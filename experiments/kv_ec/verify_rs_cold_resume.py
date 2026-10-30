"""Stage E: sealed Capture -> existing online XOR/RDMA/RS -> new-instance Decode.

Only metadata claims cross via scp. Protection uploads, reconstruction, and
delivery of recovered KV to the target host use the existing native verbs.
Every output is new; failure does not authorize local-target fallback.
"""

import argparse
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
import zipfile

import numpy as np

from . import verify_gpu_cold_resume as cold
from .cold_resume_state import load_checkpoint, write_json
from .gpu_kv_restore import _require
from .node_block_export import canonical, descriptor, sha, unpack, xor
from .online_chunk_manager import OnlineChunkManager
from .online_rs_exporter import OnlineRSExporter
from .online_rs_identity import stripe_id
from .remote_rs_client import RemoteRSClient
from .rs_cold_source import (FILES, MAX_ARCHIVE, load_cold_source, plan_identity, unwrap_node,
                             validate_claim, validate_protection)
from .rs_recovery_source import load_bundle, native_stats, recover_node
from .verify_rs_resume import rdma_options
from .perf_observer import observed

def capture_claim(directory):
    root = Path(directory)
    source, state, seal, _ = load_checkpoint(root)
    claim = dict(schema='kv-cold-rs-claim-v1', state=state, seal=seal,
        descriptor=descriptor(source.load()),
        state_document=(root / 'request_state.json').read_text(),
        seal_document=(root / 'COMMITTED.json').read_text(),
        proof_documents=[(root / f'rank_{r}.capture.json').read_text() for r in range(state['tp_size'])],
        proofs=[json.loads((root / f'rank_{r}.capture.json').read_text()) for r in range(state['tp_size'])],
        request_state_sha256=sha((root / 'request_state.json').read_bytes()),
        capture_seal_sha256=sha(canonical(seal)),
        capture_seal_file_sha256=sha((root / 'COMMITTED.json').read_bytes()))
    return validate_claim(claim)


def make_plan(args):
    plan = dict(schema='kv-cold-rs-plan-v1', run_id=args.run_id,
        captures=[json.loads(p.read_text()) for p in (args.node0_claim, args.node1_claim)])
    plan_identity(plan)
    write_json(args.output, plan)
    return dict(plan=str(args.output), plan_sha256=sha(canonical(plan)))


@observed("protection_inclusive", "protection")
def export_capture(args):
    plan = json.loads(args.plan.read_text())
    identity = plan_identity(plan)
    claim = capture_claim(args.capture)
    node = args.node_id
    _require(claim == plan['captures'][node], 'stage_e_export_wrong_capture')
    # This is the adapter: validated real rank arrays enter the existing online
    # manager, which performs local XOR and hands private CPU copies to exporter.
    source, _, _, _ = load_checkpoint(args.capture)
    meta, payloads = unpack(source.load())
    exporter = OnlineRSExporter(node_id=node, run_id=identity['run_id'],
        checkpoint_version=identity['checkpoint_version'],
        requests={meta['ranks'][0]['request_id']: identity['protection_id']},
        remote_host=args.remote_host, remote_port=args.remote_port,
        rdma_device=args.rdma_device, rdma_gid_index=args.rdma_gid_index,
        rdma_port_num=args.rdma_port_num, timeout=args.timeout,
        rdma_timeout=min(args.timeout, 180), workers=1, max_pending=1)
    try:
        manager = OnlineChunkManager(args.output, claim['state']['capture_run_id'],
            test_recovery=True, global_exporter=exporter)
        for rank in range(len(meta['ranks'])):
            result = manager.submit(meta['ranks'][rank], np.frombuffer(payloads[rank], dtype=np.uint8))
        _require(result['state'] == 'COMMITTED', 'stage_e_local_xor_not_committed')
        directory = args.output / claim['state']['restore_id']
        deadline = time.monotonic() + args.timeout
        while True:
            receipt = json.loads((directory / 'global_status.json').read_text())
            if receipt['global_state'] == 'GLOBAL_COMMITTED':
                break
            _require(receipt['global_state'] not in ('FAILED', 'EXPIRED', 'CANCELLED') and
                     time.monotonic() < deadline, 'stage_e_export_failed:' + str(receipt))
            time.sleep(0.1)
        protected = (directory / 'global_node.bin').read_bytes()
        _require(unwrap_node(protected, claim, identity) == source.load(),
                 'stage_e_export_did_not_preserve_capture')
        local = json.loads((directory / 'manifest.json').read_text())
        _require(local['parity_sha256'] == sha(xor(*payloads)) and
                 all(c['byte_exact'] for c in local['recovery_checks']), 'stage_e_local_xor_invalid')
        write_json(args.output / 'export.json', dict(schema='kv-cold-rs-export-v1',
            plan_sha256=sha(canonical(plan)), node_id=node, receipt=receipt,
            local_xor_sha256=local['parity_sha256'], local_xor_recovery_checks=local['recovery_checks']))
        return dict(receipt=str(directory / 'global_status.json'), node_block=str(directory / 'global_node.bin'))
    finally:
        exporter.close()


def seal_protection(args):
    plan = json.loads(args.plan.read_text())
    identity = plan_identity(plan)
    exports = [json.loads(p.read_text()) for p in (args.node0_export, args.node1_export)]
    receipts = [e['receipt'] for e in exports]
    client = RemoteRSClient(args.remote_host, args.remote_port, args.timeout)
    first = receipts[0]
    _require(first['stripe_id'] == stripe_id(identity), 'stage_e_wrong_stripe')
    status, raw = client.call('get_stripe_status', **{k: first[k] for k in
        ('stripe_id', 'handle', 'manifest_sha256')})
    _require(not raw, 'stage_e_tcp_payload_forbidden')
    protection = dict(schema='kv-cold-rs-protection-v1', plan=plan, receipts=receipts,
        exports=exports, status=status, control_transfers=client.transfers,
        remote_host=args.remote_host, remote_port=args.remote_port, state='GLOBAL_COMMITTED')
    validate_protection(protection)
    write_json(args.output, protection)
    return dict(protection=str(args.output), stripe_id=first['stripe_id'],
        handle=first['handle'], manifest_sha256=first['manifest_sha256'])


@observed("recover_adapter_inclusive", "recovery")
def recover(args):
    protection = json.loads(args.protection.read_text())
    identity = validate_protection(protection)  # commitment BEFORE logical loss
    survivor = unwrap_node(args.survivor.read_bytes(), protection['plan']['captures'][0], identity)
    _require(sha(xor(*unpack(survivor)[1])) == protection['exports'][0]['local_xor_sha256'],
             'stage_e_local_xor_hash_mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    # Fault model: the reconstruction API receives no target input. The server
    # already discarded both source blocks at COMMITTED and retains only parity.
    write_json(args.output / 'receipt.json', protection['receipts'][1])
    options = vars(args).copy()
    options.update(receipt=args.output / 'receipt.json', output=args.output / 'protected')
    recover_node(NS(**options))
    source, _, _, binding = load_bundle(args.output / 'protected')
    claim = protection['plan']['captures'][1]
    block = unwrap_node(source.load(), claim, identity)
    with (args.output / 'recovered_node.bin').open('xb') as stream:
        stream.write(block)
    write_json(args.output / 'stage_e.json', dict(schema='kv-rs-cold-source-v1',
        protection=protection, target_node_id=1,
        protected_source={k: binding[k] for k in ('recovery_sha256', 'block_sha256')},
        capture_block_sha256=sha(block), runtime_original_local_file_used=False,
        ordering=['CAPTURE_SEALED', 'RS_GLOBAL_COMMITTED', 'TARGET_INPUT_WITHHELD',
                  'NATIVE_RDMA_RECONSTRUCTED', 'CAPTURE_HASH_VERIFIED']))
    _, _, _, identity = load_cold_source(args.output)
    return identity


@observed("delivery_send_inclusive", "communication")
def send_source(args):
    from .multinode.rdma_transport import RDMATransport
    _, _, _, binding = load_cold_source(args.bundle)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name in FILES:
            archive.writestr(name, (args.bundle / name).read_bytes())
    payload = buffer.getvalue()
    _require(len(payload) <= MAX_ARCHIVE, 'stage_e_delivery_too_large')
    transport = RDMATransport(device=args.rdma_device, gid_index=args.rdma_gid_index,
                             port_num=args.rdma_port_num, timeout=args.timeout)
    try:
        stats = transport.send_with_stats(args.remote_host, args.remote_port, payload)
        native_stats(stats, len(payload))
    finally:
        transport.close()
    result = dict(bundle=binding, send_stats=stats, archive_bytes=len(payload), archive_sha256=sha(payload))
    write_json(args.output, result)
    return result


@observed("delivery_receive_inclusive", "communication")
def receive_source(args):
    from .multinode.rdma_transport import receive_file
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'delivery.zip'
    stats = receive_file(args.remote_port, path, device=args.rdma_device,
        gid_index=args.rdma_gid_index, port_num=args.rdma_port_num,
        timeout=args.timeout, return_stats=True)
    size = path.stat().st_size
    native_stats(stats, size)
    _require(size <= MAX_ARCHIVE, 'stage_e_delivery_too_large')
    with zipfile.ZipFile(path) as archive:
        _require(sorted(archive.namelist()) == sorted(FILES) and
                 sum(i.file_size for i in archive.infolist()) <= MAX_ARCHIVE,
                 'stage_e_delivery_archive_invalid')
        for name in FILES:
            dest = args.output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(archive.read(name))
    load_cold_source(args.output)
    write_json(args.output / 'delivery.json', dict(receive_stats=stats, archive_bytes=size,
        archive_sha256=sha(path.read_bytes())))
    return load_cold_source(args.output)[3]


@observed("independent_source_validation", "validation")
def check_source(bundle, capture):
    source, report, _, binding = load_cold_source(bundle)
    original, state, _, _ = load_checkpoint(capture)  # independent acceptance ONLY
    _require(source.load() == original.load() and
             state == report['protection']['plan']['captures'][1]['state'],
             'stage_e_independent_capture_differs')
    _require('delivery_sha256' in binding and 'delivery_send_sha256' in binding,
             'stage_e_native_delivery_required')
    return dict(pass_source=True, bundle=binding, original_sha256=sha(original.load()),
        recovered_sha256=sha(source.load()), real_rdma=True, rs_recovery_byte_exact=True,
        runtime_original_local_file_used=False)


def live_protection(report):
    """Refuse an expired/restarted parity store even if old receipts were valid."""
    protection = report['protection']
    receipt = protection['receipts'][1]
    client = RemoteRSClient(protection['remote_host'], protection['remote_port'], 30)
    status, raw = client.call('get_stripe_status', **{k: receipt[k] for k in
        ('stripe_id', 'handle', 'manifest_sha256')})
    _require(not raw, 'stage_e_tcp_payload_forbidden')
    validate_protection(dict(protection, status=status, control_transfers=client.transfers))
    # Keep the final verdict repeatable; volatile TTL/timing fields from a live
    # query must not make a second successful check overwrite old evidence.
    return dict(status={k: status[k] for k in (
        'state', 'stripe_id', 'handle', 'manifest_sha256', 'parity_sha256', 'submitted_nodes')},
        query_completed=True, control_payload_bytes=0)


def launch(args):
    _, report, _, binding = load_cold_source(args.rs_source)
    accepted = json.loads((args.rs_source / 'source-check.json').read_text())
    _require(accepted == check_source(args.rs_source, args.bundle) and accepted['bundle'] == binding,
             'stage_e_source_acceptance_required')
    live_protection(report)
    cold.launch(args)  # identical Stage D Scheduler and GPU transaction


def check_results(output):
    cfg = json.loads((Path(output) / 'run.json').read_text())
    _require(cfg['cold_mode'] == 'restore' and bool(cfg.get('rs_source')), 'stage_e_rs_cold_run_required')
    bundle = Path(cfg['rs_source'])
    source, report, _, binding = load_cold_source(bundle)
    _require(binding == cfg['rs_source_binding'] and
             json.loads((bundle / 'source-check.json').read_text()) ==
             check_source(bundle, cfg['state_checkpoint']), 'stage_e_source_changed')
    live = live_protection(report)
    result = cold.check_results(output)
    committed = len(result['ranks']) == result['tp_size'] and all(
        r['committed'] is True and r['transaction_state'] == 'COMMITTED' and
        r['rs_source'] == binding and r['runtime_original_local_file_used'] is False
        for r in result['ranks'])
    passed = all(result.get(k) is True for k in ('pass_stage_d', 'cuda_executed', 'new_instance',
        'restored_slots_used_by_decode', 'output_token_ids_match', 'no_slot_leak', 'no_request_row_leak'))
    passed = passed and committed and result['resume_decode_forward_count'] == 16 and \
        result['recomputed_prefix_tokens'] == result['original_prefix_prefill_tokens'] == 0
    receipt = report['protection']['receipts'][1]
    result.update(schema='kv-rs-cold-resume-stage-e-v1', pass_stage_e=bool(passed),
        functional_correctness_pass=bool(passed), isolated_performance_eligible=False,
        real_rdma=True, rs_recovery_byte_exact=True, gpu_restore_committed=committed,
        runtime_original_local_file_used=False, stripe_id=receipt['stripe_id'],
        handle=receipt['handle'], manifest_sha256=receipt['manifest_sha256'],
        target_node_id=1, recovered_bytes=len(source.load()), recovered_sha256=sha(source.load()),
        rs_source=binding, whole_node_failover=False,
        live_rs=live,
        failure_domain='controlled Node Block loss + same-host new-instance cold resume')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('claim')
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('plan')
    p.add_argument('--node0-claim', type=Path, required=True)
    p.add_argument('--node1-claim', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('export')
    rdma_options(p)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--node-id', type=int, choices=(0, 1), required=True)
    p = sub.add_parser('protect')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--node0-export', type=Path, required=True)
    p.add_argument('--node1-export', type=Path, required=True)
    p.add_argument('--remote-host', required=True)
    p.add_argument('--remote-port', type=int, default=30700)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('recover')
    rdma_options(p, returning=True)
    p.add_argument('--protection', type=Path, required=True)
    p.add_argument('--survivor', type=Path, required=True)
    for name in ('send-source', 'receive-source'):
        p = sub.add_parser(name)
        rdma_options(p)
        if name == 'send-source':
            p.add_argument('--bundle', type=Path, required=True)
    p = sub.add_parser('inspect')
    p.add_argument('--bundle', type=Path, required=True)
    p = sub.add_parser('check-source')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--send-receipt', type=Path, required=True)
    p = sub.add_parser('launch')
    for name in ('bundle', 'rs-source', 'reference', 'ledger', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gpus', default='auto')
    p.add_argument('--port', type=int, default=30986)
    p.add_argument('--mem-fraction-static', type=float, default=0.6)
    p.add_argument('--gpu-admission-mode', choices=('strict', 'shared'), default='strict')
    p.add_argument('--timeout', type=float, default=120)
    p.set_defaults(inject_rank_failure=None)
    p = sub.add_parser('submit', aliases=['restore'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:30986')
    p.add_argument('--wait-ready', type=float, default=600)
    p.add_argument('--timeout', type=float, default=600)
    p = sub.add_parser('check')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'claim':
        result = capture_claim(args.capture)
        write_json(args.output, result)
    elif args.command == 'plan':
        result = make_plan(args)
    elif args.command == 'export':
        result = export_capture(args)
    elif args.command == 'protect':
        result = seal_protection(args)
    elif args.command == 'recover':
        result = recover(args)
    elif args.command == 'send-source':
        result = send_source(args)
    elif args.command == 'receive-source':
        result = receive_source(args)
    elif args.command == 'inspect':
        result = load_cold_source(args.bundle)[3]
    elif args.command == 'check-source':
        # Sender metadata may travel over SCP; KV payload arrived via native RDMA.
        write_json(args.bundle / 'delivery-send.json', json.loads(args.send_receipt.read_text()))
        result = check_source(args.bundle, args.capture)
        write_json(args.bundle / 'source-check.json', result)
    elif args.command == 'launch':
        launch(args)
        return
    elif args.command in ('submit', 'restore'):
        cold.submit(args)
        return
    else:
        try:
            result = check_results(args.output)
        except Exception as exc:
            result = cold.failure_result(args.output, exc)
        cold.save_verdict(args.output, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result.get('pass_stage_e') is True else 1)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
