"""Stage E export envelope adapter; no RS codec, Scheduler or GPU ownership.

The exporter adds global_protection to metadata. Removing ONLY that extension
from a verified reconstructed node must reproduce the sealed D block exactly.
No original block/rank file is read by this recovery data source.
"""

import json
from pathlib import Path
import zipfile

from .cold_resume_state import validate_state
from .gpu_kv_restore import BytesCheckpointSource, _require
from .node_block_export import canonical, descriptor, pack, sha, unpack, xor
from .online_rs_identity import identity_for, stripe_id
from .rs_recovery_source import committed_status, load_bundle, native_stats
from .rs_stripe_builder import validate_manifest
from .rs_codec import CODE
from .perf_observer import observed

FILES = ('stage_e.json', 'recovered_node.bin', 'protected/recovery.json',
         'protected/recovered_node.bin')
MAX_ARCHIVE = 160 * 1024**2


def validate_claim(claim):
    state, seal, own = claim['state'], claim['seal'], claim['descriptor']
    validate_state(state, own['metadata'], own['sha256'])
    _require(claim['schema'] == 'kv-cold-rs-claim-v1' and
             json.loads(claim['state_document']) == state and
             json.loads(claim['seal_document']) == seal and
             sha(claim['state_document'].encode()) == claim['request_state_sha256'] and
             sha(claim['seal_document'].encode()) == claim['capture_seal_file_sha256'] and
             claim['request_state_sha256'] == seal['request_state_sha256'] and
             claim['capture_seal_sha256'] == sha(canonical(seal)) and
             seal['schema'] == 'kv-cold-seal-v1' and seal['state'] == 'GLOBAL_COMMITTED' and
             seal['kv_sha256'] == own['sha256'] == state['kv_sha256'] and
             seal['restore_id'] == state['restore_id'] and
             seal['rank_state_sha256'] == [sha(canonical(state))] * 2,
             'stage_e_capture_claim_invalid')
    _require(len(seal['origin_processes']) == 2 and
             seal['origin_processes'][0] != seal['origin_processes'][1],
             'stage_e_capture_processes_invalid')
    for rank, proof in enumerate(claim['proofs']):
        _require(proof['tp_rank'] == rank and proof['cuda_executed'] is True and
                 json.loads(claim['proof_documents'][rank]) == proof and
                 sha(claim['proof_documents'][rank].encode()) == seal['rank_capture_sha256'][rank] and
                 own['metadata']['ranks'][rank]['sha256'] == seal['rank_payload_sha256'][rank] and
                 proof['state_sha256'] == sha(canonical(state)) and
                 proof['process'] == seal['origin_processes'][rank] and
                 proof['materialized_kv_tokens'] == 256 and
                 proof['pending_token_id'] == state['pending_token_id'],
                 'stage_e_capture_proof_invalid')
    _require(len(claim['proofs']) == 2, 'stage_e_capture_proof_count')
    return claim


def plan_identity(plan):
    _require(plan['schema'] == 'kv-cold-rs-plan-v1' and bool(plan['run_id']), 'stage_e_plan_invalid')
    for node, claim in enumerate(plan['captures']):
        validate_claim(claim)
        _require(claim['descriptor']['metadata']['node_id'] == node, 'stage_e_plan_node_mismatch')
    _require(len(plan['captures']) == 2, 'stage_e_two_real_captures_required')
    _require(plan['captures'][0]['seal']['origin_processes'][0]['host'] !=
             plan['captures'][1]['seal']['origin_processes'][0]['host'],
             'stage_e_two_physical_capture_hosts_required')
    validate_manifest(dict(format='kv-rs-stripe-v1', code=CODE,
        stripe_id='stage-e-pair', nodes=[c['descriptor'] for c in plan['captures']]))
    first = plan['captures'][0]['descriptor']['metadata']['ranks'][0]
    return identity_for(plan['run_id'], sha(canonical(plan)), sha(canonical(plan)), first)


def validate_protection(protection):
    plan = protection['plan']
    identity = plan_identity(plan)
    receipts, status = protection['receipts'], protection['status']
    _require(protection['schema'] == 'kv-cold-rs-protection-v1' and len(receipts) == 2 and
             protection['state'] == 'GLOBAL_COMMITTED' and
             status['submitted_nodes'] == [0, 1], 'stage_e_protection_not_committed')
    for node, receipt in enumerate(receipts):
        manifest = committed_status(status, receipt)
        own = manifest['nodes'][node]
        expected = plan['captures'][node]['descriptor']['metadata']
        ranks = [dict(r, global_protection=identity) for r in expected['ranks']]
        _require(receipt['node_id'] == node and receipt['transport'] == 'rdma' and
                 receipt['remote_outcome'] == 'COMMITTED' and
                 receipt['global_protection'] == identity and
                 receipt['local_snapshot_id'] == plan['captures'][node]['state']['restore_id'] and
                 receipt['stripe_id'] == stripe_id(identity) and
                 own['metadata'] == dict(expected, ranks=ranks), 'stage_e_export_binding_invalid')
        remote = status['rdma'][str(node)]
        _require(remote['state'] == 'COMPLETED' and remote['payload_sha256'] == own['sha256'] and
                 remote['payload_bytes'] == own['length'], 'stage_e_upload_incomplete')
        native_stats(receipt['metrics']['rdma_native'], own['length'])
        native_stats(remote['stats'], own['length'])
        export = protection['exports'][node]
        checks = export['local_xor_recovery_checks']
        _require(export['schema'] == 'kv-cold-rs-export-v1' and
                 export['plan_sha256'] == sha(canonical(plan)) and
                 export['node_id'] == node and export['receipt'] == receipt and
                 len(export['local_xor_sha256']) == 64 and
                 checks == [dict(lost_rank=r, byte_exact=True) for r in (0, 1)],
                 'stage_e_local_xor_evidence_invalid')
    _require(protection['control_transfers'] and all(t['transport'] == 'tcp' and
                 t['tx']['payload_bytes'] == t['rx']['payload_bytes'] == 0
                 for t in protection['control_transfers']), 'stage_e_protection_tcp_payload')
    return identity


def unwrap_node(protected, claim, identity):
    meta, payloads = unpack(protected)
    expected = claim['descriptor']['metadata']
    _require(meta == dict(expected, ranks=[dict(r, global_protection=identity)
                 for r in expected['ranks']]), 'stage_e_reconstructed_envelope_mismatch')
    block = pack(meta['node_id'], payloads, expected['ranks'])
    _require(descriptor(block) == claim['descriptor'], 'stage_e_unwrapped_capture_hash_mismatch')
    validate_state(claim['state'], expected, sha(block))
    return block


@observed('rs_source_load_validate_inclusive', 'validation')
def load_cold_source(directory):
    root = Path(directory).resolve()
    raw = (root / 'stage_e.json').read_bytes()
    report = json.loads(raw)
    protection = report['protection']
    identity = validate_protection(protection)
    source, recovery, _, rs_binding = load_bundle(root / 'protected')
    node = report['target_node_id']
    _require(node == 1 and report['schema'] == 'kv-rs-cold-source-v1' and
             recovery['receipt'] == protection['receipts'][node] and
             recovery['status_before']['manifest'] == protection['status']['manifest'] and
             report['runtime_original_local_file_used'] is False and
             report['protected_source'] == {k: rs_binding[k] for k in ('recovery_sha256', 'block_sha256')},
             'stage_e_recovery_protection_mismatch')
    claim = protection['plan']['captures'][node]
    block = unwrap_node(source.load(), claim, identity)
    _require(sha(xor(*unpack(block)[1])) == protection['exports'][node]['local_xor_sha256'],
             'stage_e_local_xor_hash_mismatch')
    _require((root / 'recovered_node.bin').read_bytes() == block and
             report['capture_block_sha256'] == sha(block), 'stage_e_cold_artifact_changed')
    binding = dict(schema='kv-rs-cold-source-v1', path=str(root),
        source_sha256=sha(raw), block_sha256=sha(block),
        request_state_sha256=claim['request_state_sha256'],
        capture_seal_sha256=claim['capture_seal_file_sha256'])
    delivery = root / 'delivery.json'
    if delivery.exists():
        transfer = json.loads(delivery.read_text())
        native_stats(transfer['receive_stats'], transfer['archive_bytes'])
        archive = root / 'delivery.zip'
        _require(archive.stat().st_size == transfer['archive_bytes'] <= MAX_ARCHIVE and
                 sha(archive.read_bytes()) == transfer['archive_sha256'],
                 'stage_e_delivery_archive_changed')
        with zipfile.ZipFile(archive) as zipped:
            _require(sorted(zipped.namelist()) == sorted(FILES) and
                     sum(i.file_size for i in zipped.infolist()) <= MAX_ARCHIVE and
                     all(zipped.read(name) == (root / name).read_bytes() for name in FILES),
                     'stage_e_delivery_contents_changed')
        binding['delivery_sha256'] = sha(delivery.read_bytes())
    sender = root / 'delivery-send.json'
    if sender.exists():
        _require(delivery.exists(), 'stage_e_native_delivery_required')
        sent = json.loads(sender.read_text())
        native_stats(sent['send_stats'], transfer['archive_bytes'])
        _require(all(sent[k] == transfer[k] for k in ('archive_bytes', 'archive_sha256')) and
                 all(sent['bundle'][k] == binding[k] for k in (
                     'schema', 'source_sha256', 'block_sha256',
                     'request_state_sha256', 'capture_seal_sha256')),
                 'stage_e_delivery_sender_mismatch')
        binding['delivery_send_sha256'] = sha(sender.read_bytes())
    return BytesCheckpointSource(block), report, claim['descriptor']['metadata'], binding
