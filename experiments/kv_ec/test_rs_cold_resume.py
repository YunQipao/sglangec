"""CPU protocol/codec integration ONLY. Fixture and native evidence are mocked.

Production APIs reject these synthetic rank payloads; no CPU result is Stage E
RDMA/CUDA acceptance. Native transport and GPU regression remain separate.
"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import zipfile

from . import node_block_export as node
from . import rs_cold_source as source
from . import verify_rs_cold_resume as cli
from .cold_resume_state import SAMPLING, load_checkpoint
from .gpu_kv_restore import RestoreRejected
from .gpu_resume_hook import token_digest
from .online_rs_identity import stripe_id
from .rs_codec import encode, recover
from .rs_stripe_builder import build, manifest_sha, validate_manifest
from .test_online_rs import fixture
from .test_online_rs import MockRDMAClient
from .online_rs_exporter import OnlineRSExporter
from .remote_rs_server import StripeStore


def stats(size):
    # Deliberate CPU fixture, used only to test parser/identity contracts.
    line = (f'KV_EC_RDMA_STATS bytes={size} qp_num=9 remote_qpn=10 '
            'completion_status=SUCCESS')
    return dict(transport='rdma', bytes=size, completion_status='SUCCESS',
                qp_num=9, remote_qpn=10, helper_stdout=line)


def save(path, value):
    path.write_text(json.dumps(value))


def capture(root, node_id):
    root.mkdir()
    arrays, ranks = fixture(node_id, tp=2)
    ids, pending = [11] * 256, 7
    for rank in ranks:
        rank.update(req_pool_idx=3, req_generation=4, token_ids_sha256=token_digest(ids))
    block = node.pack(node_id, [a.tobytes() for a in arrays], ranks, allow_fixture=True)
    first = ranks[0]
    state = dict(schema='kv-cold-request-v1', checkpoint_version=1, request_version=1,
        tp_size=2, boundary='post_prefill_result_before_decode', origin_input_ids=ids,
        output_ids=[pending], materialized_kv_tokens=256, kv_allocated_len=256,
        pending_token_id=pending, pending_token_has_kv=False, next_decode_input_ids=[pending],
        next_decode_position=256, next_decode_seq_len=257, sampling=SAMPLING,
        rng_policy='greedy_no_rng', finished=False, extend_batch_idx=1, decode_batch_idx=0,
        decode_steps=16, vocab_size=32000, request_id=f'independent-request-{node_id}',
        restore_id=first['snapshot_id'], capture_run_id=first['request_version'],
        token_start=0, token_count=256, node_id=node_id, input_ids_sha256=token_digest(ids),
        kv_sha256=node.sha(block), runtime_identity={k: first[k] for k in
            ('model_sha256', 'source_commit', 'core_diff_sha256')}, origin_row=3, origin_generation=4)
    origins = [dict(host=f'node{node_id}', boot_id='boot', pid=100 + r, start_ticks=10 + r)
               for r in range(3)]
    for r in (0, 1):
        (root / f'rank_{r}.bin').write_bytes(arrays[r].tobytes())
        save(root / f'rank_{r}.meta.json', ranks[r])
        save(root / f'rank_{r}.capture.json', dict(tp_rank=r, cuda_executed=True,
            process=origins[r], state_sha256=node.sha(node.canonical(state)),
            materialized_kv_tokens=256, pending_token_id=pending))
    save(root / 'request_state.json', state)
    seal = dict(schema='kv-cold-seal-v1', state='GLOBAL_COMMITTED', restore_id=state['restore_id'],
        request_state_sha256=node.sha((root / 'request_state.json').read_bytes()),
        kv_sha256=node.sha(block), rank_state_sha256=[node.sha(node.canonical(state))] * 2,
        origin_processes=origins[:2], origin_server_process=origins[2],
        rank_capture_sha256=[node.sha((root / f'rank_{r}.capture.json').read_bytes()) for r in (0, 1)],
        rank_payload_sha256=[node.sha(a.tobytes()) for a in arrays])
    save(root / 'COMMITTED.json', seal)
    (root / 'global_node.bin').write_bytes(block)
    return block


class RSColdResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Explicit test-only fixture boundaries. CLI has no such option.
        fixture_unpack = lambda b: node.unpack(b, allow_fixture=True)
        fixture_pack = lambda n, b, r: node.pack(n, b, r, allow_fixture=True)
        fixture_descriptor = lambda b: node.descriptor(b, allow_fixture=True)
        for module in (source, cli):
            for name, fn in (('unpack', fixture_unpack), ('descriptor', fixture_descriptor)):
                if hasattr(module, name):
                    self.enterContext(patch.object(module, name, fn))
        self.enterContext(patch.object(source, 'pack', fixture_pack))
        for mod in ('rs_cold_source', 'rs_recovery_source'):
            self.enterContext(patch(f'experiments.kv_ec.{mod}.validate_manifest',
                side_effect=lambda m: validate_manifest(m, allow_fixture=True)))
        self.enterContext(patch('experiments.kv_ec.cold_resume_state.validate_checkpoint',
                                side_effect=lambda b: fixture_unpack(b)[0]))
        self.enterContext(patch('experiments.kv_ec.cold_resume_state.unpack', fixture_unpack))
        self.enterContext(patch('experiments.kv_ec.rs_recovery_source.validate_checkpoint',
                                side_effect=lambda b: fixture_unpack(b)[0]))
        self.enterContext(patch('experiments.kv_ec.rs_recovery_source.descriptor', fixture_descriptor))
        self.blocks = [capture(self.root / f'capture{n}', n) for n in (0, 1)]
        self.claims = [cli.capture_claim(self.root / f'capture{n}') for n in (0, 1)]
        self.plan = dict(schema='kv-cold-rs-plan-v1', run_id='cpu-only', captures=self.claims)
        self.identity = source.plan_identity(self.plan)
        self.protected = []
        for n, claim in enumerate(self.claims):
            meta, raw = fixture_unpack(self.blocks[n])
            ranks = [dict(r, global_protection=self.identity) for r in meta['ranks']]
            self.protected.append(fixture_pack(n, raw, ranks))
        sid = stripe_id(self.identity)
        manifest = build(sid, self.protected, allow_fixture=True)
        parity = encode(*self.protected)
        self.status = dict(state='COMMITTED', manifest=manifest, stripe_id=sid,
            handle='cpu-mock-handle', manifest_sha256=manifest_sha(manifest),
            parity_sha256=node.sha(parity), submitted_nodes=[0, 1], rdma={})
        self.receipts = []
        for n, b in enumerate(self.protected):
            self.status['rdma'][str(n)] = dict(state='COMPLETED', payload_bytes=len(b),
                payload_sha256=node.sha(b), stats=stats(len(b)))
            self.receipts.append(dict(global_state='GLOBAL_COMMITTED', node_id=n,
                transport='rdma', remote_outcome='COMMITTED', global_protection=self.identity,
                stripe_id=sid, handle=self.status['handle'], manifest_sha256=manifest_sha(manifest),
                parity_sha256=node.sha(parity), node_descriptor=manifest['nodes'][n],
                local_snapshot_id=self.claims[n]['state']['restore_id'],
                metrics=dict(rdma_native=stats(len(b)))))
        self.protection = dict(schema='kv-cold-rs-protection-v1', state='GLOBAL_COMMITTED',
            plan=self.plan, receipts=self.receipts, status=self.status,
            exports=[dict(schema='kv-cold-rs-export-v1', node_id=n,
                plan_sha256=node.sha(node.canonical(self.plan)), receipt=self.receipts[n],
                local_xor_sha256=node.sha(node.xor(*node.unpack(self.blocks[n], allow_fixture=True)[1])),
                local_xor_recovery_checks=[dict(lost_rank=r, byte_exact=True) for r in (0, 1)])
                for n in (0, 1)], remote_host='cpu-only', remote_port=30700,
            control_transfers=[dict(transport='tcp', tx=dict(payload_bytes=0), rx=dict(payload_bytes=0))])
        # Actual codec recovers Node 1 from D0 and P; target is not passed.
        rebuilt = recover([self.protected[0], None], parity, [len(b) for b in self.protected])
        self.assertEqual(rebuilt, self.protected[1])
        self.bundle = self.root / 'recovery'
        self.bundle.mkdir()
        child = self.bundle / 'protected'
        child.mkdir()
        (child / 'recovered_node.bin').write_bytes(rebuilt)
        transfer = dict(state='COMPLETED', payload_bytes=len(rebuilt), payload_sha256=node.sha(rebuilt),
            client_send_stats=stats(len(self.protected[0])), receive_stats=stats(len(self.protected[0])),
            send_stats=stats(len(rebuilt)), client_receive_stats=stats(len(rebuilt)))
        after = deepcopy(self.status)
        after['rdma']['recovery_1'] = transfer
        report = dict(schema='kv-rs-gpu-source-v1', receipt=self.receipts[1], status_before=self.status,
            status_after=after, transfer=transfer, recovered_bytes=len(rebuilt), recovered_sha256=node.sha(rebuilt),
            survivor_sha256=node.sha(self.protected[0]), real_rdma=True, rs_recovery_byte_exact=True,
            runtime_original_local_file_used=False, recovery_input='survivor_plus_remote_committed_parity',
            control_transfers=[dict(op='prepare_rdma_recovery', transport='tcp',
                                   tx=dict(payload_bytes=0), rx=dict(payload_bytes=0))])
        save(child / 'recovery.json', report)
        (self.bundle / 'recovered_node.bin').write_bytes(self.blocks[1])
        save(self.bundle / 'stage_e.json', dict(schema='kv-rs-cold-source-v1',
            protection=self.protection, target_node_id=1, runtime_original_local_file_used=False,
            capture_block_sha256=node.sha(self.blocks[1]), protected_source=dict(
                recovery_sha256=node.sha((child / 'recovery.json').read_bytes()), block_sha256=node.sha(rebuilt))))

    def test_rs_envelope_reproduces_exact_sealed_capture_and_runtime_needs_no_original(self):
        original = self.root / 'capture1'
        (original / 'global_node.bin').unlink()
        for r in (0, 1):
            (original / f'rank_{r}.bin').unlink()
        restored, state, _, binding = load_checkpoint(original, self.bundle)
        self.assertEqual(restored.load(), self.blocks[1])
        self.assertEqual(state, self.claims[1]['state'])
        self.assertEqual(binding['block_sha256'], self.claims[1]['state']['kv_sha256'])

    def test_fixture_is_rejected_by_production_node_validator(self):
        with self.assertRaisesRegex(ValueError, 'real SGLang capture'):
            node.unpack(self.protected[1])

    def test_capture_export_adapter_uses_real_manager_exporter_and_rs_store(self):
        # CPU transport injection ONLY. Actual manager aggregation, worker
        # registration, XOR, packing, RS commit and unwrapping run unchanged.
        store = StripeStore(allow_fixture=True, rdma_directory=self.root / 'mock-rdma')
        plan = self.root / 'plan.json'
        save(plan, self.plan)
        def exporter(**options):
            return OnlineRSExporter(**options, allow_fixture=True,
                client_factory=lambda: MockRDMAClient(store), poll_interval=0.005)
        def run(n):
            return cli.export_capture(NS(plan=plan, capture=self.root / f'capture{n}',
                node_id=n, output=self.root / f'export{n}', remote_host='cpu-only',
                remote_port=30700, rdma_device='cpu-only', rdma_gid_index=0,
                rdma_port_num=1, timeout=5))
        with patch.object(cli, 'OnlineRSExporter', side_effect=exporter), ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, (0, 1)))
        for n, result in enumerate(results):
            wrapped = Path(result['node_block']).read_bytes()
            self.assertEqual(source.unwrap_node(wrapped, self.claims[n], self.identity), self.blocks[n])
            export = json.loads((self.root / f'export{n}' / 'export.json').read_text())
            self.assertEqual(export['local_xor_recovery_checks'],
                [dict(lost_rank=r, byte_exact=True) for r in (0, 1)])
            self.assertEqual(export['receipt']['global_state'], 'GLOBAL_COMMITTED')
            self.assertEqual(export['receipt']['transport'], 'CPU_MOCK')
            self.assertNotEqual(node.sha(wrapped), node.sha(self.blocks[n]))

    def test_plan_rejects_historical_state_pair_and_wrong_rank_model(self):
        for key, value in (('kv_sha256', '0' * 64), ('pending_token_id', 9), ('restore_id', 'old')):
            bad = deepcopy(self.plan)
            bad['captures'][1]['state'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                source.plan_identity(bad)
        bad = deepcopy(self.plan)
        bad['captures'][1]['descriptor']['metadata']['ranks'][1]['model_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'rank_identity_mismatch'):
            source.plan_identity(bad)

    def test_noncommitted_stale_handle_and_missing_native_upload_are_rejected(self):
        for mutate in (lambda p: p.update(state='PENDING'),
                       lambda p: p['status'].update(state='PENDING'),
                       lambda p: p['receipts'][1].update(handle='old'),
                       lambda p: p['receipts'][1]['metrics'].update(rdma_native=dict(transport='CPU_MOCK')),
                       lambda p: p['exports'][1].update(local_xor_recovery_checks=[]),
                       lambda p: p['control_transfers'][0]['tx'].update(payload_bytes=1)):
            bad = deepcopy(self.protection)
            mutate(bad)
            with self.assertRaises(ValueError):
                source.validate_protection(bad)

    def test_corrupt_or_different_capture_envelope_is_rejected(self):
        other = deepcopy(self.claims[1])
        other['descriptor']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'capture_hash'):
            source.unwrap_node(self.protected[1], other, self.identity)
        with self.assertRaises(ValueError):
            source.unwrap_node(self.protected[0], self.claims[1], self.identity)
        broken = bytearray(self.protected[1])
        broken[-1] ^= 1
        with self.assertRaises(ValueError):
            source.unwrap_node(bytes(broken), self.claims[1], self.identity)

    def test_recovered_payload_must_match_local_xor_report(self):
        path = self.bundle / 'stage_e.json'
        report = json.loads(path.read_text())
        report['protection']['exports'][1]['local_xor_sha256'] = '0' * 64
        save(path, report)
        with self.assertRaisesRegex(ValueError, 'local_xor_hash_mismatch'):
            source.load_cold_source(self.bundle)

    def test_recovery_and_cold_artifact_and_state_seals_are_pinned(self):
        path = self.bundle / 'recovered_node.bin'
        original = path.read_bytes()
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, 'cold_artifact_changed'):
            source.load_cold_source(self.bundle)
        path.write_bytes(original)
        capture_root = self.root / 'capture1'
        seal = json.loads((capture_root / 'COMMITTED.json').read_text())
        seal['extra_unbound_field'] = True
        save(capture_root / 'COMMITTED.json', seal)
        with self.assertRaisesRegex(ValueError, 'state_seal_mismatch'):
            load_checkpoint(capture_root, self.bundle)

    def test_stage_e_check_cannot_accept_a_local_stage_d_pass(self):
        save(self.root / 'run.json', dict(cold_mode='restore', rs_source=None))
        with patch.object(cli.cold, 'check_results', return_value=dict(pass_stage_d=True)) as check:
            with self.assertRaisesRegex(ValueError, 'rs_cold_run_required'):
                cli.check_results(self.root)
            check.assert_not_called()

    def test_independent_comparison_and_native_delivery_are_required(self):
        with self.assertRaisesRegex(ValueError, 'native_delivery_required'):
            cli.check_source(self.bundle, self.root / 'capture1')
        self.deliver()
        result = cli.check_source(self.bundle, self.root / 'capture1')
        self.assertTrue(result['pass_source'])
        with self.assertRaises(ValueError):
            cli.check_source(self.bundle, self.root / 'capture0')

    def deliver(self):
        binding = source.load_cold_source(self.bundle)[3]
        archive = self.bundle / 'delivery.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as zipped:
            for name in source.FILES:
                zipped.write(self.bundle / name, name)
        raw = archive.read_bytes()
        info = dict(archive_bytes=len(raw), archive_sha256=node.sha(raw))
        save(self.bundle / 'delivery.json', dict(info, receive_stats=stats(len(raw))))
        save(self.bundle / 'delivery-send.json', dict(info, bundle=binding, send_stats=stats(len(raw))))

    def test_delivery_requires_matching_native_sender_and_immutable_archive(self):
        self.deliver()
        receipt = self.bundle / 'delivery-send.json'
        sent = json.loads(receipt.read_text())
        for bad in (dict(sent, archive_sha256='0' * 64),
                    dict(sent, send_stats=dict(transport='CPU_MOCK')),
                    dict(sent, bundle=dict(sent['bundle'], request_state_sha256='0' * 64))):
            save(receipt, bad)
            with self.assertRaises(ValueError):
                source.load_cold_source(self.bundle)
        save(receipt, sent)
        archive = self.bundle / 'delivery.zip'
        archive.write_bytes(archive.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'archive_changed'):
            source.load_cold_source(self.bundle)

    def test_live_check_rejects_restarted_or_expired_stripe(self):
        with patch.object(cli, 'RemoteRSClient') as client:
            client.return_value.transfers = self.protection['control_transfers']
            client.return_value.call.return_value = (self.status, b'')
            report = dict(protection=self.protection)
            cli.live_protection(report)
            for changed in (dict(self.status, handle='restarted'), dict(self.status, state='EXPIRED')):
                client.return_value.call.return_value = (changed, b'')
                with self.assertRaises(ValueError):
                    cli.live_protection(report)

    def test_same_host_captures_cannot_bypass_plan_cli(self):
        bad = deepcopy(self.plan)
        # Host independence is also enforced when loading a plan from disk.
        for process in bad['captures'][1]['seal']['origin_processes']:
            process['host'] = 'node0'
        with self.assertRaises(ValueError):
            source.plan_identity(bad)

    def test_stage_e_aggregation_requires_strict_cold_result_and_both_rank_bindings(self):
        self.deliver()
        accepted = cli.check_source(self.bundle, self.root / 'capture1')
        save(self.bundle / 'source-check.json', accepted)
        binding = accepted['bundle']
        output = self.root / 'cold-new'
        output.mkdir()
        save(output / 'run.json', dict(cold_mode='restore', rs_source=str(self.bundle),
            rs_source_binding=binding, state_checkpoint=str(self.root / 'capture1')))
        passed = dict(pass_stage_d=True, cuda_executed=True, new_instance=True,
            restored_slots_used_by_decode=True, output_token_ids_match=True,
            no_slot_leak=True, no_request_row_leak=True, resume_decode_forward_count=16,
            recomputed_prefix_tokens=0, original_prefix_prefill_tokens=0,
            ranks=[dict(committed=True, transaction_state='COMMITTED', rs_source=binding,
                        runtime_original_local_file_used=False) for _ in (0, 1)])
        with patch.object(cli, 'live_protection', return_value=dict(query_completed=True)), \
                patch.object(cli.cold, 'check_results') as cold:
            cold.return_value = deepcopy(passed)
            self.assertTrue(cli.check_results(output)['pass_stage_e'])
            cases = [dict(passed, **{k: False}) for k in (
                'pass_stage_d', 'cuda_executed', 'new_instance',
                'restored_slots_used_by_decode', 'output_token_ids_match',
                'no_slot_leak', 'no_request_row_leak')]
            cases += [dict(passed, resume_decode_forward_count=15),
                      dict(passed, original_prefix_prefill_tokens=256),
                      dict(passed, ranks=[])]
            for rank in (0, 1):
                bad = deepcopy(passed)
                bad['ranks'][rank]['transaction_state'] = 'ABORTED'
                cases.append(bad)
            for bad in cases:
                cold.return_value = deepcopy(bad)
                self.assertFalse(cli.check_results(output)['pass_stage_e'])


if __name__ == '__main__':
    unittest.main()
