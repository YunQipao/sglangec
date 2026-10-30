"""CPU-only validation of configuration/real production codecs; not GPU evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.kv_ec import configured_debug as flow, debug_resources, launch_debug
from experiments.kv_ec.gpu_memory_budget import GIB
from experiments.kv_ec.test_gpu_memory_budget import GPUMemoryBudgetTests
from experiments.kv_ec.multinode.tests.test_kv_hierarchical import make_snapshot, changed_header
from experiments.kv_ec.multinode import kv_hierarchical as hierarchy
from experiments.kv_ec.multinode.verify_hierarchical import verify


EXAMPLE = Path(__file__).with_name('config-example.json')


class ConfigurationTests(unittest.TestCase):
    def read(self, cfg):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'config.json'
            path.write_text(json.dumps(cfg))
            return flow.load(path)

    def test_four_combinations_and_commands(self):
        for tp in (2, 4):
            for mode in ('STRICT', 'SHARED'):
                cfg = {**json.loads(EXAMPLE.read_text()), 'tp_size': tp,
                       'gpu_ids': list(range(tp)), 'gpu_mode': mode}
                self.read(cfg)
                argv = flow.command(cfg, Path('/tmp/run-new'), 0, 'parity')
                self.assertEqual(argv[argv.index('--tp') + 1], str(tp))
                self.assertEqual(argv[argv.index('--gpu-mode') + 1], mode)
                self.assertIn('--tokenizer', argv)

    def test_invalid_configs_fail_closed(self):
        cfg = json.loads(EXAMPLE.read_text())
        for change in ({'tp_size': True}, {'tp_size': 3}, {'tp_size': 2.0},
                       {'gpu_ids': [0]}, {'gpu_ids': [0, 0]}, {'gpu_ids': [True, 2]},
                       {'mem_fraction_static': [.5, .6]}, {'min_free_gpu_gb': float('nan')},
                       {'gpu_mode': 'shared'}, {'rs': [4, 1]}, {'transport': 'mock'},
                       {'model_path': 'Qwen/Qwen3-8B'}, {'rs_host': 'third-host'},
                       {'context_length': 256}, {'transport': 'rdma'},
                       {'unexpected_setting': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read({**cfg, **change})

    def test_roles_and_protocol_are_not_conflated(self):
        cfg = flow.load(EXAMPLE)
        worker = flow.command(cfg, Path('/tmp/run-new'), 1, 'worker')
        sender = flow.command(cfg, Path('/tmp/run-new'), 0, 'send')
        self.assertIn('experiments.kv_ec.multinode.worker', worker)
        self.assertIn('--d0-bytes', worker)
        self.assertIn('experiments.kv_ec.multinode.kv_hierarchical', sender)
        self.assertIn('--snapshot', sender)
        self.assertNotIn('--d0-bytes', sender)
        self.assertEqual(sender[sender.index('--peer') + 1], '10.102.0.233')

    def test_model_error_names_actual_directory(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = {**flow.load(EXAMPLE), 'model_path': root}
            with self.assertRaisesRegex(ValueError, 'local_model_files_required: ' + root):
                flow.check_model(cfg)

    def test_peer_source_model_tokenizer_and_tp_drift_fail(self):
        left = dict(node=0, fingerprint={'source': 'same', 'tp': 2},
                    model_identity={'model_sha256': 'model'}, tokenizer_sha256='tokenizer')
        right = {**left, 'node': 1}
        flow.compare_preflights(left, right, left['fingerprint'], 0)
        for delta in ({'fingerprint': {'source': 'different', 'tp': 2}},
                      {'fingerprint': {'source': 'same', 'tp': 4}}, {'node': 0},
                      {'model_identity': {'model_sha256': 'other'}}, {'tokenizer_sha256': 'other'}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                flow.compare_preflights(left, {**right, **delta}, left['fingerprint'], 0)

    def test_failed_child_or_gpu_observation_cannot_report_success(self):
        cfg = flow.load(EXAMPLE)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaisesRegex(RuntimeError, 'exited 3'):
                flow.execute([sys.executable, '-c', 'raise SystemExit(3)'], cfg, root, 'worker', 1)
            with patch.object(flow, 'observe', side_effect=RuntimeError('inventory unavailable')), \
                    self.assertRaisesRegex(RuntimeError, 'memory_errors'):
                flow.execute([sys.executable, '-c', 'pass'], cfg, root, 'request', 0)
            with self.assertRaisesRegex(ValueError, 'observation incomplete'):
                flow.require_request_success(root, 0)
            (root / 'request-node0-exit.json').write_text(json.dumps(
                dict(exit_code=0, memory_observation_errors=[])))
            (root / 'parity-node0.log').write_text('CUDA out of memory')
            with self.assertRaisesRegex(ValueError, 'GPU runtime failure'):
                flow.require_request_success(root, 0)


class AdmissionTests(unittest.TestCase):
    def check(self, *, shared, low=False, minimum=0):
        with tempfile.TemporaryDirectory() as root:
            model = GPUMemoryBudgetTests().model(root)
            rows = {str(i): dict(uuid=f'GPU-{i}', total_bytes=total * GIB, free_bytes=free * GIB)
                    for i, (total, free) in enumerate(((48, 36), (24, 22)))}
            if low:
                rows['1']['free_bytes'] = 4 * GIB
            with patch.object(debug_resources, 'read_gpu_memory', return_value=rows), \
                 patch.object(debug_resources, 'read_gpu_tasks', return_value={'GPU-0': [{'pid': '123'}]}):
                return debug_resources.admit(['0', '1'], model, 2, 1024, .8,
                                             shared=shared, minimum=minimum)

    def test_shared_unequal_free_and_total_capacity_allowed(self):
        result = self.check(shared=True)
        self.assertEqual([r['gpu_index'] for r in result], ['0', '1'])
        self.assertNotEqual(result[0]['free_memory_bytes'], result[1]['free_memory_bytes'])
        self.assertIn('actual_min_free_static_budget_bytes', result[0]['estimate'])

    def test_lowest_gpu_rejects_not_average(self):
        with self.assertRaisesRegex(ValueError, 'GPU index=1|GPU_STATIC_KV_POOL_INSUFFICIENT'):
            self.check(shared=True, low=True)

    def test_minimum_and_strict_existing_tasks_reject(self):
        with self.assertRaisesRegex(ValueError, 'STRICT rejects'):
            self.check(shared=False)
        with self.assertRaisesRegex(ValueError, 'GPU 1'):
            self.check(shared=True, minimum=23)

    def test_child_policy_overrides_parent_for_both_modes(self):
        for mode, expected in (('STRICT', '1'), ('SHARED', '0')):
            with tempfile.TemporaryDirectory() as root, \
                    patch.dict(os.environ, {'SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK': 'wrong'}), \
                    patch.object(sys, 'argv', ['launch_debug', '--mode', 'baseline', '--tp', '2',
                        '--gpus', '0,1', '--gpu-mode', mode, '--output', root, '--snapshot-id', 'cpu-only']), \
                    patch('experiments.kv_ec.debug_resources.admit', return_value=[
                        {'gpu_index': str(i), 'gpu_uuid': f'GPU-test-{i}'} for i in range(2)]) as admission, \
                    patch.object(launch_debug, 'check_port', return_value='available'), \
                    patch.object(launch_debug.os, 'execv'):
                launch_debug.main()
                self.assertEqual(os.environ['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK'], expected)
                self.assertIs(admission.call_args.kwargs['shared'], mode == 'SHARED')
            self.assertNotEqual(os.environ.get('SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK'), 'wrong')


class TP2SnapshotTests(unittest.TestCase):
    def test_real_format_tp2_roundtrip_all_rank_xor_and_rs_erasures(self):
        # These are explicitly CPU-generated fixtures; never hardware acceptance.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            a = make_snapshot(root / 'a', request='A', snapshot='a', tp=2)
            b = make_snapshot(root / 'b', request='B', snapshot='b', tp=2)
            am, blocks, _, checks, envelope, header, *_ = hierarchy._load_node(a, 'node0')
            self.assertEqual(len(checks), 2)
            self.assertTrue(all(item['byte_exact'] for item in checks))
            self.assertEqual([block.size for block in blocks], [18874368] * 2)
            self.assertEqual([item['kv_head_range'] for item in header['rank_blocks']], [[0, 4], [4, 8]])
            decoded, restored = hierarchy._decode_node(envelope)
            self.assertEqual(decoded['tp_size'], 2)
            self.assertEqual(len(restored), 2)
            args = SimpleNamespace(output=root / 'result', snapshot=b, node_id='node1',
                transport='tcp', bind_host='127.0.0.1', data_port=19481, timeout=1,
                stripe_id='cpu-only', chunk_id=None, experiment_id='explicit-cpu-fixture', request_metrics=None)
            def receive(host, port, path, **kwargs):
                path.write_bytes(envelope)
                return .001
            with patch.object(hierarchy, 'tcp_receive_file', side_effect=receive):
                report = hierarchy.run_receive(args)
            self.assertEqual(report['state'], 'COMMITTED')
            self.assertEqual(report['tp_size'], 2)
            self.assertTrue(all(report['verification'].values()))
            self.assertFalse(report['shared_node_failure_recovery'])
            self.assertEqual(verify(args.output, b)['state'], 'PASS')
            with self.assertRaisesRegex(ValueError, 'does not match committed D1'):
                verify(args.output, a)
            # Historical v1 omitted the optional compatibility profile.
            # Preserve those exact envelopes and verify without rewriting them.
            original_load = hierarchy._load_node
            def legacy_load(*call_args):
                result = list(original_load(*call_args))
                result[4] = changed_header(result[4], lambda h: h.pop('capture_compatibility'))
                result[5] = {k: v for k, v in result[5].items() if k != 'capture_compatibility'}
                return result
            legacy_args = SimpleNamespace(**{**vars(args), 'output': root / 'legacy-result'})
            with patch.object(hierarchy, '_load_node', side_effect=legacy_load), \
                    patch.object(hierarchy, 'tcp_receive_file', side_effect=receive):
                hierarchy.run_receive(legacy_args)
            self.assertEqual(verify(legacy_args.output, b)['state'], 'PASS')
            for mutate in (
                lambda h: h.update(tp_size=4),
                lambda h: h['rank_blocks'][1].update(kv_head_range=[2, 4]),
                lambda h: h['rank_metadata'].reverse(),
            ):
                with self.assertRaises(ValueError):
                    hierarchy._decode_node(changed_header(envelope, mutate))
            with self.assertRaises(ValueError):
                hierarchy._decode_node(envelope[:-1])
            other = make_snapshot(root / 'tp4', request='C', snapshot='c', tp=4)
            bm, *_ = hierarchy._load_node(other, 'node2')
            with self.assertRaisesRegex(ValueError, 'tp_size|packed_byte_count'):
                hierarchy._compatible(am, bm)


if __name__ == '__main__':
    unittest.main()
