"""CPU regression for TP4; no test in this module proves CUDA or RDMA PASS."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from . import gpu_admission, launch_online, perf_preflight, perf_benchmark
from . import verify_gpu_cold_resume as cold
from .gpu_kv_restore import RestoreRejected, _metadata_identity, _validate_segments
from .gpu_memory_budget import GIB, check_shared_gpu_budget
from .gpu_restore_hook import GPURestoreHook
from .node_block_export import canonical, pack, sha, unpack
from .test_debug_capture import rank_fixture
from .test_gpu_cold_resume import ColdStateProtocolTests
from .test_gpu_memory_budget import GPUMemoryBudgetTests
from .xor_codec import encode, recover


EXAMPLE = Path(__file__).with_name('perf_stage_e_232.example.json')


def config(tp=4, selected='shared'):
    cfg = json.loads(EXAMPLE.read_text())
    cfg.update(tp_size=tp, gpu_admission={'mode': selected})
    for node in cfg['nodes']:
        node['gpus'] = ','.join(map(str, range(tp)))
    return cfg


class TP4AdmissionTests(unittest.TestCase):
    def test_rank_report_gate_rejects_each_missing_rank_and_extra_rank(self):
        from .verify_gpu_restore import require_complete_rank_reports
        for tp in (2, 4):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for rank in range(tp):
                    (root / f'rank_{rank}.json').write_text('{}')
                require_complete_rank_reports(root, tp)
                for rank in range(tp):
                    path = root / f'rank_{rank}.json'
                    path.unlink()
                    with self.subTest(tp=tp, missing=rank), self.assertRaises(FileNotFoundError):
                        require_complete_rank_reports(root, tp)
                    path.write_text('{}')
                (root / f'rank_{tp}.json').write_text('{}')
                with self.assertRaisesRegex(RestoreRejected, 'complete_tp_rank_reports_required'):
                    require_complete_rank_reports(root, tp)

    def test_tp_template_selection_preserves_literal_paths(self):
        cfg = config()
        literal = cfg['nodes'][0]['template']
        self.assertEqual(perf_preflight.template_path(cfg, 0), literal)
        for tp in (2, 4):
            cfg['tp_size'] = tp
            for node in range(2):
                cfg['nodes'][node]['gpus'] = ','.join(map(str, range(tp)))
                cfg['nodes'][node]['template'] = f'/tmp/real-tp{{tp_size}}-node{node}/node.bin'
                self.assertEqual(perf_preflight.template_path(cfg, node), f'/tmp/real-tp{tp}-node{node}/node.bin')
            perf_preflight.validate_config(cfg)
        cfg['nodes'][0]['template'] = '/tmp/{unknown}/node.bin'
        with self.assertRaisesRegex(ValueError, 'unsafe_path'):
            perf_preflight.validate_config(cfg)

    def test_runtime_device_binding_checks_every_rank_and_physical_selection(self):
        resources = [dict(gpu_index=str(i + 4), gpu_uuid=f'GPU-00000000-0000-0000-0000-{i:012d}')
                     for i in range(4)]
        record = dict(gpus='4,5,6,7', gpu_resources_before_start=resources)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=record['gpus']):
            for rank in range(4):
                scheduler = NS(model_worker=NS(ps=NS(tp_rank=rank, tp_size=4, gpu_id=rank),
                    model_runner=NS(token_to_kv_pool=NS(k_buffer=[NS(device=torch.device('cuda', rank))]))))
                with patch('torch.cuda.get_device_properties', return_value=NS(uuid=resources[rank]['gpu_uuid'])):
                    binding = gpu_admission.runtime_rank_binding(scheduler, resources)
                gpu_admission.validate_rank_binding(binding, record, rank)
                for field, value in (('tp_rank', (rank + 1) % 4), ('gpu_index', '999'),
                                     ('runtime_gpu_uuid', 'GPU-wrong'), ('cuda_device', 'cpu')):
                    with self.subTest(rank=rank, field=field), self.assertRaises(ValueError):
                        gpu_admission.validate_rank_binding(dict(binding, **{field: value}), record, rank)
                with patch('torch.cuda.get_device_properties', return_value=NS(uuid='GPU-wrong')):
                    with self.assertRaisesRegex(ValueError, 'UUID_BINDING'):
                        gpu_admission.runtime_rank_binding(scheduler, resources)

    def test_tp2_default_strict_and_all_four_combinations(self):
        old = config(2)
        del old['gpu_admission']
        self.assertEqual(gpu_admission.config_mode(perf_preflight.validate_config(old)), 'strict')
        for tp in (2, 4):
            for mode in ('strict', 'shared'):
                with self.subTest(tp=tp, mode=mode):
                    cfg = config(tp, mode)
                    cfg['nodes'][1]['gpus'] = ','.join(map(str, range(4, 4 + tp)))
                    self.assertEqual(perf_preflight.validate_config(cfg)['tp_size'], tp)
                    self.assertEqual(gpu_admission.config_mode(cfg), mode)

    def test_invalid_tp_mode_and_per_rank_fraction_rejected(self):
        for tp in (True, 2., '4', 1, 3, 8):
            cfg = config()
            cfg['tp_size'] = tp
            with self.subTest(tp=tp), self.assertRaises(ValueError):
                perf_preflight.validate_config(cfg)
        for selected in ('auto', '', 1, None):
            cfg = config()
            cfg['gpu_admission']['mode'] = selected
            with self.subTest(mode=selected), self.assertRaises(ValueError):
                perf_preflight.validate_config(cfg)
        for fraction in ([.35] * 4, {'0': .35}, True, float('nan')):
            cfg = config()
            cfg['mem_fraction_static'] = fraction
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                perf_preflight.validate_config(cfg)

    def test_exact_gpu_count_and_unique_physical_mapping(self):
        for gpus in ('0,1', '0,1,2', '0,1,2,3,4', '0,1,2,2', '0,1,2,GPU-3'):
            cfg = config()
            cfg['nodes'][1]['gpus'] = gpus
            with self.subTest(gpus=gpus), self.assertRaisesRegex(ValueError, 'gpu_indices'):
                perf_preflight.validate_config(cfg)

    def test_four_unequal_busy_gpus_shared_and_each_low_gpu_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model = str(GPUMemoryBudgetTests().model(temp))
            rows = {str(i): dict(total_bytes=t * GIB, free_bytes=f * GIB, uuid=f'GPU-{i}')
                    for i, (t, f) in enumerate(((24, 18), (48, 24), (32, 20), (80, 26)))}
            args = NS(gpus='0,1,2,3', model=model, tp=4, mem_fraction_static=.5,
                      gpu_admission_mode='shared')
            with patch.object(launch_online, 'read_gpu_memory', return_value=rows), \
                    patch.object(launch_online, 'read_gpu_tasks', return_value={'GPU-2': [{'pid': '999'}]}):
                admitted = launch_online.check_gpu_resources(args)
                self.assertEqual(len(admitted), 4)
                gpu_admission.validate_evidence(dict(tp_size=4,
                    **gpu_admission.metadata('shared', .5, admitted)), 'shared', 4)
                for rank in range(4):
                    low = deepcopy(rows)
                    low[str(rank)]['free_bytes'] = 3 * GIB
                    with self.subTest(rank=rank), self.assertRaisesRegex(ValueError, 'INSUFFICIENT'):
                        check_shared_gpu_budget(['0', '1', '2', '3'], model, 4, 4096, .5,
                            memory_rows=low, tasks_by_uuid={}, actual_free_static=True)
                args.gpu_admission_mode = 'strict'
                with self.assertRaisesRegex(ValueError, 'busy'):
                    launch_online.check_gpu_resources(args)
                for row in admitted:
                    details = row['estimate']
                    self.assertEqual(details['local_kv_heads'], 2)
                    self.assertGreater(details['communication_reserve_bytes'], 0)
                    self.assertEqual(details['capture_staging_bytes'], 36 * 2 * 256 * 2 * 128 * 2)
                    self.assertEqual(details['restore_writeback_bytes'], details['capture_staging_bytes'])

    def test_four_rank_launch_rechecks_dynamic_memory_and_does_not_leak_policy(self):
        from . import reprefill_server
        for mode in ('shared', 'strict'):
            with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
                args = NS(tp=4, output=Path(temp) / 'new', model='/model', gpus='0,1,2,3',
                    gpu_admission_mode=mode, mem_fraction_static=.5, port=31050, prefill_chunk_size=-1)
                with patch.object(reprefill_server, 'check_gpu_resources', side_effect=[[],
                        ValueError('INSUFFICIENT_GPU_MEMORY: dynamic change')]), \
                        patch.object(reprefill_server, 'source_identity', return_value={}), \
                        patch.object(reprefill_server, 'check_port'), \
                        patch.object(reprefill_server.os, 'execve') as execute:
                    with self.assertRaisesRegex(ValueError, 'dynamic change'):
                        reprefill_server.launch(args)
                    execute.assert_not_called()
                self.assertNotIn('SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK', os.environ)
                self.assertFalse(args.output.exists())

    def test_config_and_source_mismatch_fail_before_rs(self):
        cfg = config()
        policy = gpu_admission.metadata('shared', .6, [dict(gpu_index=str(i), gpu_uuid=f'GPU-{i}',
            total_memory_bytes=100, free_memory_bytes=90, required_estimated_memory_bytes=80,
            other_tasks=[]) for i in range(4)])
        for mismatch in ('config', 'source'):
            reports = [dict(status='PASS', tp_size=4, config_sha256=sha(canonical(cfg)),
                **policy, checks=dict(inventory=dict(host=f'host{i}', boot_id=f'boot{i}',
                    commit_sha='same', tracked_diff_sha256='same', benchmark_files={}),
                    model_identity=dict(model_sha256='same'))) for i in range(2)]
            if mismatch == 'config':
                reports[1]['config_sha256'] = 'different'
            else:
                reports[1]['checks']['inventory']['benchmark_files'] = {'adapter.py': 'different'}
            def run(argv, **kwargs):
                index = int(argv[argv.index('--node') + 1])
                return NS(returncode=0, stdout='KV_EC_PREFLIGHT ' + json.dumps(reports[index]), stderr='')
            with patch.object(perf_preflight.subprocess, 'run', side_effect=run), \
                    patch('experiments.kv_ec.perf_stage_e.Host.argv', lambda h, m, *a: [m, *map(str, a)]), \
                    patch('experiments.kv_ec.remote_rs_client.RemoteRSClient') as client:
                result = perf_preflight.preflight(cfg, '/tmp/cpu-only-no-write', 1)
                self.assertEqual(result['status'], 'FAIL')
                self.assertIn(mismatch.upper() + '_IDENTITY_MISMATCH', result['error'])
                client.assert_not_called()

    def test_reprefill_rejects_wrong_model_tp_and_prefix_before_request(self):
        info = dict(tp_size=4, disable_radix_cache=True, disable_overlap_schedule=True,
            chunked_prefill_size=-1, max_running_requests=1, attention_backend='triton',
            cuda_graph_config={k: {'backend': 'disabled'} for k in ('decode', 'prefill')},
            model_path='/model')
        for field in ('model', 'tp', 'prefix', 'input_identity'):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                ids = root / 'ids.json'
                ids.write_text(json.dumps([2] * (255 if field == 'prefix' else 256)))
                evidence = root / 'target-run.json'
                evidence.write_text(json.dumps(dict(tp_size=4,
                    model_identity=dict(model_sha256='wrong' if field == 'model' else 'correct'))))
                args = NS(tp=4, input_ids=ids, url='http://unused', model='/model',
                    output=root / 'output', expected_model_sha256='correct', model_identity_evidence=evidence,
                    expected_input_ids_sha256='wrong' if field == 'input_identity' else None)
                wrong_info = dict(info, tp_size=2) if field == 'tp' else info
                with patch.object(perf_benchmark, 'http', return_value=wrong_info), \
                        patch.object(perf_benchmark.urllib.request, 'urlopen') as request:
                    with self.assertRaises(ValueError):
                        perf_benchmark.inference(args)
                    request.assert_not_called()
                    self.assertFalse(args.output.exists())


class TP4LayoutAndVoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays, cls.metas = [], []
        for rank in range(4):
            raw, meta = rank_fixture(rank, 4, layers=range(36), count=256, heads=2)
            meta.update(request_version='version', checkpoint_version='checkpoint', manager_epoch='epoch')
            cls.arrays.append(raw)
            cls.metas.append(meta)

    def test_xor_every_single_rank_byte_exact_and_different_payloads(self):
        self.assertEqual(len({sha(a.tobytes()) for a in self.arrays}), 4)
        parity = encode(self.arrays)
        for lost in range(4):
            restored = recover([None if r == lost else a for r, a in enumerate(self.arrays)], parity)
            np.testing.assert_array_equal(restored, self.arrays[lost])
            _validate_segments(restored.tobytes(), self.metas[lost], token_count=256, tp_size=4)

    def test_unequal_rank_lengths_fail_before_xor_without_truncation(self):
        with self.assertRaisesRegex(ValueError, 'equal byte lengths'):
            encode([*self.arrays[:3], self.arrays[3][:-1]])
        with self.assertRaisesRegex(ValueError, 'equal non-empty rank lengths'):
            from .node_block_export import xor
            xor(*[a.tobytes() for a in self.arrays[:3]], self.arrays[3][:-1].tobytes())

    def test_each_rank_hash_and_gpu_writeback_evidence_fail_closed(self):
        from .verify_gpu_resume import validate_restore_evidence
        for rank in range(4):
            meta = self.metas[rank]
            bad = self.arrays[rank].copy()
            bad[rank] ^= 1
            with self.subTest(rank=rank), self.assertRaisesRegex(RestoreRejected, 'payload_sha256'):
                _validate_segments(bad.tobytes(), meta, token_count=256, tp_size=4)
            evidence = dict(old_slots=[], new_slots=list(range(1, 257)), allocator_size=512,
                stats=dict(loaded_kv_tokens=256, token_count=256, token_start=0, recomputed_tokens=0,
                    unexpected_prefill=False, written_bytes=meta['packed_byte_count'], tp_rank=rank, tp_size=4,
                    layers_verified=[dict(layer_id=x['layer_id'], k_byte_exact=True, v_byte_exact=True,
                        k_sha256=x['k_sha256'], v_sha256=x['v_sha256']) for x in meta['layer_records']]))
            validate_restore_evidence(evidence, meta, rank)
            evidence['stats']['layers_verified'][rank]['v_byte_exact'] = False
            with self.subTest(rank=rank), self.assertRaisesRegex(RestoreRejected, 'layer_bytes_invalid'):
                validate_restore_evidence(evidence, meta, rank)

    def test_all_four_capture_sources_bound_and_production_rejects_fixtures(self):
        raw = [a.tobytes() for a in self.arrays]
        block = pack(0, raw, self.metas, allow_fixture=True)
        self.assertEqual(len(unpack(block, allow_fixture=True)[1]), 4)
        with self.assertRaisesRegex(ValueError, 'real SGLang'):
            unpack(block)
        for rank in range(4):
            for key in ('request_id', 'model_sha256', 'token_ids_sha256', 'request_version', 'checkpoint_version'):
                bad = deepcopy(self.metas)
                bad[rank][key] = 'different'
                with self.subTest(rank=rank, key=key), self.assertRaisesRegex(ValueError, 'mixed rank'):
                    pack(0, raw, bad, allow_fixture=True)

    def test_restore_rejects_wrong_rank_layout_and_model_on_each_rank(self):
        for rank in range(4):
            pool = NS(layer_num=36, kv_cache_layout='nhd', use_hnd=False, is_quantized_kv_cache=False,
                dtype=torch.bfloat16, store_dtype=torch.bfloat16, start_layer=0, page_size=1)
            scheduler = NS(model_worker=NS(ps=NS(tp_size=4, tp_rank=rank), model_runner=NS(
                token_to_kv_pool=pool, model_config=NS(hf_config=NS(model_type='qwen3',
                    num_key_value_heads=8, head_dim=128, hidden_size=4096, num_attention_heads=32)))),
                kv_ec_source_identity={k: self.metas[rank][k] for k in
                                      ('model_sha256', 'source_commit', 'core_diff_sha256')})
            meta = self.metas[rank]
            _metadata_identity(meta, scheduler, NS(), rank=rank, tp_size=4, token_start=0, token_count=256)
            for bad in (dict(meta, tp_rank=(rank + 1) % 4), dict(meta, model_sha256='wrong')):
                with self.assertRaises(RestoreRejected):
                    _metadata_identity(bad, scheduler, NS(), rank=rank, tp_size=4, token_start=0, token_count=256)
            bad = deepcopy(meta)
            bad['layer_records'][0]['packed_shape']['k'] = [256, 4, 128]
            with self.assertRaisesRegex(RestoreRejected, 'packed_shape'):
                _validate_segments(self.arrays[rank].tobytes(), bad, token_count=256, tp_size=4)

    def test_votes_check_identity_and_readiness_of_rank_two_and_three(self):
        for failing in (None, 2, 3):
            hook = GPURestoreHook.__new__(GPURestoreHook)
            hook.tp_size, hook.deadline = 4, time.monotonic() + 10
            hook.run_id, hook.block_hash, hook.evidence = 'cpu-vote', 'hash', {}
            hook.meta, hook.scheduler = {'ranks': [{'token_ids_sha256': 'hash'}]}, NS(tp_cpu_group=object())
            hook._event = Mock()
            def gather(outputs, value, **kwargs):
                self.assertEqual(len(outputs), 4)
                for out in outputs:
                    out.copy_(value)
                if failing is not None:
                    outputs[failing][2] ^= 1
                return NS(wait=lambda **kwargs: True)
            with patch('torch.distributed.all_gather', side_effect=gather):
                self.assertEqual(hook._vote('VERIFIED', True), failing is None)


class TP4ColdStateTests(ColdStateProtocolTests):
    """Run native CPU Req/Scheduler counter tests for each of four rank IDs."""
    def setUp(self):
        super().setUp()
        self.state['tp_size'] = 4
        self.meta['ranks'] = [deepcopy(self.meta['ranks'][0]) for _ in range(4)]

    def test_each_rank_state_binding_is_required(self):
        from .cold_resume_state import validate_state
        for rank in range(4):
            for key in ('req_generation', 'model_sha256', 'token_ids_sha256', 'checkpoint_version'):
                bad = deepcopy(self.meta)
                bad['ranks'][rank][key] = 'wrong'
                with self.subTest(rank=rank, key=key), self.assertRaises(RestoreRejected):
                    validate_state(self.state, bad, 'kv-hash')

    def test_four_rank_sixteen_token_comparison_rejects_any_difference(self):
        from .verify_gpu_resume import compare_outputs
        original = list(range(16))
        logits = [np.zeros((1, 32), dtype=np.float32) for _ in original]
        for rank in range(4):
            self.assertTrue(compare_outputs(original, original, logits, logits)['output_token_ids_match'])
            bad = original.copy()
            bad[rank + 8] += 1
            self.assertFalse(compare_outputs(original, bad, logits, logits)['output_token_ids_match'])

    def test_cold_checker_cannot_ignore_rank_three_or_oom(self):
        # Production read_evidence validates files first; missing/OOM evidence
        # is fatal. No CUDA flag is accepted from these CPU-only stand-ins.
        for failure in (FileNotFoundError('rank_3.json'), RuntimeError('CUDA out of memory')):
            with tempfile.TemporaryDirectory() as temp:
                (Path(temp) / 'run.json').write_text(json.dumps(dict(cold_mode='restore')))
                with patch.object(cold, 'read_evidence', side_effect=failure):
                    with self.assertRaises(type(failure)):
                        cold.check_results(temp)
                report = cold.failure_result(temp, failure)
                self.assertFalse(report['functional_correctness_pass'])
                self.assertFalse(report['isolated_performance_eligible'])


class InferenceIdentityGroupingTests(unittest.TestCase):
    def test_tp_mode_model_input_sampling_and_source_never_pool(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = dict(scheme='reprefill', prefix_tokens=256, tp_size=2,
                gpu_admission_mode='strict', model='/model', input_ids_sha256='input',
                generation_tokens=17, sampling=dict(temperature=0), commit_sha='source')
            variants = [baseline, dict(baseline, tp_size=4),
                dict(baseline, gpu_admission_mode='shared'), dict(baseline, model='/other-model'),
                dict(baseline, input_ids_sha256='different'),
                dict(baseline, sampling=dict(temperature=1)), dict(baseline, commit_sha='other-source')]
            roots = []
            for i, metadata in enumerate(variants):
                path = root / str(i)
                path.mkdir()
                perf_benchmark.save(path / 'metadata.json', metadata)
                perf_benchmark.save(path / 'request-000.json', dict(warmup=False,
                    request_ms=1., prefill_ttft_ms=.5, output_token_ids=[3] * 17,
                    response=dict(meta_info=dict(cached_tokens=0, prompt_tokens=256, completion_tokens=17))))
                roots.append(path)
            args = NS(roots=roots, scheme='reprefill', expected_trials=1, output=root / 'summary.json')
            perf_benchmark.collect_inference(args)
            report = perf_benchmark.read(args.output)
            self.assertEqual(report['rejected'], [])
            self.assertEqual(len(report['rows']), 2 * len(variants))
            self.assertTrue(all(r['n'] == 1 and not r['isolated_performance_eligible'] for r in report['rows']))


if __name__ == '__main__':
    unittest.main()
