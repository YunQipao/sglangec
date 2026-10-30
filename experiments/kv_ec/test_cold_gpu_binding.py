"""CPU regressions for capture identity; no CUDA/RDMA acceptance is claimed."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from . import gpu_admission as admission
from .gpu_cold_resume_hook import GPUColdResumeHook
from .gpu_kv_restore import RestoreRejected
from .gpu_resume_hook import GPUResumeHook
from . import verify_gpu_cold_resume as checker


def fixture(gpus='1,2', node=0, mode='shared'):
    resources = [dict(gpu_index=index,
                      gpu_uuid=f'GPU-{node + 1:08x}-1234-5678-9012-{rank + 1:012x}')
                 for rank, index in enumerate(gpus.split(','))]
    return dict(gpus=gpus, tp_size=2, rank_gpu_binding_required=True,
                gpu_resources_before_start=resources, gpu_admission_mode=mode)


def scheduler(rank):
    return NS(model_worker=NS(ps=NS(tp_rank=rank, tp_size=2, gpu_id=rank),
              model_runner=NS(token_to_kv_pool=NS(
                  k_buffer=[NS(device=torch.device('cuda', rank))]))))


def observe(record, rank):
    with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=record['gpus']), \
            patch('torch.cuda.get_device_properties', return_value=NS(
                uuid=record['gpu_resources_before_start'][rank]['gpu_uuid'].removeprefix('GPU-'))):
        s = scheduler(rank)
        binding = admission.runtime_rank_binding(s, record['gpu_resources_before_start'])
        captured = admission.capture_rank_binding(s, record)
    return binding, captured


class ColdGPUBindingTests(unittest.TestCase):
    def test_tp2_physical_selection_and_rank_order(self):
        # 1,2 / 2,3 reproduce the incident; 1,3 is truly non-contiguous.
        for gpus in ('1,2', '2,3', '1,3', '3,1'):
            for mode in ('shared', 'strict'):
                record = fixture(gpus, mode=mode)
                for rank in (0, 1):
                    with self.subTest(gpus=gpus, mode=mode, rank=rank):
                        binding, captured = observe(record, rank)
                        self.assertEqual(binding['gpu_index'], gpus.split(',')[rank])
                        self.assertEqual(captured['gpu_id'], rank)
                        self.assertEqual(captured['cuda_device'], f'cuda:{rank}')
                        self.assertEqual(admission.validate_capture_rank_binding(
                            captured, binding, record, rank), {})

    def test_nodes_use_their_own_indices_and_uuids(self):
        left, right = fixture('1,2', 0), fixture('2,3', 1)
        for rank in (0, 1):
            a_binding, a_capture = observe(left, rank)
            b_binding, b_capture = observe(right, rank)
            self.assertEqual(admission.validate_capture_rank_binding(
                a_capture, a_binding, left, rank), {})
            self.assertEqual(admission.validate_capture_rank_binding(
                b_capture, b_binding, right, rank), {})
            self.assertIn('gpu_uuid', admission.validate_capture_rank_binding(
                a_capture, b_binding, right, rank))

    def test_missing_capture_evidence_reports_all_fields(self):
        record = fixture()
        binding, captured = observe(record, 0)
        # Original exporter retained only tp_rank/tp_size, with no device fields.
        failures = admission.validate_capture_rank_binding(
            dict(tp_rank=0, tp_size=2), binding, record, 0)
        self.assertEqual(set(failures), {'gpu_id', 'cuda_visible_devices',
                         'cuda_device', 'physical_gpu_index', 'gpu_uuid'})
        self.assertTrue(all(v['actual'] is None for v in failures.values()))

    def test_each_captured_field_is_independently_checked(self):
        record = fixture('3,1')
        binding, captured = observe(record, 1)
        for field, value in (('tp_rank', 0), ('tp_size', 4), ('gpu_id', 3),
                             ('cuda_visible_devices', '1,3'), ('cuda_device', 'cuda:0'),
                             ('physical_gpu_index', '3'),
                             ('gpu_uuid', fixture(node=1)['gpu_resources_before_start'][1]['gpu_uuid'])):
            with self.subTest(field=field):
                failures = admission.validate_capture_rank_binding(
                    dict(captured, **{field: value}), binding, record, 1)
                self.assertEqual(set(failures), {field})
        for value in (None, '', 'unavailable', 'GPU-wrong'):
            self.assertIn('gpu_uuid', admission.validate_capture_rank_binding(
                dict(captured, gpu_uuid=value), binding, record, 1))

    def test_runtime_rejects_wrong_or_missing_uuid(self):
        record = fixture()
        for value in (None, '', 'unavailable', fixture(node=1)['gpu_resources_before_start'][0]['gpu_uuid']):
            with self.subTest(uuid=value), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='1,2'), \
                    patch('torch.cuda.get_device_properties', return_value=NS(uuid=value)):
                with self.assertRaisesRegex(ValueError, 'UUID_BINDING'):
                    admission.runtime_rank_binding(scheduler(0), record['gpu_resources_before_start'])
        binding, captured = observe(record, 0)
        for field in ('gpu_uuid', 'runtime_gpu_uuid'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                admission.validate_rank_binding(dict(binding, **{field: None}), record, 0)
        missing = deepcopy(record)
        missing['gpu_resources_before_start'][0]['gpu_uuid'] = None
        with self.assertRaises(ValueError):
            admission.validate_rank_binding(dict(binding, gpu_uuid=None, runtime_gpu_uuid=None), missing, 0)

    def test_runtime_rejects_rank_reorder_and_wrong_cuda_device(self):
        record = fixture('3,1')
        for rank in (0, 1):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='1,3'), \
                    patch('torch.cuda.get_device_properties', return_value=NS(
                        uuid=record['gpu_resources_before_start'][rank]['gpu_uuid'])):
                with self.assertRaisesRegex(ValueError, 'UUID_BINDING'):
                    admission.runtime_rank_binding(scheduler(rank), record['gpu_resources_before_start'])
            s = scheduler(rank)
            s.model_worker.model_runner.token_to_kv_pool.k_buffer[0].device = torch.device('cuda', 1 - rank)
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=record['gpus']):
                with self.assertRaisesRegex(ValueError, 'DEVICE_BINDING'):
                    admission.runtime_rank_binding(s, record['gpu_resources_before_start'])

    def test_real_cold_stage_seals_fresh_runtime_identity_after_copy_fence(self):
        record = fixture('1,3')
        for rank in (0, 1):
            with tempfile.TemporaryDirectory() as temp:
                hook = GPUColdResumeHook.__new__(GPUColdResumeHook)
                hook.cold_mode, hook.prefill_seen = 'capture', False
                hook.scheduler, hook.rank, hook.run_id = scheduler(rank), rank, 'cpu-regression'
                hook.generation, hook.req = 1, NS()
                hook.cfg = dict(record, capture_bundle=temp)
                hook.evidence = dict(runtime_identity={})
                hook._barrier = lambda phase, action: action()
                item = NS(host=torch.zeros(4, dtype=torch.uint8), metadata=dict(layer_records=[]), done=Mock())
                expected_uuid = record['gpu_resources_before_start'][rank]['gpu_uuid']
                def properties(device):
                    item.done.synchronize.assert_called_once()
                    return NS(uuid=expected_uuid.removeprefix('GPU-'))
                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=record['gpus']), \
                        patch('torch.cuda.Stream'), patch('torch.cuda.get_device_properties', side_effect=properties), \
                        patch('experiments.kv_ec.online_chunk.OnlineProtector._stage', return_value=item), \
                        patch('experiments.kv_ec.gpu_cold_resume_hook.perf_enabled', return_value=False), \
                        patch.object(GPUResumeHook, 'after_forward'):
                    hook.after_forward(NS())
                meta = json.loads((Path(temp) / f'rank_{rank}.meta.json').read_text())
                binding, _ = observe(record, rank)
                self.assertEqual(admission.validate_capture_rank_binding(meta, binding, record, rank), {})
                self.assertEqual((Path(temp) / f'rank_{rank}.bin').read_bytes(), bytes(4))

    def test_checker_fails_closed_with_field_diagnostics_for_old_capture(self):
        record = fixture()
        binding, _ = observe(record, 0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'run.json').write_text(json.dumps(dict(record,
                experiment='gpu-cold-resume-stage-d', cold_mode='capture', capture_bundle='unused',
                origin_server_process={})))
            (root / 'rank_0.json').write_text(json.dumps(dict(rank_gpu_binding=binding)))
            with patch.object(checker, 'load_checkpoint', return_value=(Mock(), dict(tp_size=2),
                    dict(origin_server_process={}), None)), \
                    patch.object(checker, 'sealed_origin_processes'), \
                    patch.object(checker.stage_a, 'require_complete_rank_reports'), \
                    patch('experiments.kv_ec.gpu_restore_hook.validate_checkpoint',
                          return_value=dict(ranks=[dict(tp_rank=0, tp_size=2)])):
                with self.assertRaisesRegex(RestoreRejected, 'cold_capture_gpu_binding_failed:') as exc:
                    checker.read_evidence(root, 'capture')
            details = json.loads(str(exc.exception).split(':', 1)[1])
            self.assertEqual(details['tp_rank'], 0)
            self.assertIn('gpu_uuid', details['fields'])

    def test_capture_rechecks_uuid_instead_of_copying_startup_report(self):
        record = fixture()
        binding, _ = observe(record, 0)
        with tempfile.TemporaryDirectory() as temp:
            hook = GPUColdResumeHook.__new__(GPUColdResumeHook)
            hook.cold_mode, hook.prefill_seen = 'capture', False
            hook.scheduler, hook.rank, hook.run_id = scheduler(0), 0, 'cpu-rejected'
            hook.generation, hook.req = 1, NS()
            hook.cfg = dict(record, capture_bundle=temp)
            hook.evidence = dict(runtime_identity={}, rank_gpu_binding=binding)
            hook._barrier = lambda phase, action: action()
            item = NS(host=torch.zeros(4, dtype=torch.uint8),
                      metadata=dict(layer_records=[]), done=Mock())
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=record['gpus']), \
                    patch('torch.cuda.Stream'), \
                    patch('torch.cuda.get_device_properties', return_value=NS(uuid='unavailable')), \
                    patch('experiments.kv_ec.online_chunk.OnlineProtector._stage', return_value=item), \
                    patch('experiments.kv_ec.gpu_cold_resume_hook.perf_enabled', return_value=False), \
                    patch.object(GPUResumeHook, 'after_forward') as parent:
                with self.assertRaisesRegex(ValueError, 'UUID_BINDING'):
                    hook.after_forward(NS())
                parent.assert_not_called()
            item.done.synchronize.assert_called_once()
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
