"""CPU protocol tests; no test here establishes CUDA/SGLang acceptance."""

from copy import deepcopy
from array import array
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

from experiments.kv_ec.cold_resume_state import (SAMPLING, claim_restore, require_new_process,
    sealed_origin_processes, validate_state)
from experiments.kv_ec.gpu_kv_restore import (RestoreRejected, RestoreStats, RestoreTransaction,
    allocate_restore_slots, validate_initial_ownership)
from experiments.kv_ec.gpu_cold_resume_hook import (seed_request_from_state,
    validate_decode_scheduler_state, GPUColdResumeHook)
from experiments.kv_ec.gpu_resume_hook import token_digest
from experiments.kv_ec import verify_gpu_cold_resume as cold_cli
from experiments.kv_ec.verify_gpu_resume import compare_outputs


def native_methods(relative_path, class_name, names, namespace):
    """Execute unmodified native methods without importing CUDA-only modules."""
    path = Path(__file__).resolve().parents[2] / relative_path
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    methods = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == set(names)
    module = ast.Module(body=ast.parse('from __future__ import annotations').body + methods,
                        type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


class ColdStateProtocolTests(unittest.TestCase):
    def setUp(self):
        ids, request_id = [11] * 256, 'origin-request'
        model = dict(model_sha256='model', source_commit='commit', core_diff_sha256='diff')
        self.state = dict(schema='kv-cold-request-v1', checkpoint_version=1,
            request_version=1, tp_size=2, boundary='post_prefill_result_before_decode',
            origin_input_ids=ids, output_ids=[7], materialized_kv_tokens=256,
            kv_allocated_len=256, pending_token_id=7, pending_token_has_kv=False,
            next_decode_input_ids=[7], next_decode_position=256, next_decode_seq_len=257,
            sampling=SAMPLING, rng_policy='greedy_no_rng', finished=False,
            extend_batch_idx=1, decode_batch_idx=0, decode_steps=16, vocab_size=32000,
            request_id=request_id, restore_id='restore-id', capture_run_id='capture-run',
            token_start=0, token_count=256, node_id=0, input_ids_sha256=token_digest(ids),
            kv_sha256='kv-hash', runtime_identity=model, origin_row=3, origin_generation=4)
        self.meta = dict(node_id=0, ranks=[])
        for rank in (0, 1):
            self.meta['ranks'].append(dict(**model, token_ids_sha256=token_digest(ids),
                snapshot_id='restore-id', checkpoint_version='restore-id',
                request_id=hashlib.sha256(request_id.encode()).hexdigest(),
                request_version='capture-run', manager_epoch='capture-run',
                req_pool_idx=3, req_generation=4))

    def test_consistent_post_prefill_state_is_accepted(self):
        self.assertEqual(validate_state(self.state, self.meta, 'kv-hash'), self.state)

    def test_version_pending_token_and_materialized_length_are_required(self):
        cases = [
            ({**self.state, 'checkpoint_version': 99}, 'cold_state_version_or_boundary_mismatch'),
            ({k: v for k, v in self.state.items() if k != 'pending_token_id'},
             'cold_state_required_field_missing'),
            ({**self.state, 'output_ids': []}, 'cold_token_sequence_invalid'),
            ({**self.state, 'pending_token_has_kv': True},
             'cold_pending_token_or_kv_length_mismatch'),
            ({**self.state, 'materialized_kv_tokens': 255},
             'cold_pending_token_or_kv_length_mismatch'),
            ({**self.state, 'next_decode_position': 255},
             'cold_pending_token_or_kv_length_mismatch'),
        ]
        for damaged, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(RestoreRejected, reason):
                validate_state(damaged, self.meta, 'kv-hash')

    def test_tp_row_generation_or_token_digest_disagreement_is_rejected(self):
        broken = deepcopy(self.meta)
        broken['ranks'][1]['req_generation'] += 1
        with self.assertRaisesRegex(RestoreRejected, 'cold_state_rank_identity_mismatch'):
            validate_state(self.state, broken, 'kv-hash')
        broken = deepcopy(self.state)
        broken['origin_input_ids'][0] += 1
        with self.assertRaisesRegex(RestoreRejected, 'cold_state_kv_binding_mismatch'):
            validate_state(broken, self.meta, 'kv-hash')

    def test_restore_claim_is_one_shot(self):
        with tempfile.TemporaryDirectory() as path:
            claim_restore(Path(path), 'a' * 64, 'run-one')
            with self.assertRaisesRegex(RestoreRejected, 'cold_checkpoint_already_claimed'):
                claim_restore(Path(path), 'a' * 64, 'run-two')

    def test_new_instance_identity_includes_server_launcher_process(self):
        ranks = [dict(host='host', boot_id='boot', pid=i, start_ticks=i * 10)
                 for i in (11, 12)]
        server = dict(host='host', boot_id='boot', pid=13, start_ticks=130)
        origins = sealed_origin_processes(dict(origin_processes=ranks,
                                                origin_server_process=server))
        require_new_process(dict(host='host', boot_id='boot', pid=21, start_ticks=210), origins)
        with self.assertRaisesRegex(RestoreRejected, 'cold_requires_new_same_host_process'):
            require_new_process(server, origins)

    def test_seeded_req_restores_extend_count_but_not_pending_token_kv(self):
        class FreshReq:
            def __init__(self, origin):
                self.rid = 'origin-request'
                self.origin_input_ids = array('q', origin)
                self.output_ids = array('q')
                self.full_untruncated_fill_ids = array('q')
                self.kv_committed_len = 0
                self.extend_batch_idx = self.decode_batch_idx = 0

            def _refresh_fill_ids(self):
                self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

            def set_extend_range(self, start, end):
                self.extend_range = (start, end)

        req = FreshReq(self.state['origin_input_ids'])
        pending = seed_request_from_state(req, self.state)
        self.assertEqual(pending, self.state['pending_token_id'])
        self.assertEqual(req.extend_range, (0, 256))
        self.assertEqual((req.extend_batch_idx, req.decode_batch_idx), (1, 0))
        self.assertEqual(req.kv_committed_len, 0)
        self.assertEqual(list(req.output_ids), [pending])
        self.assertEqual(list(req.full_untruncated_fill_ids),
                         self.state['origin_input_ids'] + [pending])

    def test_first_decode_uses_restored_prefill_counter_and_pending_token(self):
        req = NS(extend_batch_idx=1, decode_batch_idx=1, output_ids=[7],
            origin_input_ids=self.state['origin_input_ids'],
            full_untruncated_fill_ids=self.state['origin_input_ids'] + [7],
            kv_committed_len=257)
        batch = NS(forward_mode=NS(name='DECODE'), seq_lens_cpu=torch.tensor([257]))
        validate_decode_scheduler_state(req, batch, self.state, 1)
        req.extend_batch_idx = 0
        with self.assertRaisesRegex(RestoreRejected, 'cold_decode_scheduler_state_mismatch'):
            validate_decode_scheduler_state(req, batch, self.state, 1)

    def test_three_native_decode_rounds_preserve_lazy_fill_and_pending_relay(self):
        # These are the actual Req, allocator, prepare, FutureMap and result
        # methods, not a simulation that manually increments scheduler counters.
        from experiments.kv_ec.test_scheduler_pause import request_pool_class
        base = 'python/sglang/srt/'
        ns = dict(torch=torch, _is_npu=False, _is_cuda=False, _is_hip=False,
            _DEBUG_ASSERT=False, ForwardMode=NS(DECODE=NS(name='DECODE')),
            Range=lambda start, end: NS(start=start, end=end),
            mamba_extra_buffer_enabled=lambda: False,
            get_observability=lambda: NS(enable_metrics=False),
            evict_from_tree_cache=lambda cache, count: None)
        native_methods(base + 'mem_cache/allocation.py', None,
                       ['alloc_for_decode', 'alloc_token_slots', '_alloc_page_size'], ns)
        req_methods = native_methods(base + 'managers/schedule_batch.py', 'Req',
            ['_refresh_fill_ids', 'set_extend_range', 'finished', 'update_finish_state'], ns)
        native_methods(base + 'managers/schedule_batch.py', 'ScheduleBatch', ['prepare_for_decode'], ns)
        relay_methods = native_methods(base + 'managers/overlap_utils.py', 'FutureMap', ['stash'], ns)
        native_methods(base + 'managers/overlap_utils.py', None, ['resolve_forward_inputs'], ns)
        processor_methods = native_methods(base + 'managers/scheduler_components/batch_result_processor.py',
            'SchedulerBatchResultProcessor', ['process_batch_result_decode', '_normalize_decode_outputs'], ns)
        req = type('NativeReq', (), req_methods)()
        req.rid, req.origin_input_ids = self.state['request_id'], array('q', self.state['origin_input_ids'])
        req.output_ids, req.full_untruncated_fill_ids = array('q'), array('q')
        req.kv_committed_len = 0
        req.finished_reason = req.to_finish = req.grammar = None
        req.sampling_params = NS(max_new_tokens=17)
        # Unsupported stop policies are already rejected by the Stage D gate.
        req._check_vocab_boundary_finish = req._check_token_based_finish = lambda tokens: False
        req._check_str_based_finish = lambda count: False
        req.time_stats = NS(set_last_decode_finish_time=lambda: None)
        req.return_logprob = req.return_sampling_mask = req.return_hidden_states = False
        seed_request_from_state(req, self.state)
        allocator = TokenToKVPoolAllocator(512, torch.bfloat16, 'cpu', NS(), need_sort=False)
        pool = request_pool_class().__new__(request_pool_class())
        pool.req_to_token = torch.zeros((2, 512), dtype=torch.int32)
        pool.req_generation = torch.tensor([0, 1])
        pool.free_slots = []
        req.req_pool_idx = 1
        req.is_retracted = False
        # Initial committed install: real CPU slots, no model forward claimed.
        pool.req_to_token[1, :256] = allocator.alloc(256).to(torch.int32)
        req.kv_committed_len, req.kv = 256, NS(kv_allocated_len=256)
        spec = NS(is_none=lambda: True, is_ngram=lambda: False)
        batch = NS(reqs=[req], spec_algorithm=spec, model_config=NS(is_encoder_decoder=False),
            sampling_info=NS(penalizer_orchestrator=NS(is_required=False)),
            seq_lens=torch.tensor([256]), seq_lens_cpu=torch.tensor([256]),
            orig_seq_lens=torch.tensor([256]), req_pool_indices=torch.tensor([1]),
            req_to_token_pool=pool, tree_cache=NS(page_size=1, token_to_kv_pool_allocator=allocator),
            maybe_evict_swa=lambda: None, hisparse_coordinator=None,
            prefill_input_ids_cpu=None, input_ids=None, enable_overlap=False, return_logprob=False)
        relay = type('NativeFutureMap', (), relay_methods)()
        relay.spec_algo, relay.output_tokens_buf = spec, torch.full((2,), -1, dtype=torch.int64)
        relay._forward_buf_initialized = True
        relay.need_topk = relay.need_hidden_states = False
        relay.draft_probs_buf = relay.dsa_topk_indices_buf = None
        relay._maybe_init_dsa_topk_indices_buf = lambda payload: None
        relay.stash(batch.req_pool_indices, NS(bonus_tokens=torch.tensor([self.state['pending_token_id']])))
        processor = type('NativeResultProcessor', (), processor_methods)()
        processor.enable_overlap = processor.enable_overlap_mlx = False
        processor.token_to_kv_pool_allocator = allocator
        processor.metrics_reporter = NS(num_generated_tokens=0, forward_ct_decode=0,
            report_decode_stats=lambda *args, **kwargs: None)
        processor._maybe_update_reasoning_tokens = lambda *args: None
        processor._handle_finish_state_updated_req = lambda *args: None
        processor.output_streamer = NS(stream_output=lambda *args: None)
        with tempfile.TemporaryDirectory() as temp:
            hook = GPUColdResumeHook.__new__(GPUColdResumeHook)
            hook.req, hook.state, hook.row, hook.generation, hook.rank = req, self.state, 1, 1, 0
            hook.cold_mode, hook.admitted, hook.done, hook.timeout = 'restore', True, False, 120
            hook.sampled_seed, hook.meta = self.state['pending_token_id'], self.meta
            hook.prefix_slots = tuple(pool.req_to_token[1, :256].tolist())
            hook.output = Path(temp)
            hook.scheduler = NS(req_to_token_pool=pool, future_map=relay, forward_ct=0,
                                token_to_kv_pool_allocator=allocator)
            hook.evidence = dict(resume_decode_forward_count=0, forward_observations=[],
                                 committed=True, scheduler_injected=True)
            phases = []
            def barrier(phase, action):
                phases.append(phase)
                return action()  # collectives are tested separately, not GPU evidence
            hook._barrier = barrier
            for step in range(1, 4):
                pending = req.output_ids[-1]
                ns['prepare_for_decode'](batch)
                hook.before_forward(batch)
                self.assertEqual(req.decode_batch_idx, step)
                self.assertEqual(req.kv_committed_len, 256 + step)  # scheduled, not materialized yet
                self.assertEqual(hook.evidence['resume_decode_forward_count'], step - 1)
                ns['resolve_forward_inputs'](batch, relay)
                self.assertEqual(batch.input_ids.tolist(), [pending])
                # CPU model stand-in completes one forward and samples a token.
                # The actual CUDA observer is separately required by GPU acceptance.
                sampled = batch.input_ids + 1
                hook.scheduler.forward_ct += 1
                hook.evidence['resume_decode_forward_count'] += 1
                hook.evidence['forward_observations'].append(dict(kv_length=int(batch.seq_lens_cpu[0]),
                    input_ids=batch.input_ids.tolist(), positions=[int(batch.seq_lens_cpu[0]) - 1]))
                relay.stash(batch.req_pool_indices, NS(bonus_tokens=sampled))
                batch.input_ids = None  # native run_batch relay contract
                hook.after_forward(batch)
                result = NS(copy_done=None, routed_experts_output=None, indexer_topk_output=None,
                    logits_output=None, next_token_ids=sampled, can_run_cuda_graph=False,
                    num_correct_drafts=0)
                processor.process_batch_result_decode(batch, result)
                hook.after_result(batch)
                self.assertEqual(list(req.output_ids), list(range(7, 8 + step)))
                self.assertEqual(len(req.full_untruncated_fill_ids), 257)
                self.assertNotEqual(list(req.full_untruncated_fill_ids),
                    list(req.origin_input_ids) + list(req.output_ids))  # old assertion is invalid
                self.assertFalse(req.finished())
            entries = [json.loads(line) for line in (Path(temp) / 'rank_0.decode_state.jsonl').read_text().splitlines()]
            self.assertEqual(len(entries), 9)
            self.assertTrue(all(f['matches'] for e in entries for f in e['fields'].values()))
            for index, step in enumerate(range(1, 4)):
                self.assertEqual(phases[index * 3:index * 3 + 3], [f'READY_{step}',
                    f'COLD_FORWARD_STATE_{step}', 'RESULT_' + token_digest(req.output_ids[:step + 1])])
            # A poisoned relay must fail with the actual and expected token.
            relay.output_tokens_buf[1] = -1
            with self.assertRaisesRegex(RestoreRejected, 'future_map_pending_token.*actual.*-1.*expected.*10'):
                hook._validate_decode_state(batch, 3, 'RESULT')

    def test_lazy_fill_is_still_validated_and_wrong_counters_are_named(self):
        req = NS(extend_batch_idx=1, decode_batch_idx=2, output_ids=[7, 8],
            origin_input_ids=self.state['origin_input_ids'],
            full_untruncated_fill_ids=self.state['origin_input_ids'] + [7], kv_committed_len=258)
        batch = NS(forward_mode=NS(name='DECODE'), seq_lens_cpu=torch.tensor([258]))
        validate_decode_scheduler_state(req, batch, self.state, 2)
        for field, value, diagnostic in (
            ('decode_batch_idx', 1, 'decode_batch_idx'),
            ('kv_committed_len', 257, 'kv_committed_len_scheduled'),
            ('output_ids', [7], 'output_ids_len'),
            ('full_untruncated_fill_ids', self.state['origin_input_ids'] + [9], 'cached_fill_sha256'),
            ('full_untruncated_fill_ids', self.state['origin_input_ids'], 'cached_fill_len'),
        ):
            with self.subTest(field=field, value=value):
                broken = deepcopy(req)
                setattr(broken, field, value)
                with self.assertRaisesRegex(RestoreRejected, 'cold_decode_scheduler_state_mismatch:.*' + diagnostic):
                    validate_decode_scheduler_state(broken, batch, self.state, 2)


class InitialInstallTransactionTests(unittest.TestCase):
    def fixture(self):
        req_pool = NS(device='cpu', free_slots=[], req_generation=torch.tensor([0, 9]),
            req_to_token=torch.full((2, 4), 77, dtype=torch.int32))
        req_pool.req_to_token[0] = 0
        req = NS(req_pool_idx=1, kv_committed_len=0,
            kv=NS(kv_allocated_len=0), cache_protected_len=0,
            prefix_indices=torch.empty(0, dtype=torch.int64))
        allocator = TokenToKVPoolAllocator(16, torch.bfloat16, 'cpu', NS(), need_sort=False)
        scheduler = NS(req_to_token_pool=req_pool, token_to_kv_pool_allocator=allocator)
        return scheduler, req

    def transaction(self, scheduler, req):
        old = validate_initial_ownership(scheduler, req)
        row, a, rp = req.req_pool_idx, scheduler.token_to_kv_pool_allocator, scheduler.req_to_token_pool
        generation = int(rp.req_generation[row])
        new, journal = allocate_restore_slots(a, 2, old)
        return RestoreTransaction(a, rp, req, row, generation, old, new, 0, 2,
            RestoreStats('cpu-fixture', 0, 0, 2, 0, 2, 2, 0), state='VERIFIED',
            savepoint=journal, initial_install=True,
            prior_mapping=tuple(rp.req_to_token[row, :2].tolist()))

    def cold_hook(self, rank=0, inject_rank_failure=None):
        # CPU observer/coordinator tests only: a mocked restore_to_gpu return
        # exercises evidence recording, never establishes actual CUDA execution.
        scheduler, req = self.fixture()
        tx = self.transaction(scheduler, req)
        hook = GPUColdResumeHook.__new__(GPUColdResumeHook)
        hook.cold_mode, hook.rank = 'restore', rank
        hook.scheduler, hook.req, hook.tx = scheduler, req, None
        hook.source, hook.source_name = object(), 'cpu-mocked-verified-return'
        hook.cfg = dict(inject_rank_failure=inject_rank_failure)
        hook.evidence = dict(cuda_executed=False, verified=False, committed=False,
            local_cold_resume_complete=False, resume_decode_forward_count=0)
        hook._event = Mock()
        return hook, tx

    def test_verified_callback_records_cuda_evidence_after_prepare_returns(self):
        from experiments.kv_ec.gpu_restore_hook import coordinated_restore
        hook, tx = self.cold_hook()
        def prepare(*args, **kwargs):
            self.assertFalse(hook.evidence['cuda_executed'])
            self.assertFalse(hook.evidence['verified'])
            return tx
        with patch('experiments.kv_ec.gpu_cold_resume_hook.restore_to_gpu', side_effect=prepare) as restore:
            result = coordinated_restore(hook._prepare_restore, lambda phase, ready: ready, hook._observe_tx)
        restore.assert_called_once_with(hook.scheduler, hook.req, hook.source,
            initial_install=True, source_name=hook.source_name)
        self.assertIs(result, tx)
        self.assertTrue(hook.evidence['cuda_executed'])
        self.assertTrue(hook.evidence['verified'])
        self.assertEqual([call.args[0] for call in hook._event.call_args_list],
                         ['VERIFIED', 'MAPPED', 'COMMITTED', 'FINISHED'])

    def test_byte_verification_failure_never_records_cuda_evidence(self):
        from experiments.kv_ec.gpu_restore_hook import coordinated_restore
        hook, tx = self.cold_hook()
        def fail(*args, **kwargs):
            tx.abort()  # restore_to_gpu owns rollback before raising
            raise RestoreRejected('gpu_kv_writeback_byte_mismatch')
        with patch('experiments.kv_ec.gpu_cold_resume_hook.restore_to_gpu', side_effect=fail):
            with self.assertRaisesRegex(RestoreRejected, 'tp_restore_verified_failed'):
                coordinated_restore(hook._prepare_restore, lambda phase, ready: ready, hook._observe_tx)
        self.assertFalse(hook.evidence['cuda_executed'])
        self.assertFalse(hook.evidence['verified'])
        hook._event.assert_not_called()
        self.assertEqual(tx.state, 'ABORTED')
        self.assertEqual(tx.allocator.available_size(), tx.allocator.size)

    def test_unverified_transaction_is_rejected_before_observer(self):
        from experiments.kv_ec.gpu_restore_hook import coordinated_restore
        hook, tx = self.cold_hook()
        tx.state = 'PREPARED'
        with patch('experiments.kv_ec.gpu_cold_resume_hook.restore_to_gpu', return_value=tx):
            with self.assertRaisesRegex(RestoreRejected, 'tp_restore_verified_failed'):
                coordinated_restore(hook._prepare_restore, lambda phase, ready: ready, hook._observe_tx)
        self.assertFalse(hook.evidence['cuda_executed'])
        hook._event.assert_not_called()
        self.assertEqual(tx.state, 'ABORTED')

    def test_commit_failure_keeps_local_verified_fact_without_claiming_success(self):
        from experiments.kv_ec.gpu_restore_hook import coordinated_restore
        hook, tx = self.cold_hook()
        with patch('experiments.kv_ec.gpu_cold_resume_hook.restore_to_gpu', return_value=tx), \
                patch.object(tx, 'commit', side_effect=RuntimeError('injected_commit_failure')):
            with self.assertRaisesRegex(RestoreRejected, 'tp_restore_committed_failed'):
                coordinated_restore(hook._prepare_restore, lambda phase, ready: ready, hook._observe_tx)
        # CUDA verification is a local completed fact, not global COMMIT/PASS.
        self.assertTrue(hook.evidence['cuda_executed'])
        self.assertTrue(hook.evidence['verified'])
        self.assertFalse(hook.evidence['committed'])
        self.assertFalse(hook.evidence['local_cold_resume_complete'])
        self.assertEqual(tx.state, 'ABORTED')
        self.assertEqual(tx.allocator.available_size(), tx.allocator.size)

    def test_single_rank_injection_records_only_peer_verification_and_aborts_both(self):
        from experiments.kv_ec.gpu_restore_hook import coordinated_restore
        pairs = [self.cold_hook(rank, inject_rank_failure=1) for rank in (0, 1)]
        prepared = {id(hook.scheduler): tx for hook, tx in pairs}
        barrier, votes = threading.Barrier(2, timeout=10), {}
        def run(rank):
            hook, tx = pairs[rank]
            def vote(phase, ready):
                votes[rank] = ready
                barrier.wait()
                unanimous = all(votes.values())
                barrier.wait()
                return unanimous
            with self.assertRaisesRegex(RestoreRejected, 'tp_restore_verified_failed'):
                coordinated_restore(hook._prepare_restore, vote, hook._observe_tx)
        with patch('experiments.kv_ec.gpu_cold_resume_hook.restore_to_gpu',
                   side_effect=lambda scheduler, *args, **kwargs: prepared[id(scheduler)]):
            with ThreadPoolExecutor(max_workers=2) as workers:
                list(workers.map(run, (0, 1)))
        self.assertTrue(pairs[0][0].evidence['cuda_executed'])
        self.assertFalse(pairs[1][0].evidence['cuda_executed'])
        pairs[1][0]._event.assert_not_called()
        for hook, tx in pairs:
            self.assertFalse(hook.evidence['committed'])
            self.assertFalse(hook.evidence['local_cold_resume_complete'])
            self.assertEqual(hook.evidence['resume_decode_forward_count'], 0)
            self.assertEqual(tx.state, 'ABORTED')
            self.assertEqual(tx.allocator.available_size(), tx.allocator.size)

    def test_other_transaction_events_do_not_infer_cuda_from_decode_or_commit(self):
        hook, tx = self.cold_hook()
        hook.evidence['resume_decode_forward_count'] = 16
        for phase in ('MAPPED', 'COMMITTED', 'FINISHED'):
            hook._observe_tx(phase, NS(state='COMMITTED'))
            self.assertFalse(hook.evidence['cuda_executed'])
            self.assertFalse(hook.evidence['verified'])
        tx.abort()

    def test_first_install_commits_without_fabricating_or_freeing_old_slots(self):
        scheduler, req = self.fixture()
        tx = self.transaction(scheduler, req)
        tx.commit()
        self.assertEqual((req.kv_committed_len, req.kv.kv_allocated_len), (2, 2))
        self.assertEqual(scheduler.req_to_token_pool.req_to_token[1, :2].tolist(), list(tx.new_slots))
        self.assertEqual(tx.old_slots, ())
        self.assertNotIn(0, scheduler.token_to_kv_pool_allocator.free_pages.tolist())

    def test_first_install_abort_restores_stale_row_bytes_and_zero_lengths(self):
        scheduler, req = self.fixture()
        tx = self.transaction(scheduler, req)
        tx.install_mapping()
        tx.abort()
        self.assertEqual(scheduler.req_to_token_pool.req_to_token[1, :2].tolist(), [77, 77])
        self.assertEqual((req.kv_committed_len, req.kv.kv_allocated_len), (0, 0))
        self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 16)

    def test_initial_install_rejects_a_row_with_existing_kv(self):
        scheduler, req = self.fixture()
        req.kv_committed_len = 1
        with self.assertRaisesRegex(RestoreRejected, 'cold_restore_request_already_has_kv'):
            validate_initial_ownership(scheduler, req)

    def test_single_rank_prepare_failure_compensates_peer_before_any_decode(self):
        # Mirrors the existing Stage B fixed-order coordinator with one local
        # prepare failure: both peers vote false and the prepared side rolls back.
        from experiments.kv_ec.gpu_restore_hook import coordinated_restore
        transactions = [self.transaction(*self.fixture()) for _ in range(2)]
        local = {'rank': 0}
        def prepare():
            tx = transactions[local['rank']]
            if local['rank'] == 1:
                tx.abort()
                raise RestoreRejected('injected_single_rank_failure')
            return tx
        # A single local coordinator cannot faithfully emulate two collectives;
        # assert its prepare failure compensates the local transaction and never
        # reaches a mapping/commit callback. The TP collective path is exercised
        # by Stage B's two-thread test below.
        local['rank'] = 1
        with self.assertRaisesRegex(RestoreRejected, 'verified_failed'):
            coordinated_restore(prepare, lambda phase, ready: ready)
        self.assertEqual(transactions[1].state, 'ABORTED')
        self.assertFalse(transactions[1].allocator._kv_ec_restore_active)


class ColdComparisonTests(unittest.TestCase):
    def test_failure_check_verdict_controls_exit_only_for_check_failure(self):
        negative = dict(pass_stage_d=False, negative_test_pass=True)
        self.assertTrue(cold_cli.check_exit_success('check-failure', negative))
        self.assertFalse(cold_cli.check_exit_success('check', negative))
        self.assertFalse(cold_cli.check_exit_success('reference', negative))
        self.assertFalse(cold_cli.check_exit_success('check-failure',
                                                     dict(pass_stage_d=False,
                                                          negative_test_pass=False)))
        self.assertFalse(cold_cli.check_exit_success('check',
                                                     dict(pass_stage_d=False,
                                                          pass_reference=False,
                                                          negative_test_pass=True)))
        self.assertTrue(cold_cli.check_exit_success('check', dict(pass_stage_d=True)))
        self.assertTrue(cold_cli.check_exit_success('reference', dict(pass_reference=True)))

    def test_difference_reports_full_output_and_decode_positions(self):
        reference = [np.array([[0.0, 1.0, 2.0]])]
        resumed = [np.array([[0.0, 2.0, 1.0]])]
        result = compare_outputs([1], [2], reference, resumed,
                                output_offset=1, step_offset=1)
        self.assertFalse(result['output_token_ids_match'])
        self.assertEqual(result['first_difference']['output_index'], 1)
        self.assertEqual(result['first_difference']['decode_step'], 1)
        self.assertEqual(result['logits_errors'][0]['step'], 1)

    def test_capture_submit_uses_run_id_before_checkpoint_is_sealed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with (output / 'run.json').open('x') as handle:
                json.dump(dict(experiment='gpu-cold-resume-stage-d', cold_mode='capture',
                    run_id='origin-rid', input_ids=[11] * 256), handle)
            for rank in (0, 1):
                with (output / f'rank_{rank}.json').open('x') as handle:
                    json.dump(dict(failure_reason=None), handle)
            args = NS(output=output, url='http://unused', wait_ready=1, timeout=1)
            with patch.object(cold_cli.stage_a, 'submit') as send:
                cold_cli.submit(args)
            self.assertEqual(send.call_args.kwargs['request_id'], 'origin-rid')


if __name__ == '__main__':
    unittest.main()
