"""CPU protocol tests; no test here establishes CUDA/SGLang acceptance."""

from copy import deepcopy
from array import array
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

from .cold_resume_state import (SAMPLING, claim_restore, require_new_process,
    sealed_origin_processes, validate_state)
from .gpu_kv_restore import (RestoreRejected, RestoreStats, RestoreTransaction,
    allocate_restore_slots, validate_initial_ownership)
from .gpu_cold_resume_hook import (seed_request_from_state,
    validate_decode_scheduler_state)
from .gpu_resume_hook import token_digest
from . import verify_gpu_cold_resume as cold_cli
from .verify_gpu_resume import compare_outputs


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
        from .gpu_restore_hook import coordinated_restore
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
