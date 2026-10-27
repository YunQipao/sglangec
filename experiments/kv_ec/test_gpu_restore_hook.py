"""CPU exception/coordination tests; these NEVER constitute GPU acceptance."""

import ast
from contextlib import ExitStack, nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import numpy as np

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from .gpu_kv_restore import (
    RestoreRejected, RestoreTransaction, allocate_restore_slots, validate_live_ownership,
)
from .gpu_restore_hook import GPURestoreHook, coordinated_restore, validate_checkpoint
from .test_scheduler_pause import request_pool_class
from .test_debug_capture import lifetime_fixture
from . import test_gpu_kv_restore as restore_tests
from . import gpu_kv_restore as restore_module
from .verify_gpu_restore import check_results, read_input_ids


def transaction(*, need_sort=False):
    allocator = TokenToKVPoolAllocator(16, torch.bfloat16, "cpu", NS(), need_sort)
    old = tuple(allocator.alloc(2).tolist())
    allocator.enable_slot_leases(active_slots=old)
    cls = request_pool_class()
    rp = cls.__new__(cls)
    rp._alloc_size, rp.size, rp.device = 2, 1, "cpu"
    rp.free_slots = []
    rp.req_generation = torch.tensor([0, 7])
    rp.req_to_token = torch.zeros((2, 8), dtype=torch.int32)
    rp.req_to_token[1, :2] = torch.tensor(old)
    req = NS(req_pool_idx=1, kv_committed_len=2, kv=NS(kv_allocated_len=2),
             cache_protected_len=0, inflight_middle_chunks=0)
    owner = "cpu-test"
    rp.protect_kv_ec_row(1, 7, owner)
    new, savepoint = allocate_restore_slots(allocator, 2, old)
    tx = RestoreTransaction(allocator, rp, req, 1, 7, old, new, 0, 2, NS(),
        state="VERIFIED", savepoint=savepoint, guard_owner=owner)
    scheduler = NS(req_to_token_pool=rp, token_to_kv_pool_allocator=allocator)
    tx.ownership_check = lambda: validate_live_ownership(scheduler, req, 0, 2,
        reserved=tx.old_slots if tx.mapping_installed else tx.new_slots)
    return tx, scheduler


def free_slots(a):
    return a.free_pages.tolist() + a.release_pages.tolist()


class AllocationFailureTests(unittest.TestCase):
    def allocator(self):
        a = TokenToKVPoolAllocator(16, torch.bfloat16, "cpu", NS(), need_sort=False)
        a.enable_slot_leases()
        return a

    def test_allocator_exception_before_and_after_mutation_rolls_back(self):
        for after in (False, True):
            with self.subTest(after_mutation=after):
                a = self.allocator()
                original = a.alloc
                def fail(n):
                    if after:
                        original(n)
                    raise RuntimeError("injected_alloc")
                a.alloc = fail
                with self.assertRaisesRegex(RuntimeError, "injected_alloc"):
                    allocate_restore_slots(a, 2, ())
                self.assertEqual(free_slots(a), list(range(1, 17)))
                self.assertEqual(a.slot_generations([1, 2]), (0, 0))
                self.assertFalse(a._kv_ec_restore_active)

    def test_short_none_duplicate_oob_and_alias_returns_do_not_leak(self):
        for returned in (None, [1], [1, 1], [0, 2], [1, 17], [15, 16], [3, 4], [1.5, 2.5]):
            with self.subTest(returned=returned):
                a = self.allocator()
                original = a.alloc
                def invalid(n):
                    original(n)
                    return None if returned is None else torch.tensor(returned)
                a.alloc = invalid
                with self.assertRaises(RestoreRejected):
                    allocate_restore_slots(a, 2, (15, 16))
                self.assertEqual(free_slots(a), list(range(1, 17)))
                self.assertFalse(a._kv_ec_restore_active)

    def test_free_cat_failure_is_atomic(self):
        a = self.allocator()
        slots = a.alloc(2)
        before = free_slots(a)
        with patch("torch.cat", side_effect=RuntimeError("injected_cat")):
            with self.assertRaisesRegex(RuntimeError, "injected_cat"):
                a.free(slots)
        self.assertEqual(free_slots(a), before)

    def test_lease_generation_validation_is_atomic(self):
        a = self.allocator()
        for bad in ([1, 17], [1, 1]):
            with self.assertRaises(RuntimeError):
                a.slot_lease_registry.on_alloc(bad)
            self.assertEqual(a.slot_generations([1, 2]), (0, 0))


class TransactionFailureTests(unittest.TestCase):
    def test_mapping_cannot_be_changed_without_an_undo_journal(self):
        tx, _ = transaction()
        journal, tx.savepoint = tx.savepoint, None
        with self.assertRaisesRegex(RestoreRejected, "undo_journal"):
            tx.commit()
        self.assertEqual(tuple(tx._mapping().tolist()), tx.old_slots)
        tx.savepoint = journal
        tx.abort()
        self.assert_rolled_back(tx)

    def assert_rolled_back(self, tx):
        self.assertEqual(tx.state, "ABORTED")
        self.assertEqual(tuple(tx._mapping().tolist()), tx.old_slots)
        self.assertEqual(set(free_slots(tx.allocator)), set(range(1, 17)) - set(tx.old_slots))
        self.assertEqual(len(free_slots(tx.allocator)), 14)
        self.assertFalse(tx.allocator._kv_ec_restore_active)
        self.assertIsNone(tx.savepoint.free_pages)
        self.assertIsNone(tx.ownership_check)

    def test_old_slot_free_failure_before_and_after_mutation_compensates(self):
        for need_sort in (False, True):
            for after in (False, True):
                with self.subTest(need_sort=need_sort, after=after):
                    tx, _ = transaction(need_sort=need_sort)
                    original = tx.allocator.free
                    def fail(slots):
                        if after:
                            original(slots)
                        raise RuntimeError("injected_old_free")
                    tx.allocator.free = fail
                    with self.assertRaisesRegex(RuntimeError, "injected_old_free"):
                        tx.commit()
                    self.assert_rolled_back(tx)

    def test_abort_waits_for_write_and_read_before_releasing_slots(self):
        tx, _ = transaction()
        order = []
        tx.fence = lambda: order.append("fence")
        restore = tx.savepoint.restore
        def recover():
            self.assertEqual(order, ["fence"])
            order.append("restore")
            restore()
        tx.savepoint.restore = recover
        tx.state = "PREPARED"  # injected readback/verification rejection
        tx.abort()
        self.assertEqual(order, ["fence", "restore"])
        self.assert_rolled_back(tx)

    def test_failed_cuda_fence_quarantines_until_a_real_fence_completes(self):
        tx, _ = transaction()
        before = free_slots(tx.allocator)
        tx.fence = Mock(side_effect=RuntimeError("cuda_pending"))
        with self.assertRaisesRegex(RuntimeError, "cuda_pending"):
            tx.abort()
        self.assertEqual(free_slots(tx.allocator), before)
        self.assertTrue(tx.allocator._kv_ec_restore_active)
        self.assertIn(tx.row, tx.req_pool._kv_ec_row_owners)
        tx.fence = Mock()
        tx.abort()
        self.assert_rolled_back(tx)

    def test_generation_changed_never_installs_or_releases_old_slots(self):
        tx, _ = transaction()
        tx.req_pool.req_generation[tx.row] += 1
        with self.assertRaisesRegex(RestoreRejected, "generation_changed"):
            tx.install_mapping()
        with self.assertRaisesRegex(RuntimeError, "identity_changed"):
            tx.abort()
        self.assertEqual(tuple(tx._mapping().tolist()), tx.old_slots)
        self.assertTrue(set(tx.new_slots).issubset(free_slots(tx.allocator)))
        self.assertFalse(set(tx.old_slots).intersection(free_slots(tx.allocator)))

    def test_changed_old_mapping_is_rejected(self):
        tx, _ = transaction()
        tx.req_pool.req_to_token[1, 0] = 16
        with self.assertRaises(RestoreRejected):
            tx.install_mapping()
        tx.abort()
        self.assertFalse(set(tx.old_slots).intersection(free_slots(tx.allocator)))

    def test_other_live_row_or_free_old_slots_reject_ownership(self):
        tx, scheduler = transaction()
        tx.abort()
        rp = tx.req_pool
        rp.req_to_token = torch.cat([rp.req_to_token, rp.req_to_token[:1]])
        with self.assertRaisesRegex(RestoreRejected, "exclusive_request_row"):
            validate_live_ownership(scheduler, tx.req, 0, 2)
        rp.free_slots = [2]
        tx.allocator.free(torch.tensor(tx.old_slots))
        with self.assertRaisesRegex(RestoreRejected, "slots_not_owned"):
            validate_live_ownership(scheduler, tx.req, 0, 2)

    def test_row_protection_blocks_reuse_clear_free_and_writes(self):
        tx, _ = transaction()
        for call in (lambda: tx.req_pool.free(tx.req), tx.req_pool.clear,
                     lambda: tx.req_pool.alloc([]), lambda: tx.req_pool.write((1, 0), 9)):
            with self.assertRaisesRegex(RuntimeError, "protected"):
                call()
        tx.abort()
        tx.req_pool.free(tx.req)
        self.assertIsNone(tx.req.req_pool_idx)
        self.assertEqual(tx.req_pool.alloc([tx.req]), [1])
        self.assertEqual(int(tx.req_pool.req_generation[1]), 8)

    def test_commit_then_normal_release_has_no_slot_leak_and_no_generation_change(self):
        tx, _ = transaction()
        tx.commit()
        self.assertEqual(tuple(tx._mapping().tolist()), tx.new_slots)
        self.assertEqual(int(tx.req_pool.req_generation[1]), 7)
        with self.assertRaises(RestoreRejected):
            tx.abort()
        with self.assertRaises(RestoreRejected):
            tx.commit()
        self.assertEqual(tuple(tx._mapping().tolist()), tx.new_slots)
        tx.allocator.free(tx._mapping().long())
        tx.req_pool.free(tx.req)
        self.assertEqual(sorted(free_slots(tx.allocator)), list(range(1, 17)))


class TPCoordinationTests(unittest.TestCase):
    def test_real_vote_checks_both_readiness_and_identity_digest(self):
        for failure in (None, "identity", "ready"):
            with self.subTest(failure=failure):
                hook = GPURestoreHook.__new__(GPURestoreHook)
                hook.deadline = time.monotonic() + 10
                hook.run_id, hook.block_hash = "cpu-vote", "hash"
                hook.meta = {"ranks": [{"token_ids_sha256": "tokens"}]}
                hook.evidence = dict(row=1, generation_before=7)
                hook.scheduler = NS(tp_cpu_group=object())
                hook._event = Mock()
                def gather(outputs, value, **kwargs):
                    for output in outputs:
                        output.copy_(value)
                    if failure == "identity":
                        outputs[1][3] ^= 1
                    elif failure == "ready":
                        outputs[1][0] = 0
                    return NS(wait=Mock(return_value=True))
                with patch("torch.distributed.all_gather", side_effect=gather):
                    self.assertEqual(hook._vote("VERIFIED", True), failure is None)

    def run_ranks(self, failure=None):
        transactions = [transaction()[0] for _ in (0, 1)]
        barrier = threading.Barrier(2, timeout=10)
        votes, phases = {}, [[], []]
        def run(rank):
            tx = transactions[rank]
            def prepare():
                if rank == 1 and failure == "verify":
                    tx.abort()  # restore_to_gpu's local exception cleanup
                    raise RestoreRejected("injected_verification")
                return tx
            def vote(phase, ready):
                phases[rank].append(phase)
                votes[rank] = ready
                barrier.wait()
                result = all(votes.values())
                barrier.wait()
                return result
            if rank == 1 and failure == "commit":
                original = tx.allocator.free
                def fail(slots):
                    original(slots)
                    raise RuntimeError("injected_rank1_commit_after_free")
                tx.allocator.free = fail
            try:
                coordinated_restore(prepare, vote)
                return "COMMITTED"
            except RestoreRejected:
                return "REJECTED"
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(run, (0, 1)))
        return transactions, phases, outcomes

    def test_single_rank_verify_failure_never_starts_mapping_or_commit(self):
        txs, phases, outcomes = self.run_ranks("verify")
        self.assertEqual(outcomes, ["REJECTED"] * 2)
        self.assertEqual(phases, [["VERIFIED"], ["VERIFIED"]])
        for tx in txs:
            TransactionFailureTests().assert_rolled_back(tx)

    def test_single_rank_commit_failure_rolls_back_both_including_successful_rank(self):
        txs, phases, outcomes = self.run_ranks("commit")
        self.assertEqual(outcomes, ["REJECTED"] * 2)
        self.assertEqual(phases, [["VERIFIED", "MAPPED", "COMMITTED"]] * 2)
        for tx in txs:
            TransactionFailureTests().assert_rolled_back(tx)

    def test_two_rank_success_uses_identical_phase_sequence(self):
        txs, phases, outcomes = self.run_ranks()
        self.assertEqual(outcomes, ["COMMITTED"] * 2)
        self.assertEqual(phases, [["VERIFIED", "MAPPED", "COMMITTED", "FINISHED"]] * 2)
        for tx in txs:
            self.assertFalse(tx.allocator._kv_ec_restore_active)

    def test_collective_timeout_aborts_local_verified_transaction(self):
        tx, _ = transaction()
        with self.assertRaisesRegex(TimeoutError, "injected_timeout"):
            coordinated_restore(lambda: tx, Mock(side_effect=TimeoutError("injected_timeout")))
        TransactionFailureTests().assert_rolled_back(tx)


class HookAdmissionTests(unittest.TestCase):
    def test_no_cuda_cannot_start_experiment(self):
        s, _, _, _ = lifetime_fixture()
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RestoreRejected, "cuda_required"):
                GPURestoreHook(s, "/no-checkpoint", "/tmp", "cpu-not-gpu")

    def test_checkpoint_fixtures_cannot_enter_real_hook(self):
        restore_tests.GPUKVRestoreValidationTests.setUpClass()
        with self.assertRaisesRegex(ValueError, "real SGLang capture required"):
            validate_checkpoint(restore_tests.GPUKVRestoreValidationTests.blocks[0])

    def test_request_input_hash_must_match_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ids.json"
            path.write_text(json.dumps([1] * 256))
            meta = {"ranks": [{"token_ids_sha256": hashlib.sha256(
                np.asarray([1] * 256, dtype="<i8").tobytes()).hexdigest()}]}
            self.assertEqual(read_input_ids(path, meta), [1] * 256)
            path.write_text(json.dumps([2] * 256))
            with self.assertRaisesRegex(RestoreRejected, "hash_mismatch"):
                read_input_ids(path, meta)

    def test_multi_output_request_is_rejected_before_processing(self):
        hook = GPURestoreHook.__new__(GPURestoreHook)
        hook.admitted = False
        hook._fail = Mock()
        message = type("TokenizedGenerateReqInput", (), {})()
        message.input_ids = [1] * 256
        message.sampling_params = NS(max_new_tokens=2)
        with self.assertRaisesRegex(RestoreRejected, "one_output_request"):
            hook.before_inputs([message])
        self.assertFalse(hook.admitted)

    def test_scheduler_hook_is_opt_in_and_between_forward_and_result_cleanup(self):
        path = Path(__file__).resolve().parents[2] / "python/sglang/srt/managers/scheduler.py"
        source = path.read_text()
        self.assertIn('os.environ.get("SGLANG_KV_EC_GPU_RESTORE") == "1"', source)
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        loop = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "event_loop_normal")
        text = ast.unparse(loop)
        self.assertLess(text.index("self.run_batch(batch)"), text.index("self._kv_ec_gpu_restore.after_forward(batch)"))
        self.assertLess(text.index("self._kv_ec_gpu_restore.after_forward(batch)"), text.index("self.process_batch_result(batch, result)"))
        self.assertLess(text.index("self.process_batch_result(batch, result)"), text.index("self._kv_ec_gpu_restore.after_result(batch)"))


class GPUPathFaultInjectionTests(unittest.TestCase):
    """Exercise the PUBLIC function's exception cleanup, mocking CUDA only.

    Metadata checks, segment hashes, row protection, allocator and transaction
    are real CPU code. Device copies/reads are injected; no GPU PASS is claimed.
    """
    @classmethod
    def setUpClass(cls):
        blocks, ranks = restore_tests.GPUKVRestoreValidationTests._rank_set(0, 0)
        cls.payloads, cls.meta = blocks, dict(node_id=0, ranks=ranks)

    def run_failure(self, failure):
        a = TokenToKVPoolAllocator(600, torch.bfloat16, "cpu", NS(), need_sort=False)
        old = a.alloc(256)
        cls = request_pool_class()
        rp = cls.__new__(cls)
        rp._alloc_size, rp.size, rp.device, rp.free_slots = 2, 1, "cpu", []
        rp.req_generation = torch.tensor([0, 7])
        rp.req_to_token = torch.zeros((2, 256), dtype=torch.int32)
        rp.req_to_token[1] = old.int()
        req = NS(req_pool_idx=1, kv_committed_len=256, kv=NS(kv_allocated_len=256))
        pool = NS(k_buffer=[torch.empty(1)], v_buffer=[torch.empty(1)], layer_num=36,
            kv_cache_layout="nhd", use_hnd=False, is_quantized_kv_cache=False,
            dtype=torch.bfloat16, store_dtype=torch.bfloat16, start_layer=0, page_size=1, size=600)
        a._kvcache = pool
        hf = NS(model_type="qwen3", num_key_value_heads=8, head_dim=128,
                hidden_size=4096, num_attention_heads=32)
        s = NS(req_to_token_pool=rp, token_to_kv_pool_allocator=a, enable_overlap=False,
            server_args=NS(disable_radix_cache=True), model_worker=NS(ps=NS(tp_rank=0, tp_size=2),
                model_runner=NS(token_to_kv_pool=pool, model_config=NS(hf_config=hf))))
        before = free_slots(a)
        cuda_fence = Mock()
        with ExitStack() as stack:
            for name, value in (
                ("is_available", Mock(return_value=True)),
                ("synchronize", cuda_fence),
                ("stream", Mock(side_effect=lambda _: nullcontext())),
                ("Stream", Mock(return_value=object())),
                ("Event", Mock(return_value=NS(record=Mock(), synchronize=Mock()))),
            ):
                stack.enter_context(patch.object(torch.cuda, name, value))
            stack.enter_context(patch.object(restore_module, "unpack", return_value=(self.meta, self.payloads)))
            copy = stack.enter_context(patch.object(restore_module, "_copy_layer"))
            read = stack.enter_context(patch.object(restore_module, "_read_layer", return_value=b"corrupt"))
            if failure == "copy":
                copy.side_effect = RuntimeError("injected_copy")
            if failure == "stream":
                stack.enter_context(patch.object(torch.cuda, "Stream", side_effect=RuntimeError("injected_stream")))
            if failure == "alloc":
                original = a.alloc
                def bad_alloc(n):
                    original(n)
                    raise RuntimeError("injected_alloc_after_mutation")
                a.alloc = bad_alloc
            with self.assertRaises((RuntimeError, RestoreRejected)):
                restore_module.restore_to_gpu(s, req, restore_module.BytesCheckpointSource(b"mock"))
        self.assertEqual(free_slots(a), before)
        self.assertEqual(rp.req_to_token[1].tolist(), old.tolist())
        self.assertFalse(getattr(rp, "_kv_ec_row_owners", {}))
        self.assertFalse(getattr(a, "_kv_ec_restore_active", False))
        self.assertGreaterEqual(cuda_fence.call_count, 2)

    def test_public_api_allocator_fault_cleans_guard_and_slots(self):
        self.run_failure("alloc")

    def test_public_api_stream_creation_fault_cleans_guard_and_slots(self):
        self.run_failure("stream")

    def test_public_api_partial_write_fault_fences_before_cleanup(self):
        self.run_failure("copy")

    def test_public_api_readback_verification_failure_reclaims_new_slots(self):
        self.run_failure("verify")


class ResultCheckerTests(unittest.TestCase):
    """Check report validation only, with a deliberately mocked metadata parser."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        block = b"CPU checker fixture, never a production checkpoint"
        self.block = self.root / "checkpoint.bin"
        self.block.write_bytes(block)
        self.digest = hashlib.sha256(block).hexdigest()
        self.meta = {"ranks": [dict(
            packed_byte_count=18874368, token_ids_sha256="tokens", model_sha256="weights",
            layer_records=[dict(k_sha256=f"k{r}:{i}", v_sha256=f"v{r}:{i}") for i in range(36)]
        ) for r in range(2)]}
        (self.root / "run.json").write_text(json.dumps(dict(
            checkpoint=str(self.block), checkpoint_sha256=self.digest, run_id="cpu-checker")))
        self.records = []
        phases = ["VERIFIED", "GLOBAL_VERIFIED_OK", "MAPPED", "GLOBAL_MAPPED_OK",
                  "COMMITTED", "GLOBAL_COMMITTED_OK", "FINISHED", "GLOBAL_FINISHED_OK",
                  "GLOBAL_COMMIT", "GLOBAL_CLEANUP_OK", "PASS"]
        for rank in range(2):
            record = dict(
                schema="kv-gpu-restore-stage-a-v1", tp_rank=rank, tp_size=2, run_id="cpu-checker",
                checkpoint_sha256=self.digest, pass_stage_a=True, cuda_executed=True,
                verified=True, committed=True, slots_disjoint=True, generation_unchanged=True,
                row_released=True, no_slot_leak=True, request_finished=True, mapping_matches_new=True,
                failure_reason=None, transaction_state="COMMITTED", generation_before=7,
                generation_after=7, original_prefill_tokens=256, original_prefill_forward_count=1,
                original_sampled_output_tokens=1, decode_forward_count=0, restore_forward_count=0,
                forward_ct_before_restore=1, forward_ct_after_restore=1,
                free_slots_after=600, allocator_size=600, row=1,
                old_slots=list(range(1, 257)), new_slots=list(range(257, 513)),
                token_ids_sha256="tokens", runtime_identity=dict(model_sha256="weights"),
                stats=dict(loaded_kv_tokens=256, token_count=256, token_start=0, recomputed_tokens=0,
                    unexpected_prefill=False, written_bytes=18874368, tp_rank=rank, tp_size=2,
                    layers_verified=[dict(layer_id=i, k_byte_exact=True, v_byte_exact=True,
                        k_sha256=f"k{rank}:{i}", v_sha256=f"v{rank}:{i}") for i in range(36)]))
            self.records.append(record)
            (self.root / f"rank_{rank}.events.jsonl").write_text("\n".join(
                json.dumps(dict(phase=p, run_id="cpu-checker", tp_rank=rank)) for p in phases))
        self.write_records(self.records)

    def write_records(self, records):
        for rank, record in enumerate(records):
            (self.root / f"rank_{rank}.json").write_text(json.dumps(record))

    def check(self):
        with patch("experiments.kv_ec.verify_gpu_restore.validate_checkpoint", return_value=self.meta):
            return check_results(self.root)

    def test_complete_report_schema_is_accepted_by_checker_only(self):
        self.assertTrue(self.check()["pass_stage_a"])

    def test_every_required_invariant_is_checked_independently(self):
        for key, value in (
            ("cuda_executed", False), ("verified", False), ("committed", False),
            ("generation_after", 8), ("original_prefill_tokens", 0), ("restore_forward_count", 1),
            ("decode_forward_count", 1), ("mapping_matches_new", False),
            ("free_slots_after", 599), ("row_released", False), ("no_slot_leak", False),
        ):
            with self.subTest(key=key):
                records = deepcopy(self.records)
                records[1][key] = value
                self.write_records(records)
                with self.assertRaises(RestoreRejected):
                    self.check()

    def test_missing_rank_and_layer_evidence_never_pass(self):
        (self.root / "rank_1.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()
        records = deepcopy(self.records)
        records[1]["stats"]["layers_verified"].pop()
        self.write_records(records)
        with self.assertRaisesRegex(RestoreRejected, "missing_layer"):
            self.check()

    def test_recompute_prefill_layer_digest_and_slot_alias_rejected(self):
        for change in ("recomputed_tokens", "unexpected_prefill", "digest", "alias"):
            with self.subTest(change=change):
                records = deepcopy(self.records)
                if change == "digest":
                    records[0]["stats"]["layers_verified"][17]["v_sha256"] = "bad"
                elif change == "alias":
                    records[0]["new_slots"][0] = records[0]["old_slots"][0]
                else:
                    records[0]["stats"][change] = True if change == "unexpected_prefill" else 1
                self.write_records(records)
                with self.assertRaises(RestoreRejected):
                    self.check()

    def test_missing_global_commit_event_and_late_failure_rejected(self):
        events = self.root / "rank_0.events.jsonl"
        events.write_text(events.read_text().replace('"GLOBAL_COMMIT"', '"NO_COMMIT"'))
        with self.assertRaisesRegex(RestoreRejected, "phase_evidence"):
            self.check()
        (self.root / "rank_1.failure.json").write_text("{}")
        with self.assertRaisesRegex(RestoreRejected, "late_failure"):
            self.check()


if __name__ == "__main__":
    unittest.main()
