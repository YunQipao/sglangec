"""CPU-only B/D safety tests. No fixture here is GPU acceptance evidence."""

import ast
from copy import deepcopy
from contextlib import redirect_stdout
from functools import partial
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from experiments.kv_ec.accept_pause import experiment_performance_record
from experiments.kv_ec.pause_control import PauseCoordinator
from experiments.kv_ec.scheduler_pause import SchedulerPauseGate, PauseRejected
from experiments.kv_ec.scheduler_retry import SchedulerPrefillRetry
from experiments.kv_ec.scheduler_shadow import SchedulerShadowRecovery, validate_checkpoint
from experiments.kv_ec.test_scheduler_pause import ROOT, fixture, observation, Event
from experiments.kv_ec.m4_abcd import MODES, summarize


def extract(relative, name, namespace, class_name=None):
    path = ROOT / relative
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class PerformanceRecordTests(unittest.TestCase):
    def check_record(self, mode, keys):
        # This is the real client shape: note is present in every successful
        # streaming response, including the failing GPU runs reported by users.
        response = dict(client_metrics_ms=dict(
            ttft_ms=210.0, tpot_ms=40.0, request_total_ms=810.0,
            note="client streaming TTFT and amortized TPOT, includes HTTP/SSE overhead"))
        records = [{mode: {key: float(rank + 1) for key in keys}} for rank in range(4)]
        original_response, original_records = deepcopy(response), deepcopy(records)
        performance = experiment_performance_record(mode, response, records, config={"tp": 4})
        self.assertEqual(performance["mode"], mode)
        self.assertEqual(performance["config"], {"tp": 4})
        for key, value in response["client_metrics_ms"].items():
            self.assertEqual(performance[key], value)
        self.assertIn("all-rank max", performance["timing_note"])
        for key in keys:
            self.assertEqual(performance[key], 4.0)
        self.assertEqual(performance["rank_metrics"], [record[mode] for record in records])
        self.assertEqual(json.loads(json.dumps(performance)), performance)
        self.assertEqual(response, original_response)
        self.assertEqual(records, original_records)

    def test_B_record_preserves_client_note_and_protection_timing(self):
        self.check_record("protection_only", ("protection_blocking_ms", "parity_commit_ms"))

    def test_D_record_preserves_client_note_and_fault_timing(self):
        self.check_record("prefill_retry",
                          ("fault_to_rebuilt_state_ms", "fault_to_first_decode_ms", "prefill_forward_ms"))


class RetryCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.c = PauseCoordinator("tx", experiment_mode="prefill_retry")
        for rank in range(4):
            self.call("paused", tp_rank=rank, observation=observation())

    def call(self, op, **kwargs):
        return self.c.dispatch(dict(epoch=self.c.epoch, transaction_id="tx", op=op, **kwargs))

    def ready(self, rank):
        return dict(event_complete=True, lease_validated=True,
                    request_hash="r", request_version="v1", input_ids_sha256="input",
                    fault_injected=rank == 0, fault_bytes_changed=rank == 0)

    def rebuilt(self):
        return dict(request_hash="r", original_request_version="v1", input_ids_sha256="input",
                    retry_rebuilt=True, byte_exact=True, history_preserved=True,
                    fresh_allocation=True, new_lease_validated=True,
                    used_committed_parity=False, rollback_used_for_recovery=False,
                    duplicate_output_emitted=False, fault_boundary_output_count=1,
                    rebuilt_prefix_tokens=768)

    def handoff(self):
        for rank in range(4):
            self.call("retry_ready", tp_rank=rank, observation=self.ready(rank))
        for rank in range(4):
            self.call("retry_released", tp_rank=rank)

    def test_four_rank_handoff_and_rebuild_before_external_authorization(self):
        with self.assertRaisesRegex(ValueError, "before_four_rank"):
            self.call("retry_released", tp_rank=0)
        with self.assertRaisesRegex(ValueError, "four_rank_shadow"):
            self.call("authorize")
        with self.assertRaisesRegex(ValueError, "missing_proof"):
            self.call("shadow_verified", tp_rank=0, observation=self.rebuilt())
        self.handoff()
        for rank in range(3):
            self.call("shadow_verified", tp_rank=rank, observation=self.rebuilt())
        with self.assertRaises(ValueError):
            self.call("authorize")
        self.call("shadow_verified", tp_rank=3, observation=self.rebuilt())
        self.assertEqual(self.call("authorize")["state"], "RESUME_REQUESTED")
        for op in ("validated", "released", "armed"):
            for rank in range(4):
                self.call(op, tp_rank=rank)
        self.assertEqual(self.c.state, "RESUME_AUTHORIZED")

    def test_bad_target_fault_or_lease_proof_rejected(self):
        for key, value in (("event_complete", False), ("lease_validated", False),
                           ("fault_injected", False), ("fault_bytes_changed", False),
                           ("request_version", "wrong")):
            proof = self.ready(0)
            proof[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing_proof"):
                self.call("retry_ready", tp_rank=0, observation=proof)

    def test_rebuild_may_not_skip_generation_history_or_kv_checks(self):
        self.handoff()
        for key, value in (("byte_exact", False), ("history_preserved", False),
                           ("fresh_allocation", False), ("new_lease_validated", False),
                           ("used_committed_parity", True), ("rollback_used_for_recovery", True),
                           ("duplicate_output_emitted", True), ("rebuilt_prefix_tokens", 769)):
            proof = self.rebuilt()
            proof[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing_proof"):
                self.call("shadow_verified", tp_rank=0, observation=proof)

    def test_conflicting_modes_and_duplicate_handoff_are_fail_closed(self):
        with self.assertRaises(ValueError):
            PauseCoordinator("bad", require_online=True, experiment_mode="prefill_retry")
        self.call("retry_ready", tp_rank=0, observation=self.ready(0))
        proof = self.ready(0)
        proof["extra_changed_metadata"] = True
        self.assertEqual(self.call("retry_ready", tp_rank=0, observation=proof)["state"], "FAILED")
        self.assertEqual(self.call("authorize")["state"], "FAILED")


class ProtectionTests(unittest.TestCase):
    def test_B_four_rank_committed_checkpoint_interlock(self):
        c = PauseCoordinator("tx", experiment_mode="protection_only")
        def dispatch(op, **values):
            return c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op=op, **values))
        checkpoint = dict(
            snapshot_id="tx-chunk0", checkpoint_version="tx-chunk0",
            manager_epoch="manager", parity_sha256="parity", request_hash="r",
            request_version="v1", input_ids_sha256="input")
        for rank in range(4):
            obs = observation()
            with self.assertRaisesRegex(ValueError, "committed_checkpoint"):
                dispatch("paused", tp_rank=rank, observation=obs)
            obs.update(checkpoint_committed=True, checkpoint=checkpoint)
            dispatch("paused", tp_rank=rank, observation=obs)
        proof = dict(checkpoint, byte_exact=True, online_kv_pool_modified=False,
                     protection_only=True, fault_injected=False, recovery_executed=False,
                     used_committed_parity=False)
        for key in ("fault_injected", "recovery_executed", "used_committed_parity"):
            bad = dict(proof, **{key: True})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "invalid_proof"):
                dispatch("shadow_verified", tp_rank=0, observation=bad)
        for rank in range(4):
            dispatch("shadow_verified", tp_rank=rank, observation=proof)
        self.assertEqual(dispatch("authorize")["state"], "RESUME_REQUESTED")

    def test_default_off_and_invalid_env(self):
        s = NS(server_args=NS(disable_overlap_schedule=True))
        for env in (
            {"SGLANG_KV_EC_EXPERIMENT_MODE": "prefill_retry"},
            {"SGLANG_KV_EC_EXPERIMENT_MODE": "protection_only"},
            {"SGLANG_KV_EC_EXPERIMENT_MODE": "bad"},
            {"SGLANG_KV_EC_EXPERIMENT_MODE": "prefill_retry", "SGLANG_KV_EC_ONLINE_RESTORE": "1"},
        ):
            with self.subTest(env=env), patch.dict("os.environ", env, clear=True):
                with self.assertRaises(PauseRejected):
                    SchedulerPauseGate.from_env(s)
        self.assertFalse(SchedulerShadowRecovery("/tmp/test.sock", "epoch").protection_only)
        with self.assertRaises(ValueError):
            SchedulerShadowRecovery("/tmp/test.sock", "epoch", online=True, protection_only=True)

    def test_B_run_only_revalidates_committed_source_without_recovery(self):
        recovery = SchedulerShadowRecovery("/tmp/test.sock", "epoch", protection_only=True)
        meta = dict(snapshot_id="tx-chunk0", token_start=0)
        recovery.chunks = [dict(metadata=meta, status={})]
        recovery.validate = Mock()
        recovery.consensus_checkpoint = Mock(return_value=dict(input_ids_sha256="input"))
        gate = NS(_revalidate=Mock(), event=object(), req=object(), mapping=object(),
                  model_identity={}, scheduler=NS(
                      model_worker=NS(model_runner=NS(token_to_kv_pool=object())),
                      forward_stream=NS(wait_event=Mock())))
        with patch("experiments.kv_ec.scheduler_shadow.torch.cuda.stream", return_value=__import__(
                "contextlib").nullcontext()), patch(
                "experiments.kv_ec.scheduler_shadow.pack_chunk", return_value=(None, meta)), patch(
                "experiments.kv_ec.scheduler_shadow.validate_checkpoint") as validate, patch(
                "experiments.kv_ec.scheduler_shadow.reconstruct",
                side_effect=AssertionError("B must never recover")), patch(
                "experiments.kv_ec.scheduler_shadow.write_mapped_bytes",
                side_effect=AssertionError("B must never write")), patch(
                "experiments.kv_ec.scheduler_shadow.zero_mapped_bytes",
                side_effect=AssertionError("B must never inject")):
            result = recovery.run(gate)
        validate.assert_called_once_with({}, meta, require_oracles=False)
        self.assertFalse(result["recovery_executed"])
        self.assertFalse(result["kv_writeback_executed"])
        self.assertFalse(result["fault_injected"])

    def test_production_commit_rejects_oracle_work_and_bad_hash(self):
        from experiments.kv_ec.parity_manager import ParityManager
        from experiments.kv_ec.test_parity_manager import metadata_and_blocks
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(test_oracle=False)
        for meta in metadata:
            meta.update(manager_epoch=manager.manager_epoch, request_version="v1", checkpoint_version="cp1")
        with patch("experiments.kv_ec.parity_manager.recover",
                   side_effect=AssertionError("B manager must never reconstruct")):
            for meta, block in zip(metadata, blocks):
                status = manager.submit(meta, block)
        self.assertEqual(status["state"], "COMMITTED")
        self.assertEqual(status["oracle_checks"], [])
        validate_checkpoint(status, metadata[0], require_oracles=False)
        with self.assertRaises(ValueError):
            validate_checkpoint(status, metadata[0])
        bad = deepcopy(status)
        bad["rank_hashes"]["0"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "version_or_rank"):
            validate_checkpoint(bad, metadata[0], require_oracles=False)
        bad = deepcopy(status)
        bad["oracle_checks"] = [{"lost_rank": r, "byte_exact": True} for r in range(4)]
        with self.assertRaises(ValueError):
            validate_checkpoint(bad, metadata[0], require_oracles=False)
        # Explicit new C still reconstructs and hashes all bytes, just without
        # doing four unrelated recovery-oracles during pre-fault parity commit.
        from experiments.kv_ec.scheduler_shadow import reconstruct
        import numpy as np
        exported, raw = manager.export_parity("checkpoint", metadata[0])
        recovered = reconstruct(np.frombuffer(raw, dtype=np.uint8), blocks[1:],
                                exported, metadata[0], require_oracles=False)
        np.testing.assert_array_equal(recovered, blocks[0])
        corrupted = blocks[1].copy()
        corrupted[0] ^= 1
        with self.assertRaisesRegex(ValueError, "survivor_version_or_bytes"):
            reconstruct(np.frombuffer(raw, dtype=np.uint8), [corrupted, *blocks[2:]],
                        exported, metadata[0], require_oracles=False)


class RetryAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("experiments.kv_ec.scheduler_pause.torch.cuda.is_available", return_value=False))

    def test_scope_and_stream_history(self):
        for tokens in (512, 768):
            with self.assertRaises(ValueError):
                SchedulerPrefillRetry(recover_tokens=tokens)
        retry = SchedulerPrefillRetry()
        req = NS(output_ids=[7], send_token_offset=1)
        retry._check_history(req, (7,), {"send_token_offset": 1})
        req.send_token_offset = 0
        with self.assertRaisesRegex(ValueError, "history_or_stream"):
            retry._check_history(req, (7,), {"send_token_offset": 1})
        req.output_ids.append(7)
        with self.assertRaises(ValueError):
            retry._check_history(req, (7,), {})

    def test_handoff_never_releases_an_unvalidated_lease(self):
        for failure in ("missing_peer", "mapping", "event", "vote"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                gate, req, batch, event = fixture(Path(tmp), PauseCoordinator("tx"))
                gate.after_result(batch)
                gate._call = Mock(return_value={"retry_ready_ranks": list(range(4))})
                gate._vote = Mock(return_value=True)
                if failure == "missing_peer":
                    gate._call.return_value = {"retry_ready_ranks": [0, 1, 2]}
                elif failure == "mapping":
                    gate.scheduler.req_to_token_pool.req_generation[2] += 1
                elif failure == "event":
                    event.complete = False
                else:
                    gate._vote.return_value = False
                with self.assertRaises((ValueError, PauseRejected)):
                    SchedulerPrefillRetry()._release_old(gate)
                self.assertEqual(gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 1)
                with self.assertRaisesRegex(RuntimeError, "row_protected"):
                    gate.scheduler.req_to_token_pool.free(req)
                with self.assertRaisesRegex(RuntimeError, "protected_slot_free"):
                    gate.scheduler.token_to_kv_pool_allocator.free(torch.arange(1, 769))
                event.complete = True

    def test_normal_release_reallocate_and_new_lease_with_actual_allocator(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate, req, batch, _ = fixture(Path(tmp), PauseCoordinator("tx"))
            gate.after_result(batch)
            retry = SchedulerPrefillRetry()
            gate._vote = lambda value: value
            gate._call = Mock(return_value={"retry_ready_ranks": list(range(4)),
                                           "retry_released_ranks": list(range(4))})
            old = gate.mapping
            s = gate.scheduler
            retry._release_old(gate)
            self.assertEqual(s.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)

            # Execute the real normal release, ChunkCache free, request reset and
            # row alloc implementations; only GPU batch preparation/forward is a
            # CPU surrogate. Lease guards are never patched out.
            common_ns = dict(HybridReqToTokenPool=type("Hybrid", (), {}),
                             _release_overallocated_kv_indices=lambda *args: None)
            release_kv = extract("python/sglang/srt/mem_cache/common.py", "release_kv_cache", common_ns)
            reset = extract("python/sglang/srt/managers/schedule_batch.py", "reset_for_retract",
                            {"torch": torch}, "Req")
            release = extract("python/sglang/srt/managers/schedule_batch.py", "release_req", {
                "release_kv_cache": release_kv, "evict_from_tree_cache": lambda *args: None,
                "envs": NS(SGLANG_RETRACT_DECODE_STEPS=NS(get=lambda: 1))})
            cache_ns = {}
            finished = extract("python/sglang/srt/mem_cache/chunk_cache.py",
                               "cache_finished_req", cache_ns, "ChunkCache")
            unfinished = extract("python/sglang/srt/mem_cache/chunk_cache.py",
                                 "cache_unfinished_req", {"torch": torch}, "ChunkCache")
            cache = NS(req_to_token_pool=s.req_to_token_pool,
                       token_to_kv_pool_allocator=s.token_to_kv_pool_allocator)
            cache.cache_finished_req = partial(finished, cache)
            cache.cache_unfinished_req = partial(unfinished, cache)
            s.tree_cache = cache
            s.server_args.disaggregation_mode = "null"
            req.kv = NS(kv_allocated_len=768)
            req.kv_committed_len = 768
            req.effective_kv_committed_len = lambda: req.kv_committed_len
            req.retraction_count, req.input_embeds = 0, None
            req.reset_for_retract = partial(reset, req)
            req.init_next_round_input = lambda tree: None
            req.set_extend_range = lambda start, end: setattr(req, "extend_range", NS(start=start, end=end))
            s.last_batch, s.running_batch = batch, NS(is_empty=lambda: True)
            s.model_config, s.spec_algorithm = NS(), NS()

            class CpuBatch:
                @classmethod
                def init_new(cls, *args):
                    return cls()

                def prepare_for_extend(self):
                    s.req_to_token_pool.alloc([req])
                    slots = s.token_to_kv_pool_allocator.alloc(768)
                    s.req_to_token_pool.write((req.req_pool_idx, slice(0, 768)), slots.to(torch.int32))
                    req.kv, req.kv_committed_len = NS(kv_allocated_len=768), 768
                    req.is_retracted = False
                    self.prefix_lens, self.extend_num_tokens = [0], 768
                    self.reqs = [req]

            def fence(g):
                retry.event = Event()
                g.event = retry.event
                s.token_to_kv_pool_allocator.attach_slot_lease_cuda_event(g.owner, g.event)
            retry._fence = fence
            s.run_batch = Mock(return_value=NS(has_sampled_token_ids=True, delay_sample_func=None,
                                               next_token_ids=torch.tensor([2])))
            event = Mock()
            event.elapsed_time.return_value = .5
            module = NS(ScheduleBatch=CpuBatch, release_req=release)
            with patch.dict(sys.modules, {"sglang.srt.managers.schedule_batch": module}), patch(
                    "experiments.kv_ec.scheduler_retry.torch.cuda.Event", return_value=event), patch(
                    "experiments.kv_ec.scheduler_retry.torch.cuda.current_stream", return_value=None):
                result = retry._rebuild(gate)
            self.assertEqual(result["prefill_forward_ms"], .5)
            self.assertEqual(req.retraction_count, 1)
            self.assertEqual(req.output_ids, [2])
            self.assertNotEqual(old.req_generation, gate.mapping.req_generation)
            self.assertNotEqual(old.slot_generations, gate.mapping.slot_generations)
            self.assertEqual(s.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 1)
            with self.assertRaisesRegex(RuntimeError, "protected_slot_free"):
                s.token_to_kv_pool_allocator.free(torch.tensor(gate.mapping.physical_slots))
            with self.assertRaisesRegex(RuntimeError, "row_protected"):
                s.req_to_token_pool.free(req)
            gate._revalidate()


class StudyTests(unittest.TestCase):
    def test_CPU_failure_starts_no_GPU_mode(self):
        from experiments.kv_ec.m4_abcd import main
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
                "m4_abcd", "--gpus", "0,1,2,3", "--shadow-evidence", "/prior",
                "--output", str(Path(tmp) / "new")]), patch(
                "experiments.kv_ec.m4_abcd.run_logged", return_value=1) as run:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            self.assertEqual(run.call_count, 1)
            manifest = json.loads((Path(tmp) / "new/trials.json").read_text())
            self.assertEqual(manifest["trials"], [])
            self.assertEqual(manifest["state"], "FAIL")

    def test_three_repeats_require_successful_smoke(self):
        from experiments.kv_ec.m4_abcd import main
        with patch("sys.argv", [
                "m4_abcd", "--gpus", "0,1,2,3", "--shadow-evidence", "/prior",
                "--repeat", "3"]), patch("sys.stderr", io.StringIO()), patch(
                "experiments.kv_ec.m4_abcd.run_logged") as run:
            with self.assertRaises(SystemExit):
                main()
            run.assert_not_called()

    def test_B_D_driver_failure_cannot_start_servers_or_claim_validation(self):
        from experiments.kv_ec.accept_pause import main
        import subprocess
        for mode in ("protection_only", "prefill_retry"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
                    "accept_pause", "--mode", mode, "--gpus", "0,1,2,3",
                    "--shadow-evidence", "/prior", "--output", str(Path(tmp) / "new")]), patch(
                    "experiments.kv_ec.accept_pause.admit_online", return_value={}), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output",
                    side_effect=subprocess.CalledProcessError(9, "nvidia-smi", output="driver unavailable")), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as spawn:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 1)
                spawn.assert_not_called()
                verdict = json.loads((Path(tmp) / "new/acceptance.json").read_text())
                self.assertFalse(verdict["GPU_VALIDATED"])
                self.assertFalse(verdict["EXPERIMENT_PASS"])
                self.assertIn("driver unavailable", verdict["reason"])

    def test_four_distinct_modes_and_incomplete_study_rejected(self):
        self.assertEqual(MODES, ("baseline", "protection_only", "online_recovery", "prefill_retry"))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "trials.json").write_text(json.dumps({"trials": []}))
            with self.assertRaisesRegex(ValueError, "no_complete"):
                summarize(tmp)


if __name__ == "__main__":
    unittest.main()
