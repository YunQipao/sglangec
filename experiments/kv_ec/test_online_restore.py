"""CPU-backed online call-path regressions, NOT GPU acceptance evidence."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import io
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.kv_ec import scheduler_shadow as recovery
from experiments.kv_ec.accept_pause import admit_online, check_online_evidence, main
from experiments.kv_ec.gpu_scratch_restore import restore_rank_to_gpu
from experiments.kv_ec.parity_manager import ParityManager
from experiments.kv_ec.pause_control import PauseCoordinator
from experiments.kv_ec.scheduler_pause import PauseRejected
from experiments.kv_ec.test_scheduler_pause import fixture, observation
from experiments.kv_ec.test_accept_pause import IDENTITY, completed_run


def pool_fixture():
    generator = torch.Generator().manual_seed(17)
    # Include arbitrary BF16 bit patterns; never compare BF16 numerically.
    pairs = [tuple(torch.randint(0, 256, (1026, 2, 256), dtype=torch.uint8,
                                 generator=generator).view(torch.bfloat16) for _ in range(2))
             for _ in range(36)]
    pool = NS(k_buffer=[pair[0] for pair in pairs], v_buffer=[pair[1] for pair in pairs])
    pool.get_kv_buffer = lambda layer: pairs[layer]
    return pool


class OnlineRestoreTests(unittest.TestCase):
    def test_shadow_evidence_is_not_online_success_and_no_flag_is_implicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed_run(root)
            with self.assertRaisesRegex(ValueError, "online_kv_pool_modified"):
                check_online_evidence(root / "ranks", IDENTITY, {}, {})
            with patch("sys.argv", ["accept_pause", "--online-restore", "--gpus", "0,1,2,3"]), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as spawn, patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
                spawn.assert_not_called()
            with self.assertRaisesRegex(ValueError, "missing_shadow_consensus"):
                admit_online(root, "/not-a-real-model")

    def test_online_verifier_requires_actual_target_fault_and_no_rollback_success(self):
        # Synthetic evidence validates the auditor only, never GPU execution.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed_run(root)
            proof_checkpoint = dict(snapshot_id="test-chunk0", checkpoint_version="test-chunk0",
                                    manager_epoch="manager", parity_sha256="parity", request_version="v1",
                                    request_hash=json.loads((root / "ranks/pause-rank-0.json").read_text())["request_hash"],
                                    input_ids_sha256="input")
            consensus = dict(proof_checkpoint, byte_exact=True, checkpoint_committed_before_fault=True,
                             fault_injected=True, fault_bytes_changed=True, online_kv_pool_modified=True,
                             used_committed_parity=True, rank0_source_excluded=True, writeback_byte_exact=True)
            checkpoint = dict(proof_checkpoint, state="COMMITTED", rank_hashes={str(r): {"sha256": "original"} for r in range(4)},
                              oracle_checks=[{"lost_rank": r, "byte_exact": True} for r in range(4)])
            for rank in range(4):
                path = root / "ranks" / f"pause-rank-{rank}.json"
                record = json.loads(path.read_text())
                slots = list(range(1, 769))
                record.update(online_kv_pool_modified=rank == 0, physical_slots_before=slots,
                              physical_slots_after=slots, paused_observation_monotonic=2.)
                record["coordinator_final"].update(require_online=True, shadow_ranks=list(range(4)))
                record["protection"] = dict(proof_checkpoint, state="COMMITTED", source_sha256="original",
                                           committed_monotonic=1., stage="normal_prefill_before_paused_observation")
                record["online"] = dict(
                    mode="online", tp_rank=rank, device=f"cuda:{rank}", recovered_rank=0, token_start=0,
                    token_count=256, layers=list(range(36)), input_ids_sha256="input",
                    request_generation=1, consensus=consensus, checkpoint=proof_checkpoint,
                    checkpoint_committed_before_fault=True, used_committed_parity=True, rank0_source_excluded=True,
                    cuda_event_complete=True, slot_generations_unchanged=True, mapping_unchanged=True,
                    rollback_used_for_recovery=False, original_sha256="original", rollback_sha256="original",
                    restored_sha256="original", survivor_unchanged=True, fault_injected=rank == 0,
                    online_kv_pool_modified=rank == 0, writeback_byte_exact=rank == 0,
                    fault_bytes_changed=rank == 0, changed_byte_count=100, fault_monotonic=3.,
                    decode_observed_monotonic=4., rollback_byte_exact=False,
                    physical_slots_sha256=recovery.digest(np.asarray(slots[:256], dtype="<i8")),
                    oracle_checks=[{"lost_rank": r, "byte_exact": True} for r in range(4)],
                    timing_ms={key: 1. for key in (
                        "protection_total_ms", "fault_injection_and_verify_ms", "survivor_gather_ms", "parity_fetch_ms",
                        "xor_and_cpu_verify_ms", "h2d_ms", "kv_writeback_ms", "writeback_gpu_and_hash_verify_ms", "fault_to_decode_ms")})
                path.write_text(json.dumps(record))
            response = {"input_ids_sha256": "input"}
            self.assertEqual(len(check_online_evidence(root / "ranks", IDENTITY, checkpoint, response)), 4)
            path = root / "ranks/pause-rank-0.json"
            original = json.loads(path.read_text())
            for key, value in (("fault_injected", False), ("fault_bytes_changed", False), ("device", "cpu"),
                               ("changed_byte_count", 0), ("restored_sha256", "bad"), ("timing_ms", {}),
                               ("rollback_byte_exact", True), ("rollback_used_for_recovery", True),
                               ("fault_monotonic", .5), ("physical_slots_sha256", "wrong")):
                record = deepcopy(original)
                record["online"][key] = value
                path.write_text(json.dumps(record))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check_online_evidence(root / "ranks", IDENTITY, checkpoint, response)

    def test_online_gpu_unavailable_never_starts_a_service(self):
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
                "accept_pause", "--online-restore", "--shadow-evidence", "/prior-shadow", "--gpus", "0,1,2,3"]), patch(
                "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                "experiments.kv_ec.accept_pause.admit_online", return_value={}), patch(
                "experiments.kv_ec.accept_pause.subprocess.check_output",
                side_effect=subprocess.CalledProcessError(9, "nvidia-smi", output="driver unavailable")), patch(
                "experiments.kv_ec.accept_pause.subprocess.Popen") as spawn:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            spawn.assert_not_called()
            verdict = json.loads((Path(tmp) / "acceptance.json").read_text())
            self.assertFalse(verdict["M2_PASS"])
            self.assertIn("driver unavailable", verdict["reason"])

    def test_scatter_and_fault_touch_only_current_mapped_chunk(self):
        pool = pool_fixture()
        slots = torch.arange(512, 0, -2, dtype=torch.int64)
        saved = [tuple(t.view(torch.uint8).index_select(0, slots).clone() for t in pool.get_kv_buffer(layer))
                 for layer in range(36)]
        other = torch.tensor([i for i in range(1026) if i not in set(slots.tolist())])
        before = [tuple(t.view(torch.uint8).index_select(0, other) for t in pool.get_kv_buffer(layer))
                  for layer in range(36)]
        recovery.zero_mapped_bytes(pool, slots)
        with self.assertRaisesRegex(ValueError, "byte_mismatch"):
            recovery.verify_mapped_bytes(pool, slots, saved)
        recovery.write_mapped_bytes(pool, slots, saved)
        recovery.verify_mapped_bytes(pool, slots, saved)
        for layer, pair in enumerate(before):
            for live, expected in zip(pool.get_kv_buffer(layer), pair):
                self.assertTrue(torch.equal(live.view(torch.uint8).index_select(0, other), expected))
        for buffers in (saved[:35], [tuple(t.view(torch.uint8) for t in pool.get_kv_buffer(0))] * 36):
            with self.assertRaises(ValueError):
                recovery.write_mapped_bytes(pool, slots, buffers)

    def test_prepare_commit_then_fault_parity_restore_and_partial_write_rollback(self):
        # Adapt only device/IPC operations; execute actual protection,
        # injection, scatter, digest checks, failure handler and gate leases.
        for failure in (None, "partial_write", "missing_parity", "manager_restart", "slot_generation",
                        "request_cancelled", "rank_missing", "coordinator_disconnect"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                coordinator = PauseCoordinator("tx", require_online=True)
                gate, req, batch, producer = fixture(Path(tmp), coordinator)
                manager = ParityManager(test_oracle=True, manager_epoch="manager")
                actual_pool = pool_fixture()
                proxy_pool = NS(k_buffer=[NS(device=torch.device("cuda:0"))])
                gate.scheduler.model_worker.model_runner = NS(token_to_kv_pool=proxy_pool)
                gate.scheduler.forward_stream = Mock()
                gate.scheduler.tp_cpu_group = object()
                gate.model_identity.update(source_commit="commit", core_diff_sha256="diff")
                online = recovery.SchedulerShadowRecovery("/tmp/parity.sock", "manager", online=True)
                gate.shadow = online
                tensor_factory, checker, packer = torch.tensor, recovery.checked_buffers, recovery.pack_chunk
                state = {"faulted": False, "parity_fetched": False, "partial": False}
                def control(path, message, **kwargs):
                    if failure == "coordinator_disconnect" and state["faulted"] and message["op"] == "status":
                        raise ConnectionError("injected_coordinator_disconnect")
                    return {"state": "OK", "status": coordinator.dispatch(message)}
                def ipc(path, metadata, payload, **kwargs):
                    if metadata.get("kind") == "status":
                        return manager.status(metadata["snapshot_id"])
                    self.assertFalse(online.write_started, "parity may NEVER be submitted after a fault")
                    return manager.submit(metadata, payload)
                def tensor(*args, **kwargs):
                    if str(kwargs.get("device", "")).startswith("cuda"):
                        kwargs["device"] = "cpu"
                    return tensor_factory(*args, **kwargs)
                def pack(*args, **kwargs):
                    return packer(actual_pool, *args[1:], **kwargs)
                def buffers(pool, layer, slots):
                    return checker(actual_pool, layer, slots)
                def gather(contribution, **kwargs):
                    self.assertEqual(torch.count_nonzero(contribution).item(), 0)
                    self.assertTrue(online.write_started)
                    self.assertIsNone(online.survivor)
                    state["faulted"] = True
                    for rank, block in enumerate(survivors, 1):
                        kwargs["gather_list"][rank].copy_(torch.from_numpy(block))
                    return NS(wait=lambda **kw: failure != "rank_missing")
                def fetch(path, expected, **kwargs):
                    self.assertTrue(state["faulted"])
                    if failure == "missing_parity":
                        raise ValueError("parity_checkpoint_not_committed")
                    if failure == "manager_restart":
                        raise ValueError("shadow_manager_epoch_changed")
                    if failure == "slot_generation":
                        gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.generation[1] += 1
                    if failure == "request_cancelled":
                        req.to_finish = object()
                    state["parity_fetched"] = True
                    status, raw = manager.export_parity(expected["snapshot_id"], expected)
                    return status, np.frombuffer(raw, dtype=np.uint8).copy()
                original_write = recovery.write_mapped_bytes
                def write(pool, slots, sources):
                    if sources is online.recovered:
                        self.assertTrue(state["parity_fetched"])
                        self.assertIsNot(sources, online.rollback)
                        if failure == "partial_write" and not state["partial"]:
                            state["partial"] = True
                            actual_pool.k_buffer[0].view(torch.uint8).index_copy_(0, slots, sources[0][0])
                            raise RuntimeError("injected_partial_layer_write_failure")
                    return original_write(actual_pool, slots, sources)
                def restore(payload, metadata, device, *, cuda_metrics):
                    cuda_metrics["h2d_copy_event_ms"] = 0.
                    return restore_rank_to_gpu(payload, metadata, torch.device("cpu"))
                with patch("experiments.kv_ec.scheduler_pause.torch.cuda.is_available", return_value=False), patch(
                        "experiments.kv_ec.scheduler_pause.request", side_effect=control):
                    gate.after_result(batch)
                admission_check = Mock()
                with patch.object(online, "check_online_admission", admission_check), patch(
                        "experiments.kv_ec.scheduler_shadow.manager_hello", return_value="manager"), patch(
                        "experiments.kv_ec.scheduler_shadow.torch.cuda.is_available", return_value=True), patch(
                        "experiments.kv_ec.scheduler_shadow.torch.cuda.stream", return_value=nullcontext()), patch(
                        "experiments.kv_ec.scheduler_shadow.torch.cuda.Event", side_effect=lambda: NS(
                            record=Mock(), query=lambda: True)), patch(
                        "experiments.kv_ec.scheduler_shadow.torch.tensor", side_effect=tensor), patch(
                        "experiments.kv_ec.scheduler_shadow.pack_chunk", side_effect=pack), patch(
                        "experiments.kv_ec.scheduler_shadow.checked_buffers", side_effect=buffers), patch(
                        "experiments.kv_ec.scheduler_shadow.submit_once", side_effect=ipc) as submit, patch(
                        "experiments.kv_ec.scheduler_pause.request", side_effect=control):
                    self.assertFalse(online.prepare(gate))
                    first_calls = submit.call_count
                    self.assertFalse(online.prepare(gate))
                    self.assertEqual(submit.call_count, first_calls + 1, "second prepare only polls status")
                    survivors = []
                    # CPU fixtures for three distinct TP shards, same committed version.
                    for rank in range(1, 4):
                        block, meta = packer(actual_pool, replace(gate.mapping, tp_rank=rank), req,
                                             gate.model_identity, snapshot_id="tx-chunk0", manager_epoch="manager")
                        block = np.bitwise_xor(block, rank)
                        derived = recovery.validate_payload(block, meta, verify_checksums=False)
                        for key in ("sha256", "k_sha256", "v_sha256"):
                            meta[key] = derived[key]
                        for record in meta["layer_records"]:
                            for kind in ("k", "v"):
                                record[kind + "_sha256"] = derived["layer_sha256"][(record["layer_id"], kind)]
                        survivors.append(block)
                        manager.submit(meta, block)
                    self.assertTrue(online.prepare(gate))
                    self.assertIsNone(online.survivor)
                    self.assertFalse(manager._checkpoints["tx-chunk0"].ranks)
                    original = deepcopy(online.metadata)
                    gate._vote = lambda ready: ready
                    gate._tick_local()
                    for rank in range(1, 4):
                        obs = dict(coordinator.observations[0])
                        coordinator.dispatch(dict(gate.identity, op="paused", tp_rank=rank, observation=obs))
                    gate.state = "SHADOW_READY"
                    with patch("experiments.kv_ec.scheduler_shadow.torch.distributed.gather", side_effect=gather), patch(
                            "experiments.kv_ec.scheduler_shadow.fetch_parity", side_effect=fetch), patch(
                            "experiments.kv_ec.scheduler_shadow.restore_rank_to_gpu", side_effect=restore), patch(
                            "experiments.kv_ec.scheduler_shadow.write_mapped_bytes", side_effect=write):
                        if failure is None:
                            result = online.run(gate)
                            self.assertTrue(result["fault_bytes_changed"])
                            self.assertTrue(result["writeback_byte_exact"])
                            self.assertGreater(result["changed_byte_count"], 0)
                            self.assertEqual(result["restored_sha256"], original["sha256"])
                            self.assertFalse(result.get("rollback_attempted", False))
                        else:
                            with self.assertRaises((ValueError, RuntimeError, OSError)):
                                online.run(gate)
                            evidence = gate.evidence["online"]
                            self.assertFalse(online.done)
                            if failure in ("slot_generation", "request_cancelled"):
                                self.assertFalse(evidence["rollback_attempted"])
                                self.assertIn("mapping_changed" if failure == "slot_generation" else "request_cancelled",
                                              evidence["rollback_failure"])
                            else:
                                self.assertTrue(evidence["rollback_byte_exact"])
                            with self.assertRaises(PauseRejected):
                                gate._fatal(RuntimeError("experiment_failed"))
                            self.assertEqual(gate.state, "ABORTED_FAIL_CLOSED")
                            self.assertTrue(gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.closed)
                            self.assertEqual(gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 1)
                            with self.assertRaisesRegex(RuntimeError, "row_protected"):
                                gate.scheduler.req_to_token_pool.free(req)
                    self.assertEqual(online.metadata, original)

    def test_incomplete_cuda_fence_never_rolls_back_or_releases(self):
        online = recovery.SchedulerShadowRecovery("/tmp/parity.sock", "manager", online=True)
        online.write_started = True
        online.rollback = object()
        gate = NS(evidence={}, _revalidate=Mock())
        with patch.object(online, "fence", side_effect=TimeoutError("cuda_incomplete")), patch.object(
                recovery, "write_mapped_bytes") as write:
            online.rollback_failure(gate, NS(), RuntimeError("copy_failed"))
        write.assert_not_called()
        self.assertFalse(gate.evidence["online"]["rollback_attempted"])
        self.assertIn("cuda_incomplete", gate.evidence["online"]["rollback_failure"])
        self.assertIsNotNone(online.rollback)

    def test_coordinator_requires_preexisting_protection_and_all_rank_repair(self):
        c = PauseCoordinator("tx", require_online=True)
        checkpoint = dict(snapshot_id="tx-chunk0", checkpoint_version="tx-chunk0", manager_epoch="m",
                          parity_sha256="sha", input_ids_sha256="tokens", request_hash="r", request_version="v1")
        def call(op, **values):
            return c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op=op, **values))
        with self.assertRaisesRegex(ValueError, "preexisting"):
            call("paused", tp_rank=0, observation=observation())
        for rank in range(4):
            call("paused", tp_rank=rank, observation=dict(observation(), checkpoint=checkpoint, checkpoint_committed=True))
        proof = dict(checkpoint, byte_exact=True, online_kv_pool_modified=True, checkpoint_committed_before_fault=True,
                     fault_injected=True, fault_bytes_changed=True, used_committed_parity=True,
                     rank0_source_excluded=True, writeback_byte_exact=True)
        for rank in range(4):
            with self.assertRaises(ValueError):
                call("authorize")
            if rank == 0:
                with self.assertRaisesRegex(ValueError, "incomplete_online"):
                    call("shadow_verified", tp_rank=rank, observation=dict(proof, writeback_byte_exact=False))
            call("shadow_verified", tp_rank=rank, observation=proof)
        self.assertEqual(call("authorize")["state"], "RESUME_REQUESTED")

    def test_no_preexisting_checkpoint_or_no_cuda_rejects_before_writes(self):
        online = recovery.SchedulerShadowRecovery("/tmp/parity.sock", "m", online=True)
        gate = NS(mapping=NS())
        with self.assertRaisesRegex(ValueError, "preexisting"):
            online.run(gate)
        online.status, online.protection_mapping = {}, gate.mapping
        with patch("experiments.kv_ec.scheduler_shadow.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "real_cuda"):
                online.run(gate)
        self.assertFalse(online.modified)


if __name__ == "__main__":
    unittest.main()
