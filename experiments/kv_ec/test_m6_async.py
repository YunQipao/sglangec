"""M6 CPU ownership tests; simulated events are never real GPU evidence."""

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from experiments.kv_ec.async_protection import (
    AsyncProtection, CPUSnapshot, COMMITTED, COMMITTING, FAILED, SNAPSHOT_READY,
    checkpoint_metadata,
)
from experiments.kv_ec.parity_manager import ParityManager
from experiments.kv_ec.pause_control import PauseCoordinator
from experiments.kv_ec.test_parity_manager import metadata_and_blocks
from experiments.kv_ec.test_scheduler_pause import fixture, observation
from experiments.kv_ec.scheduler_pause import PauseRejected
from experiments.kv_ec.scheduler_async import SchedulerAsyncProtection
from experiments.kv_ec.xor_codec import encode, recover


def snapshots():
    blocks, metadata = metadata_and_blocks(count=256)
    for meta in metadata:
        meta.update(checkpoint_version=meta["snapshot_id"], manager_epoch="manager",
                    request_version="v1", slot_generation=[1] * 256, input_ids_sha256="input")
    return blocks, metadata, [
        CPUSnapshot.freeze(block, meta, event_complete=True, started=time.monotonic())
        for block, meta in zip(blocks, metadata)]


def transport(manager):
    def call(socket, metadata, payload, timeout):
        if metadata.get("kind") == "status":
            return manager.status(metadata["snapshot_id"])
        return manager.submit(metadata, np.frombuffer(payload, dtype=np.uint8))
    return call


class AsyncTests(unittest.TestCase):
    def test_wire_envelopes_are_not_checkpoint_identity(self):
        saved = dict(state="COMMITTED", manager_epoch="manager", snapshot_id="cp",
                     checkpoint_version="cp", parity_sha256="hash", rank_hashes={"0": {"sha256": "rank"}})
        status = dict(saved, ipc_ms=1, transport_ms={"connect_ms": 1}, protocol="v1",
                      payload_bytes=0, kind="response")
        exported = dict(saved, protocol="v1", payload_bytes=9437184, kind="response")
        self.assertNotEqual(status, exported)
        self.assertEqual(checkpoint_metadata(status), saved)
        self.assertEqual(checkpoint_metadata(exported), saved)
        changed = dict(status, checkpoint_version="changed")
        self.assertNotEqual(checkpoint_metadata(status), checkpoint_metadata(changed))

    def wait(self, worker):
        self.assertTrue(worker.finished.wait(3))
        worker.thread.join(3)
        self.assertFalse(worker.thread.is_alive())

    def complete(self):
        blocks, meta, frozen = snapshots()
        manager = ParityManager(manager_epoch="manager")
        workers = [AsyncProtection("/tmp/unused", transport=transport(manager)) for _ in range(4)]
        for worker, snapshot in zip(workers, frozen):
            worker.stage(snapshot)
            worker.start()
        for worker in workers:
            self.wait(worker)
            self.assertEqual(worker.state, COMMITTED, worker.reason)
        return blocks, meta, frozen, manager, workers

    def test_normal_four_rank_byte_exact_parity_and_every_single_rank_recovery(self):
        blocks, meta, frozen, manager, workers = self.complete()
        status, raw = manager.export_parity("checkpoint", meta[0])
        parity = np.frombuffer(raw, dtype=np.uint8)
        np.testing.assert_array_equal(parity, encode(blocks))
        self.assertEqual(status["oracle_checks"], [])
        for lost in range(4):
            np.testing.assert_array_equal(
                recover([None if r == lost else b for r, b in enumerate(blocks)], parity), blocks[lost])
            self.assertEqual(workers[lost].require_recoverable(meta[lost])["state"], COMMITTED)
            self.assertEqual(workers[lost].resident_bytes, 0)
            self.assertIsNone(workers[lost].snapshot)
            self.assertEqual([t["state"] for t in workers[lost].trace],
                             [SNAPSHOT_READY, COMMITTING, COMMITTED])

    def test_snapshot_is_independent_immutable_bytes_and_deep_metadata(self):
        blocks, meta, frozen = snapshots()
        original = frozen[0].payload
        blocks[0][:] = 0
        meta[0]["physical_slots"][0] += 50
        detached = frozen[0].metadata
        detached["physical_slots"][0] += 50
        self.assertEqual(frozen[0].payload, original)
        self.assertNotEqual(frozen[0].metadata, detached)
        with self.assertRaises(ValueError):
            np.frombuffer(original, dtype=np.uint8).setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            frozen[0].payload = b"modified"

    def test_incomplete_copy_wrong_hash_and_missing_generation_rejected(self):
        blocks, meta, _ = snapshots()
        with self.assertRaisesRegex(ValueError, "copy_incomplete"):
            CPUSnapshot.freeze(blocks[0], meta[0], event_complete=False, started=time.monotonic())
        bad = blocks[0].copy()
        bad[0] ^= 1
        with self.assertRaises(ValueError):
            CPUSnapshot.freeze(bad, meta[0], event_complete=True, started=time.monotonic())
        del meta[0]["slot_generation"]
        with self.assertRaisesRegex(ValueError, "binding_incomplete"):
            CPUSnapshot.freeze(blocks[0], meta[0], event_complete=True, started=time.monotonic())

    def test_byte_budget_and_single_task_queue_include_inflight_work(self):
        _, _, frozen = snapshots()
        worker = AsyncProtection("/tmp/unused", max_bytes=1)
        with self.assertRaisesRegex(ValueError, "invalid_async"):
            worker.stage(frozen[0])
        entered, leave = threading.Event(), threading.Event()
        def pending(*a, **kw):
            entered.set()
            leave.wait(2)
            return {"state": "FAILED"}
        worker = AsyncProtection("/tmp/unused", transport=pending)
        worker.stage(frozen[0])
        worker.start()
        self.assertTrue(entered.wait(2))
        with self.assertRaisesRegex(ValueError, "queue_full"):
            worker.stage(frozen[1])
        self.assertEqual(worker.resident_bytes, len(frozen[0].payload))
        leave.set()
        self.wait(worker)
        self.assertEqual(worker.state, FAILED)

    def test_manager_error_and_timeout_never_become_recoverable(self):
        _, meta, frozen = snapshots()
        for fail in ("manager_error", "timeout"):
            def call(*a, **kw):
                if fail == "manager_error":
                    raise ConnectionError("manager_down")
                return {"state": "PENDING"}
            worker = AsyncProtection("/tmp/unused", transport=call, timeout_s=.04)
            worker.stage(frozen[0])
            worker.start()
            self.wait(worker)
            self.assertEqual(worker.state, FAILED)
            with self.assertRaisesRegex(ValueError, "not_committed"):
                worker.require_recoverable(meta[0])

    def test_fault_before_committed_fail_closed_and_rank_timing_skew(self):
        _, meta, frozen = snapshots()
        manager = ParityManager(manager_epoch="manager")
        workers = [AsyncProtection("/tmp/unused", transport=transport(manager)) for _ in range(4)]
        for r in range(3):
            workers[r].stage(frozen[r])
            workers[r].start()
        try:
            time.sleep(.03)
            self.assertEqual(manager.status("checkpoint")["state"], "PENDING")
            for r in range(3):
                with self.assertRaisesRegex(ValueError, "not_committed"):
                    workers[r].require_recoverable(meta[r])
            workers[3].stage(frozen[3])
            workers[3].start()
            for worker in workers:
                self.wait(worker)
                self.assertEqual(worker.state, COMMITTED)
        finally:
            for worker in workers:
                if worker.thread and worker.thread.is_alive():
                    worker.cancel()
                    self.wait(worker)

    def test_postcommit_stale_epoch_checkpoint_row_slots_and_request_rejected(self):
        _, meta, frozen, _, workers = self.complete()
        for key, value in (
            ("checkpoint_version", "v2"), ("manager_epoch", "restarted"),
            ("request_id", "other"), ("request_version", "v2"), ("req_generation", 999),
            ("physical_slots", list(range(1000, 1256))), ("slot_generation", [2] * 256)):
            changed = deepcopy(meta[0])
            changed[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "stale_binding"):
                    workers[0].require_recoverable(changed)
                with self.assertRaises(ValueError):
                    frozen[0].require_binding(changed)

    def test_cancel_ready_inflight_late_commit_and_after_commit_invalidate(self):
        _, meta, frozen, _, completed = self.complete()
        completed[0].cancel()
        with self.assertRaises(ValueError):
            completed[0].require_recoverable(meta[0])
        ready = AsyncProtection("/tmp/unused")
        ready.stage(frozen[0])
        ready.cancel()
        self.assertEqual(ready.resident_bytes, 0)
        with self.assertRaises(ValueError):
            ready.start()
        entered, leave = threading.Event(), threading.Event()
        status = completed[1].status
        def late(*a, **kw):
            entered.set()
            leave.wait(2)
            return status
        worker = AsyncProtection("/tmp/unused", transport=late)
        worker.stage(frozen[1])
        worker.start()
        self.assertTrue(entered.wait(2))
        worker.cancel()
        leave.set()
        self.wait(worker)
        self.assertEqual(worker.state, FAILED)
        self.assertIsNone(worker.status)
        with self.assertRaises(ValueError):
            worker.require_recoverable(meta[1])

    def test_checkpoint_changes_between_ranks_fail_manager_not_reused(self):
        _, meta, frozen = snapshots()
        manager = ParityManager(manager_epoch="manager")
        submitted = threading.Event()
        direct = transport(manager)
        def call(*a, **kw):
            result = direct(*a, **kw)
            submitted.set()
            return result
        first = AsyncProtection("/tmp/unused", transport=call)
        first.stage(frozen[0])
        first.start()
        self.assertTrue(submitted.wait(2))
        changed = frozen[1].metadata
        changed["checkpoint_version"] = "different"
        result = manager.submit(changed, np.frombuffer(frozen[1].payload, dtype=np.uint8))
        self.assertEqual(result["state"], FAILED)
        self.wait(first)
        self.assertEqual(first.state, FAILED)
        with self.assertRaises(ValueError):
            manager.export_parity("checkpoint", meta[0])

    def test_foreground_returns_while_worker_is_pending_and_commit_metrics_separate(self):
        _, _, frozen = snapshots()
        worker = AsyncProtection("/tmp/unused", transport=lambda *a, **kw: {"state": "PENDING"},
                                 timeout_s=.2)
        worker.stage(frozen[0])
        worker.start()
        self.assertFalse(worker.finished.is_set())
        self.assertIsNone(worker.record()["background_committed_ms"])
        worker.cancel()
        self.wait(worker)


class AsyncGateTests(unittest.TestCase):
    def test_actual_E_gate_releases_after_snapshot_then_decodes_before_commit(self):
        # Only device/IPC operations are faked; actual four gate handshakes,
        # row/slot leases, adapter, worker and ParityManager all execute.
        from experiments.kv_ec.m6_acceptance import check_async_evidence, async_performance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coordinator = PauseCoordinator("checkpoint", experiment_mode="async_protection_only")
            manager = ParityManager(manager_epoch="manager")
            blocks, metas = metadata_and_blocks(count=256, layers=tuple(range(36)))
            fixtures = [fixture(root, coordinator, r) for r in range(4)]
            env = dict(SGLANG_KV_EC_SHADOW_PARITY_SOCKET="/tmp/unused",
                       SGLANG_KV_EC_SHADOW_MANAGER_EPOCH="manager",
                       SGLANG_KV_EC_PAUSE_TIMEOUT_S="5")
            with patch.dict("os.environ", env, clear=True):
                for g, req, batch, event in fixtures:
                    g.shadow = SchedulerAsyncProtection()
                    g.shadow.worker.transport = transport(manager)
                    g.shadow.worker.commit_delay_s = .15
                    g.shadow.check_online_admission = Mock()
                    with patch("experiments.kv_ec.scheduler_pause.torch.cuda.is_available", return_value=False):
                        g.after_result(batch)
                    meta = metas[g.rank]
                    meta.update(
                        snapshot_id="checkpoint-chunk0", checkpoint_version="checkpoint-chunk0",
                        request_id=g.evidence["request_hash"], request_version=g.mapping.request_version,
                        manager_epoch="manager", input_ids_sha256="input", req_pool_idx=g.mapping.req_pool_idx,
                        req_generation=g.mapping.req_generation, physical_slots=list(g.mapping.physical_slots[:256]),
                        slot_generation=list(g.mapping.slot_generations[:256]))
                    for layer in meta["layer_records"]:
                        layer["device"] = f"cuda:{g.rank}"
                    g.scheduler.model_worker.model_runner = NS(
                        token_to_kv_pool=NS(k_buffer=[NS(device=NS(type="cuda"))]))
                    g.scheduler.forward_stream = Mock()
                    g.scheduler.device_module = NS(Event=lambda: NS(
                        record=Mock(), synchronize=Mock(), query=lambda: True))
                    g._vote = lambda ready: ready
                def control(path, message, **kwargs):
                    return dict(state="OK", status=coordinator.dispatch(message))
                def pack(pool, mapping, *a, **kw):
                    return blocks[mapping.tp_rank], metas[mapping.tp_rank]
                with patch("experiments.kv_ec.scheduler_pause.request", side_effect=control), patch(
                        "experiments.kv_ec.scheduler_async.pack_chunk", side_effect=pack), patch(
                        "experiments.kv_ec.scheduler_async.torch.cuda.is_available", return_value=True), patch(
                        "experiments.kv_ec.scheduler_async.torch.cuda.stream", return_value=nullcontext()), patch(
                        "experiments.kv_ec.scheduler_async.torch.cuda.max_memory_allocated", return_value=100), patch(
                        "experiments.kv_ec.scheduler_async.torch.cuda.max_memory_reserved", return_value=200):
                    for g, *_ in fixtures:
                        g._tick_local()
                    for g, *_ in fixtures:
                        g.next_poll = 0
                        g._tick_local()
                        self.assertEqual(g.state, "SHADOW_READY")
                    for g, *_ in fixtures:
                        g.next_poll = 0
                        g.tick()
                        self.assertEqual(g.shadow.worker.state, SNAPSHOT_READY)
                        self.assertEqual(g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 1)
                    coordinator.dispatch(dict(
                        epoch=coordinator.epoch, transaction_id="checkpoint", op="authorize"))
                    for _ in range(20):
                        for g, *_ in fixtures:
                            g.next_poll = 0
                            if g.state != "RESUMED":
                                g.tick()
                        if all(g.state == "RESUMED" for g, *_ in fixtures):
                            break
                    self.assertTrue(all(g.state == "RESUMED" for g, *_ in fixtures))
                    self.assertNotEqual(manager.status("checkpoint-chunk0")["state"], COMMITTED)
                    for g, req, batch, _ in fixtures:
                        self.assertEqual(g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)
                        req.output_ids.append(3)
                        batch.forward_mode = NS(is_decode=lambda: True, is_extend=lambda: False)
                        g.after_result(batch)
                        req.finished = lambda: True
                        g.after_result(batch)
                    for g, *_ in fixtures:
                        self.assertTrue(g.shadow.worker.finished.wait(3))
                        g.shadow.worker.thread.join(3)
                        self.assertEqual(g.shadow.worker.state, COMMITTED, g.shadow.worker.reason)
                # All four ranks use separate evidence directories in fixture;
                # assemble metadata-only records for the production auditor.
                combined = root / "combined"
                combined.mkdir()
                for g, *_ in fixtures:
                    for path in g.output.glob("*.json"):
                        (combined / path.name).write_bytes(path.read_bytes())
                response = dict(input_ids_sha256="input", client_metrics_ms=dict(
                    request_total_ms=1., ttft_ms=.1, tpot_ms=.1, note="synthetic CPU fixture"))
                checkpoint = manager.status("checkpoint-chunk0")
                records = check_async_evidence(
                    combined, fixtures[0][0].identity, [checkpoint], response, require_overlap=True)
                perf = async_performance(response, records, config={})
                self.assertTrue(perf["all_ranks_decode_before_commit"])
                for filename, key, value in (
                    ("async-rank-0.json", "state", FAILED),
                    ("async-lifecycle-rank-0.json", "cancelled", True),
                    ("async-rank-0.json", "unprotected_window_ms", 0),
                    ("async-rank-0.json", "submit_started_monotonic", 0)):
                    path = combined / filename
                    original = path.read_bytes()
                    changed = json.loads(original)
                    changed[key] = value
                    path.write_text(json.dumps(changed))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        check_async_evidence(combined, fixtures[0][0].identity, [checkpoint], response)
                    path.write_bytes(original)
                path = combined / "pause-rank-0.json"
                original = path.read_bytes()
                changed = json.loads(original)
                changed["async_protection_only"]["foreground_protection_blocking_ms"] += 1
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "scheduler_ready"):
                    check_async_evidence(combined, fixtures[0][0].identity, [checkpoint], response)
                path.write_bytes(original)

    def test_coordinator_waits_for_four_snapshot_proofs_but_never_parity(self):
        c = PauseCoordinator("tx", experiment_mode="async_protection_only")
        def call(op, **kw):
            return c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op=op, **kw))
        proof = dict(snapshot_id="tx-chunk0", checkpoint_version="tx-chunk0",
                     manager_epoch="manager", input_ids_sha256="input",
                     request_hash="r", request_version="v1", snapshot_ready=True,
                     copy_complete=True, independent_cpu_bytes=True, checkpoint_committed=False,
                     used_committed_parity=False, online_kv_pool_modified=False, fault_injected=False)
        for rank in range(4):
            call("paused", tp_rank=rank, observation=observation())
        for key, value in (("copy_complete", False), ("independent_cpu_bytes", False),
                           ("checkpoint_committed", True), ("checkpoint_version", "wrong")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing_proof"):
                call("shadow_verified", tp_rank=0, observation=dict(proof, **{key: value}))
        for rank in range(3):
            call("shadow_verified", tp_rank=rank, observation=proof)
        with self.assertRaises(ValueError):
            call("authorize")
        call("shadow_verified", tp_rank=3, observation=proof)
        self.assertEqual(call("authorize")["state"], "RESUME_REQUESTED")
        for op in ("validated", "released", "armed"):
            for rank in range(4):
                call(op, tp_rank=rank)
        self.assertEqual(c.state, "RESUME_AUTHORIZED")

    def test_incomplete_event_holds_real_slot_and_row_leases(self):
        with tempfile.TemporaryDirectory() as root:
            c = PauseCoordinator("tx", experiment_mode="async_protection_only")
            gate, req, batch, event = fixture(Path(root), c)
            with patch("experiments.kv_ec.scheduler_pause.torch.cuda.is_available", return_value=False):
                gate.after_result(batch)
            event.complete = False
            gate._vote = lambda ready: ready
            self.assertTrue(gate.tick())
            with self.assertRaisesRegex(RuntimeError, "protected_slot_free"):
                gate.scheduler.token_to_kv_pool_allocator.free(
                    __import__("torch").tensor(gate.mapping.physical_slots))
            with self.assertRaisesRegex(RuntimeError, "row_protected"):
                gate.scheduler.req_to_token_pool.free(req)
            gate.deadline = time.monotonic() - 1
            with self.assertRaises(PauseRejected):
                gate.tick()
            self.assertEqual(gate.state, "ABORTED_FAIL_CLOSED")


if __name__ == "__main__":
    unittest.main()
