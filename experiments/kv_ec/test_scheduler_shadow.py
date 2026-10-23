"""CPU regression only. These tests are not real GPU recovery evidence."""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.kv_ec.accept_pause import check_shadow_evidence, main
from experiments.kv_ec.parity_manager import ParityManager, ParityManagerServer
from experiments.kv_ec.pause_control import PauseCoordinator
from experiments.kv_ec.scheduler_pause import PauseRejected, SchedulerPauseGate
from experiments.kv_ec.scheduler_shadow import (
    SchedulerShadowRecovery, checked_buffers, pack_chunk, reconstruct, validate_checkpoint,
)
from experiments.kv_ec.test_accept_pause import IDENTITY, completed_run
from experiments.kv_ec.test_parity_manager import metadata_and_blocks
from experiments.kv_ec.test_scheduler_pause import fixture, observation


def committed():
    blocks, metadata = metadata_and_blocks()
    manager = ParityManager(test_oracle=True)
    for meta in metadata:
        meta.update(manager_epoch=manager.manager_epoch, request_version="v1", checkpoint_version="cp1")
    for meta, block in zip(metadata, blocks):
        status = manager.submit(meta, block)
    return manager, blocks, metadata, status


class ShadowTests(unittest.TestCase):
    def test_committed_parity_reconstructs_from_only_three_survivors(self):
        manager, blocks, metadata, status = committed()
        exported, raw = manager.export_parity("checkpoint", metadata[0])
        self.assertFalse(manager._checkpoints["checkpoint"].ranks)
        self.assertIsNone(manager._checkpoints["checkpoint"].oracle)
        restored = reconstruct(np.frombuffer(raw, dtype=np.uint8), blocks[1:], exported, metadata[0])
        np.testing.assert_array_equal(restored, blocks[0])
        # Erasing or corrupting the original oracle does not affect recovery.
        blocks[0][:] = 0
        again = reconstruct(np.frombuffer(raw, dtype=np.uint8), blocks[1:], status, metadata[0])
        np.testing.assert_array_equal(again, restored)
        changed = bytearray(raw)
        changed[0] ^= 1
        with self.assertRaisesRegex(ValueError, "parity_or_survivor"):
            reconstruct(np.frombuffer(changed, dtype=np.uint8), blocks[1:], status, metadata[0])

    def test_parity_export_is_version_bound_and_owns_bytes(self):
        manager, blocks, metadata, status = committed()
        for key in ("manager_epoch", "request_version", "checkpoint_version", "snapshot_id",
                    "model_sha256", "token_start", "token_ids_sha256", "input_ids_sha256"):
            expected = deepcopy(metadata[0])
            expected[key] = "wrong"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity_mismatch"):
                manager.export_parity("checkpoint", expected)
        pending = ParityManager()
        with self.assertRaisesRegex(ValueError, "not_committed"):
            pending.export_parity("checkpoint", metadata[0])
        status["rank_hashes"]["0"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "version_or_rank"):
            validate_checkpoint(status, metadata[0])

    def test_survivor_corruption_and_missing_oracles_rejected(self):
        manager, blocks, metadata, status = committed()
        _, raw = manager.export_parity("checkpoint", metadata[0])
        blocks[2][0] ^= 1
        with self.assertRaisesRegex(ValueError, "survivor_version_or_bytes"):
            reconstruct(np.frombuffer(raw, dtype=np.uint8), blocks[1:], status, metadata[0])
        status["oracle_checks"] = []
        with self.assertRaisesRegex(ValueError, "four_parity_oracles"):
            validate_checkpoint(status, metadata[0])

    def test_parity_socket_handler_exports_payload_without_rank_retention(self):
        manager, blocks, metadata, status = committed()
        conn = Mock()
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=False)
        with patch("experiments.kv_ec.parity_manager.recv_message", return_value=(
                {"kind": "parity", "snapshot_id": "checkpoint", "expected": metadata[0]}, b"")), patch(
                "experiments.kv_ec.parity_manager.send_message") as send:
            ParityManagerServer("/tmp/unused.sock", manager)._handle(conn)
        args = send.call_args.args
        self.assertEqual(args[1]["state"], "COMMITTED")
        self.assertIsInstance(args[2], bytes)
        self.assertEqual(len(args[2]), len(blocks[0]))
        self.assertFalse(manager._checkpoints["checkpoint"].ranks)

    def test_real_layout_packing_uses_noncontiguous_physical_slot_order(self):
        # All BF16 bit patterns, including NaNs, must remain uninterpreted.
        raw = torch.arange(300 * 2 * 128 * 2, dtype=torch.int64).to(torch.uint8)
        k = raw.view(torch.bfloat16).reshape(300, 2, 128)
        v = raw.bitwise_xor(91).view(torch.bfloat16).reshape(300, 2, 128)
        pool = NS(k_buffer=[k] * 36, v_buffer=[v] * 36,
                  get_kv_buffer=lambda layer: (k, v))
        slots = tuple(range(256, 0, -1))
        mapping = NS(physical_slots=slots, slot_generations=(7,) * 256, req_pool_idx=1,
                     req_generation=5, request_version="v1", tp_rank=0)
        req = NS(rid="fixture", origin_input_ids=[1] * 768)
        before = (raw.clone(), v.view(torch.uint8).clone())
        payload, meta = pack_chunk(pool, mapping, req, {}, snapshot_id="cp", manager_epoch="epoch")
        expected_k = k.view(torch.uint8)[list(slots)].reshape(-1).numpy()
        expected_v = v.view(torch.uint8)[list(slots)].reshape(-1).numpy()
        for record in meta["layer_records"]:
            for kind, expected in (("k", expected_k), ("v", expected_v)):
                offset, count = record[kind + "_offset"], record[kind + "_nbytes"]
                np.testing.assert_array_equal(payload[offset:offset + count], expected)
        self.assertTrue(torch.equal(raw, before[0]))
        self.assertTrue(torch.equal(v.view(torch.uint8), before[1]))
        self.assertEqual(meta["physical_slots"], list(slots))
        self.assertEqual(meta["slot_generation"], [7] * 256)

    def test_real_layout_packing_selects_requested_independent_chunk(self):
        raw = torch.arange(1024 * 2 * 128 * 2, dtype=torch.int64).to(torch.uint8)
        k = raw.view(torch.bfloat16).reshape(1024, 2, 128)
        v = raw.bitwise_xor(37).view(torch.bfloat16).reshape(1024, 2, 128)
        pool = NS(k_buffer=[k] * 36, v_buffer=[v] * 36,
                  get_kv_buffer=lambda layer: (k, v))
        slots = tuple(range(768, 0, -1))
        mapping = NS(physical_slots=slots, slot_generations=(9,) * 768, req_pool_idx=1,
                     req_generation=5, request_version="v1", tp_rank=2)
        req = NS(rid="fixture", origin_input_ids=list(range(768)))
        payload, meta = pack_chunk(
            pool, mapping, req, {}, snapshot_id="cp-chunk1", manager_epoch="epoch",
            token_start=256, token_count=256)
        self.assertEqual(meta["token_start"], 256)
        self.assertEqual(meta["token_count"], 256)
        self.assertEqual(meta["physical_slots"], list(slots[256:512]))
        self.assertEqual(meta["slot_generation"], [9] * 256)
        self.assertEqual(meta["token_ids_sha256"], __import__(
            "experiments.kv_ec.snapshot", fromlist=["digest"]).digest(
                np.asarray(req.origin_input_ids[256:512], dtype="<i8")))
        self.assertEqual(payload.size, 36 * 2 * 256 * 2 * 128 * 2)
        with self.assertRaisesRegex(ValueError, "256_aligned"):
            pack_chunk(pool, mapping, req, {}, snapshot_id="bad", manager_epoch="epoch",
                       token_start=0, token_count=512)

    def test_invalid_mapping_stride_or_owner_is_rejected(self):
        tensor = torch.empty((300, 2, 128), dtype=torch.bfloat16)
        pool = NS(k_buffer=[tensor], v_buffer=[tensor], get_kv_buffer=lambda layer: (tensor, tensor))
        good = torch.arange(1, 257, dtype=torch.int64)
        checked_buffers(pool, 0, good)
        for slots in (good[:255], good * 0, good + 299, good.int()):
            with self.subTest(slots=slots.shape), self.assertRaisesRegex(ValueError, "mapping"):
                checked_buffers(pool, 0, slots)
        pool.get_kv_buffer = lambda layer: (tensor.clone(), tensor)
        with self.assertRaisesRegex(ValueError, "mapping"):
            checked_buffers(pool, 0, good)
        pool.get_kv_buffer = lambda layer: (tensor.transpose(1, 2), tensor)
        with self.assertRaisesRegex(ValueError, "mapping"):
            checked_buffers(pool, 0, good)

    def test_coordinator_requires_all_shadow_acks_before_authorizing(self):
        c = PauseCoordinator("tx", require_shadow=True)
        def dispatch(op, **values):
            return c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op=op, **values))
        for rank in range(4):
            dispatch("paused", tp_rank=rank, observation=observation())
        proof = dict(snapshot_id="tx-chunk0", checkpoint_version="tx-chunk0", manager_epoch="manager",
                     request_hash="r", request_version="v1", input_ids_sha256="input-hash",
                     parity_sha256="digest", byte_exact=True, online_kv_pool_modified=False)
        for rank in range(4):
            with self.assertRaisesRegex(ValueError, "four_rank_shadow"):
                dispatch("authorize")
            dispatch("shadow_verified", tp_rank=rank, observation=proof)
        self.assertEqual(dispatch("authorize")["state"], "RESUME_REQUESTED")
        for op in ("validated", "released", "armed"):
            for rank in range(4):
                dispatch(op, tp_rank=rank)
        self.assertEqual(c.state, "RESUME_AUTHORIZED")

    def test_shadow_version_disagreement_is_terminal(self):
        c = PauseCoordinator("tx", require_shadow=True)
        for rank in range(4):
            c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op="paused",
                            tp_rank=rank, observation=observation()))
        proof = dict(snapshot_id="tx-chunk0", checkpoint_version="tx-chunk0", manager_epoch="manager",
                     request_hash="r", request_version="v1", input_ids_sha256="input-hash",
                     parity_sha256="digest", byte_exact=True, online_kv_pool_modified=False)
        for rank in range(2):
            status = c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op="shadow_verified",
                                     tp_rank=rank, observation=dict(proof, manager_epoch=str(rank))))
        self.assertEqual(status["state"], "FAILED")
        self.assertEqual(status["failure_reason"], "shadow_identity_mismatch")

    def test_shadow_failure_keeps_existing_row_and_slot_lease(self):
        for failure in ("rank0_copy_failed", "shadow_manager_epoch_changed", "shadow_cuda_event_incomplete"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, patch(
                    "experiments.kv_ec.scheduler_pause.torch.cuda.is_available", return_value=False):
                c = PauseCoordinator("tx", require_shadow=True)
                gate, req, batch, event = fixture(Path(tmp), c)
                with patch("experiments.kv_ec.scheduler_pause.request", side_effect=lambda path, msg, **kw: {
                        "state": "OK", "status": c.dispatch(msg)}):
                    gate.after_result(batch)
                    gate._vote = lambda ready: ready
                    gate.tick()
                    obs = next(iter(c.observations.values()))
                    for rank in range(1, 4):
                        c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op="paused", tp_rank=rank,
                                        observation=obs))
                    gate.shadow = NS(done=False, run=Mock(side_effect=RuntimeError(failure)))
                    gate.next_poll = 0
                    with self.assertRaisesRegex(PauseRejected, failure):
                        gate.tick()
                self.assertEqual(gate.state, "ABORTED_FAIL_CLOSED")
                self.assertFalse(c.shadow)
                registry = gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry
                self.assertEqual(registry.active_lease_count, 1)
                self.assertTrue(registry.closed)
                with self.assertRaisesRegex(RuntimeError, "row_protected"):
                    gate.scheduler.req_to_token_pool.free(req)

    def test_four_gates_shadow_then_authorized_decode_preserves_lease_lifetime(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "experiments.kv_ec.scheduler_pause.torch.cuda.is_available", return_value=False):
            c = PauseCoordinator("tx", require_shadow=True)
            fixtures = [fixture(Path(tmp), c, rank) for rank in range(4)]
            gates = [item[0] for item in fixtures]
            def dispatch(path, message, **kwargs):
                return {"state": "OK", "status": c.dispatch(message)}
            with patch("experiments.kv_ec.scheduler_pause.request", side_effect=dispatch):
                for gate, req, batch, event in fixtures:
                    gate.after_result(batch)
                    gate._tick_local()
                    gate.shadow = NS(done=False, validate=Mock())
                for gate in gates:
                    gate.next_poll = 0
                    gate._tick_local()
                    self.assertEqual(gate.state, "SHADOW_READY")
                def mock_shadow(gate):
                    # Mock compute only; ownership and coordinator are real CPU objects.
                    allocator = gate.scheduler.token_to_kv_pool_allocator
                    self.assertEqual(allocator.slot_lease_registry.active_lease_count, 1)
                    with self.assertRaisesRegex(RuntimeError, "protected_slot_free"):
                        allocator.free(torch.arange(1, 769))
                    with self.assertRaisesRegex(RuntimeError, "row_protected"):
                        gate.scheduler.req_to_token_pool.free(gate.req)
                    return {"consensus": dict(
                        snapshot_id="tx-chunk0", checkpoint_version="tx-chunk0", manager_epoch="manager",
                        input_ids_sha256="input-hash",
                        request_hash=gate.evidence["request_hash"], request_version=gate.mapping.request_version,
                        parity_sha256="digest", byte_exact=True, online_kv_pool_modified=False)}
                for gate in gates:
                    gate.shadow.run = mock_shadow
                    gate._vote = lambda ready: ready
                    self.assertTrue(gate.tick())
                self.assertEqual(sorted(c.shadow), list(range(4)))
                self.assertTrue(all(gate.shadow.done and not gate.released for gate in gates))
                c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op="authorize"))
                for gate in gates:
                    gate._vote = lambda ready: ready and all(
                        other.state in ("RESUME_READY", "RESUMED") for other in gates)
                for _ in range(20):
                    for gate in gates:
                        gate.next_poll = 0
                        gate.tick()
                    if all(gate.state == "RESUMED" for gate in gates):
                        break
                self.assertTrue(all(gate.state == "RESUMED" for gate in gates))
                for gate, req, batch, event in fixtures:
                    req.output_ids.append(9)
                    gate.after_result(NS(reqs=[req], forward_mode=NS(is_decode=lambda: True)))
                    self.assertTrue(gate.evidence["decode_observed"])
                    self.assertEqual(gate.evidence["slot_generations_before"], gate.evidence["slot_generations_after"])
                    self.assertEqual(gate.evidence["physical_slots_before"], gate.evidence["physical_slots_after"])
                    self.assertTrue(gate.evidence["row_protection_released"])

    def test_no_cuda_and_online_switch_never_fall_back_to_live_writes(self):
        shadow = SchedulerShadowRecovery("/tmp/parity.sock", "epoch")
        with patch("experiments.kv_ec.scheduler_shadow.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "real_cuda"):
                shadow.run(NS())
        with patch.dict("os.environ", {"SGLANG_KV_EC_ONLINE_RESTORE": "1"}):
            with self.assertRaisesRegex(ValueError, "online_restore_disabled"):
                SchedulerShadowRecovery.from_env()
            with self.assertRaisesRegex(PauseRejected, "online_restore_disabled"):
                SchedulerPauseGate.from_env(NS(server_args=NS()))
        with patch("experiments.kv_ec.scheduler_shadow.manager_hello", return_value="restarted"):
            with self.assertRaisesRegex(ValueError, "manager_epoch_changed"):
                shadow.validate(NS(deadline=__import__("time").monotonic() + 10))
        shadow.event = NS(query=lambda: False)
        with patch("experiments.kv_ec.scheduler_shadow.manager_hello", return_value="epoch"):
            with self.assertRaisesRegex(ValueError, "cuda_event_incomplete"):
                shadow.validate(NS(deadline=__import__("time").monotonic() + 10))

    def test_m1_evidence_cannot_be_misreported_as_shadow_or_m2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed_run(root)
            with self.assertRaisesRegex(ValueError, "missing_shadow"):
                check_shadow_evidence(root / "ranks", IDENTITY)
            with patch("sys.argv", ["accept_pause", "--verify-run", tmp, "--shadow",
                                    "--report", str(root / "shadow-audit.json")]), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as spawn:
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 1)
                spawn.assert_not_called()
            verdict = json.loads((root / "shadow-audit.json").read_text())
            self.assertFalse(verdict["M2_PASS"])
            self.assertFalse(verdict["SHADOW_PASS"])

    def test_shadow_acceptance_checks_mapping_consensus_and_gpu_timing(self):
        # Synthetic evidence exercises the verifier, not GPU execution.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed_run(root)
            for rank in range(4):
                path = root / "ranks" / f"pause-rank-{rank}.json"
                record = json.loads(path.read_text())
                slots = list(range(1, 769))
                from experiments.kv_ec.snapshot import digest
                record.update(row=2, physical_slots_before=slots, physical_slots_after=slots)
                record["coordinator_final"].update(require_shadow=True, shadow_ranks=list(range(4)))
                record["shadow"] = dict(
                    mode="shadow", tp_rank=rank, recovered_rank=0, token_start=0, token_count=256,
                    layers=list(range(36)), device=f"cuda:{rank}", source_unchanged=True,
                    byte_exact=True, cuda_event_complete=True, used_committed_parity=True,
                    online_kv_pool_modified=False, rank0_contribution="zeros_excluded_from_recovery",
                    request_row=2, request_generation=1, request_version="v1", input_ids_sha256="fixture",
                    physical_slots_sha256=digest(np.asarray(slots[:256], dtype="<i8")),
                    slot_generations_sha256=digest(np.asarray([1] * 256, dtype="<i8")),
                    oracle_checks=[{"lost_rank": r, "byte_exact": True} for r in range(4)],
                    timing_ms={key: 1. for key in ("shadow_h2d_ms", "shadow_verify_ms", "shadow_vs_live_ms",
                                                  "xor_and_cpu_verify_ms", "parity_fetch_ms", "shadow_h2d_cuda_event_ms")},
                    consensus=dict(snapshot_id="test-chunk0", checkpoint_version="test-chunk0",
                                   request_hash=record["request_hash"], request_version="v1",
                                   input_ids_sha256="fixture", manager_epoch="manager", parity_sha256="digest",
                                   byte_exact=True, online_kv_pool_modified=False))
                path.write_text(json.dumps(record))
            self.assertEqual(set(check_shadow_evidence(root / "ranks", IDENTITY)), {"0", "1", "2", "3"})
            path = root / "ranks" / "pause-rank-0.json"
            good = json.loads(path.read_text())
            for key, value in (("device", "cpu"), ("request_generation", 2), ("byte_exact", False),
                               ("physical_slots_sha256", "wrong"), ("online_kv_pool_modified", True),
                               ("timing_ms", {}), ("oracle_checks", []), ("input_ids_sha256", "wrong")):
                bad = deepcopy(good)
                bad["shadow"][key] = value
                path.write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check_shadow_evidence(root / "ranks", IDENTITY)

    def test_shadow_driver_failure_reports_m2_false_without_starting_processes(self):
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp, patch(
                "sys.argv", ["accept_pause", "--shadow", "--gpus", "0,1,2,3"]), patch(
                "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                "experiments.kv_ec.accept_pause.subprocess.check_output",
                side_effect=subprocess.CalledProcessError(9, "nvidia-smi", output="driver unavailable")), patch(
                "experiments.kv_ec.accept_pause.subprocess.Popen") as spawn:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(), 1)
            spawn.assert_not_called()
            self.assertIn("M2_PASS=false", output.getvalue())
            self.assertIn("SHADOW_PASS=false", output.getvalue())
            verdict = json.loads((Path(tmp) / "acceptance.json").read_text())
            self.assertFalse(verdict["M2_PASS"])
            self.assertIn("driver unavailable", verdict["reason"])


if __name__ == "__main__":
    unittest.main()
