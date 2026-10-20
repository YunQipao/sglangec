"""CPU checks for the acceptance verifier, not GPU experiment evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec.accept_pause import check_evidence, main, verify_run


IDENTITY = {"transaction_id": "test", "epoch": "test-epoch"}


def rank_record(rank):
    return dict(
        tp_rank=rank, state="RESUMED", transaction_id="test", coordinator_epoch="test-epoch",
        decode_observed=True, decode_forward_mode="decode", decode_output_count=2,
        scheduler_skipped_iterations=1, resume_authorized=True, tp_cpu_resume_vote=True,
        allocator_lease_released=True, row_protection_released=True,
        active_slot_leases_after_release=0, cuda_event_complete=True,
        cuda_event_recorded=True, online_kv_pool_modified=False, failure=None,
        request_generation_before=1, request_generation_after=1,
        slot_generations_before=[1] * 768, slot_generations_after=[1] * 768,
        pause_ms=1, token_count=768,
        request_hash=hashlib.sha256(b"artificial").hexdigest(),
        coordinator_final=dict(IDENTITY, state="RESUME_AUTHORIZED", paused_ranks=list(range(4)),
                               validated_ranks=list(range(4)), released_ranks=list(range(4)),
                               armed_ranks=list(range(4))))


def completed_run(root):
    (root / "ranks").mkdir()
    (root / "identity.json").write_text(json.dumps(IDENTITY))
    for rank in range(4):
        (root / "ranks" / f"pause-rank-{rank}.json").write_text(json.dumps(rank_record(rank)))
    (root / "server.log").write_text("server started\n")
    events = [dict(op=op, response={"state": "OK", "status": dict(
        IDENTITY, state=state, paused_ranks=list(range(4)))})
        for op, state in (("paused", "PAUSED"), ("authorize", "RESUME_REQUESTED"),
                          ("authorize", "RESUME_AUTHORIZED"))]
    (root / "coordinator.log").write_text("".join(
        "INFO:module:KV_EC_PAUSE_CONTROL " + json.dumps(event) + "\n" for event in events))
    response = dict(kind="real_server_artificial_prompt", request_id="artificial",
                    model_identifier="fixture", input_ids_sha256="fixture", input_token_count=768,
                    sampling={"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
                    output_ids=list(range(16)), finish_reason={"type": "length", "length": 16})
    for name in ("baseline-response.json", "pause-response.json"):
        (root / name).write_text(json.dumps(response))
    return root / "baseline-response.json", root / "pause-response.json"


class AcceptanceTests(unittest.TestCase):
    def test_incomplete_or_failed_rank_evidence_is_rejected(self):
        identity = IDENTITY
        base = rank_record(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank in range(4):
                (root / f"pause-rank-{rank}.json").write_text(json.dumps(dict(base, tp_rank=rank)))
            check_evidence(root, identity)
            for key, value in (
                ("decode_observed", False), ("state", "PAUSED"), ("row_protection_released", False),
                ("active_slot_leases_after_release", 1), ("online_kv_pool_modified", True),
                ("slot_generations_after", [2]), ("transaction_id", "stale"),
                ("decode_output_count", 1), ("tp_cpu_resume_vote", 1),
            ):
                with self.subTest(key=key):
                    record = dict(deepcopy(base), tp_rank=0)
                    record[key] = value
                    (root / "pause-rank-0.json").write_text(json.dumps(record))
                    with self.assertRaises(ValueError):
                        check_evidence(root, identity)
            (root / "pause-rank-0.json").write_text(json.dumps(dict(base, tp_rank=0)))
            (root / "pause-rank-0.error.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "failure"):
                check_evidence(root, identity)

    def test_unavailable_gpu_reports_fail_without_starting_processes(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "sys.argv", ["accept_pause", "--gpus", "0,1,2,3"]), patch(
                    "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output",
                    side_effect=subprocess.CalledProcessError(9, "nvidia-smi", output="driver unavailable")), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as popen:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(), 1)
            popen.assert_not_called()
            self.assertIn("FAIL:", output.getvalue())
            verdict = json.loads((Path(tmp) / "acceptance.json").read_text())
            self.assertFalse(verdict["M1"])
            self.assertIn("driver unavailable", verdict["reason"])

    def test_busy_gpu_never_starts_or_stops_existing_processes(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "sys.argv", ["accept_pause", "--gpus", "0,1,2,3"]), patch(
                    "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output", side_effect=[
                        "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n", "GPU-2, 1234\n"]), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as popen, patch(
                    "experiments.kv_ec.accept_pause.os.killpg") as killpg:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            popen.assert_not_called()
            killpg.assert_not_called()
            verdict = json.loads((Path(tmp) / "acceptance.json").read_text())
            self.assertFalse(verdict["M1"])
            self.assertIn("active compute tasks", verdict["reason"])

    def test_read_only_audit_accepts_complete_matching_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, paused = completed_run(root)
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            verdict = verify_run(root, baseline, paused)
            self.assertTrue(verdict["M1"], verdict)
            self.assertTrue(verdict["checks"]["output_tokens_identical"])
            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_identical_tokens_do_not_allow_sampling_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, paused = completed_run(root)
            record = json.loads(baseline.read_text())
            del record["sampling"]["ignore_eos"]
            baseline.write_text(json.dumps(record))
            verdict = verify_run(root, baseline, paused)
            self.assertFalse(verdict["M1"])
            self.assertTrue(verdict["checks"]["output_tokens_identical"])
            self.assertIn("mismatch: sampling", verdict["reason"])

    def test_missing_authorization_allocator_error_and_unbound_response_reject(self):
        for failure in ("authorization", "allocator", "request", "identity", "same_file"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline, paused = completed_run(root)
                if failure == "authorization":
                    (root / "coordinator.log").write_text("")
                elif failure == "allocator":
                    (root / "server.log").write_text("RuntimeError: protected_slot_free\n")
                elif failure == "request":
                    record = json.loads(paused.read_text())
                    record["request_id"] = "different"
                    paused.write_text(json.dumps(record))
                elif failure == "identity":
                    (root / "identity.json").write_text(json.dumps(dict(IDENTITY, epoch="restarted")))
                else:
                    baseline = paused
                verdict = verify_run(root, baseline, paused)
                self.assertFalse(verdict["M1"], verdict)
                self.assertEqual(verdict["state"], "FAIL")

    def test_offline_cli_never_checks_or_starts_gpu_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed_run(root)
            with patch("sys.argv", ["accept_pause", "--verify-run", tmp,
                                    "--report", str(root / "audit.json")]), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as popen, patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output") as check_output:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
                popen.assert_not_called()
                check_output.assert_not_called()
                self.assertTrue(json.loads((root / "audit.json").read_text())["M1"])


if __name__ == "__main__":
    unittest.main()
