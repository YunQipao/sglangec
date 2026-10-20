"""CPU checks for the acceptance verifier, not GPU experiment evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec.accept_pause import check_evidence, main


class AcceptanceTests(unittest.TestCase):
    def test_incomplete_or_failed_rank_evidence_is_rejected(self):
        identity = {"transaction_id": "test", "epoch": "test-epoch"}
        base = dict(
            state="RESUMED", transaction_id="test", coordinator_epoch="test-epoch",
            decode_observed=True, decode_forward_mode="decode", decode_output_count=2,
            scheduler_skipped_iterations=1, resume_authorized=True, tp_cpu_resume_vote=True,
            allocator_lease_released=True, row_protection_released=True,
            active_slot_leases_after_release=0, cuda_event_complete=True,
            cuda_event_recorded=True, online_kv_pool_modified=False, failure=None,
            request_generation_before=1, request_generation_after=1,
            slot_generations_before=[1], slot_generations_after=[1],
            coordinator_final=dict(identity, state="RESUME_AUTHORIZED", paused_ranks=list(range(4)),
                                   validated_ranks=list(range(4)), released_ranks=list(range(4)),
                                   armed_ranks=list(range(4))))
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


if __name__ == "__main__":
    unittest.main()
