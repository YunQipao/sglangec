"""CPU-only M6 study admission checks. Never starts a GPU subprocess."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec import m6_abe
from experiments.kv_ec.m6_acceptance import async_performance


class M6StudyTests(unittest.TestCase):
    def test_B_window_uses_same_prefill_origin_and_rejects_impossible_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ranks").mkdir()
            for rank in range(4):
                record = dict(prefill_completed_monotonic=1.,
                              protection=dict(committed_monotonic=1. + (rank + 1) / 100),
                              paused_observation_monotonic=2.)
                (root / "ranks" / f"pause-rank-{rank}.json").write_text(json.dumps(record))
            self.assertAlmostEqual(m6_abe.synchronous_unprotected_window(root), 40.)
            record["protection"]["committed_monotonic"] = .5
            (root / "ranks" / "pause-rank-0.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "clock_invalid"):
                m6_abe.synchronous_unprotected_window(root)

    def test_E_dirty_source_is_rejected_before_GPU_probe(self):
        from experiments.kv_ec import accept_pause
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
            "accept_pause", "--mode", "async_protection_only", "--gpus", "0,1,2,3",
            "--shadow-evidence", "/not-used", "--output", tmp + "/e"
        ]), patch("experiments.kv_ec.study_validation.check_live_sources",
                  side_effect=ValueError("dirty_sources")), patch.object(
                      accept_pause, "admit_online", side_effect=AssertionError("must not admit")), patch.object(
                          accept_pause.subprocess, "check_output",
                          side_effect=AssertionError("must not probe GPU")), redirect_stdout(io.StringIO()):
            self.assertEqual(accept_pause.main(), 1)
            saved = json.loads((Path(tmp) / "e" / "acceptance.json").read_text())
            self.assertEqual(saved["state"], "FAIL")
            self.assertFalse(saved["GPU_VALIDATED"])
            self.assertIn("dirty_sources", saved["reason"])

    def test_CPU_failure_never_launches_any_GPU_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "study"
            with patch("sys.argv", [
                "m6_abe", "--gpus", "0,1,2,3", "--shadow-evidence", "/not-used",
                "--diagnostic-overlap-smoke", "--async-commit-delay", "2", "--output", str(root)
            ]), patch.object(m6_abe, "check_live_sources"), patch.object(
                    m6_abe, "run_logged", return_value=1) as run, redirect_stdout(io.StringIO()):
                self.assertEqual(m6_abe.main(), 1)
            run.assert_called_once()
            self.assertIn("unittest", run.call_args.args[0])
            manifest = json.loads((root / "trials.json").read_text())
            self.assertEqual(manifest["state"], "FAIL")
            self.assertEqual(manifest["trials"], [])
            self.assertIn("no_GPU_launched", manifest["reason"])

    def test_undelayed_performance_requires_smoke_before_CPU_or_GPU(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
            "m6_abe", "--gpus", "0,1,2,3", "--shadow-evidence", "/not-used",
            "--repeat", "3", "--output", tmp + "/study"
        ]), patch.object(m6_abe, "run_logged") as run, patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                m6_abe.main()
            run.assert_not_called()

    def test_artificial_delay_cannot_leak_into_performance(self):
        with patch("sys.argv", [
            "m6_abe", "--gpus", "0,1,2,3", "--shadow-evidence", "/not-used",
            "--async-commit-delay", "2", "--smoke-evidence", "/not-used", "--output", "/not-used"
        ]), patch.object(m6_abe, "run_logged") as run, patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                m6_abe.main()
            run.assert_not_called()

    def test_incomplete_study_rejected_even_when_manifest_says_PASS(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cpu-regression.log").write_text("\nRan 12 tests in 1s\n\nOK\n")
            (root / "trials.json").write_text(json.dumps(dict(
                state="PASS", cpu_returncode=0, repeat=3, trials=[{}, {}])))
            with self.assertRaisesRegex(ValueError, "missing_repeats"):
                m6_abe.summarize(root)

    def test_offline_entry_cannot_launch_CPU_server_or_GPU(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
            "m6_abe", "--verify-study", "/saved", "--output", tmp + "/audit.json"
        ]), patch.object(m6_abe, "summarize", return_value=dict(state="PASS")) as audit, patch.object(
                m6_abe, "run_logged", side_effect=AssertionError("offline cannot launch")), redirect_stdout(io.StringIO()):
            self.assertEqual(m6_abe.main(), 0)
            audit.assert_called_once_with(Path("/saved"))
            self.assertEqual(json.loads((Path(tmp) / "audit.json").read_text())["state"], "PASS")

    def test_mode_normalization_removes_only_documented_async_settings(self):
        source = dict(tp=4, manager_epoch="manager", async_queue_capacity=1,
                      async_pending_policy="bounded_off_path_drain",
                      async_snapshot_budget_bytes=9437184, async_commit_delay_s=2,
                      unknown_protection_change=True)
        self.assertEqual(m6_abe.base_config(source), dict(
            tp=4, manager_epoch="manager", unknown_protection_change=True))
        self.assertEqual(source["async_commit_delay_s"], 2)

    def test_background_completion_is_not_added_to_request_total(self):
        response = dict(client_metrics_ms=dict(
            request_total_ms=200, ttft_ms=100, tpot_ms=10, note="original client"))
        metrics = dict(snapshot_ms=20, background_committed_ms=500,
                       unprotected_window_ms=530, foreground_protection_blocking_ms=30,
                       peak_snapshot_bytes=9437184, peak_cpu_rss_bytes=100000000,
                       peak_gpu_allocated_bytes=1000000000, peak_gpu_reserved_bytes=2000000000,
                       decode_before_commit=True)
        perf = async_performance(response, [dict(async_metrics=metrics) for _ in range(4)], config={})
        self.assertEqual(perf["request_total_ms"], 200)
        self.assertEqual(perf["foreground_protection_blocking_ms"], 30)
        self.assertEqual(perf["background_committed_ms"], 500)
        self.assertTrue(perf["all_ranks_decode_before_commit"])


if __name__ == "__main__":
    unittest.main()
