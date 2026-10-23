"""CPU-only M6 study admission checks. Never starts a GPU subprocess."""

from contextlib import redirect_stdout
import io
import json
from copy import deepcopy
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
        ]), patch.object(m6_abe, "reaudit_study", return_value=dict(state="PASS")) as audit, patch.object(
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


class M6SummaryComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "study"
        self.root.mkdir()
        self.response = dict(
            kind="real_server_artificial_prompt", model_identifier="model", input_ids_sha256="input",
            input_token_count=768, sampling=dict(temperature=0, max_new_tokens=16, ignore_eos=True),
            output_ids=list(range(16)), finish_reason=dict(type="length", length=16))
        self.config = dict(model_identifier="model", input_ids_sha256="input",
                           sampling=self.response["sampling"], gpu_uuids=["g0", "g1", "g2", "g3"],
                           experiment_source_sha256={"source": "hash"})
        manifest = dict(state="PASS", repeat=1, source_sha256={"source": "hash"},
                        diagnostic_overlap_smoke=True, async_commit_delay_s=2, cpu_returncode=0,
                        trials=[dict(state="PASS", run=str(self.root / "repeat1"), repeat=1,
                                     completed_modes=list(m6_abe.MODES))])
        (self.root / "trials.json").write_text(json.dumps(manifest))
        names = (
            "test_actual_E_gate_releases_after_snapshot_then_decodes_before_commit",
            "test_postcommit_stale_epoch_checkpoint_row_slots_and_request_rejected",
            "test_cancel_ready_inflight_late_commit_and_after_commit_invalidate",
            "test_manager_error_and_timeout_never_become_recoverable",
        )
        (self.root / "cpu-regression.log").write_text(
            "\n".join(f"{name} (fixture.Test) ... ok" for name in names))
        self.paths = {}
        for mode in m6_abe.MODES:
            directory = self.root / "repeat1" / mode
            directory.mkdir(parents=True)
            (directory / "acceptance.json").write_text(json.dumps(dict(
                transaction_id=mode, capture_profile=False, async_commit_delay_s=2 if mode.startswith("async") else 0)))
            (directory / "baseline-response.json").write_text(json.dumps(self.response))
            self.paths[mode] = directory / (
                "baseline-response.json" if mode == "baseline" else "pause-response.json")
            self.paths[mode].write_text(json.dumps(self.response))
        self.enterContext(patch.object(m6_abe, "validate_cpu", return_value=dict(state="PASS")))
        self.enterContext(patch.object(m6_abe, "historical_sources", return_value=dict(state="PASS")))
        self.enterContext(patch.object(m6_abe, "synchronous_unprotected_window", return_value=5.))
        self.enterContext(patch.object(m6_abe, "audit_mode", side_effect=self.audit))
        self.enterContext(patch.object(m6_abe, "audit_e", side_effect=lambda path, **kw: self.audit(path, "async_protection_only")))

    def audit(self, path, mode):
        # Only upstream GPU/CPU evidence is simulated; summarize and the strict
        # request_debug.compare read all three actual response files unchanged.
        response = json.loads(self.paths[mode].read_text())
        if mode == "baseline":
            return dict(request_total_ms=10., gpu_uuids=self.config["gpu_uuids"]), response
        config = deepcopy(self.config)
        perf = dict(request_total_ms=20., config=config)
        if mode == "protection_only":
            perf["protection_blocking_ms"] = 5.
        else:
            config["async_commit_delay_s"] = 2
            perf.update(foreground_protection_blocking_ms=3., unprotected_window_ms=2000.)
        return perf, response

    def run_summary(self):
        with redirect_stdout(io.StringIO()):
            return m6_abe.summarize(self.root)

    def change(self, mode, key, value):
        record = json.loads(self.paths[mode].read_text())
        record[key] = value
        self.paths[mode].write_text(json.dumps(record))

    def test_A_B_E_identical_PASS_and_both_comparisons_anchor_A(self):
        with patch.object(m6_abe, "compare", wraps=m6_abe.compare) as compare:
            result = self.run_summary()
        self.assertEqual(result["state"], "PASS")
        self.assertEqual([c.args[0] for c in compare.call_args_list], [
            [self.paths["baseline"], self.paths["protection_only"]],
            [self.paths["baseline"], self.paths["async_protection_only"]]])
        self.assertEqual(set(result["modes"]), set(m6_abe.MODES))
        self.assertEqual(result["paired_deltas"][0]["E_minus_B_request_ms"], 0.)

    def test_B_output_different_FAIL(self):
        self.change("protection_only", "output_ids", [100] * 16)
        with self.assertRaisesRegex(AssertionError, "output_ids"):
            self.run_summary()

    def test_E_output_different_FAIL(self):
        self.change("async_protection_only", "output_ids", [100] * 16)
        with self.assertRaisesRegex(AssertionError, "output_ids"):
            self.run_summary()

    def test_B_E_equal_but_different_from_A_FAIL(self):
        self.change("baseline", "output_ids", [100] * 16)
        with self.assertRaisesRegex(AssertionError, "output_ids"):
            self.run_summary()

    def test_missing_response_in_any_mode_FAIL(self):
        for mode, path in self.paths.items():
            original = path.read_bytes()
            path.unlink()
            with self.subTest(mode=mode), self.assertRaises(FileNotFoundError):
                self.run_summary()
            path.write_bytes(original)

    def test_input_sampling_model_count_finish_reason_still_strict(self):
        for mode in m6_abe.MODES[1:]:
            for key, value in (
                ("model_identifier", "other"), ("input_ids_sha256", "other"),
                ("input_token_count", 767), ("sampling", dict(temperature=1)),
                ("finish_reason", dict(type="stop")),
            ):
                original = self.paths[mode].read_bytes()
                self.change(mode, key, value)
                with self.subTest(mode=mode, key=key), self.assertRaises((AssertionError, ValueError)):
                    self.run_summary()
                self.paths[mode].write_bytes(original)

    def test_original_summary_failure_reaudits_PASS_preserving_all_artifacts(self):
        path = self.root / "trials.json"
        manifest = json.loads(path.read_text())
        manifest.update(state="FAIL", reason=m6_abe.RECOVERABLE_SUMMARY_REASON)
        path.write_text(json.dumps(manifest))
        before = m6_abe.artifact_hashes(self.root)
        with self.assertRaisesRegex(ValueError, "manifest_not_PASS"):
            self.run_summary()
        with redirect_stdout(io.StringIO()):
            report = m6_abe.reaudit_study(self.root)
        self.assertEqual(report["state"], "PASS", report["reason"])
        self.assertEqual(report["original_manifest_state"], "FAIL")
        self.assertTrue(report["original_artifacts_unchanged"])
        self.assertEqual(before, m6_abe.artifact_hashes(self.root))
        self.assertFalse((self.root / "summary.json").exists())
        self.assertEqual(json.loads(path.read_text())["state"], "FAIL")

    def test_reaudit_rejects_other_failure_incomplete_modes_or_bad_output(self):
        path = self.root / "trials.json"
        original = json.loads(path.read_text())
        for failure in ("different_reason", "missing_mode", "bad_output"):
            manifest = deepcopy(original)
            manifest.update(state="FAIL", reason=m6_abe.RECOVERABLE_SUMMARY_REASON)
            if failure == "different_reason":
                manifest["reason"] = "GPU_failed"
            elif failure == "missing_mode":
                manifest["trials"][0]["completed_modes"].pop()
            else:
                self.change("async_protection_only", "output_ids", [123] * 16)
            path.write_text(json.dumps(manifest))
            with redirect_stdout(io.StringIO()):
                report = m6_abe.reaudit_study(self.root)
            self.assertEqual(report["state"], "FAIL", failure)
            self.assertTrue(report["original_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
