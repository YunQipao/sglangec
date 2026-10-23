"""Offline audit/statistics tests; synthetic records are not GPU evidence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec import m4_analysis as analysis
from experiments.kv_ec import m4_trials as trials


MATRIX = Path("/tmp/kv-ec-m3-matrix-9h4va7t5")


class M4StatisticsTests(unittest.TestCase):
    def test_single_and_pilot_never_claim_stable_median_p95(self):
        for values in ([7.], [1., 2., 9.]):
            result = trials.statistics(values)
            self.assertEqual(result["samples_ms"], values)
            self.assertNotIn("median_ms", result)
            self.assertNotIn("p95_ms", result)
        result = trials.statistics(list(range(1, 11)))
        self.assertEqual(result["median_ms"], 5.5)
        self.assertAlmostEqual(result["p95_ms"], 9.55)
        for bad in ([], [True], [float("nan")], [float("inf")], [-1.]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                trials.statistics(bad)

    def test_duplicate_trials_and_config_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("one", "two"):
                (root / name).mkdir()
                (root / name / "baseline-response.json").write_text("{}")
            def record(path):
                path = Path(path)
                return dict(lost_rank=0, recover_tokens=256, fault_to_decode_ms=2.), dict(
                    transaction_id=path.name, baseline_path=str(path / "baseline-response.json"),
                    config={"model": "same"})
            with patch.object(trials, "audit_case", side_effect=record):
                result = trials.summarize_trials([root / "one", root / "two"])
                self.assertEqual(result["groups"][0]["n"], 2)
                self.assertNotIn("median_ms", result["groups"][0]["metrics"]["fault_to_decode_ms"])
                self.assertEqual(result["modes_unavailable"], ["B_protection_only", "D_real_prefill_retry"])
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    trials.summarize_trials([root / "one", root / "one"])
            a, b = record(root / "one"), record(root / "two")
            b[1]["config"] = {"model": "different"}
            with patch.object(trials, "audit_case", side_effect=[a, b]), self.assertRaisesRegex(
                    ValueError, "configuration_drift"):
                trials.summarize_trials([root / "one", root / "two"])

    def test_experiments_only_revision_change_requires_identical_measured_files_and_core(self):
        before = dict(source_commit="old", model="m", experiment_source_sha256={"live.py": "same"})
        after = dict(before, source_commit="new")
        with patch.object(trials.subprocess, "check_output", return_value=b"") as git:
            trials.compatible_config(after, before)
            git.assert_called_once()
        with patch.object(trials.subprocess, "check_output", return_value=b"core changed"), \
                self.assertRaisesRegex(ValueError, "inference_source_changed"):
            trials.compatible_config(after, before)
        changed = dict(after, experiment_source_sha256={"live.py": "changed"})
        with self.assertRaisesRegex(ValueError, "measurement_configuration_drift"):
            trials.compatible_config(changed, before)

    def test_ten_runs_require_pilot_before_work(self):
        with patch("sys.argv", ["m4_trials", "--matrix", "/prior", "--gpus", "0,1,2,3",
                                "--shadow-evidence", "/shadow", "--repeat", "10"]), \
                patch("sys.stderr"), patch.object(trials, "run_logged") as run:
            with self.assertRaises(SystemExit):
                trials.main()
            run.assert_not_called()

    def test_cpu_failure_starts_no_gpu_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new"
            with patch("sys.argv", ["m4_trials", "--matrix", "/prior", "--gpus", "0,1,2,3",
                                    "--shadow-evidence", "/shadow", "--repeat", "1",
                                    "--case", "0:256", "--output", str(root)]), \
                    patch.object(trials, "audit_matrix", return_value={"cases": [{"config": {}}]}), \
                    patch.object(trials, "run_logged", return_value=1) as run:
                self.assertEqual(trials.main(), 1)
                self.assertEqual(run.call_count, 1)
                manifest = json.loads((root / "trials.json").read_text())
                self.assertEqual(manifest["state"], "FAIL")
                self.assertFalse(manifest["trials"])

    def test_failed_gpu_trial_stops_remaining_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new"
            with patch("sys.argv", ["m4_trials", "--matrix", "/prior", "--gpus", "0,1,2,3",
                                    "--shadow-evidence", "/shadow", "--repeat", "3",
                                    "--output", str(root)]), \
                    patch.object(trials, "audit_matrix", return_value={
                        "cases": [{"config": {"model_identifier": "/model"}}]}), \
                    patch.object(trials, "run_logged", side_effect=[0, 1]) as run:
                self.assertEqual(trials.main(), 1)
                self.assertEqual(run.call_count, 2)
                manifest = json.loads((root / "trials.json").read_text())
                self.assertEqual(len(manifest["trials"]), 1)
                self.assertEqual(manifest["trials"][0]["state"], "FAIL")


@unittest.skipUnless((MATRIX / "matrix.json").exists(), "optional real saved M3 matrix not available")
class M4SavedGPUAuditTests(unittest.TestCase):
    def test_real_matrix_reaudits_without_cuda_or_subprocess(self):
        with patch("subprocess.Popen", side_effect=AssertionError("read-only audit must start nothing")):
            result = analysis.audit_matrix(MATRIX)
        self.assertEqual(result["case_count"], 12)
        self.assertEqual(result["independent_baselines"], 12)
        self.assertTrue(all(row["n"] == 1 for row in result["rows"]))

    def test_saved_performance_tamper_rejected(self):
        original = analysis.load
        def load(path):
            value = original(path)
            if Path(path).name == "performance-online.json":
                value["cpu_xor_ms"] += 100
            return value
        with patch.object(analysis, "load", side_effect=load), self.assertRaisesRegex(
                ValueError, "performance_value_mismatch:cpu_xor_ms"):
            analysis.audit_case(MATRIX / "rank0-tokens256")

    def test_missing_matrix_case_rejected(self):
        value = deepcopy(analysis.load(MATRIX / "matrix.json"))
        value["cases"].pop()
        with patch.object(analysis, "load", return_value=value), self.assertRaisesRegex(
                ValueError, "requires_12"):
            analysis.audit_matrix(MATRIX)

    def test_report_has_raw_observations_not_quantiles(self):
        result = analysis.audit_matrix(MATRIX)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report"
            analysis.write_report(result, path)
            text = (path / "REPORT.md").read_text()
            self.assertIn("n=1", text)
            self.assertIn("not pure XOR", text)
            self.assertTrue((path / "performance.csv").exists())
            with self.assertRaises(FileExistsError):
                analysis.write_report(result, path)


if __name__ == "__main__":
    unittest.main()
