"""CPU provenance tests; Git fixtures are not simulated GPU acceptance."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec import study_validation as source
from experiments.kv_ec.study_audit import validate_completed, verify_study


class SourceProofTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.directory = self.repo / "experiments/kv_ec"
        self.directory.mkdir(parents=True)
        for name in source.MINIMUM_SOURCES:
            (self.directory / name).write_text("# unchanged fixture\n")
        (self.repo / "python").mkdir()
        (self.repo / "python/core.py").write_text("# inference\n")
        self.commit("initial")
        self.before = self.config()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo,
                                       stderr=subprocess.PIPE).decode().strip()

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)

    def config(self):
        config = {k: k for k in source.CONFIG_KEYS}
        config.update(source_commit=self.git("rev-parse", "HEAD"),
                      core_diff_sha256=source.EMPTY_SHA256,
                      experiment_source_sha256={
                          "scheduler_shadow.py": source.sha256_file(
                              self.directory / "scheduler_shadow.py")})
        return config

    def test_evidence_only_commit_and_missing_hashes_supplemented(self):
        (self.repo / "kv-ec-fixture").mkdir()
        (self.repo / "kv-ec-fixture/acceptance.json").write_text("{}")
        self.commit("archive")
        result = source.compatible_config(self.before, self.config(), repo=self.repo)
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["commit_change"], "evidence_only_additions")
        self.assertEqual(result["before"]["files"]["parity_manager.py"]["evidence"],
                         "git_blob_supplement")
        self.assertNotEqual(result["before"]["source_commit"], result["after"]["source_commit"])

    def test_real_runtime_change_rejected_even_when_legacy_anchor_unchanged(self):
        (self.directory / "parity_manager.py").write_text("# changed validation\n")
        self.commit("change manager")
        with self.assertRaisesRegex(ValueError, "runtime_source"):
            source.compatible_config(self.before, self.config(), repo=self.repo)

    def test_untracked_tracked_and_staged_source_changes_rejected(self):
        for kind in ("untracked", "tracked", "staged"):
            with self.subTest(kind=kind):
                target = self.directory / ("untracked.py" if kind == "untracked" else "parity_manager.py")
                original = target.read_text() if target.exists() else None
                target.write_text("# dirty\n")
                if kind == "staged":
                    self.git("add", str(target.relative_to(self.repo)))
                with self.assertRaisesRegex(ValueError, "dirty_or_untracked"):
                    source.check_live_sources(repo=self.repo)
                if kind == "staged":
                    self.git("restore", "--staged", str(target.relative_to(self.repo)))
                if original is None:
                    target.unlink()
                else:
                    target.write_text(original)

    def test_model_input_gpu_sampling_protection_and_software_changes_rejected(self):
        for key in ("model_identifier", "model_sha256", "input_ids_sha256", "gpu_uuids",
                    "sampling", "parity_test_oracle", "recover_tokens", "lost_rank",
                    "torch_version", "cuda_version", "dtype", "layout", "page_size", "tp"):
            with self.subTest(key=key):
                changed = dict(self.before, **{key: "different"})
                with self.assertRaisesRegex(ValueError, "incompatible_config:" + key):
                    source.compatible_config(self.before, changed, repo=self.repo)

    def test_missing_git_or_dependency_is_inconclusive(self):
        changed = dict(self.before, source_commit="f" * 40)
        with self.assertRaises(source.Inconclusive):
            source.compatible_config(self.before, changed, repo=self.repo)
        (self.directory / "checkpoint_protocol.py").unlink()
        self.commit("missing dependency")
        with self.assertRaises(source.Inconclusive):
            source.historical_sources(self.config(), repo=self.repo)

    def test_wrong_recorded_hash_and_non_evidence_diff_rejected(self):
        changed = deepcopy(self.before)
        changed["experiment_source_sha256"]["scheduler_shadow.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "recorded_source_differs"):
            source.historical_sources(changed, repo=self.repo)
        (self.repo / "requirements.txt").write_text("different-package\n")
        self.commit("dependency change")
        with self.assertRaisesRegex(ValueError, "non_evidence"):
            source.compatible_config(self.before, self.config(), repo=self.repo)

    def test_missing_recorded_source_anchors_is_not_treated_as_equality(self):
        changed = dict(self.before, experiment_source_sha256={})
        with self.assertRaises(source.Inconclusive):
            source.compatible_config(self.before, changed, repo=self.repo)

    def test_launch_parameter_changes_and_missing_parameters_rejected(self):
        parameters = dict(startup_timeout=600, pause_timeout=120, baseline_port=30359)
        with self.assertRaisesRegex(ValueError, "launch_parameters_changed"):
            source.compatible_config(
                self.before, self.before, repo=self.repo,
                before_parameters=parameters, after_parameters=dict(parameters, pause_timeout=240))
        with self.assertRaises(source.Inconclusive):
            source.compatible_config(
                self.before, self.before, repo=self.repo, before_parameters=parameters)

    def test_fingerprint_covers_all_requested_and_transitive_experiment_modules(self):
        from experiments.kv_ec.accept_pause import MEASURED_SOURCES
        self.assertTrue(set(source.MINIMUM_SOURCES) <= set(MEASURED_SOURCES))
        self.assertTrue({"xor_codec.py", "gpu_scratch_restore.py", "debug_capture.py",
                         "recovery_lease.py", "cross_rank_control.py", "cross_rank_lease.py",
                         "kv_pool_audit.py", "request_debug.py", "launch_debug.py",
                         "chunk_protection.py"} <= set(MEASURED_SOURCES))
        self.assertFalse(any(n.startswith("test_") for n in MEASURED_SOURCES))

    def test_preflight_mismatch_starts_no_CPU_or_GPU_process(self):
        from experiments.kv_ec.m4_abcd import main
        config = dict(self.before, model_sha256="different")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "smoke"
            smoke.mkdir()
            (smoke / "trials.json").write_text(json.dumps(dict(
                state="PASS", source_sha256=self.before["experiment_source_sha256"],
                run_parameters=dict(
                    gpus="0,1,2,3", model="/media/sda1/iic_data/luotao/models/Qwen3-8B",
                    baseline_port=30359, pause_port=30360, startup_timeout=600., pause_timeout=120.),
                trials=[])))
            with patch("sys.argv", [
                    "m4_abcd", "--gpus", "0,1,2,3", "--shadow-evidence", "/prior",
                    "--repeat", "3", "--smoke-evidence", str(smoke),
                    "--output", str(root / "study")]), patch(
                        "experiments.kv_ec.m4_abcd.summarize",
                        return_value=dict(repeat=1, capture_profile=False, config=self.before)), patch(
                        "experiments.kv_ec.m4_abcd.preflight_config", return_value=config), patch(
                        "experiments.kv_ec.m4_abcd.run_logged") as runner, redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
                runner.assert_not_called()
            self.assertIn("incompatible_config:model_sha256",
                          json.loads((root / "study/trials.json").read_text())["reason"])


STUDY = Path("/tmp/kv-ec-m5-hash-three")
SMOKE = Path("/tmp/kv-ec-m5-next-2Yab76/smoke")


@unittest.skipUnless((STUDY / "trials.json").is_file(), "saved real GPU study unavailable")
class SavedStudyAuditTests(unittest.TestCase):
    def test_original_FAIL_reaudits_PASS_without_mutation_or_GPU(self):
        original = (STUDY / "trials.json").read_bytes()
        with patch("experiments.kv_ec.m4_abcd.run_logged",
                   side_effect=AssertionError("no execution allowed")), patch(
                       "torch.cuda.is_available", side_effect=AssertionError("no GPU query")):
            report = verify_study(STUDY, SMOKE)
        self.assertEqual(report["state"], "PASS", report.get("reason"))
        self.assertEqual(report["original_manifest_state"], "FAIL")
        self.assertEqual(report["gpu_trial_validation"]["completed_modes"], 12)
        self.assertEqual(report["original_artifact_hashes"][str(STUDY / "trials.json")],
                         hashlib.sha256(original).hexdigest())
        self.assertEqual((STUDY / "trials.json").read_bytes(), original)
        self.assertFalse((STUDY / "summary.json").exists())

    def test_only_two_of_three_repeats_rejected(self):
        manifest = json.loads((STUDY / "trials.json").read_text())
        manifest["trials"] = manifest["trials"][:2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trials.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incomplete_repeat_count"):
                validate_completed(root, expected_repeat=3, allow_terminal_config_failure=True)

    def test_one_recovery_acceptance_FAIL_rejected(self):
        original_read = Path.read_text
        failing_path = STUDY / "repeat2/online_recovery/acceptance.json"

        def changed(path, *args, **kwargs):
            text = original_read(path, *args, **kwargs)
            if path == failing_path:
                value = json.loads(text)
                value["state"] = "FAIL"
                return json.dumps(value)
            return text
        with patch.object(Path, "read_text", changed):
            report = verify_study(STUDY, SMOKE)
        self.assertEqual(report["state"], "FAIL")
        self.assertIn("failed_online_recovery_acceptance", report["reason"])

    def test_missing_source_history_is_INCONCLUSIVE_not_PASS(self):
        with patch("experiments.kv_ec.study_audit.historical_sources",
                   side_effect=source.Inconclusive("missing_blob")):
            report = verify_study(STUDY, SMOKE)
        self.assertEqual(report["state"], "INCONCLUSIVE")

    def test_cpu_exit_zero_without_successful_log_rejected(self):
        from experiments.kv_ec.study_audit import validate_cpu
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cpu-regression.log").write_text("Ran 239 tests in 10s\nFAILED\n")
            with self.assertRaisesRegex(ValueError, "CPU_log"):
                validate_cpu(root, dict(cpu_returncode=0))

    def test_offline_CLI_never_runs_live_preflight(self):
        from experiments.kv_ec.m4_abcd import main
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
                "m4_abcd", "--verify-study", str(STUDY), "--smoke-evidence", str(SMOKE),
                "--output", str(Path(tmp) / "report.json")]), patch(
                "experiments.kv_ec.m4_abcd.preflight_config",
                side_effect=AssertionError("offline cannot probe GPU")), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
            self.assertEqual(json.loads((Path(tmp) / "report.json").read_text())["state"], "PASS")

    def test_comparison_accepts_reaudit_and_rejects_hand_edited_report(self):
        from experiments.kv_ec.m5_compare import audited_summary
        report = verify_study(STUDY, SMOKE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            summary, responses = audited_summary(path)
            self.assertEqual(summary["acceptance_origin"], "completed_run_offline_reaudit_PASS")
            self.assertEqual(len(responses), 3)
            report["reconstructed_summary"]["modes"]["baseline"]["request_total_ms"]["samples_ms"][0] = 1.
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "stale_or_modified"):
                audited_summary(path)

    def test_original_failed_manifest_cannot_be_used_as_original_PASS(self):
        from experiments.kv_ec.m5_compare import audited_summary
        with self.assertRaisesRegex(ValueError, "unrecoverable_manifest_state"):
            audited_summary(STUDY)


if __name__ == "__main__":
    unittest.main()
