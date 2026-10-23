"""Re-audit saved GPU evidence without starting any server or GPU operation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re

from .study_validation import (
    Inconclusive, compatible_config, historical_sources, runtime_sources, sha256_file,
)


RECOVERABLE_REASON = "ValueError: configuration_changed_since_smoke"
CRITICAL_CPU_TESTS = (
    "test_bad_version_generation_or_partial_write_fails_closed",
    "test_four_rank_handoff_and_rebuild_before_external_authorization",
    "test_committed_parity_reconstructs_from_only_three_survivors",
    "test_survivor_corruption_and_missing_oracles_rejected",
    "test_generation_and_manager_epoch_fail_closed",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def artifact_hashes(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"symlink_in_evidence:{path}")
        if path.is_file():
            result[str(path)] = sha256_file(path)
    return result


def validate_cpu(root, manifest):
    require(manifest.get("cpu_returncode") == 0, "CPU_regression_exit_not_zero")
    path = root / "cpu-regression.log"
    if not path.is_file():
        raise Inconclusive(f"missing_CPU_regression_log:{path}")
    text = path.read_text()
    match = re.search(r"\nRan (\d+) tests? in ([\d.]+)s\s*\n\s*"
                      r"OK(?: \(skipped=(\d+)\))?\s*\Z", text)
    require(match is not None and int(match[1]) > 0, "CPU_log_missing_successful_terminal_summary")
    for name in CRITICAL_CPU_TESTS:
        require(re.search(r"^" + re.escape(name) + r" \([^\n]+\) \.\.\. ok$",
                          text, re.MULTILINE) is not None,
                f"critical_CPU_test_missing_or_skipped:{name}")
    # Do not mistake expected negative-case tracebacks embedded in unittest
    # output for runner failure; the terminal unittest summary is authoritative.
    return dict(state="PASS", returncode=0, tests=int(match[1]),
                elapsed_s=float(match[2]), skipped=int(match[3] or 0),
                critical_tests_passed=list(CRITICAL_CPU_TESTS),
                log=str(path), log_sha256=sha256_file(path))


def validate_completed(root, *, expected_repeat=None, allow_terminal_config_failure=False):
    from .m4_abcd import MODES, summarize
    root = Path(root).resolve()
    manifest = json.loads((root / "trials.json").read_text())
    allowed_failure = (allow_terminal_config_failure
                       and manifest.get("reason") == RECOVERABLE_REASON)
    require(manifest.get("state") == "PASS" or (
        manifest.get("state") == "FAIL" and allowed_failure), "unrecoverable_manifest_state")
    repeat = manifest.get("repeat")
    require(type(repeat) is int and repeat in (1, 3), "unsupported_repeat_count")
    require(expected_repeat is None or repeat == expected_repeat, "unexpected_repeat_count")
    trials = manifest.get("trials", [])
    require(len(trials) == repeat, "incomplete_repeat_count")
    require([t.get("repeat") for t in trials] == list(range(1, repeat + 1)),
            "duplicate_or_missing_repeat")
    for trial in trials:
        require(trial.get("state") == "PASS"
                and trial.get("completed_modes") == list(MODES), "incomplete_ABCD_trial")
        require(Path(trial["run"]).resolve() == root / f"repeat{trial['repeat']}",
                "trial_path_outside_study")
    cpu = validate_cpu(root, manifest)
    with redirect_stdout(io.StringIO()):
        summary = summarize(root)
    require(summary["repeat"] == repeat, "reconstructed_repeat_count_mismatch")
    reference = None
    ports = None
    for trial in trials:
        base = Path(trial["run"])
        for mode in MODES:
            acceptance = json.loads((base / mode / "acceptance.json").read_text())
            require(acceptance.get("GPU_VALIDATED") is True, "missing_GPU_VALIDATED")
            current_ports = {k: acceptance[k] for k in ("baseline_port", "pause_port")}
            require(ports is None or ports == current_ports, "study_port_configuration_drift")
            ports = current_ports
            response = json.loads((base / mode / (
                "baseline-response.json" if mode == "baseline" else "pause-response.json")).read_text())
            values = {k: response[k] for k in (
                "model_identifier", "input_ids_sha256", "input_token_count",
                "sampling", "output_ids", "finish_reason")}
            require(reference is None or reference == values, "cross_repeat_response_mismatch")
            reference = values
    proof = historical_sources(summary["config"], manifest["source_sha256"])
    return dict(manifest=manifest, summary=summary, cpu=cpu, source=proof,
                effective_parameters=dict(ports, chunk_tokens=256, modes=list(MODES)),
                gpu=dict(state="PASS", repeat=repeat, completed_modes=repeat * len(MODES),
                         all_rank_runtime_checks=True, exact_input_output_tokens=True,
                         parity_and_writeback_byte_exact=True, real_prefill_retry=True))


def verify_study(root, smoke_root):
    """Independent verdict; original manifest/acceptance files are never written."""
    root, smoke_root = Path(root).resolve(), Path(smoke_root).resolve()
    report = dict(
        schema="kv-ec-study-reaudit-v1", state="INCONCLUSIVE",
        study=str(root), smoke_evidence=str(smoke_root),
        original_manifest_state=None, original_manifest_reason=None,
        cpu_validation=None, gpu_trial_validation=None,
        source_equivalence_proof=None, config_compatibility=None,
        reconstructed_summary=None, original_artifact_hashes={},
        auditor_source_sha256={name: sha256_file(Path(__file__).with_name(name))
                               for name in runtime_sources()},
        execution="read_only_saved_evidence_no_server_no_GPU")
    before = {}
    try:
        before = {**artifact_hashes(root), **artifact_hashes(smoke_root)}
        report["original_artifact_hashes"] = before
        manifest = json.loads((root / "trials.json").read_text())
        report.update(original_manifest_state=manifest["state"],
                      original_manifest_reason=manifest.get("reason"))
        current = validate_completed(root, expected_repeat=3, allow_terminal_config_failure=True)
        smoke = validate_completed(smoke_root, expected_repeat=1)
        report["cpu_validation"] = dict(study=current["cpu"], smoke=smoke["cpu"])
        report["gpu_trial_validation"] = current["gpu"]
        require(not current["summary"]["capture_profile"]
                and not smoke["summary"]["capture_profile"], "profiled_study_cannot_admit_performance")
        proof = compatible_config(
            smoke["summary"]["config"], current["summary"]["config"],
            before_recorded=smoke["manifest"]["source_sha256"],
            after_recorded=current["manifest"]["source_sha256"],
            before_parameters=smoke["manifest"].get("run_parameters"),
            after_parameters=current["manifest"].get("run_parameters"))
        require(current["effective_parameters"] == smoke["effective_parameters"],
                "effective_experiment_parameters_changed_since_smoke")
        report.update(
            state="PASS", source_equivalence_proof=proof,
            config_compatibility=dict(state="PASS",
                                      effective_parameters=current["effective_parameters"]),
            reconstructed_summary=current["summary"],
            acceptance_origin="completed_run_offline_reaudit_PASS")
    except Inconclusive as exc:
        report.update(state="INCONCLUSIVE", reason=str(exc))
    except (FileNotFoundError, PermissionError) as exc:
        report.update(state="INCONCLUSIVE", reason=f"missing_evidence:{exc}")
    except (ValueError, KeyError, TypeError, AssertionError) as exc:
        report.update(state="FAIL", reason=f"{type(exc).__name__}: {exc}")
    if before:
        after = {**artifact_hashes(root), **artifact_hashes(smoke_root)}
        if before != after:
            report.update(state="FAIL", reason="evidence_changed_during_audit")
    return report


def write_report(report, output):
    output = Path(output).resolve()
    for key in ("study", "smoke_evidence"):
        root = Path(report[key]).resolve()
        require(output != root and root not in output.parents, "report_must_be_outside_original_evidence")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
