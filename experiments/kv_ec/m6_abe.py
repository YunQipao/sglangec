"""M6 A/B/E CPU-first studies. A delayed correctness smoke admits performance.

No M3 matrix reruns. Never compares delayed/profiled smoke latency with normal
performance, never promotes a failed mode, and never changes historical M5 data.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import traceback

from .accept_pause import MEASURED_SOURCES, verify_run
from .m4_abcd import audit_mode, preflight_config
from .m4_analysis import load, require
from .m4_trials import run_logged
from .m6_acceptance import async_performance, check_async_evidence
from .request_debug import compare
from .study_audit import artifact_hashes, validate_cpu
from .study_validation import (
    Inconclusive, check_live_sources, compatible_config, historical_sources, runtime_sources, sha256_file,
)


MODES = ("baseline", "protection_only", "async_protection_only")
RECOVERABLE_SUMMARY_REASON = "ValueError:too many values to unpack (expected 2)"
ASYNC_CONFIG_KEYS = (
    "async_queue_capacity", "async_snapshot_budget_bytes", "async_commit_delay_s",
    "async_pending_policy",
)


def base_config(config):
    return {k: v for k, v in config.items() if k not in ASYNC_CONFIG_KEYS}

def compare_abe(responses):
    """Anchor both protected modes to A using the unchanged strict pair oracle."""
    require(set(responses) == set(MODES), "M6_missing_response_mode")
    for mode in MODES[1:]:
        compare([responses["baseline"], responses[mode]])


def synchronous_unprotected_window(root):
    """Use the same completed-Prefill origin as E, without editing B evidence."""
    windows = []
    for rank in range(4):
        record = load(Path(root) / "ranks" / f"pause-rank-{rank}.json")
        prefill = record["prefill_completed_monotonic"]
        committed = record["protection"]["committed_monotonic"]
        require(all(type(t) in (int, float) and math.isfinite(t) for t in (prefill, committed))
                and prefill <= committed <= record["paused_observation_monotonic"],
                "B_unprotected_window_clock_invalid")
        windows.append((committed - prefill) * 1000)
    return max(windows)


def audit_e(root, *, require_overlap):
    root = Path(root)
    acceptance = load(root / "acceptance.json")
    require(acceptance.get("state") == "PASS"
            and acceptance.get("GPU_VALIDATED") is True
            and acceptance.get("EXPERIMENT_PASS") is True
            and acceptance.get("mode") == "async_protection_only"
            and not acceptance.get("capture_profile"), "E_acceptance_not_real_unprofiled_PASS")
    require(verify_run(root, root / "baseline-response.json",
                       root / "pause-response.json")["state"] == "PASS", "E_pause_audit_failed")
    response = load(root / "pause-response.json")
    identity = load(root / "identity.json")
    checkpoints = load(root / "checkpoints.json")
    records = check_async_evidence(root / "ranks", identity, checkpoints, response,
                                   require_overlap=require_overlap)
    export = load(root / "parity-export-verification.json")
    require(export == dict(state="PASS", parity_bytes=checkpoints[0]["parity_bytes"],
                           parity_sha256=checkpoints[0]["parity_sha256"],
                           snapshot_id=checkpoints[0]["snapshot_id"]), "E_export_missing_or_mismatched")
    perf = load(root / "performance-async-protection-only.json")
    require(perf == async_performance(response, records, config=perf["config"]),
            "E_performance_not_reconstructible")
    require(perf["config"]["async_commit_delay_s"] == acceptance["async_commit_delay_s"],
            "E_diagnostic_delay_mismatch")
    require(perf["config"]["async_queue_capacity"] == 1
            and perf["config"]["async_snapshot_budget_bytes"] == 9437184
            and perf["config"]["async_pending_policy"] == "bounded_off_path_drain"
            and perf["config"]["parity_test_oracle"] is False,
            "E_unsupported_protection_configuration")
    for rank in range(4):
        require(records[rank]["model_identity"] ==
                {k: perf["config"][k] for k in records[rank]["model_identity"]},
                "E_runtime_model_source_mismatch")
    for name in ("baseline-server.log", "pause-server.log"):
        text = (root / name).read_text(errors="replace")
        require(all(f"TP{r}]" in text for r in range(4)) and "Load weight end." in text
                and "Qwen3ForCausalLM" in text and '"POST /generate HTTP/1.1" 200' in text,
                "E_missing_real_TP4_execution")
    return perf, response


def summarize(root, *, allow_running=False, reaudit_summary_failure=False):
    root = Path(root)
    manifest = load(root / "trials.json")
    recoverable = (reaudit_summary_failure and manifest["state"] == "FAIL"
                   and manifest.get("reason") == RECOVERABLE_SUMMARY_REASON)
    require(manifest["state"] == "PASS" or allow_running and manifest["state"] == "RUNNING"
            or recoverable,
            "M6_manifest_not_PASS")
    require(manifest["repeat"] in (1, 3)
            and len(manifest["trials"]) == manifest["repeat"], "M6_missing_repeats")
    cpu = validate_cpu(root, manifest)
    text = (root / "cpu-regression.log").read_text()
    for name in (
        "test_actual_E_gate_releases_after_snapshot_then_decodes_before_commit",
        "test_postcommit_stale_epoch_checkpoint_row_slots_and_request_rejected",
        "test_cancel_ready_inflight_late_commit_and_after_commit_invalidate",
        "test_manager_error_and_timeout_never_become_recoverable",
    ):
        require(re.search(r"^" + re.escape(name) + r" \([^\n]+\) \.\.\. ok$",
                          text, re.MULTILINE), "M6_critical_CPU_missing_or_skipped:" + name)
    groups = {mode: [] for mode in MODES}
    transactions, inodes = set(), set()
    config = None
    raw = []
    for trial in manifest["trials"]:
        require(trial["state"] == "PASS" and trial["completed_modes"] == list(MODES),
                "M6_trial_incomplete")
        responses = {}
        for mode in MODES:
            path = Path(trial["run"]) / mode
            perf, response = (audit_e(path, require_overlap=manifest["diagnostic_overlap_smoke"])
                              if mode == "async_protection_only" else audit_mode(path, mode))
            if mode == "protection_only":
                perf = dict(perf, unprotected_window_ms=synchronous_unprotected_window(path))
            saved = load(path / "acceptance.json")
            require(not saved.get("capture_profile"), "M6_profiled_not_performance")
            if mode != "async_protection_only":
                require(saved.get("async_commit_delay_s", 0) == 0, "delay_in_non_E_mode")
            tx = saved["transaction_id"]
            stat = (path / "baseline-response.json").stat()
            inode = (stat.st_dev, stat.st_ino)
            require(tx not in transactions and inode not in inodes, "M6_reused_trial")
            transactions.add(tx)
            inodes.add(inode)
            if mode == "baseline":
                uuids = perf["gpu_uuids"]
            else:
                current = base_config(perf["config"])
                require(config is None or current == config, "M6_config_drift")
                config = current
                require(config["experiment_source_sha256"] == manifest["source_sha256"],
                        "M6_launch_runtime_fingerprint_drift")
                require(current["gpu_uuids"] == uuids, "M6_GPU_drift")
                require(current["input_ids_sha256"] == response["input_ids_sha256"]
                        and current["sampling"] == response["sampling"]
                        and current["model_identifier"] == response["model_identifier"],
                        "M6_config_response_mismatch")
            if mode == "async_protection_only":
                require(perf["config"]["async_commit_delay_s"] == manifest["async_commit_delay_s"],
                        "M6_study_delay_drift")
            groups[mode].append(perf)
            responses[mode] = path / ("baseline-response.json" if mode == "baseline" else "pause-response.json")
        compare_abe(responses)
        a, b, e = [groups[mode][-1] for mode in MODES]
        raw.append(dict(
            repeat=trial["repeat"], B_minus_A_request_ms=b["request_total_ms"] - a["request_total_ms"],
            E_minus_A_request_ms=e["request_total_ms"] - a["request_total_ms"],
            E_minus_B_request_ms=e["request_total_ms"] - b["request_total_ms"],
            E_minus_B_foreground_ms=e["foreground_protection_blocking_ms"] - b["protection_blocking_ms"],
            E_minus_B_unprotected_window_ms=e["unprotected_window_ms"] - b["unprotected_window_ms"]))
    metrics = {}
    for mode, rows in groups.items():
        metrics[mode] = {
            k: dict(samples=[row[k] for row in rows],
                    median=statistics.median(row[k] for row in rows))
            for k, value in rows[0].items()
            if type(value) in (int, float) and (k.endswith("_ms") or k.endswith("_bytes"))}
    source_proof = historical_sources(config, manifest["source_sha256"])
    return dict(
        state="PASS", revision="M6-blocking-D2H-async-CPU-v1", repeat=manifest["repeat"],
        diagnostic_overlap_smoke=manifest["diagnostic_overlap_smoke"], config=config,
        source_proof=source_proof, cpu_validation=cpu,
        modes=metrics, paired_deltas=raw,
        performance_verdict="NOT_EVALUATED_DIAGNOSTIC" if manifest["diagnostic_overlap_smoke"]
                            else "DESCRIPTIVE_ONLY_NO_PREDECLARED_M6_THRESHOLD",
        memory_limitations="A/B process and device memory not instrumented; E reports process lifetime "
                           "CPU high-water and CUDA allocator high-water at snapshot, not whole-request GPU peak",
        note="Delayed correctness smoke latencies must not be compared with undelayed performance. "
             "Commit observation bounds the unprotected window conservatively; n=3 is a pilot.")

def reaudit_study(root):
    """Independent metadata-only report; never repairs a historical manifest."""
    root = Path(root).resolve()
    report = dict(
        state="FAIL", study=str(root), original_manifest_state=None, original_manifest_reason=None,
        cpu_validation=None, gpu_trial_validation=None, source_equivalence_proof=None,
        reconstructed_summary=None, original_artifact_hashes=None, reason=None,
        auditor_source_sha256={n: sha256_file(Path(__file__).with_name(n)) for n in runtime_sources()})
    before = None
    try:
        before = artifact_hashes(root)
        report["original_artifact_hashes"] = before
        manifest = load(root / "trials.json")
        report.update(original_manifest_state=manifest.get("state"),
                      original_manifest_reason=manifest.get("reason"))
        result = summarize(root, reaudit_summary_failure=True)
        report.update(
            cpu_validation=result["cpu_validation"], reconstructed_summary=result,
            source_equivalence_proof=result["source_proof"],
            gpu_trial_validation=dict(
                state="PASS", repeat=result["repeat"], modes=list(MODES),
                diagnostic_overlap_smoke=result["diagnostic_overlap_smoke"],
                checks="reexecuted A/B/E acceptance, leases/events, parity, rank timelines, exact A-B/A-E oracle"))
        report["state"] = "PASS"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(state="INCONCLUSIVE" if isinstance(exc, Inconclusive) else "FAIL",
                      reason=f"{type(exc).__name__}:{exc}", traceback=traceback.format_exc())
    finally:
        if before is not None:
            try:
                report["original_artifacts_unchanged"] = before == artifact_hashes(root)
                if not report["original_artifacts_unchanged"]:
                    report.update(state="FAIL", reason="original_artifacts_changed_during_reaudit")
            except Exception as exc:
                report.update(state="FAIL", reason=f"artifact_recheck_failed:{exc}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus")
    parser.add_argument("--shadow-evidence", type=Path)
    parser.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    parser.add_argument("--baseline-port", type=int, default=30359)
    parser.add_argument("--pause-port", type=int, default=30360)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--pause-timeout", type=float, default=120)
    parser.add_argument("--repeat", type=int, choices=(1, 3), default=1)
    parser.add_argument("--diagnostic-overlap-smoke", action="store_true")
    parser.add_argument("--async-commit-delay", type=float, default=0)
    parser.add_argument("--smoke-evidence", type=Path)
    parser.add_argument("--verify-study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.verify_study:
        require(not args.output.resolve().is_relative_to(args.verify_study.resolve()),
                "offline_report_must_be_outside_original_study")
        result = reaudit_study(args.verify_study)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(f"M6_OFFLINE={result['state']} report={args.output}")
        return 0 if result["state"] == "PASS" else 1
    if not args.gpus or not args.shadow_evidence:
        parser.error("live studies require --gpus and --shadow-evidence")
    if args.diagnostic_overlap_smoke:
        if args.repeat != 1 or not 0 < args.async_commit_delay < args.pause_timeout:
            parser.error("diagnostic smoke requires repeat=1 and 0 < delay < pause-timeout")
    elif args.async_commit_delay != 0:
        parser.error("artificial delay is only permitted in diagnostic smoke")
    if not args.diagnostic_overlap_smoke and not args.smoke_evidence:
        parser.error("undelayed performance requires a PASS diagnostic overlap --smoke-evidence")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
              for n in MEASURED_SOURCES}
    parameters = {k: getattr(args, k) for k in (
        "gpus", "model", "baseline_port", "pause_port", "startup_timeout", "pause_timeout")}
    manifest = dict(
        state="RUNNING", repeat=args.repeat, trials=[], source_sha256=source,
        run_parameters=parameters, diagnostic_overlap_smoke=args.diagnostic_overlap_smoke,
        async_commit_delay_s=args.async_commit_delay)
    def save():
        (root / "trials.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        check_live_sources()
        if args.smoke_evidence:
            smoke_manifest = load(args.smoke_evidence / "trials.json")
            require(smoke_manifest["state"] == "PASS", "M6_smoke_manifest_not_PASS")
            smoke = summarize(args.smoke_evidence)
            require(smoke["repeat"] == 1 and smoke["diagnostic_overlap_smoke"],
                    "M6_requires_single_diagnostic_overlap_smoke")
            # E delay is intentionally not in base_config: smoke proves the
            # overlap, never supplies performance reference samples.
            manifest["smoke_source_equivalence_proof"] = compatible_config(
                smoke["config"], preflight_config(args, smoke),
                before_recorded=smoke_manifest["source_sha256"], after_recorded=source,
                before_parameters=smoke_manifest["run_parameters"], after_parameters=parameters)
        manifest["cpu_returncode"] = run_logged(
            [sys.executable, "-m", "unittest", "discover", "-s", "experiments/kv_ec", "-p", "test_*.py", "-v"],
            root / "cpu-regression.log", cwd=repo, env=env, label="CPU")
        require(manifest["cpu_returncode"] == 0, "CPU_failed_no_GPU_launched")
        for repeat in range(1, args.repeat + 1):
            run = root / f"repeat{repeat}"
            run.mkdir()
            trial = dict(state="RUNNING", run=str(run), repeat=repeat, completed_modes=[])
            manifest["trials"].append(trial)
            save()
            for mode in MODES:
                command = [
                    sys.executable, "-u", "-m", "experiments.kv_ec.accept_pause",
                    "--mode", mode, "--gpus", args.gpus, "--model", args.model,
                    "--baseline-port", str(args.baseline_port), "--pause-port", str(args.pause_port),
                    "--startup-timeout", str(args.startup_timeout),
                    "--pause-timeout", str(args.pause_timeout), "--output", str(run / mode)]
                if mode != "baseline":
                    command += ["--shadow-evidence", str(args.shadow_evidence.resolve())]
                if mode == "async_protection_only" and args.diagnostic_overlap_smoke:
                    command += ["--async-commit-delay", str(args.async_commit_delay), "--require-async-overlap"]
                print(f"START repeat={repeat} mode={mode} run={run / mode}", flush=True)
                code = run_logged(command, run / f"{mode}.log", cwd=repo, env=env, label=mode)
                require(code == 0, f"M6_mode_failed:{mode};no_later_GPU_launched")
                if mode == "async_protection_only":
                    audit_e(run / mode, require_overlap=args.diagnostic_overlap_smoke)
                else:
                    audit_mode(run / mode, mode)
                trial["completed_modes"].append(mode)
                save()
            trial["state"] = "PASS"
            save()
        result = summarize(root, allow_running=True)
        check_live_sources()
        if args.smoke_evidence:
            manifest["final_source_equivalence_proof"] = compatible_config(
                smoke["config"], result["config"],
                before_recorded=smoke_manifest["source_sha256"], after_recorded=source,
                before_parameters=smoke_manifest["run_parameters"], after_parameters=parameters)
        (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest["state"] = "PASS"
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(state="FAIL", reason=f"{type(exc).__name__}:{exc}",
                        traceback=traceback.format_exc())
    finally:
        save()
    print(f"M6_ABE_PASS={manifest['state'] == 'PASS'} Evidence: {root}", flush=True)
    if manifest.get("reason"):
        print(manifest["reason"], flush=True)
    return 0 if manifest["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
