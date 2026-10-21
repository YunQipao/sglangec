"""Independent A/B/C/D studies at rank0/256 lost tokens, 768 prompt, 16 output.

CPU first, one single-run smoke before a three-repeat study. No busy GPU
takeover, no background GPU polling, no retries after failed correctness.
"""

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
import tempfile

from .accept_pause import MEASURED_SOURCES, check_experiment_evidence, verify_run
from .m4_analysis import audit_case, load, require
from .m4_trials import run_logged, statistics
from .request_debug import compare

MODES = ("baseline", "protection_only", "online_recovery", "prefill_retry")
PERFORMANCE = {
    "baseline": "performance-baseline.json",
    "protection_only": "performance-protection-only.json",
    "online_recovery": "performance-online.json",
    "prefill_retry": "performance-prefill-retry.json",
}
COMMON_CONFIG = (
    "lost_rank", "recover_tokens",
    "model_identifier", "model_sha256", "source_commit", "core_diff_sha256",
    "gpu_uuids", "tp", "dtype", "layout", "page_size", "overlap", "prefix_cache",
    "input_tokens", "output_tokens", "input_ids_sha256", "sampling",
    "torch_version", "cuda_version", "experiment_source_sha256", "parity_test_oracle",
)


def audit_mode(root, mode):
    root = Path(root)
    acceptance = load(root / "acceptance.json")
    require(acceptance.get("state") == "PASS", f"failed_{mode}_acceptance")
    perf = load(root / PERFORMANCE[mode])
    if mode == "baseline":
        response = load(root / "baseline-response.json")
        require(acceptance.get("mode") == mode and len(response["output_ids"]) == 16,
                "baseline_incomplete")
        require(perf["model_identifier"] == response["model_identifier"]
                and perf["sampling"] == response["sampling"]
                and perf["input_ids_sha256"] == response["input_ids_sha256"],
                "baseline_config_mismatch")
        for key in ("ttft_ms", "tpot_ms", "request_total_ms"):
            require(perf[key] == response["client_metrics_ms"][key], "baseline_timing_mismatch")
        return perf, response
    compare([root / "baseline-response.json", root / "pause-response.json"])
    response = load(root / "pause-response.json")
    require(perf["config"]["parity_test_oracle"] is False, "study_requires_production_parity")
    if mode == "online_recovery":
        audit_case(root)
        ranks = [load(root / "ranks" / f"pause-rank-{r}.json") for r in range(4)]
        lost = perf["config"]["lost_rank"]
        fault = min(c["fault_monotonic"] for c in ranks[lost]["online"]["chunks"])
        require(perf["fault_to_first_decode_ms"] ==
                max((r["decode_observed_monotonic"] - fault) * 1000 for r in ranks),
                "C_all_rank_fault_to_decode_mismatch")
    else:
        require(acceptance.get("EXPERIMENT_PASS") is True and acceptance.get("mode") == mode,
                "missing_B_D_acceptance")
        require(verify_run(root, root / "baseline-response.json", root / "pause-response.json",
                           retry=mode == "prefill_retry")["state"] == "PASS", "B_D_pause_audit_failed")
        identity = load(root / "identity.json")
        checkpoints = load(root / "checkpoints.json") if (root / "checkpoints.json").exists() else []
        records = check_experiment_evidence(root / "ranks", identity, checkpoints, response, mode)
        require(perf["rank_metrics"] == [r[mode] for r in records], "rank_metrics_mismatch")
        keys = (("protection_blocking_ms", "parity_commit_ms") if mode == "protection_only" else
                ("fault_to_rebuilt_state_ms", "fault_to_first_decode_ms", "prefill_forward_ms"))
        for key in keys:
            require(perf[key] == max(r[mode][key] for r in records), "performance_timing_mismatch")
        for name in ("baseline-server.log", "pause-server.log"):
            log = (root / name).read_text(errors="replace")
            require(all(f"TP{r}]" in log for r in range(4)) and "Load weight end." in log
                    and "Qwen3ForCausalLM" in log and '"POST /generate HTTP/1.1" 200' in log,
                    "missing_real_TP4_execution")
            require(not any(s in log for s in (
                "Scheduler hit an exception", "Traceback (most recent call last)",
                "SlotLeaseRejected", "protected_slot_free", "registry_closed")),
                "failed_scheduler_or_lease")
        require(all(check["actual"] == check["expected"] and
                    type(check["actual"]) is type(check["expected"])
                    for r in records for check in r["runtime_checks"].values()),
                "failed_runtime_safety_checks")
        require(all(r.get("runtime_checks") and r["model_identity"] ==
                    {k: perf["config"][k] for k in r["model_identity"]} for r in records),
                "missing_runtime_checks_or_model_source_mismatch")
    config = perf["config"]
    require(config["input_ids_sha256"] == response["input_ids_sha256"]
            and config["sampling"] == response["sampling"]
            and config["model_identifier"] == response["model_identifier"],
            "performance_response_mismatch")
    for key in ("ttft_ms", "tpot_ms", "request_total_ms"):
        require(perf[key] == response["client_metrics_ms"][key], "client_timing_mismatch")
    require(all(p.suffix in (".json", ".log") for p in root.rglob("*") if p.is_file()),
            "unexpected_binary_artifact")
    return perf, response


def summarize(root):
    manifest = load(Path(root) / "trials.json")
    groups = {mode: [] for mode in MODES}
    config = None
    transactions, baseline_inodes = set(), set()
    for trial in manifest["trials"]:
        require(trial.get("state") == "PASS", "incomplete_trial")
        modes = {}
        for mode in MODES:
            path = Path(trial["run"]) / mode
            perf, response = audit_mode(path, mode)
            acceptance = load(path / "acceptance.json")
            require(acceptance.get("capture_profile", False) ==
                    (manifest.get("capture_profile", False) and mode != "baseline"),
                    "capture_profile_mode_mismatch")
            tx = acceptance["transaction_id"]
            baseline = (path / "baseline-response.json").stat()
            inode = (baseline.st_dev, baseline.st_ino)
            require(tx not in transactions and inode not in baseline_inodes, "duplicate_trial")
            transactions.add(tx)
            baseline_inodes.add(inode)
            modes[mode] = response
            if mode == "baseline":
                baseline_uuids = perf["gpu_uuids"]
            else:
                current = {k: perf["config"][k] for k in COMMON_CONFIG}
                require(config is None or config == current, "mode_or_repeat_configuration_drift")
                config = current
                require(baseline_uuids == config["gpu_uuids"], "baseline_GPU_UUID_mismatch")
            groups[mode].append(perf)
        for mode in MODES[1:]:
            for key in ("model_identifier", "input_ids_sha256", "input_token_count",
                        "sampling", "output_ids", "finish_reason"):
                require(modes[mode][key] == modes["baseline"][key], f"cross_mode_mismatch:{key}")
    require(config is not None, "no_complete_ABCD_trial")
    require(all(config.get(k) == v for k, v in dict(
        tp=4, dtype="bf16", layout="nhd", page_size=1, overlap=False, prefix_cache=False,
        input_tokens=768, output_tokens=16, lost_rank=0, recover_tokens=256).items()), "unsupported_study_config")
    return dict(
        state="PASS", measurement_revision="ABCD-production-parity-v1",
        capture_profile=manifest.get("capture_profile", False),
        repeat=len(groups["baseline"]), config=config,
        modes={mode: {k: statistics([row[k] for row in rows])
                      for k, v in rows[0].items()
                      if k.endswith("_ms") and type(v) in (float, int)}
               for mode, rows in groups.items()},
        note="fresh servers, no warmup; B/C no recovery-based parity oracle; strict byte validation enabled; "
             "no speedup against legacy test-oracle Pilot; n=3 has no stable P95")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--shadow-evidence", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 3), default=1)
    parser.add_argument("--smoke-evidence", type=Path,
                        help="audited single-run complete A/B/C/D required for repeat=3")
    parser.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    parser.add_argument("--baseline-port", type=int, default=30359)
    parser.add_argument("--pause-port", type=int, default=30360)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--pause-timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-profile", action="store_true",
                        help="M5 capture traces; diagnostic latencies are not production measurements")
    args = parser.parse_args()
    if args.capture_profile and args.repeat != 1:
        parser.error("--capture-profile supports only a single diagnostic smoke")
    if args.repeat == 3 and args.smoke_evidence is None:
        parser.error("repeat=3 requires complete --smoke-evidence; never promote failed B/D")
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="kv-ec-abcd-"))
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    manifest = dict(state="RUNNING", run=str(root), repeat=args.repeat, trials=[],
                    capture_profile=args.capture_profile,
                    modes=MODES, lost_rank=0, recover_tokens=256,
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in MEASURED_SOURCES})
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def save():
        (root / "trials.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    try:
        if args.repeat == 3:
            require(load(args.smoke_evidence / "trials.json")["state"] == "PASS", "smoke_not_pass")
            smoke = summarize(args.smoke_evidence)
            require(smoke["repeat"] == 1, "requires_single_run_smoke")
            require(not smoke["capture_profile"], "profiled_smoke_not_performance_admission")
            require(load(args.smoke_evidence / "trials.json")["source_sha256"] ==
                    manifest["source_sha256"], "source_changed_since_smoke")
        code = run_logged(
            [sys.executable, "-m", "unittest", "discover", "-s", "experiments/kv_ec", "-p", "test_*.py", "-v"],
            root / "cpu-regression.log", cwd=repo, env=env, label="CPU")
        manifest["cpu_returncode"] = code
        require(code == 0, "CPU_regression_failed_no_GPU_trial_launched")
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
                    if args.capture_profile:
                        command.append("--capture-profile")
                print(f"START repeat={repeat} mode={mode} run={run / mode}", flush=True)
                code = run_logged(command, run / f"{mode}.log", cwd=repo, env=env, label=mode)
                require(code == 0, f"mode_failed:{mode}; no later GPU experiment launched")
                audit_mode(run / mode, mode)
                trial["completed_modes"].append(mode)
                save()
            trial["state"] = "PASS"
            save()
        result = summarize(root)
        if args.repeat == 3:
            require(result["config"] == smoke["config"], "configuration_changed_since_smoke")
        (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest["state"] = "PASS"
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(state="FAIL", reason=f"{type(exc).__name__}: {exc}")
        for trial in manifest["trials"]:
            if trial["state"] == "RUNNING":
                trial["state"] = "FAIL"
    finally:
        save()
    print(f"ABCD_PASS={str(manifest['state'] == 'PASS').lower()} Evidence: {root}", flush=True)
    if manifest.get("reason"):
        print(manifest["reason"], flush=True)
    return 0 if manifest["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
