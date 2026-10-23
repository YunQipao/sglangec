"""Independent paired A/C pilot trials; B and real D are explicitly unavailable."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from .m4_analysis import audit_case, audit_matrix, load, require


REPRESENTATIVE_CASES = ((0, 256), (1, 256), (0, 768), (3, 768))


def compatible_config(actual, expected):
    """Allow experiments-only commits while retaining both recorded revisions."""
    require({k: v for k, v in actual.items() if k != "source_commit"} ==
            {k: v for k, v in expected.items() if k != "source_commit"},
            "measurement_configuration_drift")
    if actual.get("source_commit") != expected.get("source_commit"):
        repo = Path(__file__).resolve().parents[2]
        difference = subprocess.check_output(
            ["git", "diff", expected["source_commit"], actual["source_commit"], "--", "python"], cwd=repo)
        require(not difference, "inference_source_changed")


def statistics(values):
    require(bool(values) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "invalid_measurement")
    result = dict(n=len(values), samples_ms=values, min_ms=min(values), max_ms=max(values))
    if len(values) >= 10:
        result.update(status="descriptive_statistics", median_ms=float(np.median(values)),
                      p95_ms=float(np.percentile(values, 95)),
                      percentile_method="numpy linear interpolation; n=10 P95 is tail-uncertain")
    else:
        result["status"] = "single_observation" if len(values) == 1 else "pilot_only_no_stable_quantiles"
    return result


def summarize_trials(paths):
    groups, identities, baseline_files = {}, set(), set()
    rows = []
    for path in paths:
        row, audit = audit_case(path)
        tx = audit["transaction_id"]
        baseline = Path(audit["baseline_path"]).stat()
        inode = (baseline.st_dev, baseline.st_ino)
        require(tx not in identities and inode not in baseline_files, "duplicate_independent_trial")
        identities.add(tx)
        baseline_files.add(inode)
        key = (row["lost_rank"], row["recover_tokens"])
        group = groups.setdefault(key, dict(config=audit["config"], rows=[]))
        require(group["config"] == audit["config"], "trial_configuration_drift")
        group["rows"].append(row)
        rows.append(row)
    require(bool(groups), "no_accepted_trials")
    results = []
    for (rank, tokens), group in sorted(groups.items()):
        metrics = {}
        for key, value in group["rows"][0].items():
            if key.endswith("_ms") and type(value) in (int, float):
                metrics[key] = statistics([row[key] for row in group["rows"]])
        results.append(dict(lost_rank=rank, recover_tokens=tokens, n=len(group["rows"]),
                            config=group["config"], metrics=metrics))
    return dict(state="PASS", modes_measured=["A_baseline", "C_online_recovery"],
                modes_unavailable=["B_protection_only", "D_real_prefill_retry"],
                correctness_heavy=True, independent_trial_count=len(rows), groups=results, rows=rows,
                interpretation="Paired fresh-server A/C observations, not a complete A/B/C/D experiment. "
                               "Request total deltas are descriptive and include harness/validation cost.")


def validate_pilot(root):
    root = Path(root)
    manifest = load(root / "trials.json")
    require(manifest.get("state") == "PASS" and manifest.get("repeat") == 3, "pilot_not_complete")
    require(set(tuple(case) for case in manifest.get("cases", [])) == set(REPRESENTATIVE_CASES),
            "pilot_missing_representative_cases")
    paths = [Path(t["run"]) for t in manifest["trials"] if t["state"] == "PASS"]
    summary = summarize_trials(paths)
    require(len(summary["groups"]) == 4 and all(g["n"] == 3 for g in summary["groups"]),
            "pilot_requires_three_independent_runs_per_case")
    return summary


def run_logged(command, log, *, cwd, env, label):
    """Show heartbeat/log tail without touching any discovered GPU process."""
    with Path(log).open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            while True:
                try:
                    return process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    text = Path(log).read_text(errors="replace").splitlines()
                    last = text[-1][-240:] if text else "(no log line yet)"
                    print(f"RUNNING {label} elapsed_s={time.monotonic() - started:.0f} "
                          f"log={log}\n  {last}", flush=True)
        except BaseException:
            # Ask the owned acceptance process to execute its own finally
            # cleanup before resorting to killing its private process group.
            for sig, timeout in ((signal.SIGINT, 60), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="previous real 12/12 M3 matrix")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--shadow-evidence", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 3, 10), default=3)
    parser.add_argument("--pilot-evidence", type=Path,
                        help="complete three-run representative pilot required before repeat=10")
    parser.add_argument("--case", choices=("0:256", "1:256", "0:768", "3:768"),
                        help="one-case smoke; omitted runs all representative cases")
    parser.add_argument("--baseline-port", type=int, default=30359)
    parser.add_argument("--pause-port", type=int, default=30360)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--pause-timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat == 10 and not args.pilot_evidence:
        parser.error("--repeat 10 requires --pilot-evidence from a successful repeat=3 pilot")
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="kv-ec-m4-trials-"))
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    cases = (tuple(map(int, args.case.split(":"))),) if args.case else REPRESENTATIVE_CASES
    manifest = dict(state="RUNNING", repeat=args.repeat, cases=cases, trials=[], run=str(root),
                    modes=["A_baseline", "C_online_recovery"],
                    unimplemented=["B_protection_only", "D_real_prefill_retry"],
                    analysis_source_sha256={
                        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("m4_analysis.py", "m4_trials.py")})
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def save():
        (root / "trials.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    print(f"M4_RUN={root}\nB_protection_only=NOT_IMPLEMENTED D_prefill_retry=NOT_IMPLEMENTED", flush=True)
    try:
        matrix = audit_matrix(args.matrix)
        (root / "reference-audit.json").write_text(json.dumps(matrix, indent=2) + "\n")
        config = matrix["cases"][0]["config"]
        pilot = validate_pilot(args.pilot_evidence) if args.repeat == 10 else None
        if pilot:
            for group in pilot["groups"]:
                compatible_config(group["config"], next(
                    c["config"] for c in matrix["cases"] if
                    (c["lost_rank"], c["recover_tokens"]) == (group["lost_rank"], group["recover_tokens"])))
        print("M3_AUDIT_PASS=true; starting complete CPU regression", flush=True)
        code = run_logged([sys.executable, "-m", "unittest", "discover", "-s", "experiments/kv_ec",
                           "-p", "test_*.py", "-v"], root / "cpu-regression.log", cwd=repo, env=env,
                          label="CPU")
        require(code == 0, "CPU_regression_failed_no_GPU_trial_launched")
        manifest["cpu_returncode"] = code
        print("CPU_PASS=true", flush=True)
        # Repeat-major ordering interleaves representative configurations.
        # Every trial owns fresh baseline/pause server processes.
        for repeat in range(args.repeat):
            for rank, tokens in cases:
                run = root / f"repeat{repeat + 1}-rank{rank}-tokens{tokens}"
                trial = dict(state="RUNNING", repeat=repeat + 1, lost_rank=rank,
                             recover_tokens=tokens, run=str(run))
                manifest["trials"].append(trial)
                save()
                label = f"repeat={repeat + 1}/{args.repeat} rank={rank} tokens={tokens}"
                print(f"START {label}", flush=True)
                command = [sys.executable, "-u", "-m", "experiments.kv_ec.accept_pause",
                           "--online-restore", "--lost-rank", str(rank), "--recover-tokens", str(tokens),
                           "--gpus", args.gpus, "--shadow-evidence", str(args.shadow_evidence.resolve()),
                           "--model", config["model_identifier"], "--baseline-port", str(args.baseline_port),
                           "--pause-port", str(args.pause_port), "--startup-timeout", str(args.startup_timeout),
                           "--pause-timeout", str(args.pause_timeout), "--output", str(run)]
                trial["command"] = command
                code = run_logged(command, root / (run.name + ".log"), cwd=repo, env=env, label=label)
                trial["returncode"] = code
                if (run / "acceptance.json").exists():
                    trial["acceptance_reason"] = load(run / "acceptance.json").get("reason")
                require(code == 0, f"GPU_trial_failed:{run}:{trial.get('acceptance_reason', '')}; "
                                  "later_trials_not_run")
                row, audited = audit_case(run)
                reference = next(c["config"] for c in matrix["cases"]
                                 if (c["lost_rank"], c["recover_tokens"]) == (rank, tokens))
                # Analysis/runner changes do not affect the live measurements.
                # Any changed measured source/hardware/model must start a new study.
                compatible_config(audited["config"], reference)
                trial.update(state="PASS", row=row)
                save()
                print(f"PASS {label} fault_to_decode_ms={row['fault_to_decode_ms']:.2f}", flush=True)
        paths = [Path(t["run"]) for t in manifest["trials"]]
        summary = summarize_trials(paths)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["state"] = "PASS"
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(state="FAIL", reason=f"{type(exc).__name__}: {exc}")
        for trial in manifest["trials"]:
            if trial["state"] == "RUNNING":
                trial.update(state="FAIL", reason=manifest["reason"])
    finally:
        save()
    if manifest.get("reason"):
        print(f"reason={manifest['reason']}", flush=True)
    print(f"M4_AC_PASS={str(manifest['state'] == 'PASS').lower()}\nEvidence: {root}", flush=True)
    return 0 if manifest["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
