"""Audit unprofiled GPU trials and gate repeats on preliminary B/C results."""

import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from .m4_abcd import COMMON_CONFIG, summarize
from .m4_analysis import load, require

METRICS = {
    "A_request_total_ms": ("baseline", "request_total_ms"),
    "B_request_total_ms": ("protection_only", "request_total_ms"),
    "B_protection_blocking_ms": ("protection_only", "protection_blocking_ms"),
    "C_fault_to_decode_ms": ("online_recovery", "fault_to_first_decode_ms"),
    "C_request_total_ms": ("online_recovery", "request_total_ms"),
    "D_fault_to_decode_ms": ("prefill_retry", "fault_to_first_decode_ms"),
}


def compare_summaries(baseline, candidate, *, min_b_improvement_pct=1., max_c_regression_pct=5.):
    require(math.isfinite(min_b_improvement_pct) and 0 <= min_b_improvement_pct < 100,
            "invalid_B_threshold")
    require(math.isfinite(max_c_regression_pct) and 0 <= max_c_regression_pct <= 100,
            "invalid_C_threshold")
    for row in (baseline, candidate):
        require(row["state"] == "PASS" and not row.get("capture_profile", False),
                "comparison_requires_unprofiled_PASS")
    # Experimental source hashes and commits are intentionally different.
    # Model, hardware, sampling, layout, and dirty inference-source hash are not.
    for key in COMMON_CONFIG:
        if key not in ("source_commit", "experiment_source_sha256"):
            require(baseline["config"][key] == candidate["config"][key],
                    f"incomparable_config:{key}")
    metrics = {}
    for name, (mode, key) in METRICS.items():
        values = []
        for summary in (baseline, candidate):
            samples = summary["modes"][mode][key]["samples_ms"]
            require(len(samples) == summary["repeat"] and all(
                type(x) in (float, int) and math.isfinite(x) and x > 0 for x in samples),
                f"invalid_metric:{name}")
            values.append(statistics.median(samples))
        before, after = values
        metrics[name] = dict(baseline_ms=before, candidate_ms=after,
                             reduction_pct=(before - after) / before * 100)
    before = metrics["B_request_total_ms"]["baseline_ms"] - metrics["A_request_total_ms"]["baseline_ms"]
    after = metrics["B_request_total_ms"]["candidate_ms"] - metrics["A_request_total_ms"]["candidate_ms"]
    require(before > 0, "nonpositive_baseline_B_minus_A")
    metrics["B_minus_A_request_ms"] = dict(
        baseline_ms=before, candidate_ms=after, reduction_pct=(before - after) / before * 100)
    gates = dict(
        B_blocking_reduced=metrics["B_protection_blocking_ms"]["reduction_pct"] > min_b_improvement_pct,
        B_extra_request_reduced=metrics["B_minus_A_request_ms"]["reduction_pct"] > min_b_improvement_pct,
        C_fault_to_decode_within_limit=metrics["C_fault_to_decode_ms"]["reduction_pct"] >= -max_c_regression_pct,
        C_request_within_limit=metrics["C_request_total_ms"]["reduction_pct"] >= -max_c_regression_pct)
    return dict(
        state="PASS", baseline_repeat=baseline["repeat"], candidate_repeat=candidate["repeat"],
        metrics=metrics, gates=gates, preliminary_positive=all(gates.values()),
        thresholds=dict(min_B_improvement_pct=min_b_improvement_pct,
                        max_C_regression_pct=max_c_regression_pct),
        gpu_peak_memory_comparison=None, cpu_thread_time_comparison=None,
        note="Saved real GPU evidence; not a new experiment. B-A is the difference of medians. "
             "A single smoke only gates repeats; n=3 is still a pilot, not proof of efficacy. "
             "GPU peak memory and unprofiled CPU thread costs are not recorded in M4.")


def audited_summary(root):
    manifest = load(root / "trials.json")
    require(manifest["state"] == "PASS" and manifest.get("cpu_returncode") == 0,
            "requires_completed_GPU_run_and_CPU_regression")
    with redirect_stdout(io.StringIO()):
        summary = summarize(root)
    return summary, [load(Path(trial["run"]) / "baseline" / "baseline-response.json")
                     for trial in manifest["trials"]]


def compare_runs(baseline, candidate, **thresholds):
    before, old_responses = audited_summary(baseline)
    after, new_responses = audited_summary(candidate)
    for response in old_responses + new_responses:
        for key in ("input_ids_sha256", "sampling", "output_ids", "finish_reason", "model_identifier"):
            require(response[key] == old_responses[0][key], f"cross_revision_response_mismatch:{key}")
    repo = Path(__file__).resolve().parents[2]
    commits = [s["config"]["source_commit"] for s in (before, after)]
    require(all(re.fullmatch("[0-9a-f]{40}", c) for c in commits), "invalid_source_commit")
    difference = subprocess.check_output(["git", "diff", *commits, "--", "python"], cwd=repo)
    require(not difference, "inference_source_changed_since_baseline")
    result = compare_summaries(before, after, **thresholds)
    result.update(baseline=str(baseline), candidate=str(candidate),
                  baseline_source=before["config"]["experiment_source_sha256"],
                  candidate_source=after["config"]["experiment_source_sha256"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/kv-ec-abcd-PYvwwJ/three"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="exclusive new JSON")
    parser.add_argument("--require-positive", action="store_true", help="nonzero exit if B/C gates fail")
    parser.add_argument("--expect-repeat", type=int, choices=(1, 3))
    parser.add_argument("--min-b-improvement-pct", type=float, default=1.)
    parser.add_argument("--max-c-regression-pct", type=float, default=5.)
    args = parser.parse_args()
    try:
        result = compare_runs(args.baseline.resolve(), args.candidate.resolve(),
                              min_b_improvement_pct=args.min_b_improvement_pct,
                              max_c_regression_pct=args.max_c_regression_pct)
        require(args.expect_repeat is None or result["candidate_repeat"] == args.expect_repeat,
                "unexpected_candidate_repeat_count")
    except (Exception, KeyboardInterrupt) as exc:
        result = dict(state="FAIL", reason=f"{type(exc).__name__}: {exc}")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    passed = result["state"] == "PASS" and (
        not args.require_positive or result["preliminary_positive"])
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
