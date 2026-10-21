"""Audit unprofiled GPU trials and gate repeats on preliminary B/C results."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from .m4_analysis import load, require
from .study_audit import validate_completed, verify_study
from .study_validation import Inconclusive, compare_config_fields

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
    try:
        compare_config_fields(baseline["config"], candidate["config"])
    except ValueError as exc:
        raise ValueError(f"incomparable_config:{exc}") from exc
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
    root = Path(root)
    if root.is_file():
        saved = load(root)
        require(saved.get("schema") == "kv-ec-study-reaudit-v1"
                and saved.get("state") == "PASS", "requires_independent_reaudit_PASS")
        # A report is not an override: rerun every audit and compare its hashes
        # and reconstructed values, rejecting stale or hand-edited reports.
        fresh = verify_study(saved["study"], saved["smoke_evidence"])
        require(fresh["state"] == "PASS" and fresh == saved, "stale_or_modified_reaudit_report")
        summary = fresh["reconstructed_summary"]
        summary = dict(summary, acceptance_origin="completed_run_offline_reaudit_PASS",
                       original_manifest_state=fresh["original_manifest_state"],
                       source_validation=fresh["source_equivalence_proof"]["after"])
        root = Path(saved["study"])
        manifest = load(root / "trials.json")
    else:
        checked = validate_completed(root)
        summary = dict(checked["summary"], acceptance_origin="original_run_PASS",
                       original_manifest_state="PASS",
                       source_validation=checked["source"])
        manifest = checked["manifest"]
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
                  candidate_source=after["config"]["experiment_source_sha256"],
                  baseline_acceptance_origin=before["acceptance_origin"],
                  candidate_acceptance_origin=after["acceptance_origin"],
                  candidate_original_manifest_state=after["original_manifest_state"],
                  baseline_source_proof=before["source_validation"],
                  candidate_source_proof=after["source_validation"],
                  inference_tree_equal=(before["source_validation"]["inference_tree"]
                                        == after["source_validation"]["inference_tree"]))
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
        result = dict(state="INCONCLUSIVE" if isinstance(exc, Inconclusive) else "FAIL",
                      reason=f"{type(exc).__name__}: {exc}")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    passed = result["state"] == "PASS" and (
        not args.require_positive or result["preliminary_positive"])
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
