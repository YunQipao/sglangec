"""Stage B CLI: real checkpoint, native Decode, independent reference comparison."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .gpu_kv_restore import _require
from .gpu_resume_hook import token_digest, validate_decode_observation
from .gpu_restore_hook import validate_checkpoint
from .node_block_export import sha
from . import verify_gpu_restore as stage_a


def reference_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), run_sha256=sha((path / "run.json").read_bytes()),
                rank_sha256=[sha((path / f"rank_{r}.json").read_bytes()) for r in (0, 1)])


def validate_restore_evidence(result, expected, rank):
    old, new = result["old_slots"], result["new_slots"]
    _require(len(new) == len(set(new)) == 256 and not set(old) & set(new) and
             all(0 < slot <= result["allocator_size"] for slot in new),
             "stage_b_restored_slots_overlap")
    stats = result["stats"]
    _require(stats["loaded_kv_tokens"] == stats["token_count"] == 256 and
             stats["token_start"] == stats["recomputed_tokens"] == 0 and
             stats["unexpected_prefill"] is False and
             stats["written_bytes"] == expected["packed_byte_count"] and
             stats["tp_rank"] == rank and stats["tp_size"] == 2,
             "stage_b_restore_stats_invalid")
    layers = stats["layers_verified"]
    _require([x["layer_id"] for x in layers] == list(range(36)), "stage_b_missing_layers")
    for actual, ref in zip(layers, expected["layer_records"]):
        _require(actual["k_byte_exact"] is True and actual["v_byte_exact"] is True and
                 actual["k_sha256"] == ref["k_sha256"] and
                 actual["v_sha256"] == ref["v_sha256"], "stage_b_layer_bytes_invalid")


def read_run(output, expected_mode=None):
    """Fail closed on missing ranks, stale identities, misleading counters or logs."""
    output = Path(output)
    _require(not list(output.glob("rank_*.failure.json")), "stage_b_late_rank_failure")
    config = json.loads((output / "run.json").read_text())
    mode, steps = config["mode"], config["decode_steps"]
    _require(config["experiment"] == "gpu-resume-stage-b" and
             mode in ("reference", "resume", "corrupt") and 16 <= steps <= 128 and
             (expected_mode is None or mode == expected_mode), "stage_b_run_config_invalid")
    block = Path(config["checkpoint"]).read_bytes()
    meta = validate_checkpoint(block)
    _require(sha(block) == config["checkpoint_sha256"], "stage_b_checkpoint_changed")
    reports, all_logits = [], []
    for rank in (0, 1):
        r = json.loads((output / f"rank_{rank}.json").read_text())
        _require(r["schema"] == "kv-gpu-resume-stage-b-v1" and r["mode"] == mode and
                 r["run_id"] == config["run_id"] and r["tp_rank"] == rank and
                 r["tp_size"] == 2 and r["checkpoint_sha256"] == sha(block) and
                 r["decode_steps"] == steps, "stage_b_rank_identity_invalid")
        _require(r.get("failure_reason") is None and all(r.get(k) is True for k in (
            "cuda_executed", "generation_unchanged", "row_released", "no_slot_leak",
            "request_finished", "all_rows_released")), "stage_b_rank_lifecycle_failed")
        _require(r["runtime_identity"] == config["runtime_identity"] and
                 r["runtime_identity"]["model_sha256"] == meta["ranks"][rank]["model_sha256"] and
                 r["generation_before"] == r["generation_after"] > 0 and
                 r["free_slots_after"] == r["allocator_size"] and
                 r["free_rows_after"] == r["request_pool_size"] and
                 r["original_prefill_tokens"] == 256 and
                 r["original_prefill_forward_count"] == 1 and
                 r["recomputed_prefix_tokens"] == 0 and
                 r["token_ids_sha256"] == meta["ranks"][rank]["token_ids_sha256"],
                 "stage_b_rank_counters_invalid")
        outputs = r["output_token_ids"]
        count = 1 if mode == "corrupt" else steps + 1
        _require(len(outputs) == count and all(type(x) is int and x >= 0 for x in outputs) and
                 r["prefill_seed_token_id"] == outputs[0] and
                 r["resumed_generated_tokens"] == count - 1 and
                 r["output_token_ids_sha256"] == token_digest(outputs) and
                 r["resume_decode_forward_count"] == count - 1,
                 "stage_b_output_count_or_hash_invalid")
        old = r["old_slots"]
        _require(len(old) == len(set(old)) == 256, "stage_b_old_slots_invalid")
        if mode == "reference":
            _require(r["pass_reference"] is True and r["reference_no_restore"] is True and
                     r["committed"] is False and r["restored_kv_tokens"] == 0 and
                     r["restored_slots_used_by_decode"] is False and r["mapping_version"] == 0,
                     "stage_b_reference_restored_kv")
            prefix = old
        elif mode == "resume":
            _require(all(r.get(k) is True for k in (
                "local_resume_complete", "verified", "committed", "slots_disjoint",
                "restored_slots_used_by_decode", "request_state_unchanged_by_restore")) and
                r["restored_kv_tokens"] == 256 and r["transaction_state"] == "COMMITTED" and
                r["mapping_version"] == 1 and
                r["forward_ct_before_restore"] == r["forward_ct_after_restore"],
                "stage_b_restore_not_committed")
            state = r["request_state_before_restore"]
            _require(state == r["request_state_after_restore"] and
                     state["output_ids"] == [] and state["pending_token"] == [outputs[0]] and
                     state["kv_committed_len"] == state["kv_allocated_len"] == 256 and
                     state["seq_lens"] == [256] and state["extend_batch_idx"] == 1 and
                     state["decode_batch_idx"] == 0 and state["row"] == r["row"] and
                     state["generation"] == r["generation_before"] and
                     state["input_sha256"] == r["token_ids_sha256"],
                     "stage_b_request_boundary_state_invalid")
            prefix = r["new_slots"]
        else:
            _require(r["negative_test_pass"] is True and r["committed"] is False and
                     r["transaction_state"] == "ABORTED" and r["restored_kv_tokens"] == 0 and
                     r["restored_slots_used_by_decode"] is False and
                     r["expected_rejection"] == "tp_restore_verified_failed" and
                     (rank != 0 or (r["corruption_injected"] is True and
                                    r["corruption_detected"] is True)),
                     "stage_b_corruption_not_safely_rejected")
            prefix = old
        if mode != "reference":
            validate_restore_evidence(r, meta["ranks"][rank], rank)
        observations = r["forward_observations"]
        _require(len(observations) == len(r["logits"]) == count,
                 "stage_b_forward_observations_missing")
        logits = []
        for step, (o, logit) in enumerate(zip(observations, r["logits"])):
            _require(o["step"] == step and o["generation"] == r["generation_before"] and
                     o["req_pool_indices"] == [r["row"]] and
                     o["forward_ct"] == observations[0]["forward_ct"] + step and
                     o["kv_length"] == 256 + step and
                     len(o["mapping"]) == len(set(o["mapping"])) == 256 + step and
                     all(0 < slot <= r["allocator_size"] for slot in o["mapping"]),
                     "stage_b_forward_state_invalid")
            if step == 0:
                _require(o["mode"] == "EXTEND" and o["mapping"] == old and
                         o["mapping_version"] == 0 and o["positions"] == list(range(256)) and
                         len(o["input_ids"]) == 256 and token_digest(o["input_ids"]) ==
                         r["token_ids_sha256"], "stage_b_original_prefill_invalid")
            else:
                _require(o["mode"] == "DECODE" and
                         o["mapping_version"] == (1 if mode == "resume" else 0),
                         "stage_b_prefix_reprefilled")
                validate_decode_observation(step=step, outputs=outputs[:step],
                    seq_len=o["kv_length"], committed_len=o["kv_length"],
                    input_ids=o["input_ids"], positions=o["positions"],
                    indices=o["kv_indices"], mapping=o["mapping"], prefix_slots=prefix,
                    out_slots=o["out_cache_loc"])
            _require(logit["step"] == step and logit["file"] ==
                     f"rank_{rank}.logits_{step:03d}.npy", "stage_b_logit_filename_invalid")
            path = output / logit["file"]
            _require(sha(path.read_bytes()) == logit["sha256"], "stage_b_logits_changed")
            arr = np.load(path, allow_pickle=False)
            _require(arr.ndim == 2 and arr.shape[0] == 1 and
                     list(arr.shape) == logit["shape"] and np.isfinite(arr).all() and
                     int(np.argmax(arr[0])) == outputs[step], "stage_b_logits_or_sampling_invalid")
            logits.append(arr)
        events = [json.loads(line) for line in
                  (output / f"rank_{rank}.events.jsonl").read_text().splitlines()]
        phases = ["GLOBAL_READY_0_OK", "GLOBAL_OBSERVED_0_OK"]
        if mode == "resume":
            phases += ["VERIFIED", "GLOBAL_VERIFIED_OK", "MAPPED", "GLOBAL_MAPPED_OK",
                       "COMMITTED", "GLOBAL_COMMITTED_OK", "FINISHED", "GLOBAL_FINISHED_OK",
                       "GLOBAL_COMMIT"]
        elif mode == "corrupt":
            phases += ([] if rank == 0 else ["VERIFIED"]) + [
                "GLOBAL_VERIFIED_FAILED", "GLOBAL_NEGATIVE_ABORTED_OK"]
        phases += ["GLOBAL_RESULT_" + token_digest(outputs[:1]) + "_OK"]
        for step in range(1, count):
            phases += [f"GLOBAL_READY_{step}_OK", f"GLOBAL_OBSERVED_{step}_OK",
                       "GLOBAL_RESULT_" + token_digest(outputs[:step + 1]) + "_OK"]
        phases += ["COMPLETE"]
        _require([e["phase"] for e in events] == phases and all(
            e["tp_rank"] == rank and e["run_id"] == config["run_id"] for e in events),
            "stage_b_tp_barrier_evidence_invalid")
        reports.append(r)
        all_logits.append(logits)
    for key in ("output_token_ids", "output_token_ids_sha256", "generation_before",
                "generation_after", "row", "resume_decode_forward_count", "mapping_version"):
        _require(reports[0][key] == reports[1][key], "stage_b_rank_disagreement:" + key)
    return config, reports, all_logits


def compare_outputs(reference, resumed, reference_logits, resumed_logits):
    _require(len(reference) == len(resumed) == len(reference_logits) == len(resumed_logits),
             "stage_b_reference_length_mismatch")
    first = next((i for i, pair in enumerate(zip(reference, resumed)) if pair[0] != pair[1]), None)
    errors = []
    for step, (ref, actual) in enumerate(zip(reference_logits, resumed_logits)):
        _require(ref.shape == actual.shape, "stage_b_reference_logits_shape_mismatch")
        delta = actual.astype(np.float64) - ref.astype(np.float64)
        errors.append(dict(step=step, max_abs_error=float(np.max(np.abs(delta))),
                           rms_error=float(np.sqrt(np.mean(delta * delta)))))
    difference = None if first is None else dict(
        output_index=first, decode_step=first,
        reference_token_id=reference[first], resumed_token_id=resumed[first],
        logits_error=errors[first],
        reference_logits_at_tokens=reference_logits[first][0, [reference[first], resumed[first]]].tolist(),
        resumed_logits_at_tokens=resumed_logits[first][0, [reference[first], resumed[first]]].tolist())
    return dict(output_token_ids_match=first is None,
        reference_token_ids_sha256=token_digest(reference), resumed_token_ids_sha256=token_digest(resumed),
        first_difference=difference, logits_errors=errors,
        logits_after_first_difference_are_different_contexts=first is not None)


def check_results(output, reference=None):
    config, reports, logits = read_run(output)
    mode = config["mode"]
    verdict = dict(pass_stage_b=False, pass_reference=mode == "reference",
        negative_test_pass=mode == "corrupt", cuda_executed=True, tp_size=2,
        restored_kv_tokens=reports[0]["restored_kv_tokens"],
        restored_slots_used_by_decode=reports[0]["restored_slots_used_by_decode"],
        resume_decode_forward_count=reports[0]["resume_decode_forward_count"],
        recomputed_prefix_tokens=0, generation_unchanged=True, no_slot_leak=True,
        reference_token_ids_sha256=reports[0]["output_token_ids_sha256"] if mode == "reference" else None,
        resumed_token_ids_sha256=reports[0]["output_token_ids_sha256"] if mode == "resume" else None,
        output_token_ids_match=False, run_id=config["run_id"], ranks=reports)
    if mode != "resume":
        return verdict
    ref_path = Path(reference or config["reference"]["path"]).resolve()
    _require(reference_identity(ref_path) == config["reference"], "stage_b_reference_identity_changed")
    ref_config, refs, ref_logits = read_run(ref_path, "reference")
    def normalized_argv(cfg):
        argv = list(cfg["argv"])
        argv[argv.index("--port") + 1] = "<port>"
        return argv
    _require(ref_config["decode_steps"] == config["decode_steps"] and
             ref_config["checkpoint_sha256"] == config["checkpoint_sha256"] and
             ref_config["runtime_identity"] == config["runtime_identity"] and
             normalized_argv(ref_config) == normalized_argv(config),
             "stage_b_reference_runtime_mismatch")
    comparisons = [compare_outputs(refs[r]["output_token_ids"], reports[r]["output_token_ids"],
                                  ref_logits[r], logits[r]) for r in (0, 1)]
    passed = all(c["output_token_ids_match"] for c in comparisons)
    for r, comparison in zip(reports, comparisons):
        r.update(comparison, pass_stage_b=passed)
    verdict.update(pass_stage_b=passed, output_token_ids_match=passed,
        reference_token_ids_sha256=comparisons[0]["reference_token_ids_sha256"],
        resumed_token_ids_sha256=comparisons[0]["resumed_token_ids_sha256"],
        first_difference=comparisons[0]["first_difference"],
        logits_errors_by_rank=[c["logits_errors"] for c in comparisons], reference=str(ref_path))
    return verdict


def launch(args):
    _require(16 <= args.decode_steps <= 128, "stage_b_invalid_decode_steps")
    reference = None
    if args.mode == "resume":
        _require(args.reference is not None, "stage_b_reference_required")
        cfg, _, _ = read_run(args.reference, "reference")
        _require(cfg["decode_steps"] == args.decode_steps and cfg["checkpoint_sha256"] ==
                 sha(args.checkpoint.read_bytes()), "stage_b_reference_checkpoint_or_steps_mismatch")
        reference = reference_identity(args.reference)
    stage_a.launch(args, extra_config=dict(experiment="gpu-resume-stage-b", mode=args.mode,
        decode_steps=args.decode_steps, reference=reference), extra_env=dict(
        SGLANG_KV_EC_GPU_RESUME="1", SGLANG_KV_EC_GPU_RESUME_MODE=args.mode,
        SGLANG_KV_EC_GPU_RESUME_STEPS=str(args.decode_steps)))


def save_verdict(output, result):
    path = Path(output) / "check.json"
    if path.exists():
        _require(json.loads(path.read_text()) == result, "stage_b_existing_verdict_differs")
    else:
        with path.open("x") as handle:
            json.dump(result, handle, indent=2)


def submit(args):
    config = json.loads((args.output / "run.json").read_text())
    _require(config["experiment"] == "gpu-resume-stage-b", "stage_b_run_required")
    stage_a.submit(args, max_new_tokens=1 if config["mode"] == "corrupt" else
                   config["decode_steps"] + 1)
    # HTTP streaming can finish before the post-result cleanup vote saves JSON.
    # Wait for complete rank reports instead of treating HTTP as acceptance.
    deadline = time.monotonic() + args.timeout
    while True:
        reports = []
        for rank in (0, 1):
            try:
                report = json.loads((args.output / f"rank_{rank}.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            _require(report.get("failure_reason") is None, "stage_b_rank_reported_failure")
            reports.append(report)
        _require(not list(args.output.glob("rank_*.failure.json")), "stage_b_late_rank_failure")
        if len(reports) == 2:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("stage_b_rank_cleanup_reports_timeout")
        time.sleep(0.1)
    print("Both rank reports saved; run check for token/logit comparison and Stage B verdict.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser("launch")
    start.add_argument("--model", required=True)
    start.add_argument("--checkpoint", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--mode", choices=("reference", "resume", "corrupt"), required=True)
    start.add_argument("--reference", type=Path)
    start.add_argument("--decode-steps", type=int, default=16)
    start.add_argument("--gpus", default="auto")
    start.add_argument("--port", type=int, default=30482)
    start.add_argument("--mem-fraction-static", type=float, default=0.6)
    start.add_argument("--timeout", type=float, default=120)
    send = sub.add_parser("submit")
    send.add_argument("--output", type=Path, required=True)
    send.add_argument("--input-ids", type=Path, required=True)
    send.add_argument("--url", default="http://127.0.0.1:30482")
    send.add_argument("--wait-ready", type=float, default=600)
    send.add_argument("--timeout", type=float, default=300)
    check = sub.add_parser("check")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--reference", type=Path)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    if args.command == "inspect":
        print(json.dumps(stage_a.inspect_checkpoint(args.checkpoint), indent=2))
    elif args.command == "launch":
        launch(args)
    elif args.command == "submit":
        submit(args)
    else:
        try:
            result = check_results(args.output, args.reference)
        except Exception as exc:
            result = dict(pass_stage_b=False, pass_reference=False, negative_test_pass=False,
                          failure_reason=f"{type(exc).__name__}:{exc}")
        save_verdict(args.output, result)
        print(json.dumps(result, indent=2))
        if not any(result.get(k) is True for k in (
                "pass_stage_b", "pass_reference", "negative_test_pass")):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
