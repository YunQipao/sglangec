"""Launch, submit and inspect the real TP2 Stage A scheduler experiment."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from .gpu_restore_hook import validate_checkpoint
from .gpu_kv_restore import _require
from .node_block_export import sha


def read_input_ids(path, meta):
    ids = json.loads(Path(path).read_text())
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    _require(isinstance(ids, list) and len(ids) == 256 and
             all(type(x) is int and x >= 0 for x in ids), "exactly_256_input_ids_required")
    digest = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    _require(digest == meta["ranks"][0]["token_ids_sha256"],
             "checkpoint_input_ids_hash_mismatch")
    return ids


def public_test_inputs(args):
    """Reconstruct ONLY request_debug's public prompt and verify its digest.

    Checkpoints deliberately do not store private input tokens. For any other
    source request the caller must supply its own original --input-ids file.
    """
    from transformers import AutoTokenizer
    meta = validate_checkpoint(args.checkpoint.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer.encode("KV cache research with artificial public test data. " * 160,
                           add_special_tokens=False)[:256]
    digest = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    _require(len(ids) == 256 and digest == meta["ranks"][0]["token_ids_sha256"],
             "checkpoint_does_not_match_request_debug_public_prompt")
    with args.output.open("x") as handle:
        json.dump(ids, handle)
    print(json.dumps(dict(input_ids_file=str(args.output), token_count=256,
                          token_ids_sha256=digest, gpu_acceptance=False)))


def inspect_checkpoint(path):
    block = Path(path).read_bytes()
    meta = validate_checkpoint(block)
    return dict(checkpoint=str(Path(path).resolve()), checkpoint_sha256=sha(block),
                node_id=meta["node_id"], tp_size=2, token_start=0, token_count=256,
                model_sha256=meta["ranks"][0]["model_sha256"],
                token_ids_sha256=meta["ranks"][0]["token_ids_sha256"],
                rank_bytes=[r["packed_byte_count"] for r in meta["ranks"]],
                provenance=[r["provenance"] for r in meta["ranks"]],
                cpu_checkpoint_validation=True, gpu_acceptance=False)


def check_results(output):
    output = Path(output)
    _require(not list(output.glob("rank_*.failure.json")), "rank_reported_late_failure")
    config = json.loads((output / "run.json").read_text())
    block = Path(config["checkpoint"]).read_bytes()
    meta = validate_checkpoint(block)
    _require(sha(block) == config["checkpoint_sha256"], "checkpoint_reference_changed")
    results = [json.loads((output / f"rank_{r}.json").read_text()) for r in (0, 1)]
    for rank, result in enumerate(results):
        expected = meta["ranks"][rank]
        _require(result["schema"] == "kv-gpu-restore-stage-a-v1" and
                 result["tp_rank"] == rank and result["tp_size"] == 2 and
                 result["run_id"] == config["run_id"] and
                 result["checkpoint_sha256"] == sha(block), "result_identity_mismatch")
        _require(all(result.get(k) is True for k in (
            "pass_stage_a", "cuda_executed", "verified", "committed", "slots_disjoint",
            "generation_unchanged", "row_released", "no_slot_leak", "request_finished",
            "mapping_matches_new")),
            "rank_did_not_pass_gpu_restore")
        _require(result.get("failure_reason") is None and
                 result["transaction_state"] == "COMMITTED" and
                 result["generation_before"] > 0 and
                 result["generation_before"] == result["generation_after"] and
                 result["original_prefill_tokens"] == 256 and
                 result["original_prefill_forward_count"] == 1 and
                 result["original_sampled_output_tokens"] == 1 and
                 result["decode_forward_count"] == result["restore_forward_count"] == 0 and
                 result["forward_ct_before_restore"] == result["forward_ct_after_restore"] and
                 result["free_slots_after"] == result["allocator_size"], "rank_lifecycle_mismatch")
        old, new = result["old_slots"], result["new_slots"]
        _require(len(set(old)) == len(set(new)) == len(old) == len(new) == 256 and
                 not set(old) & set(new) and
                 all(0 < x <= result["allocator_size"] for x in old + new),
                 "rank_slot_evidence_invalid")
        stats = result["stats"]
        _require(stats["loaded_kv_tokens"] == stats["token_count"] == 256 and
                 stats["token_start"] == stats["recomputed_tokens"] == 0 and
                 stats["unexpected_prefill"] is False and
                 stats["written_bytes"] == expected["packed_byte_count"] and
                 stats["tp_rank"] == rank and stats["tp_size"] == 2 and
                 result["token_ids_sha256"] == expected["token_ids_sha256"] and
                 result["runtime_identity"]["model_sha256"] == expected["model_sha256"],
                 "rank_restore_stats_mismatch")
        layers = stats["layers_verified"]
        _require([x["layer_id"] for x in layers] == list(range(36)), "missing_layer_verification")
        for actual, ref in zip(layers, expected["layer_records"]):
            _require(actual["k_byte_exact"] is True and actual["v_byte_exact"] is True and
                     actual["k_sha256"] == ref["k_sha256"] and
                     actual["v_sha256"] == ref["v_sha256"], "layer_byte_evidence_mismatch")
        events = [json.loads(line) for line in
                  (output / f"rank_{rank}.events.jsonl").read_text().splitlines()]
        _require([event["phase"] for event in events] == [
            "VERIFIED", "GLOBAL_VERIFIED_OK", "MAPPED", "GLOBAL_MAPPED_OK",
            "COMMITTED", "GLOBAL_COMMITTED_OK", "FINISHED", "GLOBAL_FINISHED_OK",
            "GLOBAL_COMMIT", "GLOBAL_CLEANUP_OK", "PASS"], "tp_phase_evidence_incomplete")
        _require(all(e["tp_rank"] == rank and e["run_id"] == config["run_id"] for e in events),
                 "tp_event_identity_mismatch")
    for key in ("checkpoint_sha256", "token_ids_sha256", "original_prefill_tokens",
                "generation_before", "generation_after", "row", "transaction_state"):
        _require(results[0][key] == results[1][key], "tp_rank_results_disagree:" + key)
    return dict(pass_stage_a=True, cuda_executed=True, tp_ranks=[0, 1],
                loaded_kv_tokens=256, recomputed_tokens=0, unexpected_prefill=False,
                original_prefill_tokens=256, decode_forward_count=0,
                checkpoint_sha256=sha(block), run_id=config["run_id"], ranks=results)


def launch(args, *, extra_config=None, extra_env=None):
    from .debug_capture import source_identity
    from .gpu_memory_budget import read_gpu_memory, read_gpu_tasks
    from .launch_online import check_gpu_resources, server_argv
    from .launch_debug import check_port
    block = args.checkpoint.resolve().read_bytes()
    validate_checkpoint(block)
    args.tp, args.prefill_chunk_size, args.allow_shared_gpu_debug = 2, -1, False
    _require(0.05 <= args.mem_fraction_static <= 0.95, "invalid_memory_fraction")
    if args.gpus == "auto":
        rows, tasks = read_gpu_memory(), read_gpu_tasks()
        idle = sorted((g for g in rows if not tasks.get(rows[g]["uuid"])),
                      key=lambda g: rows[g]["free_bytes"], reverse=True)
        chosen = None
        for pair in itertools.combinations(idle, 2):
            args.gpus = ",".join(pair)
            try:
                check_gpu_resources(args)
                chosen = args.gpus
                break
            except ValueError:
                continue
        _require(chosen is not None, "two_idle_gpus_with_sufficient_memory_required")
    else:
        gpus = args.gpus.split(",")
        _require(len(gpus) == len(set(gpus)) == 2 and all(g.isdigit() for g in gpus),
                 "two_distinct_gpu_indices_required")
        check_gpu_resources(args)
    check_port(args.port)
    # Do not trust a parent's prepared identity when creating new evidence.
    os.environ.pop("SGLANG_KV_EC_PREPARED_IDENTITY", None)
    identity = source_identity(args.model)
    _require(identity["model_sha256"] == validate_checkpoint(block)["ranks"][0]["model_sha256"],
             "checkpoint_model_hash_mismatch")
    output = args.output.resolve()
    output.mkdir(exist_ok=False)  # no existing results can be overwritten
    config = dict(run_id=output.name, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=sha(block), gpus=args.gpus, model=args.model,
                  runtime_identity=identity, argv=server_argv(args))
    config.update(extra_config or {})
    with (output / "run.json").open("x") as f:
        json.dump(config, f, indent=2)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SGLANG_KV_EC_GPU_RESTORE="1",
        SGLANG_KV_EC_GPU_RESUME="0",
        SGLANG_KV_EC_GPU_COLD_RESUME="0",
        SGLANG_KV_EC_RS_RESUME="0",
        SGLANG_KV_EC_GPU_RESTORE_CHECKPOINT=str(args.checkpoint.resolve()),
        SGLANG_KV_EC_GPU_RESTORE_OUTPUT=str(output),
        SGLANG_KV_EC_GPU_RESTORE_RUN_ID=output.name,
        SGLANG_KV_EC_GPU_RESTORE_TIMEOUT=str(args.timeout),
        SGLANG_KV_EC_PREPARED_IDENTITY=json.dumps(identity),
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0",
    )
    os.environ.update(extra_env or {})
    print(json.dumps(config, indent=2), flush=True)
    os.execv(sys.executable, server_argv(args))


def submit(args, *, max_new_tokens=1, sampling_overrides=None, request_id=None):
    output = args.output.resolve()
    config = json.loads((output / "run.json").read_text())
    block = Path(config["checkpoint"]).read_bytes()
    _require(sha(block) == config["checkpoint_sha256"], "checkpoint_reference_changed")
    ids = read_input_ids(args.input_ids, validate_checkpoint(block))
    deadline = time.monotonic() + args.wait_ready
    while True:
        try:
            with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError("stage_a_server_not_ready")
            time.sleep(1)
    sampling = dict(max_new_tokens=max_new_tokens, temperature=0, ignore_eos=True)
    sampling.update(sampling_overrides or {})
    request = dict(input_ids=ids, rid=request_id or config["run_id"],
                   sampling_params=sampling,
                   stream=False)
    # Reserve the submission marker before sending: retry cannot silently create
    # another request in a run whose one-shot request may have been accepted.
    with (output / "submitted.json").open("x") as f:
        json.dump(dict(run_id=config["run_id"], token_count=len(ids)), f)
    req = urllib.request.Request(args.url.rstrip("/") + "/generate",
        data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        result = json.load(response)
    with (output / "http_response.json").open("x") as f:
        json.dump(result, f, indent=2)
    print("HTTP complete; GPU acceptance requires the check command and BOTH rank reports.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser("launch")
    start.add_argument("--model", required=True)
    start.add_argument("--checkpoint", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True, help="new directory only")
    start.add_argument("--gpus", default="auto", help="two idle physical GPU indices or auto")
    start.add_argument("--port", type=int, default=30481)
    start.add_argument("--mem-fraction-static", type=float, default=0.6)
    start.add_argument("--timeout", type=float, default=120)
    send = sub.add_parser("submit")
    send.add_argument("--output", type=Path, required=True)
    send.add_argument("--input-ids", type=Path, required=True,
                      help="original capture's first 256 token IDs, JSON array or input_ids object")
    send.add_argument("--url", default="http://127.0.0.1:30481")
    send.add_argument("--wait-ready", type=float, default=600)
    send.add_argument("--timeout", type=float, default=300)
    check = sub.add_parser("check")
    check.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("inspect", help="CPU-only real checkpoint validation")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inputs = sub.add_parser("public-test-inputs", help="recover request_debug public prompt IDs only")
    inputs.add_argument("--checkpoint", type=Path, required=True)
    inputs.add_argument("--model", required=True)
    inputs.add_argument("--output", type=Path, required=True, help="new JSON file only")
    args = p.parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "submit":
        submit(args)
    elif args.command == "inspect":
        print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))
    elif args.command == "public-test-inputs":
        public_test_inputs(args)
    else:
        print(json.dumps(check_results(args.output), indent=2))


if __name__ == "__main__":
    main()
