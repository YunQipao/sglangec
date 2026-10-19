"""Artificial sequential profiling requests and live protection validation.

Only target warmups and measurements are protected; ordinary model warmups are
separate. A fresh manager is required for each bounded plan (16 checkpoint cap).
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from .capture_plan import capture_plan
from .parity_manager import submit_once
from .request_debug import compare, request


def validate_timing(record):
    phase = record["performance_ms"]
    if phase["timing_schema"] != "kv-ec-exclusive-wall-v2":
        raise ValueError("requires v2 exclusive timing")
    wall = phase["wall_regions_ms"]
    if any(value < 0 for value in wall.values()):
        raise ValueError("negative wall region")
    if abs(sum(wall.values()) - phase["accounted_ms"]) > 1e-6:
        raise ValueError("accounted sum mismatch")
    if abs(phase["accounted_ms"] + phase["unaccounted_ms"] - phase["scheduler_capture_ms"]) > 1e-6:
        raise ValueError("exclusive timing balance failed")
    return phase


def wait_commit(socket, snapshot, deadline_seconds):
    deadline = time.monotonic() + deadline_seconds
    while True:
        status = submit_once(socket, {"kind": "status", "snapshot_id": snapshot}, b"", timeout=5)
        if status["state"] == "COMMITTED":
            return status
        if status["state"] in ("FAILED", "EXPIRED"):
            raise RuntimeError(status)
        if time.monotonic() >= deadline:
            raise TimeoutError(status)
        time.sleep(0.2)


def exclusive_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def summarize(values):
    ordered = sorted(values)
    # Nearest-rank P95, not interpolated; same definition for all modes.
    import math
    return {"median": statistics.median(values), "p95": ordered[math.ceil(.95 * len(values)) - 1],
            "n": len(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--request-id", default="kv-ec-artificial-request")
    p.add_argument("--ec-warmups", type=int, default=2)
    p.add_argument("--ec-repeat", type=int, default=10)
    p.add_argument("--model-warmups", type=int, default=2)
    p.add_argument("--socket", type=Path, help="omit for baseline; identical request plan")
    p.add_argument("--tp", type=int, choices=(2, 4), required=True)
    p.add_argument("--token-count", type=int, choices=(128, 256, 512), default=256)
    p.add_argument("--protection-mode", choices=("FAST_PARITY", "DEBUG_SAFE"), default="FAST_PARITY")
    p.add_argument("--transfer", choices=("pinned", "blocking"), default="pinned")
    p.add_argument("--compare-dir", type=Path, help="reference response files from the same snapshot/request plan")
    p.add_argument("--compare-snapshot-id", help="reference plan prefix; keeps snapshot IDs unique across modes")
    p.add_argument("--commit-timeout", type=float, default=60)
    args = p.parse_args()
    plan = capture_plan(args.snapshot_id, args.request_id, args.ec_warmups, args.ec_repeat)
    reference_plan = list(capture_plan(args.compare_snapshot_id or args.snapshot_id,
                                     args.request_id, args.ec_warmups, args.ec_repeat).values())
    if args.model_warmups < 0:
        p.error("model-warmups must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    # Reserve summary before starting, to reject accidental reruns early.
    summary_path = args.output / f"{args.snapshot_id}-profile-summary.json"
    if summary_path.exists() or any((args.output / f"{snap}-response.json").exists() for snap in plan.values()):
        p.error("evidence already exists; choose a fresh snapshot/output")

    def send(rid, path):
        request(SimpleNamespace(model=args.model, url=args.url, request_id=rid,
                                prompt_tokens=768, stream_metrics=True, output=path))
        response = json.loads(path.read_text())
        if len(response["output_ids"]) != 16:
            raise AssertionError("requires exactly 16 output tokens")
        return response

    for i in range(args.model_warmups):
        send(f"model-warmup-{args.snapshot_id}-{i}", args.output / f"{args.snapshot_id}-model-warmup-{i}.json")
    records = []
    for index, (rid, snapshot) in enumerate(plan.items()):
        response_path = args.output / f"{snapshot}-response.json"
        response = send(rid, response_path)
        if args.compare_dir:
            compare([args.compare_dir / f"{reference_plan[index]}-response.json", response_path])
        item = {"snapshot_id": snapshot, "request_id": rid,
                "kind": "ec_warmup" if index < args.ec_warmups else "measurement",
                "client_metrics_ms": response["client_metrics_ms"]}
        if args.socket:
            status = wait_commit(args.socket, snapshot, args.commit_timeout)
            expected = 36 * args.token_count * (8 // args.tp) * 128 * 2 * 2
            assert status["tp_size"] == args.tp and status["snapshot_id"] == snapshot
            assert status["request_id"] == hashlib.sha256(rid.encode()).hexdigest()
            assert status["format"] == "kv-ec-rank-bin-v1" and status["model_identifier"] == args.model
            assert status["layers"] == list(range(36)) and status["token_start"] == 0
            assert status["token_count"] == args.token_count and status["dtype"] == "torch.bfloat16"
            assert status["parity_bytes"] == status["steady_parity_bytes"] == status["packed_byte_count"] == expected
            assert status["oracle_checks"] == [{"lost_rank": i, "byte_exact": True} for i in range(args.tp)]
            assert set(status["rank_hashes"]) == {str(i) for i in range(args.tp)}
            for layer in status["layer_records"]:
                for kind in ("k", "v"):
                    assert layer["packed_shape"][kind] == [args.token_count, 8 // args.tp, 128]
            directory = args.output / snapshot
            deadline = time.monotonic() + 10
            while not all((directory / f"rank_{rank}.diagnostic.json").exists() for rank in range(args.tp)):
                if time.monotonic() >= deadline:
                    raise TimeoutError("missing timing/diagnostic evidence")
                time.sleep(.1)
            ranks = []
            for rank in range(args.tp):
                record = json.loads((directory / f"rank_{rank}.timing.json").read_text())
                phase = validate_timing(record)
                assert record["snapshot_id"] == snapshot and record["tp_rank"] == rank
                assert record["parity_protection"] == "ENQUEUED"
                assert phase["protection_mode"] == args.protection_mode and phase["transfer_mode"] == args.transfer
                assert phase["d2h_bytes"] == expected
                if args.protection_mode == "FAST_PARITY":
                    assert phase["source_recheck_ms"] == 0
                tail = json.loads((directory / f"rank_{rank}.diagnostic.json").read_text())
                ranks.append({"rank": rank, "performance_ms": phase, "diagnostic": tail})
            assert not list(directory.glob("*.bin"))
            assert not any((directory / f"rank_{rank}.json").exists() for rank in range(args.tp))
            exclusive_json(args.output / f"{snapshot}-commit.json", status)
            item.update(ranks=ranks, xor_ms=status["xor_ms"], commit_ms=status["parity_commit_ms"],
                        peak_process_rss_bytes=status["peak_process_rss_bytes"])
            print(snapshot, item["kind"], "xor_ms=", item["xor_ms"], "commit_ms=", item["commit_ms"])
            for rank in ranks:
                phase = rank["performance_ms"]
                top = sorted(phase["wall_regions_ms"].items(), key=lambda kv: kv[1], reverse=True)[:6]
                print("rank", rank["rank"], "capture_ms", phase["scheduler_capture_ms"],
                      "unaccounted_ms", phase["unaccounted_ms"], "top wall regions", top)
        records.append(item)
    measurements = [r for r in records if r["kind"] == "measurement"]
    summary = {"snapshot_id": args.snapshot_id, "records": records,
               "note": "sequential same-process requests, NOT independent model starts; profiler runs are diagnostic only",
               "metrics": {key: summarize([r["client_metrics_ms"][key] for r in measurements])
                           for key in ("ttft_ms", "tpot_ms", "request_total_ms")}}
    if args.socket:
        summary["protection_success_rate"] = 1.0  # every attempted planned capture was verified above
        summary["manager_metrics"] = {key: summarize([r[key] for r in measurements])
                                      for key in ("xor_ms", "commit_ms", "peak_process_rss_bytes")}
        summary["rank_metrics_ms"] = {
            str(rank): {key: summarize([r["ranks"][rank]["performance_ms"][key] for r in measurements])
                        for key in ("scheduler_capture_ms", "kv_gather_ms", "d2h_ms", "sha256_ms", "ipc_enqueue_ms", "unaccounted_ms")}
            for rank in range(args.tp)}
    exclusive_json(summary_path, summary)
    print(json.dumps(summary["metrics"], indent=2))
    print("PASS: request plan completed", summary_path)


if __name__ == "__main__":
    main()
