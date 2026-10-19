"""Foreground isolated baseline/capture server. Never terminates existing workers."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("baseline", "capture", "parity"), required=True)
    p.add_argument("--parity-socket", type=Path)
    p.add_argument("--protection-mode", choices=("DEBUG_SAFE", "FAST_PARITY"), default="FAST_PARITY",
                   help="parity validation mode; EC remains disabled in baseline")
    p.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    p.add_argument("--gpus", required=True, help="PCI ordered GPU indices, e.g. 0 or 0,1,2,3")
    p.add_argument("--port", type=int, default=30281)
    p.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--request-id", default="kv-ec-artificial-request")
    p.add_argument("--token-count", type=int, choices=(128, 256, 512), default=128)
    p.add_argument("--layers", choices=("0", "all"), default="0")
    p.add_argument("--transfer", choices=("pinned", "blocking"), default="pinned",
                   help="safe D2H A/B; pinned allocation errors reject capture rather than silently changing mode")
    p.add_argument("--profile-detail", action="store_true", help="per-layer wall/thread CPU timings (perturbs latency)")
    p.add_argument("--cuda-events", action="store_true", help="diagnostic CUDA stream elapsed times, not additive CPU costs")
    p.add_argument("--cpu-profile", action="store_true", help="cProfile capture only; diagnostic run, not latency benchmark")
    p.add_argument("--ec-warmups", type=int, default=0)
    p.add_argument("--ec-repeat", type=int, default=1, help="explicit bounded capture plan, total warmups+repeat <=16")
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.kv_ec.capture_plan import capture_plan
    try:
        plan = capture_plan(args.snapshot_id, args.request_id, args.ec_warmups, args.ec_repeat)
    except ValueError as exc:
        p.error(str(exc))
    if args.mode == "parity" and (args.tp not in (2, 4) or not args.parity_socket or not args.parity_socket.is_absolute()):
        p.error("parity requires TP=2/4 and an absolute --parity-socket")
    gpus = args.gpus.split(",")
    if len(gpus) != args.tp or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        p.error("select exactly TP distinct physical GPU indices")
    if not args.output.is_absolute():
        p.error("output must be absolute")
    if any((args.output / snapshot_id).exists() for snapshot_id in plan.values()):
        p.error("snapshot directory exists; use a new snapshot-id (never overwrite evidence)")
    # Read-only readiness check: not an authorization to take over this port.
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError:
            p.error("port unavailable; choose another port, do not kill other services")
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      SGLANG_KV_EC_DEBUG="0" if args.mode == "baseline" else "1",
                      SGLANG_KV_EC_PROTECTION_MODE=(args.protection_mode if args.mode == "parity" else "DEBUG_SAFE"),
                      SGLANG_KV_EC_OUTPUT_DIR=str(args.output), SGLANG_KV_EC_SNAPSHOT_ID=args.snapshot_id,
                      SGLANG_KV_EC_REQUEST_ID=args.request_id, SGLANG_KV_EC_TOKEN_START="0",
                      SGLANG_KV_EC_TOKEN_COUNT=str(args.token_count), SGLANG_KV_EC_LAYERS=args.layers)
    os.environ.update(SGLANG_KV_EC_TRANSFER=args.transfer,
                      SGLANG_KV_EC_PROFILE_DETAIL="1" if args.profile_detail else "0",
                      SGLANG_KV_EC_CUDA_EVENTS="1" if args.cuda_events else "0",
                      SGLANG_KV_EC_CPU_PROFILE="1" if args.cpu_profile else "0",
                      SGLANG_KV_EC_CAPTURE_PLAN=json.dumps(plan))
    os.environ.pop("SGLANG_KV_EC_PARITY_SOCKET", None)
    os.environ.pop("SGLANG_KV_EC_PREPARED_IDENTITY", None)
    if args.mode == "parity":
        os.environ["SGLANG_KV_EC_PARITY_SOCKET"] = str(args.parity_socket)
        os.environ["SGLANG_KV_EC_PARITY_TIMEOUT"] = "2"
    if args.mode != "baseline":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from experiments.kv_ec.debug_capture import source_identity

        started = time.perf_counter()
        identity = source_identity(args.model)
        initialization_ms = (time.perf_counter() - started) * 1000
        os.environ["SGLANG_KV_EC_PREPARED_IDENTITY"] = json.dumps(identity)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / (args.snapshot_id + "-initialization.json")).open("x") as stream:
            json.dump({"identity": identity, "model_fingerprint_initialization_ms": initialization_ms}, stream, indent=2)
        print("KV_EC model fingerprint initialized before server startup:", initialization_ms, "ms", flush=True)
    # Both protected modes use identical inference settings; only protection
    # validation/transport work differs. FAST_PARITY remains experimental.
    argv = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
            "--host", "127.0.0.1", "--port", str(args.port), "--tp-size", str(args.tp),
            "--mem-fraction-static", "0.60", "--context-length", "4096", "--dtype", "bfloat16",
            "--kv-cache-dtype", "auto", "--page-size", "1", "--attention-backend", "triton",
            "--disable-radix-cache", "--chunked-prefill-size", "-1", "--disable-overlap-schedule",
            "--max-running-requests", "1", "--skip-server-warmup", "--random-seed", "42",
            "--cuda-graph-config", '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}']
    print("Starting isolated", args.mode, "server; no existing processes will be stopped.", flush=True)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
