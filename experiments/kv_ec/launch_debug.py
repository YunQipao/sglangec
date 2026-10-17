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
    p.add_argument("--tp", type=int, choices=(1, 4), required=True)
    p.add_argument("--gpus", required=True, help="PCI ordered GPU indices, e.g. 0 or 0,1,2,3")
    p.add_argument("--port", type=int, default=30281)
    p.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--request-id", default="kv-ec-artificial-request")
    p.add_argument("--token-count", type=int, choices=(128, 256, 512), default=128)
    p.add_argument("--layers", choices=("0", "all"), default="0")
    args = p.parse_args()
    if args.mode == "parity" and (args.tp != 4 or not args.parity_socket or not args.parity_socket.is_absolute()):
        p.error("parity requires TP=4 and an absolute --parity-socket")
    gpus = args.gpus.split(",")
    if len(gpus) != args.tp or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        p.error("select exactly TP distinct physical GPU indices")
    if not args.output.is_absolute():
        p.error("output must be absolute")
    if (args.output / args.snapshot_id).exists():
        p.error("snapshot directory exists; use a new snapshot-id (never overwrite evidence)")
    # Read-only readiness check: not an authorization to take over this port.
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError:
            p.error("port unavailable; choose another port, do not kill other services")
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      SGLANG_KV_EC_DEBUG="0" if args.mode == "baseline" else "1",
                      SGLANG_KV_EC_OUTPUT_DIR=str(args.output), SGLANG_KV_EC_SNAPSHOT_ID=args.snapshot_id,
                      SGLANG_KV_EC_REQUEST_ID=args.request_id, SGLANG_KV_EC_TOKEN_START="0",
                      SGLANG_KV_EC_TOKEN_COUNT=str(args.token_count), SGLANG_KV_EC_LAYERS=args.layers)
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
    # Both modes use identical inference settings; only the hook switch differs.
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
