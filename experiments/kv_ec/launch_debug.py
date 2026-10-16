"""Foreground isolated baseline/capture server. Never terminates existing workers."""

import argparse
import os
from pathlib import Path
import socket
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("baseline", "capture"), required=True)
    p.add_argument("--tp", type=int, choices=(1, 4), required=True)
    p.add_argument("--gpus", required=True, help="PCI ordered GPU indices, e.g. 0 or 0,1,2,3")
    p.add_argument("--port", type=int, default=30281)
    p.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--request-id", default="kv-ec-artificial-request")
    p.add_argument("--token-count", type=int, choices=(128, 256), default=128)
    p.add_argument("--layers", choices=("0", "all"), default="0")
    args = p.parse_args()
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
                      SGLANG_KV_EC_DEBUG="1" if args.mode == "capture" else "0",
                      SGLANG_KV_EC_OUTPUT_DIR=str(args.output), SGLANG_KV_EC_SNAPSHOT_ID=args.snapshot_id,
                      SGLANG_KV_EC_REQUEST_ID=args.request_id, SGLANG_KV_EC_TOKEN_START="0",
                      SGLANG_KV_EC_TOKEN_COUNT=str(args.token_count), SGLANG_KV_EC_LAYERS=args.layers)
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
