"""Foreground isolated baseline/capture server. Never terminates existing workers."""

import argparse
import errno
import json
import os
from pathlib import Path
import socket
import sys
import time


def check_port(port, host="127.0.0.1"):
    """Probe without SO_REUSEPORT; listen also rejects reusable live bindings.

    A failed plain bind alone is not evidence of a live listener. Retry with
    the server's address-reuse semantics, retaining both errors on failure.
    This is a preflight, not a reservation; the server must still bind itself.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(f"invalid TCP port: {port!r}")
    original = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
            probe.listen(1)
        return "available"
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise OSError(exc.errno, f"port_probe_socket_error {host}:{port}: {exc}") from exc
        original = exc
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
            probe.listen(1)
        return "available_with_reuseaddr (no listener; consistent with TIME_WAIT)"
    except OSError as exc:
        # Do not assert TIME_WAIT from EADDRINUSE: a non-listening socket can
        # own the address too, and historical TCP state is not observable here.
        kind = "socket_error"
        if exc.errno == errno.EADDRINUSE:
            kind = "address_in_use_non_listener_or_nonreusable_TIME_WAIT"
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.settimeout(.2)
                    if probe.connect_ex((host, port)) == 0:
                        kind = "active_listener"
            except OSError:
                pass
        raise OSError(exc.errno, f"port_unavailable {host}:{port} kind={kind}; "
                      f"plain_bind={original!r}; reuseaddr_bind_or_listen={exc!r}; "
                      "no existing service was stopped") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("baseline", "capture", "parity"), required=True)
    p.add_argument("--parity-socket", type=Path)
    p.add_argument("--protection-mode", choices=("DEBUG_SAFE", "FAST_PARITY"), default="FAST_PARITY",
                   help="parity validation mode; EC remains disabled in baseline")
    p.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    p.add_argument("--gpus", required=True, help="PCI ordered GPU indices, e.g. 0 or 0,1,2,3")
    p.add_argument("--port", type=int, default=30281)
    p.add_argument("--host", default="127.0.0.1", help="inference HTTP bind address")
    p.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--request-id", default="kv-ec-artificial-request")
    p.add_argument("--token-count", type=int, choices=(128, 256, 512), default=128)
    p.add_argument("--chunk-size", type=int, choices=(128, 256, 512),
                   help="capture a complete prompt as independent logical chunks")
    p.add_argument("--layers", choices=("0", "all"), default="0")
    p.add_argument("--transfer", choices=("pinned", "blocking"), default="pinned",
                   help="safe D2H A/B; pinned allocation errors reject capture rather than silently changing mode")
    p.add_argument("--profile-detail", action="store_true", help="per-layer wall/thread CPU timings (perturbs latency)")
    p.add_argument("--cuda-events", action="store_true", help="diagnostic CUDA stream elapsed times, not additive CPU costs")
    p.add_argument("--cpu-profile", action="store_true", help="cProfile capture only; diagnostic run, not latency benchmark")
    p.add_argument("--mem-fraction-static", type=float, default=0.60)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--allow-shared-gpu-debug", action="store_true",
                   help="explicit debug-only mode; read-only per-GPU memory admission")
    p.add_argument("--ec-warmups", type=int, default=0)
    p.add_argument("--ec-repeat", type=int, default=1, help="explicit bounded capture plan, total warmups+repeat <=16")
    args = p.parse_args()
    if args.chunk_size is not None and args.token_count != args.chunk_size:
        p.error("--chunk-size must equal --token-count")
    if not 0 < args.mem_fraction_static <= 1:
        p.error("--mem-fraction-static must be in (0, 1]")
    # Existing requests contain 768 input tokens and generate up to 16 more.
    if args.context_length < 784:
        p.error("--context-length must be at least 784 for the 768-token prompt and output")
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
    if not 0.05 <= args.mem_fraction_static <= 0.95:
        p.error("--mem-fraction-static must be in [0.05, 0.95]")
    if not 784 <= args.context_length <= 131072:
        p.error("--context-length must be in [784, 131072] for the 768+16-token request")
    if args.allow_shared_gpu_debug:
        from experiments.kv_ec.gpu_memory_budget import check_shared_gpu_budget
        try:
            admission = check_shared_gpu_budget(
                gpus, args.model, args.tp, args.context_length,
                args.mem_fraction_static)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            p.error(str(exc))
        print("Shared GPU debug memory admission:", json.dumps(admission), flush=True)
    chunk_plan = {}
    for request, snapshot in plan.items():
        if args.chunk_size is None:
            chunk_plan[request] = [snapshot]
        else:
            chunk_plan[request] = [
                f"{snapshot}-chunk-{index}"
                for index in range(1, 1 + (768 + args.chunk_size - 1) // args.chunk_size)
            ]
    if any((args.output / snapshot_id).exists()
           for snapshots in chunk_plan.values() for snapshot_id in snapshots):
        p.error("snapshot directory exists; use a new snapshot-id (never overwrite evidence)")
    # Read-only readiness check: not an authorization to take over this port.
    try:
        port_status = check_port(args.port, args.host)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(f"Port {args.port}: {port_status}", flush=True)
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
                      SGLANG_KV_EC_CAPTURE_PLAN=json.dumps(plan),
                      SGLANG_KV_EC_CHUNK_PLAN=json.dumps(chunk_plan),
                      SGLANG_KV_EC_REQUEST_VERSION=os.environ.get(
                          "SGLANG_KV_EC_REQUEST_VERSION", "request-v1"),
                      SGLANG_KV_EC_CHECKPOINT_VERSION=os.environ.get(
                          "SGLANG_KV_EC_CHECKPOINT_VERSION", args.snapshot_id),
                      SGLANG_KV_EC_MANAGER_EPOCH=os.environ.get(
                          "SGLANG_KV_EC_MANAGER_EPOCH", "local-scheduler-epoch"))
    if args.chunk_size is not None:
        # Three chunks are submitted independently. Keep their bounded IPC
        # ownership explicit; the Manager must be started with matching
        # --max-pending capacity.
        os.environ["SGLANG_KV_EC_PARITY_QUEUE"] = str(len(chunk_plan[args.request_id]))
    os.environ.pop("SGLANG_KV_EC_PARITY_SOCKET", None)
    os.environ.pop("SGLANG_KV_EC_PREPARED_IDENTITY", None)
    if args.mode == "parity":
        os.environ["SGLANG_KV_EC_PARITY_SOCKET"] = str(args.parity_socket)
        os.environ["SGLANG_KV_EC_PARITY_TIMEOUT"] = "2"
        from experiments.kv_ec.parity_manager import manager_hello

        os.environ["SGLANG_KV_EC_MANAGER_EPOCH"] = manager_hello(
            args.parity_socket
        )
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
            "--host", args.host, "--port", str(args.port), "--tp-size", str(args.tp),
            "--mem-fraction-static", str(args.mem_fraction_static),
            "--context-length", str(args.context_length), "--dtype", "bfloat16",
            "--kv-cache-dtype", "auto", "--page-size", "1", "--attention-backend", "triton",
            "--disable-radix-cache", "--chunked-prefill-size", "-1", "--disable-overlap-schedule",
            "--max-running-requests", "1", "--skip-server-warmup", "--random-seed", "42",
            "--cuda-graph-config", '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}']
    print("Starting isolated", args.mode, "server; shared_gpu_debug=",
          args.allow_shared_gpu_debug, "no existing workers will be stopped or modified.", flush=True)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
