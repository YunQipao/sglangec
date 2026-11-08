"""Foreground stage-1 online server. Does not stop or modify existing processes."""

import argparse
import itertools
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys

from .checkpoint_protocol import recv_message, send_message
from .debug_capture import source_identity
from .gpu_memory_budget import check_shared_gpu_budget, read_gpu_memory, read_gpu_tasks
from .launch_debug import check_port
from .gpu_admission import args_mode, environment


def check_gpu_resources(args):
    """Read-only admission using physical indices, independent of parent CVD."""
    selected_mode = args_mode(args)
    tasks = read_gpu_tasks()
    rows = read_gpu_memory()
    if args.gpus == 'auto':
        candidates = [g for g in rows if selected_mode == 'shared' or not tasks.get(rows[g]['uuid'])]
        choices = []
        for pair in itertools.combinations(candidates, args.tp):
            try:
                budget = check_shared_gpu_budget(pair, args.model, args.tp, getattr(args, "context_length", 8192),
                    args.mem_fraction_static, memory_rows=rows, tasks_by_uuid=tasks,
                    actual_free_static=selected_mode == 'shared')
                choices.append((min(r['remaining_safety_margin_bytes'] for r in budget), pair))
            except ValueError:
                continue
        if not choices:
            raise ValueError('INSUFFICIENT_GPU_MEMORY: no admissible GPU rank set in ' + selected_mode)
        args.gpus = ','.join(max(choices)[1])
    gpus = args.gpus.split(',')
    if len(gpus) != args.tp or len(set(gpus)) != args.tp or not all(g.isdigit() for g in gpus):
        raise ValueError('distinct_physical_gpu_indices_required')
    for gpu in gpus:
        if gpu not in rows:
            raise ValueError(f"GPU index={gpu} reason=GPU_NOT_FOUND")
    if selected_mode == 'strict':
        selected = {rows[gpu]["uuid"] for gpu in gpus}
        if selected.intersection(tasks):
            raise ValueError(
                "selected GPU is busy; choose free GPUs (no process was stopped)"
            )
    return check_shared_gpu_budget(
        gpus, args.model, args.tp, getattr(args, "context_length", 8192), args.mem_fraction_static,
        memory_rows=rows, tasks_by_uuid=tasks,
        actual_free_static=selected_mode == 'shared',
        report=lambda admission: print(
            "GPU memory admission (bytes; required is additional memory; "
            "adjust --mem-fraction-static):", json.dumps(admission), flush=True),
    )


def server_argv(args):
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--context-length",
        str(getattr(args, "context_length", 8192)),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--page-size",
        "1",
        "--attention-backend",
        "triton",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        str(args.prefill_chunk_size),
        "--disable-overlap-schedule",
        "--max-running-requests",
        "1",
        "--skip-server-warmup",
        "--random-seed",
        "42",
        "--cuda-graph-config",
        '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}',
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protection-mode", choices=("xor", "replication", "rs-only", "xor-rs"),
                        help="default: manager mode (legacy local XOR without exporter)")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="same serving flags with protection disabled",
    )
    parser.add_argument(
        "--re-prefill-baseline",
        action="store_true",
        help="logical fault with hot in-process Re-Prefill; no EC protector",
    )
    parser.add_argument(
        "--gpus", required=True,
        help="physical nvidia-smi GPU indices; overrides CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--node-id", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--failure-scope", choices=("gpu-rank", "node"), default="node",
        help=(
            "gpu-rank: recover from locally committed XOR while node RS export "
            "continues asynchronously; node: require remote RS global commit"
        ),
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--manager-pid", type=int, help="owned manager PID; verify Unix SO_PEERCRED")
    parser.add_argument(
        "--run-id",
        required=True,
        help="must match local manager, unique per server lifetime",
    )
    parser.add_argument("--port", type=int, default=30381)
    parser.add_argument(
        "--capacity", type=int, default=2, help="queued + active snapshots per rank"
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=256,
        help="-1 disables chunked prefill",
    )
    parser.add_argument(
        "--protection-grain-tokens", choices=("auto", "128", "256", "512", "1024"),
        default="auto",
        help="auto: 128 for <=1K, 256 for <=4K, 512 for <=8K, otherwise 1024",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.60)
    parser.add_argument('--gpu-admission-mode', choices=('strict', 'shared'), default=None)
    parser.add_argument(
        "--live-checkpoint", type=Path,
        help="immutable global_node.bin preloaded into host memory for a fault trial",
    )
    parser.add_argument(
        "--live-control", type=Path,
        help="optional atomic JSON fault-control file",
    )
    parser.add_argument("--fault-after-decode", type=int)
    parser.add_argument(
        "--fault-at-kv-boundary", type=int,
        help="pause exactly when committed KV reaches this token boundary",
    )
    parser.add_argument(
        "--prefill-tokens-per-s", type=float,
        help="same-model warm Re-Prefill calibration for the hybrid selector",
    )
    parser.add_argument(
        "--restore-bytes-per-s", type=float,
        help="same-boundary EC reconstruction+writeback calibration",
    )
    parser.add_argument(
        "--disable-opt", action="append", default=[], choices=(
            "hot-recovery", "memory-recovery", "streaming", "pipeline",
            "hash-reuse", "incremental", "adaptive-grain", "hybrid",
            "cuda-rs", "fused-xor-write", "async-verify", "async-d2h",
            "same-process-switch",
        ),
        help="all optimizations are enabled by default; repeat to disable one",
    )
    parser.add_argument(
        "--allow-shared-gpu-debug", action="store_true",
        help="legacy alias for shared admission when --gpu-admission-mode is omitted",
    )
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if (
        (args.baseline and args.re_prefill_baseline)
        or
        len(gpus) != args.tp
        or len(set(gpus)) != args.tp
        or not all(g.isdigit() for g in gpus)
        or not args.socket.is_absolute()
        or args.capacity < 1
        or not args.run_id
        or len(args.run_id) > 96
        or not 0.05 <= args.mem_fraction_static <= 0.95
        or args.prefill_chunk_size not in (-1, 256, 512, 768, 1024)
        or (args.fault_after_decode is not None and args.fault_after_decode < 1)
        or (
            args.fault_at_kv_boundary is not None
            and (
                args.fault_at_kv_boundary < 1
                or args.fault_at_kv_boundary > args.context_length
            )
        )
        or (args.live_checkpoint is not None and not args.live_checkpoint.is_file())
        or (args.live_control is not None and not args.live_control.is_absolute())
        or (args.prefill_tokens_per_s is not None and args.prefill_tokens_per_s <= 0)
        or (args.restore_bytes_per_s is not None and args.restore_bytes_per_s <= 0)
    ):
        parser.error(
            "invalid GPU selection, socket, capacity, run id or memory/prefill limits"
        )
    try:
        check_gpu_resources(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    check_port(args.port)
    if not args.baseline and not args.re_prefill_baseline:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            sock.connect(str(args.socket))
            if args.manager_pid is not None:
                pid, uid, _ = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if pid != args.manager_pid or uid != os.getuid():
                    parser.error("online manager socket peer ownership mismatch")
            send_message(sock, {"kind": "hello"}, b"")
            ready, _ = recv_message(sock)
        if ready.get("state") != "READY" or ready.get("run_id") != args.run_id:
            parser.error("online manager is not ready for this run id")
        if args.protection_mode is None:
            args.protection_mode = ready.get("protection_mode", "xor")
        if ready.get("protection_mode", "xor") != args.protection_mode:
            parser.error("online manager protection mode mismatch")
    if args.protection_mode is None:
        args.protection_mode = "xor"
    # This launcher is the identity producer. Never inherit a stale prepared
    # identity from a previous service/model lifetime in the parent's env.
    os.environ.pop("SGLANG_KV_EC_PREPARED_IDENTITY", None)
    identity = source_identity(args.model)
    check_gpu_resources(args)  # resource competition can change during hashing/handshake
    # The resource policy is passed explicitly to execve, never exported to
    # the caller's shell or persisted in the Conda/system environment.
    disabled = set(args.disable_opt)
    def enabled(name):
        return "0" if args.baseline or name in disabled else "1"

    if "adaptive-grain" in disabled:
        protection_grain = 256
    elif args.protection_grain_tokens == "auto":
        protection_grain = 128 if args.context_length <= 1024 else (
            256 if args.context_length <= 4096 else (
                512 if args.context_length <= 8192 else 1024
            )
        )
    else:
        protection_grain = int(args.protection_grain_tokens)

    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SGLANG_KV_EC_ONLINE=(
            "0" if args.baseline or args.re_prefill_baseline else "1"
        ),
        SGLANG_KV_EC_ONLINE_SOCKET=str(args.socket),
        SGLANG_KV_EC_ONLINE_RUN_ID=args.run_id,
        SGLANG_KV_EC_ONLINE_PROTECTION_MODE=args.protection_mode,
        SGLANG_KV_EC_NODE_ID=str(args.node_id),
        SGLANG_KV_EC_REQUIRE_GLOBAL_RS=(
            "1" if (
                args.failure_scope == "node"
                and args.protection_mode in ("xor-rs", "rs-only")
            ) else "0"
        ),
        SGLANG_KV_EC_FAILURE_SCOPE=args.failure_scope,
        SGLANG_KV_EC_DEBUG="0",
        SGLANG_KV_EC_PAUSE_RESUME="0",
        SGLANG_KV_EC_GPU_RESTORE="0",
        SGLANG_KV_EC_GPU_RESUME="0",
        SGLANG_KV_EC_GPU_COLD_RESUME="0",
        SGLANG_KV_EC_RS_RESUME="0",
        SGLANG_KV_EC_LIVE_RECOVERY=(
            "1" if args.re_prefill_baseline else enabled("hot-recovery")
        ),
        SGLANG_KV_EC_HOT_MEMORY=enabled("memory-recovery"),
        SGLANG_KV_EC_STREAMING_RECOVERY=enabled("streaming"),
        SGLANG_KV_EC_PIPELINE=enabled("pipeline"),
        SGLANG_KV_EC_HASH_REUSE=enabled("hash-reuse"),
        SGLANG_KV_EC_INCREMENTAL=enabled("incremental"),
        SGLANG_KV_EC_ADAPTIVE_GRAIN=enabled("adaptive-grain"),
        SGLANG_KV_EC_HYBRID=enabled("hybrid"),
        SGLANG_KV_EC_CUDA_RS=enabled("cuda-rs"),
        SGLANG_KV_EC_FUSED_XOR_WRITE=enabled("fused-xor-write"),
        SGLANG_KV_EC_ASYNC_VERIFY=enabled("async-verify"),
        SGLANG_KV_EC_DEFER_FULL_AUDIT=enabled("async-verify"),
        SGLANG_KV_EC_ASYNC_D2H=enabled("async-d2h"),
        SGLANG_KV_EC_SAME_PROCESS_SWITCH=(
            "0" if args.re_prefill_baseline else enabled("same-process-switch")
        ),
        SGLANG_KV_EC_ONLINE_CAPACITY=str(args.capacity),
        SGLANG_KV_EC_ONLINE_GRAIN_TOKENS=str(protection_grain),
        SGLANG_KV_EC_PREPARED_IDENTITY=json.dumps(identity),
    )
    if args.live_checkpoint is not None:
        os.environ["SGLANG_KV_EC_LIVE_CHECKPOINT"] = str(args.live_checkpoint.resolve())
    else:
        os.environ.pop("SGLANG_KV_EC_LIVE_CHECKPOINT", None)
    if args.live_control is not None:
        os.environ["SGLANG_KV_EC_LIVE_CONTROL"] = str(args.live_control)
    else:
        os.environ.pop("SGLANG_KV_EC_LIVE_CONTROL", None)
    if args.fault_after_decode is not None:
        os.environ["SGLANG_KV_EC_FAULT_AFTER_DECODE"] = str(args.fault_after_decode)
    else:
        os.environ.pop("SGLANG_KV_EC_FAULT_AFTER_DECODE", None)
    if args.fault_at_kv_boundary is not None:
        os.environ["SGLANG_KV_EC_FAULT_AT_KV_BOUNDARY"] = str(
            args.fault_at_kv_boundary
        )
    else:
        os.environ.pop("SGLANG_KV_EC_FAULT_AT_KV_BOUNDARY", None)
    if args.prefill_tokens_per_s is not None:
        os.environ["SGLANG_KV_EC_PREFILL_TOKENS_PER_S"] = str(args.prefill_tokens_per_s)
    else:
        os.environ.pop("SGLANG_KV_EC_PREFILL_TOKENS_PER_S", None)
    if args.restore_bytes_per_s is not None:
        os.environ["SGLANG_KV_EC_RESTORE_BYTES_PER_S"] = str(args.restore_bytes_per_s)
    else:
        os.environ.pop("SGLANG_KV_EC_RESTORE_BYTES_PER_S", None)
    os.execve(sys.executable, server_argv(args), dict(os.environ, **environment(args_mode(args))))


if __name__ == "__main__":
    main()
