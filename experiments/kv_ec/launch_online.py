"""Foreground stage-1 online server. Does not stop or modify existing processes."""

import argparse
import itertools
import json
import os
from pathlib import Path
import socket
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
                budget = check_shared_gpu_budget(pair, args.model, args.tp, 4096,
                    args.mem_fraction_static, memory_rows=rows, tasks_by_uuid=tasks,
                    actual_free_static=selected_mode == 'shared')
                choices.append((min(r['remaining_safety_margin_bytes'] for r in budget), pair))
            except ValueError:
                continue
        if not choices:
            raise ValueError('INSUFFICIENT_GPU_MEMORY: no admissible GPU pair in ' + selected_mode)
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
        gpus, args.model, args.tp, 4096, args.mem_fraction_static,
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
        "4096",
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
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="same serving flags with protection disabled",
    )
    parser.add_argument(
        "--gpus", required=True,
        help="physical nvidia-smi GPU indices; overrides CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--socket", type=Path, required=True)
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
    parser.add_argument("--mem-fraction-static", type=float, default=0.60)
    parser.add_argument('--gpu-admission-mode', choices=('strict', 'shared'), default=None)
    parser.add_argument(
        "--allow-shared-gpu-debug", action="store_true",
        help="legacy alias for shared admission when --gpu-admission-mode is omitted",
    )
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if (
        len(gpus) != args.tp
        or len(set(gpus)) != args.tp
        or not all(g.isdigit() for g in gpus)
        or not args.socket.is_absolute()
        or args.capacity < 1
        or not args.run_id
        or len(args.run_id) > 96
        or not 0.05 <= args.mem_fraction_static <= 0.95
        or args.prefill_chunk_size not in (-1, 256, 512, 768, 1024)
    ):
        parser.error(
            "invalid GPU selection, socket, capacity, run id or memory/prefill limits"
        )
    try:
        check_gpu_resources(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    check_port(args.port)
    if not args.baseline:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            sock.connect(str(args.socket))
            send_message(sock, {"kind": "hello"}, b"")
            ready, _ = recv_message(sock)
        if ready.get("state") != "READY" or ready.get("run_id") != args.run_id:
            parser.error("online manager is not ready for this run id")
    identity = source_identity(args.model) if not args.baseline else {}
    check_gpu_resources(args)  # resource competition can change during hashing/handshake
    # The resource policy is passed explicitly to execve, never exported to
    # the caller's shell or persisted in the Conda/system environment.
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SGLANG_KV_EC_ONLINE="0" if args.baseline else "1",
        SGLANG_KV_EC_ONLINE_SOCKET=str(args.socket),
        SGLANG_KV_EC_ONLINE_RUN_ID=args.run_id,
        SGLANG_KV_EC_DEBUG="0",
        SGLANG_KV_EC_PAUSE_RESUME="0",
        SGLANG_KV_EC_ONLINE_CAPACITY=str(args.capacity),
        SGLANG_KV_EC_PREPARED_IDENTITY=json.dumps(identity),
    )
    os.execve(sys.executable, server_argv(args), dict(os.environ, **environment(args_mode(args))))


if __name__ == "__main__":
    main()
