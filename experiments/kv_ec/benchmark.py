"""Synthetic TP-shaped blocks, NOT a running TP model or node-failure experiment.

CUDA mode copies one block from each selected device sequentially into pinned DRAM.
CPU mode leaves d2h_ms empty: it is not a GPU transfer measurement.
"""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

if __package__:
    from .xor_codec import encode, recover
else:
    from xor_codec import encode, recover

BLOCK_SIZES = (64 * 1024, 256 * 1024, 1024**2, 4 * 1024**2, 16 * 1024**2)


def measured(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--devices", default="0,1,2,3", help="visible CUDA ordinals")
    parser.add_argument("--tp", type=int, choices=(2, 4), nargs="+", default=[2, 4])
    parser.add_argument("--block-bytes", type=int, nargs="+", default=list(BLOCK_SIZES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or min(args.block_bytes) < 1:
        parser.error("invalid warmup, iterations or block length")
    devices = [int(i) for i in args.devices.split(",")]
    if args.source == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is not accessible; no CUDA results will be generated")
        if len(devices) < max(args.tp) or len(set(devices)) != len(devices):
            parser.error("select a distinct visible CUDA device for each shard")
        if any(i < 0 or i >= torch.cuda.device_count() for i in devices):
            parser.error("invalid visible CUDA ordinal")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite existing experiments. Companion metadata is reserved too.
    meta_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or meta_path.exists():
        parser.error("output or metadata already exists; choose a new output path")
    metadata = {
        "kind": "synthetic_" + args.source,
        "real_sglang_kv": False,
        "node_failure_test": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "cpu_threads": args.cpu_threads,
        "utc": datetime.now(timezone.utc).isoformat(),
        "timing": "wall clock; excludes allocation, H2D setup, validation; includes output allocation for XOR/recovery",
        "d2h": "sum wall time of sequential blocking copies of N blocks to pinned CPU DRAM" if args.source == "cuda" else "not measured",
        "devices": [
            {"ordinal": i, "name": torch.cuda.get_device_name(i)} for i in devices
        ] if args.source == "cuda" else [],
    }
    try:
        metadata["sglang_installed_version"] = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        metadata["sglang_installed_version"] = None
    with meta_path.open("x") as f:
        json.dump(metadata, f, indent=2)
    columns = ["tp", "block_bytes", "d2h_ms", "xor_ms", "recovery_ms", "correct",
               "source", "iteration", "lost_rank", "total_d2h_bytes"]
    rng = np.random.default_rng(args.seed)
    with args.output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for tp in args.tp:
            for length in args.block_bytes:
                original = [rng.integers(0, 256, length, dtype=np.uint8) for _ in range(tp)]
                # Keep the original only as a validation oracle, never pass a lost block
                # into recover(). Random bytes include arbitrary FP16/BF16 bit patterns.
                if args.source == "cuda":
                    gpu_blocks = [torch.from_numpy(b).to(f"cuda:{d}")
                                  for b, d in zip(original, devices)]
                    host_blocks = [torch.empty(length, dtype=torch.uint8, pin_memory=True)
                                   for _ in range(tp)]
                    for d in devices[:tp]:
                        torch.cuda.synchronize(d)
                    blocks = [b.numpy() for b in host_blocks]
                else:
                    blocks = [b.copy() for b in original]
                for iteration in range(-args.warmup, args.iterations):
                    d2h_ms = ""
                    if args.source == "cuda":
                        def transfer():
                            for cpu, gpu in zip(host_blocks, gpu_blocks):
                                cpu.copy_(gpu, non_blocking=False)
                        _, d2h_ms = measured(transfer)
                    parity, xor_ms = measured(lambda: encode(blocks))
                    lost = iteration % tp
                    survivors = [None if i == lost else b for i, b in enumerate(blocks)]
                    restored, recovery_ms = measured(lambda: recover(survivors, parity))
                    correct = np.array_equal(restored, original[lost]) and all(
                        np.array_equal(b, expected) for b, expected in zip(blocks, original)
                    )
                    if not correct:
                        raise AssertionError(f"byte mismatch at tp={tp}, bytes={length}, lost={lost}")
                    if iteration >= 0:
                        writer.writerow(dict(tp=tp, block_bytes=length, d2h_ms=d2h_ms,
                                             xor_ms=xor_ms, recovery_ms=recovery_ms,
                                             correct=correct, source="synthetic_" + args.source,
                                             iteration=iteration, lost_rank=lost,
                                             total_d2h_bytes=tp * length if args.source == "cuda" else 0))
                f.flush()
                print(f"PASS synthetic_{args.source}: tp={tp}, block_bytes={length}, measured={args.iterations}", flush=True)
                if args.source == "cuda":
                    del gpu_blocks, host_blocks
    print(f"CSV: {args.output}; metadata: {meta_path}")


if __name__ == "__main__":
    main()
