"""Offline GPU scratch restore for real KV-EC rank snapshots.

This is an evidence/benchmark tool. It never imports SGLang, attaches to a
server, or writes an online KV pool. Restored tensors are temporary allocations
owned by this process and are released when each measurement finishes.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

if __package__:
    from .rank_snapshot import assemble, validate_payload
    from .xor_codec import encode, recover
    from .snapshot import digest
else:
    from rank_snapshot import assemble, validate_payload
    from xor_codec import encode, recover
    from snapshot import digest


def _dtype(name):
    value = getattr(torch, name.removeprefix("torch."), None)
    if value not in (torch.float16, torch.bfloat16):
        raise ValueError("only BF16/FP16 snapshots are supported")
    return value


def load_real_snapshot(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("manifest.json is required")
    manifest = json.loads(manifest_path.read_text())
    loaded_manifest, blocks = assemble(directory)
    # Assemble validates every rank sidecar and payload. Also compare the
    # persisted manifest so a stale/mixed directory cannot be benchmarked.
    for key in ("format", "snapshot_id", "request_id", "tp_size", "layers",
                "token_start", "token_count", "dtype", "packed_byte_count",
                "source_commit", "model_sha256"):
        if manifest.get(key) != loaded_manifest.get(key):
            raise ValueError(f"manifest mismatch: {key}")
    if manifest.get("provenance") != "sglang_request":
        raise ValueError("GPU restore requires real sglang_request provenance")
    if manifest.get("snapshot_completion_status") != "complete":
        raise ValueError("snapshot is not complete")
    if manifest["tp_size"] != 4:
        raise ValueError("this experiment requires TP=4")
    return manifest, blocks


def split_layer_payload(payload, record):
    """Return K/V byte views using only sidecar offsets and byte counts."""
    k0, kn = int(record["k_offset"]), int(record["k_nbytes"])
    v0, vn = int(record["v_offset"]), int(record["v_nbytes"])
    k = payload[k0:k0 + kn]
    v = payload[v0:v0 + vn]
    if k.size != kn or v.size != vn:
        raise ValueError("truncated layer payload")
    return k, v


def restore_rank_to_gpu(payload, metadata, device):
    """Restore one packed rank as temporary BF16/FP16 K/V tensors."""
    dtype = _dtype(metadata["dtype"])
    restored = []
    h2d_ms = 0.0
    verify_ms = 0.0
    for record in metadata["layer_records"]:
        k_bytes, v_bytes = split_layer_payload(payload, record)
        entries = []
        for kind, raw in (("k", k_bytes), ("v", v_bytes)):
            shape = tuple(record["packed_shape"][kind])
            expected = int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
            if raw.size != expected:
                raise ValueError(f"{kind} byte size does not match BF16/FP16 shape")
            # Allocate the final temporary tensor, then copy raw bytes through
            # a byte view. No float conversion or numerical reinterpretation.
            started = time.perf_counter_ns()
            tensor = torch.empty(shape, dtype=dtype, device=device)
            destination = tensor.view(torch.uint8).reshape(-1)
            source = torch.from_numpy(np.ascontiguousarray(raw))
            destination.copy_(source, non_blocking=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_ms += (time.perf_counter_ns() - started) / 1e6
            started = time.perf_counter_ns()
            actual = tensor.view(torch.uint8).detach().cpu().numpy().reshape(-1)
            if not np.array_equal(actual, raw):
                raise AssertionError(f"GPU byte verification failed for layer {record['layer_id']} {kind}")
            verify_ms += (time.perf_counter_ns() - started) / 1e6
            entries.append(tensor)
        restored.append((entries[0], entries[1]))
    return restored, h2d_ms, verify_ms


def one_restore(manifest, blocks, lost_rank, device):
    parity_started = time.perf_counter_ns()
    parity = encode(blocks)
    xor_ms = (time.perf_counter_ns() - parity_started) / 1e6
    survivors = [None if i == lost_rank else block for i, block in enumerate(blocks)]
    started = time.perf_counter_ns()
    restored = recover(survivors, parity)
    cpu_recovery_ms = (time.perf_counter_ns() - started) / 1e6
    expected = blocks[lost_rank]
    if digest(restored) != digest(expected) or not np.array_equal(restored, expected):
        raise AssertionError(f"CPU recovery mismatch for lost rank {lost_rank}")
    metadata = manifest["ranks"][lost_rank]
    # Validate before allocating GPU scratch tensors; this also checks every K/V
    # checksum and all shape/stride metadata for the restored rank.
    validate_payload(restored, metadata)
    tensors, h2d_ms, gpu_verification_ms = restore_rank_to_gpu(restored, metadata, device)
    # Keep tensors alive until all checks for this lost rank are complete.
    del tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"lost_rank": lost_rank, "byte_exact": True, "xor_ms": xor_ms,
            "cpu_recovery_ms": cpu_recovery_ms, "h2d_restore_ms": h2d_ms,
            "gpu_verification_ms": gpu_verification_ms,
            "restored_bytes": int(restored.size)}


def benchmark(directory, device="cuda", warmup=1, repeat=5):
    manifest, blocks = load_real_snapshot(directory)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not accessible; run this tool on the GPU server")
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")
    for _ in range(warmup):
        for lost in range(4):
            one_restore(manifest, blocks, lost, device)
    samples = {rank: [] for rank in range(4)}
    for _ in range(repeat):
        for lost in range(4):
            samples[lost].append(one_restore(manifest, blocks, lost, device))
    def summary(key, rank):
        values = [item[key] for item in samples[rank]]
        return {"median_ms": statistics.median(values),
                "p95_ms": float(np.percentile(values, 95)), "samples_ms": values}
    ranks = []
    for lost in range(4):
        ranks.append({"lost_rank": lost, "byte_exact": True,
                      "xor": summary("xor_ms", lost),
                      "cpu_recovery": summary("cpu_recovery_ms", lost),
                      "h2d_restore": summary("h2d_restore_ms", lost),
                      "gpu_verification": summary("gpu_verification_ms", lost),
                      "restored_bytes": samples[lost][0]["restored_bytes"]})
    return {"tool": "gpu_scratch_restore", "snapshot_dir": str(Path(directory).resolve()),
            "snapshot_id": manifest["snapshot_id"], "tp": 4, "layers": manifest["layers"],
            "token_count": manifest["token_count"], "dtype": manifest["dtype"],
            "device": str(device), "warmup": warmup, "repeat": repeat,
            "provenance": manifest["provenance"], "online_kv_pool_modified": False,
            "ranks": ranks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.snapshot_dir, args.device, args.warmup, args.repeat)
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
