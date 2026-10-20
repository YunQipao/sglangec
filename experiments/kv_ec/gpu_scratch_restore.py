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
        timing_files = sorted(directory.glob("rank_*.timing.json"))
        binary_files = sorted(directory.glob("rank_*.bin"))
        if timing_files and not binary_files:
            raise ValueError(
                "control-plane evidence directory has timing/diagnostic JSON "
                "but no rank Snapshot payload; FAST_PARITY intentionally does "
                "not persist rank_*.bin or manifest.json. Run an explicit "
                "DEBUG_SAFE/capture job that writes complete rank files."
            )
        raise ValueError(
            f"real Snapshot directory requires manifest.json and rank_*.bin/json: "
            f"{directory}"
        )
    manifest = json.loads(manifest_path.read_text())
    loaded_manifest, blocks = assemble(directory)
    # Assemble validates every rank sidecar and payload. Also compare the
    # persisted manifest so a stale/mixed directory cannot be benchmarked.
    for key in loaded_manifest:
        if manifest.get(key) != loaded_manifest.get(key):
            raise ValueError(f"manifest mismatch: {key}")
    if manifest.get("provenance") != "sglang_request":
        raise ValueError("GPU restore requires real sglang_request provenance")
    if manifest.get("snapshot_completion_status") != "complete":
        raise ValueError("snapshot is not complete")
    if manifest["tp_size"] not in (2, 4):
        raise ValueError("this experiment requires TP=2/4")
    return manifest, blocks


def validate_snapshot_sequence(manifests, *, expected_chunks=3, chunk_tokens=256):
    """Validate contiguous same-request chunks without retaining KV bytes."""
    if len(manifests) != expected_chunks:
        raise ValueError(f"expected {expected_chunks} snapshots")
    first = manifests[0]
    required = ("request_id", "request_id_encoding", "model_identifier",
                "model_sha256", "source_commit", "core_diff_sha256", "tp_size",
                "layers", "dtype", "provenance", "packed_byte_count")
    for key in required:
        if key not in first:
            raise ValueError(f"missing sequence metadata: {key}")
    if first["tp_size"] != 4 or first["layers"] != list(range(36)):
        raise ValueError("sequence experiment requires TP4 and all 36 layers")
    next_start = 0
    seen = set()
    for index, manifest in enumerate(manifests):
        if int(manifest["token_start"]) != next_start:
            raise ValueError("snapshot token ranges are not contiguous")
        if int(manifest["token_count"]) != chunk_tokens:
            raise ValueError("snapshot token count does not match chunk size")
        if manifest["provenance"] != "sglang_request":
            raise ValueError("GPU restore requires real sglang_request provenance")
        for key in required:
            if manifest[key] != first[key]:
                raise ValueError(f"snapshot sequence mismatch: {key}")
        next_start += chunk_tokens
        if manifest["snapshot_id"] in seen:
            raise ValueError("snapshot IDs must be unique per chunk")
        seen.add(manifest["snapshot_id"])
        if "ranks" in manifest:
            for rank, metadata in enumerate(manifest["ranks"]):
                reference = first["ranks"][rank]
                for key in ("request_version", "checkpoint_version", "manager_epoch",
                            "process_epoch", "req_pool_idx", "req_generation"):
                    if metadata.get(key) != reference.get(key):
                        raise ValueError(f"snapshot rank identity mismatch: {key}")
                shapes = [record["packed_shape"] for record in metadata["layer_records"]]
                expected_shapes = [
                    {"k": [chunk_tokens, 2, 128], "v": [chunk_tokens, 2, 128]}
                    for _ in range(36)
                ]
                if shapes != expected_shapes:
                    raise ValueError("sequence requires Qwen3-8B TP4 NHD K/V layout")
    return {
        "chunk_count": len(manifests),
        "token_start": int(first["token_start"]),
        "token_count": next_start - int(first["token_start"]),
        "tp_size": int(first["tp_size"]),
        "layers": list(first["layers"]),
        "request_id": first["request_id"],
    }


def benchmark_sequence(directories, device="cuda", warmup=1, repeat=5):
    """Restore three real chunks one at a time into isolated GPU scratch."""
    directories = [Path(directory) for directory in directories]
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("sequence restore requires CUDA scratch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not accessible; run this tool on the GPU server")

    manifests = []
    for directory in directories:
        manifest, blocks = load_real_snapshot(directory)
        manifests.append(manifest)
        del blocks
    coverage = validate_snapshot_sequence(manifests)
    chunk_results = []
    for index, directory in enumerate(directories):
        manifest, blocks = load_real_snapshot(directory)
        for _ in range(warmup):
            for lost in range(manifest["tp_size"]):
                one_restore(manifest, blocks, lost, device)
        samples = {rank: [] for rank in range(manifest["tp_size"])}
        for _ in range(repeat):
            for lost in range(manifest["tp_size"]):
                samples[lost].append(one_restore(manifest, blocks, lost, device))
        chunk_results.append({
            "index": index,
            "snapshot_dir": str(directory.resolve()),
            "snapshot_id": manifest["snapshot_id"],
            "token_start": manifest["token_start"],
            "token_count": manifest["token_count"],
            "rank_sha256": [rank["sha256"] for rank in manifest["ranks"]],
            "ranks": [
                {
                    "lost_rank": lost,
                    "byte_exact": all(item["byte_exact"] for item in samples[lost]),
                    "xor_ms": _summary(samples[lost], "xor_ms"),
                    "cpu_recovery_ms": _summary(samples[lost], "cpu_recovery_ms"),
                    "h2d_restore_ms": _summary(samples[lost], "h2d_restore_ms"),
                    "gpu_verification_ms": _summary(
                        samples[lost], "gpu_verification_ms"
                    ),
                    "restored_bytes": samples[lost][0]["restored_bytes"],
                }
                for lost in range(manifest["tp_size"])
            ],
        })
        del blocks
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "tool": "gpu_scratch_restore_sequence",
        "chunks": chunk_results,
        "coverage": coverage,
        "device": str(device),
        "warmup": warmup,
        "repeat": repeat,
        "online_kv_pool_modified": False,
        "status": "byte_exact",
        "erasure_cases": sum(len(chunk["ranks"]) for chunk in chunk_results),
    }


def _summary(samples, key):
    values = [item[key] for item in samples]
    return {
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
        "samples_ms": values,
    }


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
    if not 0 <= lost_rank < manifest["tp_size"] or len(blocks) != manifest["tp_size"]:
        raise ValueError("invalid lost rank or incomplete TP blocks")
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
    tp = manifest["tp_size"]
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not accessible; run this tool on the GPU server")
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")
    for _ in range(warmup):
        for lost in range(tp):
            one_restore(manifest, blocks, lost, device)
    samples = {rank: [] for rank in range(tp)}
    for _ in range(repeat):
        for lost in range(tp):
            samples[lost].append(one_restore(manifest, blocks, lost, device))
    ranks = []
    for lost in range(tp):
        ranks.append({"lost_rank": lost, "byte_exact": True,
                      "xor": _summary(samples[lost], "xor_ms"),
                      "cpu_recovery": _summary(samples[lost], "cpu_recovery_ms"),
                      "h2d_restore": _summary(samples[lost], "h2d_restore_ms"),
                      "gpu_verification": _summary(
                          samples[lost], "gpu_verification_ms"
                      ),
                      "restored_bytes": samples[lost][0]["restored_bytes"]})
    return {"tool": "gpu_scratch_restore", "snapshot_dir": str(Path(directory).resolve()),
            "snapshot_id": manifest["snapshot_id"], "tp": tp, "layers": manifest["layers"],
            "token_count": manifest["token_count"], "dtype": manifest["dtype"],
            "device": str(device), "warmup": warmup, "repeat": repeat,
            "provenance": manifest["provenance"], "online_kv_pool_modified": False,
            "ranks": ranks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if len(args.snapshot_dir) == 1:
        result = benchmark(args.snapshot_dir[0], args.device, args.warmup, args.repeat)
    else:
        result = benchmark_sequence(
            args.snapshot_dir, args.device, args.warmup, args.repeat
        )
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
