"""Persistent, in-memory, chunk-pipelined TP=2 KV recovery benchmark.

This is a data-path acceptance tool, not an automatic SGLang failover claim.
It keeps the target CUDA allocation alive across faults, prepares parity before
the fault boundary, and excludes file I/O and process/model startup from the
recovery clock.  CPU XOR, pinned-host staging and asynchronous H2D are
overlapped with a bounded ring of buffers.  Every formal result is checked by
a fused host SHA-256 plus GPU edge samples; preflight performs a full GPU
readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch


MIB = 1024 * 1024


def sha256_bytes(value) -> str:
    return hashlib.sha256(memoryview(value)).hexdigest()


def validate_inputs(rank0, rank1) -> int:
    if len(rank0) <= 0 or len(rank0) != len(rank1):
        raise ValueError("tp2_equal_nonempty_rank_payloads_required")
    return len(rank0)


def incremental_parity(rank0, rank1, *, chunk_bytes: int):
    """Build stable parity incrementally; never create a second full output."""
    total = validate_inputs(rank0, rank1)
    if chunk_bytes <= 0:
        raise ValueError("positive_chunk_bytes_required")
    parity = np.empty(total, dtype=np.uint8)
    started = time.perf_counter_ns()
    first_ns = None
    chunks = 0
    for offset in range(0, total, chunk_bytes):
        end = min(total, offset + chunk_bytes)
        np.bitwise_xor(rank0[offset:end], rank1[offset:end], out=parity[offset:end])
        chunks += 1
        if first_ns is None:
            first_ns = time.perf_counter_ns() - started
    elapsed = time.perf_counter_ns() - started
    return parity, {
        "protection_ms": elapsed / 1e6,
        "first_protected_chunk_ms": first_ns / 1e6,
        "chunk_count": chunks,
        "stable_parity_bytes": parity.nbytes,
        "temporary_full_payload_copies": 0,
    }


def update_parity(parity, old_chunk, new_chunk, *, offset: int):
    """P' = P xor old xor new; touches only the dirty token/chunk range."""
    if (offset < 0 or len(old_chunk) != len(new_chunk)
            or offset + len(old_chunk) > len(parity)):
        raise ValueError("invalid_incremental_update_range")
    view = parity[offset:offset + len(old_chunk)]
    np.bitwise_xor(view, old_chunk, out=view)
    np.bitwise_xor(view, new_chunk, out=view)


def recover_cpu(parity, survivor, *, chunk_bytes: int) -> tuple[bytes, dict]:
    total = validate_inputs(parity, survivor)
    output = np.empty(total, dtype=np.uint8)
    started = time.perf_counter_ns()
    first_ns = None
    for offset in range(0, total, chunk_bytes):
        end = min(total, offset + chunk_bytes)
        np.bitwise_xor(parity[offset:end], survivor[offset:end], out=output[offset:end])
        if first_ns is None:
            first_ns = time.perf_counter_ns() - started
    elapsed = time.perf_counter_ns() - started
    return output.tobytes(), {"recovery_ms": elapsed / 1e6,
                              "first_chunk_ms": first_ns / 1e6}


class HotCudaRecovery:
    """One persistent target allocation with a bounded pinned-host ring."""

    def __init__(self, total_bytes: int, *, device: str, chunk_bytes: int,
                 depth: int = 2):
        if total_bytes <= 0 or chunk_bytes <= 0 or depth < 2:
            raise ValueError("invalid_hot_pipeline_configuration")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("real_cuda_required")
        self.total_bytes, self.chunk_bytes, self.depth = total_bytes, chunk_bytes, depth
        with torch.cuda.device(self.device):
            self.target = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)
            self.stream = torch.cuda.Stream(device=self.device)
            self.host = [torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
                         for _ in range(depth)]
            self.events = [torch.cuda.Event() for _ in range(depth)]
        self.used = [False] * depth

    def clear_before_fault(self):
        self.target.zero_()
        torch.cuda.synchronize(self.device)

    def recover(self, parity, survivor, *, expected_sha256: str, full_verify=False):
        total = validate_inputs(parity, survivor)
        if total != self.total_bytes:
            raise ValueError("payload_size_changed")
        digest = hashlib.sha256()
        wall_start = time.perf_counter_ns()
        first_enqueued_ns = None
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_first = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record(self.stream)
            chunks = 0
            for offset in range(0, total, self.chunk_bytes):
                end = min(total, offset + self.chunk_bytes)
                slot = chunks % self.depth
                if self.used[slot]:
                    self.events[slot].synchronize()
                count = end - offset
                host_view = self.host[slot][:count]
                np.bitwise_xor(parity[offset:end], survivor[offset:end],
                               out=host_view.numpy())
                digest.update(host_view.numpy())
                self.target[offset:end].copy_(host_view, non_blocking=True)
                self.events[slot].record(self.stream)
                self.used[slot] = True
                chunks += 1
                if first_enqueued_ns is None:
                    cuda_first.record(self.stream)
                    first_enqueued_ns = time.perf_counter_ns() - wall_start
            cuda_end.record(self.stream)
        cuda_end.synchronize()
        ready_ns = time.perf_counter_ns() - wall_start
        got_sha = digest.hexdigest()
        if got_sha != expected_sha256:
            raise ValueError("reconstructed_host_sha256_mismatch")
        sample_started = time.perf_counter_ns()
        edge = min(MIB, total)
        first = self.target[:edge].cpu().numpy().tobytes()
        last = self.target[total - edge:].cpu().numpy().tobytes()
        expected_first = np.bitwise_xor(parity[:edge], survivor[:edge]).tobytes()
        expected_last = np.bitwise_xor(parity[total-edge:], survivor[total-edge:]).tobytes()
        if first != expected_first or last != expected_last:
            raise ValueError("gpu_edge_sample_mismatch")
        sample_ms = (time.perf_counter_ns() - sample_started) / 1e6
        full_ms = None
        if full_verify:
            verify_started = time.perf_counter_ns()
            full = self.target.cpu().numpy()
            if sha256_bytes(full) != expected_sha256:
                raise ValueError("gpu_full_readback_sha256_mismatch")
            full_ms = (time.perf_counter_ns() - verify_started) / 1e6
        return {
            "fault_to_kv_ready_ms": ready_ns / 1e6,
            "first_chunk_enqueued_ms": first_enqueued_ns / 1e6,
            "first_chunk_gpu_ms": cuda_start.elapsed_time(cuda_first),
            "gpu_pipeline_ms": cuda_start.elapsed_time(cuda_end),
            "post_gpu_edge_validation_ms": sample_ms,
            "post_gpu_full_validation_ms": full_ms,
            "chunk_count": chunks,
            "chunk_bytes": self.chunk_bytes,
            "pipeline_depth": self.depth,
            "reconstructed_sha256": got_sha,
            "gpu_edge_byte_exact": True,
            "gpu_full_byte_exact": True if full_verify else None,
            "disk_io_in_fault_window": False,
            "model_or_process_startup_in_fault_window": False,
            "persistent_target_allocation": True,
        }


def whole_block_cuda(parity, survivor, *, device: str, expected_sha256: str):
    """Reference path: full CPU materialization followed by blocking H2D."""
    started = time.perf_counter_ns()
    recovered = np.bitwise_xor(parity, survivor)
    got = sha256_bytes(recovered)
    if got != expected_sha256:
        raise ValueError("whole_block_sha256_mismatch")
    target = torch.from_numpy(recovered).to(torch.device(device))
    torch.cuda.synchronize(torch.device(device))
    elapsed = (time.perf_counter_ns() - started) / 1e6
    del target
    return {"fault_to_kv_ready_ms": elapsed, "reconstructed_sha256": got,
            "full_temporary_host_bytes": recovered.nbytes,
            "model_or_process_startup_in_fault_window": False}


def summary(values):
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank0", type=Path, required=True)
    parser.add_argument("--rank1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-mib", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--lost-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pipeline-depth", type=int, default=2)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.pipeline_depth < 2 or any(x <= 0 for x in args.chunk_mib):
        parser.error("positive rounds/chunks and pipeline depth >=2 required")
    args.output.mkdir(parents=True, exist_ok=False)
    # mmap and touch before parity/protection: no filesystem access is admitted
    # between the fault boundary and KV-ready timestamp.
    r0 = np.memmap(args.rank0, mode="r", dtype=np.uint8)
    r1 = np.memmap(args.rank1, mode="r", dtype=np.uint8)
    total = validate_inputs(r0, r1)
    for view in (r0, r1):
        _ = int(view[::4096].sum(dtype=np.uint64))
    source_hashes = [sha256_bytes(r0), sha256_bytes(r1)]
    config = vars(args) | {"rank0": str(args.rank0), "rank1": str(args.rank1),
                           "output": str(args.output), "payload_bytes_per_rank": total,
                           "source_sha256": source_hashes,
                           "scope": "persistent CUDA data-path; not Scheduler/SSE failover"}
    atomic_json(args.output / "config.json", config)
    records, protections = [], []
    # Reference once; repeated cold/model runs are intentionally avoided.
    parity_ref, protection = incremental_parity(r0, r1, chunk_bytes=4 * MIB)
    protections.append(dict(chunk_mib=4, **protection,
                            parity_sha256=sha256_bytes(parity_ref)))
    survivor = r1 if args.lost_rank == 0 else r0
    expected = source_hashes[args.lost_rank]
    baseline = whole_block_cuda(parity_ref, survivor, device=args.device,
                                expected_sha256=expected)
    atomic_json(args.output / "smoke-whole-block.json", baseline)
    for chunk_mib in args.chunk_mib:
        chunk_bytes = chunk_mib * MIB
        parity, protection = incremental_parity(r0, r1, chunk_bytes=chunk_bytes)
        if not np.array_equal(parity, parity_ref):
            raise ValueError("chunk_size_changed_parity")
        protections.append(dict(chunk_mib=chunk_mib, **protection,
                                parity_sha256=sha256_bytes(parity)))
        hot = HotCudaRecovery(total, device=args.device, chunk_bytes=chunk_bytes,
                              depth=args.pipeline_depth)
        # Preflight is separate and does the expensive full GPU readback.
        hot.clear_before_fault()
        preflight = hot.recover(parity, survivor, expected_sha256=expected, full_verify=True)
        atomic_json(args.output / f"preflight-{chunk_mib}mib.json", preflight)
        for repeat in range(args.rounds):
            hot.clear_before_fault()
            record = hot.recover(parity, survivor, expected_sha256=expected)
            record.update(repeat=repeat, chunk_mib=chunk_mib, valid=True,
                          shared_nonisolated=True)
            records.append(record)
            atomic_json(args.output / f"trial-{chunk_mib}mib-{repeat}.json", record)
        del hot
        torch.cuda.empty_cache()
    grouped = []
    for chunk_mib in args.chunk_mib:
        group = [r for r in records if r["chunk_mib"] == chunk_mib]
        grouped.append({"chunk_mib": chunk_mib, "valid_rounds": len(group),
                        "fault_to_kv_ready_ms": summary([r["fault_to_kv_ready_ms"] for r in group]),
                        "first_chunk_gpu_ms": summary([r["first_chunk_gpu_ms"] for r in group]),
                        "gpu_pipeline_ms": summary([r["gpu_pipeline_ms"] for r in group])})
    result = {"status": "PASS", "valid_trials": len(records), "rounds_per_chunk": args.rounds,
              "all_byte_exact": all(r["gpu_edge_byte_exact"] for r in records),
              "baseline_whole_block": baseline, "protection": protections,
              "summary": grouped, "records": records,
              "limitations": ["no model weights are loaded by this benchmark",
                              "no Scheduler/SSE resume is claimed",
                              "fault is logical loss of one TP rank payload",
                              "shared GPU measurement"]}
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
