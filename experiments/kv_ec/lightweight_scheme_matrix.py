"""Lightweight five-round EC scheme/fault matrix on authentic TP=2 KV bytes.

The target CUDA allocation is persistent and all protection data are prepared
before each logical fault.  Node loss means that both node-local data and local
protection are unavailable; it is not a physical server outage or RTO test.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .hot_recovery_pipeline import HotCudaRecovery, incremental_parity
from .multinode import rs_codec


SCHEMES = ("noec", "local-replication", "xor", "rs-only", "xor-rs", "remote-replication")
FAULTS = ("gpu-rank-kv-erasure", "logical-node-kv-unavailable")


def digest(value):
    return hashlib.sha256(memoryview(value)).hexdigest()


def coverage(scheme, fault):
    if fault == "gpu-rank-kv-erasure":
        return scheme in ("local-replication", "xor", "xor-rs")
    return scheme in ("rs-only", "xor-rs", "remote-replication")


def stat(values):
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank0", type=Path, required=True)
    p.add_argument("--rank1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--chunk-mib", type=int, default=4)
    a = p.parse_args()
    if a.output.exists() or a.rounds not in (1, 5) or a.chunk_mib <= 0:
        p.error("new output, one/five rounds and positive chunk size required")
    a.output.mkdir(parents=True)
    d0 = np.memmap(a.rank0, mode="r", dtype=np.uint8)
    d1 = np.memmap(a.rank1, mode="r", dtype=np.uint8)
    if len(d0) != len(d1) or len(d0) == 0:
        raise ValueError("equal_authentic_tp2_rank_payloads_required")
    for data in (d0, d1):
        _ = int(data[::4096].sum(dtype=np.uint64))
    hashes = (digest(d0), digest(d1))
    chunk = a.chunk_mib * 1024**2
    protect_started = time.perf_counter_ns()
    xor_parity, xor_stats = incremental_parity(d0, d1, chunk_bytes=chunk)
    xor_protect_ms = (time.perf_counter_ns() - protect_started) / 1e6
    protect_started = time.perf_counter_ns()
    _, _, rs_parity = rs_codec.encode(d0, d1)
    rs_protect_ms = (time.perf_counter_ns() - protect_started) / 1e6
    replicas = (np.asarray(d0).copy(), np.asarray(d1).copy())
    zeros = np.zeros(len(d0), dtype=np.uint8)
    hot = HotCudaRecovery(len(d0), device=a.device, chunk_bytes=chunk, depth=2)
    records = []
    for fault in FAULTS:
        for scheme in SCHEMES:
            for lost in (0, 1):
                for repeat in range(a.rounds):
                    trial = {"scheme": scheme, "fault": fault, "lost": lost,
                             "repeat": repeat, "shared_nonisolated": True,
                             "physical_gpu_failure": False,
                             "physical_node_failure": False}
                    if not coverage(scheme, fault):
                        trial.update(status="EXPECTED_NOT_COVERED", recovery_success=False,
                                     expected_outcome=True, fault_to_kv_ready_ms=None,
                                     reason="scheme_fault_model_mismatch")
                    else:
                        hot.clear_before_fault()
                        started = time.perf_counter_ns()
                        rs_decode_ms = None
                        if fault == "gpu-rank-kv-erasure":
                            if scheme == "local-replication":
                                left, right = zeros, replicas[lost]
                            else:
                                left, right = xor_parity, d1 if lost == 0 else d0
                        elif scheme == "remote-replication":
                            # Remote transfer is deliberately excluded here;
                            # E4 supplies the separately measured RDMA/TCP cost.
                            left, right = zeros, replicas[lost]
                        else:
                            rs_started = time.perf_counter_ns()
                            r0, r1 = rs_codec.recover(
                                data0=None if lost == 0 else d0,
                                data1=None if lost == 1 else d1,
                                parity=rs_parity, lengths=[len(d0), len(d1)])
                            recovered = r0 if lost == 0 else r1
                            rs_decode_ms = (time.perf_counter_ns() - rs_started) / 1e6
                            left, right = zeros, recovered
                        gpu = hot.recover(left, right, expected_sha256=hashes[lost])
                        elapsed = (time.perf_counter_ns() - started) / 1e6
                        trial.update(status="PASS", recovery_success=True,
                                     expected_outcome=True, byte_exact=True,
                                     fault_to_kv_ready_ms=elapsed,
                                     rs_decode_ms=rs_decode_ms,
                                     gpu_pipeline_ms=gpu["gpu_pipeline_ms"],
                                     first_chunk_gpu_ms=gpu["first_chunk_gpu_ms"],
                                     remote_transfer_included=False,
                                     model_scheduler_sse_included=False)
                    records.append(trial)
                    write_json(a.output / (f"{fault}-{scheme}-lost{lost}-r{repeat}.json"), trial)
    groups = []
    for fault in FAULTS:
        for scheme in SCHEMES:
            selected = [r for r in records if r["fault"] == fault and r["scheme"] == scheme]
            successful = [r for r in selected if r["recovery_success"]]
            groups.append({"fault": fault, "scheme": scheme,
                           "cases": len(selected), "recovered": len(successful),
                           "expected_not_covered": len(selected) - len(successful),
                           "expected_outcome_checks": sum(r["expected_outcome"] for r in selected),
                           "fault_to_kv_ready_ms": stat([r["fault_to_kv_ready_ms"] for r in successful]) if successful else None})
    original = 2 * len(d0)
    redundancy = {
        "noec": 0,
        "local-replication": original,
        "xor": len(xor_parity),
        "rs-only": len(rs_parity),
        "xor-rs": len(xor_parity) + len(rs_parity),
        "remote-replication": original,
    }
    result = {"status": "PASS", "rounds": a.rounds, "payload_bytes_per_shard": len(d0),
              "source_sha256": hashes, "records": records, "groups": groups,
              "all_expected_outcomes_correct": all(r["expected_outcome"] for r in records),
              "successful_gpu_recoveries": sum(r["recovery_success"] for r in records),
              "expected_not_covered_cases": sum(not r["recovery_success"] for r in records),
              "protection": {"xor_ms": xor_protect_ms, "rs_ms": rs_protect_ms,
                             "xor_first_chunk_ms": xor_stats["first_protected_chunk_ms"]},
              "redundancy_bytes": redundancy,
              "redundancy_ratio": {k: v / original for k, v in redundancy.items()},
              "limitations": ["logical byte erasure, not physical device failure",
                              "node fault is logical unavailability, not independent third-domain outage",
                              "remote replication transfer excluded from recovery clock",
                              "no model/Scheduler/SSE in this matrix", "shared GPU"]}
    write_json(a.output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "successful_gpu_recoveries", "expected_not_covered_cases", "protection", "groups")}, indent=2))


if __name__ == "__main__":
    main()
