"""Measure bounded dirty-chunk parity updates on authentic TP=2 KV bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from .hot_recovery_pipeline import incremental_parity, update_parity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank0", type=Path, required=True)
    p.add_argument("--rank1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-mib", type=int, default=4)
    p.add_argument("--rounds", type=int, default=5)
    a = p.parse_args()
    if a.chunk_mib <= 0 or a.rounds < 1 or a.output.exists():
        p.error("positive chunk/rounds and a new output path are required")
    a.output.mkdir(parents=True)
    x = np.memmap(a.rank0, mode="r", dtype=np.uint8)
    y = np.memmap(a.rank1, mode="r", dtype=np.uint8)
    chunk = min(a.chunk_mib * 1024 * 1024, len(x))
    offset = max(0, (len(x) // 2 // chunk) * chunk)
    if offset + chunk > len(x):
        offset = len(x) - chunk
    parity, protection = incremental_parity(x, y, chunk_bytes=chunk)
    before = hashlib.sha256(parity).hexdigest()
    old = x[offset:offset + chunk].copy()
    new = old.copy()
    new ^= np.uint8(0x5A)
    records = []
    for repeat in range(a.rounds):
        started = time.perf_counter_ns()
        update_parity(parity, old, new, offset=offset)
        forward = (time.perf_counter_ns() - started) / 1e6
        expected = np.bitwise_xor(new, y[offset:offset + chunk])
        if not np.array_equal(parity[offset:offset + chunk], expected):
            raise ValueError("incremental_forward_mismatch")
        started = time.perf_counter_ns()
        update_parity(parity, new, old, offset=offset)
        reverse = (time.perf_counter_ns() - started) / 1e6
        records.append({"repeat": repeat, "dirty_bytes": chunk,
                        "forward_update_ms": forward, "rollback_update_ms": reverse,
                        "byte_exact": True})
    if hashlib.sha256(parity).hexdigest() != before:
        raise ValueError("incremental_rollback_changed_full_parity")
    values = [r["forward_update_ms"] for r in records]
    result = {"status": "PASS", "rounds": a.rounds, "dirty_bytes": chunk,
              "full_rank_bytes": len(x), "dirty_fraction": chunk / len(x),
              "full_protection_ms": protection["protection_ms"],
              "incremental_update_ms_mean": statistics.mean(values),
              "incremental_update_ms_sample_std": statistics.stdev(values) if len(values) > 1 else 0,
              "all_byte_exact": all(r["byte_exact"] for r in records),
              "records": records,
              "scope": "CPU-resident authentic KV parity update; no GPU capture/RDMA"}
    (a.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
