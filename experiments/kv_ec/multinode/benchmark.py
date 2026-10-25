"""CPU-only local RS codec microbenchmark at the requested block sizes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from .rs_codec import encode, recover

DEFAULT_SIZES = (1, 4, 16, 64)


def run(sizes, output, iterations=3):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for size_mib in sizes:
        size = size_mib * 1024 * 1024
        rng = np.random.default_rng(100 + size_mib)
        a = rng.integers(0, 256, size, dtype=np.uint8)
        b = rng.integers(0, 256, size, dtype=np.uint8)
        enc, dec = [], []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            _, _, p = encode(a, b)
            enc.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            got, _ = recover(data0=None, data1=b, parity=p, lengths=(size, size))
            dec.append((time.perf_counter_ns() - start) / 1e6)
            if got.tobytes() != a.tobytes():
                raise ValueError("codec benchmark correctness mismatch")
        rows.append({"transport": "cpu_only", "block_bytes": size,
                     "rs_encode_ms": sorted(enc)[len(enc)//2],
                     "rs_decode_ms": sorted(dec)[len(dec)//2],
                     "transfer_ms": None, "effective_network_gbps": None,
                     "protection_completion_ms": None, "recovery_ms": sorted(dec)[len(dec)//2],
                     "total_transferred_bytes": 0, "cpu_percent": None,
                     "host_dram_bytes": int(a.nbytes + b.nbytes + p.nbytes),
                     "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "sha256_D0": hashlib.sha256(a).hexdigest(), "state": "PASS"})
    (output / "benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes-mib", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.iterations < 1 or any(size <= 0 for size in args.sizes_mib):
        p.error("sizes and iterations must be positive")
    print(json.dumps(run(args.sizes_mib, args.output.resolve(), args.iterations), indent=2))


if __name__ == "__main__":
    main()
