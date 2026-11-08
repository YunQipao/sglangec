"""Fair persistent-target whole-block XOR+H2D reference."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .hot_recovery_pipeline import incremental_parity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank0", type=Path, required=True)
    p.add_argument("--rank1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rounds", type=int, default=5)
    a = p.parse_args()
    if a.output.exists() or a.rounds < 1:
        p.error("new output and positive rounds required")
    a.output.mkdir(parents=True)
    r0 = np.memmap(a.rank0, mode="r", dtype=np.uint8)
    r1 = np.memmap(a.rank1, mode="r", dtype=np.uint8)
    parity, protection = incremental_parity(r0, r1, chunk_bytes=4 * 1024**2)
    expected = hashlib.sha256(r0).hexdigest()
    device = torch.device(a.device)
    target = torch.empty(len(r0), dtype=torch.uint8, device=device)
    # Discard CUDA/context/page warmup, matching the optimized formal path.
    warm = np.bitwise_xor(parity, r1)
    target.copy_(torch.from_numpy(warm))
    torch.cuda.synchronize(device)
    del warm
    records = []
    for repeat in range(a.rounds):
        target.zero_()
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        recovered = np.bitwise_xor(parity, r1)
        digest = hashlib.sha256(recovered).hexdigest()
        target.copy_(torch.from_numpy(recovered))
        torch.cuda.synchronize(device)
        ready = (time.perf_counter_ns() - started) / 1e6
        if digest != expected:
            raise ValueError("whole_block_host_mismatch")
        edge = min(1024**2, len(r0))
        if (target[:edge].cpu().numpy().tobytes() != r0[:edge].tobytes()
                or target[-edge:].cpu().numpy().tobytes() != r0[-edge:].tobytes()):
            raise ValueError("whole_block_gpu_edge_mismatch")
        records.append({"repeat": repeat, "fault_to_kv_ready_ms": ready,
                        "host_sha256": digest, "gpu_edge_byte_exact": True})
        del recovered
    values = [r["fault_to_kv_ready_ms"] for r in records]
    result = {"status": "PASS", "rounds": a.rounds, "all_byte_exact": True,
              "fault_to_kv_ready_ms_mean": statistics.mean(values),
              "fault_to_kv_ready_ms_sample_std": statistics.stdev(values) if len(values)>1 else 0,
              "full_temporary_host_bytes": len(r0), "persistent_target_allocation": True,
              "protection_ms": protection["protection_ms"], "records": records,
              "scope": "warm persistent CUDA allocation; whole-block CPU materialization and blocking H2D"}
    (a.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
