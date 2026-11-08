"""Two-GPU Shared-mode TP=2 logical-rank-loss hot recovery acceptance."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .hot_recovery_pipeline import HotCudaRecovery, incremental_parity


def edge_bytes(tensor, count):
    return (tensor[:count].cpu().numpy().tobytes(), tensor[-count:].cpu().numpy().tobytes())


def stats(values):
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values), "max": max(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank0", type=Path, required=True)
    p.add_argument("--rank1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-mib", type=int, default=4)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--devices", default="cuda:0,cuda:1")
    a = p.parse_args()
    devices = a.devices.split(",")
    if (a.output.exists() or a.rounds < 1 or a.chunk_mib < 1
            or len(devices) != 2 or devices[0] == devices[1]):
        p.error("new output, two devices, positive rounds/chunk required")
    a.output.mkdir(parents=True)
    ranks = [np.memmap(a.rank0, mode="r", dtype=np.uint8),
             np.memmap(a.rank1, mode="r", dtype=np.uint8)]
    if len(ranks[0]) != len(ranks[1]):
        raise ValueError("tp2_rank_size_mismatch")
    total = len(ranks[0])
    chunk = a.chunk_mib * 1024**2
    parity, protection = incremental_parity(*ranks, chunk_bytes=chunk)
    hashes = [hashlib.sha256(x).hexdigest() for x in ranks]
    # Both TP ranks exist on distinct physical CUDA devices for the entire run.
    resident = []
    for rank, device in enumerate(devices):
        value = torch.from_numpy(np.asarray(ranks[rank])).to(torch.device(device))
        resident.append(value)
    torch.cuda.synchronize()
    edge = min(1024**2, total)
    original_edges = [edge_bytes(x, edge) for x in resident]
    records = []
    for lost in (0, 1):
        survivor = 1 - lost
        hot = HotCudaRecovery(total, device=devices[lost], chunk_bytes=chunk, depth=2)
        # Use the persistent target as the actual lost-rank allocation. The
        # survivor allocation remains resident and is never a recovery target.
        del resident[lost]
        resident.insert(lost, hot.target)
        hot.target.copy_(torch.from_numpy(np.asarray(ranks[lost])))
        torch.cuda.synchronize(torch.device(devices[lost]))
        for repeat in range(a.rounds):
            survivor_before = edge_bytes(resident[survivor], edge)
            hot.clear_before_fault()
            record = hot.recover(parity, ranks[survivor],
                                 expected_sha256=hashes[lost],
                                 full_verify=(repeat == 0))
            survivor_after = edge_bytes(resident[survivor], edge)
            if survivor_before != survivor_after or survivor_after != original_edges[survivor]:
                raise ValueError("survivor_gpu_changed_during_peer_recovery")
            if edge_bytes(hot.target, edge) != original_edges[lost]:
                raise ValueError("lost_rank_gpu_edge_restore_mismatch")
            record.update(repeat=repeat, lost_rank=lost, survivor_rank=survivor,
                          tp_size=2, shared_mode=True, distinct_cuda_devices=True,
                          survivor_gpu_unchanged=True, byte_exact=True)
            records.append(record)
            (a.output / f"lost{lost}-trial{repeat}.json").write_text(json.dumps(record, indent=2))
    groups = []
    for lost in (0, 1):
        rows = [r for r in records if r["lost_rank"] == lost]
        groups.append({"lost_rank": lost, "valid_rounds": len(rows),
                       "fault_to_kv_ready_ms": stats([r["fault_to_kv_ready_ms"] for r in rows]),
                       "first_chunk_gpu_ms": stats([r["first_chunk_gpu_ms"] for r in rows])})
    result = {"status": "PASS", "tp_size": 2, "shared_mode": True,
              "physical_gpu_count": 2, "devices": devices,
              "rank_bytes": total, "chunk_mib": a.chunk_mib,
              "valid_trials": len(records), "rounds_per_loss_direction": a.rounds,
              "all_byte_exact": all(r["byte_exact"] for r in records),
              "all_survivor_gpu_unchanged": all(r["survivor_gpu_unchanged"] for r in records),
              "protection": protection, "summary": groups, "records": records,
              "scope": "two physical GPUs, TP=2 data path; no loaded model/Scheduler/SSE"}
    (a.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
