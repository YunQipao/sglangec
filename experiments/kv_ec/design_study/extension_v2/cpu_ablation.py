"""Real-capture CPU shard ablation, not GPU/process/physical-node failure.

The recovery adapter owns its active KV, actual complete CPU copies and parity.
Erased buffers are zeroed before references are removed. Recovery never receives
the original lost payload. The input vault is used only to initialize each new
trial; prefault hashes (not lost bytes) are used for correctness afterwards.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ... import rs_codec, xor_codec
from ...node_block_export import descriptor, pack, unpack
from ...rs_stripe_builder import build
from .evidence import RSSSampler, code_hashes, save, sha
from .cpu_host_gate import cpu_gate

MODES = ("noec", "replication", "xor", "rs-only", "xor-rs")


def gpu_idle(mode="strict_idle"):
    """Compatibility entrypoint; shared is an explicit production CPU policy."""
    return cpu_gate(mode)


class Unrecoverable(Exception):
    pass


class ShardAdapter:
    def __init__(self, nodes, mode, *, allow_fixture=False):
        if mode not in MODES:
            raise ValueError("unknown ablation mode")
        self.mode, self.allow_fixture = mode, allow_fixture
        decoded = [unpack(b, allow_fixture=allow_fixture) for b in nodes]
        self.meta = [m for m, _ in decoded]
        self.hashes = [[sha(b) for b in ranks] for _, ranks in decoded]
        self.lengths = [len(b) for b in nodes]
        self.node_hashes = [sha(b) for b in nodes]
        self.active = [[np.frombuffer(b, dtype=np.uint8).copy() for b in ranks] for _, ranks in decoded]
        self.source_bytes = sum(b.nbytes for ranks in self.active for b in ranks)
        self.tp = len(self.active[0])
        begin = time.perf_counter_ns()
        self.replicas = [[b.copy() for b in ranks] for ranks in self.active] if mode == "replication" else [None, None]
        self.local = [xor_codec.encode(ranks) for ranks in self.active] if mode in ("xor", "xor-rs") else [None, None]
        self.global_parity = rs_codec.encode(*nodes) if mode in ("rs-only", "xor-rs") else None
        self.prepare_ms = (time.perf_counter_ns()-begin)/1e6
        self.ledger = dict(source_rank_bytes=self.source_bytes,
            stable_replica_bytes=sum(b.nbytes for ranks in self.replicas if ranks for b in ranks),
            stable_local_parity_bytes=sum(p.nbytes for p in self.local if p is not None),
            stable_remote_parity_bytes=len(self.global_parity) if self.global_parity is not None else 0,
            input_node_snapshot_bytes=sum(self.lengths),
            active_working_copy_bytes=self.source_bytes,
            temporary_serialization_peak_bytes=None, measured_peak_rss_bytes=None,
            note="actual allocated arrays/bytes; source vault retained by caller for next independent trial; logical domains in one process")

    def erase(self, node, rank=None):
        if node not in (0,1) or (rank is not None and not 0 <= rank < self.tp):
            raise ValueError("invalid fault target")
        targets = list(range(self.tp)) if rank is None else [rank]
        checks = []
        for r in targets:
            block = self.active[node][r]
            before = sha(block)
            block.fill(0)
            checks.append(dict(rank=r, before_sha256=before, erased_sha256=sha(block),
                zeroed=not np.any(block), erased_bytes=block.nbytes))
            self.active[node][r] = None
        if rank is None:
            # Model loss of access to ALL node-local state, not merely its GPU
            # tensors. This is only an adapter access boundary, not node death.
            if self.replicas[node] is not None:
                for b in self.replicas[node]:
                    b.fill(0)
                self.replicas[node] = None
            if self.local[node] is not None:
                self.local[node].fill(0)
                self.local[node] = None
        return dict(kind="host_rank_kv_erasure" if rank is not None else "logical_node_kv_unavailable",
            node=node, rank=rank, scope="CPU adapter access isolation only", actual_erasure=checks,
            physical_failure=False, gpu_erased=False)

    def recover(self, node, rank=None):
        targets = list(range(self.tp)) if rank is None else [rank]
        if self.replicas[node] is not None:
            restored = {r:self.replicas[node][r].copy() for r in targets}
            route = "complete_cpu_replica"
        elif rank is not None and self.local[node] is not None:
            restored = {rank:xor_codec.recover(self.active[node], self.local[node])}
            route = "local_cpu_xor"
        elif self.global_parity is not None:
            peer = 1-node
            survivor = pack(peer, [b.tobytes() for b in self.active[peer]], self.meta[peer]["ranks"], allow_fixture=self.allow_fixture)
            recovered = rs_codec.recover([None if i == node else survivor for i in (0,1)], self.global_parity, self.lengths)
            if sha(recovered) != self.node_hashes[node]:
                raise ValueError("complete reconstructed node hash mismatch")
            _, ranks = unpack(recovered, allow_fixture=self.allow_fixture)
            restored = {r:np.frombuffer(ranks[r], dtype=np.uint8).copy() for r in targets}
            route = "global_cpu_rs_2_1"
        else:
            raise Unrecoverable("no accessible complete replica/parity route")
        if any(sha(b) != self.hashes[node][r] for r,b in restored.items()):
            raise ValueError("reconstructed KV shard hash mismatch")
        for r,b in restored.items():
            self.active[node][r] = b
        return dict(route=route, byte_exact=True, recovered_bytes=sum(b.nbytes for b in restored.values()),
            recovered_ranks=targets, network_bytes=0, h2d_bytes=0, gpu_writeback=False)


def load_nodes(paths, *, allow_fixture=False):
    nodes = [Path(p).read_bytes() for p in paths]
    if len(nodes) != 2:
        raise ValueError("two independent ordered D0/D1 exports required")
    build("cpu-ablation-input-check", nodes, allow_fixture=allow_fixture)
    sources = []
    for i,(path,raw) in enumerate(zip(paths,nodes)):
        own = descriptor(raw, allow_fixture=allow_fixture)
        if own["metadata"]["node_id"] != i:
            raise ValueError("ordered D0/D1 real node exports required")
        sources.append(dict(path=str(Path(path).resolve()), sha256=sha(raw), descriptor=own))
    return nodes, sources


def run(paths, output, *, rounds=5, max_seconds=1800, allow_fixture=False, idle_check=None,
        cpu_host_mode="strict_idle"):
    if rounds not in (1,5) or not 0 < max_seconds <= 7200:
        raise ValueError("bounded smoke or five-round run required")
    if idle_check is not None and not allow_fixture:
        raise ValueError("idle bypass is test-fixture only")
    check = idle_check or (lambda: gpu_idle(cpu_host_mode))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="ec-cpu-ablation-v2", status="PARTIAL", rounds=rounds, records=[],
        evidence_class="synthetic_fixture" if allow_fixture else ("current_smoke" if rounds == 1 else "current_five_round"),
        code_sha256=code_hashes(), cpu_host_mode=cpu_host_mode, isolated_performance_eligible=False,
        physical_node_failure=False, gpu_failover_verified=False,
        node_rto_ms=None, fault_to_token_ms=None, scope="CPU real-KV erasure/reconstruction only; no GPU or physical node failure")
    try:
        report["gpu_idle_before"] = check()
        nodes, report["sources"] = load_nodes(paths, allow_fixture=allow_fixture)
        tp = len(report["sources"][0]["descriptor"]["metadata"]["ranks"])
        report["expected_records"] = len(MODES)*rounds*(2*tp+2)
        deadline = time.monotonic()+max_seconds
        with RSSSampler() as rss, (output/"records.jsonl").open("x") as stream:
            for mode in MODES:
                for repeat in range(rounds):
                    report.setdefault("gpu_idle_checks", []).append(check())
                    for node in (0,1):
                        for rank in [*range(tp), None]:
                            rec = dict(mode=mode, repeat=repeat, node=node, rank=rank, isolated_performance_eligible=False,
                                fault="rank_kv_erasure" if rank is not None else "node_kv_unavailable",
                                expected_recoverable=(mode in ("rs-only","xor-rs") or rank is not None and mode in ("replication","xor")),
                                recovery_success=False, success=False)
                            try:
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("CPU ablation wall budget exhausted")
                                adapter = ShardAdapter(nodes, mode, allow_fixture=allow_fixture)
                                rec.update(memory_ledger=adapter.ledger, checkpoint_ms=adapter.prepare_ms)
                                rss.sample()
                                started = time.perf_counter_ns()
                                rec["erasure"] = adapter.erase(node,rank)
                                try:
                                    rec["recovery"] = adapter.recover(node,rank)
                                    rec["recovery_success"] = True
                                finally:
                                    rec["fault_to_host_kv_ready_ms"] = (time.perf_counter_ns()-started)/1e6 if rec["recovery_success"] else None
                                rec["success"] = rec["expected_recoverable"] is True
                            except Unrecoverable as exc:
                                rec.update(expected_unrecoverable=not rec["expected_recoverable"],
                                    success=not rec["expected_recoverable"], error=str(exc))
                            except Exception as exc:
                                rec.update(error=f"{type(exc).__name__}: {exc}", timeout=isinstance(exc,TimeoutError))
                            finally:
                                if "adapter" in locals():
                                    del adapter
                            report["records"].append(rec)
                            stream.write(json.dumps(rec)+"\n")
                            stream.flush()
        report["rss"] = rss.report()
        report["gpu_idle_after"] = check()
        report["status"] = "PASS" if all(r["success"] for r in report["records"]) and len(report["records"]) == report["expected_records"] else "PARTIAL"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["recovery_successes"] = sum(r["recovery_success"] for r in report["records"])
        report["expected_unrecoverable"] = sum(r.get("expected_unrecoverable",False) for r in report["records"])
        save(output/"raw.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=Path, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, choices=(1,5), default=5)
    p.add_argument("--max-seconds", type=float, default=1800)
    p.add_argument("--cpu-host-mode", choices=("shared", "strict_idle"), default="strict_idle")
    args = p.parse_args()
    result = run(args.nodes,args.output,rounds=args.rounds,max_seconds=args.max_seconds,cpu_host_mode=args.cpu_host_mode)
    print(json.dumps({k:result.get(k) for k in ("status","recovery_successes","expected_unrecoverable","error")}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
