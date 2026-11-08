"""CPU-only online TP2/TP4 aggregation and XOR; optional background RDMA handoff.

The socket is private to one run. Bounded pending groups expire; terminal
identities are retained in a bounded LRU and on disk. No socket is unlinked or
checkpoint overwritten at startup. Disk I/O and XOR never run in the scheduler.
"""

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np

from .checkpoint_protocol import recv_message, send_message
from .rank_snapshot import COMMON, rank_manifest, validate_payload, write_rank_files
from .snapshot import digest
from .xor_codec import encode, recover
from .perf_observer import span


ONLINE_IDENTITY = (
    "chunk_id",
    "token_range",
    "layer_range",
    "model_version",
    "checkpoint_version",
    "request_version",
    "manager_epoch",
)


class OnlineChunkManager:
    def __init__(
        self,
        output,
        run_id,
        *,
        timeout=30,
        max_pending=8,
        retain_ranks=False,
        test_recovery=False,
        global_exporter=None,
        protection_mode="xor",
        max_replica_bytes=2 * 1024**3,
    ):
        if timeout <= 0 or max_pending < 1 or not run_id:
            raise ValueError("invalid online manager limits")
        if protection_mode not in ("replication", "rs-only", "xor", "xor-rs"):
            raise ValueError("invalid protection mode")
        # Preserve the historical xor + exporter invocation.
        if protection_mode == "xor" and global_exporter is not None:
            protection_mode = "xor-rs"
        if (protection_mode in ("rs-only", "xor-rs")) != (global_exporter is not None):
            raise ValueError("global mode requires exporter; replication forbids exporter")
        if type(max_replica_bytes) is not int or max_replica_bytes <= 0:
            raise ValueError("positive replica byte budget required")
        self.protection_mode = protection_mode
        self.max_replica_bytes = max_replica_bytes
        self.replicas, self.replica_bytes = {}, 0
        self.local_parities, self.local_parity_bytes = {}, 0
        self.manifests = {}  # immutable recovery metadata; fault path does no disk I/O
        self.output, self.run_id = Path(output), run_id
        # Run directories must be fresh. This also prevents duplicate/replayed
        # checkpoints from becoming new groups after a service restart.
        self.output.mkdir(parents=True, exist_ok=False)
        self.timeout, self.max_pending = timeout, max_pending
        self.retain_ranks, self.test_recovery = retain_ranks, test_recovery
        self.global_exporter = global_exporter
        self.pending, self.terminal = {}, OrderedDict()
        (self.output / "service_manifest.json").write_text(json.dumps(dict(
            schema="kv-online-service-v2", run_id=run_id,
            protection_mode=protection_mode, retain_ranks_debug=retain_ranks,
            replication_storage="independent_complete_rank_arrays_in_host_dram" if protection_mode == "replication" else None,
            max_replica_bytes=max_replica_bytes,
        ), indent=2))

    def status(self, key):
        """RS-only snapshots are not protected until the bound global commit."""
        result = dict(self.terminal[key])
        global_path = self.output / key / "global_status.json"
        if global_path.exists():
            receipt = json.loads(global_path.read_text())
            result["global_state"] = receipt.get("global_state")
            result["remote_outcome"] = receipt.get("remote_outcome")
        if self.protection_mode == "rs-only" and result["state"] == "LOCAL_SNAPSHOT":
            if global_path.exists():
                receipt = json.loads(global_path.read_text())
                if (receipt.get("global_state") == "GLOBAL_COMMITTED"
                        and receipt.get("local_snapshot_id") == key
                        and receipt.get("remote_outcome") == "COMMITTED"):
                    result.update(state="COMMITTED", protection_state="GLOBAL_COMMITTED")
        return result

    def load_replica(self, key, rank, *, request_id):
        """Load from the actual DRAM replica, never from original/debug files."""
        if self.protection_mode != "replication" or key not in self.replicas:
            raise ValueError("complete live replica required")
        manifest = self.manifests.get(key)
        if manifest is None:
            raise ValueError("live replica metadata unavailable")
        if (manifest["request_id"] != request_id or type(rank) is not int
                or not 0 <= rank < manifest["tp_size"]):
            raise ValueError("replica request/rank mismatch")
        block = self.replicas[key][rank].copy()
        validate_payload(block, manifest["ranks"][rank])
        return block

    def recover_xor(self, key, lost_rank, survivor, *, request_id):
        """Reconstruct one TP rank strictly from live DRAM parity + survivor.

        No rank file or parity file is read.  The survivor is supplied by its
        still-live TP rank and authenticated against the committed manifest.
        """
        if self.protection_mode not in ("xor", "xor-rs"):
            raise ValueError("live XOR parity required")
        if key not in self.local_parities or key not in self.terminal:
            raise ValueError("committed live parity required")
        manifest = self.manifests.get(key)
        if manifest is None:
            raise ValueError("live parity metadata unavailable")
        if (
            manifest["request_id"] != request_id
            or type(lost_rank) is not int
            or not 0 <= lost_rank < manifest["tp_size"]
            or manifest["tp_size"] != 2
        ):
            raise ValueError("xor recovery identity mismatch")
        survivor_rank = 1 - lost_rank
        validate_payload(survivor, manifest["ranks"][survivor_rank])
        parity = self.local_parities[key]
        if digest(parity) != manifest["parity_sha256"]:
            raise ValueError("live parity corruption")
        started = time.perf_counter()
        restored = np.bitwise_xor(parity, survivor)
        validate_payload(restored, manifest["ranks"][lost_rank])
        return restored, {
            "state": "COMMITTED",
            "snapshot_id": key,
            "lost_rank": lost_rank,
            "source": "bounded_host_dram_parity_plus_authenticated_survivor",
            "disk_reads": 0,
            "reconstruct_ms": (time.perf_counter() - started) * 1000,
        }

    def release_protection(self, key, *, request_id):
        """Explicit experiment lifetime end; never silently evict live protection."""
        directory = self.output / key
        if key not in self.terminal or not directory.is_dir():
            raise ValueError("known checkpoint required")
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest["request_id"] != request_id:
            raise ValueError("protection release request mismatch")
        replicas = self.replicas.pop(key, ())
        self.replica_bytes -= sum(b.nbytes for b in replicas)
        parity = self.local_parities.pop(key, None)
        if parity is not None:
            self.local_parity_bytes -= parity.nbytes
        self.manifests.pop(key, None)
        manifest["dram_retention"] = "RELEASED_BY_CALLER"
        manifest["live_local_redundancy_bytes"] = 0
        temporary = directory / "manifest.release.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(path)
        self.terminal[key] = dict(self.terminal[key], state="RELEASED")
        return dict(state="RELEASED", snapshot_id=key)

    def expire(self):
        now = time.monotonic()
        for key, group in list(self.pending.items()):
            if now - group["created"] >= self.timeout:
                self._end(key, {"state": "EXPIRED", "reason": "missing_ranks"})

    def _end(self, key, status):
        self.pending.pop(key, None)
        self.terminal[key] = status
        while len(self.terminal) > 256:
            self.terminal.popitem(last=False)
        return status

    def submit(self, meta, payload):
        self.expire()
        key = meta.get("snapshot_id")
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
        ):
            raise ValueError("invalid checkpoint id")
        validate_payload(payload, meta)
        if meta.get("protection_mode", self.protection_mode) != self.protection_mode:
            raise ValueError("runtime/manager protection mode mismatch")
        if payload.nbytes > 512 * 1024**2:
            raise ValueError("rank payload exceeds online byte budget")
        tp, rank = meta["tp_size"], meta["tp_rank"]
        grain = meta.get("capture_grain_tokens", 256)
        start = meta["chunk_id"] * grain
        identity_text = (
            f"{self.run_id}:{meta['request_id']}:{meta['chunk_id']}"
            if grain == 256 else
            f"{self.run_id}:{meta['request_id']}:{start}:{start + grain}"
        )
        expected_id = hashlib.sha256(identity_text.encode()).hexdigest()
        layer_range = meta.get("layer_range")
        layer_count = (
            layer_range[1] if isinstance(layer_range, list)
            and len(layer_range) == 2 and layer_range[0] == 0
            and type(layer_range[1]) is int else -1
        )
        if (
            tp not in (2, 4)
            or type(meta["chunk_id"]) is not int
            or meta["chunk_id"] < 0
            or key != expected_id
            or meta["checkpoint_version"] != key
            or meta["manager_epoch"] != self.run_id
            or meta["request_version"] != self.run_id
            or meta["token_start"] != start
            or grain not in (128, 256, 512, 1024)
            or meta["token_count"] != grain
            or meta["token_range"] != [start, start + meta["token_count"]]
            or not 0 < layer_count <= 128
            or meta["layers"] != list(range(layer_count))
            or meta["model_version"] != meta["model_sha256"]
        ):
            raise ValueError("invalid online identity")
        if key in self.terminal:
            status = self.status(key)
            if status["state"] in ("COMMITTED", "LOCAL_SNAPSHOT", "RELEASED"):
                manifest = json.loads((self.output / key / "manifest.json").read_text())
                original = manifest["ranks"][rank]
                if meta["sha256"] != original["sha256"] or any(
                    meta.get(k) != original.get(k) for k in COMMON + ONLINE_IDENTITY
                ):
                    return {
                        "state": "FAILED",
                        "reason": "conflicting_closed_checkpoint",
                    }
            return {**status, "duplicate": True}
        if (self.output / key).exists():
            return {"state": "CLOSED", "reason": "checkpoint_already_persisted"}
        group = self.pending.get(key)
        if group is None:
            if len(self.pending) >= self.max_pending:
                return {"state": "DROPPED", "reason": "pending_capacity"}
            group = self.pending[key] = {"created": time.monotonic(), "ranks": {}}
        ranks = group["ranks"]
        if ranks:
            first = next(iter(ranks.values()))[1]
            if any(meta[k] != first[k] for k in ONLINE_IDENTITY):
                return self._end(
                    key, {"state": "FAILED", "reason": "identity_mismatch"}
                )
            # Reuse the canonical offline rank/layout validator as the common
            # boundary. A partial set is checked by filling only rank indices,
            # retaining every other identity/layout field verbatim.
            rank_manifest(
                [dict(first, tp_rank=i) if i != rank else meta for i in range(tp)]
            )
        if rank in ranks:
            if ranks[rank][1] == meta and np.array_equal(ranks[rank][0], payload):
                return {"state": "PENDING", "duplicate": True}
            return self._end(
                key, {"state": "FAILED", "reason": "conflicting_duplicate"}
            )
        ranks[rank] = (payload.copy(), deepcopy(meta))
        if set(ranks) != set(range(tp)):
            return {"state": "PENDING", "ranks": sorted(ranks)}
        entries = [ranks[i][1] for i in range(tp)]
        manifest = rank_manifest(entries)
        blocks = [ranks[i][0] for i in range(tp)]
        commit_started = time.perf_counter()
        source_bytes = sum(b.nbytes for b in blocks)
        if (self.protection_mode == "replication"
                and self.replica_bytes + source_bytes > self.max_replica_bytes):
            return self._end(key, {"state": "DROPPED", "reason": "replica_capacity"})
        if (self.protection_mode in ("xor", "xor-rs")
                and self.local_parity_bytes + blocks[0].nbytes > self.max_replica_bytes):
            return self._end(key, {"state": "DROPPED", "reason": "parity_capacity"})
        parity, xor_ms = None, None
        if self.protection_mode in ("xor", "xor-rs"):
            with span("local_xor_encode", "ec_compute"):
                started = time.perf_counter()
                parity = encode(blocks)
                xor_ms = (time.perf_counter() - started) * 1000
            self.local_parities[key] = parity
            self.local_parity_bytes += parity.nbytes
        replica_copy_ms = None
        if self.protection_mode == "replication":
            started = time.perf_counter()
            self.replicas[key] = tuple(b.copy() for b in blocks)
            self.replica_bytes += source_bytes
            replica_copy_ms = (time.perf_counter() - started) * 1000
        recovery_checks = []
        if self.test_recovery and parity is not None:
            with span("local_xor_validation", "validation"):
                for lost in range(tp):
                    restored = recover(
                        [None if i == lost else b for i, b in enumerate(blocks)], parity
                    )
                    if not np.array_equal(restored, blocks[lost]):
                        raise ValueError("XOR recovery mismatch")
                    recovery_checks.append({"lost_rank": lost, "byte_exact": True})
        directory = self.output / key
        directory.mkdir()
        # Atomic *completion marker*: partial files after a crash cannot be
        # mistaken for protection. These writes are on the CPU service only.
        if self.retain_ranks:
            for block, entry in zip(blocks, entries):
                write_rank_files(directory, block, entry)
        if parity is not None:
            with (directory / "parity.bin").open("xb") as stream:
                stream.write(parity.tobytes())
        state = "LOCAL_SNAPSHOT" if self.protection_mode == "rs-only" else "COMMITTED"
        manifest.update(
            state=state,
            protection_mode=self.protection_mode,
            dram_retention="LIVE",
            live_local_redundancy_bytes=source_bytes if self.protection_mode == "replication" else (parity.nbytes if parity is not None else 0),
            protection_state="AWAITING_GLOBAL" if state == "LOCAL_SNAPSHOT" else "LOCAL_COMMITTED",
            online_identity={k: meta[k] for k in ONLINE_IDENTITY},
            xor_ms=xor_ms,
            aggregation_ms=(time.monotonic() - group["created"]) * 1000,
            rank_performance_ms=[m["performance_ms"] for m in entries],
            parity_sha256=digest(parity) if parity is not None else None,
            recovery_checks=recovery_checks,
            retained_rank_files=self.retain_ranks,
            memory_ledger=dict(source_rank_bytes=source_bytes,
                stable_local_parity_bytes=parity.nbytes if parity is not None else 0,
                stable_replica_bytes=source_bytes if self.protection_mode == "replication" else 0,
                stable_remote_parity_bytes=None if self.global_exporter else 0,
                snapshot_bytes=source_bytes, temporary_rank_copy_bytes=source_bytes,
                debug_disk_rank_bytes=source_bytes if self.retain_ranks else 0,
                measured_rss_bytes=None, note="allocation ledger, not RSS; overlapping lifetimes not summed"),
            timing_ms=dict(capture_per_rank=[m["performance_ms"] for m in entries],
                encode=xor_ms, replica_copy=replica_copy_ms,
                local_prepare=(time.perf_counter()-commit_started)*1000,
                commit=None, global_commit=None),
        )
        # Recovery reads this immutable copy.  The JSON files below are only
        # experiment evidence and are never consulted by the fault path.
        self.manifests[key] = deepcopy(manifest)
        temporary = directory / "manifest.tmp"
        with temporary.open("x") as stream:
            json.dump(manifest, stream, indent=2)
        temporary.rename(directory / "manifest.json")
        manifest["timing_ms"]["commit"] = (time.perf_counter()-commit_started)*1000
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(directory / "manifest.json")
        # LOCAL commit is authoritative regardless of remote admission/failure.
        # Only private CPU copies cross this optional boundary. No GPU lifetime
        # is extended, and no RDMA/remote wait runs in this service thread.
        if self.global_exporter is not None:
            try:
                self.global_exporter.enqueue(directory, entries, blocks)
            except Exception as exc:
                error = {
                    "local_state": state,
                    "global_state": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                try:
                    (directory / "global_handoff_error.json").write_text(
                        json.dumps(error)
                    )
                except OSError:
                    print(
                        json.dumps(dict(kind="global_handoff_error", **error)),
                        flush=True,
                    )
        return self._end(
            key, {"state": state, "snapshot_id": key, "xor_ms": xor_ms,
                  "protection_mode": self.protection_mode}
        )


def serve(path, manager, *, stop=None, ready=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to replace existing socket")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        path.chmod(0o600)
        server.listen(8)
        server.settimeout(0.5)
        print(
            json.dumps(
                {"state": "READY", "socket": str(path), "run_id": manager.run_id,
                 "protection_mode": manager.protection_mode}
            ),
            flush=True,
        )
        if ready is not None:
            ready.set()
        try:
            while stop is None or not stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    manager.expire()
                    continue
                with conn:
                    conn.settimeout(5)
                    response_raw = b""
                    try:
                        meta, raw = recv_message(conn)
                        if meta.get("kind") == "hello" and not raw:
                            response = {"state": "READY", "run_id": manager.run_id,
                                        "protection_mode": manager.protection_mode}
                        elif meta.get("kind") == "status" and not raw:
                            key = meta.get("snapshot_id")
                            if key in manager.terminal:
                                response = manager.status(key)
                            elif key in manager.pending:
                                response = {"state": "PENDING", "snapshot_id": key}
                            else:
                                response = {"state": "UNKNOWN", "snapshot_id": key}
                        elif meta.get("kind") == "load_replica" and not raw:
                            restored = manager.load_replica(meta["snapshot_id"], meta["tp_rank"], request_id=meta["request_id"])
                            response_raw = restored.tobytes()
                            response = dict(state="COMMITTED", snapshot_id=meta["snapshot_id"],
                                tp_rank=meta["tp_rank"], request_id=meta["request_id"], sha256=digest(restored))
                        elif meta.get("kind") == "recover_xor" and raw:
                            restored, response = manager.recover_xor(
                                meta["snapshot_id"], meta["lost_rank"],
                                np.frombuffer(raw, dtype=np.uint8),
                                request_id=meta["request_id"],
                            )
                            response_raw = restored.tobytes()
                            response["sha256"] = digest(restored)
                        elif meta.get("kind") == "release_protection" and not raw:
                            response = manager.release_protection(meta["snapshot_id"], request_id=meta["request_id"])
                        else:
                            response = manager.submit(
                                meta, np.frombuffer(raw, dtype=np.uint8)
                            )
                    except Exception as exc:
                        response = {"state": "FAILED", "reason": type(exc).__name__}
                    try:
                        send_message(conn, response, response_raw)
                    except OSError:
                        pass
        finally:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protection-mode", choices=("replication", "rs-only", "xor", "xor-rs"), default="xor")
    parser.add_argument("--max-replica-bytes", type=int, default=2 * 1024**3,
                        help="bounded live local replica OR parity DRAM bytes; no silent eviction")
    parser.add_argument(
        "--retain-ranks", action="store_true", help="debug byte-comparison evidence"
    )
    parser.add_argument(
        "--test-recovery", action="store_true", help="debug all single-rank erasures"
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-pending", type=int, default=8)
    parser.add_argument(
        "--global-rs-config",
        type=Path,
        help="opt-in immutable request pairing + RDMA worker config",
    )
    args = parser.parse_args()
    exporter = None
    if args.global_rs_config:
        from .online_rs_exporter import OnlineRSExporter

        exporter = OnlineRSExporter.from_config(args.global_rs_config)
    try:
        manager = OnlineChunkManager(
            args.output,
            args.run_id,
            timeout=args.timeout,
            max_pending=args.max_pending,
            retain_ranks=args.retain_ranks,
            test_recovery=args.test_recovery,
            global_exporter=exporter,
            protection_mode=args.protection_mode,
            max_replica_bytes=args.max_replica_bytes,
        )
        serve(args.socket, manager)
    finally:
        if exporter is not None:
            exporter.close()


if __name__ == "__main__":
    main()
