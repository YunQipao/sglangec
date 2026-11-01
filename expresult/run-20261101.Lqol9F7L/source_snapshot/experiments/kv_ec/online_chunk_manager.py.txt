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
    ):
        if timeout <= 0 or max_pending < 1 or not run_id:
            raise ValueError("invalid online manager limits")
        self.output, self.run_id = Path(output), run_id
        # Run directories must be fresh. This also prevents duplicate/replayed
        # checkpoints from becoming new groups after a service restart.
        self.output.mkdir(parents=True, exist_ok=False)
        self.timeout, self.max_pending = timeout, max_pending
        self.retain_ranks, self.test_recovery = retain_ranks, test_recovery
        self.global_exporter = global_exporter
        self.pending, self.terminal = {}, OrderedDict()

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
        if payload.nbytes > 64 * 1024**2:
            raise ValueError("rank payload exceeds online byte budget")
        tp, rank = meta["tp_size"], meta["tp_rank"]
        start = meta["chunk_id"] * 256
        expected_id = hashlib.sha256(
            f"{self.run_id}:{meta['request_id']}:{meta['chunk_id']}".encode()
        ).hexdigest()
        if (
            tp not in (2, 4)
            or type(meta["chunk_id"]) is not int
            or meta["chunk_id"] < 0
            or key != expected_id
            or meta["checkpoint_version"] != key
            or meta["manager_epoch"] != self.run_id
            or meta["request_version"] != self.run_id
            or meta["token_start"] != start
            or meta["token_count"] != 256
            or meta["token_range"] != [start, start + 256]
            or meta["layer_range"] != [0, 36]
            or meta["layers"] != list(range(36))
            or meta["model_version"] != meta["model_sha256"]
        ):
            raise ValueError("invalid online identity")
        if key in self.terminal:
            status = self.terminal[key]
            if status["state"] == "COMMITTED":
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
        with span("local_xor_encode", "ec_compute"):
            started = time.perf_counter()
            parity = encode(blocks)
            xor_ms = (time.perf_counter() - started) * 1000
        recovery_checks = []
        if self.test_recovery:
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
        with (directory / "parity.bin").open("xb") as stream:
            stream.write(parity.tobytes())
        manifest.update(
            state="COMMITTED",
            online_identity={k: meta[k] for k in ONLINE_IDENTITY},
            xor_ms=xor_ms,
            aggregation_ms=(time.monotonic() - group["created"]) * 1000,
            rank_performance_ms=[m["performance_ms"] for m in entries],
            parity_sha256=digest(parity),
            recovery_checks=recovery_checks,
            retained_rank_files=self.retain_ranks,
        )
        temporary = directory / "manifest.tmp"
        with temporary.open("x") as stream:
            json.dump(manifest, stream, indent=2)
        temporary.rename(directory / "manifest.json")
        # LOCAL commit is authoritative regardless of remote admission/failure.
        # Only private CPU copies cross this optional boundary. No GPU lifetime
        # is extended, and no RDMA/remote wait runs in this service thread.
        if self.global_exporter is not None:
            try:
                self.global_exporter.enqueue(directory, entries, blocks)
            except Exception as exc:
                error = {
                    "local_state": "LOCAL_COMMITTED",
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
            key, {"state": "COMMITTED", "snapshot_id": key, "xor_ms": xor_ms}
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
                {"state": "READY", "socket": str(path), "run_id": manager.run_id}
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
                    try:
                        meta, raw = recv_message(conn)
                        if meta.get("kind") == "hello" and not raw:
                            response = {"state": "READY", "run_id": manager.run_id}
                        else:
                            response = manager.submit(
                                meta, np.frombuffer(raw, dtype=np.uint8)
                            )
                    except Exception as exc:
                        response = {"state": "FAILED", "reason": type(exc).__name__}
                    try:
                        send_message(conn, response, b"")
                    except OSError:
                        pass
        finally:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
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
        )
        serve(args.socket, manager)
    finally:
        if exporter is not None:
            exporter.close()


if __name__ == "__main__":
    main()
