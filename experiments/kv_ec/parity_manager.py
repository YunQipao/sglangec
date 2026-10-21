"""CPU DRAM parity-only checkpoint manager and Unix-socket service.

The manager owns only pending rank buffers while a checkpoint is incomplete.
After all valid TP=2/4 ranks arrive, it replaces them with one XOR parity block
and immutable metadata. It never writes rank files and never touches SGLang.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import argparse
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import queue
import resource
import socket
import threading
import time
import uuid

import numpy as np

if __package__:
    from .checkpoint_protocol import recv_message, send_message
    from .rank_snapshot import validate_payload
    from .xor_codec import encode, recover
else:
    from checkpoint_protocol import recv_message, send_message
    from rank_snapshot import validate_payload
    from xor_codec import encode, recover

logger = logging.getLogger(__name__)


PENDING, COMMITTED, FAILED, EXPIRED = "PENDING", "COMMITTED", "FAILED", "EXPIRED"


def _identity(metadata):
    keys = ("format", "snapshot_id", "request_id", "model_identifier", "model_sha256", "provenance",
            "source_commit", "core_diff_sha256", "tp_size", "layers",
            "token_start", "token_count", "token_ids_sha256", "dtype",
            "packed_byte_count", "pool_class", "lifetime", "completion_dependency")
    keys += ("request_version", "checkpoint_version", "manager_epoch", "input_ids_sha256")
    return tuple((key, json.dumps(metadata.get(key), sort_keys=True)) for key in keys)


def _layout_identity(metadata):
    return tuple((record.get("layer_id"), record.get("dtype"),
                   json.dumps(record.get("packed_shape"), sort_keys=True),
                   record.get("k_offset"), record.get("k_nbytes"),
                   record.get("v_offset"), record.get("v_nbytes"))
                  for record in metadata.get("layer_records", []))


@dataclass
class Checkpoint:
    snapshot_id: str
    created: float
    deadline: float
    state: str = PENDING
    ranks: dict = field(default_factory=dict)
    parity: np.ndarray | None = None
    metadata: dict | None = None
    error: str | None = None
    oracle: dict | None = None


class ParityManager:
    """Thread-safe TP=2/4 manager; no cross-rank blocking or collectives."""

    def __init__(self, *, timeout=30.0, max_pending=1, retain_oracle=False,
                 test_oracle=False, manager_epoch=None):
        if timeout <= 0 or max_pending < 1:
            raise ValueError("timeout and max_pending must be positive")
        self.timeout = float(timeout)
        self.max_pending = int(max_pending)
        self.retain_oracle = bool(retain_oracle)
        self.test_oracle = test_oracle
        self.manager_epoch = manager_epoch or str(uuid.uuid4())
        self._checkpoints = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, metadata, payload):
        """Submit one CPU-owned rank. Returns status without waiting for peers."""
        try:
            snapshot_id = metadata.get("snapshot_id")
            rank = int(metadata.get("tp_rank", -1))
            with self._lock:
                existing = self._checkpoints.get(snapshot_id)
                if existing is not None and existing.state == PENDING and existing.ranks:
                    first_metadata = next(iter(existing.ranks.values()))[1]
                    if (_identity(metadata) != _identity(first_metadata)
                            or _layout_identity(metadata) != _layout_identity(first_metadata)):
                        existing.state, existing.error = FAILED, "METADATA_MISMATCH"
                        existing.ranks.clear()
                        existing.oracle = None
                        return {"state": FAILED, "reason": existing.error}
            self._validate_submission(metadata, payload)
            snapshot_id = metadata["snapshot_id"]
            rank = int(metadata["tp_rank"])
            now = time.monotonic()
            with self._lock:
                self._expire_locked(now)
                checkpoint = self._checkpoints.get(snapshot_id)
                if checkpoint is None:
                    active = sum(c.state == PENDING for c in self._checkpoints.values())
                    if active >= self.max_pending or len(self._checkpoints) >= 16:
                        return {"state": FAILED, "reason": "PENDING_CAPACITY"}
                    checkpoint = Checkpoint(snapshot_id, now, now + self.timeout)
                    self._checkpoints[snapshot_id] = checkpoint
                if checkpoint.state != PENDING:
                    if checkpoint.state == COMMITTED:
                        return {"state": FAILED, "reason": "CHECKPOINT_CLOSED", "checkpoint_state": COMMITTED}
                    return {"state": checkpoint.state, "reason": checkpoint.error}
                first_metadata = next(iter(checkpoint.ranks.values()))[1] if checkpoint.ranks else None
                if checkpoint.ranks and (
                    _identity(metadata) != _identity(first_metadata)
                    or _layout_identity(metadata) != _layout_identity(first_metadata)
                ):
                    checkpoint.state, checkpoint.error = FAILED, "METADATA_MISMATCH"
                    checkpoint.ranks.clear()
                    checkpoint.oracle = None
                    return {"state": FAILED, "reason": checkpoint.error}
                if rank in checkpoint.ranks:
                    checkpoint.state, checkpoint.error = FAILED, "DUPLICATE_RANK"
                    checkpoint.ranks.clear()
                    checkpoint.oracle = None
                    return {"state": FAILED, "reason": "DUPLICATE_RANK"}
                checkpoint.ranks[rank] = (np.ascontiguousarray(payload, dtype=np.uint8).copy(), deepcopy(metadata))
                if self.retain_oracle:
                    checkpoint.oracle = checkpoint.oracle or {}
                    checkpoint.oracle[rank] = checkpoint.ranks[rank][0].copy()
                tp = metadata["tp_size"]
                expected_ranks = set(range(tp))
                if set(checkpoint.ranks) != expected_ranks:
                    return {"state": PENDING, "ranks": sorted(checkpoint.ranks)}
                blocks = [checkpoint.ranks[i][0] for i in range(tp)]
                xor_started = time.perf_counter()
                checkpoint.parity = encode(blocks)
                xor_ms = (time.perf_counter() - xor_started) * 1000
                oracle_checks = []
                oracle_started = time.perf_counter()
                if self.test_oracle:
                    for lost in range(tp):
                        restored = recover([None if i == lost else b for i, b in enumerate(blocks)], checkpoint.parity)
                        if not np.array_equal(restored, blocks[lost]):
                            raise ValueError("oracle recovery mismatch")
                        oracle_checks.append({"lost_rank": lost, "byte_exact": True})
                    del restored
                checkpoint.metadata = dict(next(iter(checkpoint.ranks.values()))[1])
                checkpoint.metadata.update(state=COMMITTED, parity_bytes=int(checkpoint.parity.size),
                                          parity_sha256=hashlib.sha256(checkpoint.parity).hexdigest(),
                                          committed_at=time.time(), xor_ms=xor_ms,
                                          oracle_validation_ms=(time.perf_counter()-oracle_started)*1000,
                                          parity_commit_ms=(time.monotonic()-checkpoint.created)*1000,
                                          oracle_checks=oracle_checks,
                                          peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                          steady_parity_bytes=int(checkpoint.parity.nbytes))
                checkpoint.metadata["rank_hashes"] = {
                    str(i): {key: checkpoint.ranks[i][1][key] for key in ("sha256", "k_sha256", "v_sha256")}
                    for i in range(tp)
                }
                checkpoint.state = COMMITTED
                # Input rank buffers are never retained after commit unless the
                # explicit test-only oracle was requested.
                checkpoint.ranks.clear()
                logger.info("KV_EC parity COMMITTED snapshot=%s bytes=%d xor_ms=%.3f", snapshot_id, checkpoint.parity.size, xor_ms)
                return self._status_locked(checkpoint)
        except Exception as exc:
            with self._lock:
                checkpoint = self._checkpoints.get(metadata.get("snapshot_id"))
                if checkpoint is None and len(self._checkpoints) < 16:
                    checkpoint = Checkpoint(metadata.get("snapshot_id"), time.monotonic(), time.monotonic())
                    self._checkpoints[metadata.get("snapshot_id")] = checkpoint
                if checkpoint is not None and checkpoint.state == PENDING:
                    checkpoint.state, checkpoint.error = FAILED, type(exc).__name__
                    checkpoint.ranks.clear()
                    checkpoint.oracle = None
                    checkpoint.parity = None
            logger.warning("KV_EC parity submission failed (%s)", type(exc).__name__)
            return {"state": FAILED, "reason": type(exc).__name__}

    def _validate_submission(self, metadata, payload):
        if not isinstance(metadata.get("snapshot_id"), str) or not metadata["snapshot_id"]:
            raise ValueError("snapshot_id required")
        for key in ("request_id", "model_identifier", "model_sha256", "source_commit", "core_diff_sha256", "token_ids_sha256"):
            if not isinstance(metadata.get(key), str) or not metadata[key]:
                raise ValueError("checkpoint identity incomplete")
        tp, rank = metadata.get("tp_size"), metadata.get("tp_rank")
        if type(tp) is not int or tp not in (2, 4) or type(rank) is not int or not 0 <= rank < tp:
            raise ValueError("TP=2/4 and a rank in the complete TP set required")
        if metadata.get("dtype") != "torch.bfloat16":
            raise ValueError("BF16 required")
        if metadata.get("provenance") not in ("sglang_request", "synthetic_fixture"):
            raise ValueError("explicit request provenance required")
        if (metadata.get("provenance") == "sglang_request"
                and metadata.get("manager_epoch") != self.manager_epoch):
            raise ValueError("manager_epoch_mismatch")
        if metadata.get("snapshot_completion_status") != "complete":
            raise ValueError("incomplete snapshot")
        if metadata.get("provenance") == "sglang_request":
            count = metadata.get("token_count")
            if not isinstance(count, int) or count not in (128, 256, 512):
                raise ValueError("prototype supports 128/256/512 tokens")
            for record in metadata.get("layer_records", []):
                if (not 0 <= record["layer_id"] < 36 or
                        any(record["packed_shape"][kind] != [count, 8 // tp, 128]
                            for kind in ("k", "v"))):
                    raise ValueError("Qwen3 TP-sharded shape required")
        payload = np.asarray(payload)
        if payload.dtype != np.uint8 or payload.ndim != 1:
            raise ValueError("flat uint8 payload required")
        if payload.size != metadata.get("packed_byte_count"):
            raise ValueError("payload length mismatch")
        if payload.size > 64 * 1024**2:
            raise ValueError("checkpoint byte budget exceeded")
        deferred = metadata.get("checksum_deferred") is True
        if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
            raise ValueError("payload checksum mismatch")
        # FAST_PARITY may defer checksum calculation, but only explicitly. The
        # manager validates the complete CPU-owned payload before taking the
        # rank into the checkpoint, then writes the derived digests into the
        # metadata used for COMMITTED rank identity and future recovery checks.
        derived = validate_payload(payload, metadata, verify_checksums=not deferred)
        if deferred:
            metadata["sha256"] = derived["sha256"]
            metadata["k_sha256"] = derived["k_sha256"]
            metadata["v_sha256"] = derived["v_sha256"]
            for record in metadata["layer_records"]:
                for kind in ("k", "v"):
                    record[kind + "_sha256"] = derived["layer_sha256"][(record["layer_id"], kind)]
            metadata["checksum_deferred"] = False

    def _expire_locked(self, now):
        for checkpoint in self._checkpoints.values():
            if checkpoint.state == PENDING and now >= checkpoint.deadline:
                checkpoint.state, checkpoint.error = EXPIRED, "TIMEOUT"
                checkpoint.ranks.clear()
                checkpoint.oracle = None

    def _status_locked(self, checkpoint):
        result = {"snapshot_id": checkpoint.snapshot_id, "state": checkpoint.state}
        if checkpoint.error:
            result["reason"] = checkpoint.error
        if checkpoint.state == PENDING:
            result["ranks"] = sorted(checkpoint.ranks)
        if checkpoint.state == COMMITTED:
            result.update(deepcopy(checkpoint.metadata))
        return result

    def status(self, snapshot_id):
        with self._lock:
            self._expire_locked(time.monotonic())
            checkpoint = self._checkpoints.get(snapshot_id)
            return {"snapshot_id": snapshot_id, "state": "UNKNOWN"} if checkpoint is None else self._status_locked(checkpoint)

    def release(self, snapshot_id):
        """Explicitly end a checkpoint's protection and drop its DRAM parity."""
        with self._lock:
            self._checkpoints.pop(snapshot_id, None)

    def export_parity(self, snapshot_id, expected):
        """Export only a committed parity block, bound to the caller's version.

        Survivor bytes stay in the paused workers, not in a second manager
        state machine. The returned copy cannot mutate the checkpoint.
        """
        with self._lock:
            checkpoint = self._checkpoints.get(snapshot_id)
            if checkpoint is None or checkpoint.state != COMMITTED:
                raise ValueError("parity_checkpoint_not_committed")
            metadata = checkpoint.metadata
            if (expected.get("manager_epoch") != self.manager_epoch
                    or _identity(expected) != _identity(metadata)
                    or _layout_identity(expected) != _layout_identity(metadata)):
                raise ValueError("parity_checkpoint_identity_mismatch")
            return self._status_locked(checkpoint), checkpoint.parity.tobytes()

    def recover_for_test(self, snapshot_id, lost_rank):
        """Explicit test oracle; production manager does not retain originals."""
        if not self.retain_oracle:
            raise RuntimeError("test oracle is disabled")
        with self._lock:
            checkpoint = self._checkpoints[snapshot_id]
            if checkpoint.state != COMMITTED or checkpoint.oracle is None:
                raise RuntimeError("checkpoint is not committed")
            tp = checkpoint.metadata["tp_size"]
            if type(lost_rank) is not int or not 0 <= lost_rank < tp:
                raise ValueError("invalid lost rank")
            blocks = [checkpoint.oracle[i] for i in range(tp)]
            return recover([None if i == lost_rank else block for i, block in enumerate(blocks)], checkpoint.parity)


class ParityManagerServer:
    def __init__(self, path, manager=None):
        self.path = Path(path)
        self.manager = manager or ParityManager()
        self._stop = threading.Event()
        self._server = None
        self.ready = threading.Event()

    def serve_forever(self):
        if self.path.exists():
            raise FileExistsError("refusing to replace existing socket")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            self._server = server
            server.bind(str(self.path))
            self.path.chmod(0o600)
            server.listen(16)
            server.settimeout(0.5)
            self.ready.set()
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    with self.manager._lock:
                        self.manager._expire_locked(time.monotonic())
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True
                ).start()
        self.path.unlink(missing_ok=True)

    def _handle(self, conn):
        with conn:
            conn.settimeout(2)
            try:
                metadata, payload = recv_message(conn)
                response_payload = b""
                if metadata.get("kind") == "hello":
                    response = {"state": "READY", "manager_epoch": self.manager.manager_epoch}
                elif metadata.get("kind") == "status":
                    response = self.manager.status(metadata.get("snapshot_id"))
                elif metadata.get("kind") == "parity":
                    if payload:
                        raise ValueError("parity_export_requires_empty_payload")
                    response, response_payload = self.manager.export_parity(
                        metadata.get("snapshot_id"), metadata.get("expected", {}))
                else:
                    response = self.manager.submit(metadata, np.frombuffer(payload, dtype=np.uint8))
            except Exception as exc:
                response = {"state": FAILED, "reason": type(exc).__name__}
                response_payload = b""
            try:
                send_message(conn, {"kind": "response", **response}, response_payload)
            except OSError:
                pass

    def stop(self):
        self._stop.set()
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass


class AsyncParityClient:
    """Bounded, fire-and-forget Worker-side IPC client.

    The submit call only copies an owned CPU payload into a bounded queue. A
    background thread performs socket transfer and never participates in model
    collectives or scheduler control flow.
    """

    def __init__(self, path, *, max_queue=1, timeout=0.25):
        if max_queue < 1 or timeout <= 0:
            raise ValueError("bounded queue and positive timeout required")
        self.path = str(path)
        self.timeout = float(timeout)
        self.queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kv-ec-parity-ipc", daemon=True)
        self._thread.start()

    def submit(self, metadata, payload):
        started = time.perf_counter_ns()
        copied = 0
        copies = 0
        detail = {}
        try:
            # Numpy payloads require one immutable queue-ownership copy.
            # Existing bytes are reused by submit_once and send_message;
            # kernel/socket buffering is separate from Python payload copies.
            copy_started = time.perf_counter_ns()
            copies = int(not isinstance(payload, bytes))
            payload = bytes(payload)
            copied = len(payload) if copies else 0
            detail["payload_copy_ms"] = (time.perf_counter_ns() - copy_started) / 1e6
            metadata_started = time.perf_counter_ns()
            owned_metadata = deepcopy(metadata)
            detail["metadata_deepcopy_ms"] = (time.perf_counter_ns() - metadata_started) / 1e6
            put_started = time.perf_counter_ns()
            try:
                self.queue.put_nowait((owned_metadata, payload))
            finally:
                detail["queue_put_ms"] = (time.perf_counter_ns() - put_started) / 1e6
            result = {"accepted": True}
        except queue.Full:
            result = {"accepted": False, "reason": "QUEUE_FULL"}
        elapsed = (time.perf_counter_ns() - started) / 1e6
        # Count attempted copies even when queue.put_nowait subsequently fails.
        return dict(result, enqueue_ms=elapsed, copy_bytes=copied, copy_count=copies,
                    unaccounted_ms=max(0.0, elapsed - sum(detail.values())), **detail)

    def _run(self):
        while not self._stop.is_set() or not self.queue.empty():
            try:
                metadata, payload = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                response = submit_once(self.path, metadata, payload, timeout=self.timeout)
                logger.info("KV_EC parity IPC snapshot=%s rank=%s state=%s ipc_ms=%.3f",
                            metadata.get("snapshot_id"), metadata.get("tp_rank"),
                            response.get("state"), response.get("ipc_ms", -1.0))
                logger.info("KV_EC IPC background phases snapshot=%s rank=%s phases=%s",
                            metadata.get("snapshot_id"), metadata.get("tp_rank"),
                            json.dumps(response.get("transport_ms", {}), sort_keys=True))
            except Exception as exc:
                logger.warning("KV_EC parity IPC failed snapshot=%s rank=%s (%s)",
                               metadata.get("snapshot_id"), metadata.get("tp_rank"),
                               type(exc).__name__)
            finally:
                self.queue.task_done()
                # Do not pin the last full rank payload in an idle worker's
                # local variables after the queue task has completed.
                del metadata, payload

    def close(self, *, drain=False):
        self._stop.set()
        # Bounded join even when the manager is unavailable. Never called by
        # the scheduler; queued data is owned only by this background client.
        self._thread.join(timeout=(self.timeout * (self.queue.maxsize + 1) + 1) if drain else 0)


def submit_once(path, metadata, payload, timeout=0.25):
    # Direct callers may submit a numpy rank block; the wire protocol accepts
    # only bytes-like values. For the background client's already-owned bytes,
    # bytes(payload) returns the same object (no additional payload copy).
    payload = bytes(payload)
    started = time.perf_counter_ns()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        connect_started = time.perf_counter_ns()
        client.connect(str(path))
        connect_ms = (time.perf_counter_ns() - connect_started) / 1e6
        transport = send_message(client, metadata, payload)
        response_started = time.perf_counter_ns()
        response, _ = recv_message(client)
        response_ms = (time.perf_counter_ns() - response_started) / 1e6
    response["ipc_ms"] = (time.perf_counter_ns() - started) / 1e6
    response["transport_ms"] = {"connect_ms": connect_ms, "response_wait_ms": response_ms,
                                **(transport if isinstance(transport, dict) else {})}
    return response


def manager_hello(path, timeout=2.0):
    response = submit_once(path, {"kind": "hello"}, b"", timeout=timeout)
    if response.get("state") != "READY" or not response.get("manager_epoch"):
        raise RuntimeError("parity manager did not provide a live epoch")
    return response["manager_epoch"]


def fetch_parity(path, expected, timeout=2.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        send_message(client, {"kind": "parity", "snapshot_id": expected["snapshot_id"],
                              "expected": expected}, b"")
        metadata, raw = recv_message(client)
    if metadata.get("state") != COMMITTED:
        raise ValueError(f"parity_export_failed:{metadata.get('reason')}")
    if (len(raw) != expected["packed_byte_count"]
            or hashlib.sha256(raw).hexdigest() != metadata.get("parity_sha256")):
        raise ValueError("parity_export_byte_mismatch")
    return metadata, np.frombuffer(raw, dtype=np.uint8).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-pending", type=int, default=1)
    parser.add_argument("--test-oracle", action="store_true", help="verify all erasures transiently before releasing inputs")
    parser.add_argument("--status", help="query a snapshot instead of starting a server")
    parser.add_argument("--output", type=Path, help="exclusive-create status report (metadata only)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.status:
        encoded = json.dumps(submit_once(args.socket, {"kind": "status", "snapshot_id": args.status}, b"", timeout=2), indent=2)
        if args.output:
            with args.output.open("x") as stream:
                stream.write(encoded + "\n")
        print(encoded)
        return
    ParityManagerServer(args.socket, ParityManager(timeout=args.timeout, max_pending=args.max_pending, test_oracle=args.test_oracle)).serve_forever()


if __name__ == "__main__":
    main()
