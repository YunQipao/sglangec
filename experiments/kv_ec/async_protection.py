"""M6 CPU-only, bounded snapshot ownership and observable commit lifecycle.

Unlike AsyncParityClient (diagnostic fire-and-forget), this worker tracks the
manager's terminal response. It never touches CUDA, an allocator or collectives.
"""

from dataclasses import dataclass
import json
import math
import threading
import time

import numpy as np

from .parity_manager import submit_once
from .rank_snapshot import validate_payload
from .scheduler_shadow import validate_checkpoint


UNPROTECTED, SNAPSHOT_READY, COMMITTING, COMMITTED, FAILED = (
    "UNPROTECTED", "SNAPSHOT_READY", "COMMITTING", "COMMITTED", "FAILED")
RANK_BYTES = 36 * 2 * 256 * 2 * 128 * 2
BINDING = (
    "request_id", "request_version", "snapshot_id", "checkpoint_version",
    "manager_epoch", "tp_rank", "tp_size", "req_pool_idx", "req_generation",
    "physical_slots", "slot_generation", "token_start", "token_count",
    "input_ids_sha256", "sha256",
)


def checkpoint_metadata(status):
    """Drop transport envelopes/timings, never checkpoint identity fields."""
    envelope = {"ipc_ms", "transport_ms", "kind", "protocol", "payload_bytes"}
    return {k: v for k, v in status.items() if k not in envelope}


@dataclass(frozen=True)
class CPUSnapshot:
    # bytes and a serialized JSON document have no writable aliases, including
    # through NumPy's setflags(write=True) or a caller's nested metadata dict.
    payload: bytes
    metadata_json: str
    started: float
    ready: float

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    @classmethod
    def freeze(cls, payload, metadata, *, event_complete, started, max_bytes=RANK_BYTES):
        if event_complete is not True or metadata.get("snapshot_completion_status") != "complete":
            raise ValueError("snapshot_copy_incomplete")
        if (not isinstance(payload, np.ndarray) or payload.dtype != np.uint8
                or payload.ndim != 1 or not 0 < payload.nbytes <= max_bytes):
            raise ValueError("snapshot_byte_budget_or_type")
        if any(key not in metadata for key in BINDING):
            raise ValueError("snapshot_binding_incomplete")
        if (metadata["checkpoint_version"] != metadata["snapshot_id"]
                or metadata["tp_size"] != 4 or metadata["token_count"] != 256
                or metadata["token_start"] != 0
                or metadata.get("checksum_deferred", False)
                or len(metadata["slot_generation"]) != 256
                or any(type(g) is not int or g <= 0 for g in metadata["slot_generation"])):
            raise ValueError("unsupported_async_snapshot")
        owned = payload.tobytes()
        # Independently verify the immutable bytes, not the mutable capture.
        validate_payload(np.frombuffer(owned, dtype=np.uint8), metadata)
        return cls(owned, json.dumps(metadata, sort_keys=True), started, time.monotonic())

    def require_binding(self, current):
        if any(current.get(key) != self.metadata[key] for key in BINDING):
            raise ValueError("snapshot_request_chunk_epoch_row_slot_or_version_changed")


class AsyncProtection:
    """One request/chunk/rank, one in-flight task, no unbounded task history.

    Capacity counts queued AND executing work. Normal request completion drains
    off-path until the fixed deadline. Cancellation invalidates local protection
    even if an already executing socket request later commits at the manager.
    """

    def __init__(self, socket, *, timeout_s=30, max_bytes=RANK_BYTES,
                 transport=submit_once, commit_delay_s=0,
                 test_hold_state=""):
        if (not math.isfinite(timeout_s) or timeout_s <= 0 or max_bytes < 1
                or not math.isfinite(commit_delay_s) or not 0 <= commit_delay_s < timeout_s):
            raise ValueError("invalid_async_budget")
        self.socket, self.timeout_s, self.max_bytes = socket, timeout_s, max_bytes
        self.transport, self.commit_delay_s = transport, commit_delay_s
        if test_hold_state not in ("", SNAPSHOT_READY, COMMITTING, COMMITTED):
            raise ValueError("unsupported_test_hold_state")
        self.test_hold_state = test_hold_state
        self.test_state_reached = threading.Event()
        self.test_state_release = threading.Event()
        self.state = UNPROTECTED
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.finished = threading.Event()
        self.snapshot = None
        self.status = None
        self.reason = None
        self.trace = []
        self.thread = None
        self.resident_bytes = self.peak_snapshot_bytes = 0

    def _transition(self, state):
        self.state = state
        self.trace.append(dict(state=state, monotonic=time.monotonic()))

    def stage(self, snapshot):
        with self._lock:
            if self.state != UNPROTECTED:
                raise ValueError("async_queue_full_or_closed")
            if (not isinstance(snapshot, CPUSnapshot) or len(snapshot.payload) > self.max_bytes
                    or type(snapshot.payload) is not bytes or type(snapshot.metadata_json) is not str
                    or snapshot.ready < snapshot.started):
                raise ValueError("invalid_async_snapshot")
            self.snapshot = snapshot
            self.metadata = snapshot.metadata
            self.started, self.ready = snapshot.started, snapshot.ready
            self.resident_bytes = self.peak_snapshot_bytes = len(snapshot.payload)
            self._transition(SNAPSHOT_READY)

    def start(self, *, on_complete=None):
        with self._lock:
            if self.state != SNAPSHOT_READY or self.thread is not None:
                raise ValueError("async_start_requires_ready")
            self.thread = threading.Thread(
                target=self._run, args=(on_complete,), name="kv-ec-m6-commit", daemon=True)
            self.thread.start()

    def _check(self):
        if self._cancel.is_set():
            raise ValueError("request_cancelled")
        remaining = self.ready + self.timeout_s - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("async_commit_deadline")
        return remaining

    def _hold_test_state(self, state):
        if self.test_hold_state != state:
            return
        self.test_state_reached.set()
        if not self.test_state_release.wait(timeout=self.timeout_s):
            raise TimeoutError("m7_test_state_barrier_timeout:" + state)
        self._check()

    def release_test_state(self):
        self.test_state_release.set()

    def _run(self, on_complete):
        try:
            if self.test_hold_state == SNAPSHOT_READY:
                self._hold_test_state(SNAPSHOT_READY)
            with self._lock:
                self._check()
                self._transition(COMMITTING)
                snapshot = self.snapshot
            self._hold_test_state(COMMITTING)
            if self._cancel.wait(self.commit_delay_s):
                raise ValueError("request_cancelled")
            remaining = self._check()
            with self._lock:
                self.submit_started = time.monotonic()
            status = self.transport(self.socket, snapshot.metadata, snapshot.payload,
                                    timeout=min(10., remaining))
            while status.get("state") == "PENDING":
                self._cancel.wait(min(.01, self._check()))
                status = self.transport(
                    self.socket, {"kind": "status", "snapshot_id": self.metadata["snapshot_id"]},
                    b"", timeout=min(10., self._check()))
            validate_checkpoint(status, self.metadata, require_oracles=False)
            with self._lock:
                self._check()
                self.status = checkpoint_metadata(status)
                self._transition(COMMITTED)
            self._hold_test_state(COMMITTED)
        except Exception as exc:
            with self._lock:
                self.reason = f"{type(exc).__name__}:{exc}"
                self._transition(FAILED)
        finally:
            # Drop payload before signalling completion. A cancelled in-flight
            # transmission owns its bytes until it exits, never GPU references.
            snapshot = None
            with self._lock:
                self.snapshot = None
                self.resident_bytes = 0
            self.finished.set()
            if on_complete is not None:
                on_complete(self.record())

    def cancel(self):
        self._cancel.set()
        self.release_test_state()
        with self._lock:
            self.reason = "request_cancelled"
            self.status = None
            self._transition(FAILED)
            if self.thread is None:
                self.snapshot = None
                self.resident_bytes = 0
                self.finished.set()

    def require_recoverable(self, current):
        """Authorization only, NOT a new online writeback path.

        A real recovery must additionally acquire/validate RecoveryLeaseManager's
        live row/slot lease and fetch committed parity from the manager.
        """
        with self._lock:
            if self.state != COMMITTED or self._cancel.is_set():
                raise ValueError("async_not_committed_fail_closed_no_retry")
            if any(current.get(k) != self.metadata[k] for k in BINDING):
                raise ValueError("async_recovery_stale_binding")
            validate_checkpoint(self.status, current, require_oracles=False)
            return json.loads(json.dumps(self.status))

    def record(self):
        with self._lock:
            committed = next((e["monotonic"] for e in reversed(self.trace)
                              if e["state"] == COMMITTED), None)
            return dict(
                state=self.state, reason=self.reason, metadata=getattr(self, "metadata", None),
                trace=list(self.trace), snapshot_started_monotonic=getattr(self, "started", None),
                snapshot_ready_monotonic=getattr(self, "ready", None),
                submit_started_monotonic=getattr(self, "submit_started", None),
                committed_observed_monotonic=committed,
                snapshot_ms=(self.ready - self.started) * 1000 if hasattr(self, "ready") else None,
                background_committed_ms=(committed - self.ready) * 1000 if committed else None,
                capture_to_committed_ms=(committed - self.started) * 1000 if committed else None,
                resident_snapshot_bytes=self.resident_bytes,
                peak_snapshot_bytes=self.peak_snapshot_bytes,
                status=json.loads(json.dumps(self.status)),
                memory_note="snapshot bytes only; capture/transport/manager temporaries are separate")
