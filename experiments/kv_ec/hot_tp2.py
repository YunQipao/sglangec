"""Legacy standalone TP2 data-path helpers.

Immutable bytes only: file-backed/mutable inputs are not accepted in the fault
path. Transport and sink adapters must implement finite I/O timeouts. This
module creates no services, files, CUDA contexts, or scheduler hooks on import.
The production experiment adapter is :mod:`live_recovery`; these helpers remain
as its CPU oracle and bounded-pipeline test surface.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import math
import time

import numpy as np

from .multinode.rs_codec import CODEC, PRIMITIVE_POLYNOMIAL

MIB = 1024**2
CHUNK_MIB = (1, 4, 16, 64)


class Rejected(RuntimeError):
    pass


@dataclass(frozen=True)
class Options:
    enabled: bool = False
    chunk_mib: int = 4
    depth: int = 2
    max_bytes: int = 512 * MIB

    def require(self):
        if self.enabled is not True:
            raise Rejected("hot_tp2_disabled")
        if (type(self.chunk_mib) is not int or self.chunk_mib not in CHUNK_MIB
                or type(self.depth) is not int or not 1 <= self.depth <= 8
                or type(self.max_bytes) is not int or self.max_bytes <= 0):
            raise Rejected("invalid_bounded_configuration")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def _bytes(value, size=None):
    if type(value) is not bytes or not value or (size is not None and len(value) != size):
        raise Rejected("immutable_nonempty_memory_bytes_required")
    return np.frombuffer(value, dtype=np.uint8)


def rs_chunk(a, b, operation):
    """Vectorized CPU implementation; oracle is multinode.rs_codec in tests.

    encode/recover0: a XOR 2*b; recover1: (a XOR b)/2, polynomial 0x11d.
    This is NOT the XOR-only code used by hot_recovery_pipeline.
    """
    a, b = _bytes(a), _bytes(b, len(a))
    if operation in ("encode", "recover0"):
        out = np.left_shift(b, np.uint8(1))
        out ^= (b >> np.uint8(7)) * np.uint8(PRIMITIVE_POLYNOMIAL & 255)
        out ^= a
    elif operation == "recover1":
        out = np.bitwise_xor(a, b)
        out = (out >> np.uint8(1)) ^ ((out & np.uint8(1)) * np.uint8(0x8e))
    else:
        raise Rejected("unknown_rs_operation")
    return out.tobytes()


def bounded_map(items, compute, consume, *, depth):
    """At most depth futures/results; ordered ACK/sink provides backpressure.

    Workers overlap input/EC with the consumer's send or GPU staging. No eager
    Executor.map, full-payload reassembly, or unbounded producer queue. On error
    cancel queued work and join finite in-flight callbacks before returning.
    """
    if type(depth) is not int or not 1 <= depth <= 8:
        raise Rejected("invalid_pipeline_depth")
    iterator, pending = iter(items), deque()
    count = peak = 0
    with ThreadPoolExecutor(max_workers=depth, thread_name_prefix="hot-tp2") as pool:
        try:
            for _ in range(depth):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(compute, item))
            peak = len(pending)
            while pending:
                result = pending.popleft().result()
                consume(result)  # ACK or sink acceptance BEFORE pulling more
                del result
                count += 1
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.append(pool.submit(compute, item))
                peak = max(peak, len(pending))
        finally:
            for future in pending:
                future.cancel()
    return dict(chunks=count, peak_pending=peak, pipeline_depth=depth,
                gpudirect=False)


@dataclass(frozen=True)
class Chunk:
    checkpoint: str
    index: int
    generation: int
    data_sha: tuple[str, str]
    parity_sha: str
    parity: bytes
    codec: str = CODEC

    @property
    def receipt(self):
        return (self.checkpoint, self.index, self.generation, self.parity_sha, self.codec, self.data_sha)


@dataclass(frozen=True)
class MemoryCheckpoint:
    checkpoint: str
    chunk_bytes: int
    total_bytes: int
    chunks: tuple[Chunk, ...]

    def validate(self, options):
        options.require()
        if (type(self.checkpoint) is not str or not self.checkpoint
                or type(self.chunks) is not tuple
                or self.chunk_bytes != options.chunk_mib * MIB
                or not 0 < self.total_bytes <= options.max_bytes
                or len(self.chunks) != (self.total_bytes + self.chunk_bytes - 1) // self.chunk_bytes):
            raise Rejected("checkpoint_geometry_mismatch")
        for index, c in enumerate(self.chunks):
            size = min(self.chunk_bytes, self.total_bytes - index * self.chunk_bytes)
            if (type(c) is not Chunk or c.checkpoint != self.checkpoint or c.index != index or c.codec != CODEC
                    or type(c.generation) is not int or c.generation < 0
                    or type(c.parity) is not bytes or len(c.parity) != size
                    or type(c.data_sha) is not tuple or len(c.data_sha) != 2
                    or any(type(h) is not str or len(h) != 64
                           or any(x not in "0123456789abcdef" for x in h)
                           for h in (*c.data_sha, c.parity_sha))):
                raise Rejected("invalid_committed_chunk_manifest")


def protect(data0, data1, *, checkpoint, options=Options(), send=None, codec=rs_chunk):
    """Return only a fully ACKed local checkpoint; send(chunk) must ACK receipt.

    A remote partial send is not a committed checkpoint. Actual RS-service
    transaction publication and cross-host integration remain adapter work.
    """
    options.require()
    _bytes(data0)
    _bytes(data1, len(data0))
    if type(checkpoint) is not str or not checkpoint or len(data0) > options.max_bytes:
        raise Rejected("checkpoint_identity_or_memory_budget")
    width, chunks = options.chunk_mib * MIB, []

    def encode(index):
        start = index * width
        a, b = data0[start:start + width], data1[start:start + width]
        p = codec(a, b, "encode")
        _bytes(p, len(a))
        return Chunk(checkpoint, index, 0, (sha(a), sha(b)), sha(p), p)

    def commit(chunk):
        if send is not None and send(chunk) != chunk.receipt:
            raise Rejected("parity_ack_binding_mismatch")
        chunks.append(chunk)

    stats = bounded_map(range((len(data0) + width - 1) // width), encode, commit,
                        depth=options.depth)
    cp = MemoryCheckpoint(checkpoint, width, len(data0), tuple(chunks))
    cp.validate(options)
    return cp, stats


def update_dirty(cp, changes, *, options=Options(), send=None, codec=rs_chunk):
    """Atomic copy-on-write local update; unchanged Chunk objects are reused.

    changes = {index: (expected_generation, old_d0, old_d1, new_d0, new_d1)}.
    Producer explicitly marks dirty chunks, and supplies authenticated old
    bytes. No scan/hash of clean chunks; callers keep the old CP on any error.
    Remote adapter must stage receipts, not publish partial versions in place.
    """
    cp.validate(options)
    if type(changes) is not dict or any(type(i) is not int or not 0 <= i < len(cp.chunks)
                                        for i in changes):
        raise Rejected("invalid_dirty_indices")
    replacements = {}
    for index, change in changes.items():
        old = cp.chunks[index]
        if (type(change) is not tuple or len(change) != 5
                or type(change[0]) is not int or change[0] != old.generation):
            raise Rejected("stale_dirty_generation")
        old0, old1, new0, new1 = change[1:]
        for value in (old0, old1, new0, new1):
            _bytes(value, len(old.parity))
        if (sha(old0), sha(old1)) != old.data_sha or sha(old.parity) != old.parity_sha:
            raise Rejected("dirty_old_bytes_mismatch")
        # GF delta: P' = P XOR (old0 XOR new0) XOR 2*(old1 XOR new1).
        delta0 = np.bitwise_xor(_bytes(old0), _bytes(new0)).tobytes()
        delta1 = np.bitwise_xor(_bytes(old1), _bytes(new1)).tobytes()
        delta = codec(delta0, delta1, "encode")
        parity = np.bitwise_xor(_bytes(old.parity), _bytes(delta, len(old.parity))).tobytes()
        new = replace(old, generation=old.generation + 1,
                      data_sha=(sha(new0), sha(new1)), parity=parity, parity_sha=sha(parity))
        if send is not None and send(new) != new.receipt:
            raise Rejected("dirty_ack_binding_mismatch")
        replacements[index] = new
    return replace(cp, chunks=tuple(replacements.get(i, c) for i, c in enumerate(cp.chunks))), dict(
        dirty_chunks=len(replacements), updated_bytes=sum(len(c.parity) for c in replacements.values()),
        clean_chunks_hashed=0)


def recover(cp, *, lost_rank, fetch_survivor, sink, options=Options(), codec=rs_chunk):
    """fetch_survivor(index, size)->immutable bytes, sink.stage(index, bytes).

    Sink must quarantine partial output, implement finish() including full
    destination validation, and abort() on any failure. No Scheduler commit.
    fetch may be timeout-bounded network I/O; adapters are never GPUDirect.
    """
    start, first = time.perf_counter_ns(), None
    cp.validate(options)
    if type(lost_rank) is not int or lost_rank not in (0, 1):
        raise Rejected("exactly_one_tp2_rank_must_be_lost")

    def decode(c):
        survivor = fetch_survivor(c.index, len(c.parity))
        _bytes(survivor, len(c.parity))
        if sha(c.parity) != c.parity_sha or sha(survivor) != c.data_sha[1 - lost_rank]:
            raise Rejected("recovery_input_sha256_mismatch")
        restored = codec(c.parity, survivor, f"recover{lost_rank}")
        _bytes(restored, len(c.parity))
        if sha(restored) != c.data_sha[lost_rank]:
            raise Rejected("reconstructed_sha256_mismatch")
        return c.index, restored

    def write(result):
        nonlocal first
        sink.stage(*result)
        if first is None:
            first = (time.perf_counter_ns() - start) / 1e6

    try:
        stats = bounded_map(cp.chunks, decode, write, depth=options.depth)
        if sink.finish() is not True:
            raise Rejected("destination_full_verification_required")
    except BaseException:
        sink.abort()
        raise
    return stats | dict(restore_call_to_verified_data_ready_ms=(time.perf_counter_ns() - start) / 1e6,
                        first_chunk_staged_ms=first, strict_sha256=True,
                        scheduler_ready=False, model_loaded_by_prototype=False)


@dataclass(frozen=True)
class PredictionKey:
    model: str
    resident_process: str
    epoch: str
    kv_bytes: int
    tokens: int
    chunk_mib: int
    transport: str
    boundary: str = "fault_detected_to_scheduler_verified_ready"
    verification: str = "strict_full"
    tp: int = 2
    shared: bool = True


class OnlinePredictor:
    """EWMA per EXACT live-model workload/boundary; no cold Prefill samples."""

    def __init__(self, *, enabled=False, min_samples=2, max_age_s=300):
        if (type(min_samples) is not int or min_samples < 2
                or not math.isfinite(max_age_s) or max_age_s <= 0):
            raise Rejected("invalid_prediction_budget")
        self.enabled, self.minimum, self.max_age = enabled, min_samples, max_age_s
        self.samples = {}

    def observe(self, key, path, ms, *, model_already_resident, full_boundary, now=None):
        if self.enabled is not True:
            raise Rejected("hybrid_policy_disabled")
        if (path not in ("EC", "RE_PREFILL") or not model_already_resident or not full_boundary
                or key.boundary != "fault_detected_to_scheduler_verified_ready"
                or key.verification != "strict_full" or key.tp != 2 or key.shared is not True
                or not all((key.model, key.resident_process, key.epoch, key.transport))
                or key.kv_bytes <= 0 or key.tokens <= 0 or key.chunk_mib not in CHUNK_MIB
                or not math.isfinite(ms) or ms <= 0):
            raise Rejected("incomparable_prediction_sample")
        now = time.monotonic() if now is None else now
        if not math.isfinite(now):
            raise Rejected("invalid_sample_clock")
        prior = self.samples.get((key, path))
        if prior and now < prior[2]:
            raise Rejected("sample_clock_regressed")
        count, mean = (prior[0], prior[1]) if prior and now - prior[2] <= self.max_age else (0, ms)
        self.samples[key, path] = (count + 1, .3 * ms + .7 * mean, now)

    def choose(self, key, *, ec_committed, live_request_valid, now=None):
        result = dict(action="FAIL_CLOSED", reason="hybrid_policy_disabled", predictions_ms={})
        if self.enabled is not True:
            return result
        if not live_request_valid:
            return result | dict(reason="live_request_or_lease_invalid")
        now = time.monotonic() if now is None else now
        for path in ("EC", "RE_PREFILL"):
            sample = self.samples.get((key, path))
            if (not math.isfinite(now) or not sample or sample[0] < self.minimum
                    or not 0 <= now - sample[2] <= self.max_age):
                return result | dict(reason="missing_stale_or_incomparable_online_samples")
            result["predictions_ms"][path] = sample[1]
        action = min(result["predictions_ms"], key=result["predictions_ms"].get)
        if not ec_committed:
            action, reason = "RE_PREFILL", "ec_uncommitted_live_model_reprefill_only"
        else:
            reason = "minimum_same_boundary_online_ewma"
        return result | dict(action=action, reason=reason, key=key.__dict__)


class SameProcessSwitchPrototype:
    """Compatibility wrapper for an injected live Scheduler commit callback.

    New code uses ``LiveRecoveryHook`` directly.  This wrapper no longer claims
    to perform a switch by itself and cannot bypass the real transaction.
    """

    def __init__(self, *, enabled=False):
        self.enabled = enabled

    def submit(self, *, before, current, model_ready_ns, fault_ns,
               stage_b_prefill_count, verification_event, scheduler_commit=None):
        if self.enabled is not True:
            raise Rejected("same_process_hot_switch_disabled")
        if before != current or not before:
            raise Rejected("live_request_process_model_or_generation_changed")
        if not 0 < model_ready_ns < fault_ns or stage_b_prefill_count != 0:
            raise Rejected("hot_boundary_requires_resident_model_and_zero_stage_b_prefill")
        if verification_event is None:
            raise Rejected("required_validation_event_missing")
        verification_event.synchronize()
        if verification_event.query() is not True:
            raise Rejected("validation_event_incomplete")
        if not callable(scheduler_commit):
            raise Rejected("live_scheduler_commit_callback_required")
        result = scheduler_commit()
        if not isinstance(result, dict) or result.get("state") != "COMMITTED":
            raise Rejected("live_scheduler_commit_failed")
        return result
