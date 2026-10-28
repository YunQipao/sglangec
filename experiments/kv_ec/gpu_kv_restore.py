"""Explicit GPU KV checkpoint restore for the single-request TP2 experiment.

This module is deliberately not imported by the normal scheduler.  It is a
fail-closed experiment hook: callers provide a scheduler, a live request and a
complete ``global_node.bin`` for *this* TP rank.  New KV slots are allocated,
written on a private CUDA stream, byte-verified, and only then installed in the
request row.  No prefill or model forward is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Protocol

import torch

from .node_block_export import unpack, sha
from .perf_observer import emit


class RestoreRejected(ValueError):
    """Checkpoint, allocator, ownership, or byte verification failed."""


class CheckpointSource(Protocol):
    def load(self) -> bytes: ...


@dataclass(frozen=True)
class LocalCheckpointSource:
    path: Path
    expected_sha256: str | None = None

    def load(self) -> bytes:
        block = self.path.read_bytes()
        if self.expected_sha256 is not None:
            _require(sha(block) == self.expected_sha256, "checkpoint_source_changed")
        return block


@dataclass(frozen=True)
class BytesCheckpointSource:
    payload: bytes

    def load(self) -> bytes:
        return bytes(self.payload)


@dataclass
class RestoreStats:
    source: str
    node_id: int
    tp_rank: int
    tp_size: int
    token_start: int
    token_count: int
    loaded_kv_tokens: int
    written_bytes: int
    recomputed_tokens: int = 0
    unexpected_prefill: bool = False
    h2d_ms: float = 0.0
    verify_ms: float = 0.0
    layers_verified: list = field(default_factory=list)


class AllocatorSavepoint:
    """Undo journal for the isolated scheduler thread, NOT a concurrent allocator.

    Tensor free lists are replaced, never mutated in place by this allocator.
    Retain their references before alloc, including the lease generations. This
    covers injected exceptions even *after* alloc/free published their change.
    The row guard and the blocking TP votes must remain held until finish/abort.
    """

    def __init__(self, allocator):
        _require(allocator.is_not_in_free_group and not allocator.free_group,
                 "restore_free_group_active")
        _require(not getattr(allocator, "_kv_ec_restore_active", False),
                 "restore_allocator_already_owned")
        self.allocator = allocator
        self.free_pages = allocator.free_pages
        self.release_pages = allocator.release_pages
        self.generation = allocator.slot_lease_registry.generation.clone()
        _require(allocator.slot_lease_registry.active_lease_count == 0,
                 "restore_active_slot_leases")
        allocator._kv_ec_restore_active = True

    def restore(self):
        a = self.allocator
        _require(a._kv_ec_restore_active, "restore_allocator_guard_lost")
        a.free_pages, a.release_pages = self.free_pages, self.release_pages
        a.is_not_in_free_group, a.free_group = True, []
        a.slot_lease_registry.generation.copy_(self.generation)

    def close(self):
        self.allocator._kv_ec_restore_active = False
        # Do not retain superseded CUDA index tensors / CPU generation clones
        # for the lifetime of the experiment's diagnostic transaction object.
        self.free_pages = self.release_pages = self.generation = None


@dataclass
class RestoreTransaction:
    """A committed mapping replacement, or an explicitly abortable prepare."""

    allocator: object
    req_pool: object
    req: object
    row: int
    generation: int
    old_slots: tuple[int, ...]
    new_slots: tuple[int, ...]
    token_start: int
    token_count: int
    stats: RestoreStats
    state: str = "PREPARED"
    savepoint: object = None
    fence: object = None
    guard_owner: str | None = None
    mapping_installed: bool = False
    ownership_check: object = None
    finalized: bool = False
    # First installation has no old physical owner. Stale row bytes are an
    # undo image only, never slots to free or evidence of ownership.
    initial_install: bool = False
    prior_mapping: tuple[int, ...] | None = None

    def _identity(self):
        _require(self.req.req_pool_idx == self.row and self.row not in self.req_pool.free_slots,
                 "request_row_changed_before_commit")
        _require(int(self.req_pool.req_generation[self.row]) == self.generation,
                 "request_generation_changed_before_commit")
        if self.guard_owner is not None:
            _require(getattr(self.req_pool, "_kv_ec_row_owners", {}).get(self.row)
                     == self.guard_owner, "restore_row_guard_lost")

    def _mapping(self):
        return self.req_pool.req_to_token[
            self.row, self.token_start:self.token_start + self.token_count]

    def _fence(self):
        if self.fence is not None:
            # Also used on exceptions before an event could be recorded.
            # A failed fence deliberately quarantines the slots (fail closed).
            self.fence()

    def _close(self):
        if self.guard_owner is not None:
            self.req_pool.release_kv_ec_row(self.row, self.generation, self.guard_owner)
            self.guard_owner = None
        if self.savepoint is not None:
            self.savepoint.close()
        self.ownership_check = None

    def install_mapping(self):
        _require(self.state == "VERIFIED", "restore_commit_requires_verified")
        _require(self.savepoint is not None and self.allocator._kv_ec_restore_active,
                 "restore_commit_requires_undo_journal")
        self._identity()
        if self.ownership_check is not None:
            self.ownership_check()
        expected = self.prior_mapping if self.initial_install else self.old_slots
        _require(tuple(self._mapping().cpu().tolist()) == expected,
                 "request_slot_mapping_changed_before_commit")
        _require(not set(self.old_slots) & set(self.new_slots), "restore_slot_alias")
        self._fence()
        # Mark first: a throwing/asynchronous assignment is still compensatable.
        self.mapping_installed = True
        self.state = "MAPPING_PENDING"
        self._mapping().copy_(torch.tensor(self.new_slots, dtype=torch.int32,
                                           device=self.req_pool.device))
        if self.initial_install:
            self.req.kv_committed_len = self.token_count
            self.req.kv.kv_allocated_len = self.token_count
        self._fence()
        self.state = "MAPPED"

    def commit(self, *, retain_rollback: bool = False) -> None:
        """Local commit; TP coordinator keeps the journal until BOTH commits vote.

        A transaction without the exclusive undo journal cannot change mappings.
        """
        _require(not self.finalized and self.state in ("VERIFIED", "MAPPED"),
                 "restore_commit_requires_verified_or_mapped")
        _require(self.savepoint is not None and self.allocator._kv_ec_restore_active,
                 "restore_commit_requires_undo_journal")
        try:
            if self.state == "VERIFIED":
                self.install_mapping()
            _require(self.state == "MAPPED", "restore_commit_requires_mapped")
            self._identity()
            if self.ownership_check is not None:
                self.ownership_check()
            _require(tuple(self._mapping().cpu().tolist()) == self.new_slots,
                     "restore_mapping_changed_after_install")
            if self.old_slots:
                self.allocator.free(torch.tensor(self.old_slots, dtype=torch.int64,
                                                 device=self.allocator.device))
            self._fence()
        except BaseException:
            self.state = "COMMIT_FAILED"
            self.abort()
            raise
        self.state = "COMMITTED"
        if not retain_rollback:
            self.finish()

    def finish(self):
        _require(self.state == "COMMITTED", "restore_finish_requires_commit")
        self._close()
        self.finalized = True

    def abort(self) -> None:
        if self.state == "ABORTED":
            return
        if self.finalized or (self.savepoint is not None and
                              not self.allocator._kv_ec_restore_active):
            raise RestoreRejected("cannot_abort_committed_restore")
        if self.state == "COMMITTED" and self.savepoint is None:
            raise RestoreRejected("cannot_abort_committed_restore")
        self._fence()
        if self.mapping_installed:
            self._identity()
            previous = self.prior_mapping if self.initial_install else self.old_slots
            self._mapping().copy_(torch.tensor(previous, dtype=torch.int32,
                                               device=self.req_pool.device))
            if self.initial_install:
                self.req.kv_committed_len = self.req.kv.kv_allocated_len = 0
            self._fence()
            self.mapping_installed = False
        if self.savepoint is not None:
            self.savepoint.restore()
        else:
            self.allocator.free(torch.tensor(self.new_slots, dtype=torch.int64,
                                             device=self.allocator.device))
        self.state = "ABORTED"
        try:
            self._close()
        except BaseException:
            self.state = "QUARANTINED"
            raise


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RestoreRejected(reason)


def _metadata_identity(meta: dict, scheduler, req, *, rank: int, tp_size: int,
                       token_start: int, token_count: int) -> None:
    """Check the immutable fields before allocating any new GPU slot."""
    ps = scheduler.model_worker.ps
    pool = scheduler.model_worker.model_runner.token_to_kv_pool
    _require(ps.tp_rank == rank and ps.tp_size == tp_size, "tp_rank_or_size_mismatch")
    _require(tp_size == 2, "restore_experiment_requires_tp2")
    _require(meta["tp_rank"] == rank and meta["tp_size"] == tp_size,
             "checkpoint_rank_or_size_mismatch")
    _require(meta["dtype"] == "torch.bfloat16", "checkpoint_dtype_mismatch")
    _require(meta["layers"] == list(range(36)) and pool.layer_num == 36,
             "checkpoint_layer_layout_mismatch")
    _require(meta["token_start"] == token_start and meta["token_count"] == token_count,
             "checkpoint_token_range_mismatch")
    _require(token_count in (256, 512, 768), "unsupported_restore_token_count")
    _require(meta["snapshot_completion_status"] == "complete" and
             meta.get("source_unchanged") is True, "checkpoint_incomplete")
    _require(meta["packed_byte_count"] > 0, "checkpoint_empty")
    _require(pool.kv_cache_layout == "nhd" and not pool.use_hnd and
             not pool.is_quantized_kv_cache and pool.dtype == pool.store_dtype == torch.bfloat16,
             "runtime_kv_layout_unsupported")
    _require(pool.start_layer == 0 and pool.page_size == 1,
             "runtime_pool_must_be_dense_page1")
    hf = scheduler.model_worker.model_runner.model_config.hf_config
    _require(hf.model_type == "qwen3" and hf.num_key_value_heads == 8 and
             hf.head_dim == 128 and hf.hidden_size == 4096 and
             hf.num_attention_heads == 32, "runtime_model_layout_mismatch")
    # A source identity is required; when the scheduler exposes the capture
    # identity, compare content fingerprints and source provenance exactly.
    expected = getattr(scheduler, "kv_ec_source_identity", None)
    if expected:
        for key in ("model_sha256", "source_commit", "core_diff_sha256"):
            _require(meta.get(key) == expected.get(key), "checkpoint_model_identity_mismatch")


def _validate_segments(raw: bytes, meta: dict, *, token_count: int, tp_size: int) -> None:
    _require(sha(raw) == meta["sha256"], "checkpoint_payload_sha256_mismatch")
    expected_heads = 8 // tp_size
    offset = 0
    kh = hashlib.sha256()
    vh = hashlib.sha256()
    for record in meta["layer_records"]:
        _require(record["dtype"] == "torch.bfloat16", "checkpoint_layer_dtype_mismatch")
        for kind in ("k", "v"):
            shape = record["packed_shape"][kind]
            nbytes = int(record[f"{kind}_nbytes"])
            _require(shape == [token_count, expected_heads, 128],
                     "checkpoint_packed_shape_mismatch")
            _require(record[f"{kind}_offset"] == offset and offset + nbytes <= len(raw),
                     "checkpoint_segment_offset_mismatch")
            segment = raw[offset : offset + nbytes]
            _require(sha(segment) == record[f"{kind}_sha256"],
                     "checkpoint_segment_sha256_mismatch")
            (kh if kind == "k" else vh).update(segment)
            offset += nbytes
    _require(offset == len(raw) and kh.hexdigest() == meta["k_sha256"] and
             vh.hexdigest() == meta["v_sha256"], "checkpoint_aggregate_sha256_mismatch")


def _copy_layer(pool, layer_id: int, slots: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, stream: object) -> None:
    """Use the pool's normal writer, with a null RadixAttention only for layer id."""
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    with torch.cuda.stream(stream):
        # MHATokenToKVPool.set_kv_buffer only needs layer.layer_id when a layer
        # override is absent; passing None plus the explicit override preserves
        # the allocator/pool writer and avoids fabricating a model layer object.
        pool.set_kv_buffer(None, KVWriteLoc(slots), k, v,
                           layer_id_override=layer_id)


def _read_layer(pool, layer_id: int, slots: torch.Tensor, stream: object) -> bytes:
    with torch.cuda.stream(stream):
        k, v = pool.get_kv_buffer(layer_id)
        return b"".join((k.index_select(0, slots).contiguous().view(torch.uint8).cpu().numpy().tobytes(),
                         v.index_select(0, slots).contiguous().view(torch.uint8).cpu().numpy().tobytes()))


def validate_live_ownership(scheduler, req, token_start, token_count, reserved=()):
    """Dense page1, private single request: no shared prefix or unknown owner.

    Free rows contain stale mappings; never infer ownership from those bytes.
    All allocated rows must be accounted for and this experiment admits one.
    """
    rp, a = scheduler.req_to_token_pool, scheduler.token_to_kv_pool_allocator
    row = req.req_pool_idx
    _require(row is not None and 0 < row < rp.req_to_token.shape[0],
             "request_row_not_allocated")
    active = set(range(1, rp.req_to_token.shape[0])) - set(rp.free_slots)
    _require(active == {row}, "restore_requires_exclusive_request_row")
    _require(int(rp.req_generation[row]) > 0, "request_row_invalid")
    length = int(req.kv_committed_len)
    _require(token_start >= 0 and token_start + token_count <= length,
             "restore_unmaterialized_tokens")
    _require(req.kv is not None and req.kv.kv_allocated_len == length and
             getattr(req, "cache_protected_len", 0) == 0,
             "restore_shared_or_overallocated_slots")
    slots = tuple(int(x) for x in rp.req_to_token[row, :length].cpu().tolist())
    _require(len(slots) == length and len(set(slots)) == length and
             all(0 < x <= a.size for x in slots), "request_slot_mapping_invalid")
    free_values = a.free_pages.cpu().tolist() + a.release_pages.cpu().tolist()
    free = set(free_values)
    _require(len(free_values) == len(free) and all(0 < x <= a.size for x in free),
             "restore_allocator_free_list_invalid")
    _require(not free.intersection(slots), "request_slots_not_owned")
    # With no radix cache, every allocated physical slot must belong to this row.
    _require(not set(reserved).intersection(free) and not set(reserved).intersection(slots)
             and len(free) + len(slots) + len(reserved) == a.size,
             "restore_unknown_physical_owner")
    return slots[token_start:token_start + token_count]


def allocate_restore_slots(allocator, count, old_slots):
    """Catch allocation exceptions, short/invalid returns and post-mutation faults."""
    savepoint = AllocatorSavepoint(allocator)
    try:
        tensor = allocator.alloc(count)
        _require(tensor is not None and tensor.ndim == 1 and len(tensor) == count
                 and tensor.dtype == torch.int64,
                 "restore_slots_insufficient")
        slots = tuple(int(x) for x in tensor.cpu().tolist())
        free_before = set(savepoint.free_pages.cpu().tolist()) | set(
            savepoint.release_pages.cpu().tolist())
        _require(len(set(slots)) == count and all(0 < x <= allocator.size for x in slots)
                 and set(slots).issubset(free_before) and not set(slots) & set(old_slots),
                 "allocator_returned_invalid_slots")
        after_values = allocator.free_pages.cpu().tolist() + allocator.release_pages.cpu().tolist()
        free_after = set(after_values)
        _require(len(free_after) == len(after_values) and free_after == free_before - set(slots),
                 "allocator_allocation_accounting_mismatch")
        return slots, savepoint
    except BaseException:
        # No writes have been launched here, but alloc/registry may use CUDA.
        if str(allocator.device).startswith("cuda"):
            torch.cuda.synchronize(allocator.device)
        savepoint.restore()
        savepoint.close()
        raise


def validate_initial_ownership(scheduler, req, reserved=()):
    """A freshly allocated, exclusive row with ZERO owned KV, not fake old KV."""
    rp, a = scheduler.req_to_token_pool, scheduler.token_to_kv_pool_allocator
    row = req.req_pool_idx
    _require(row is not None and 0 < row < rp.req_to_token.shape[0] and
             set(range(1, rp.req_to_token.shape[0])) - set(rp.free_slots) == {row}
             and int(rp.req_generation[row]) > 0, "cold_restore_requires_exclusive_row")
    _require(req.kv_committed_len == 0 and req.kv is not None and
             req.kv.kv_allocated_len == 0 and req.cache_protected_len == 0 and
             len(req.prefix_indices) == 0, "cold_restore_request_already_has_kv")
    free = a.free_pages.cpu().tolist() + a.release_pages.cpu().tolist()
    _require(len(free) == len(set(free)) and len(reserved) == len(set(reserved)) and
             not set(free).intersection(reserved) and
             set(free).union(reserved) == set(range(1, a.size + 1)),
             "cold_restore_unknown_physical_owner")
    return ()


def restore_to_gpu(scheduler, req, source: CheckpointSource, *, token_start: int = 0,
                   stream=None, source_name: str = "checkpoint",
                   initial_install: bool = False) -> RestoreTransaction:
    """Prepare and verify a real GPU writeback; caller explicitly commits mapping."""
    if not torch.cuda.is_available():
        raise RestoreRejected("cuda_required_for_gpu_restore")
    block = source.load()
    meta, rank_blocks = unpack(block)
    rank = int(scheduler.model_worker.ps.tp_rank)
    tp_size = int(scheduler.model_worker.ps.tp_size)
    raw = rank_blocks[rank] if rank < len(rank_blocks) else b""
    token_count = int(meta["ranks"][rank]["token_count"])
    rank_meta = meta["ranks"][rank]
    _metadata_identity(rank_meta, scheduler, req, rank=rank, tp_size=tp_size,
                       token_start=token_start, token_count=token_count)
    _validate_segments(raw, rank_meta, token_count=token_count, tp_size=tp_size)
    req_pool = scheduler.req_to_token_pool
    allocator = scheduler.token_to_kv_pool_allocator
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    _require(type(allocator) is TokenToKVPoolAllocator, "restore_allocator_unsupported")
    _require(scheduler.server_args.disable_radix_cache and not scheduler.enable_overlap,
             "restore_requires_private_nonoverlap_pool")
    pool = scheduler.model_worker.model_runner.token_to_kv_pool
    _require(allocator.get_kvcache() is pool and allocator.size == pool.size,
             "restore_allocator_pool_mismatch")
    device = pool.k_buffer[0].device
    torch.cuda.synchronize(device)
    if initial_install:
        _require(token_start == 0, "cold_restore_requires_prefix_start_zero")
        old = validate_initial_ownership(scheduler, req)
    else:
        old = validate_live_ownership(scheduler, req, token_start, token_count)
    row = int(req.req_pool_idx)
    generation = int(req_pool.req_generation[row])
    owner = f"gpu-restore:{row}:{generation}"
    req_pool.protect_kv_ec_row(row, generation, owner)
    tx = RestoreTransaction(allocator, req_pool, req, row, generation, old, (),
                            token_start, token_count,
                            RestoreStats(source_name, meta["node_id"], rank, tp_size,
                                         token_start, token_count, 0, 0),
                            fence=lambda: torch.cuda.synchronize(device), guard_owner=owner,
                            initial_install=initial_install)
    try:
        if initial_install:
            tx.prior_mapping = tuple(tx._mapping().cpu().tolist())
        new, tx.savepoint = allocate_restore_slots(allocator, token_count, old)
        tx.new_slots = new
        def ownership():
            if initial_install and not tx.mapping_installed:
                return validate_initial_ownership(scheduler, req, reserved=tx.new_slots)
            return validate_live_ownership(scheduler, req, token_start, token_count,
                reserved=tx.old_slots if tx.mapping_installed else tx.new_slots)
        tx.ownership_check = ownership
        stream = stream or torch.cuda.Stream(device=device)
        event = torch.cuda.Event()
        started = time.perf_counter()
        perf_started = time.monotonic_ns()
        gpu_slots = torch.tensor(new, dtype=torch.int64, device=device)
        # Slots created on the current stream must be ready on the private writer.
        torch.cuda.synchronize(device)
        offset = 0
        with torch.cuda.stream(stream):
            for record in rank_meta["layer_records"]:
                k_n = int(record["k_nbytes"])
                v_n = int(record["v_nbytes"])
                k = torch.frombuffer(bytearray(raw[offset:offset + k_n]), dtype=torch.bfloat16)
                offset += k_n
                v = torch.frombuffer(bytearray(raw[offset:offset + v_n]), dtype=torch.bfloat16)
                offset += v_n
                k = k.reshape(token_count, 8 // tp_size, 128).to(pool.k_buffer[0].device)
                v = v.reshape(token_count, 8 // tp_size, 128).to(pool.v_buffer[0].device)
                _copy_layer(pool, int(record["layer_id"]), gpu_slots, k, v, stream)
            event.record(stream)
        event.synchronize()
        tx.stats.written_bytes = len(raw)
        tx.stats.h2d_ms = (time.perf_counter() - started) * 1000
        emit("gpu_kv_writeback", "recovery", start_ns=perf_started,
             tp_rank=rank, bytes=len(raw), includes="host_tensor_prepare,H2D,scatter,existing_fences")
        verify_start = time.perf_counter()
        perf_verify_started = time.monotonic_ns()
        cursor = 0
        for record in rank_meta["layer_records"]:
            got = _read_layer(pool, int(record["layer_id"]),
                              gpu_slots, stream)
            expected = raw[cursor:cursor + int(record["k_nbytes"]) + int(record["v_nbytes"])]
            _require(got == expected, "gpu_kv_writeback_byte_mismatch")
            kn = int(record["k_nbytes"])
            tx.stats.layers_verified.append(dict(layer_id=record["layer_id"],
                k_byte_exact=True, v_byte_exact=True,
                k_sha256=sha(got[:kn]), v_sha256=sha(got[kn:])))
            cursor += len(expected)
        torch.cuda.synchronize(pool.k_buffer[0].device)
        tx.stats.verify_ms = (time.perf_counter() - verify_start) * 1000
        emit("gpu_layer_readback_validation", "validation", start_ns=perf_verify_started,
             tp_rank=rank, layers=len(tx.stats.layers_verified))
        tx.stats.loaded_kv_tokens = token_count
        tx.state = "VERIFIED"
        return tx
    except BaseException as exc:
        try:
            tx.abort()
        except BaseException as cleanup:
            tx.state = "QUARANTINED"
            raise RestoreRejected("restore_cleanup_failed_slots_quarantined") from cleanup
        # Preserve diagnostics without handing callers a live, unowned tensor.
        exc.restore_stats = tx.stats
        raise


def writeback_cli_help() -> str:
    return "python -m experiments.kv_ec.verify_gpu_restore; global commit requires BOTH TP ranks VERIFIED"
