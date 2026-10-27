"""Explicit GPU KV checkpoint restore for the single-request TP2 experiment.

This module is deliberately not imported by the normal scheduler.  It is a
fail-closed experiment hook: callers provide a scheduler, a live request and a
complete ``global_node.bin`` for *this* TP rank.  New KV slots are allocated,
written on a private CUDA stream, byte-verified, and only then installed in the
request row.  No prefill or model forward is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Protocol

import torch

from .node_block_export import unpack, sha


class RestoreRejected(ValueError):
    """Checkpoint, allocator, ownership, or byte verification failed."""


class CheckpointSource(Protocol):
    def load(self) -> bytes: ...


@dataclass(frozen=True)
class LocalCheckpointSource:
    path: Path

    def load(self) -> bytes:
        return self.path.read_bytes()


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

    def commit(self) -> None:
        if self.state != "VERIFIED":
            raise RestoreRejected("restore_commit_requires_verified")
        if self.req.req_pool_idx != self.row or self.row in self.req_pool.free_slots:
            raise RestoreRejected("request_row_changed_before_commit")
        if int(self.req_pool.req_generation[self.row]) != self.generation:
            raise RestoreRejected("request_generation_changed_before_commit")
        if set(self.old_slots) & set(self.new_slots):
            raise RestoreRejected("restore_slot_alias")
        self.req_pool.req_to_token[
            self.row, self.token_start : self.token_start + self.token_count
        ] = torch.tensor(self.new_slots, dtype=torch.int32, device=self.req_pool.device)
        try:
            if self.old_slots:
                self.allocator.free(torch.tensor(self.old_slots, dtype=torch.int64,
                                                 device=self.allocator.device))
        except Exception:
            self.state = "COMMIT_RELEASE_FAILED"
            raise
        self.state = "COMMITTED"

    def abort(self) -> None:
        if self.state in ("COMMITTED", "COMMIT_RELEASE_FAILED"):
            raise RestoreRejected("cannot_abort_committed_restore")
        if self.state == "ABORTED":
            return
        self.allocator.free(torch.tensor(self.new_slots, dtype=torch.int64,
                                         device=self.allocator.device))
        self.state = "ABORTED"


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
             hf.head_dim == 128, "runtime_model_layout_mismatch")
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


def restore_to_gpu(scheduler, req, source: CheckpointSource, *, token_start: int = 0,
                   stream=None, source_name: str = "checkpoint") -> RestoreTransaction:
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
    _require(req.req_pool_idx is not None and req.req_pool_idx > 0,
             "request_row_not_allocated")
    req_pool = scheduler.req_to_token_pool
    row = int(req.req_pool_idx)
    generation = int(req_pool.req_generation[row])
    _require(row not in req_pool.free_slots and generation > 0, "request_row_invalid")
    old = tuple(int(x) for x in req_pool.req_to_token[row, token_start:token_start + token_count].cpu().tolist())
    _require(len(old) == token_count and len(set(old)) == token_count and
             all(slot > 0 for slot in old), "request_slot_mapping_invalid")
    allocator = scheduler.token_to_kv_pool_allocator
    new_tensor = allocator.alloc(token_count)
    if new_tensor is None or len(new_tensor) != token_count:
        raise RestoreRejected("restore_slots_insufficient")
    new = tuple(int(x) for x in new_tensor.cpu().tolist())
    _require(len(set(new)) == token_count and all(x > 0 for x in new), "allocator_returned_invalid_slots")
    pool = scheduler.model_worker.model_runner.token_to_kv_pool
    stream = stream or torch.cuda.Stream(device=pool.k_buffer[0].device)
    event = torch.cuda.Event()
    started = time.perf_counter()
    tx = RestoreTransaction(allocator, req_pool, req, row, generation, old, new,
                            token_start, token_count,
                            RestoreStats(source_name, meta["node_id"], rank, tp_size,
                                         token_start, token_count, token_count, len(raw)))
    try:
        offset = 0
        with torch.cuda.stream(stream):
            for record in rank_meta["layer_records"]:
                k_n = int(record["k_nbytes"])
                v_n = int(record["v_nbytes"])
                k = torch.frombuffer(memoryview(raw[offset:offset + k_n]), dtype=torch.bfloat16).clone()
                offset += k_n
                v = torch.frombuffer(memoryview(raw[offset:offset + v_n]), dtype=torch.bfloat16).clone()
                offset += v_n
                k = k.reshape(token_count, 8 // tp_size, 128).to(pool.k_buffer[0].device)
                v = v.reshape(token_count, 8 // tp_size, 128).to(pool.v_buffer[0].device)
                _copy_layer(pool, int(record["layer_id"]), torch.tensor(new, device=pool.k_buffer[0].device), k, v, stream)
            event.record(stream)
        event.synchronize()
        tx.stats.h2d_ms = (time.perf_counter() - started) * 1000
        verify_start = time.perf_counter()
        cursor = 0
        for record in rank_meta["layer_records"]:
            got = _read_layer(pool, int(record["layer_id"]),
                              torch.tensor(new, device=pool.k_buffer[0].device), stream)
            expected = raw[cursor:cursor + int(record["k_nbytes"]) + int(record["v_nbytes"])]
            _require(got == expected, "gpu_kv_writeback_byte_mismatch")
            cursor += len(expected)
        torch.cuda.synchronize(pool.k_buffer[0].device)
        tx.stats.verify_ms = (time.perf_counter() - verify_start) * 1000
        tx.state = "VERIFIED"
        return tx
    except Exception:
        tx.abort()
        raise


def writeback_cli_help() -> str:
    return "restore_to_gpu(scheduler, req, LocalCheckpointSource(path)); commit only after VERIFIED"
