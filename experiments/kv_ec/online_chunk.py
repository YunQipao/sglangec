"""Opt-in, non-overlap online capture. Only the scheduler thread reads KV.

Workers receive independent GPU/Host staging, never a request, pool or slots.
The device fence + completed gather is the lifetime boundary; a D2H event is
the host-read boundary. No inference collectives or allocator leases are added.
"""

import atexit
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import socket
import threading
import time

import numpy as np
import torch

from .checkpoint_protocol import recv_message, send_message
from .debug_capture import require, select_raw_bytes, source_identity
from .snapshot import digest

logger = logging.getLogger(__name__)
CHUNK_TOKENS = 256


def validate_runtime(s):
    a = s.server_args
    require(
        a.disable_overlap_schedule and not s.enable_overlap,
        "online_overlap_unsupported",
    )
    require(
        a.disable_radix_cache and s.tree_cache.disable,
        "online_shared_cache_unsupported",
    )
    require(
        a.pp_size == a.dp_size == a.dcp_size == a.nnodes == 1,
        "online_single_node_plain_tp_required",
    )
    require(
        a.max_running_requests == 1 and not a.enable_mixed_chunk,
        "online_single_request_required",
    )
    require(
        s.spec_algorithm.is_none() and not a.enable_dp_attention,
        "online_speculation_or_dp_unsupported",
    )
    require(
        not any(
            (
                a.enable_unified_memory,
                a.enable_page_major_kv_layout,
                a.enable_hierarchical_cache,
                a.enable_hisparse,
                a.enable_lora,
                a.cpu_offload_gb,
                getattr(a, "enable_pdmux", False),
            )
        ),
        "online_dynamic_storage_unsupported",
    )
    require(
        str(a.disaggregation_mode) == "null" and a.quantization is None,
        "online_disaggregation_or_quantization_unsupported",
    )
    require(
        a.page_size == 1 and a.kv_cache_dtype == "auto", "online_plain_slots_required"
    )
    require(
        a.cuda_graph_config.decode.backend
        == a.cuda_graph_config.prefill.backend
        == "disabled",
        "online_graphs_unsupported",
    )
    require(
        not any(
            os.environ.get(k) == "1"
            for k in ("SGLANG_KV_EC_DEBUG", "SGLANG_KV_EC_PAUSE_RESUME")
        ),
        "online_conflicting_experiment",
    )


def completed_chunks(req, batch, index, next_chunk, grain=CHUNK_TOKENS):
    """Only called after forward; sampled-but-not-forwarded tokens are excluded."""
    if (
        req.finished()
        or req.is_retracted
        or getattr(req, "to_finish", None)
        or req.req_pool_idx is None
    ):
        return range(0)
    require(
        req.session_id is None and req.multimodal_inputs is None,
        "online_text_non_session_required",
    )
    require(
        batch.forward_mode.name in ("EXTEND", "DECODE"),
        "online_forward_mode_unsupported",
    )
    written = int(batch.seq_lens_cpu[index])
    require(written == req.kv_committed_len, "online_written_length_mismatch")
    require(
        written <= len(req.origin_input_ids) + len(req.output_ids),
        "online_unmaterialized_tokens",
    )
    require(grain in (128, 256, 512, 1024), "online_invalid_capture_grain")
    return range(next_chunk, written // grain)


@dataclass
class StagedChunk:
    metadata: dict
    host: torch.Tensor
    device_copy: torch.Tensor
    done: object
    begin: object
    queued_at: float
    cancelled: object = None


@dataclass
class RequestProgress:
    req: object
    row: int
    generation: int
    next_chunk: int
    started: float
    cancelled: threading.Event


def finish_metadata(meta, payload):
    """CPU-only canonical rank-bin metadata, after D2H event completion."""
    meta["sha256"] = digest(payload)
    hashes = {k: hashlib.sha256() for k in ("k", "v")}
    for record in meta["layer_records"]:
        for kind in ("k", "v"):
            start = record[kind + "_offset"]
            segment = payload[start : start + record[kind + "_nbytes"]]
            record[kind + "_sha256"] = digest(segment)
            hashes[kind].update(segment)
    meta.update({kind + "_sha256": h.hexdigest() for kind, h in hashes.items()})
    meta["snapshot_completion_status"] = "complete"
    return meta


def gather_layers(pool, slots, tp_size, packed):
    """Gather canonical bytes into caller-owned storage; no copy of the pool.

    The caller owns the completion/lifetime fence. Kept device-agnostic so CPU
    tests can compare this exact implementation with the offline tensor layout.
    """
    records, offset = [], 0
    layer_num = int(getattr(pool, "layer_num", 36))
    require(layer_num > 0, "online_invalid_layer_count")
    for layer in range(layer_num):
        k, v = pool.get_kv_buffer(layer)
        shape = [int(slots.numel()), *map(int, k.shape[1:])]
        require(
            k.dtype == v.dtype == torch.bfloat16
            and k.device == v.device == packed.device
            and k.ndim == v.ndim == 3
            and k.is_contiguous()
            and v.is_contiguous()
            and list(k.shape[1:]) == list(v.shape[1:]) == shape[1:]
            and shape[1] > 0 and shape[2] > 0,
            "online_invalid_tensor_layout",
        )
        size = int(np.prod(shape)) * 2
        require(offset + 2 * size <= packed.numel(), "online_staging_too_small")
        records.append(
            dict(
                layer_id=layer,
                dtype=str(k.dtype),
                packed_shape={"k": shape, "v": shape},
                k_offset=offset,
                k_nbytes=size,
                v_offset=offset + size,
                v_nbytes=size,
            )
        )
        for tensor in (k, v):
            packed[offset : offset + size].copy_(select_raw_bytes(tensor, slots))
            offset += size
    require(offset == packed.numel(), "online_staging_length_mismatch")
    return records


class OnlineProtector:
    def __init__(
        self,
        scheduler,
        *,
        path,
        run_id,
        identity,
        capacity=2,
        timeout=5,
        max_bytes=64 * 1024**2,
    ):
        require(
            capacity > 0 and timeout > 0 and 0 < max_bytes <= 64 * 1024**2,
            "online_invalid_budget",
        )
        self.scheduler, self.path = scheduler, str(path)
        self.run_id, self.identity = run_id, identity
        self.enabled = True
        self.incremental_enabled = os.environ.get("SGLANG_KV_EC_INCREMENTAL", "1") == "1"
        self.async_d2h_enabled = os.environ.get("SGLANG_KV_EC_ASYNC_D2H", "1") == "1"
        self.grain = int(os.environ.get("SGLANG_KV_EC_ONLINE_GRAIN_TOKENS", "256"))
        require(self.grain in (128, 256, 512, 1024), "online_invalid_capture_grain")
        self.timeout, self.max_bytes = timeout, max_bytes
        self.queue = queue.Queue(maxsize=capacity)
        # Includes the in-flight worker, not just queued entries. Reserve before
        # allocating any Host/GPU staging memory.
        self.capacity = threading.BoundedSemaphore(capacity)
        self.stop = threading.Event()
        self.active = {}  # scheduler-only state; never passed to the worker
        self.metrics = queue.Queue(maxsize=128)
        self.protected_lock = threading.Lock()
        self.protected_chunks = {}
        self.protected_bytes = 0
        self.protected_byte_limit = int(os.environ.get(
            "SGLANG_KV_EC_LIVE_CACHE_BYTES", str(1024**3)
        ))
        require(self.protected_byte_limit > 0, "online_invalid_live_cache_budget")
        self.require_global_rs = os.environ.get(
            "SGLANG_KV_EC_REQUIRE_GLOBAL_RS", "0"
        ) == "1"
        failure_scope = os.environ.get("SGLANG_KV_EC_FAILURE_SCOPE", "node")
        require(failure_scope in ("gpu-rank", "node"), "online_invalid_failure_scope")
        require(
            failure_scope == "node" or not self.require_global_rs,
            "gpu_rank_recovery_must_not_wait_for_global_rs",
        )
        self.failure_scope = failure_scope
        self.stream = torch.cuda.Stream(
            device=scheduler.model_worker.model_runner.token_to_kv_pool.k_buffer[
                0
            ].device
        )
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="kv-ec-online-host"
        )
        self.thread.start()
        atexit.register(self.close)

    @classmethod
    def from_env(cls, scheduler):
        validate_runtime(scheduler)
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

        runner, ps = scheduler.model_worker.model_runner, scheduler.model_worker.ps
        pool = runner.token_to_kv_pool
        require(
            type(pool) is MHATokenToKVPool
            and type(scheduler.req_to_token_pool) is ReqToTokenPool,
            "online_static_mha_pool_required",
        )
        require(
            runner.req_to_token_pool is scheduler.req_to_token_pool
            and scheduler.token_to_kv_pool_allocator.get_kvcache() is pool,
            "online_pool_reference_mismatch",
        )
        require(
            ps.tp_size in (2, 4)
            and ps.attn_tp_size == ps.tp_size
            and ps.attn_cp_size == ps.attn_dcp_size == 1,
            "online_tp2_or_tp4_required",
        )
        hf = runner.model_config.hf_config
        require(
            hf.model_type == "qwen3"
            and hf.num_hidden_layers == pool.layer_num
            and 0 < hf.num_hidden_layers <= 128
            and hf.num_key_value_heads == 8
            and hf.head_dim == 128,
            "online_supported_qwen3_required",
        )
        require(
            pool.kv_cache_layout == "nhd"
            and not pool.use_hnd
            and not pool.post_capture_active
            and pool.layer_transfer_counter is None
            and not pool.is_quantized_kv_cache
            and pool.dtype == pool.store_dtype == torch.bfloat16
            and pool.start_layer == 0,
            "online_bf16_static_nhd_required",
        )
        path = os.environ.get("SGLANG_KV_EC_ONLINE_SOCKET", "")
        run_id = os.environ.get("SGLANG_KV_EC_ONLINE_RUN_ID", "")
        require(
            Path(path).is_absolute() and bool(run_id) and len(run_id) <= 96,
            "online_socket_and_unique_run_id_required",
        )
        identity = source_identity(scheduler.server_args.model_path)
        return cls(
            scheduler,
            path=path,
            run_id=run_id,
            identity=identity,
            capacity=int(os.environ.get("SGLANG_KV_EC_ONLINE_CAPACITY", "2")),
        )

    def before_inputs(self, inputs):
        # Fingerprints describe startup weights. End this protection epoch
        # before dispatching a hot update or storage release/resume.
        if any(
            type(req).__name__.startswith(
                ("UpdateWeight", "ReleaseMemoryOccupation", "ResumeMemoryOccupation")
            )
            for req in inputs
        ):
            self.enabled = False
            for state in self.active.values():
                state.cancelled.set()
            self.active.clear()
            self._metric(
                {
                    "kind": "protection_disabled",
                    "reason": "model_or_storage_epoch_changed",
                }
            )

    def after_forward(self, batch):
        # The caller is the non-overlap scheduler, before result processing.
        # Never catch a CUDA safety failure and resume with outstanding pool reads.
        if not getattr(self, "enabled", True):
            return
        self._retire_invalid()
        for i, req in enumerate(batch.reqs):
            if req.finished() or req.is_retracted or getattr(req, "to_finish", None):
                continue
            row = req.req_pool_idx
            if row is None:
                continue
            generation = int(self.scheduler.req_to_token_pool.req_generation[row])
            key = id(req)
            state = self.active.get(key)
            if state is None:
                state = RequestProgress(
                    req, row, generation, 0, time.perf_counter(), threading.Event()
                )
                self.active[key] = state
            require(state.generation == generation, "online_request_generation_changed")
            grain = getattr(self, "grain", CHUNK_TOKENS)
            for chunk in completed_chunks(req, batch, i, state.next_chunk, grain):
                if not getattr(self, "incremental_enabled", True) and chunk > 0:
                    break
                if not self.capacity.acquire(blocking=False):
                    # Retry on the next completed forward while ownership is live.
                    self._metric({"kind": "backpressure", "request_id": self._rid(req)})
                    break
                try:
                    staged = (
                        self._stage(req, chunk, generation)
                        if grain == CHUNK_TOKENS
                        else self._stage(req, chunk, generation, token_count=grain)
                    )
                    staged.cancelled = state.cancelled
                    self.queue.put_nowait(staged)
                except BaseException:
                    self.capacity.release()
                    raise
                state.next_chunk = chunk + 1

    def _rid(self, req):
        return hashlib.sha256(str(req.rid).encode()).hexdigest()

    def recover_xor_from_memory(self, snapshot_id, request_id, lost_rank, survivor):
        """Fault-path client for the manager's DRAM-only XOR reconstruction."""
        if not isinstance(survivor, bytes) or not survivor:
            raise ValueError("immutable survivor bytes required")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.path)
            send_message(sock, {
                "kind": "recover_xor", "snapshot_id": snapshot_id,
                "request_id": request_id, "lost_rank": lost_rank,
            }, survivor)
            status, raw = recv_message(sock)
        if status.get("state") != "COMMITTED" or digest(raw) != status.get("sha256"):
            raise RuntimeError("live XOR reconstruction failed")
        return raw, status

    def _checkpoint_status(self, snapshot_id):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.path)
            send_message(sock, {"kind": "status", "snapshot_id": snapshot_id}, b"")
            status, _ = recv_message(sock)
        return status

    def snapshot_prefix(self, req, boundary, *, timeout=None):
        """Return this TP rank's immutable, committed chunks for [0,boundary)."""
        if type(boundary) is not int or boundary <= 0 or boundary % self.grain:
            raise ValueError("live recovery boundary must be a complete protection chunk")
        deadline = time.monotonic() + (timeout or self.timeout)
        while self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        if self.queue.unfinished_tasks:
            raise TimeoutError("incremental protection queue did not drain")
        rid = self._rid(req)
        count = boundary // self.grain
        # Fault pause and after_forward run on adjacent scheduler turns. A
        # chunk can be complete in the KV pool while its staging worker has
        # not yet published protected_chunks. Wait until the same deadline
        # used for committed status before declaring a real missing prefix.
        chunks = None
        while time.monotonic() < deadline:
            with self.protected_lock:
                candidate = [self.protected_chunks.get((rid, i)) for i in range(count)]
            if all(item is not None for item in candidate):
                chunks = candidate
                break
            time.sleep(0.005)
        if chunks is None:
            raise TimeoutError("incremental protected prefix did not publish")
        result = []
        for meta, raw in chunks:
            while True:
                status = self._checkpoint_status(meta["snapshot_id"])
                local_ready = status.get("state") in ("COMMITTED", "LOCAL_SNAPSHOT")
                global_ready = (
                    not self.require_global_rs
                    or status.get("global_state") == "GLOBAL_COMMITTED"
                )
                if local_ready and global_ready:
                    break
                if time.monotonic() >= deadline:
                    tier = "globally" if self.require_global_rs else "locally"
                    raise TimeoutError(
                        f"incremental checkpoint not {tier} committed"
                    )
                time.sleep(0.01)
            result.append((deepcopy(meta), raw))
        return result

    def _stage(self, req, chunk, generation, token_count=CHUNK_TOKENS):
        require(type(token_count) is int and token_count in (128, 256, 512, 1024),
                "unsupported_online_capture_range")
        s = self.scheduler
        pool, rp = s.model_worker.model_runner.token_to_kv_pool, s.req_to_token_pool
        ps = s.model_worker.ps
        row, start = req.req_pool_idx, chunk * token_count
        end = start + token_count
        require(
            row > 0
            and row not in rp.free_slots
            and int(rp.req_generation[row]) == generation,
            "online_request_row_released",
        )
        layer_num = int(getattr(pool, "layer_num", 36))
        k0, v0 = pool.get_kv_buffer(0)
        expected = layer_num * token_count * int(k0[0].numel()) * 2 * 2
        require(k0.shape == v0.shape, "online_kv_shape_mismatch")
        require(expected <= self.max_bytes, "online_byte_budget_exceeded")
        device = pool.k_buffer[0].device
        require(
            device.type == "cuda" and not torch.cuda.is_current_stream_capturing(),
            "online_cuda_stream_required",
        )
        started = time.perf_counter()
        from .perf_observer import enabled as perf_enabled, emit
        perf_started = time.monotonic_ns() if perf_enabled() else None
        # Conservative stage-1 fence covers KV writes on all device streams.
        torch.cuda.synchronize(device)
        slots = rp.req_to_token[row, start:end].clone().long()
        physical = slots.cpu().tolist()
        require(
            len(physical) == token_count
            and len(set(physical)) == token_count
            and all(0 < p <= pool.size for p in physical),
            "online_invalid_slots",
        )
        packed = torch.empty(expected, dtype=torch.uint8, device=device)
        host = torch.empty(expected, dtype=torch.uint8, pin_memory=True)
        # All reads of pool tensors are on the scheduler's current stream.
        try:
            records = gather_layers(pool, slots, ps.tp_size, packed)
        finally:
            # Even a rejected partial gather must finish its pool reads before
            # propagating. No queued task ever accesses pool storage.
            torch.cuda.current_stream(device).synchronize()
        if perf_started is not None:
            emit('gpu_snapshot_gather', 'protection', start_ns=perf_started,
                 tp_rank=ps.tp_rank, includes='existing_fences,allocation,gather')
        require(
            req.req_pool_idx == row
            and int(rp.req_generation[row]) == generation
            and row not in rp.free_slots
            and torch.equal(slots, rp.req_to_token[row, start:end].long()),
            "online_ownership_changed",
        )
        tokens = (req.origin_input_ids + req.output_ids)[start:end]
        rid = self._rid(req)
        identity_text = (
            f"{self.run_id}:{rid}:{chunk}" if token_count == 256
            else f"{self.run_id}:{rid}:{start}:{end}"
        )
        checkpoint = hashlib.sha256(identity_text.encode()).hexdigest()
        meta = dict(
            self.identity,
            format="kv-ec-rank-bin-v1",
            provenance="sglang_request",
            snapshot_id=checkpoint,
            checkpoint_version=checkpoint,
            request_id=rid,
            request_id_encoding="sha256",
            protection_mode=os.environ.get("SGLANG_KV_EC_ONLINE_PROTECTION_MODE", "xor"),
            request_version=self.run_id,
            manager_epoch=self.run_id,
            model_version=self.identity["model_sha256"],
            chunk_id=chunk,
            capture_grain_tokens=token_count,
            token_range=[start, end],
            layer_range=[0, layer_num],
            tp_size=ps.tp_size,
            tp_rank=ps.tp_rank,
            layers=list(range(layer_num)),
            token_start=start,
            token_count=token_count,
            token_ids_sha256=hashlib.sha256(
                np.asarray(tokens, dtype="<i8").tobytes()
            ).hexdigest(),
            dtype=str(pool.dtype),
            packed_byte_count=expected,
            layer_records=records,
            req_pool_idx=row,
            req_generation=generation,
            physical_slots=physical,
            pool_class=type(pool).__module__ + "." + type(pool).__name__,
            source_unchanged=True,
            snapshot_completion_status="copy_pending",
            lifetime="non_overlap_before_result_free_independent_gpu_staging",
            completion_dependency="device_fence_gather_fence_then_private_stream_d2h_event",
            performance_ms={
                "scheduler_stage_ms": (time.perf_counter() - started) * 1000
            },
        )
        begin, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        with torch.cuda.stream(self.stream):
            begin.record(self.stream)
            async_d2h = getattr(self, "async_d2h_enabled", True)
            host.copy_(packed, non_blocking=async_d2h)
            done.record(self.stream)
            packed.record_stream(self.stream)
        if not async_d2h:
            # Explicit comparison path for --disable-opt async-d2h.
            done.synchronize()
        return StagedChunk(meta, host, packed, done, begin, time.perf_counter())

    def after_result(self, batch):
        self._retire_invalid()

    def _retire_invalid(self):
        rp = self.scheduler.req_to_token_pool
        for key, state in list(self.active.items()):
            req = state.req
            invalid = (
                req.req_pool_idx != state.row
                or state.row in rp.free_slots
                or int(rp.req_generation[state.row]) != state.generation
            )
            reason = getattr(req, "finished_reason", None)
            cancelled = bool(
                req.is_retracted
                or getattr(req, "to_finish", None)
                or type(reason).__name__ == "FINISH_ABORT"
            )
            if req.finished() or cancelled or invalid:
                self.active.pop(key)
                # A normal completed request drains already detached snapshots.
                # An abort/retraction invalidates queued protection attempts.
                if cancelled or (invalid and not req.finished()):
                    state.cancelled.set()
                now = time.perf_counter()
                arrival = getattr(
                    getattr(req, "time_stats", None), "wait_queue_entry_time", 0
                )
                self._metric(
                    {
                        "kind": "request_end",
                        "request_id": self._rid(req),
                        "observation_to_end_ms": (now - state.started) * 1000,
                        "scheduler_request_ms": (
                            (now - arrival) * 1000 if arrival else None
                        ),
                        "submitted_chunks": state.next_chunk,
                        "unsubmitted_complete_chunks": max(
                            0, req.kv_committed_len // getattr(self, "grain", CHUNK_TOKENS) - state.next_chunk
                        ),
                        "cancelled": state.cancelled.is_set(),
                    }
                )

    def _metric(self, metric):
        try:
            self.metrics.put_nowait(metric)
        except queue.Full:
            pass  # bounded diagnostic sampling must not stall inference

    def _run(self):
        while (
            not self.stop.is_set() or not self.queue.empty() or not self.metrics.empty()
        ):
            try:
                item = self.queue.get(timeout=0.05)
            except queue.Empty:
                self._flush_metrics()
                continue
            try:
                meta = item.metadata
                payload = None
                meta["performance_ms"]["queue_wait_ms"] = (
                    time.perf_counter() - item.queued_at
                ) * 1000
                item.done.synchronize()
                if item.cancelled is not None and item.cancelled.is_set():
                    logger.info(
                        "KV_EC_ONLINE rank=%d chunk=%d state=CANCELLED",
                        meta["tp_rank"],
                        meta["chunk_id"],
                    )
                    continue
                meta["performance_ms"]["gpu_to_host_ms"] = item.begin.elapsed_time(
                    item.done
                )
                payload = item.host.numpy()
                finish_metadata(meta, payload)
                # Socket and serialization happen only after the worker owns
                # completed Host bytes. No re-read after request end/reuse.
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout)
                    sock.connect(self.path)
                    send_message(sock, meta, payload.tobytes())
                    status, _ = recv_message(sock)
                if status.get("state") in ("PENDING", "COMMITTED", "LOCAL_SNAPSHOT"):
                    immutable = payload.tobytes()
                    key = (meta["request_id"], meta["chunk_id"])
                    with self.protected_lock:
                        old = self.protected_chunks.get(key)
                        added = len(immutable) - (len(old[1]) if old else 0)
                        if self.protected_bytes + added > self.protected_byte_limit:
                            raise RuntimeError("live incremental cache byte budget exhausted")
                        self.protected_chunks[key] = (deepcopy(meta), immutable)
                        self.protected_bytes += added
                logger.info(
                    "KV_EC_ONLINE rank=%d chunk=%d result=%s metrics=%s",
                    meta["tp_rank"],
                    meta["chunk_id"],
                    json.dumps(status),
                    json.dumps(meta["performance_ms"]),
                )
            except Exception as exc:
                logger.warning("KV_EC_ONLINE task failed (%s)", type(exc).__name__)
            finally:
                # Do not retain the previous iteration's pinned buffer while
                # admitting another task into the bounded staging pool.
                del item, payload
                self.queue.task_done()
                self.capacity.release()
                self._flush_metrics()

    def _flush_metrics(self):
        while not self.metrics.empty():
            logger.info("KV_EC_ONLINE metrics=%s", json.dumps(self.metrics.get()))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=self.timeout + 1)
