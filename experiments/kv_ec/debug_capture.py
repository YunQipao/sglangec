"""Read-only, single-request non-overlap prefill capture. No collectives.

Only called by the result processor before token append and cache/free. Explicit
guards reject unsupported lifetimes before touching any mapping/KV Tensor.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import numpy as np
import torch

from .snapshot import digest
from .rank_snapshot import write_rank_files

logger = logging.getLogger(__name__)
PROCESS_EPOCH = str(uuid.uuid4())
# Each TP worker has its own Python process. Keep attempt state here instead of
# mutating the frozen+slots result processor.
_ATTEMPTED = set()
_SOURCE_IDENTITY_CACHE = {}
_PARITY_CLIENTS = {}


class CaptureRejected(ValueError):
    """Messages must be fixed reason codes; never include request contents."""


def require(condition, reason):
    if not condition:
        raise CaptureRejected(reason)


@dataclass(frozen=True)
class CaptureConfig:
    output: Path
    snapshot_id: str
    target_rid: str
    token_start: int
    token_count: int
    layers_spec: str
    byte_budget: int

    @classmethod
    def from_env(cls):
        snapshot_id = os.environ.get("SGLANG_KV_EC_SNAPSHOT_ID", "")
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,96}", snapshot_id)), "invalid_snapshot_id")
        output = Path(os.environ.get("SGLANG_KV_EC_OUTPUT_DIR", ""))
        require(output.is_absolute(), "absolute_output_dir_required")
        rid = os.environ.get("SGLANG_KV_EC_REQUEST_ID", "")
        require(bool(rid), "explicit_target_request_id_required")
        start = int(os.environ.get("SGLANG_KV_EC_TOKEN_START", "0"))
        count = int(os.environ.get("SGLANG_KV_EC_TOKEN_COUNT", "128"))
        budget = int(os.environ.get("SGLANG_KV_EC_MAX_BYTES", str(64 * 1024**2)))
        require(start >= 0 and 0 < count <= 4096 and budget > 0, "invalid_range_or_budget")
        return cls(output, snapshot_id, rid, start, count,
                   os.environ.get("SGLANG_KV_EC_LAYERS", "0"), budget)
    def layers(self, layer_count):
        layers = list(range(layer_count)) if self.layers_spec == "all" else [int(x) for x in self.layers_spec.split(",")]
        require(bool(layers) and layers == sorted(set(layers)), "layers_must_be_sorted_unique")
        require(all(0 <= i < layer_count for i in layers), "invalid_layer")
        return layers


def protection_mode():
    mode = os.environ.get("SGLANG_KV_EC_PROTECTION_MODE", "DEBUG_SAFE").upper()
    require(mode in ("DEBUG_SAFE", "FAST_PARITY"), "invalid_protection_mode")
    return mode


def validate_lifetime(scheduler, batch, req, config):
    """CPU-only guards; these do NOT themselves establish GPU completion."""
    a = scheduler.server_args
    require(a.disable_overlap_schedule and not scheduler.enable_overlap, "overlap_not_supported")
    require(a.disable_radix_cache and scheduler.tree_cache.disable, "prefix_cache_not_supported")
    require(a.chunked_prefill_size == -1 and not a.enable_mixed_chunk, "chunked_or_mixed_not_supported")
    require(a.max_running_requests == 1, "single_running_request_required")
    require(a.pp_size == a.dp_size == a.dcp_size == a.nnodes == 1, "only_single_node_plain_tp_supported")
    spec = getattr(scheduler, "spec_algorithm", None)
    if spec is not None:
        no_spec = spec.is_none()
    else:
        no_spec = getattr(scheduler, "draft_worker", None) is None and getattr(
            a, "speculative_algorithm", None
        ) in (None, "none", "")
    require(not a.enable_dp_attention and no_spec, "dp_or_speculation_not_supported")
    require(not a.enable_unified_memory and not a.enable_page_major_kv_layout, "nonstatic_layout_not_supported")
    require(not a.enable_hierarchical_cache and not a.enable_hisparse, "offload_not_supported")
    require(a.cpu_offload_gb == 0 and not getattr(a, "enable_pdmux", False), "offload_or_pdmux_not_supported")
    require(str(a.disaggregation_mode) == "null", "disaggregation_not_supported")
    require(a.quantization is None and not a.enable_lora, "quantization_or_lora_not_supported")
    require(a.kv_cache_dtype == "auto", "kv_dtype_must_be_auto")
    require(a.page_size == 1, "first_version_requires_page_size_one")
    require(a.cuda_graph_config.decode.backend == "disabled" and
            a.cuda_graph_config.prefill.backend == "disabled", "graphs_must_be_disabled")
    require(batch.forward_mode.name == "EXTEND" and not batch.decoding_reqs, "plain_prefill_only")
    require(len(batch.reqs) == 1 and batch.reqs[0] is req, "single_request_batch_required")
    require(req.inflight_middle_chunks == 0 and not req.output_ids, "first_complete_prefill_only")
    require(not req.finished() and not req.is_retracted, "request_not_live")
    require(req.session_id is None and req.multimodal_inputs is None, "text_non_session_only")
    require(len(req.prefix_indices) == 0 and batch.prefix_lens == [0], "cached_prefix_not_supported")
    prompt_len = len(req.origin_input_ids)
    require(req.extend_range.start == 0 and req.extend_range.end == prompt_len, "incomplete_prefill_range")
    require(int(batch.seq_lens_cpu[0]) == prompt_len and req.kv_committed_len == prompt_len,
            "prefill_length_ledger_mismatch")
    require(config.token_start + config.token_count <= prompt_len, "requested_tokens_not_written")
    require(req.req_pool_idx is not None and req.req_pool_idx > 0, "request_row_released")


def source_identity(model_path):
    """Hash configuration AND weight bytes once in debug mode, before synchronizing.

    This intentionally expensive read gives all ranks a comparable local model
    fingerprint; it is outside forward and logs no prompt or token content.
    """
    root = Path(model_path).resolve()
    prepared = os.environ.get("SGLANG_KV_EC_PREPARED_IDENTITY")
    if prepared:
        identity = json.loads(prepared)
        require(identity["model_identifier"] == str(root), "prepared_model_mismatch")
        return dict(identity)
    cached = _SOURCE_IDENTITY_CACHE.get(str(root))
    if cached is not None:
        return dict(cached)
    files = sorted(root.glob("*.safetensors"))
    require((root / "config.json").is_file() and bool(files), "local_model_files_required")
    hasher = hashlib.sha256()
    for path in [root / "config.json"] + files:
        hasher.update(path.name.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b""):
                hasher.update(chunk)
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "python"], cwd=repo)
    identity = {"model_identifier": str(root), "model_sha256": hasher.hexdigest(),
                "source_commit": commit, "core_diff_sha256": hashlib.sha256(diff).hexdigest()}
    _SOURCE_IDENTITY_CACHE[str(root)] = identity
    return dict(identity)


def _tp_rank(scheduler):
    """Return the TP rank owned by the result processor's worker."""
    return int(scheduler.model_worker.ps.tp_rank)


def select_raw_bytes(tensor, slots):
    """Read bytes in logical token order; view changes interpretation, not values."""
    return tensor.detach().view(torch.uint8).index_select(0, slots).contiguous().reshape(-1)


def capture(scheduler, batch, req, config):
    end_to_end_started = time.perf_counter_ns()
    mode = protection_mode()
    phase = {"protection_mode": mode}
    # These are mutually exclusive wall-clock regions.  CUDA event timings are
    # deliberately not substituted: the scheduler pays CPU wall time here.
    phase_started = time.perf_counter_ns()
    validate_lifetime(scheduler, batch, req, config)
    # Import current checkout's exact classes ONLY after the lifetime checks.
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

    worker = scheduler.model_worker
    runner = worker.model_runner
    pool = runner.token_to_kv_pool
    req_pool = scheduler.req_to_token_pool
    ps = worker.ps
    require(type(pool) is MHATokenToKVPool and type(req_pool) is ReqToTokenPool, "unexpected_runtime_pool_class")
    require(runner.req_to_token_pool is req_pool and
            scheduler.token_to_kv_pool_allocator.get_kvcache() is pool, "pool_reference_mismatch")
    require(ps.tp_size in (1, 2, 4) and ps.attn_tp_size == ps.tp_size and
            ps.attn_cp_size == ps.attn_dcp_size == 1, "unexpected_parallel_layout")
    require(pool.kv_cache_layout == "nhd" and not pool.use_hnd and not pool.post_capture_active,
            "plain_nhd_static_pool_required")
    require(pool.layer_transfer_counter is None, "layer_transfer_in_progress_or_enabled")
    require(not pool.is_quantized_kv_cache and pool.dtype == pool.store_dtype == torch.bfloat16,
            "plain_bfloat16_storage_required")
    hf = runner.model_config.hf_config
    require(hf.model_type == "qwen3" and hf.num_hidden_layers == pool.layer_num == 36 and
            hf.num_key_value_heads == 8 and hf.head_dim == 128, "qwen3_8b_layout_required")
    require(pool.start_layer == 0 and pool.page_size == 1, "full_model_plain_slots_required")
    layers = config.layers(pool.layer_num)
    expected_bytes = len(layers) * config.token_count * (8 // ps.tp_size) * 128 * 2 * 2
    require(expected_bytes <= config.byte_budget, "snapshot_exceeds_byte_budget")
    device = pool.k_buffer[0].device
    require(device.type == "cuda" and torch.cuda.is_available(), "cuda_not_accessible")
    require(not torch.cuda.is_current_stream_capturing(), "capture_context_not_allowed")
    phase["lifetime_and_pool_validation_ms"] = (time.perf_counter_ns() - phase_started) / 1e6
    started = time.perf_counter_ns()
    identity = source_identity(scheduler.server_args.model_path)
    phase["model_fingerprint_ms"] = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter()
    # Coarse debug fence: includes set_kv_buffer on any stream on this device.
    # Not a lifetime lock: ownership is protected by this stopped scheduler thread
    # plus the exclusions above, until all copies/checks return and normal free runs.
    sync_started = time.perf_counter_ns()
    torch.cuda.synchronize(device)
    phase["gpu_sync_ms"] = (time.perf_counter_ns() - sync_started) / 1e6
    phase["gpu_sync_ms"] = phase.get("gpu_sync_ms", 0.0)
    slot_started = time.perf_counter_ns()
    row = req.req_pool_idx
    require(row not in req_pool.free_slots and row < req_pool.req_to_token.shape[0], "request_row_not_allocated")
    generation = int(req_pool.req_generation[row])
    require(generation > 0, "invalid_request_generation")
    end = config.token_start + config.token_count
    slots = req_pool.req_to_token[row, config.token_start:end].clone().long()
    slots_cpu = slots.cpu().tolist()
    require(len(set(slots_cpu)) == config.token_count and
            all(0 < slot <= pool.size for slot in slots_cpu), "invalid_or_duplicate_physical_slots")
    phase["slot_query_metadata_ms"] = (time.perf_counter_ns() - slot_started) / 1e6
    chunks, layer_records = [], []
    offset = 0
    kh, vh = hashlib.sha256(), hashlib.sha256()
    gather_ms = d2h_ms = gpu_wait_ms = source_recheck_ms = sha_ms = 0.0
    # Host staging buffers are reused for every layer.  Packed chunks still
    # own numpy storage because the staging buffers are overwritten next layer.
    cpu_k_buffer = cpu_v_buffer = None
    for layer_id in layers:
        # Getter's layer loading semantics are excluded above, not ignored.
        k, v = pool.get_kv_buffer(layer_id)
        expected_tail = (8 // ps.tp_size, 128)
        require(k.device == v.device == device and k.dtype == v.dtype == torch.bfloat16,
                "runtime_tensor_dtype_device_mismatch")
        require(k.is_contiguous() and v.is_contiguous() and k.ndim == v.ndim == 3 and
                tuple(k.shape[1:]) == tuple(v.shape[1:]) == expected_tail, "unsupported_actual_tensor_layout")
        # Only selected token rows are read/copied; no entire-pool clone.
        gather_started = time.perf_counter_ns()
        k_selected = select_raw_bytes(k, slots)
        v_selected = select_raw_bytes(v, slots)
        gather_ms += (time.perf_counter_ns() - gather_started) / 1e6
        # Complete gather before timing D2H; this is a CPU wall-clock fence.
        wait_started = time.perf_counter_ns()
        torch.cuda.current_stream(device).synchronize()
        gpu_wait_ms += (time.perf_counter_ns() - wait_started) / 1e6
        d2h_started = time.perf_counter_ns()
        # Reusable pinned buffers are an optional safe optimization.  The
        # synchronous fence above remains mandatory until slot leasing exists.
        try:
            if cpu_k_buffer is None or cpu_k_buffer.numel() != k_selected.numel():
                cpu_k_buffer = torch.empty(k_selected.numel(), dtype=torch.uint8, device="cpu", pin_memory=True)
                cpu_v_buffer = torch.empty(v_selected.numel(), dtype=torch.uint8, device="cpu", pin_memory=True)
            cpu_k_buffer.copy_(k_selected, non_blocking=True); cpu_v_buffer.copy_(v_selected, non_blocking=True)
            torch.cuda.current_stream(device).synchronize()
            kb = cpu_k_buffer.numpy().copy(); vb = cpu_v_buffer.numpy().copy()
        except (RuntimeError, TypeError):
            kb = k_selected.to("cpu", non_blocking=False).numpy().copy()
            vb = v_selected.to("cpu", non_blocking=False).numpy().copy()
        d2h_ms += (time.perf_counter_ns() - d2h_started) / 1e6
        # Re-read selected SOURCE bytes before release to prove no local mutation.
        # This checks the selected range, not unrelated rows or other requests.
        if mode == "DEBUG_SAFE":
            recheck_started = time.perf_counter_ns()
            require(np.array_equal(kb, select_raw_bytes(k, slots).cpu().numpy()) and
                    np.array_equal(vb, select_raw_bytes(v, slots).cpu().numpy()), "source_changed_during_copy")
            source_recheck_ms += (time.perf_counter_ns() - recheck_started) / 1e6
        shape = [config.token_count, *expected_tail]
        hash_started = time.perf_counter_ns()
        k_hash, v_hash = digest(kb), digest(vb)
        sha_ms += (time.perf_counter_ns() - hash_started) / 1e6
        record = {"layer_id": layer_id, "dtype": str(k.dtype), "original_shape": {"k": list(k.shape), "v": list(v.shape)},
                  "source_stride": {"k": list(k.stride()), "v": list(v.stride())},
                  "packed_shape": {"k": shape, "v": shape}, "device": str(device),
                  "k_offset": offset, "k_nbytes": int(kb.size), "k_sha256": k_hash,
                  "v_offset": offset + int(kb.size), "v_nbytes": int(vb.size), "v_sha256": v_hash}
        layer_records.append(record)
        offset += kb.size + vb.size
        hash_started = time.perf_counter_ns()
        kh.update(kb); vh.update(vb)
        sha_ms += (time.perf_counter_ns() - hash_started) / 1e6
        chunks.extend([kb, vb])
    final_sync_started = time.perf_counter_ns()
    torch.cuda.synchronize(device)
    gpu_wait_ms += (time.perf_counter_ns() - final_sync_started) / 1e6
    require(req.req_pool_idx == row and int(req_pool.req_generation[row]) == generation and
            row not in req_pool.free_slots, "request_generation_changed")
    require(torch.equal(slots, req_pool.req_to_token[row, config.token_start:end].long()), "slot_mapping_changed")
    concat_started = time.perf_counter_ns()
    payload = np.concatenate(chunks)
    phase["numpy_concatenate_ms"] = (time.perf_counter_ns() - concat_started) / 1e6
    require(payload.size == expected_bytes, "unexpected_packed_byte_count")
    payload_hash_started = time.perf_counter_ns()
    payload_sha256 = digest(payload)
    sha_ms += (time.perf_counter_ns() - payload_hash_started) / 1e6
    props = torch.cuda.get_device_properties(device)
    object_started = time.perf_counter_ns()
    meta = dict(identity, format="kv-ec-rank-bin-v1", provenance="sglang_request",
                snapshot_id=config.snapshot_id, request_id=hashlib.sha256(str(req.rid).encode()).hexdigest(),
                request_id_encoding="sha256", req_pool_idx=row, req_generation=generation,
                slot_generation=None, process_epoch=PROCESS_EPOCH, pid=os.getpid(),
                tp_rank=ps.tp_rank, tp_size=ps.tp_size, gpu_id=ps.gpu_id,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_device=str(device), gpu_name=props.name, gpu_uuid=str(getattr(props, "uuid", "unavailable")),
                layers=layers, token_start=config.token_start, token_count=config.token_count,
                prompt_token_count=len(req.origin_input_ids),
                token_ids_sha256=hashlib.sha256(np.asarray(req.origin_input_ids[config.token_start:end], dtype="<i8").tobytes()).hexdigest(),
                dtype=str(pool.dtype), pool_class=type(pool).__module__ + "." + type(pool).__name__,
                req_pool_class=type(req_pool).__module__ + "." + type(req_pool).__name__,
                layer_records=layer_records, original_shape=[r["original_shape"] for r in layer_records],
                physical_slots=slots_cpu, packed_byte_count=int(payload.size), sha256=payload_sha256,
                k_sha256=kh.hexdigest(), v_sha256=vh.hexdigest(), source_unchanged=True,
                lifetime="non_overlap_single_scheduler_before_append_cache_free",
                completion_dependency="device_synchronize_before_read_blocking_d2h_before_return",
                selected_copy_and_check_ms=(time.perf_counter() - started) * 1000,
                snapshot_completion_status="complete")
    phase["python_object_construction_ms"] = (time.perf_counter_ns() - object_started) / 1e6
    directory = config.output / config.snapshot_id
    phase["kv_gather_ms"] = gather_ms
    phase["gpu_sync_ms"] += gpu_wait_ms
    phase["d2h_ms"] = d2h_ms
    phase["source_recheck_ms"] = source_recheck_ms
    phase["sha256_ms"] = sha_ms
    meta["performance_ms"] = phase
    parity_socket = os.environ.get("SGLANG_KV_EC_PARITY_SOCKET")
    if parity_socket:
        # The client owns a bytes copy in a bounded background queue. No rank
        # file is written and this scheduler thread never waits for peers.
        from .parity_manager import AsyncParityClient

        client = _PARITY_CLIENTS.get(parity_socket)
        if client is None:
            client = AsyncParityClient(
                parity_socket,
                max_queue=int(os.environ.get("SGLANG_KV_EC_PARITY_QUEUE", "1")),
                timeout=float(os.environ.get("SGLANG_KV_EC_PARITY_TIMEOUT", "0.25")),
            )
            _PARITY_CLIENTS[parity_socket] = client
        enqueue = client.submit(meta, payload)
        phase["ipc_enqueue_ms"] = enqueue["enqueue_ms"]
        phase["ipc_copy_bytes"] = enqueue.get("copy_bytes", 0)
        phase["ipc_copy_count"] = enqueue.get("copy_count", 0)
        meta["parity_protection"] = "ENQUEUED" if enqueue["accepted"] else enqueue["reason"]
        if not enqueue["accepted"]:
            logger.warning("KV_EC parity not protected snapshot=%s rank=%d reason=%s",
                           config.snapshot_id, ps.tp_rank, enqueue["reason"])
    else:
        write_rank_files(directory, payload, meta)
    phase["python_object_serialization_ms"] = phase.get("ipc_enqueue_ms", 0.0)
    phase["diagnostic_write_ms"] = 0.0
    phase["snapshot_end_to_end_ms"] = (time.perf_counter_ns() - end_to_end_started) / 1e6
    phase["scheduler_blocking_ms"] = phase["snapshot_end_to_end_ms"]
    phase["steady_snapshot_ms"] = phase["snapshot_end_to_end_ms"] - phase["model_fingerprint_ms"]
    accounted = sum(float(phase.get(k, 0.0)) for k in (
        "lifetime_and_pool_validation_ms", "model_fingerprint_ms", "gpu_sync_ms",
        "slot_query_metadata_ms", "kv_gather_ms", "d2h_ms", "source_recheck_ms",
        "sha256_ms", "numpy_concatenate_ms", "python_object_construction_ms",
        "python_object_serialization_ms"))
    # Residual is intentionally reported instead of guessed: it includes
    # tensor metadata checks, final guards, object construction and any future
    # uninstrumented code, with no nested-time double counting.
    phase["unaccounted_ms"] = max(0.0, phase["scheduler_blocking_ms"] - accounted)
    phase["other_uninstrumented_ms"] = phase["unaccounted_ms"]
    directory.mkdir(parents=True, exist_ok=True)
    # Final timing data is separate from the immutable rank evidence. Its own
    # small diagnostic write is explicitly outside scheduler_blocking_ms.
    diagnostic_started = time.perf_counter_ns()
    with (directory / f"rank_{ps.tp_rank}.timing.json").open("x") as stream:
        json.dump({"snapshot_id": config.snapshot_id, "tp_rank": ps.tp_rank,
                   "parity_protection": meta.get("parity_protection", "DEBUG_DISK_ONLY"),
                   "performance_ms": phase,
                   "diagnostic_write_included": False}, stream, indent=2)
    # The diagnostic write is deliberately outside scheduler_blocking_ms; its
    # measured value is logged after the immutable timing record is written.
    phase["diagnostic_write_ms"] = (time.perf_counter_ns() - diagnostic_started) / 1e6
    logger.info(
        "KV_EC_DEBUG timing rank=%d snapshot=%s model_fingerprint_ms=%.3f "
        "gpu_sync_ms=%.3f kv_gather_ms=%.3f d2h_ms=%.3f source_recheck_ms=%.3f "
        "disk_write_ms=%.3f snapshot_end_to_end_ms=%.3f",
        ps.tp_rank, config.snapshot_id, phase["model_fingerprint_ms"],
        phase["gpu_sync_ms"], phase["kv_gather_ms"], phase["d2h_ms"],
        phase["source_recheck_ms"], phase.get("disk_write_ms", float("nan")),
        phase["snapshot_end_to_end_ms"],
    )
    logger.info("KV_EC_DEBUG complete snapshot=%s rank=%d pool=%s bytes=%d K=%s V=%s source_unchanged=true",
                config.snapshot_id, ps.tp_rank, meta["pool_class"], payload.size, meta["k_sha256"], meta["v_sha256"])
    return meta


def maybe_capture(scheduler, batch, req):
    if os.environ.get("SGLANG_KV_EC_DEBUG") != "1":
        return
    target = os.environ.get("SGLANG_KV_EC_REQUEST_ID")
    if not target or req.rid != target:
        return
    snapshot_id = os.environ.get("SGLANG_KV_EC_SNAPSHOT_ID", "")
    try:
        rank = _tp_rank(scheduler)
    except Exception:
        rank = -1
    attempt_key = (os.getpid(), PROCESS_EPOCH, snapshot_id, target, rank)
    if attempt_key in _ATTEMPTED:
        return
    _ATTEMPTED.add(attempt_key)
    config = None
    try:
        config = CaptureConfig.from_env()
        capture(scheduler, batch, req, config)
    except Exception as exc:
        # Failure is local, never changes sampling or distributed control flow.
        reason = str(exc) if isinstance(exc, CaptureRejected) else type(exc).__name__
        logger.warning("KV_EC_DEBUG rejected rank=%d reason=%s; inference continues", rank, reason)
        try:
            if config is None:
                config = CaptureConfig.from_env()
            directory = config.output / config.snapshot_id
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"rank_{rank}.error.json").open("x") as f:
                json.dump({"snapshot_completion_status": "failed", "reason": reason,
                           "tp_rank": rank}, f)
        except Exception as write_exc:
            # Never overwrite prior evidence, but make sidecar failures visible.
            logger.warning("KV_EC_DEBUG error sidecar failed rank=%d (%s)",
                           rank, type(write_exc).__name__)
