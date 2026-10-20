"""Shadow-only recovery inside SchedulerPauseGate's existing row/slot lease.

No live KV destination is exposed. Parity comes from the committed manager;
Gloo gathers only survivors (rank 0 contributes zeros, never its oracle).
"""

from datetime import timedelta
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from .debug_capture import select_raw_bytes
from .gpu_scratch_restore import restore_rank_to_gpu
from .parity_manager import (
    COMMITTED, PENDING, _identity, _layout_identity, fetch_parity,
    manager_hello, submit_once,
)
from .rank_snapshot import FORMAT, validate_payload
from .snapshot import digest
from .xor_codec import recover


def checked_buffers(pool, layer, slots):
    """Resolve actual owner tensors, never calculate offsets from pool size."""
    k, v = pool.get_kv_buffer(layer)
    for kind, tensor, owner in (("k", k, pool.k_buffer[layer]),
                                ("v", v, pool.v_buffer[layer])):
        if (tensor is not owner or tensor.dtype != torch.bfloat16
                or tensor.ndim != 3 or tuple(tensor.shape[1:]) != (2, 128)
                or not tensor.is_contiguous() or tuple(tensor.stride()) != (256, 128, 1)
                or tensor.device != slots.device or slots.dtype != torch.int64
                or slots.ndim != 1 or slots.numel() != 256
                or int(slots.min()) <= 0 or int(slots.max()) >= tensor.shape[0]
                or slots.unique().numel() != 256):
            raise ValueError(f"unsupported_shadow_tensor_or_mapping:layer={layer}:{kind}")
    return k, v


def pack_chunk(pool, mapping, req, identity, *, snapshot_id, manager_epoch):
    """Read only mapped chunk 0, in the established rank-bin canonical format."""
    device = pool.k_buffer[0].device
    slots = torch.tensor(mapping.physical_slots[:256], dtype=torch.int64, device=device)
    chunks, records = [], []
    hashes = {kind: hashlib.sha256() for kind in ("k", "v")}
    offset = 0
    for layer in range(36):
        pair = checked_buffers(pool, layer, slots)
        record = dict(layer_id=layer, dtype="torch.bfloat16", packed_shape={},
                      original_shape={}, source_stride={}, device=str(device))
        for kind, tensor in zip(("k", "v"), pair):
            raw = select_raw_bytes(tensor, slots).cpu().numpy().copy()
            record["packed_shape"][kind] = [256, 2, 128]
            record["original_shape"][kind] = list(tensor.shape)
            record["source_stride"][kind] = list(tensor.stride())
            record.update({kind + "_offset": offset, kind + "_nbytes": int(raw.size),
                           kind + "_sha256": digest(raw)})
            hashes[kind].update(raw)
            chunks.append(raw)
            offset += raw.size
        records.append(record)
    payload = np.concatenate(chunks)
    metadata = dict(
        identity, format=FORMAT, provenance="sglang_request", snapshot_id=snapshot_id,
        request_id=hashlib.sha256(req.rid.encode()).hexdigest(), request_id_encoding="sha256",
        req_pool_idx=mapping.req_pool_idx, req_generation=mapping.req_generation,
        request_version=mapping.request_version, checkpoint_version=snapshot_id,
        manager_epoch=manager_epoch, tp_rank=mapping.tp_rank, tp_size=4,
        layers=list(range(36)), token_start=0, token_count=256, prompt_token_count=768,
        token_ids_sha256=digest(np.asarray(req.origin_input_ids[:256], dtype="<i8")),
        input_ids_sha256=digest(np.asarray(req.origin_input_ids, dtype="<i8")),
        dtype="torch.bfloat16", pool_class=type(pool).__module__ + "." + type(pool).__name__,
        physical_slots=list(mapping.physical_slots[:256]),
        slot_generation=list(mapping.slot_generations[:256]), layer_records=records,
        packed_byte_count=int(payload.size), sha256=digest(payload),
        k_sha256=hashes["k"].hexdigest(), v_sha256=hashes["v"].hexdigest(),
        source_unchanged=True, snapshot_completion_status="complete",
        lifetime="paused_request_row_and_slot_lease",
        completion_dependency="producer_event_then_forward_stream_blocking_d2h",
    )
    validate_payload(payload, metadata)
    return payload, metadata


def validate_checkpoint(status, expected):
    if (status.get("state") != COMMITTED or status.get("checksum_deferred")
            or _identity(status) != _identity(expected)
            or _layout_identity(status) != _layout_identity(expected)
            or status.get("rank_hashes", {}).get(str(expected["tp_rank"]), {}).get("sha256")
            != expected["sha256"]):
        raise ValueError("shadow_checkpoint_version_or_rank_mismatch")
    if status.get("oracle_checks") != [
            {"lost_rank": rank, "byte_exact": True} for rank in range(4)]:
        raise ValueError("shadow_requires_four_parity_oracles")


def reconstruct(parity, survivors, status, expected):
    validate_checkpoint(status, expected)
    if len(survivors) != 3 or digest(parity) != status["parity_sha256"]:
        raise ValueError("shadow_parity_or_survivor_count_mismatch")
    for rank, block in enumerate(survivors, 1):
        if (block.dtype != np.uint8 or block.ndim != 1 or block.size != parity.size
                or digest(block) != status["rank_hashes"][str(rank)]["sha256"]):
            raise ValueError(f"shadow_survivor_version_or_bytes_mismatch:rank={rank}")
    restored = recover([None, *survivors], parity)
    validate_payload(restored, expected)
    return restored


class SchedulerShadowRecovery:
    def __init__(self, socket, manager_epoch):
        if not Path(socket).is_absolute() or not manager_epoch:
            raise ValueError("shadow_requires_absolute_socket_and_manager_epoch")
        self.socket, self.manager_epoch = socket, manager_epoch
        self.done = False
        self.metadata = self.status = self.event = None

    @classmethod
    def from_env(cls):
        if os.environ.get("SGLANG_KV_EC_ONLINE_RESTORE", "0") != "0":
            raise ValueError("online_restore_disabled_until_real_shadow_gpu_acceptance")
        return cls(os.environ["SGLANG_KV_EC_SHADOW_PARITY_SOCKET"],
                   os.environ["SGLANG_KV_EC_SHADOW_MANAGER_EPOCH"])

    def timeout(self, gate):
        remaining = gate.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("shadow_total_deadline")
        return min(10., remaining)

    def validate(self, gate):
        if manager_hello(self.socket, timeout=self.timeout(gate)) != self.manager_epoch:
            raise ValueError("shadow_manager_epoch_changed")
        if self.event is not None and not self.event.query():
            raise ValueError("shadow_cuda_event_incomplete")

    def run(self, gate):
        # Called on all four ranks only after a bounded TP CPU readiness vote.
        # The gate still owns the original full 768-token row/slot lease.
        if not torch.cuda.is_available():
            raise RuntimeError("shadow_requires_real_cuda")
        self.validate(gate)
        gate._revalidate()
        scheduler = gate.scheduler
        pool = scheduler.model_worker.model_runner.token_to_kv_pool
        hf = scheduler.model_worker.model_runner.model_config.hf_config
        if (hf.hidden_size != 4096 or hf.num_attention_heads != 32
                or hf.architectures != ["Qwen3ForCausalLM"]):
            raise ValueError("shadow_requires_qwen3_8b")
        device = pool.k_buffer[0].device
        if device.type != "cuda":
            raise ValueError("shadow_requires_cuda_pool")
        timing, started = {}, time.perf_counter()
        snapshot_id = gate.identity["transaction_id"] + "-chunk0"
        with torch.cuda.stream(scheduler.forward_stream):
            scheduler.forward_stream.wait_event(gate.event)
            stage = time.perf_counter()
            payload, self.metadata = pack_chunk(
                pool, gate.mapping, gate.req, gate.model_identity,
                snapshot_id=snapshot_id, manager_epoch=self.manager_epoch)
            timing["capture_d2h_and_hash_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            response = submit_once(self.socket, self.metadata, payload, timeout=self.timeout(gate))
            if response.get("state") not in (PENDING, COMMITTED):
                raise ValueError(f"shadow_submission_failed:{response.get('reason')}")
            timing["checkpoint_submit_ms"] = (time.perf_counter() - stage) * 1000

            # Explicitly exclude rank 0's pre-fault oracle from EC recovery.
            contribution = torch.zeros(payload.size, dtype=torch.uint8) if gate.rank == 0 else torch.from_numpy(payload)
            gathered = [torch.empty_like(contribution) for _ in range(4)] if gate.rank == 0 else None
            stage = time.perf_counter()
            work = torch.distributed.gather(
                contribution, gather_list=gathered, dst=0,
                group=scheduler.tp_cpu_group, async_op=True)
            if work.wait(timeout=timedelta(seconds=self.timeout(gate))) is False:
                raise TimeoutError("shadow_survivor_gather_timeout")
            timing["survivor_gather_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            while True:
                status = submit_once(self.socket, {"kind": "status", "snapshot_id": snapshot_id},
                                     b"", timeout=self.timeout(gate))
                if status.get("state") == COMMITTED:
                    break
                if status.get("state") != PENDING:
                    raise ValueError("shadow_checkpoint_not_committed")
                time.sleep(min(.01, self.timeout(gate)))
            validate_checkpoint(status, self.metadata)
            timing["checkpoint_wait_ms"] = (time.perf_counter() - stage) * 1000
            if gate.rank == 0:
                stage = time.perf_counter()
                exported, parity = fetch_parity(self.socket, self.metadata, timeout=self.timeout(gate))
                validate_checkpoint(exported, self.metadata)
                if exported["parity_sha256"] != status["parity_sha256"]:
                    raise ValueError("shadow_committed_parity_changed")
                timing["parity_fetch_ms"] = (time.perf_counter() - stage) * 1000
                stage = time.perf_counter()
                restored = reconstruct(parity, [item.numpy() for item in gathered[1:]],
                                       exported, self.metadata)
                if not np.array_equal(restored, payload):
                    raise ValueError("shadow_reconstruction_not_byte_exact")
                timing["xor_and_cpu_verify_ms"] = (time.perf_counter() - stage) * 1000
                # Reuse the existing byte-preserving GPU scratch implementation.
                cuda_metrics = {}
                shadow, timing["shadow_h2d_ms"], timing["shadow_verify_ms"] = restore_rank_to_gpu(
                    restored, self.metadata, device, cuda_metrics=cuda_metrics)
                timing["shadow_h2d_cuda_event_ms"] = cuda_metrics["h2d_copy_event_ms"]
                stage = time.perf_counter()
                slots = torch.tensor(gate.mapping.physical_slots[:256], device=device, dtype=torch.int64)
                for layer, pair in enumerate(shadow):
                    for live, destination in zip(checked_buffers(pool, layer, slots), pair):
                        if destination.untyped_storage().data_ptr() == live.untyped_storage().data_ptr():
                            raise ValueError("shadow_aliases_live_pool")
                        original = select_raw_bytes(live, slots)
                        if not torch.equal(original, destination.view(torch.uint8).reshape(-1)):
                            raise ValueError(f"shadow_live_layout_byte_mismatch:layer={layer}")
                timing["shadow_vs_live_ms"] = (time.perf_counter() - stage) * 1000
            # All ranks re-read source bytes; no source mutation is permitted.
            stage = time.perf_counter()
            current, current_meta = pack_chunk(
                pool, gate.mapping, gate.req, gate.model_identity,
                snapshot_id=snapshot_id, manager_epoch=self.manager_epoch)
            if current_meta != self.metadata or not np.array_equal(current, payload):
                raise ValueError("shadow_source_changed")
            timing["source_recheck_ms"] = (time.perf_counter() - stage) * 1000
            self.event = torch.cuda.Event()
            self.event.record(scheduler.forward_stream)
            while not self.event.query():
                time.sleep(min(.001, self.timeout(gate)))
        self.validate(gate)
        gate._revalidate()
        # Lease release must also observe the final shadow completion fence.
        scheduler.token_to_kv_pool_allocator.attach_slot_lease_cuda_event(gate.owner, self.event)
        gate.event = self.event
        self.status = status
        timing["local_total_ms"] = (time.perf_counter() - started) * 1000
        return dict(
            mode="shadow", device=str(device), tp_rank=gate.rank, recovered_rank=0,
            token_start=0, token_count=256, layers=list(range(36)),
            request_version=gate.mapping.request_version, source_sha256=self.metadata["sha256"],
            input_ids_sha256=self.metadata["input_ids_sha256"],
            physical_slots_sha256=digest(np.asarray(gate.mapping.physical_slots[:256], dtype="<i8")),
            slot_generations_sha256=digest(np.asarray(gate.mapping.slot_generations[:256], dtype="<i8")),
            request_row=gate.mapping.req_pool_idx, request_generation=gate.mapping.req_generation,
            source_unchanged=True, cuda_event_complete=True, byte_exact=True,
            rank0_contribution="zeros_excluded_from_recovery", used_committed_parity=True,
            oracle_checks=status["oracle_checks"], timing_ms=timing,
            timing_clock="perf_counter_wall_ms_except_explicit_cuda_event_ms",
            online_kv_pool_modified=False,
            consensus=dict(snapshot_id=snapshot_id, checkpoint_version=snapshot_id,
                           request_hash=gate.evidence["request_hash"],
                           request_version=gate.mapping.request_version,
                           input_ids_sha256=self.metadata["input_ids_sha256"],
                           manager_epoch=self.manager_epoch, parity_sha256=status["parity_sha256"],
                           byte_exact=True, online_kv_pool_modified=False))
