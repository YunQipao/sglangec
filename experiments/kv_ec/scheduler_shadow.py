"""Opt-in recovery inside SchedulerPauseGate's existing row/slot lease.

Online writes require an isolated admitted experiment. Parity comes from the
committed manager; Gloo gathers only survivors, never rank 0's rollback bytes.
"""

from datetime import timedelta
import hashlib
import json
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
                or slots.unique().numel() != slots.numel()):
            raise ValueError(f"unsupported_shadow_tensor_or_mapping:layer={layer}:{kind}")
    return k, v


def pack_chunk(pool, mapping, req, identity, *, snapshot_id, manager_epoch,
               token_start=0, token_count=256):
    """Read exactly one 256-token chunk at an explicit prompt offset."""
    if type(token_start) is not int or type(token_count) is not int:
        raise ValueError("chunk_range_must_be_integer")
    if token_count != 256 or token_start < 0 or token_start % 256:
        raise ValueError("chunk_range_must_be_256_aligned")
    if token_start + token_count > len(mapping.physical_slots):
        raise ValueError("chunk_range_exceeds_mapping")
    device = pool.k_buffer[0].device
    slots = torch.tensor(mapping.physical_slots[token_start:token_start + token_count],
                         dtype=torch.int64, device=device)
    chunks, records = [], []
    hashes = {kind: hashlib.sha256() for kind in ("k", "v")}
    offset = 0
    for layer in range(36):
        pair = checked_buffers(pool, layer, slots)
        record = dict(layer_id=layer, dtype="torch.bfloat16", packed_shape={},
                      original_shape={}, source_stride={}, device=str(device))
        for kind, tensor in zip(("k", "v"), pair):
            raw = select_raw_bytes(tensor, slots).cpu().numpy().copy()
            record["packed_shape"][kind] = [token_count, 2, 128]
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
        layers=list(range(36)), token_start=token_start, token_count=token_count,
        prompt_token_count=len(req.origin_input_ids),
        token_ids_sha256=digest(np.asarray(req.origin_input_ids[token_start:token_start + token_count],
                                           dtype="<i8")),
        input_ids_sha256=digest(np.asarray(req.origin_input_ids, dtype="<i8")),
        dtype="torch.bfloat16", pool_class=type(pool).__module__ + "." + type(pool).__name__,
        physical_slots=list(mapping.physical_slots[token_start:token_start + token_count]),
        slot_generation=list(mapping.slot_generations[token_start:token_start + token_count]),
        layer_records=records,
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


def write_mapped_bytes(pool, slots, buffers):
    """Scatter exact bytes through actual NHD tensor views, not pointer offsets."""
    if len(buffers) != 36:
        raise ValueError("online_write_requires_36_layers")
    # Validate all destinations and sources before issuing the first write.
    destinations = [checked_buffers(pool, layer, slots) for layer in range(36)]
    for pair, sources in zip(destinations, buffers):
        if len(sources) != 2:
            raise ValueError("online_write_requires_k_and_v")
        for live, source in zip(pair, sources):
            if (source.dtype != torch.uint8 or source.shape != (slots.numel(), 2, 256)
                    or source.device != live.device or not source.is_contiguous()
                    or source.untyped_storage().data_ptr() == live.untyped_storage().data_ptr()):
                raise ValueError("online_write_invalid_or_aliased_source")
    for pair, sources in zip(destinations, buffers):
        for live, source in zip(pair, sources):
            live.view(torch.uint8).index_copy_(0, slots, source)


def verify_mapped_bytes(pool, slots, buffers):
    for layer, sources in enumerate(buffers):
        for live, source in zip(checked_buffers(pool, layer, slots), sources):
            actual = select_raw_bytes(live, slots)
            if not torch.equal(actual, source.reshape(-1)):
                raise ValueError(f"online_writeback_byte_mismatch:layer={layer}")


def zero_mapped_bytes(pool, slots):
    destinations = [checked_buffers(pool, layer, slots) for layer in range(36)]
    for pair in destinations:
        for live in pair:
            live.view(torch.uint8).index_fill_(0, slots, 0)


class SchedulerShadowRecovery:
    def __init__(self, socket, manager_epoch, *, online=False):
        if not Path(socket).is_absolute() or not manager_epoch:
            raise ValueError("shadow_requires_absolute_socket_and_manager_epoch")
        self.socket, self.manager_epoch = socket, manager_epoch
        self.done = False
        self.metadata = self.status = self.event = None
        self.online = online
        self.protection_mapping = self.survivor = None
        self.protection_timing = {}
        self.rollback = self.recovered = self.slots = None
        self.write_started = False
        self.modified = False
        self.fault_monotonic = None

    @classmethod
    def from_env(cls):
        online = os.environ.get("SGLANG_KV_EC_ONLINE_RESTORE", "0") == "1"
        if os.environ.get("SGLANG_KV_EC_ONLINE_RESTORE", "0") not in ("0", "1") or (online and (
                os.environ.get("SGLANG_KV_EC_ISOLATED_TX") != json.loads(
                    os.environ.get("SGLANG_KV_EC_PAUSE_IDENTITY", "{} ")).get("transaction_id")
                or not os.environ.get("SGLANG_KV_EC_ISOLATED_TX")
                or not os.environ.get("SGLANG_KV_EC_ONLINE_ADMISSION"))):
            raise ValueError("online_restore_disabled_until_real_shadow_gpu_acceptance")
        return cls(os.environ["SGLANG_KV_EC_SHADOW_PARITY_SOCKET"],
                   os.environ["SGLANG_KV_EC_SHADOW_MANAGER_EPOCH"], online=online)

    def check_online_admission(self, gate):
        admission = json.loads(os.environ["SGLANG_KV_EC_ONLINE_ADMISSION"])
        if any(admission.get(key) != gate.model_identity[key]
               for key in ("model_identifier", "model_sha256", "source_commit", "core_diff_sha256")):
            raise ValueError("online_shadow_admission_model_or_source_mismatch")
        args = gate.scheduler.server_args
        hf = gate.scheduler.model_worker.model_runner.model_config.hf_config
        if (hf.hidden_size != 4096 or hf.num_attention_heads != 32
                or hf.architectures != ["Qwen3ForCausalLM"]):
            raise ValueError("online_requires_qwen3_8b")
        if any(getattr(args.cuda_graph_config, phase).backend != "disabled"
               for phase in ("decode", "prefill")):
            raise ValueError("online_requires_disabled_cuda_graphs")

    def prepare(self, gate):
        """Protection after normal prefill, BEFORE the PAUSED observation.

        The existing gate already blocks scheduling and owns leases here. No
        fault exists yet. Each rank submits once, then polls without blocking
        the other ranks' event loops or introducing a second lease protocol.
        """
        if not self.online or self.status is not None:
            return True
        self.validate(gate)
        gate._revalidate()
        pool = gate.scheduler.model_worker.model_runner.token_to_kv_pool
        if self.metadata is None:
            if not torch.cuda.is_available() or pool.k_buffer[0].device.type != "cuda":
                raise ValueError("online_protection_requires_cuda")
            self.check_online_admission(gate)
            self.protection_started = time.monotonic()
            self.protection_mapping = gate.mapping
            with torch.cuda.stream(gate.scheduler.forward_stream):
                gate.scheduler.forward_stream.wait_event(gate.event)
                started = time.perf_counter()
                payload, self.metadata = pack_chunk(
                    pool, gate.mapping, gate.req, gate.model_identity,
                    snapshot_id=gate.identity["transaction_id"] + "-chunk0",
                    manager_epoch=self.manager_epoch)
                self.protection_timing["capture_d2h_and_hash_ms"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            result = submit_once(self.socket, self.metadata, payload, timeout=self.timeout(gate))
            if result.get("state") not in (PENDING, COMMITTED):
                raise ValueError("online_protection_submission_failed")
            self.protection_timing["checkpoint_submit_ms"] = (time.perf_counter() - started) * 1000
            # Only survivors retain their CPU blocks. Rank 0's payload is not
            # retained as a recovery source; its digests/layout remain metadata.
            if gate.rank != 0:
                self.survivor = payload
        status = submit_once(self.socket, {"kind": "status", "snapshot_id": self.metadata["snapshot_id"]},
                             b"", timeout=self.timeout(gate))
        if status.get("state") == PENDING:
            return False
        validate_checkpoint(status, self.metadata)
        self.status = status
        self.commit_observed = time.monotonic()
        self.protection_timing["protection_total_ms"] = (self.commit_observed - self.protection_started) * 1000
        gate.evidence["protection"] = dict(
            snapshot_id=self.metadata["snapshot_id"], checkpoint_version=self.metadata["checkpoint_version"],
            manager_epoch=self.manager_epoch, parity_sha256=status["parity_sha256"],
            source_sha256=self.metadata["sha256"], request_version=gate.mapping.request_version,
            layer_records=self.metadata["layer_records"],
            committed_monotonic=self.commit_observed, state=COMMITTED,
            stage="normal_prefill_before_paused_observation", timing_ms=self.protection_timing)
        return True

    def consensus_checkpoint(self, gate):
        return dict(snapshot_id=self.metadata["snapshot_id"],
                    checkpoint_version=self.metadata["checkpoint_version"], manager_epoch=self.manager_epoch,
                    parity_sha256=self.status["parity_sha256"], request_hash=gate.evidence["request_hash"],
                    request_version=gate.mapping.request_version,
                    input_ids_sha256=self.metadata["input_ids_sha256"])

    def fence(self, gate):
        self.event = torch.cuda.Event()
        self.event.record(gate.scheduler.forward_stream)
        while not self.event.query():
            time.sleep(min(.001, self.timeout(gate)))
        gate.scheduler.token_to_kv_pool_allocator.attach_slot_lease_cuda_event(gate.owner, self.event)
        gate.event = self.event

    def rollback_failure(self, gate, pool, failure):
        """Never resume after failure, even when rollback proves byte-exact.

        A missing completion fence or changed mapping makes rollback unsafe.
        Keep all GPU buffers referenced and leases closed until process exit.
        """
        evidence = gate.evidence.setdefault("online", {})
        evidence.update(failure=str(failure), rollback_attempted=False, rollback_byte_exact=False)
        if not self.write_started:
            return
        try:
            self.fence(gate)
            gate._revalidate()
            # Manager/control availability is not needed to restore the local
            # original bytes, but row/slot identities must still be provable.
            evidence["rollback_attempted"] = True
            with torch.cuda.stream(gate.scheduler.forward_stream):
                write_mapped_bytes(pool, self.slots, self.rollback)
                self.fence(gate)
                verify_mapped_bytes(pool, self.slots, self.rollback)
            gate._revalidate()
            evidence["rollback_byte_exact"] = True
        except Exception as exc:
            evidence["rollback_failure"] = f"{type(exc).__name__}:{exc}"
        # _fatal closes the registry and retains both row and slot ownership.

    def run_online(self, gate):
        if not self.online or self.status is None or self.protection_mapping != gate.mapping:
            raise ValueError("online_requires_preexisting_committed_checkpoint_and_mapping")
        if not torch.cuda.is_available():
            raise RuntimeError("online_requires_real_cuda")
        self.validate(gate)
        gate._revalidate()
        status = gate._call("status")
        if (status["state"] != "PAUSED" or status.get("paused_ranks") != list(range(4))
                or status.get("require_online") is not True):
            raise ValueError("online_requires_four_rank_paused_interlock")
        pool = gate.scheduler.model_worker.model_runner.token_to_kv_pool
        device = pool.k_buffer[0].device
        if device.type != "cuda":
            raise ValueError("online_requires_cuda_pool")
        self.slots = torch.tensor(gate.mapping.physical_slots[:256], dtype=torch.int64, device=device)
        timing = dict(self.protection_timing)
        result = dict(mode="online", tp_rank=gate.rank, device=str(device), recovered_rank=0,
                      token_start=0, token_count=256, layers=list(range(36)),
                      checkpoint_committed_before_fault=True, used_committed_parity=True,
                      rank0_source_excluded=True, fault_injected=False, fault_bytes_changed=False,
                      rollback_used_for_recovery=False,
                      online_kv_pool_modified=False, writeback_byte_exact=False,
                      original_sha256=self.metadata["sha256"], rollback_byte_exact=False,
                      input_ids_sha256=self.metadata["input_ids_sha256"],
                      physical_slots_sha256=digest(np.asarray(gate.mapping.physical_slots[:256], dtype="<i8")),
                      timing_ms=timing, oracle_checks=self.status["oracle_checks"],
                      timing_clock="perf_counter_wall_ms_except_explicit_cuda_event_ms",
                      checkpoint=self.consensus_checkpoint(gate))
        gate.evidence["online"] = result
        try:
            with torch.cuda.stream(gate.scheduler.forward_stream):
                gate.scheduler.forward_stream.wait_event(gate.event)
                started = time.perf_counter()
                current, metadata = pack_chunk(pool, gate.mapping, gate.req, gate.model_identity,
                    snapshot_id=self.metadata["snapshot_id"], manager_epoch=self.manager_epoch)
                if metadata != self.metadata:
                    raise ValueError("online_live_bytes_or_version_changed_since_protection")
                if gate.rank != 0 and not np.array_equal(current, self.survivor):
                    raise ValueError("online_survivor_changed_since_protection")
                del current, metadata
                timing["prefault_live_validation_ms"] = (time.perf_counter() - started) * 1000
                committed = submit_once(self.socket, {"kind": "status", "snapshot_id": self.metadata["snapshot_id"]},
                                        b"", timeout=self.timeout(gate))
                validate_checkpoint(committed, self.metadata)
                if committed["parity_sha256"] != self.status["parity_sha256"]:
                    raise ValueError("online_committed_checkpoint_changed")
                self.validate(gate)
                gate._revalidate()
                # All ranks validate the preexisting checkpoint BEFORE rank 0
                # can inject. Existing CPU TP voting, not a new state machine.
                if not gate._vote(True):
                    raise ValueError("online_prefault_vote_failed")
                if gate.rank == 0:
                    started = time.perf_counter()
                    self.rollback = [tuple(live.detach().view(torch.uint8).index_select(0, self.slots).contiguous()
                                           for live in checked_buffers(pool, layer, self.slots)) for layer in range(36)]
                    self.fence(gate)
                    for layer, pair in enumerate(self.rollback):
                        for live, saved in zip(checked_buffers(pool, layer, self.slots), pair):
                            if saved.untyped_storage().data_ptr() == live.untyped_storage().data_ptr():
                                raise ValueError("online_rollback_aliases_live_pool")
                    rollback_hash = hashlib.sha256()
                    for pair in self.rollback:
                        for raw in pair:
                            rollback_hash.update(raw.cpu().numpy())
                    if rollback_hash.hexdigest() != self.metadata["sha256"]:
                        raise ValueError("online_rollback_capture_mismatch")
                    timing["rollback_capture_ms"] = (time.perf_counter() - started) * 1000
                    gate._revalidate()
                    self.validate(gate)
                    gate._call("status")
                    self.fault_monotonic = time.monotonic()
                    result["fault_monotonic"] = self.fault_monotonic
                    self.write_started = self.modified = True
                    result.update(fault_injected=True, online_kv_pool_modified=True)
                    zero_mapped_bytes(pool, self.slots)
                    self.fence(gate)
                    changed = 0
                    for layer, pair in enumerate(self.rollback):
                        for live, saved in zip(checked_buffers(pool, layer, self.slots), pair):
                            actual = select_raw_bytes(live, self.slots)
                            if torch.count_nonzero(actual).item() != 0:
                                raise ValueError("online_fault_clear_incomplete")
                            changed += int(torch.count_nonzero(actual != saved.reshape(-1)).item())
                    if changed == 0:
                        raise ValueError("online_fault_did_not_change_bytes")
                    result.update(fault_bytes_changed=True, changed_byte_count=changed)
                    timing["fault_injection_and_verify_ms"] = (time.monotonic() - self.fault_monotonic) * 1000
                contribution = (torch.zeros(self.metadata["packed_byte_count"], dtype=torch.uint8)
                                if gate.rank == 0 else torch.from_numpy(self.survivor))
                gathered = [torch.empty_like(contribution) for _ in range(4)] if gate.rank == 0 else None
                started = time.perf_counter()
                work = torch.distributed.gather(contribution, gather_list=gathered, dst=0,
                                               group=gate.scheduler.tp_cpu_group, async_op=True)
                if work.wait(timeout=timedelta(seconds=self.timeout(gate))) is False:
                    raise TimeoutError("online_survivor_gather_timeout")
                timing["survivor_gather_ms"] = (time.perf_counter() - started) * 1000
                if gate.rank == 0:
                    started = time.perf_counter()
                    exported, parity = fetch_parity(self.socket, self.metadata, timeout=self.timeout(gate))
                    validate_checkpoint(exported, self.metadata)
                    if exported["parity_sha256"] != self.status["parity_sha256"]:
                        raise ValueError("online_parity_changed_after_fault")
                    timing["parity_fetch_ms"] = (time.perf_counter() - started) * 1000
                    started = time.perf_counter()
                    # No rollback argument and no rank 0 payload: only three
                    # survivors plus the BEFORE-fault committed parity.
                    restored = reconstruct(parity, [block.numpy() for block in gathered[1:]], exported, self.metadata)
                    timing["xor_and_cpu_verify_ms"] = (time.perf_counter() - started) * 1000
                    metrics = {}
                    tensors, timing["h2d_ms"], timing["recovered_gpu_verify_ms"] = restore_rank_to_gpu(
                        restored, self.metadata, device, cuda_metrics=metrics)
                    self.recovered = [tuple(tensor.view(torch.uint8) for tensor in pair) for pair in tensors]
                    if any(source.untyped_storage().data_ptr() == saved.untyped_storage().data_ptr()
                           for pair, originals in zip(self.recovered, self.rollback)
                           for source, saved in zip(pair, originals)):
                        raise ValueError("online_recovery_aliases_rollback")
                    timing["h2d_cuda_event_ms"] = metrics["h2d_copy_event_ms"]
                    gate._revalidate()
                    self.validate(gate)
                    gate._call("status")
                    started = time.perf_counter()
                    write_mapped_bytes(pool, self.slots, self.recovered)
                    self.fence(gate)
                    timing["kv_writeback_ms"] = (time.perf_counter() - started) * 1000
                    started = time.perf_counter()
                    verify_mapped_bytes(pool, self.slots, self.recovered)
                    verify_mapped_bytes(pool, self.slots, self.rollback)
                    final, final_meta = pack_chunk(pool, gate.mapping, gate.req, gate.model_identity,
                        snapshot_id=self.metadata["snapshot_id"], manager_epoch=self.manager_epoch)
                    if final_meta != self.metadata:
                        raise ValueError("online_final_hash_or_mapping_mismatch")
                    result.update(writeback_byte_exact=True, restored_sha256=digest(final),
                                  rollback_sha256=rollback_hash.hexdigest())
                    timing["writeback_gpu_and_hash_verify_ms"] = (time.perf_counter() - started) * 1000
                else:
                    current, metadata = pack_chunk(pool, gate.mapping, gate.req, gate.model_identity,
                        snapshot_id=self.metadata["snapshot_id"], manager_epoch=self.manager_epoch)
                    if metadata != self.metadata:
                        raise ValueError("online_survivor_changed_during_recovery")
                    result["survivor_unchanged"] = True
                self.fence(gate)
            self.validate(gate)
            gate._revalidate()
            result.update(cuda_event_complete=True, request_generation=gate.mapping.req_generation,
                          slot_generations_unchanged=True, mapping_unchanged=True,
                          consensus=dict(self.consensus_checkpoint(gate), byte_exact=True,
                              online_kv_pool_modified=True, checkpoint_committed_before_fault=True,
                              fault_injected=True, fault_bytes_changed=True, used_committed_parity=True,
                              rank0_source_excluded=True, writeback_byte_exact=True))
            return result
        except BaseException as exc:
            self.rollback_failure(gate, pool, exc)
            raise

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
        if self.online:
            return self.run_online(gate)
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
