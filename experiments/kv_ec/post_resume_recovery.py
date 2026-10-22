"""M6-B1/B2: post-Resume four-rank lease and parity recovery transactions.

Every enabled TP rank executes the same CPU-group all-gather at each scheduler
boundary. B1 proves a fresh mapping/lease; B2 reuses M3's committed-parity
XOR and GPU writeback path while retaining the same fail-closed boundary.
"""

from datetime import timedelta
import hashlib
import json
import time

import torch

from .scheduler_shadow import (
    pack_chunk, reconstruct, write_mapped_bytes, verify_mapped_bytes,
    zero_mapped_bytes, checked_buffers, validate_checkpoint,
)
from .rank_snapshot import validate_payload
from .gpu_scratch_restore import restore_rank_to_gpu
from .parity_manager import COMMITTED, fetch_parity, submit_once
from .snapshot import digest

from .scheduler_recovery_guard import read_scheduler_mapping


RUNNING, REQUESTED, PAUSED, LEASED, VALIDATED, RESUMING, FAILED = (
    "RUNNING", "RECOVERY_REQUESTED", "RECOVERY_PAUSED", "LEASE_ACQUIRED",
    "VALIDATED", "RESUMING", "FAILED")


def _word(value):
    raw = hashlib.sha256(str(value).encode()).digest()[:8]
    return int.from_bytes(raw, "little") & ((1 << 63) - 1)


class PostResumeRecovery:
    """One-shot transaction called only from the Scheduler's pre-forward tick."""

    def __init__(self):
        self.state = RUNNING
        self.completed = False
        self.trace = []
        self.owner = None
        self.row_protected = False
        self.slot_leased = False

    def _record(self, state, **fields):
        self.state = state
        self.trace.append(dict(state=state, monotonic=time.monotonic(), **fields))

    def _all_gather(self, gate, local):
        """One fixed-order collective; local eligibility never gates its call."""
        remaining = gate.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("m6b1_collective_deadline")
        values = [torch.empty_like(local) for _ in range(4)]
        work = torch.distributed.all_gather(
            values, local, group=gate.scheduler.tp_cpu_group, async_op=True)
        if work.wait(timeout=timedelta(seconds=remaining)) is False:
            raise TimeoutError("m6b1_collective_timeout")
        return [tuple(int(x) for x in item.tolist()) for item in values]

    def _collect(self, gate, *, readiness, checkpoint_id, checkpoint_epoch):
        # Transaction identity is deterministic from the existing coordinator
        # admission; mismatched request/checkpoint/epoch prevents unanimous entry.
        fields = (gate.identity["transaction_id"], gate.target,
                  checkpoint_id, checkpoint_epoch)
        local = torch.tensor([readiness, *(_word(x) for x in fields)],
                             dtype=torch.int64, device="cpu")
        gathered = self._all_gather(gate, local)
        identities = [row[1:] for row in gathered]
        unanimous_identity = all(value == identities[0] for value in identities)
        return gathered, unanimous_identity

    @staticmethod
    def _local_readiness(gate):
        req = gate.req
        shadow = gate.shadow
        worker = getattr(shadow, "worker", None)
        metadata = getattr(worker, "metadata", {}) or {}
        checkpoint = str(metadata.get("checkpoint_version", ""))
        epoch = str(metadata.get("manager_epoch", ""))
        if req is None or req.rid != gate.target:
            return 3, checkpoint, epoch
        if req.finished() or req.is_retracted or req.to_finish is not None:
            return 3, checkpoint, epoch
        if not gate.evidence.get("decode_observed", False):
            return 0, checkpoint, epoch
        if worker is None or getattr(worker, "state", None) != "COMMITTED":
            return 0, checkpoint, epoch
        return 1, checkpoint, epoch

    def _release_local(self, gate):
        if getattr(self, "rollback_failed", False):
            raise RuntimeError("m6b2_rollback_unproven_keep_lease")
        if self.slot_leased:
            event = torch.cuda.Event()
            event.record(gate.scheduler.forward_stream)
            event.synchronize()
            if not event.query():
                raise ValueError("m6b2_cuda_work_incomplete_before_lease_release")
            gate.scheduler.token_to_kv_pool_allocator.attach_slot_lease_cuda_event(
                self.owner, event)
            gate.scheduler.token_to_kv_pool_allocator.release_slot_lease(
                self.owner, manager_epoch=self.epoch)
            self.slot_leased = False
        if self.row_protected:
            gate.scheduler.req_to_token_pool.release_kv_ec_row(
                self.mapping.req_pool_idx, self.mapping.req_generation, self.owner)
            self.row_protected = False

    def _reject(self, gate, reason):
        self._record(FAILED, reason=reason)
        record = dict(
            state=FAILED, failure_reason=reason, transaction_id=self.txid,
            checkpoint_id=self.checkpoint_id, checkpoint_epoch=self.epoch,
            parity_used=False, kv_written=False, trace=self.trace)
        gate.evidence["post_resume_recovery"] = record
        self._write_record(gate, record)
        # A failed mapping/checkpoint proof must not allow this request to
        # decode against a mapping we could not bind to its checkpoint.
        gate._fatal(RuntimeError("m6b1_fail_closed:" + reason))

    def _write_record(self, gate, record):
        suffix = "m6b2" if gate.post_resume_mode == "m6b2" else "m6b1"
        path = gate.output / f"{suffix}-rank-{gate.rank}.json"
        with path.open("x") as stream:
            json.dump(record, stream, indent=2)

    def _b2(self, gate, mapping, current):
        """Execute the single-chunk M3 data path under the fresh B1 lease."""
        transaction_started = time.perf_counter()
        shadow, rank = gate.shadow, gate.rank
        meta = current[1]
        committed = None
        pool = device = slots = source = None
        try:
            if (meta.get("snapshot_id") != self.checkpoint_id
                    or meta.get("checkpoint_version") != self.checkpoint_id
                    or meta.get("manager_epoch") != self.epoch
                    or meta.get("token_start") != 0 or meta.get("token_count") != 256
                    or meta.get("tp_size") != 4 or meta.get("tp_rank") != rank):
                raise ValueError("m6b2_checkpoint_binding_mismatch")
            committed = submit_once(shadow.socket, {
                "kind": "status", "snapshot_id": self.checkpoint_id}, b"",
                timeout=gate.timeout_s)
            validate_checkpoint(committed, meta, require_oracles=False)
            if committed.get("manager_epoch") != self.epoch:
                raise ValueError("m6b2_manager_epoch_changed")
            pool = gate.scheduler.model_worker.model_runner.token_to_kv_pool
            self._assert_mapping(gate, mapping)
            device = pool.k_buffer[0].device
            slots = torch.tensor(mapping.physical_slots[:256], dtype=torch.int64, device=device)
            source = current[0]
            preflight_ok, preflight_reason = True, ""
        except Exception as exc:
            preflight_ok = False
            preflight_reason = type(exc).__name__ + ":" + str(exc)
        preflight, identity_ok = self._collect(
            gate, readiness=int(preflight_ok), checkpoint_id=self.checkpoint_id,
            checkpoint_epoch=self.epoch)
        if (not identity_ok or not all(row[0] == 1 for row in preflight)):
            raise ValueError(preflight_reason or "m6b2_prefault_vote_rejected")

        # Keep the target copy only as a rollback image. It is never passed to
        # reconstruct or used to synthesize the expected restored bytes.
        rollback = None
        fault_ok, fault_reason = True, ""
        if rank == shadow.lost_rank:
            try:
                rollback = [tuple(t.detach().view(torch.uint8).index_select(0, slots).contiguous()
                                  for t in checked_buffers(pool, layer, slots))
                            for layer in range(36)]
                self._b2_rollback = (gate, mapping, pool, slots, rollback)
                self.fault_monotonic = time.monotonic()
                zero_mapped_bytes(pool, slots)
                torch.cuda.synchronize(device)
                changed = any(not torch.equal(before.reshape(-1), live.view(torch.uint8).index_select(0, slots).reshape(-1))
                              for layer, pair in enumerate(rollback)
                              for live, before in zip(checked_buffers(pool, layer, slots), pair))
                if not changed:
                    raise ValueError("m6b2_fault_injection_did_not_change_bytes")
            except Exception as exc:
                fault_ok = False
                fault_reason = type(exc).__name__ + ":" + str(exc)

        contribution = torch.zeros(meta["packed_byte_count"], dtype=torch.uint8) if rank == shadow.lost_rank else torch.from_numpy(source.copy())
        gathered = [torch.empty_like(contribution) for _ in range(4)] if rank == shadow.lost_rank else None
        gather_started = time.perf_counter()
        work = torch.distributed.gather(contribution, gather_list=gathered,
                                        dst=shadow.lost_rank, group=gate.scheduler.tp_cpu_group,
                                        async_op=True)
        if work.wait(timeout=timedelta(seconds=gate.timeout_s)) is False:
            raise TimeoutError("m6b2_survivor_gather_timeout")

        timings = {"survivor_gather_ms": (time.perf_counter() - gather_started) * 1000}
        recovery_ok, recovery_reason = fault_ok, fault_reason
        if rank == shadow.lost_rank:
            try:
                if not recovery_ok:
                    raise ValueError(recovery_reason)
                stage_started = time.perf_counter()
                exported, parity = fetch_parity(shadow.socket, meta, timeout=gate.timeout_s)
                validate_checkpoint(exported, meta, require_oracles=False)
                if exported.get("state") != COMMITTED or exported.get("parity_sha256") != committed.get("parity_sha256"):
                    raise ValueError("m6b2_parity_not_committed_or_changed")
                timings["parity_fetch_ms"] = (time.perf_counter() - stage_started) * 1000
                stage_started = time.perf_counter()
                restored = reconstruct(parity, [value.numpy() for i, value in enumerate(gathered)
                                                if i != shadow.lost_rank], exported, meta,
                                      lost_rank=shadow.lost_rank, require_oracles=False)
                oracle = meta["sha256"]
                if digest(restored) != oracle:
                    raise ValueError("m6b2_independent_checkpoint_hash_mismatch")
                timings["xor_and_independent_hash_ms"] = (time.perf_counter() - stage_started) * 1000
                tensors, h2d_ms, gpu_verify_ms = restore_rank_to_gpu(restored, meta, device)
                timings["h2d_ms"] = h2d_ms
                timings["restored_gpu_hash_verify_ms"] = gpu_verify_ms
                buffers = [tuple(t.view(torch.uint8) for t in pair) for pair in tensors]
                self._assert_mapping(gate, mapping)
                gate.scheduler.token_to_kv_pool_allocator.validate_slot_lease(
                    self.owner, manager_epoch=self.epoch, request_id=gate.req.rid,
                    request_version=mapping.request_version,
                    slots=list(mapping.physical_slots[:256]))
                write_mapped_bytes(pool, slots, buffers)
                torch.cuda.synchronize(device)
                verify_mapped_bytes(pool, slots, buffers)
                stage_started = time.perf_counter()
                # Re-read the live GPU bytes through the canonical capture
                # path. Only the payload bytes are compared; the lifecycle
                # metadata of the live lease intentionally differs from the
                # M6-A checkpoint metadata. validate_payload re-checks the
                # layout and every K/V segment against the checkpoint history.
                final, _ = pack_chunk(pool, mapping, gate.req, gate.model_identity,
                    snapshot_id=self.checkpoint_id, manager_epoch=self.epoch)
                if digest(final) != oracle:
                    raise ValueError("m6b2_gpu_writeback_hash_mismatch")
                validate_payload(final, meta)
                timings["writeback_and_full_layer_verify_ms"] = (time.perf_counter() - stage_started) * 1000
            except Exception as exc:
                recovery_ok = False
                recovery_reason = type(exc).__name__ + ":" + str(exc)

        coordination_started = time.perf_counter()
        complete = torch.tensor([int(recovery_ok)], dtype=torch.int64, device="cpu")
        results = self._all_gather(gate, complete)
        if any(row[0] != 1 for row in results):
            raise ValueError(recovery_reason or "m6b2_completion_vote_rejected")
        timings["completion_coordination_ms"] = (time.perf_counter() - coordination_started) * 1000
        timings["recovery_transaction_ms"] = (time.perf_counter() - transaction_started) * 1000
        return dict(fault_injected=rank == shadow.lost_rank, parity_used=rank == shadow.lost_rank,
                    kv_written=rank == shadow.lost_rank, byte_exact=True,
                    four_rank_vote=True, oracle_sha256=meta["sha256"],
                    fault_monotonic=getattr(self, "fault_monotonic", None), timings=timings)

    @staticmethod
    def _assert_mapping(gate, mapping):
        current = read_scheduler_mapping(gate.scheduler, gate.req,
                                         token_start=0, token_count=256)
        if current != mapping:
            raise ValueError("m6b2_row_slot_generation_changed")

    def _rollback_b2(self):
        saved = getattr(self, "_b2_rollback", None)
        if saved is None:
            return None
        gate, mapping, pool, slots, rollback = saved
        self._assert_mapping(gate, mapping)
        gate.scheduler.token_to_kv_pool_allocator.validate_slot_lease(
            self.owner, manager_epoch=self.epoch, request_id=gate.req.rid,
            request_version=mapping.request_version,
            slots=list(mapping.physical_slots[:256]))
        for layer, pair in enumerate(rollback):
            for live, before in zip(checked_buffers(pool, layer, slots), pair):
                live.view(torch.uint8).index_copy_(0, slots, before)
        torch.cuda.synchronize(slots.device)
        for layer, pair in enumerate(rollback):
            for live, before in zip(checked_buffers(pool, layer, slots), pair):
                if not torch.equal(live.view(torch.uint8).index_select(0, slots), before):
                    raise RuntimeError(f"m6b2_rollback_byte_mismatch_layer_{layer}")
        return True

    def tick(self, gate):
        if self.state != RUNNING or self.completed:
            return False
        gate.deadline = time.monotonic() + gate.timeout_s
        readiness, checkpoint, epoch = self._local_readiness(gate)
        gathered, identity_ok = self._collect(
            gate, readiness=readiness, checkpoint_id=checkpoint,
            checkpoint_epoch=epoch)
        codes = [row[0] for row in gathered]
        self.txid = f"{gate.identity['transaction_id']}:m6b1:{checkpoint}"
        self.checkpoint_id, self.epoch = checkpoint, epoch
        if not identity_ok or any(code == 3 for code in codes):
            # A missing/cancelled/inconsistent request is a unanimous safe
            # refusal, not a rank-local branch into another collective.
            return self._reject(gate, "rank_identity_or_request_not_unanimous")
        if not all(code == 1 for code in codes):
            return False  # checkpoint not committed or decode not yet observed

        self._record(REQUESTED)
        b2_preparation_voted = False
        try:
            # This call is before batch selection/forward. The normal scheduler
            # is non-overlap; require all prior producers to be fenced below.
            self._record(PAUSED)
            req = gate.req
            pool = gate.scheduler.model_worker.model_runner.token_to_kv_pool
            self.mapping = read_scheduler_mapping(
                gate.scheduler, req, token_start=0, token_count=256)
            if (self.mapping.tp_size != 4 or self.mapping.tp_rank != gate.rank
                    or not self.mapping.slot_generation_available):
                raise ValueError("m6b1_live_mapping_or_generation_unsupported")

            # Fence preceding producer work before acquiring a fresh lease.
            if not torch.cuda.is_available() or pool.k_buffer[0].device.type != "cuda":
                raise RuntimeError("m6b1_requires_real_cuda")
            producer = torch.cuda.Event()
            producer.record(torch.cuda.current_stream())
            gate.scheduler.forward_stream.wait_event(producer)

            allocator = gate.scheduler.token_to_kv_pool_allocator
            registry = allocator.slot_lease_registry
            old_owner = getattr(gate, "owner", None)
            if not gate.released or (old_owner and old_owner in registry._leases):
                raise ValueError("m6b1_old_pause_lease_not_released")
            self.owner = f"{self.txid}:rank{gate.rank}"
            self.epoch = str(gate.shadow.manager_epoch)
            allocator.enable_slot_leases(
                timeout_s=gate.timeout_s,
                active_slots=self.mapping.physical_slots)
            gate.scheduler.req_to_token_pool.protect_kv_ec_row(
                self.mapping.req_pool_idx, self.mapping.req_generation, self.owner)
            self.row_protected = True
            allocator.begin_slot_lease(
                self.owner, manager_epoch=self.epoch, request_id=req.rid,
                request_version=self.mapping.request_version,
                slots=list(self.mapping.physical_slots))
            self.slot_leased = True
            event = allocator.record_slot_lease_cuda_event(
                self.owner, stream=gate.scheduler.forward_stream)
            event.synchronize()
            if not event.query():
                raise ValueError("m6b1_cuda_event_incomplete")
            allocator.validate_slot_lease(
                self.owner, manager_epoch=self.epoch, request_id=req.rid,
                request_version=self.mapping.request_version,
                slots=list(self.mapping.physical_slots))

            # Capture metadata only, under the fresh lease. Strict binding
            # comparison proves that this current logical chunk is the one
            # committed by M6-A; a remapped slot/generation is rejected.
            current_bytes, current = pack_chunk(
                pool, self.mapping, req, gate.model_identity,
                snapshot_id=self.checkpoint_id, manager_epoch=self.epoch,
                token_start=0, token_count=256,
                # Re-capture the live bytes, but retain the identity of the
                # committed M6-A asynchronous CPU snapshot. These fields are
                # part of validate_checkpoint's binding, not presentation.
                lifetime="independent_immutable_cpu_snapshot",
                completion_dependency=(
                    "producer_event_then_blocking_d2h_then_capture_event"))
            gate.shadow.worker.require_recoverable(current)
            self._record(LEASED)
            self._record(VALIDATED,
                         row=self.mapping.req_pool_idx,
                         row_generation=self.mapping.req_generation,
                         request_version=self.mapping.request_version,
                         physical_slots_sha256=hashlib.sha256(
                             json.dumps(self.mapping.physical_slots).encode()).hexdigest(),
                         slot_generations=list(self.mapping.slot_generations),
                         cuda_event_complete=True, old_lease_reused=False)
            if gate.post_resume_mode == "m6b2":
                # This vote is the fixed boundary between local preparation
                # and B2's parity/data collectives. A rank that failed any
                # mapping, event, lease, or checkpoint step joins this same
                # vote from the exception path below.
                prepared, same_identity = self._collect(
                    gate, readiness=1, checkpoint_id=self.checkpoint_id,
                    checkpoint_epoch=self.epoch)
                b2_preparation_voted = True
                if not same_identity or any(row[0] != 1 for row in prepared):
                    local_ok = False
                    reason = "m6b2_rank_preparation_vote_rejected"
                    recovery = None
                else:
                    recovery = self._b2(gate, self.mapping, (current_bytes, current))
            else:
                recovery = None
        except Exception as exc:
            local_ok = False
            reason = type(exc).__name__ + ":" + str(exc)
            if gate.post_resume_mode == "m6b2":
                # Match the successful rank's preparation vote before any
                # rank is allowed to enter _b2's gather sequence.
                if not b2_preparation_voted:
                    try:
                        prepared, same_identity = self._collect(
                            gate, readiness=0, checkpoint_id=self.checkpoint_id,
                            checkpoint_epoch=self.epoch)
                        b2_preparation_voted = True
                        if not same_identity or any(row[0] != 1 for row in prepared):
                            reason += ";rank_preparation_vote_rejected"
                    except Exception as vote_exc:
                        reason += ";rank_preparation_vote=" + type(vote_exc).__name__
                try:
                    self._rollback_b2()
                except Exception as rollback_exc:
                    self.rollback_failed = True
                    reason += ";rollback=" + type(rollback_exc).__name__ + ":" + str(rollback_exc)
        else:
            local_ok, reason = True, ""

        # Every rank votes in this same position, even after a local exception.
        gathered, identity_ok = self._collect(
            gate, readiness=int(local_ok), checkpoint_id=self.checkpoint_id,
            checkpoint_epoch=self.epoch)
        unanimous = identity_ok and all(row[0] == 1 for row in gathered)
        cleanup_error = None
        try:
            self._release_local(gate)
        except Exception as exc:
            cleanup_error = exc
        released = cleanup_error is None
        gathered_release, _ = self._collect(
            gate, readiness=int(released), checkpoint_id=self.checkpoint_id,
            checkpoint_epoch=self.epoch)
        if not all(row[0] == 1 for row in gathered_release):
            # A resource that cannot be safely released must never be followed
            # by Decode. Existing gate fatal path closes allocator admission.
            gate._fatal(cleanup_error or RuntimeError("m6b1_peer_cleanup_failed"))
        if not unanimous:
            return self._reject(gate, reason or "peer_mapping_lease_validation_failed")

        self._record(RESUMING)
        rank_recovery = recovery or dict(fault_injected=False, parity_used=False,
                                         kv_written=False, byte_exact=False,
                                         four_rank_vote=False, timings={})
        record = dict(
            state="RUNNING", transaction_id=self.txid,
            checkpoint_id=self.checkpoint_id, checkpoint_epoch=self.epoch,
            row=self.mapping.req_pool_idx, row_generation=self.mapping.req_generation,
            request_version=self.mapping.request_version,
            physical_slots=list(self.mapping.physical_slots),
            slot_generations=list(self.mapping.slot_generations),
            new_lease_created=True, new_lease_released=True, old_lease_reused=False,
            all_rank_validate_vote=True, all_rank_resume_vote=True,
            fault_injected=rank_recovery["fault_injected"],
            parity_used=rank_recovery["parity_used"], kv_written=rank_recovery["kv_written"],
            byte_exact=rank_recovery["byte_exact"], four_rank_vote=rank_recovery["four_rank_vote"],
            recovery=rank_recovery, trace=self.trace)
        gate.evidence["post_resume_recovery"] = record
        self._write_record(gate, record)
        self._record(RUNNING)
        self.completed = True
        return False
