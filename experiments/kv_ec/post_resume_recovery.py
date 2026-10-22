"""M6-B1: fail-closed, no-write post-Resume lease transaction.

Every enabled TP rank executes the same CPU-group all-gather at each scheduler
boundary. This phase proves a fresh live mapping/lease can be acquired and
released; it deliberately performs no parity decode or KV writeback.
"""

from datetime import timedelta
import hashlib
import json
import time

import torch

from .scheduler_recovery_guard import read_scheduler_mapping
from .scheduler_shadow import pack_chunk


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
        if self.slot_leased:
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
        path = gate.output / f"m6b1-rank-{gate.rank}.json"
        with path.open("x") as stream:
            json.dump(record, stream, indent=2)

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
            _, current = pack_chunk(
                pool, self.mapping, req, gate.model_identity,
                snapshot_id=self.checkpoint_id, manager_epoch=self.epoch,
                token_start=0, token_count=256)
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
        except Exception as exc:
            local_ok = False
            reason = type(exc).__name__ + ":" + str(exc)
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
        record = dict(
            state="RUNNING", transaction_id=self.txid,
            checkpoint_id=self.checkpoint_id, checkpoint_epoch=self.epoch,
            row=self.mapping.req_pool_idx, row_generation=self.mapping.req_generation,
            request_version=self.mapping.request_version,
            physical_slots=list(self.mapping.physical_slots),
            slot_generations=list(self.mapping.slot_generations),
            new_lease_created=True, new_lease_released=True, old_lease_reused=False,
            all_rank_validate_vote=True, all_rank_resume_vote=True,
            parity_used=False, kv_written=False, trace=self.trace)
        gate.evidence["post_resume_recovery"] = record
        self._write_record(gate, record)
        self._record(RUNNING)
        self.completed = True
        return False
