"""Read-only Scheduler/Allocator integration boundary for KV recovery.

The guard is opt-in and fail-closed.  It can inspect a real SGLang scheduler
and request mapping, but it refuses to create a writeback-capable lease until
the allocator exposes physical-slot generations.  No allocator or KV tensor is
modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import uuid
from typing import Any

import torch

from .recovery_lease import LeaseRejected, RecoveryLeaseManager


class RecoveryGuardRejected(RuntimeError):
    """The real scheduler is outside the supported non-destructive scope."""


@dataclass(frozen=True)
class SchedulerMapping:
    tp_rank: int
    tp_size: int
    request_id: str
    req_pool_idx: int
    req_generation: int
    token_start: int
    token_count: int
    physical_slots: tuple[int, ...]
    slot_generation_available: bool
    writeback_supported: bool
    reason: str
    slot_generations: tuple[int, ...] = ()
    request_version: str = ""


def guard_enabled() -> bool:
    return os.environ.get("SGLANG_KV_EC_RECOVERY_GUARD") == "1"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RecoveryGuardRejected(reason)


def read_scheduler_mapping(
    scheduler: Any,
    req: Any,
    *,
    token_start: int,
    token_count: int,
) -> SchedulerMapping:
    """Read one rank's actual request-to-token mapping without copying KV data."""

    args = scheduler.server_args
    _require(getattr(args, "disable_overlap_schedule", False), "overlap_not_disabled")
    _require(getattr(args, "disable_radix_cache", False), "prefix_cache_not_disabled")
    _require(getattr(args, "chunked_prefill_size", -1) == -1, "chunked_prefill_enabled")
    _require(getattr(args, "max_running_requests", 1) == 1, "concurrent_requests_enabled")
    worker = scheduler.model_worker
    ps = worker.ps
    _require(int(ps.tp_size) == 4 and int(ps.attn_tp_size) == 4, "only_tp4_supported")
    row = getattr(req, "req_pool_idx", None)
    _require(type(row) is int and row > 0, "request_row_not_allocated")
    pool = scheduler.req_to_token_pool
    free_slots = {int(x) for x in getattr(pool, "free_slots", [])}
    _require(row not in free_slots, "request_row_released")
    generation_tensor = getattr(pool, "req_generation", None)
    mapping = getattr(pool, "req_to_token", None)
    _require(isinstance(generation_tensor, torch.Tensor), "request_generation_unavailable")
    _require(isinstance(mapping, torch.Tensor) and mapping.ndim == 2,
             "request_mapping_unavailable")
    _require(token_start >= 0 and token_count > 0 and
             token_start + token_count <= mapping.shape[1], "token_range_invalid")
    generation = int(generation_tensor[row].item())
    _require(generation > 0, "request_generation_invalid")
    slots = tuple(int(x) for x in mapping[row, token_start:token_start + token_count]
                  .detach().to("cpu", dtype=torch.int64).tolist())
    _require(len(set(slots)) == token_count and all(slot > 0 for slot in slots),
             "physical_slot_mapping_invalid")
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    generations = ()
    supported = False
    reason = "allocator_slot_generation_unavailable"
    if (
        getattr(allocator, "page_size", None) == 1
        and callable(getattr(allocator, "slot_generations", None))
    ):
        try:
            generations = tuple(allocator.slot_generations(slots))
            supported = len(generations) == len(slots) and all(
                generation > 0 for generation in generations
            )
            if not supported:
                reason = "allocator_slot_generation_invalid"
            else:
                reason = "allocator_slot_generation_available"
        except Exception:
            reason = "allocator_slot_generation_unavailable"
    request_version = os.environ.get(
        "SGLANG_KV_EC_REQUEST_VERSION", f"req-generation-{generation}"
    )
    return SchedulerMapping(
        tp_rank=int(ps.tp_rank),
        tp_size=int(ps.tp_size),
        request_id=str(req.rid),
        req_pool_idx=row,
        req_generation=generation,
        token_start=token_start,
        token_count=token_count,
        physical_slots=slots,
        slot_generation_available=supported,
        writeback_supported=False,
        reason=reason,
        slot_generations=generations,
        request_version=request_version,
    )


class RecoveryGuard:
    """Opt-in read-only guard used by a future scheduler hook."""

    def __init__(self, *, manager: RecoveryLeaseManager | None = None):
        self.enabled = guard_enabled()
        self.manager = manager or RecoveryLeaseManager()
        self._active_lease: str | None = None

    def observe(self, scheduler: Any, req: Any, *, token_start: int, token_count: int) -> dict:
        if not self.enabled:
            return {"enabled": False, "state": "DISABLED"}
        mapping = read_scheduler_mapping(
            scheduler, req, token_start=token_start, token_count=token_count
        )
        return {
            "enabled": True,
            "state": "READ_ONLY_BLOCKED",
            "tp_rank": mapping.tp_rank,
            "tp_size": mapping.tp_size,
            "request_id": mapping.request_id,
            "req_pool_idx": mapping.req_pool_idx,
            "req_generation": mapping.req_generation,
            "token_start": mapping.token_start,
            "token_count": mapping.token_count,
            "physical_slots_sha256": __import__("hashlib").sha256(
                ",".join(map(str, mapping.physical_slots)).encode()
            ).hexdigest(),
            "slot_generation_available": mapping.slot_generation_available,
            "writeback_supported": mapping.writeback_supported,
            "reason": mapping.reason,
            "slot_generations": list(mapping.slot_generations),
            "request_version": mapping.request_version,
        }

    def before_free(self, slots: Any, *, tp_rank: int) -> None:
        if not self.enabled:
            return
        if self._active_lease is None:
            return
        self.manager.before_free(slots, tp_rank=tp_rank)

    def before_alloc(self, slots: Any, *, tp_rank: int) -> None:
        if not self.enabled:
            return
        if self._active_lease is None:
            return
        self.manager.before_alloc(slots, tp_rank=tp_rank)

    def close(self) -> None:
        self.manager.close()
        self._active_lease = None


_ACTIVE_SCHEDULER_LEASES: dict[str, "SchedulerLeaseTransaction"] = {}


class SchedulerLeaseTransaction:
    """One-rank, non-destructive Scheduler-owned allocator transaction.

    This transaction intentionally has no cross-rank commit and no writeback.
    It only proves that the local allocator can pin the observed slots across
    a CUDA completion event and revalidate the mapping before releasing them.
    """

    def __init__(self, key: str, allocator: Any, lease_id: str, metadata: dict):
        self.key = key
        self.allocator = allocator
        self.lease_id = lease_id
        self.metadata = dict(metadata)
        self._active = True

    @classmethod
    def begin(
        cls,
        scheduler: Any,
        req: Any,
        *,
        token_start: int,
        token_count: int,
        snapshot_id: str,
        slots: tuple[int, ...],
        req_generation: int,
    ) -> "SchedulerLeaseTransaction":
        allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
        _require(
            getattr(allocator, "page_size", None) == 1,
            "slot_lease_requires_page_size_one_allocator",
        )
        required = (
            "enable_slot_leases",
            "slot_generations",
            "begin_slot_lease",
            "validate_slot_lease",
            "record_slot_lease_cuda_event",
            "release_slot_lease",
        )
        _require(all(callable(getattr(allocator, name, None)) for name in required),
                 "allocator_slot_lease_api_unavailable")
        allocator.enable_slot_leases(
            timeout_s=float(os.environ.get("SGLANG_KV_EC_LEASE_TIMEOUT_S", "5")),
            active_slots=slots,
        )
        generations = tuple(allocator.slot_generations(list(slots)))
        _require(len(generations) == len(slots) and all(g > 0 for g in generations),
                 "slot_generation_unavailable")
        rank = int(scheduler.model_worker.ps.tp_rank)
        tp_size = int(scheduler.model_worker.ps.tp_size)
        _require(tp_size == 4 and 0 <= rank < tp_size, "only_tp4_supported")
        request_version = os.environ.get(
            "SGLANG_KV_EC_REQUEST_VERSION", f"req-generation-{req_generation}"
        )
        manager_epoch = os.environ.get(
            "SGLANG_KV_EC_MANAGER_EPOCH", "local-scheduler-epoch"
        )
        transaction_id = os.environ.get(
            "SGLANG_KV_EC_TRANSACTION_ID", f"{snapshot_id}:{req.rid}"
        )
        lease_id = (
            f"{snapshot_id}-{req.rid}-rank{rank}-"
            f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        )
        allocator.begin_slot_lease(
            lease_id,
            manager_epoch=os.environ.get(
                "SGLANG_KV_EC_MANAGER_EPOCH", "local-scheduler-epoch"),
            request_id=str(req.rid),
            request_version=request_version,
            slots=list(slots),
        )
        key = f"{os.getpid()}:{rank}:{snapshot_id}:{req.rid}"
        txn = cls(
            key,
            allocator,
            lease_id,
            {
                "state": "ACTIVE",
                "lease_id": lease_id,
                "tp_rank": rank,
                "tp_size": tp_size,
                "request_id": str(req.rid),
                "request_version": request_version,
                "manager_epoch": manager_epoch,
                "checkpoint_version": os.environ.get(
                    "SGLANG_KV_EC_CHECKPOINT_VERSION", snapshot_id
                ),
                "transaction_id": transaction_id,
                "req_generation": int(req_generation),
                "req_pool_idx": int(req.req_pool_idx),
                "token_start": int(token_start),
                "token_count": int(token_count),
                "slot_count": len(slots),
                "slot_generations_before": list(generations),
                "event_recorded": False,
                "event_complete": False,
                "writeback_supported": False,
                "cross_rank_coordinated": False,
                "global_commit": False,
            },
        )
        _ACTIVE_SCHEDULER_LEASES[key] = txn
        return txn

    def validate_completion(self, scheduler: Any, req: Any, *, slots: tuple[int, ...]) -> dict:
        if not self._active:
            raise RecoveryGuardRejected("scheduler_lease_not_active")
        if self.metadata.get("event_complete"):
            raise RecoveryGuardRejected("scheduler_lease_already_validated")
        event = self.allocator.record_slot_lease_cuda_event(
            self.lease_id,
            stream=torch.cuda.current_stream(),
        )
        self.metadata["event_recorded"] = True
        try:
            event.synchronize()
            self.metadata["event_complete"] = bool(event.query())
        except Exception as exc:
            raise RecoveryGuardRejected("cuda_event_completion_failed") from exc
        _require(self.metadata["event_complete"], "cuda_event_incomplete")
        epoch = self.metadata["manager_epoch"]
        _require(
            os.environ.get("SGLANG_KV_EC_MANAGER_EPOCH", "local-scheduler-epoch")
            == epoch,
            "manager_epoch_changed",
        )
        _require(str(req.rid) == self.metadata["request_id"], "request_identity_changed")
        _require(
            int(req.req_pool_idx) == self.metadata["req_pool_idx"]
            and req.req_pool_idx not in scheduler.req_to_token_pool.free_slots,
            "request_row_changed_or_freed",
        )
        current_generation = int(
            scheduler.req_to_token_pool.req_generation[req.req_pool_idx]
        )
        _require(
            current_generation == self.metadata["req_generation"],
            "request_generation_changed",
        )
        current_slots = tuple(
            int(x)
            for x in scheduler.req_to_token_pool.req_to_token[
                req.req_pool_idx,
                self.metadata["token_start"] : self.metadata["token_start"]
                + self.metadata["token_count"],
            ]
            .detach()
            .to("cpu", dtype=torch.int64)
            .tolist()
        )
        _require(current_slots == tuple(slots), "slot_mapping_changed")
        generations = tuple(self.allocator.slot_generations(list(current_slots)))
        _require(
            generations == tuple(self.metadata["slot_generations_before"]),
            "slot_generation_changed",
        )
        self.allocator.validate_slot_lease(
            self.lease_id,
            manager_epoch=epoch,
            request_id=str(req.rid),
            request_version=self.metadata["request_version"],
            slots=list(current_slots),
        )
        self.metadata.update(
            {
                "state": "LOCAL_VALIDATED",
                "slot_generations_after": list(generations),
            }
        )
        return dict(self.metadata)

    def release_local(self) -> dict:
        if not self._active:
            raise RecoveryGuardRejected("scheduler_lease_not_active")
        if self.metadata.get("state") != "LOCAL_VALIDATED":
            raise RecoveryGuardRejected("scheduler_lease_not_validated")
        epoch = self.metadata["manager_epoch"]
        self.allocator.release_slot_lease(self.lease_id, manager_epoch=epoch)
        self.metadata.update(
            {
                "state": "LOCAL_RELEASED",
                "slot_generations_after": list(
                    self.metadata["slot_generations_after"]
                ),
                "allocator_available_size_after": int(
                    self.allocator.available_size()
                ),
            }
        )
        self._active = False
        _ACTIVE_SCHEDULER_LEASES.pop(self.key, None)
        return dict(self.metadata)

    def complete(self, scheduler: Any, req: Any, *, slots: tuple[int, ...]) -> dict:
        """Compatibility local-only transaction; never means cross-rank release."""
        self.validate_completion(scheduler, req, slots=slots)
        return self.release_local()

    def abort(self) -> dict:
        if not self._active:
            return dict(self.metadata)
        # A failed transaction is deliberately fail-closed.  If an event can
        # be recorded and synchronized, release safely; otherwise leave the
        # allocator owner in place instead of guessing that GPU work finished.
        try:
            if torch.cuda.is_available():
                event = self.allocator.record_slot_lease_cuda_event(
                    self.lease_id, stream=torch.cuda.current_stream()
                )
                event.synchronize()
                if event.query():
                    self.allocator.abort_slot_lease(
                        self.lease_id,
                        manager_epoch=self.metadata["manager_epoch"],
                    )
                    self._active = False
                    self.metadata.update(
                        state="ABORTED_RELEASED",
                        abort_event_recorded=True,
                        abort_event_complete=True,
                    )
        except Exception:
            self.metadata.update(
                abort_event_recorded=True,
                abort_event_complete=False,
            )
        finally:
            if not self._active:
                _ACTIVE_SCHEDULER_LEASES.pop(self.key, None)
            else:
                self.metadata["state"] = "ABORTED_FAIL_CLOSED"
        return dict(self.metadata)


def abort_scheduler_lease(snapshot_id: str, req: Any, scheduler: Any) -> None:
    rank = getattr(getattr(scheduler, "model_worker", None), "ps", None)
    rank = getattr(rank, "tp_rank", -1)
    key_prefix = f"{os.getpid()}:{rank}:{snapshot_id}:{getattr(req, 'rid', '')}"
    for key, txn in list(_ACTIVE_SCHEDULER_LEASES.items()):
        if key == key_prefix:
            txn.abort()
