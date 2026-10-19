"""Read-only Scheduler/Allocator integration boundary for KV recovery.

The guard is opt-in and fail-closed.  It can inspect a real SGLang scheduler
and request mapping, but it refuses to create a writeback-capable lease until
the allocator exposes physical-slot generations.  No allocator or KV tensor is
modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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
    # The current allocator has no physical slot generation or lease API.
    return SchedulerMapping(
        tp_rank=int(ps.tp_rank),
        tp_size=int(ps.tp_size),
        request_id=str(req.rid),
        req_pool_idx=row,
        req_generation=generation,
        token_start=token_start,
        token_count=token_count,
        physical_slots=slots,
        slot_generation_available=False,
        writeback_supported=False,
        reason="allocator_has_no_slot_generation_or_lease",
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
