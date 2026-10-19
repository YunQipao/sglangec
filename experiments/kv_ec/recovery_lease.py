"""Scheduler-owned, CPU-modelled KV recovery lease protocol.

This is deliberately an integration boundary, not a monkey patch for SGLang.
It models the invariants a future allocator hook must enforce:

* one request, one recovery transaction, all TP ranks agreeing;
* request-row and physical-slot generations are checked together;
* protected slots cannot be freed or allocated while the lease is active;
* CUDA copy completion is an explicit prerequisite for validation/commit;
* timeout, abort, manager close, and rank failure are fail-closed.

No GPU tensor is modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Iterable, Mapping


class LeaseRejected(RuntimeError):
    """An operation would violate recovery ownership or state rules."""


class LeaseState(str, Enum):
    ACTIVE = "ACTIVE"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class RankLeaseObservation:
    tp_rank: int
    tp_size: int
    request_id: str
    request_version: str
    req_pool_idx: int
    req_generation: int
    physical_slots: tuple[int, ...]
    slot_generations: tuple[int, ...]
    checkpoint_version: str
    manager_epoch: str
    token_start: int = 0
    token_count: int = 0


@dataclass
class _Lease:
    lease_id: str
    request_id: str
    request_version: str
    tp_size: int
    slots_by_rank: dict[int, frozenset[int]]
    slot_generations_by_rank: dict[int, dict[int, int]]
    req_generations: dict[int, tuple[int, int]]
    checkpoint_version: str
    manager_epoch: str
    deadline: float
    state: LeaseState = LeaseState.ACTIVE
    copy_complete: set[int] | None = None


class RecoveryLeaseManager:
    """Single-scheduler lease registry; callers invoke it at allocator hooks."""

    def __init__(self, *, timeout_s: float = 5.0, clock: Callable[[], float] = time.monotonic):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = float(timeout_s)
        self._clock = clock
        self._leases: dict[str, _Lease] = {}
        self._slot_owner: dict[int, str] = {}
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise LeaseRejected("manager_closed")

    def _get(self, lease_id: str) -> _Lease:
        self._check_open()
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseRejected("unknown_lease")
        if lease.state in (LeaseState.ACTIVE, LeaseState.VALIDATED) and self._clock() >= lease.deadline:
            self._release_slots(lease)
            lease.state = LeaseState.EXPIRED
        return lease

    @staticmethod
    def _check_observations(observations: Iterable[RankLeaseObservation]) -> tuple[list[RankLeaseObservation], int]:
        obs = sorted(list(observations), key=lambda item: item.tp_rank)
        if not obs:
            raise LeaseRejected("missing_rank_observations")
        tp = obs[0].tp_size
        if tp not in (2, 4) or len(obs) != tp or {item.tp_rank for item in obs} != set(range(tp)):
            raise LeaseRejected("incomplete_tp_rank_set")
        common = (obs[0].request_id, obs[0].request_version,
                  obs[0].checkpoint_version,
                  obs[0].manager_epoch, obs[0].token_start, obs[0].token_count)
        for item in obs:
            if item.tp_size != tp or (
                item.request_id, item.request_version, item.checkpoint_version,
                item.manager_epoch,
                item.token_start, item.token_count,
            ) != common:
                raise LeaseRejected("rank_observation_mismatch")
            if len(item.physical_slots) != len(item.slot_generations):
                raise LeaseRejected("slot_generation_length_mismatch")
            if any(slot <= 0 or generation <= 0 for slot, generation in
                   zip(item.physical_slots, item.slot_generations)):
                raise LeaseRejected("invalid_slot_generation")
            if len(set(item.physical_slots)) != len(item.physical_slots):
                raise LeaseRejected("duplicate_physical_slot")
        if any(item.token_count <= 0 for item in obs):
            raise LeaseRejected("invalid_token_range")
        return obs, tp

    def begin_recovery(
        self,
        lease_id: str,
        observations: Iterable[RankLeaseObservation],
        *,
        manager_epoch: str | None = None,
        checkpoint_version: str | None = None,
    ) -> dict:
        self._check_open()
        if lease_id in self._leases:
            raise LeaseRejected("duplicate_begin")
        obs, tp = self._check_observations(observations)
        epoch = manager_epoch or obs[0].manager_epoch
        version = checkpoint_version or obs[0].checkpoint_version
        if epoch != obs[0].manager_epoch or version != obs[0].checkpoint_version:
            raise LeaseRejected("checkpoint_identity_mismatch")
        slots_by_rank = {item.tp_rank: frozenset(item.physical_slots) for item in obs}
        owners = {(item.tp_rank, slot) for item in obs for slot in item.physical_slots}
        if any(key in self._slot_owner for key in owners):
            raise LeaseRejected("slot_already_leased")
        lease = _Lease(
            lease_id=lease_id,
            request_id=obs[0].request_id,
            request_version=obs[0].request_version,
            tp_size=tp,
            slots_by_rank=slots_by_rank,
            slot_generations_by_rank={
                item.tp_rank: dict(zip(item.physical_slots, item.slot_generations))
                for item in obs
            },
            req_generations={
                item.tp_rank: (item.req_pool_idx, item.req_generation) for item in obs
            },
            checkpoint_version=obs[0].checkpoint_version,
            manager_epoch=obs[0].manager_epoch,
            deadline=self._clock() + self.timeout_s,
            copy_complete=set(),
        )
        self._leases[lease_id] = lease
        for key in owners:
            self._slot_owner[key] = lease_id
        return self.status(lease_id)

    def mark_copy_complete(
        self, lease_id: str, tp_rank: int, *, cuda_event_complete: bool = False
    ) -> None:
        lease = self._get(lease_id)
        if lease.state != LeaseState.ACTIVE:
            raise LeaseRejected("copy_after_terminal_state")
        if tp_rank < 0 or tp_rank >= lease.tp_size:
            raise LeaseRejected("invalid_tp_rank")
        if not cuda_event_complete:
            raise LeaseRejected("cuda_event_incomplete")
        lease.copy_complete.add(tp_rank)

    def validate_recovery(
        self,
        lease_id: str,
        observations: Iterable[RankLeaseObservation],
        *,
        checkpoint_state: str,
        manager_epoch: str,
        checkpoint_version: str | None = None,
    ) -> dict:
        lease = self._get(lease_id)
        if lease.state != LeaseState.ACTIVE:
            raise LeaseRejected("validate_requires_active")
        if checkpoint_state != "COMMITTED":
            raise LeaseRejected("checkpoint_not_committed")
        if manager_epoch != lease.manager_epoch:
            raise LeaseRejected("manager_epoch_changed")
        if checkpoint_version is not None and checkpoint_version != lease.checkpoint_version:
            raise LeaseRejected("checkpoint_version_changed")
        obs, tp = self._check_observations(observations)
        if tp != lease.tp_size or {item.tp_rank for item in obs} != set(range(tp)):
            raise LeaseRejected("tp_changed")
        for item in obs:
            if item.request_id != lease.request_id or item.request_version != lease.request_version:
                raise LeaseRejected("request_identity_changed")
            if item.manager_epoch != lease.manager_epoch:
                raise LeaseRejected("manager_epoch_changed")
            if item.checkpoint_version != lease.checkpoint_version:
                raise LeaseRejected("checkpoint_version_changed")
            if lease.req_generations.get(item.tp_rank) != (
                item.req_pool_idx, item.req_generation
            ):
                raise LeaseRejected("request_generation_changed")
            for slot, generation in zip(item.physical_slots, item.slot_generations):
                if slot not in lease.slots_by_rank[item.tp_rank] or (
                    lease.slot_generations_by_rank[item.tp_rank][slot] != generation
                ):
                    raise LeaseRejected("slot_generation_changed")
        if lease.copy_complete != set(range(tp)):
            raise LeaseRejected("cuda_copy_incomplete")
        lease.state = LeaseState.VALIDATED
        return self.status(lease_id)

    def commit_recovery(self, lease_id: str) -> dict:
        lease = self._get(lease_id)
        if lease.state != LeaseState.VALIDATED:
            raise LeaseRejected("commit_requires_validated")
        self._release_slots(lease)
        lease.state = LeaseState.COMMITTED
        return self.status(lease_id)

    def abort_recovery(self, lease_id: str, reason: str = "aborted") -> dict:
        lease = self._get(lease_id)
        if lease.state in (LeaseState.COMMITTED, LeaseState.ABORTED, LeaseState.EXPIRED):
            raise LeaseRejected("abort_after_terminal_state")
        self._release_slots(lease)
        lease.state = LeaseState.ABORTED
        return {**self.status(lease_id), "reason": reason}

    def before_free(self, slots: Iterable[int], *, tp_rank: int = 0) -> None:
        self._check_open()
        conflict = {(tp_rank, int(slot)) for slot in slots} & set(self._slot_owner)
        if conflict:
            raise LeaseRejected("protected_slot_free")

    def before_alloc(self, slots: Iterable[int], *, tp_rank: int = 0) -> None:
        self._check_open()
        conflict = {(tp_rank, int(slot)) for slot in slots} & set(self._slot_owner)
        if conflict:
            raise LeaseRejected("protected_slot_reuse")

    def close(self) -> None:
        self._closed = True
        for lease in self._leases.values():
            if lease.state in (LeaseState.ACTIVE, LeaseState.VALIDATED):
                self._release_slots(lease)
                lease.state = LeaseState.ABORTED

    def _release_slots(self, lease: _Lease) -> None:
        for rank, slots in lease.slots_by_rank.items():
            for slot in slots:
                key = (rank, slot)
                if self._slot_owner.get(key) == lease.lease_id:
                    del self._slot_owner[key]

    def status(self, lease_id: str) -> dict:
        lease = self._leases.get(lease_id)
        if lease is None:
            return {"lease_id": lease_id, "state": "UNKNOWN"}
        return {
            "lease_id": lease.lease_id,
            "state": lease.state.value,
            "request_id": lease.request_id,
            "request_version": lease.request_version,
            "tp_size": lease.tp_size,
            "slot_count": sum(len(slots) for slots in lease.slots_by_rank.values()),
            "copy_complete_ranks": sorted(lease.copy_complete or ()),
            "checkpoint_version": lease.checkpoint_version,
            "manager_epoch": lease.manager_epoch,
        }
