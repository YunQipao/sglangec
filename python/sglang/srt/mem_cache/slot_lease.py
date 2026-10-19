"""Minimal per-rank KV slot generation and lease registry.

This module is intentionally local to one scheduler/allocator process.  It
does not coordinate TP ranks and does not write KV tensors.  A caller must
coordinate identical request/version metadata across ranks before beginning
one lease per rank.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable

import torch


class SlotLeaseRejected(RuntimeError):
    """A slot lease operation would violate allocator ownership."""


@dataclass(frozen=True)
class SlotLease:
    lease_id: str
    manager_epoch: str
    request_id: str
    request_version: str
    slots: tuple[int, ...]
    generations: tuple[int, ...]
    deadline: float
    cuda_event: object | None = None
    expired: bool = False


class SlotLeaseRegistry:
    """Per-rank registry used by the non-destructive allocator integration."""

    def __init__(
        self,
        size: int,
        *,
        timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if size <= 0 or timeout_s <= 0:
            raise ValueError("invalid slot lease registry configuration")
        self._clock = clock
        self._timeout_s = float(timeout_s)
        self._generation = torch.zeros(size + 1, dtype=torch.int64)
        self._leases: dict[str, SlotLease] = {}
        self._owners: dict[int, str] = {}
        self._enabled = False
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._closed

    @property
    def generation(self) -> torch.Tensor:
        return self._generation

    @property
    def active_lease_count(self) -> int:
        self._expire()
        return len(self._leases)

    def enable(self, active_slots: Iterable[int] = ()) -> None:
        if self._closed:
            raise SlotLeaseRejected("registry_closed")
        for slot in active_slots:
            slot = int(slot)
            if 0 < slot < self._generation.numel() and self._generation[slot] == 0:
                self._generation[slot] = 1
        self._enabled = True

    def disable(self) -> None:
        if self._leases:
            raise SlotLeaseRejected("active_leases")
        self._enabled = False

    def on_alloc(self, slots: Iterable[int]) -> None:
        if not self.enabled and not self._closed:
            return
        values = tuple(int(slot) for slot in slots)
        self._expire()
        if any(slot in self._owners for slot in values):
            raise SlotLeaseRejected("protected_slot_reuse")
        for slot in values:
            if slot <= 0 or slot >= self._generation.numel():
                raise SlotLeaseRejected("slot_out_of_bounds")
            self._generation[slot] += 1

    def on_free(self, slots: Iterable[int]) -> None:
        if not self.enabled and not self._closed:
            return
        self._expire()
        conflict = {int(slot) for slot in slots} & set(self._owners)
        if conflict:
            raise SlotLeaseRejected("protected_slot_free")

    def begin(
        self,
        lease_id: str,
        *,
        manager_epoch: str,
        request_id: str,
        request_version: str,
        slots: Iterable[int],
        generations: Iterable[int],
    ) -> SlotLease:
        if not self.enabled:
            raise SlotLeaseRejected("slot_lease_disabled")
        self._expire()
        if lease_id in self._leases:
            raise SlotLeaseRejected("duplicate_lease")
        if not manager_epoch or not request_id or not request_version:
            raise SlotLeaseRejected("missing_lease_identity")
        slot_tuple = tuple(int(slot) for slot in slots)
        generation_tuple = tuple(int(value) for value in generations)
        if not slot_tuple or len(slot_tuple) != len(generation_tuple):
            raise SlotLeaseRejected("invalid_lease_slots")
        if len(set(slot_tuple)) != len(slot_tuple):
            raise SlotLeaseRejected("duplicate_lease_slot")
        if any(slot <= 0 or slot >= self._generation.numel() for slot in slot_tuple):
            raise SlotLeaseRejected("slot_out_of_bounds")
        if any(slot in self._owners for slot in slot_tuple):
            raise SlotLeaseRejected("slot_already_leased")
        for slot, expected in zip(slot_tuple, generation_tuple):
            actual = int(self._generation[slot].item())
            if actual <= 0 or actual != expected:
                raise SlotLeaseRejected("slot_generation_mismatch")
        lease = SlotLease(
            lease_id=lease_id,
            manager_epoch=manager_epoch,
            request_id=request_id,
            request_version=request_version,
            slots=slot_tuple,
            generations=generation_tuple,
            deadline=self._clock() + self._timeout_s,
        )
        self._leases[lease_id] = lease
        for slot in slot_tuple:
            self._owners[slot] = lease_id
        return lease

    def record_cuda_event(self, lease_id: str, *, stream=None) -> object:
        """Record a real CUDA event owned by the worker stream."""
        if self._closed:
            raise SlotLeaseRejected("registry_closed")
        self._expire()
        lease = self._leases.get(lease_id)
        if lease is None:
            raise SlotLeaseRejected("unknown_lease")
        if lease.expired:
            raise SlotLeaseRejected("lease_expired")
        if not torch.cuda.is_available():
            raise SlotLeaseRejected("cuda_unavailable")
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(stream)
        updated = self._replace(lease, cuda_event=event)
        self._leases[lease_id] = updated
        return event

    def attach_cuda_event(self, lease_id: str, event: object) -> None:
        """Attach a worker-created event; completion is checked via query()."""
        if self._closed:
            raise SlotLeaseRejected("registry_closed")
        self._expire()
        lease = self._leases.get(lease_id)
        if lease is None:
            raise SlotLeaseRejected("unknown_lease")
        if not callable(getattr(event, "query", None)):
            raise SlotLeaseRejected("invalid_cuda_event")
        updated = self._replace(lease, cuda_event=event)
        self._leases[lease_id] = updated
        if updated.expired and self._event_complete(updated):
            self._release(updated)

    @staticmethod
    def _replace(lease: SlotLease, **changes) -> SlotLease:
        values = {
            "lease_id": lease.lease_id,
            "manager_epoch": lease.manager_epoch,
            "request_id": lease.request_id,
            "request_version": lease.request_version,
            "slots": lease.slots,
            "generations": lease.generations,
            "deadline": lease.deadline,
            "cuda_event": lease.cuda_event,
            "expired": lease.expired,
        }
        values.update(changes)
        return SlotLease(**values)

    @staticmethod
    def _event_complete(lease: SlotLease) -> bool:
        if lease.cuda_event is None:
            return False
        try:
            return bool(lease.cuda_event.query())
        except Exception as exc:
            raise SlotLeaseRejected("cuda_event_query_failed") from exc

    def validate(
        self,
        lease_id: str,
        *,
        manager_epoch: str,
        request_id: str,
        request_version: str,
        slots: Iterable[int],
    ) -> SlotLease:
        lease = self._get(lease_id)
        if lease.expired:
            raise SlotLeaseRejected("lease_expired")
        if not self._event_complete(lease):
            raise SlotLeaseRejected("cuda_event_incomplete")
        if manager_epoch != lease.manager_epoch:
            raise SlotLeaseRejected("manager_epoch_changed")
        if request_id != lease.request_id or request_version != lease.request_version:
            raise SlotLeaseRejected("request_identity_changed")
        slot_tuple = tuple(int(slot) for slot in slots)
        if slot_tuple != lease.slots:
            raise SlotLeaseRejected("slot_mapping_changed")
        for slot, expected in zip(lease.slots, lease.generations):
            if int(self._generation[slot].item()) != expected:
                raise SlotLeaseRejected("slot_generation_changed")
        return lease

    def release(
        self, lease_id: str, *, manager_epoch: str,
    ) -> None:
        lease = self._get(lease_id)
        if lease.expired:
            raise SlotLeaseRejected("lease_expired")
        if manager_epoch != lease.manager_epoch:
            raise SlotLeaseRejected("manager_epoch_changed")
        if not self._event_complete(lease):
            raise SlotLeaseRejected("cuda_event_incomplete")
        self._release(lease)

    def close(self) -> None:
        # Fail closed: never make in-flight GPU slots reusable on shutdown.
        self._closed = True

    def clear(self) -> None:
        if self._leases:
            raise SlotLeaseRejected("active_leases")
        self._owners.clear()
        self._generation.zero_()

    def _get(self, lease_id: str) -> SlotLease:
        if self._closed:
            raise SlotLeaseRejected("registry_closed")
        self._expire()
        lease = self._leases.get(lease_id)
        if lease is None:
            raise SlotLeaseRejected("unknown_lease")
        if lease.expired:
            raise SlotLeaseRejected("lease_expired")
        return lease

    def _expire(self) -> None:
        now = self._clock()
        for lease in list(self._leases.values()):
            if now >= lease.deadline and not lease.expired:
                if lease.cuda_event is None or not self._event_complete(lease):
                    self._leases[lease.lease_id] = SlotLease(
                        lease_id=lease.lease_id,
                        manager_epoch=lease.manager_epoch,
                        request_id=lease.request_id,
                        request_version=lease.request_version,
                        slots=lease.slots,
                        generations=lease.generations,
                        deadline=lease.deadline,
                        cuda_event=lease.cuda_event,
                        expired=True,
                    )
                    continue
                self._release(lease)

    def _release(self, lease: SlotLease) -> None:
        self._leases.pop(lease.lease_id, None)
        for slot in lease.slots:
            if self._owners.get(slot) == lease.lease_id:
                del self._owners[slot]
