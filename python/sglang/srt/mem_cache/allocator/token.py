"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.slot_lease import (
    SlotLease,
    SlotLeaseRegistry,
    SlotLeaseRejected,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.slot_lease_registry = SlotLeaseRegistry(size)
        self.clear()

    def clear(self):
        self.slot_lease_registry.clear()
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        registry = self.slot_lease_registry
        if registry.enabled or registry.closed:
            registry.on_alloc(select_index.tolist())
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            registry = self.slot_lease_registry
            if registry.enabled or registry.closed:
                registry.on_free(free_index.tolist())
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

    def enable_slot_leases(
        self, *, timeout_s: float = 5.0, active_slots=None
    ) -> None:
        if timeout_s != self.slot_lease_registry._timeout_s:
            if self.slot_lease_registry.active_lease_count:
                raise SlotLeaseRejected("active_leases")
            self.slot_lease_registry = SlotLeaseRegistry(
                self.size, timeout_s=timeout_s
            )
        self.slot_lease_registry.enable(active_slots or ())

    def slot_generations(self, slots: torch.Tensor | list[int]) -> tuple[int, ...]:
        values = [int(slot) for slot in (slots.tolist() if isinstance(slots, torch.Tensor) else slots)]
        if any(slot <= 0 or slot >= self.slot_lease_registry.generation.numel() for slot in values):
            raise ValueError("slot out of bounds")
        return tuple(int(self.slot_lease_registry.generation[slot].item()) for slot in values)

    def begin_slot_lease(
        self, lease_id: str, *, manager_epoch: str, request_id: str,
        request_version: str, slots: torch.Tensor | list[int],
    ) -> SlotLease:
        values = [int(slot) for slot in (slots.tolist() if isinstance(slots, torch.Tensor) else slots)]
        return self.slot_lease_registry.begin(
            lease_id, manager_epoch=manager_epoch, request_id=request_id,
            request_version=request_version, slots=values,
            generations=self.slot_generations(values),
        )

    def validate_slot_lease(
        self, lease_id: str, *, manager_epoch: str, request_id: str,
        request_version: str, slots: torch.Tensor | list[int],
    ) -> SlotLease:
        values = [int(slot) for slot in (slots.tolist() if isinstance(slots, torch.Tensor) else slots)]
        return self.slot_lease_registry.validate(
            lease_id, manager_epoch=manager_epoch, request_id=request_id,
            request_version=request_version, slots=values,
        )

    def record_slot_lease_cuda_event(self, lease_id: str, *, stream=None) -> object:
        """Record completion on the worker's actual CUDA stream."""
        return self.slot_lease_registry.record_cuda_event(lease_id, stream=stream)

    def attach_slot_lease_cuda_event(self, lease_id: str, event: object) -> None:
        """Attach a worker-created CUDA event for controlled integration tests."""
        self.slot_lease_registry.attach_cuda_event(lease_id, event)

    def release_slot_lease(self, lease_id: str, *, manager_epoch: str) -> None:
        """Release only after the recorded CUDA event reports completion."""
        self.slot_lease_registry.release(lease_id, manager_epoch=manager_epoch)

    def abort_slot_lease(self, lease_id: str, *, manager_epoch: str) -> None:
        """Explicit abort after a real completion fence, even on timeout."""
        self.slot_lease_registry.abort(lease_id, manager_epoch=manager_epoch)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
