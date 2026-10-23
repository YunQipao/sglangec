"""CPU tests for the real page-size-one allocator lease integration."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.slot_lease import SlotLeaseRejected


class _Event:
    def __init__(self, complete=False):
        self.complete = complete

    def query(self):
        return self.complete


class SlotLeaseAllocatorTests(unittest.TestCase):
    def allocator(self):
        return TokenToKVPoolAllocator(
            16, torch.bfloat16, "cpu", NS(), need_sort=False
        )

    def test_default_disabled_preserves_allocator(self):
        allocator = self.allocator()
        on_alloc = Mock(wraps=allocator.slot_lease_registry.on_alloc)
        on_free = Mock(wraps=allocator.slot_lease_registry.on_free)
        allocator.slot_lease_registry.on_alloc = on_alloc
        allocator.slot_lease_registry.on_free = on_free
        slots = allocator.alloc(2)
        allocator.free(slots)
        on_alloc.assert_not_called()
        on_free.assert_not_called()
        self.assertEqual(allocator.available_size(), 16)

    def test_enabled_path_runs_slot_checks(self):
        allocator = self.allocator()
        allocator.enable_slot_leases()
        on_alloc = Mock(wraps=allocator.slot_lease_registry.on_alloc)
        on_free = Mock(wraps=allocator.slot_lease_registry.on_free)
        allocator.slot_lease_registry.on_alloc = on_alloc
        allocator.slot_lease_registry.on_free = on_free
        slots = allocator.alloc(2)
        allocator.free(slots)
        on_alloc.assert_called_once()
        on_free.assert_called_once()

    def test_closed_registry_rejects_alloc_and_free(self):
        allocator = self.allocator()
        slots = allocator.alloc(1)
        allocator.slot_lease_registry.close()
        with self.assertRaisesRegex(SlotLeaseRejected, "registry_closed"):
            allocator.alloc(1)
        with self.assertRaisesRegex(SlotLeaseRejected, "registry_closed"):
            allocator.free(slots)

    def test_generation_increments_on_real_alloc_and_lease_blocks_free_reuse(self):
        allocator = self.allocator()
        allocator.enable_slot_leases()
        slots = allocator.alloc(2)
        self.assertEqual(allocator.slot_generations(slots), (1, 1))
        lease = allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        self.assertEqual(lease.generations, (1, 1))
        with self.assertRaisesRegex(SlotLeaseRejected, "protected_slot_free"):
            allocator.free(slots)
        with self.assertRaisesRegex(SlotLeaseRejected, "protected_slot_reuse"):
            allocator.slot_lease_registry.on_alloc(slots.tolist())
        with self.assertRaisesRegex(SlotLeaseRejected, "cuda_event"):
            allocator.slot_lease_registry.release("lease", manager_epoch="m1")
        with self.assertRaisesRegex(SlotLeaseRejected, "cuda_event"):
            allocator.validate_slot_lease(
                "lease", manager_epoch="m1", request_id="r1",
                request_version="v1", slots=slots,
            )
        event = _Event()
        allocator.attach_slot_lease_cuda_event("lease", event)
        with self.assertRaisesRegex(SlotLeaseRejected, "cuda_event"):
            allocator.release_slot_lease("lease", manager_epoch="m1")
        event.complete = True
        allocator.validate_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        allocator.release_slot_lease("lease", manager_epoch="m1")
        allocator.free(slots)
        allocator.alloc(14)
        reused = allocator.alloc(2)
        self.assertEqual(allocator.slot_generations(reused), (2, 2))

    def test_generation_and_manager_epoch_fail_closed(self):
        allocator = self.allocator()
        allocator.enable_slot_leases()
        slots = allocator.alloc(1)
        allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        allocator.attach_slot_lease_cuda_event("lease", _Event(True))
        with self.assertRaisesRegex(SlotLeaseRejected, "manager_epoch"):
            allocator.validate_slot_lease(
                "lease", manager_epoch="m2", request_id="r1",
                request_version="v1", slots=slots,
            )
        with self.assertRaisesRegex(SlotLeaseRejected, "request_identity"):
            allocator.validate_slot_lease(
                "lease", manager_epoch="m1", request_id="r2",
                request_version="v1", slots=slots,
            )

    def test_clear_rejects_active_lease_and_then_resets_generation(self):
        allocator = self.allocator()
        allocator.enable_slot_leases()
        slots = allocator.alloc(1)
        allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        with self.assertRaisesRegex(SlotLeaseRejected, "active_leases"):
            allocator.clear()
        allocator.attach_slot_lease_cuda_event("lease", _Event(True))
        allocator.release_slot_lease("lease", manager_epoch="m1")
        allocator.clear()
        self.assertEqual(int(allocator.slot_lease_registry.generation[1]), 0)

    def test_timeout_does_not_release_before_cuda_event(self):
        now = [0.0]
        allocator = TokenToKVPoolAllocator(
            16, torch.bfloat16, "cpu", NS(), need_sort=False
        )
        allocator.slot_lease_registry = __import__(
            "sglang.srt.mem_cache.slot_lease", fromlist=["SlotLeaseRegistry"]
        ).SlotLeaseRegistry(16, timeout_s=1, clock=lambda: now[0])
        allocator.slot_lease_registry.enable()
        slots = allocator.alloc(1)
        allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        now[0] = 2
        with self.assertRaisesRegex(SlotLeaseRejected, "expired"):
            allocator.validate_slot_lease(
                "lease", manager_epoch="m1", request_id="r1",
                request_version="v1", slots=slots,
            )
        with self.assertRaisesRegex(SlotLeaseRejected, "protected_slot_free"):
            allocator.free(slots)
        allocator.attach_slot_lease_cuda_event("lease", _Event(True))
        with self.assertRaisesRegex(SlotLeaseRejected, "protected_slot_free"):
            allocator.free(slots)
        allocator.abort_slot_lease("lease", manager_epoch="m1")
        allocator.free(slots)

    def test_free_group_checks_protected_slots_before_mutating_pool(self):
        self._check_free_group_protection()

    def test_completed_event_timeout_still_requires_explicit_abort(self):
        from sglang.srt.mem_cache.slot_lease import SlotLeaseRegistry

        now = [0.0]
        allocator = self.allocator()
        allocator.slot_lease_registry = SlotLeaseRegistry(
            16, timeout_s=1, clock=lambda: now[0]
        )
        allocator.enable_slot_leases(timeout_s=1)
        slots = allocator.alloc(1)
        allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        allocator.attach_slot_lease_cuda_event("lease", _Event(True))
        now[0] = 2
        self.assertEqual(allocator.slot_lease_registry.active_lease_count, 1)
        with self.assertRaisesRegex(SlotLeaseRejected, "protected_slot_free"):
            allocator.free(slots)
        with self.assertRaisesRegex(SlotLeaseRejected, "expired"):
            allocator.release_slot_lease("lease", manager_epoch="m1")
        with self.assertRaisesRegex(SlotLeaseRejected, "manager_epoch"):
            allocator.abort_slot_lease("lease", manager_epoch="wrong")
        allocator.abort_slot_lease("lease", manager_epoch="m1")
        allocator.free(slots)
        self.assertEqual(allocator.available_size(), 16)

    def _check_free_group_protection(self):
        allocator = self.allocator()
        allocator.enable_slot_leases()
        slots = allocator.alloc(2)
        allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots[:1],
        )
        available_before = allocator.available_size()
        allocator.free_group_begin()
        allocator.free(slots)
        with self.assertRaisesRegex(SlotLeaseRejected, "protected_slot_free"):
            allocator.free_group_end()
        self.assertEqual(allocator.available_size(), available_before)
        self.assertEqual(len(allocator.free_group), 0)

    def test_registry_cannot_be_replaced_while_lease_is_active(self):
        allocator = self.allocator()
        allocator.enable_slot_leases()
        slots = allocator.alloc(1)
        allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        with self.assertRaisesRegex(SlotLeaseRejected, "active_leases"):
            allocator.enable_slot_leases(timeout_s=10)

    def test_close_is_fail_closed_and_does_not_reuse_protected_slot(self):
        allocator = self.allocator()
        allocator.enable_slot_leases()
        slots = allocator.alloc(1)
        allocator.begin_slot_lease(
            "lease", manager_epoch="m1", request_id="r1",
            request_version="v1", slots=slots,
        )
        allocator.slot_lease_registry.close()
        with self.assertRaisesRegex(SlotLeaseRejected, "protected_slot_free"):
            allocator.free(slots)


if __name__ == "__main__":
    unittest.main()
