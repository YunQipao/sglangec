"""CPU tests for the real page-size-one allocator lease integration."""

import unittest
from types import SimpleNamespace as NS

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.slot_lease import SlotLeaseRejected


class SlotLeaseAllocatorTests(unittest.TestCase):
    def allocator(self):
        return TokenToKVPoolAllocator(
            16, torch.bfloat16, "cpu", NS(), need_sort=False
        )

    def test_default_disabled_preserves_allocator(self):
        allocator = self.allocator()
        slots = allocator.alloc(2)
        allocator.free(slots)
        self.assertEqual(allocator.available_size(), 16)

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
        allocator.mark_slot_lease_cuda_complete("lease")
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
        allocator.mark_slot_lease_cuda_complete("lease")
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
        allocator.mark_slot_lease_cuda_complete("lease")
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
        allocator.mark_slot_lease_cuda_complete("lease")
        allocator.free(slots)


if __name__ == "__main__":
    unittest.main()
