"""CPU-only tests for the real-scheduler mapping guard."""

from types import SimpleNamespace as NS
import os
import unittest
from unittest.mock import patch

import torch

from experiments.kv_ec.scheduler_recovery_guard import (
    RecoveryGuard,
    RecoveryGuardRejected,
    SchedulerLeaseTransaction,
    read_scheduler_mapping,
)


def scheduler(tp=4, prefix_disabled=True, overlap_disabled=True, chunk=-1):
    pool = NS(
        free_slots=[1, 2],
        req_generation=torch.tensor([0, 0, 0, 9], dtype=torch.int64),
        req_to_token=torch.tensor([[0] * 8, [0] * 8, [0] * 8, [8, 9, 13, 17, 21, 22, 27, 31]]),
    )
    return NS(
        server_args=NS(
            disable_overlap_schedule=overlap_disabled,
            disable_radix_cache=prefix_disabled,
            chunked_prefill_size=chunk,
            max_running_requests=1,
        ),
        model_worker=NS(ps=NS(tp_rank=0, tp_size=tp, attn_tp_size=tp)),
        req_to_token_pool=pool,
    )


class SchedulerRecoveryGuardTests(unittest.TestCase):
    def test_reads_real_mapping_but_blocks_writeback(self):
        record = read_scheduler_mapping(scheduler(), NS(req_pool_idx=3, rid="rid"),
                                        token_start=0, token_count=8)
        self.assertEqual(record.physical_slots, (8, 9, 13, 17, 21, 22, 27, 31))
        self.assertEqual(record.req_generation, 9)
        self.assertFalse(record.slot_generation_available)
        self.assertFalse(record.writeback_supported)

    def test_unsupported_runtime_modes_fail_closed(self):
        for kwargs in (
            {"prefix_disabled": False},
            {"overlap_disabled": False},
            {"chunk": 128},
        ):
            with self.subTest(kwargs):
                with self.assertRaises(RecoveryGuardRejected):
                    read_scheduler_mapping(
                        scheduler(**kwargs), NS(req_pool_idx=3, rid="rid"),
                        token_start=0, token_count=8,
                    )

    def test_disabled_guard_is_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            result = RecoveryGuard().observe(
                scheduler(), NS(req_pool_idx=3, rid="rid"),
                token_start=0, token_count=8,
            )
        self.assertEqual(result, {"enabled": False, "state": "DISABLED"})

    def test_enabled_guard_reports_read_only_block(self):
        with patch.dict(os.environ, {"SGLANG_KV_EC_RECOVERY_GUARD": "1"}, clear=True):
            result = RecoveryGuard().observe(
                scheduler(), NS(req_pool_idx=3, rid="rid"),
                token_start=0, token_count=8,
            )
        self.assertEqual(result["state"], "READ_ONLY_BLOCKED")
        self.assertFalse(result["writeback_supported"])

    def test_scheduler_lease_roundtrip_revalidates_generation_and_releases(self):
        class Event:
            def __init__(self):
                self.done = False
            def synchronize(self):
                self.done = True
            def query(self):
                return self.done

        class Allocator:
            page_size = 1
            def __init__(self):
                self.generations = {8: 1, 9: 1}
                self.leases = set()
                self.event = Event()
            def enable_slot_leases(self, **kwargs):
                for slot in kwargs["active_slots"]:
                    self.generations.setdefault(int(slot), 1)
            def slot_generations(self, slots):
                return tuple(self.generations[int(slot)] for slot in slots)
            def begin_slot_lease(self, lease_id, **kwargs):
                self.leases.add(lease_id)
            def validate_slot_lease(self, lease_id, **kwargs):
                assert lease_id in self.leases
            def record_slot_lease_cuda_event(self, lease_id, **kwargs):
                return self.event
            def release_slot_lease(self, lease_id, **kwargs):
                self.leases.remove(lease_id)
            def available_size(self):
                return 12

        alloc = Allocator()
        s = scheduler()
        s.token_to_kv_pool_allocator = alloc
        req = NS(req_pool_idx=3, rid="rid")
        with patch.dict(os.environ, {"SGLANG_KV_EC_SCHEDULER_LEASE": "1"}, clear=False), \
             patch("experiments.kv_ec.scheduler_recovery_guard.torch.cuda.current_stream", return_value=None):
            txn = SchedulerLeaseTransaction.begin(
                s, req, token_start=0, token_count=2,
                snapshot_id="snap", slots=(8, 9), req_generation=9,
            )
            result = txn.complete(s, req, slots=(8, 9))
        self.assertEqual(result["state"], "LOCAL_RELEASED")
        self.assertTrue(result["event_recorded"])
        self.assertTrue(result["event_complete"])
        self.assertEqual(alloc.leases, set())

    def test_scheduler_lease_rejects_generation_change(self):
        class Event:
            def synchronize(self): pass
            def query(self): return True
        class Allocator:
            page_size = 1
            def __init__(self): self.generations = {8: 1}
            def enable_slot_leases(self, **kwargs): pass
            def slot_generations(self, slots): return tuple(self.generations[int(x)] for x in slots)
            def begin_slot_lease(self, *args, **kwargs): pass
            def validate_slot_lease(self, *args, **kwargs): pass
            def record_slot_lease_cuda_event(self, *args, **kwargs): return Event()
            def release_slot_lease(self, *args, **kwargs): pass
        s = scheduler()
        s.token_to_kv_pool_allocator = Allocator()
        req = NS(req_pool_idx=3, rid="rid")
        txn = SchedulerLeaseTransaction.begin(
            s, req, token_start=0, token_count=1,
            snapshot_id="snap", slots=(8,), req_generation=9,
        )
        s.req_to_token_pool.req_generation[3] = 10
        with patch("experiments.kv_ec.scheduler_recovery_guard.torch.cuda.current_stream", return_value=None):
            with self.assertRaisesRegex(RecoveryGuardRejected, "request_generation_changed"):
                txn.complete(s, req, slots=(8,))

    def test_scheduler_lease_rejects_manager_epoch_change(self):
        class Event:
            def synchronize(self): pass
            def query(self): return True
        class Allocator:
            page_size = 1
            def enable_slot_leases(self, **kwargs): pass
            def slot_generations(self, slots): return (1,)
            def begin_slot_lease(self, *args, **kwargs): pass
            def validate_slot_lease(self, *args, **kwargs): pass
            def record_slot_lease_cuda_event(self, *args, **kwargs): return Event()
            def release_slot_lease(self, *args, **kwargs): pass
        s = scheduler()
        s.token_to_kv_pool_allocator = Allocator()
        req = NS(req_pool_idx=3, rid="rid")
        with patch.dict(os.environ, {"SGLANG_KV_EC_MANAGER_EPOCH": "epoch-1"}, clear=False):
            txn = SchedulerLeaseTransaction.begin(
                s, req, token_start=0, token_count=1,
                snapshot_id="snap", slots=(8,), req_generation=9,
            )
        with patch.dict(os.environ, {"SGLANG_KV_EC_MANAGER_EPOCH": "epoch-2"}, clear=False), \
             patch("experiments.kv_ec.scheduler_recovery_guard.torch.cuda.current_stream", return_value=None):
            with self.assertRaisesRegex(RecoveryGuardRejected, "manager_epoch_changed"):
                txn.complete(s, req, slots=(8,))


if __name__ == "__main__":
    unittest.main()
