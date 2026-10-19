"""CPU-only tests for the scheduler-owned recovery lease state machine."""

from dataclasses import replace
import unittest

from experiments.kv_ec.recovery_lease import (
    LeaseRejected,
    LeaseState,
    RankLeaseObservation,
    RecoveryLeaseManager,
)


def observations(tp=4, generation=7, slot_generation=11):
    slots = (8, 11, 15)
    return [
        RankLeaseObservation(
            tp_rank=rank, tp_size=tp, request_id="rid", request_version="v1",
            req_pool_idx=3, req_generation=generation, physical_slots=slots,
            slot_generations=tuple(slot_generation for _ in slots),
            checkpoint_version="cp-v1",
        )
        for rank in range(tp)
    ]


class RecoveryLeaseTests(unittest.TestCase):
    def test_begin_blocks_free_and_reuse_until_commit(self):
        manager = RecoveryLeaseManager(timeout_s=10)
        manager.begin_recovery("lease", observations())
        with self.assertRaisesRegex(LeaseRejected, "protected_slot"):
            manager.before_free([11])
        with self.assertRaisesRegex(LeaseRejected, "protected_slot"):
            manager.before_alloc([15])
        for rank in range(4):
            manager.mark_copy_complete("lease", rank)
        manager.validate_recovery("lease", observations(), checkpoint_state="COMMITTED",
                                 manager_epoch="manager-1")
        self.assertEqual(manager.commit_recovery("lease")["state"], LeaseState.COMMITTED.value)
        manager.before_free([11])

    def test_request_release_race_and_generation_change_fail_closed(self):
        manager = RecoveryLeaseManager(timeout_s=10)
        manager.begin_recovery("lease", observations())
        stale = [replace(item, req_generation=8) for item in observations()]
        for rank in range(4):
            manager.mark_copy_complete("lease", rank)
        with self.assertRaisesRegex(LeaseRejected, "request_generation"):
            manager.validate_recovery("lease", stale, checkpoint_state="COMMITTED",
                                      manager_epoch="manager-1")

    def test_slot_reuse_and_rank_mismatch_are_rejected(self):
        manager = RecoveryLeaseManager(timeout_s=10)
        manager.begin_recovery("lease", observations())
        with self.assertRaisesRegex(LeaseRejected, "slot_already"):
            manager.begin_recovery("other", observations())
        bad = observations()
        bad[2] = replace(bad[2], slot_generations=(99, 11, 15))
        for rank in range(4):
            manager.mark_copy_complete("lease", rank)
        with self.assertRaisesRegex(LeaseRejected, "rank_observation|slot_generation"):
            manager.validate_recovery("lease", bad, checkpoint_state="COMMITTED",
                                      manager_epoch="manager-1")

    def test_timeout_and_abort_release_slots(self):
        now = [0.0]
        manager = RecoveryLeaseManager(timeout_s=2, clock=lambda: now[0])
        manager.begin_recovery("lease", observations())
        now[0] = 3.0
        with self.assertRaisesRegex(LeaseRejected, "unknown|expired|active|copy_after"):
            manager.mark_copy_complete("lease", 0)
        self.assertEqual(manager.status("lease")["state"], LeaseState.EXPIRED.value)
        manager.before_free([8])
        manager = RecoveryLeaseManager(timeout_s=10)
        manager.begin_recovery("lease", observations())
        self.assertEqual(manager.abort_recovery("lease")["state"], LeaseState.ABORTED.value)
        with self.assertRaisesRegex(LeaseRejected, "abort_after"):
            manager.abort_recovery("lease")

    def test_duplicate_operations_checkpoint_manager_and_copy_guards(self):
        manager = RecoveryLeaseManager(timeout_s=10)
        manager.begin_recovery("lease", observations())
        with self.assertRaisesRegex(LeaseRejected, "duplicate_begin"):
            manager.begin_recovery("lease", observations())
        with self.assertRaisesRegex(LeaseRejected, "checkpoint"):
            manager.validate_recovery("lease", observations(), checkpoint_state="PENDING",
                                      manager_epoch="manager-1")
        for rank in range(4):
            manager.mark_copy_complete("lease", rank)
        manager.validate_recovery("lease", observations(), checkpoint_state="COMMITTED",
                                 manager_epoch="manager-1")
        manager.commit_recovery("lease")
        with self.assertRaisesRegex(LeaseRejected, "commit_requires"):
            manager.commit_recovery("lease")
        manager.close()
        with self.assertRaisesRegex(LeaseRejected, "manager_closed"):
            manager.begin_recovery("new", observations())

    def test_cuda_copy_incomplete_and_invalid_tp_are_rejected(self):
        manager = RecoveryLeaseManager(timeout_s=10)
        manager.begin_recovery("lease", observations())
        for rank in range(3):
            manager.mark_copy_complete("lease", rank)
        with self.assertRaisesRegex(LeaseRejected, "cuda_copy"):
            manager.validate_recovery("lease", observations(), checkpoint_state="COMMITTED",
                                      manager_epoch="manager-1")
        manager = RecoveryLeaseManager()
        with self.assertRaisesRegex(LeaseRejected, "incomplete_tp"):
            manager.begin_recovery("bad", observations()[:3])


if __name__ == "__main__":
    unittest.main()
