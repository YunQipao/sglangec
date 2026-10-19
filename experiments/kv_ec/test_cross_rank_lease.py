import unittest

from experiments.kv_ec.cross_rank_lease import (
    CrossRankLeaseCoordinator,
    CrossRankLeaseState,
)
from experiments.kv_ec.recovery_lease import LeaseRejected, RankLeaseObservation


def observations(*, version="req-v1", checkpoint="cp-v1", epoch="epoch-1"):
    return [
        RankLeaseObservation(
            tp_rank=rank,
            tp_size=4,
            request_id="rid",
            request_version=version,
            req_pool_idx=10 + rank,
            req_generation=7,
            physical_slots=(100 + rank * 10, 101 + rank * 10),
            slot_generations=(1, 1),
            checkpoint_version=checkpoint,
            manager_epoch=epoch,
            token_start=0,
            token_count=256,
        )
        for rank in range(4)
    ]


class CrossRankLeaseTests(unittest.TestCase):
    def coordinator(self, **kwargs):
        epoch = kwargs.pop("manager_epoch", "epoch-1")
        return CrossRankLeaseCoordinator(
            manager_epoch=epoch,
            **kwargs,
        )

    def test_full_state_machine_requires_all_ranks(self):
        manager = self.coordinator()
        tx = "tx-1"
        self.assertEqual(
            manager.prepare(tx, observations(), checkpoint_version="cp-v1").state,
            CrossRankLeaseState.PREPARE,
        )
        for rank in range(3):
            self.assertEqual(
                manager.mark_local_leased(tx, rank).state,
                CrossRankLeaseState.LOCAL_LEASED,
            )
        with self.assertRaisesRegex(LeaseRejected, "invalid_transition"):
            manager.mark_cuda_completed(tx, 0)
        self.assertEqual(
            manager.mark_local_leased(tx, 3).state,
            CrossRankLeaseState.ALL_RANKS_READY,
        )
        for rank in range(4):
            state = manager.mark_cuda_completed(tx, rank).state
        self.assertEqual(state, CrossRankLeaseState.CUDA_COMPLETED)
        for rank in range(3):
            self.assertEqual(
                manager.mark_validated(tx, rank).state,
                CrossRankLeaseState.CUDA_COMPLETED,
            )
        self.assertEqual(
            manager.mark_validated(tx, 3).state,
            CrossRankLeaseState.ALL_RANKS_VALIDATED,
        )
        self.assertEqual(
            manager.release_global(tx).state,
            CrossRankLeaseState.GLOBAL_RELEASED,
        )

    def test_any_rank_failure_blocks_global_release_until_abort_acks(self):
        manager = self.coordinator()
        tx = "tx-fail"
        manager.prepare(tx, observations(), checkpoint_version="cp-v1")
        for rank in range(4):
            manager.mark_local_leased(tx, rank)
        self.assertEqual(
            manager.fail(tx, "rank-2-validation-failed").state,
            CrossRankLeaseState.FAILED,
        )
        with self.assertRaisesRegex(LeaseRejected, "invalid_transition"):
            manager.release_global(tx)
        manager.acknowledge_abort(tx, 0)
        with self.assertRaisesRegex(LeaseRejected, "slot_already_coordinated"):
            manager.prepare("tx-reuse-before-cleanup", observations(), checkpoint_version="cp-v1")
        for rank in range(1, 4):
            manager.acknowledge_abort(tx, rank)
        self.assertEqual(
            manager.prepare("tx-reuse", observations(), checkpoint_version="cp-v1").state,
            CrossRankLeaseState.PREPARE,
        )

    def test_metadata_mismatch_and_manager_restart_fail_closed(self):
        manager = self.coordinator()
        mixed = observations()
        mixed[2] = observations(version="other")[2]
        with self.assertRaisesRegex(LeaseRejected, "rank_metadata_mismatch"):
            manager.prepare("tx-mixed", mixed, checkpoint_version="cp-v1")
        with self.assertRaisesRegex(LeaseRejected, "manager_epoch_mismatch"):
            self.coordinator(manager_epoch="epoch-2").prepare(
                "tx-old", observations(), checkpoint_version="cp-v1"
            )

    def test_timeout_does_not_allow_global_release(self):
        now = [0.0]
        manager = self.coordinator(timeout_s=1, clock=lambda: now[0])
        manager.prepare("tx-timeout", observations(), checkpoint_version="cp-v1")
        now[0] = 2.0
        self.assertEqual(
            manager.status("tx-timeout").state,
            CrossRankLeaseState.EXPIRED,
        )
        with self.assertRaisesRegex(LeaseRejected, "invalid_transition"):
            manager.release_global("tx-timeout")


if __name__ == "__main__":
    unittest.main()
