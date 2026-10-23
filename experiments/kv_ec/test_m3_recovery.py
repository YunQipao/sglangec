import unittest
from dataclasses import replace

from experiments.kv_ec.m3_recovery import (
    ChunkRecoveryProof,
    M3RecoveryPlan,
    recovery_ranges,
)
from experiments.kv_ec.recovery_lease import LeaseRejected


def proof(start, lost_rank=0):
    index = start // 256
    return ChunkRecoveryProof(
        snapshot_id=f"tx-chunk{index}",
        checkpoint_version=f"tx-chunk{index}",
        manager_epoch="epoch",
        token_start=start,
        token_count=256,
        lost_rank=lost_rank,
        committed_ranks=frozenset(range(4)),
        survivor_ranks_used=frozenset(set(range(4)) - {lost_rank}),
        parity_committed_before_fault=True,
        parity_version_valid=True,
        request_generation_unchanged=True,
        slot_generations_unchanged=True,
        layer_count=36,
        kv_writeback_byte_exact=True,
        validation_byte_exact=True,
    )


class M3RecoveryPlanTests(unittest.TestCase):
    def test_supported_ranges_are_contiguous_and_fixed_size(self):
        self.assertEqual(recovery_ranges(256), ((0, 256),))
        self.assertEqual(recovery_ranges(512), ((0, 256), (256, 256)))
        self.assertEqual(recovery_ranges(768), ((0, 256), (256, 256), (512, 256)))
        for value in (0, 255, 1024, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recovery_ranges(value)

    def test_each_rank_uses_exactly_the_other_three_survivors(self):
        for lost_rank in range(4):
            with self.subTest(lost_rank=lost_rank):
                plan = M3RecoveryPlan(transaction_id="tx", recover_tokens=256,
                                      lost_rank=lost_rank, manager_epoch="epoch")
                self.assertEqual(plan.add(proof(0, lost_rank)), "COMPLETE")
                report = plan.authorize_resume()
                self.assertEqual(report["chunks"][0]["survivor_ranks_used"],
                                 sorted(set(range(4)) - {lost_rank}))

    def test_resume_waits_for_all_chunks_and_accepts_out_of_order_proofs(self):
        plan = M3RecoveryPlan(transaction_id="tx", recover_tokens=768,
                              lost_rank=2, manager_epoch="epoch")
        self.assertEqual(plan.add(proof(512, 2)), "PENDING")
        with self.assertRaisesRegex(LeaseRejected, "all_chunks"):
            plan.authorize_resume()
        self.assertEqual(plan.add(proof(0, 2)), "PENDING")
        self.assertEqual(plan.add(proof(256, 2)), "COMPLETE")
        result = plan.authorize_resume()
        self.assertEqual([item["token_start"] for item in result["chunks"]], [0, 256, 512])

    def test_bad_version_generation_or_partial_write_fails_closed(self):
        bad_proofs = (
            replace(proof(0), checkpoint_version="stale"),
            replace(proof(0), manager_epoch="restarted"),
            replace(proof(0), slot_generations_unchanged=False),
            replace(proof(0), parity_committed_before_fault=False),
            replace(proof(0), kv_writeback_byte_exact=False),
            replace(proof(0), validation_byte_exact=False),
        )
        for bad in bad_proofs:
            with self.subTest(bad=bad), self.assertRaises(LeaseRejected):
                plan = M3RecoveryPlan(transaction_id="tx", recover_tokens=256,
                                      lost_rank=0, manager_epoch="epoch")
                plan.add(bad)
            self.assertFalse(plan.complete)
            with self.assertRaisesRegex(LeaseRejected, "failed"):
                plan.add(proof(0))

    def test_duplicate_missing_and_wrong_target_chunks_are_rejected(self):
        plan = M3RecoveryPlan(transaction_id="tx", recover_tokens=512,
                              lost_rank=1, manager_epoch="epoch")
        with self.assertRaisesRegex(LeaseRejected, "lost_rank"):
            plan.add(proof(0, 0))
        self.assertTrue(plan.failed)
        duplicate = M3RecoveryPlan(transaction_id="tx", recover_tokens=512,
                                   lost_rank=1, manager_epoch="epoch")
        duplicate.add(proof(0, 1))
        with self.assertRaisesRegex(LeaseRejected, "duplicate"):
            duplicate.add(proof(0, 1))


if __name__ == "__main__":
    unittest.main()
