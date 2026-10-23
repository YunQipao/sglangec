import unittest

from experiments.kv_ec.chunk_protection import (
    CommittedChunk,
    CompletePrefillProtection,
)
from experiments.kv_ec.recovery_lease import LeaseRejected


def chunk(index, *, state="COMMITTED", version="req-v1", checkpoint="cp-v1"):
    return CommittedChunk(
        snapshot_id=f"snapshot-chunk-{index}",
        request_id="rid",
        request_version=version,
        checkpoint_version=checkpoint,
        manager_epoch="epoch-1",
        token_start=index * 256,
        token_count=256,
        tp_size=4,
        committed_ranks=frozenset(range(4)),
        state=state,
    )


class ChunkProtectionTests(unittest.TestCase):
    def group(self):
        return CompletePrefillProtection(
            request_id="rid",
            request_version="req-v1",
            prompt_tokens=768,
            chunk_tokens=256,
            manager_epoch="epoch-1",
            checkpoint_version="cp-v1",
        )

    def test_three_committed_chunks_form_complete_prefill(self):
        group = self.group()
        self.assertEqual(group.add(chunk(0)), "PENDING")
        self.assertEqual(group.add(chunk(1)), "PENDING")
        self.assertEqual(group.add(chunk(2)), "COMPLETE")
        status = group.status()
        self.assertEqual(status["state"], "COMPLETE")
        self.assertEqual(status["ranges"], [[0, 256], [256, 512], [512, 768]])
        self.assertTrue(status["all_ranks_committed"])

    def test_metadata_only_or_incomplete_chunk_is_not_protected(self):
        with self.assertRaisesRegex(LeaseRejected, "chunk_not_committed"):
            self.group().add(chunk(0, state="PENDING"))
        group = self.group()
        group.add(chunk(0))
        self.assertEqual(group.status()["state"], "PENDING")

    def test_mixed_version_checkpoint_and_range_are_rejected(self):
        with self.assertRaisesRegex(LeaseRejected, "chunk_metadata_mismatch"):
            self.group().add(chunk(0, version="other"))
        with self.assertRaisesRegex(LeaseRejected, "chunk_metadata_mismatch"):
            self.group().add(chunk(0, checkpoint="other"))
        group = self.group()
        group.add(chunk(0))
        with self.assertRaisesRegex(LeaseRejected, "chunk_gap_or_overlap"):
            group.add(CommittedChunk(
                snapshot_id="snapshot-chunk-2",
                request_id="rid",
                request_version="req-v1",
                checkpoint_version="cp-v1",
                manager_epoch="epoch-1",
                token_start=512,
                token_count=256,
                tp_size=4,
                committed_ranks=frozenset(range(4)),
            ))


if __name__ == "__main__":
    unittest.main()
