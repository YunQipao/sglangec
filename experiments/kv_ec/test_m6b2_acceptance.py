import json
from pathlib import Path
import tempfile
import unittest

from experiments.kv_ec.m6_acceptance import check_m6b2_evidence


class M6B2AcceptanceTests(unittest.TestCase):
    def write_records(self, root):
        identity = {"transaction_id": "tx"}
        for rank in range(4):
            target = rank == 0
            record = dict(
                state="RUNNING", checkpoint_id="tx-chunk0",
                new_lease_created=True, new_lease_released=True,
                all_rank_validate_vote=True, all_rank_resume_vote=True,
                four_rank_vote=True, byte_exact=True,
                fault_injected=target, parity_used=target, kv_written=target,
                recovery=dict(oracle_sha256="independent-checkpoint-hash",
                              fault_injected=target, parity_used=target,
                              kv_written=target, byte_exact=True,
                              four_rank_vote=True, timings={}),
            )
            (root / f"m6b2-rank-{rank}.json").write_text(json.dumps(record))
        return identity

    def test_accepts_real_rank0_restore_with_four_rank_vote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = self.write_records(root)
            records = check_m6b2_evidence(root, identity, {"output_ids": [1, 2]})
            self.assertEqual(len(records), 4)

    def test_rejects_missing_vote_or_byte_exact_proof(self):
        for key in ("four_rank_vote", "byte_exact", "new_lease_released"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                identity = self.write_records(root)
                path = root / "m6b2-rank-2.json"
                record = json.loads(path.read_text())
                record[key] = False
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "rank2_transaction_proof"):
                    check_m6b2_evidence(root, identity, {"output_ids": [1, 2]})

    def test_rejects_wrong_fault_rank_and_missing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = self.write_records(root)
            path = root / "m6b2-rank-1.json"
            record = json.loads(path.read_text())
            record.update(fault_injected=True, parity_used=True, kv_written=True)
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "only_fault_and_write_target"):
                check_m6b2_evidence(root, identity, {"output_ids": [1, 2]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = self.write_records(root)
            with self.assertRaisesRegex(ValueError, "restore_or_decode_proof"):
                check_m6b2_evidence(root, identity, {})


if __name__ == "__main__":
    unittest.main()
