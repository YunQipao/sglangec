import unittest

from experiments.kv_ec.cross_rank_control import CrossRankControlServer
from experiments.kv_ec.cross_rank_lease import CrossRankLeaseCoordinator
from experiments.kv_ec.recovery_lease import LeaseRejected
from experiments.kv_ec.test_cross_rank_lease import observations


class CrossRankControlTests(unittest.TestCase):
    def setUp(self):
        self.server = CrossRankControlServer(
            "/tmp/kv-ec-control-test.sock",
            CrossRankLeaseCoordinator(manager_epoch="epoch-1"),
        )

    def test_control_dispatches_all_rank_protocol(self):
        tx = "control-tx"
        raw = [item.__dict__ for item in observations()]
        response = self.server._dispatch({
            "op": "prepare", "transaction_id": tx,
            "observations": raw, "checkpoint_version": "cp-v1",
        })
        self.assertEqual(response["status"]["state"], "PREPARE")
        for rank in range(4):
            self.assertEqual(
                self.server._dispatch({
                    "op": "local_leased", "transaction_id": tx, "tp_rank": rank,
                })["status"]["state"],
                "ALL_RANKS_READY" if rank == 3 else "LOCAL_LEASED",
            )
        for rank in range(4):
            self.server._dispatch({
                "op": "cuda_completed", "transaction_id": tx, "tp_rank": rank,
            })
        for rank in range(4):
            result = self.server._dispatch({
                "op": "validated", "transaction_id": tx, "tp_rank": rank,
            })
        self.assertEqual(result["status"]["state"], "ALL_RANKS_VALIDATED")
        self.assertEqual(
            self.server._dispatch({"op": "release_global", "transaction_id": tx})
            ["status"]["state"],
            "GLOBAL_RELEASED",
        )

    def test_duplicate_and_unknown_messages_are_errors(self):
        with self.assertRaises(LeaseRejected):
            self.server._dispatch({"op": "status", "transaction_id": "missing"})
        with self.assertRaises(LeaseRejected):
            self.server._dispatch({"op": "unknown", "transaction_id": "x"})


if __name__ == "__main__":
    unittest.main()
