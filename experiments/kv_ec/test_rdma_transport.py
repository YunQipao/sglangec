"""Contract tests for the native RDMA path.

The CI/container environment normally has no HCA.  These tests therefore
cover parsing and fail-closed behavior; the real verbs test is opt-in and is
only reported as a test when the caller explicitly enables it on a machine
with an active device.
"""

import os
from pathlib import Path
import unittest

from experiments.kv_ec.multinode.rdma_transport import (
    RDMATransport,
    RDMATransportUnavailable,
)
from experiments.kv_ec.node_block_export import descriptor, sha
from experiments.kv_ec.remote_rs_server import StripeStore
from experiments.kv_ec.test_tp4_physical_rs import tp4_block


class RDMATransportContractTests(unittest.TestCase):
    def test_machine_readable_completion_stats(self):
        stats = RDMATransport._parse_stats(
            "KV_EC_RDMA_STATS bytes=1048576 device_setup_ms=1.250 "
            "memory_registration_ms=2.500 qp_setup_ms=3.000 exchange_ms=4.000 "
            "completion_ms=5.000 total_ms=16.000 qp_num=7 remote_qpn=8 "
            "gid_index=5 port_num=1 completion_status=SUCCESS\n", 0.02
        )
        self.assertEqual(stats["bytes"], 1048576)
        self.assertEqual(stats["qp_num"], 7)
        self.assertEqual(stats["remote_qpn"], 8)
        self.assertEqual(stats["gid_index"], 5)
        self.assertEqual(stats["completion_status"], "SUCCESS")

    def test_missing_verbs_fails_closed_without_tcp_fallback(self):
        if RDMATransport.available():
            self.skipTest("verbs build prerequisites are present; hardware test is separate")
        with self.assertRaises(RDMATransportUnavailable):
            RDMATransport(device="mlx5_0", gid_index=5)

    def test_server_marks_rdma_receiver_failure_without_tcp_commit(self):
        if RDMATransport.available():
            self.skipTest("native prerequisites are present; use the opt-in verbs test")
        store = StripeStore(allow_fixture=True, rdma_port_base=39600, rdma_timeout=0.1)
        blocks = (tp4_block(0), tp4_block(1))
        status = None
        for node, block in enumerate(blocks):
            status, _ = store.dispatch({
                "op": "register_node", "stripe_id": "rdma-failure",
                "node_id": node, "descriptor": descriptor(block, allow_fixture=True),
            }, b"")
        binding = {key: status[key] for key in ("stripe_id", "handle", "manifest_sha256")}
        status, _ = store.dispatch({
            "op": "prepare_rdma", "node_id": 0, "payload_length": len(blocks[0]),
            "payload_sha256": sha(blocks[0]), **binding,
        }, b"")
        for _ in range(40):
            item = store.status(store.stripes["rdma-failure"])["rdma"][0]
            if item["state"] == "FAILED":
                break
            import time
            time.sleep(0.02)
        self.assertEqual(item["state"], "FAILED")
        self.assertEqual(store.stripes["rdma-failure"]["blocks"], {})

    @unittest.skipUnless(
        os.environ.get("KV_EC_RUN_REAL_RDMA") == "1"
        and Path("/dev/infiniband").exists(),
        "NOT VERIFIED: set KV_EC_RUN_REAL_RDMA=1 on a host with an active HCA",
    )
    def test_real_rdma_is_explicitly_opt_in(self):
        # The paired-node command in DUAL_NODE_REMOTE_RS.md performs the actual
        # transfer.  Keeping this test opt-in avoids confusing a local fixture
        # or TCP socket with RDMA evidence.
        transport = RDMATransport(device=os.environ.get("KV_EC_RDMA_DEVICE", "mlx5_0"))
        transport.close()


if __name__ == "__main__":
    unittest.main()
