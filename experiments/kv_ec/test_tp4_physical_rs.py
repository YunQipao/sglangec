"""CPU fixture coverage for the TP4 node-block and rendezvous protocol."""

import unittest

from experiments.kv_ec.node_block_export import descriptor, pack, unpack, xor
from experiments.kv_ec.remote_rs_server import StripeStore
from experiments.kv_ec.rs_stripe_builder import build
from experiments.kv_ec.test_debug_capture import rank_fixture


def tp4_block(node):
    blocks, metadata = [], []
    for rank in range(4):
        payload, meta = rank_fixture(rank, 4, layers=(0,), count=4, heads=2)
        meta.update(snapshot_id=f"snapshot-{node}", request_id=f"request-{node}",
                   request_version=f"version-{node}", checkpoint_version=f"checkpoint-{node}",
                   manager_epoch=f"manager-{node}")
        blocks.append(payload.tobytes())
        metadata.append(meta)
    return pack(node, blocks, metadata, allow_fixture=True)


class TP4PhysicalRSFixtureTests(unittest.TestCase):
    def test_complete_four_rank_block_and_local_xor(self):
        block = tp4_block(0)
        metadata, ranks = unpack(block, allow_fixture=True)
        self.assertEqual(metadata["node_id"], 0)
        self.assertEqual(len(ranks), 4)
        parity = xor(*ranks)
        for lost in range(4):
            restored = bytes(len(parity))
            from experiments.kv_ec.rs_codec import xor as xor_pair
            for rank, payload in enumerate(ranks):
                if rank != lost:
                    restored = xor_pair(restored, payload)
            self.assertEqual(xor_pair(parity, restored), ranks[lost])

    def test_two_independent_tp4_descriptors_register_then_commit(self):
        blocks = [tp4_block(0), tp4_block(1)]
        manifest = build("tp4-fixture", blocks, allow_fixture=True)
        self.assertEqual(manifest["nodes"][0]["metadata"]["ranks"][0]["tp_size"], 4)
        store = StripeStore(allow_fixture=True)
        status, _ = store.dispatch({"op": "register_node", "stripe_id": "tp4-fixture",
                                    "node_id": 0, "descriptor": descriptor(blocks[0], allow_fixture=True)}, b"")
        self.assertEqual(status["state"], "REGISTERING")
        status, _ = store.dispatch({"op": "register_node", "stripe_id": "tp4-fixture",
                                    "node_id": 1, "descriptor": descriptor(blocks[1], allow_fixture=True)}, b"")
        self.assertEqual(status["state"], "PENDING")
        key = {k: status[k] for k in ("stripe_id", "handle", "manifest_sha256")}
        status, _ = store.dispatch({"op": "submit_node_block", "node_id": 0, **key}, blocks[0])
        self.assertEqual(status["state"], "PENDING")
        status, _ = store.dispatch({"op": "submit_node_block", "node_id": 1, **key}, blocks[1])
        self.assertEqual(status["state"], "COMMITTED")
        self.assertEqual(store.stripes["tp4-fixture"]["blocks"], {})


if __name__ == "__main__":
    unittest.main()
