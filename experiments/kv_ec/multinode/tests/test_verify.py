import unittest

import numpy as np

from experiments.kv_ec.multinode.rs_codec import CODEC, CODEC_VERSION, encode
from experiments.kv_ec.multinode.verify import verify_stripe, verify_synthetic


def fixture():
    d0 = np.array([1, 2, 3, 0], dtype=np.uint8)
    d1 = np.array([4, 5, 6, 7], dtype=np.uint8)
    _, _, parity = encode(d0, d1)
    blocks = {"D0": d0, "D1": d1, "P": parity}
    import hashlib
    metadata = {}
    for role, block in blocks.items():
        metadata[role] = {"experiment_id": "e", "stripe_id": "s", "chunk_id": "c",
                          "source_node_id": "node-a" if role == "D0" else "node-b",
                          "model_fingerprint": "synthetic-seed-1",
                          "payload_length": 3 if role == "D0" else 4,
                          "padded_length": 4, "padding_length": 1 if role == "D0" else 0,
                          "codec": CODEC, "codec_version": CODEC_VERSION,
                          "rs_parameters": {"k": 2, "m": 1, "field": "GF(2^8)"},
                          "checksum": hashlib.sha256(block.tobytes()).hexdigest(),
                          "checkpoint_version": "v1", "protection_state": "COMMITTED",
                          "block_role": role}
    return metadata, blocks


class StripeVerificationTests(unittest.TestCase):
    def test_synthetic_verification_command_path_supports_unequal_lengths(self):
        self.assertEqual(verify_synthetic(4097, 2049)["state"], "PASS")

    def test_valid_complete_stripe_passes(self):
        metadata, blocks = fixture()
        self.assertEqual(verify_stripe(metadata, blocks)["state"], "PASS")

    def test_checksum_state_and_identity_mismatches_reject(self):
        for mutation in (
            lambda m: m["D1"].update(checksum="bad"),
            lambda m: m["P"].update(protection_state="PENDING"),
            lambda m: m["P"].update(stripe_id="another"),
        ):
            metadata, blocks = fixture()
            mutation(metadata)
            with self.assertRaises(ValueError):
                verify_stripe(metadata, blocks)


if __name__ == "__main__":
    unittest.main()
