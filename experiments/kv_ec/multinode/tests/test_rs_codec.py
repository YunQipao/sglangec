import unittest

import numpy as np

from experiments.kv_ec.multinode.rs_codec import encode, recover


class RSCodecTests(unittest.TestCase):
    def test_all_three_block_erasures_with_unequal_lengths(self):
        rng = np.random.default_rng(42)
        raw0 = rng.integers(0, 256, 1003, dtype=np.uint8)
        raw1 = rng.integers(0, 256, 819, dtype=np.uint8)
        width = max(len(raw0), len(raw1))
        d0 = np.pad(raw0, (0, width - len(raw0)))
        d1 = np.pad(raw1, (0, width - len(raw1)))
        _, _, parity = encode(d0, d1)
        got0, _ = recover(data0=None, data1=d1, parity=parity,
                          lengths=(len(raw0), len(raw1)))
        _, got1 = recover(data0=d0, data1=None, parity=parity,
                          lengths=(len(raw0), len(raw1)))
        _, _ = recover(data0=d0, data1=d1, parity=None,
                       lengths=(len(raw0), len(raw1)))
        self.assertEqual(got0.tobytes(), raw0.tobytes())
        self.assertEqual(got1.tobytes(), raw1.tobytes())

    def test_bad_padding_in_surviving_or_recovered_block_is_rejected(self):
        d0 = np.array([1, 2, 3, 0], dtype=np.uint8)
        d1 = np.array([4, 5, 6, 7], dtype=np.uint8)
        _, _, parity = encode(d0, d1)
        parity[3] ^= 1
        with self.assertRaisesRegex(ValueError, "padding"):
            recover(data0=None, data1=d1, parity=parity,
                    lengths=(3, 4))

    def test_requires_exactly_one_erasure_and_valid_size(self):
        d = np.ones(8, dtype=np.uint8)
        _, _, p = encode(d, d)
        with self.assertRaises(ValueError):
            recover(data0=None, data1=None, parity=p, lengths=(8, 8))
        with self.assertRaises(ValueError):
            recover(data0=d, data1=None, parity=p[:4], lengths=(8, 8))


if __name__ == "__main__":
    unittest.main()
