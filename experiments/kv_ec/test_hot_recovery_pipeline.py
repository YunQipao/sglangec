import unittest

import numpy as np

from experiments.kv_ec.hot_recovery_pipeline import (
    incremental_parity, recover_cpu, update_parity,
)


class HotRecoveryPipelineTest(unittest.TestCase):
    def test_incremental_parity_and_both_loss_directions(self):
        a = np.arange(10003, dtype=np.uint16).view(np.uint8)
        b = np.arange(10003, 20006, dtype=np.uint16).view(np.uint8)
        parity, stats = incremental_parity(a, b, chunk_bytes=997)
        self.assertGreater(stats["chunk_count"], 1)
        for expected, survivor in ((a, b), (b, a)):
            restored, timing = recover_cpu(parity, survivor, chunk_bytes=733)
            self.assertEqual(restored, expected.tobytes())
            self.assertGreaterEqual(timing["recovery_ms"], 0)

    def test_incremental_dirty_range_update(self):
        a = np.arange(4096, dtype=np.uint8)
        b = np.arange(4096, dtype=np.uint8)[::-1].copy()
        parity, _ = incremental_parity(a, b, chunk_bytes=256)
        changed = a[512:768].copy()
        changed ^= 91
        update_parity(parity, a[512:768], changed, offset=512)
        new_a = a.copy()
        new_a[512:768] = changed
        self.assertTrue(np.array_equal(parity, np.bitwise_xor(new_a, b)))

    def test_invalid_update_rejected(self):
        parity = np.zeros(10, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "invalid_incremental"):
            update_parity(parity, b"aa", b"b", offset=0)


if __name__ == "__main__":
    unittest.main()
