"""CPU-only guards for the real-snapshot GPU scratch restore tool."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.kv_ec.gpu_scratch_restore import (
    load_real_snapshot, split_layer_payload, benchmark,
)


ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "kv_ec_runs" / "tp4-all-256-20261016T142603"


class ScratchRestoreTests(unittest.TestCase):
    @unittest.skipUnless((REAL / "manifest.json").is_file(), "real evidence directory unavailable")
    def test_real_manifest_and_all_rank_payloads_validate(self):
        manifest, blocks = load_real_snapshot(REAL)
        self.assertEqual(manifest["tp_size"], 4)
        self.assertEqual(manifest["layers"], list(range(36)))
        self.assertEqual(manifest["token_count"], 256)
        self.assertTrue(all(block.dtype == np.uint8 for block in blocks))
        self.assertTrue(all(block.size == 9437184 for block in blocks))
        k, v = split_layer_payload(blocks[0], manifest["ranks"][0]["layer_records"][0])
        self.assertEqual(k.size, 131072)
        self.assertEqual(v.size, 131072)

    @unittest.skipUnless((REAL / "manifest.json").is_file(), "real evidence directory unavailable")
    def test_gpu_is_required_and_never_falls_back_to_cpu(self):
        if __import__("torch").cuda.is_available():
            self.skipTest("GPU environment; run the real restore command instead")
        with self.assertRaisesRegex(RuntimeError, "CUDA is not accessible"):
            benchmark(REAL, warmup=0, repeat=1)

    def test_split_rejects_truncated_payload(self):
        record = {"k_offset": 0, "k_nbytes": 4, "v_offset": 4, "v_nbytes": 4}
        with self.assertRaises(ValueError):
            split_layer_payload(np.zeros(7, dtype=np.uint8), record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
