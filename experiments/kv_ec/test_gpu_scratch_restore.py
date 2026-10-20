"""CPU-only guards for the real-snapshot GPU scratch restore tool."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.kv_ec.gpu_scratch_restore import (
    load_real_snapshot, split_layer_payload, benchmark, one_restore,
    validate_snapshot_sequence,
)
from experiments.kv_ec.rank_snapshot import verify, write_rank_files
from experiments.kv_ec.test_parity_manager import metadata_and_blocks


ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "kv_ec_runs" / "tp4-all-256-20261016T142603"


class ScratchRestoreTests(unittest.TestCase):
    def test_three_chunk_sequence_requires_contiguous_ranges(self):
        base = {
            "request_id": "request",
            "request_id_encoding": "sha256",
            "model_identifier": "model",
            "model_sha256": "model",
            "source_commit": "commit",
            "core_diff_sha256": "diff",
            "packed_byte_count": 9437184,
            "tp_size": 4,
            "layers": list(range(36)),
            "dtype": "torch.bfloat16",
            "provenance": "sglang_request",
        }
        manifests = []
        for index in range(3):
            manifests.append({
                **base,
                "snapshot_id": f"chunk-{index}",
                "token_start": index * 256,
                "token_count": 256,
            })
        result = validate_snapshot_sequence(manifests)
        self.assertEqual(result["token_count"], 768)
        self.assertEqual(result["token_start"], 0)

        manifests[1]["token_start"] = 512
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            validate_snapshot_sequence(manifests)

    def test_control_plane_evidence_is_not_a_gpu_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "rank_0.timing.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "control-plane evidence"):
                load_real_snapshot(path)

    def test_tp2_snapshot_load_and_each_scratch_restore_dispatch(self):
        # Synthetic CPU bytes declared as request format ONLY to exercise the
        # loader; mocked H2D is not evidence of a real GPU restore.
        blocks, metadata = metadata_and_blocks(2, count=128)
        with tempfile.TemporaryDirectory() as directory:
            for meta, block in zip(metadata, blocks):
                meta["provenance"] = "sglang_request"
                write_rank_files(directory, block, meta)
            verify(directory)
            manifest, loaded = load_real_snapshot(directory)
            self.assertEqual(manifest["tp_size"], 2)
            for lost in range(2):
                with patch("experiments.kv_ec.gpu_scratch_restore.restore_rank_to_gpu",
                           return_value=([], 0.0, 0.0)) as restore:
                    result = one_restore(manifest, loaded, lost, __import__("torch").device("cpu"))
                self.assertTrue(result["byte_exact"])
                np.testing.assert_array_equal(restore.call_args.args[0], blocks[lost])
                self.assertEqual(restore.call_args.args[1]["tp_rank"], lost)
            with patch("experiments.kv_ec.gpu_scratch_restore.torch.cuda.is_available", return_value=True), \
                 patch("experiments.kv_ec.gpu_scratch_restore.one_restore",
                       return_value={"byte_exact": True, "xor_ms": 0.0, "cpu_recovery_ms": 0.0,
                                     "h2d_restore_ms": 0.0, "gpu_verification_ms": 0.0,
                                     "restored_bytes": blocks[0].size}) as restore:
                result = benchmark(directory, warmup=1, repeat=1)
            self.assertEqual(result["tp"], 2)
            self.assertEqual([r["lost_rank"] for r in result["ranks"]], [0, 1])
            self.assertEqual([c.args[2] for c in restore.call_args_list], [0, 1, 0, 1])

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
