import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from benchmark import BLOCK_SIZES
from snapshot import read_snapshot, verify_snapshots, write_snapshot, raw_tensor
from xor_codec import encode, recover


class XorTests(unittest.TestCase):
    def test_required_sizes_all_erasure_positions(self):
        rng = np.random.default_rng(17)
        for tp in (2, 4):
            for size in BLOCK_SIZES:
                with self.subTest(tp=tp, size=size):
                    blocks = [rng.integers(0, 256, size, dtype=np.uint8) for _ in range(tp)]
                    originals = [b.copy() for b in blocks]
                    for b in blocks:
                        b.flags.writeable = False
                    parity = encode(blocks)
                    for lost in range(tp):
                        restored = recover([None if i == lost else b for i, b in enumerate(blocks)], parity)
                        np.testing.assert_array_equal(restored, originals[lost])
                    for b, original in zip(blocks, originals):
                        np.testing.assert_array_equal(b, original)

    def test_all_half_bit_patterns(self):
        # Exhaust all 65536 encodings, including NaN payloads, infinities and -0.
        bits = np.arange(65536, dtype=np.uint16)
        raw = bits.view(np.uint8)
        for dtype in (torch.float16, torch.bfloat16):
            t = torch.from_numpy(raw.copy()).view(dtype)
            np.testing.assert_array_equal(raw_tensor(t), raw)
            blocks = [raw_tensor(t), raw[::-1].copy()]
            p = encode(blocks)
            np.testing.assert_array_equal(recover([None, blocks[1]], p), raw)

    def test_fail_closed(self):
        b = np.zeros(8, dtype=np.uint8)
        for blocks in ([], [b], [b, b[:4]]):
            with self.assertRaises(ValueError):
                encode(blocks)
        for blocks in ([b, b], [None, None]):
            with self.assertRaises(ValueError):
                recover(blocks, b)
        with self.assertRaises(TypeError):
            encode([np.zeros(8, dtype=np.float16), b])
        with self.assertRaises(ValueError):
            encode([b[::2], b[::2]])
        with self.assertRaises(ValueError):
            recover([None, b[:4]], b)


def fixture(rank, tp, dtype):
    raw = torch.from_numpy(np.arange(2 * 3 * 128 * 2, dtype=np.uint8).reshape(3, 2, 128, 2).copy())
    raw.bitwise_xor_(rank * 37)
    k = raw.reshape(-1).view(dtype).reshape(3, 2, 128).transpose(0, 1)
    # Noncontiguous selected CPU tensors exercise packing without float conversion.
    v = k.clone()
    metadata = dict(request_id="synthetic-request", req_pool_idx=1, request_row_generation=2,
                    slot_generation=None, capture_id="fixture", layer_id=0, tp_rank=rank,
                    tp_size=tp, token_range=[0, 2], physical_slots=[5, 8],
                    source_k_stride=list(k.stride()), source_v_stride=list(v.stride()),
                    source_k_shape=list(k.shape), source_v_shape=list(v.shape), process_epoch="synthetic-process",
                    source_device="cpu", pool_class="synthetic-only", provenance="synthetic_fixture",
                    source_commit="fixture", prefix_sharing="unknown-not-exclusive",
                    token_ids_sha256="fixture-not-real-tokens")
    return k, v, metadata


class SnapshotTests(unittest.TestCase):
    def test_roundtrip_and_recovery(self):
        for dtype in (torch.float16, torch.bfloat16):
            for tp in (2, 4):
                with self.subTest(dtype=dtype, tp=tp), tempfile.TemporaryDirectory() as d:
                    paths = []
                    for rank in range(tp):
                        k, v, m = fixture(rank, tp, dtype)
                        paths.append(Path(d) / f"rank{rank}.npz")
                        write_snapshot(paths[-1], k, v, m)
                        payload, meta = read_snapshot(paths[-1])
                        np.testing.assert_array_equal(payload, np.concatenate([raw_tensor(k), raw_tensor(v)]))
                        self.assertEqual(meta["source_k_stride"], list(k.stride()))
                    report = verify_snapshots(paths)
                    self.assertEqual(report["provenance"], "synthetic_fixture")
                    self.assertTrue(all(c["correct"] for c in report["checks"]))
                    with self.assertRaises(ValueError):
                        verify_snapshots([paths[0]] * tp)
                    mismatched = dict(m, capture_id="different-capture")
                    other = Path(d) / "mismatched.npz"
                    write_snapshot(other, k, v, mismatched)
                    with self.assertRaises(ValueError):
                        verify_snapshots(paths[:-1] + [other])
                    with self.assertRaises(FileExistsError):
                        write_snapshot(paths[0], k, v, m)

    def test_metadata_and_corruption_rejected(self):
        k, v, m = fixture(0, 2, torch.float16)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "one.npz"
            with self.assertRaises(ValueError):
                write_snapshot(p, k, v, {})
            write_snapshot(p, k, v, m)
            payload, meta = read_snapshot(p)
            payload[0] ^= 1
            import json
            bad = Path(d) / "bad.npz"
            np.savez(bad, payload=payload, metadata=np.array(json.dumps(meta)))
            with self.assertRaises(ValueError):
                read_snapshot(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
