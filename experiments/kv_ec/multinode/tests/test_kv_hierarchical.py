import json
from pathlib import Path
import tempfile
import unittest

from experiments.kv_ec.test_debug_capture import rank_fixture
from experiments.kv_ec.rank_snapshot import write_rank_files
from experiments.kv_ec.multinode.kv_hierarchical import (
    _compatible, _decode_node, _load_node,
)


def make_snapshot(root, *, request, snapshot, layers=tuple(range(36))):
    directory = Path(root)
    directory.mkdir()
    for rank in range(4):
        payload, metadata = rank_fixture(rank, 4, layers=layers, count=256, heads=2)
        metadata.update(snapshot_id=snapshot, request_id=request,
                        pool_class="sglang.srt.mem_cache.memory_pool.MHATokenToKVPool",
                        model_identifier="/models/Qwen3-8B", model_sha256="same-model-sha",
                        source_commit="test-commit", core_diff_sha256="test-diff",
                        request_version=request, checkpoint_version=snapshot)
        write_rank_files(directory, payload, metadata)
    (directory / "manifest.json").write_text(json.dumps({"test": True}))
    return directory


class HierarchicalSnapshotTests(unittest.TestCase):
    def test_independent_requests_with_same_layout_form_node_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = make_snapshot(root / "a", request="request-A", snapshot="chunk-A")
            b = make_snapshot(root / "b", request="request-B", snapshot="chunk-B")
            am, _, _, checks, envelope, ah, *_ = _load_node(a, "node-a")
            bm, _, _, _, _, bh, *_ = _load_node(b, "node-b")
            self.assertEqual(len(checks), 4)
            self.assertNotEqual(ah["request_id"], bh["request_id"])
            _compatible(am, {**am, "ranks": bh["rank_metadata"]})
            header, recovered = _decode_node(envelope)
            self.assertEqual(header["node_id"], "node-a")
            self.assertEqual(len(recovered), 4)
            self.assertEqual(header["token_range"], [0, 256])
            self.assertEqual(header["layer_range"], [0, 36])
            self.assertTrue(all(item["kv_head_range"] == [i * 2, i * 2 + 2]
                                for i, item in enumerate(header["rank_blocks"])))

    def test_truncated_or_corrupt_node_payload_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = make_snapshot(Path(temporary) / "a", request="request-A", snapshot="chunk-A")
            *_, envelope, _, _, _ = _load_node(snapshot, "node-a")
            with self.assertRaises(ValueError):
                _decode_node(envelope[:-1])
            corrupted = bytearray(envelope)
            corrupted[-1] ^= 1
            with self.assertRaises(ValueError):
                _decode_node(corrupted)

    def test_mismatched_rank_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            a = make_snapshot(Path(temporary) / "a", request="request-A", snapshot="chunk-A")
            am, *_ = _load_node(a, "node-a")
            with self.assertRaisesRegex(ValueError, "layers"):
                _compatible(am, {**am, "layers": [*am["layers"][:-1], 37]})


if __name__ == "__main__":
    unittest.main()
