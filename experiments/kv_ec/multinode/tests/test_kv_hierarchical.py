from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from experiments.kv_ec.test_debug_capture import rank_fixture
from experiments.kv_ec.rank_snapshot import (
    AUDITED_V1_PROFILE, EMPTY_DIFF_SHA256, assemble, capture_compatibility,
    validate_payload, write_rank_files,
)
from experiments.kv_ec.multinode.kv_hierarchical import (
    MAGIC, _compatible, _decode_node, _load_node, run_receive,
)

SOURCE_A = "c78be46f66b40bac78e01304040576effe9c3eca"
SOURCE_B = "d058fb3a791dd57489c6efe31d594f56e8f68981"
MODEL_SHA256 = "7b9847a1e1530259c3b60505e32f1c8aa99ca20bd26c8de4ac5ede00262cc347"


def make_snapshot(root, *, request, snapshot, layers=tuple(range(36)),
                  model_identifier="/models/Qwen3-8B", model_sha256=MODEL_SHA256,
                  source_commit=SOURCE_A):
    directory = Path(root)
    directory.mkdir()
    for rank in range(4):
        payload, metadata = rank_fixture(rank, 4, layers=layers, count=256, heads=2)
        metadata.update(snapshot_id=snapshot, request_id=request,
                        pool_class="sglang.srt.mem_cache.memory_pool.MHATokenToKVPool",
                        model_identifier=model_identifier, model_sha256=model_sha256,
                        source_commit=source_commit, core_diff_sha256=EMPTY_DIFF_SHA256,
                        request_version=request, checkpoint_version=snapshot)
        write_rank_files(directory, payload, metadata)
    (directory / "manifest.json").write_text(json.dumps({"test": True}))
    return directory


def changed_common(manifest, **changes):
    result = deepcopy(manifest)
    result.update(changes)
    for meta in result["ranks"]:
        meta.update(changes)
    return result


def changed_header(envelope, mutate):
    offset = len(MAGIC) + 8
    length = struct.unpack_from("!Q", envelope, len(MAGIC))[0]
    header = json.loads(envelope[offset:offset + length])
    mutate(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return MAGIC + struct.pack("!Q", len(encoded)) + encoded + envelope[offset + length:]


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
            _compatible(am, bm)
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


class HierarchicalCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = make_snapshot(
            Path(cls.temporary.name) / "a", request="request-A", snapshot="chunk-A",
            model_identifier="/media/sda1/iic/luotao/models/Qwen3-8B")
        cls.loaded = _load_node(cls.directory, "node-a")
        cls.manifest, cls.blocks, _, _, cls.envelope, cls.header, *_ = cls.loaded

    def test_different_absolute_paths_same_model_sha256_are_compatible(self):
        other = changed_common(self.manifest,
                               model_identifier="/media/sda1/iic_data/luotao/models/Qwen3-8B",
                               snapshot_id="chunk-B", request_id="request-B")
        _compatible(self.manifest, other)
        self.assertNotEqual(self.manifest["model_identifier"], other["model_identifier"])

    def test_audited_source_pair_compatible_in_both_directions(self):
        other = changed_common(self.manifest, source_commit=SOURCE_B,
                               model_identifier="/different/install/location")
        _compatible(self.manifest, other)
        _compatible(other, self.manifest)
        self.assertEqual(capture_compatibility(other), AUDITED_V1_PROFILE)
        self.assertEqual(self.header["capture_compatibility"], AUDITED_V1_PROFILE)

    def test_receive_commits_different_paths_and_audited_source_versions(self):
        from experiments.kv_ec.multinode import kv_hierarchical

        local = changed_common(self.manifest, source_commit=SOURCE_B,
                               model_identifier="/media/sda1/iic_data/luotao/models/Qwen3-8B")

        def local_header(header):
            header["node_id"] = "node-b"
            header["rank_metadata"] = local["ranks"]

        envelope = changed_header(self.envelope, local_header)
        loaded = list(self.loaded)
        loaded[0], loaded[4] = local, envelope
        loaded[5] = json.loads(envelope[len(MAGIC) + 8:
                                       len(MAGIC) + 8 + struct.unpack_from("!Q", envelope, len(MAGIC))[0]])

        def receive(host, port, path, *, timeout):
            Path(path).write_bytes(self.envelope)
            return .01

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary) / "received", snapshot=self.directory,
                                   node_id="node-b", transport="tcp", bind_host="127.0.0.1",
                                   data_port=19481, timeout=1, request_metrics=None,
                                   chunk_id="chunk", stripe_id="stripe", experiment_id="test")
            with mock.patch.object(kv_hierarchical, "_load_node", return_value=loaded), \
                    mock.patch.object(kv_hierarchical, "tcp_receive_file", side_effect=receive):
                report = run_receive(args)
            self.assertEqual(report["state"], "COMMITTED")
            self.assertTrue(all(report["verification"].values()))
            self.assertEqual(report["snapshot_compatibility"]["profile"], AUDITED_V1_PROFILE)
            sources = report["snapshot_compatibility"]["sources"]
            self.assertNotEqual(sources[0]["model_identifier"], sources[1]["model_identifier"])
            self.assertEqual({s["source_commit"] for s in sources}, {SOURCE_A, SOURCE_B})
            self.assertTrue((args.output / "hierarchical-result.json").is_file())

    def test_other_model_sha256_is_rejected(self):
        other = changed_common(self.manifest, model_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            _compatible(self.manifest, other)

    def test_invalid_model_sha256_is_rejected(self):
        for value in (None, "", "same-model-sha", "z" * 64):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "model_sha256"):
                _compatible(self.manifest, changed_common(self.manifest, model_sha256=value))

    def test_unknown_source_cannot_use_empty_diff_as_compatibility_proof(self):
        other = changed_common(self.manifest, source_commit="1" * 40)
        other["capture_compatibility"] = AUDITED_V1_PROFILE
        with self.assertRaisesRegex(ValueError, "source_commit.*recapture"):
            _compatible(self.manifest, other)

    def test_unknown_source_requires_exact_same_revision(self):
        left = changed_common(self.manifest, source_commit="1" * 40)
        _compatible(left, deepcopy(left))
        with self.assertRaisesRegex(ValueError, "recapture"):
            _compatible(left, changed_common(left, source_commit="2" * 40))

    def test_dirty_audited_pair_is_not_automatically_compatible(self):
        dirty = hashlib.sha256(b"unreviewed Python changes").hexdigest()
        left = changed_common(self.manifest, core_diff_sha256=dirty)
        right = changed_common(left, source_commit=SOURCE_B)
        with self.assertRaisesRegex(ValueError, "recapture"):
            _compatible(left, right)
        with self.assertRaisesRegex(ValueError, "core_diff_sha256"):
            _compatible(self.manifest, left)

    def test_unknown_rank_format_requires_recapture(self):
        with self.assertRaisesRegex(ValueError, "format.*recapture"):
            _compatible(self.manifest, changed_common(self.manifest, format="kv-ec-rank-bin-v2"))

    def test_rank_order_and_cross_rank_identity_mismatches_are_rejected(self):
        mutations = (
            lambda m: m["ranks"].reverse(),
            lambda m: m["ranks"][1].update(tp_rank=0),
            lambda m: m["ranks"][1].update(model_sha256="0" * 64),
            lambda m: m["ranks"][1].update(source_commit=SOURCE_B),
            lambda m: m["ranks"][1].update(token_start=256),
            lambda m: m["ranks"][1].update(dtype="torch.float16"),
        )
        for mutate in mutations:
            other = deepcopy(self.manifest)
            mutate(other)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                _compatible(self.manifest, other)

    def test_changed_logical_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "token_start"):
            _compatible(self.manifest, changed_common(self.manifest, token_start=256))
        with self.assertRaises(ValueError):
            _compatible(self.manifest, changed_common(self.manifest, layers=list(reversed(range(36)))))

    def test_kv_offsets_lengths_shapes_and_segment_hashes_remain_strict(self):
        original = self.manifest["ranks"][0]
        mutations = (
            lambda m: m["layer_records"][0].update(k_offset=1),
            lambda m: m["layer_records"][0].update(v_offset=0),
            lambda m: m["layer_records"][0].update(k_nbytes=1),
            lambda m: m["layer_records"][0]["packed_shape"].update(k=[256, 1, 256]),
            lambda m: m["layer_records"].reverse(),
            lambda m: m["layer_records"][0].update(k_sha256="0" * 64),
            lambda m: m.update(k_sha256="0" * 64),
            lambda m: m.update(sha256="0" * 64),
            lambda m: m.update(packed_byte_count=m["packed_byte_count"] - 1),
        )
        for mutate in mutations:
            metadata = deepcopy(original)
            mutate(metadata)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                if metadata["layer_records"][0]["packed_shape"]["k"] == [256, 1, 256]:
                    other = deepcopy(self.manifest)
                    other["ranks"][0] = metadata
                    _compatible(self.manifest, other)
                else:
                    validate_payload(self.blocks[0], metadata)

    def test_wire_header_cannot_override_rank_model_layout_or_source(self):
        mutations = (
            lambda h: h.update(model_fingerprint="0" * 64),
            lambda h: h.update(dtype="torch.float16"),
            lambda h: h.update(token_range=[1, 257]),
            lambda h: h.update(layer_range=[1, 37]),
            lambda h: h.update(token_range=[0.0, 256.0]),
            lambda h: h["rank_blocks"][0].update(tp_rank=False),
            lambda h: h["rank_blocks"][0].update(kv_head_range=[2, 4]),
            lambda h: h["rank_blocks"][0].update(layer_records=[]),
            lambda h: h["rank_metadata"][1].update(tp_rank=0),
            lambda h: h["rank_metadata"][1].update(model_sha256="0" * 64),
            lambda h: h["rank_metadata"][1].update(source_commit=SOURCE_B),
            lambda h: h["rank_metadata"][1].update(checksum_deferred=True),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                _decode_node(changed_header(self.envelope, mutate))

    def test_wire_capture_profile_cannot_claim_unreviewed_source_is_compatible(self):
        def mutate(header):
            for meta in header["rank_metadata"]:
                meta["source_commit"] = "1" * 40

        with self.assertRaisesRegex(ValueError, "compatibility profile mismatch"):
            _decode_node(changed_header(self.envelope, mutate))

    def test_existing_v1_envelope_without_profile_is_still_validated(self):
        envelope = changed_header(self.envelope, lambda h: h.pop("capture_compatibility"))
        header, blocks = _decode_node(envelope)
        self.assertEqual(header["rank_metadata"][0]["source_commit"], SOURCE_A)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(blocks, self.blocks)))

    def test_load_and_manifest_assembly_do_not_rewrite_snapshot_evidence(self):
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.directory.iterdir()}
        manifest, _ = assemble(self.directory)
        self.assertEqual(manifest["model_identifier"], self.manifest["model_identifier"])
        self.assertEqual(manifest["layout"], "ascending_layer_then_K_then_V_logical_token_order")
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in self.directory.iterdir()}
        self.assertEqual(before, after)

    def test_model_content_fingerprint_does_not_include_installation_path(self):
        from experiments.kv_ec.debug_capture import source_identity

        with tempfile.TemporaryDirectory() as temporary:
            identities = []
            for name, weights in (("node-a", b"same weights"), ("node-b", b"same weights"),
                                  ("other-model", b"different weights")):
                root = Path(temporary) / name
                root.mkdir()
                (root / "config.json").write_text("{}")
                (root / "model.safetensors").write_bytes(weights)
                with mock.patch.dict("os.environ", {"SGLANG_KV_EC_PREPARED_IDENTITY": ""}):
                    identities.append(source_identity(root))
            self.assertNotEqual(identities[0]["model_identifier"], identities[1]["model_identifier"])
            self.assertEqual(identities[0]["model_sha256"], identities[1]["model_sha256"])
            self.assertNotEqual(identities[0]["model_sha256"], identities[2]["model_sha256"])


if __name__ == "__main__":
    unittest.main()
