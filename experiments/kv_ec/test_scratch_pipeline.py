"""Synthetic CPU fixtures test guards, not real-request/GPU attestation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec.accept_scratch import accept
from experiments.kv_ec.gpu_scratch_restore import (
    benchmark_sequence, load_real_snapshot, validate_snapshot_sequence,
)
from experiments.kv_ec.prepare_scratch import prepare, validate_capture_evidence
from experiments.kv_ec.rank_snapshot import assemble, verify, write_rank_files
from experiments.kv_ec.test_parity_manager import metadata_and_blocks


class ScratchPipelineTests(unittest.TestCase):
    def test_diagnostics_allowed_but_unknown_sidecar_rejected(self):
        blocks, metadata = metadata_and_blocks(2)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            for block, meta in zip(blocks, metadata):
                write_rank_files(path, block, meta)
                for suffix in ("timing", "diagnostic"):
                    (path / f"rank_{meta['tp_rank']}.{suffix}.json").write_text("{}")
            assemble(path)
            (path / "rank_0.unknown.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "sidecars"):
                assemble(path)

    def test_deferred_corrupt_missing_and_failed_payloads_rejected(self):
        for failure in ("deferred", "corrupt", "missing", "worker"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                path = Path(root)
                blocks, metadata = metadata_and_blocks(2)
                if failure == "deferred":
                    metadata[0]["checksum_deferred"] = True
                if failure == "corrupt":
                    blocks[0][0] ^= 1
                for block, meta in zip(blocks, metadata):
                    write_rank_files(path, block, meta)
                if failure == "missing":
                    (path / "rank_1.bin").unlink()
                if failure == "worker":
                    (path / "rank_1.error.json").write_text("{}")
                with self.assertRaises(ValueError):
                    assemble(path)

    def test_persisted_rank_manifest_cannot_bypass_byte_checks(self):
        with tempfile.TemporaryDirectory() as root:
            blocks, metadata = metadata_and_blocks(2)
            for block, meta in zip(blocks, metadata):
                meta["provenance"] = "sglang_request"
                write_rank_files(root, block, meta)
            verify(root)
            path = Path(root) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["ranks"][0]["layer_records"][0]["k_offset"] = 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest mismatch: ranks"):
                load_real_snapshot(root)

    def test_prepare_timeout_failure_and_invalid_counts(self):
        with tempfile.TemporaryDirectory() as root:
            directories = [Path(root) / str(index) for index in range(3)]
            with self.assertRaisesRegex(TimeoutError, "missing files"):
                prepare(directories, "unused", "unused", wait_s=0)
            directories[0].mkdir()
            (directories[0] / "rank_0.error.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "worker capture failed"):
                prepare(directories, "unused", "unused", wait_s=0)
        for warmup, repeat in ((-1, 10), (0, 0)):
            with self.assertRaises(ValueError):
                benchmark_sequence([], warmup=warmup, repeat=repeat)

    def test_three_chunk_preparation_restore_dispatch_and_acceptance(self):
        # The GPU copy alone is mocked; actual disk assembly, checksums, CPU
        # XOR, all 120 erasure recoveries and evidence validation run here.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            request = {"kind": "real_server_artificial_prompt", "request_id": "artificial",
                       "model_identifier": "/synthetic-test-model", "input_token_count": 768,
                       "output_ids": [1], "finish_reason": {"type": "length"},
                       "input_chunks": [
                           {"token_start": index * 256, "token_count": 256,
                            "token_ids_sha256": f"fixture-chunk-{index}"}
                           for index in range(3)
                       ]}
            identity = {"model_identifier": request["model_identifier"],
                        "model_sha256": "fixture", "source_commit": "fixture",
                        "core_diff_sha256": "fixture"}
            initialization = {"identity": identity}
            paths = [root / f"chunk-{index}" for index in range(3)]
            blocks, metadata = metadata_and_blocks(4, layers=tuple(range(36)), count=256)
            for index, path in enumerate(paths):
                for block, original in zip(blocks, metadata):
                    meta = deepcopy(original)
                    meta.update(identity)
                    meta.update(request["input_chunks"][index])
                    meta.update(snapshot_id=path.name, provenance="sglang_request",
                                prompt_token_count=768, request_id_encoding="sha256",
                                request_id=hashlib.sha256(b"artificial").hexdigest())
                    write_rank_files(path, block, meta)
                    for suffix in ("timing", "diagnostic"):
                        (path / f"rank_{meta['tp_rank']}.{suffix}.json").write_text("{}")
            del blocks
            request_path, init_path = root / "request.json", root / "init.json"
            init_path.write_text(json.dumps(initialization))
            bad_request = deepcopy(request)
            bad_request["request_id"] = "wrong-request"
            request_path.write_text(json.dumps(bad_request))
            with self.assertRaisesRegex(ValueError, "request ID mismatch"):
                prepare(paths, request_path, init_path, wait_s=0)
            self.assertTrue(all(not (path / "manifest.json").exists() for path in paths))
            request_path.write_text(json.dumps(request))
            result = prepare(paths, request_path, init_path, wait_s=0)
            self.assertEqual(sum(len(chunk["checks"]) for chunk in result["chunks"]), 12)
            self.assertFalse(result["online_kv_pool_modified"])
            with self.assertRaises(FileExistsError):
                prepare(paths, request_path, init_path, wait_s=0)
            with patch("experiments.kv_ec.gpu_scratch_restore.torch.cuda.is_available",
                       return_value=True), \
                 patch("experiments.kv_ec.gpu_scratch_restore.torch.cuda.empty_cache"), \
                 patch("experiments.kv_ec.gpu_scratch_restore.restore_rank_to_gpu",
                       return_value=([], 1.0, 2.0)) as copy:
                report = benchmark_sequence(paths, device="cuda:0", warmup=0, repeat=10)
            self.assertEqual(copy.call_count, 120)
            self.assertEqual(accept(report, request, initialization)["measurements"], 120)
            for failure in ("byte_exact", "samples", "rank", "checksum", "online"):
                broken = deepcopy(report)
                rank = broken["chunks"][0]["ranks"][0]
                if failure == "byte_exact":
                    rank["byte_exact"] = False
                elif failure == "samples":
                    rank["xor_ms"]["samples_ms"][0] = float("nan")
                elif failure == "rank":
                    broken["chunks"][0]["ranks"].pop()
                elif failure == "checksum":
                    broken["chunks"][0]["rank_sha256"][0] = "modified"
                else:
                    broken["online_kv_pool_modified"] = True
                with self.subTest(failure=failure), self.assertRaises(ValueError):
                    accept(broken, request, initialization)
            manifests = [load_real_snapshot(path)[0] for path in paths]
            for key in ("model_identifier", "request_id", "core_diff_sha256"):
                bad = deepcopy(manifests)
                bad[1][key] = "other"
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_snapshot_sequence(bad)
            bad = deepcopy(manifests)
            bad[2]["snapshot_id"] = bad[0]["snapshot_id"]
            with self.assertRaisesRegex(ValueError, "unique"):
                validate_snapshot_sequence(bad)
            bad = deepcopy(manifests)
            bad[1]["ranks"][0]["req_generation"] += 1
            with self.assertRaisesRegex(ValueError, "rank identity mismatch"):
                validate_snapshot_sequence(bad)
            bad = deepcopy(manifests)
            bad[1]["ranks"][0]["layer_records"][0]["packed_shape"]["k"][1] = 1
            with self.assertRaisesRegex(ValueError, "NHD K/V layout"):
                validate_snapshot_sequence(bad)
            bad_request = deepcopy(request)
            bad_request["input_chunks"][1]["token_ids_sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "token chunk mismatch"):
                validate_capture_evidence(manifests, bad_request, initialization)
            # A real restore failure must raise before a success report exists.
            with patch("experiments.kv_ec.gpu_scratch_restore.torch.cuda.is_available",
                       return_value=True), \
                 patch("experiments.kv_ec.gpu_scratch_restore.restore_rank_to_gpu",
                       side_effect=RuntimeError("copy failed")):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    benchmark_sequence(paths, warmup=0, repeat=10)


if __name__ == "__main__":
    unittest.main()
