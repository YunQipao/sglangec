import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.kv_ec.online_chunk_manager import OnlineChunkManager
from experiments.kv_ec.test_online_chunk import online_fixture


class ModeTests(unittest.TestCase):
    def submit(self, manager):
        blocks = [online_fixture(r) for r in range(2)]
        for block, meta in blocks:
            result = manager.submit(meta, block)
        return result, blocks

    def test_replica_independent_full_bytes_no_xor_and_checked_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = OnlineChunkManager(Path(tmp)/"new", "run", protection_mode="replication")
            with patch("experiments.kv_ec.online_chunk_manager.encode", side_effect=AssertionError("not replication")):
                result, blocks = self.submit(m)
            key = result["snapshot_id"]
            directory = m.output/key
            self.assertFalse((directory/"parity.bin").exists())
            self.assertFalse(list(directory.glob("rank_*.bin")))
            original = blocks[0][0].copy()
            blocks[0][0].fill(0)
            restored = m.load_replica(key, 0, request_id=blocks[0][1]["request_id"])
            self.assertTrue(np.array_equal(original, restored))
            restored.fill(0)
            self.assertTrue(np.array_equal(original, m.load_replica(key, 0, request_id=blocks[0][1]["request_id"])))
            with self.assertRaises(ValueError):
                m.load_replica(key, 0, request_id="wrong")
            manifest = json.loads((directory/"manifest.json").read_text())
            self.assertEqual(manifest["memory_ledger"]["stable_replica_bytes"], 2*original.nbytes)
            m.replicas[key][0][0] ^= 1
            with self.assertRaises(ValueError):
                m.load_replica(key, 0, request_id=blocks[0][1]["request_id"])

    def test_rs_only_waits_for_remote_and_keeps_conflict_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = Mock()
            m = OnlineChunkManager(Path(tmp)/"new", "run", protection_mode="rs-only", global_exporter=exporter)
            with patch("experiments.kv_ec.online_chunk_manager.encode", side_effect=AssertionError("no local XOR")):
                result, blocks = self.submit(m)
            self.assertEqual(result["state"], "LOCAL_SNAPSHOT")
            key = result["snapshot_id"]
            self.assertFalse((m.output/key/"parity.bin").exists())
            exporter.enqueue.assert_called_once()
            self.assertEqual(m.submit(blocks[0][1], blocks[0][0])["state"], "LOCAL_SNAPSHOT")
            path = m.output/key/"global_status.json"
            path.write_text(json.dumps(dict(global_state="GLOBAL_COMMITTED", local_snapshot_id="wrong", remote_outcome="COMMITTED")))
            self.assertEqual(m.status(key)["state"], "LOCAL_SNAPSHOT")
            path.write_text(json.dumps(dict(global_state="GLOBAL_COMMITTED", local_snapshot_id=key, remote_outcome="COMMITTED")))
            self.assertEqual(m.status(key)["state"], "COMMITTED")
            bad = dict(blocks[0][1], token_ids_sha256="wrong")
            self.assertEqual(m.submit(bad, blocks[0][0])["state"], "FAILED")

    def test_modes_fail_closed_before_creating_output_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode, exporter in (("rs-only", None), ("xor-rs", None), ("replication", Mock()), ("bogus", None)):
                with self.assertRaises(ValueError):
                    OnlineChunkManager(Path(tmp)/mode, "run", protection_mode=mode, global_exporter=exporter)
                self.assertFalse((Path(tmp)/mode).exists())
            m = OnlineChunkManager(Path(tmp)/"bounded", "run", protection_mode="replication", max_replica_bytes=1)
            result, _ = self.submit(m)
            self.assertEqual(result["state"], "DROPPED")
            self.assertEqual(m.replica_bytes, 0)

    def test_legacy_default_and_exporter_remain_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, exporter in enumerate((None, Mock())):
                m = OnlineChunkManager(Path(tmp)/str(i), "run", global_exporter=exporter)
                result, _ = self.submit(m)
                self.assertEqual(result["state"], "COMMITTED")
                self.assertEqual(m.protection_mode, "xor-rs" if exporter else "xor")
                self.assertTrue((m.output/result["snapshot_id"]/"parity.bin").is_file())

    def test_actual_parity_dram_and_explicit_release_do_not_silently_evict(self):
        for mode in ("xor","replication"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                m=OnlineChunkManager(Path(tmp)/"new","run",protection_mode=mode)
                result,blocks=self.submit(m)
                key=result["snapshot_id"]
                request=blocks[0][1]["request_id"]
                if mode=="xor":
                    self.assertEqual(m.local_parity_bytes,blocks[0][0].nbytes)
                    self.assertTrue(np.array_equal(m.local_parities[key],np.bitwise_xor(blocks[0][0],blocks[1][0])))
                with self.assertRaises(ValueError):
                    m.release_protection(key,request_id="wrong")
                self.assertEqual(m.release_protection(key,request_id=request)["state"],"RELEASED")
                self.assertEqual(m.replica_bytes+m.local_parity_bytes,0)
                self.assertEqual(m.status(key)["state"],"RELEASED")
                self.assertEqual(m.submit(blocks[0][1],blocks[0][0])["state"],"RELEASED")
                if mode=="replication":
                    with self.assertRaises(ValueError):
                        m.load_replica(key,0,request_id=request)
