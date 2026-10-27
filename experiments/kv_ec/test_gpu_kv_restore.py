"""CPU tests for GPU KV restore validation and transaction ownership.

These tests intentionally do not claim CUDA writeback. The real writeback path
is guarded by ``torch.cuda.is_available()`` and is covered by the server command
documented in ONLINE_RS_STAGE2.md.
"""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

import torch

from .gpu_kv_restore import (
    BytesCheckpointSource, LocalCheckpointSource, RestoreRejected,
    RestoreTransaction, _validate_segments,
)
from .node_block_export import pack, unpack, xor
from .test_debug_capture import rank_fixture


class GPUKVRestoreValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = []
        cls.nodes = []
        for node in (0, 1):
            rank_blocks, entries = cls._rank_set(node, 0)
            block = pack(node, rank_blocks, entries,
                         allow_fixture=True)
            cls.blocks.append(block)
            cls.nodes.append((rank_blocks, entries))

    @staticmethod
    def _rank_set(node, chunk):
        blocks, entries = [], []
        for rank in (0, 1):
            raw, meta = rank_fixture(rank, 2, layers=range(36), count=256, heads=4)
            meta = dict(meta, snapshot_id=f"node{node}-chunk{chunk}",
                        checkpoint_version=f"node{node}-chunk{chunk}",
                        request_version=f"node{node}-run", manager_epoch=f"node{node}-run",
                        model_version="fixture", lifetime="fixture",
                        completion_dependency="fixture",
                        chunk_id=chunk, token_start=chunk * 256,
                        token_range=[chunk * 256, (chunk + 1) * 256])
            blocks.append(raw.tobytes())
            entries.append(meta)
        return blocks, entries

    def test_both_tp_ranks_and_each_chunk_validate_exactly(self):
        for chunk in (0, 1, 2):
            for node in (0, 1):
                rank_blocks, entries = self._rank_set(node, chunk)
                block = pack(node, rank_blocks, entries,
                             allow_fixture=True)
                meta, payloads = unpack(block, allow_fixture=True)
                self.assertEqual(meta["node_id"], node)
                for rank, raw in enumerate(payloads):
                    _validate_segments(raw, meta["ranks"][rank], token_count=256, tp_size=2)

    def test_xor_rebuilds_either_rank_before_validation(self):
        rank_blocks, entries = self.nodes[0]
        survivor = rank_blocks[1]
        parity = xor(rank_blocks[0], survivor)
        rebuilt0 = bytes(a ^ b for a, b in zip(parity, survivor))
        self.assertEqual(rebuilt0, rank_blocks[0])
        block = pack(0, [rebuilt0, survivor], entries, allow_fixture=True)
        _, payloads = unpack(block, allow_fixture=True)
        _validate_segments(payloads[0], entries[0], token_count=256, tp_size=2)

    def test_corrupt_and_layout_mismatch_fail_closed(self):
        _, payloads = unpack(self.blocks[0], allow_fixture=True)
        bad = bytearray(payloads[0]); bad[0] ^= 1
        with self.assertRaisesRegex(RestoreRejected, "checkpoint_payload_sha256_mismatch"):
            _validate_segments(bytes(bad), self.nodes[0][1][0], token_count=256, tp_size=2)
        rank_meta = self.nodes[0][1][0]
        broken = {**rank_meta, "layer_records": [{
            **rank_meta["layer_records"][0],
            "packed_shape": {"k": [256, 3, 128], "v": [256, 3, 128]},
        }] + rank_meta["layer_records"][1:]}
        with self.assertRaisesRegex(RestoreRejected, "checkpoint_packed_shape_mismatch"):
            _validate_segments(payloads[0], broken, token_count=256, tp_size=2)

    def test_sources_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_node.bin"
            path.write_bytes(self.blocks[0])
            self.assertEqual(LocalCheckpointSource(path).load(), self.blocks[0])
            self.assertEqual(BytesCheckpointSource(self.blocks[0]).load(), self.blocks[0])

    def test_transaction_commit_uses_new_slots_and_abort_releases_them(self):
        req_pool = NS(device="cpu", free_slots=[], req_generation=torch.tensor([0, 7]),
                      req_to_token=torch.zeros((2, 4), dtype=torch.int32))
        req = NS(req_pool_idx=1)
        freed = []
        allocator = NS(device="cpu", free=lambda slots: freed.extend(slots.tolist()))
        tx = RestoreTransaction(allocator, req_pool, req, 1, 7, (4, 5), (8, 9),
                                0, 2, NS(), state="VERIFIED")
        tx.commit()
        self.assertEqual(req_pool.req_to_token[1].tolist()[:2], [8, 9])
        self.assertEqual(tx.state, "COMMITTED")
        self.assertEqual(freed[:2], [4, 5])
        tx2 = RestoreTransaction(allocator, req_pool, req, 1, 7, (8, 9), (10, 11),
                                 2, 2, NS())
        tx2.abort()
        self.assertEqual(freed, [4, 5, 10, 11])

    def test_transaction_rejects_mapping_generation_change(self):
        req_pool = NS(device="cpu", free_slots=[], req_generation=torch.tensor([0, 7]),
                      req_to_token=torch.zeros((2, 4), dtype=torch.int32))
        req = NS(req_pool_idx=1)
        allocator = NS(device="cpu", free=lambda slots: None)
        tx = RestoreTransaction(allocator, req_pool, req, 1, 7, (), (8, 9), 0, 2,
                                NS(), state="VERIFIED")
        req_pool.req_generation[1] = 8
        with self.assertRaisesRegex(RestoreRejected, "request_generation_changed"):
            tx.commit()


if __name__ == "__main__":
    unittest.main()
