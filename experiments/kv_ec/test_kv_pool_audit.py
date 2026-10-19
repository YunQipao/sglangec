"""CPU-only tests for fail-closed KV lifecycle and chunk planning helpers."""

from types import SimpleNamespace as NS
import unittest
from dataclasses import replace

import torch

from experiments.kv_ec.kv_pool_audit import (
    KVPoolAuditRejected,
    inspect_request_mapping,
    make_chunk_plan,
    reject_writeback,
    validate_chunk_group,
)


class KVPoolAuditTests(unittest.TestCase):
    def _pool(self):
        return NS(
            size=32,
            free_slots=[1, 3],
            req_generation=torch.tensor([0, 0, 7, 0], dtype=torch.int64),
            req_to_token=torch.tensor(
                [[0, 0, 0], [0, 0, 0], [8, 11, 15], [0, 0, 0]],
                dtype=torch.int32,
            ),
        )

    def test_three_chunks_are_contiguous_and_same_request_version(self):
        chunks = make_chunk_plan("snap", "rid", prompt_tokens=768)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([(c.token_start, c.token_end) for c in chunks],
                         [(0, 256), (256, 512), (512, 768)])
        report = validate_chunk_group(chunks, expected_prompt_tokens=768)
        self.assertEqual(report["chunk_count"], 3)
        self.assertEqual(report["token_count"], 768)

    def test_chunk_group_rejects_mixed_version_and_gap(self):
        chunks = make_chunk_plan("snap", "rid", prompt_tokens=768)
        mixed = list(chunks)
        mixed[1] = replace(chunks[1], request_version="other")
        with self.assertRaises(KVPoolAuditRejected):
            validate_chunk_group(mixed, expected_prompt_tokens=768)
        gap = list(chunks)
        gap[1] = replace(chunks[1], token_start=300, token_end=556)
        with self.assertRaises(KVPoolAuditRejected):
            validate_chunk_group(gap, expected_prompt_tokens=768)

    def test_mapping_is_read_only_and_records_row_generation(self):
        result = inspect_request_mapping(self._pool(), NS(req_pool_idx=2),
                                         token_start=0, token_count=3)
        self.assertEqual(result["physical_slots"], [8, 11, 15])
        self.assertEqual(result["req_generation"], 7)
        self.assertFalse(result["slot_generation_available"])
        self.assertFalse(result["writeback_supported"])

    def test_mapping_rejects_free_row_duplicate_and_stale_generation(self):
        pool = self._pool()
        pool.free_slots.append(2)
        with self.assertRaises(KVPoolAuditRejected):
            inspect_request_mapping(pool, NS(req_pool_idx=2), token_start=0, token_count=3)
        pool = self._pool()
        pool.req_to_token[2, 1] = 8
        with self.assertRaises(KVPoolAuditRejected):
            inspect_request_mapping(pool, NS(req_pool_idx=2), token_start=0, token_count=3)
        with self.assertRaises(KVPoolAuditRejected):
            inspect_request_mapping(self._pool(), NS(req_pool_idx=2),
                                    token_start=0, token_count=3, expected_generation=8)

    def test_writeback_is_explicitly_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "slot generation"):
            reject_writeback()


if __name__ == "__main__":
    unittest.main()
