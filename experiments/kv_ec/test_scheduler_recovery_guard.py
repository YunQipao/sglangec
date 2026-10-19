"""CPU-only tests for the real-scheduler mapping guard."""

from types import SimpleNamespace as NS
import os
import unittest
from unittest.mock import patch

import torch

from experiments.kv_ec.scheduler_recovery_guard import (
    RecoveryGuard,
    RecoveryGuardRejected,
    read_scheduler_mapping,
)


def scheduler(tp=4, prefix_disabled=True, overlap_disabled=True, chunk=-1):
    pool = NS(
        free_slots=[1, 2],
        req_generation=torch.tensor([0, 0, 0, 9], dtype=torch.int64),
        req_to_token=torch.tensor([[0] * 8, [0] * 8, [0] * 8, [8, 9, 13, 17, 21, 22, 27, 31]]),
    )
    return NS(
        server_args=NS(
            disable_overlap_schedule=overlap_disabled,
            disable_radix_cache=prefix_disabled,
            chunked_prefill_size=chunk,
            max_running_requests=1,
        ),
        model_worker=NS(ps=NS(tp_rank=0, tp_size=tp, attn_tp_size=tp)),
        req_to_token_pool=pool,
    )


class SchedulerRecoveryGuardTests(unittest.TestCase):
    def test_reads_real_mapping_but_blocks_writeback(self):
        record = read_scheduler_mapping(scheduler(), NS(req_pool_idx=3, rid="rid"),
                                        token_start=0, token_count=8)
        self.assertEqual(record.physical_slots, (8, 9, 13, 17, 21, 22, 27, 31))
        self.assertEqual(record.req_generation, 9)
        self.assertFalse(record.slot_generation_available)
        self.assertFalse(record.writeback_supported)

    def test_unsupported_runtime_modes_fail_closed(self):
        for kwargs in (
            {"prefix_disabled": False},
            {"overlap_disabled": False},
            {"chunk": 128},
        ):
            with self.subTest(kwargs):
                with self.assertRaises(RecoveryGuardRejected):
                    read_scheduler_mapping(
                        scheduler(**kwargs), NS(req_pool_idx=3, rid="rid"),
                        token_start=0, token_count=8,
                    )

    def test_disabled_guard_is_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            result = RecoveryGuard().observe(
                scheduler(), NS(req_pool_idx=3, rid="rid"),
                token_start=0, token_count=8,
            )
        self.assertEqual(result, {"enabled": False, "state": "DISABLED"})

    def test_enabled_guard_reports_read_only_block(self):
        with patch.dict(os.environ, {"SGLANG_KV_EC_RECOVERY_GUARD": "1"}, clear=True):
            result = RecoveryGuard().observe(
                scheduler(), NS(req_pool_idx=3, rid="rid"),
                token_start=0, token_count=8,
            )
        self.assertEqual(result["state"], "READ_ONLY_BLOCKED")
        self.assertFalse(result["writeback_supported"])


if __name__ == "__main__":
    unittest.main()
