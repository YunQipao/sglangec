"""Real CUDA KV-pool acceptance without loading model weights or serving."""

import os
from types import SimpleNamespace as NS
import unittest

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

from .cuda_rs_codec import CudaRSCodec
from .gpu_kv_restore import BytesCheckpointSource, restore_xor_to_gpu
from .node_block_export import pack, sha, xor
from .test_debug_capture import rank_fixture


def _checkpoint(count=128, layer_num=36):
    blocks, ranks = [], []
    for rank in (0, 1):
        raw, meta = rank_fixture(rank, 2, layers=range(layer_num), count=count, heads=4)
        meta.update(
            provenance="sglang_request",
            snapshot_id="live-fused-restore",
            checkpoint_version="live-fused-restore",
            request_version="live-fused-run",
            manager_epoch="live-fused-run",
            model_version="live-fused-model",
            lifetime="non_overlap_before_result_free_independent_gpu_staging",
            completion_dependency="test_real_cuda_pool",
            token_start=0,
        )
        blocks.append(raw.tobytes()); ranks.append(meta)
    return pack(0, blocks, ranks), blocks


def test_fused_xor_restores_real_mha_pool_and_atomically_switches_mapping(
    rank, layer_num
):
    if not torch.cuda.is_available():
        return
    device = "cuda:0"
    # This acceptance test explicitly covers the fail-closed precommit audit.
    # Production defaults to the separately tested postcommit asynchronous audit.
    os.environ["SGLANG_KV_EC_DEFER_FULL_AUDIT"] = "0"
    count, pool_size = 128, 320
    pool = MHATokenToKVPool(
        size=pool_size, page_size=1, dtype=torch.bfloat16,
        head_num=4, head_dim=128, layer_num=layer_num, device=device,
        enable_memory_saver=False, enable_alt_stream=False,
        kv_cache_layout="nhd",
    )
    allocator = TokenToKVPoolAllocator(
        pool_size, torch.bfloat16, device, pool, need_sort=False
    )
    req_pool = ReqToTokenPool(1, count, device, enable_memory_saver=False)
    req = NS(
        req_pool_idx=None, kv_committed_len=count,
        kv=NS(kv_allocated_len=count), cache_protected_len=0,
        prefix_indices=torch.empty(0, dtype=torch.int64),
    )
    req_pool.alloc([req])
    old = allocator.alloc(count)
    req_pool.req_to_token[req.req_pool_idx, :count] = old.to(torch.int32)
    hf = NS(
        model_type="qwen3", num_key_value_heads=8, head_dim=128,
        hidden_size=2048 if layer_num == 28 else 4096,
        num_attention_heads=16 if layer_num == 28 else 32,
        num_hidden_layers=layer_num,
    )
    scheduler = NS(
        server_args=NS(disable_radix_cache=True), enable_overlap=False,
        req_to_token_pool=req_pool, token_to_kv_pool_allocator=allocator,
        model_worker=NS(
            ps=NS(tp_rank=rank, tp_size=2),
            model_runner=NS(
                token_to_kv_pool=pool,
                model_config=NS(hf_config=hf),
            ),
        ),
    )
    checkpoint, blocks = _checkpoint(count, layer_num)
    parity = xor(*blocks)
    codec = CudaRSCodec("cuda", device=device)
    tx = restore_xor_to_gpu(
        scheduler, req, BytesCheckpointSource(checkpoint),
        parity=parity, survivor=blocks[1 - rank], parity_sha256=sha(parity),
        codec=codec,
    )
    assert tx.state == "VERIFIED"
    assert len(tx.stats.layers_verified) == layer_num
    assert tx.stats.fused_xor_scatter is True
    assert tx.stats.streaming_layer_writeback is True
    assert tx.stats.async_validation_stream is True
    assert tx.stats.full_reconstructed_copy_bytes == 0
    assert tx.stats.pipeline_depth == 2
    assert set(tx.old_slots).isdisjoint(tx.new_slots)
    tx.commit()
    assert tx.state == "COMMITTED" and tx.finalized is True
    assert tuple(req_pool.req_to_token[req.req_pool_idx, :count].cpu().tolist()) == tx.new_slots
    assert set(tx.old_slots).issubset(set(allocator.free_pages.cpu().tolist()))
    audit = tx.post_commit_audit_factory()
    assert audit.close(timeout=30) is True
    assert audit.layers_verified == layer_num
    del pool, allocator, req_pool, scheduler, tx
    torch.cuda.empty_cache()


class LiveFusedRestoreTests(unittest.TestCase):
    def test_rank_and_layer_variants(self):
        for layer_num in (28, 36):
            for rank in (0, 1):
                with self.subTest(layer_num=layer_num, rank=rank):
                    test_fused_xor_restores_real_mha_pool_and_atomically_switches_mapping(
                        rank, layer_num
                    )


if __name__ == "__main__":
    unittest.main()
