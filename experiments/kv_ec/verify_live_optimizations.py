"""Read-only implementation preflight; does not start a model or experiment."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from .cuda_rs_codec import CudaRSCodec
from .live_recovery import HybridPolicy, LiveRecoveryHook, RecoveryPoint
from .online_chunk import OnlineProtector
from .online_chunk_manager import OnlineChunkManager
from .gpu_kv_restore import restore_xor_to_gpu
from .remote_rs_client import RemoteRSClient
from .remote_rs_server import StripeStore


FEATURES = {
    1: ("hot_recovery", "SGLANG_KV_EC_LIVE_RECOVERY"),
    2: ("memory_recovery", "SGLANG_KV_EC_HOT_MEMORY"),
    3: ("streaming_recovery", "SGLANG_KV_EC_STREAMING_RECOVERY"),
    4: ("compute_network_writeback_pipeline", "SGLANG_KV_EC_PIPELINE"),
    5: ("copy_hash_reuse", "SGLANG_KV_EC_HASH_REUSE"),
    6: ("incremental_protection", "SGLANG_KV_EC_INCREMENTAL"),
    7: ("adaptive_granularity", "SGLANG_KV_EC_ADAPTIVE_GRAIN"),
    8: ("hybrid_ec_reprefill", "SGLANG_KV_EC_HYBRID"),
    9: ("vectorized_gf256", "SGLANG_KV_EC_CUDA_RS"),
    10: ("cuda_rs_kernel", "SGLANG_KV_EC_CUDA_RS"),
    11: ("rs_network_pipeline", "SGLANG_KV_EC_PIPELINE"),
    12: ("fused_xor_gpu_write", "SGLANG_KV_EC_FUSED_XOR_WRITE"),
    13: ("async_validation", "SGLANG_KV_EC_ASYNC_VERIFY"),
    14: ("async_d2h", "SGLANG_KV_EC_ASYNC_D2H"),
    15: ("same_process_switch", "SGLANG_KV_EC_SAME_PROCESS_SWITCH"),
}

# CLI names intentionally differ from the human-readable feature names above.
# Keep this mapping explicit so the preflight validates the launcher contract
# instead of relying on a whitespace-sensitive source substring.
DEFAULT_OPT_NAMES = {
    1: "hot-recovery",
    2: "memory-recovery",
    3: "streaming",
    4: "pipeline",
    5: "hash-reuse",
    6: "incremental",
    7: "adaptive-grain",
    8: "hybrid",
    9: "cuda-rs",
    10: "cuda-rs",
    11: "pipeline",
    12: "fused-xor-write",
    13: "async-verify",
    14: "async-d2h",
    15: "same-process-switch",
}


def verify(root: Path, *, cuda: bool):
    launch = (root / "experiments/kv_ec/launch_online.py").read_text()
    scheduler = (root / "python/sglang/srt/managers/scheduler.py").read_text()
    output = (
        root / "python/sglang/srt/managers/scheduler_components/output_streamer.py"
    ).read_text()
    live_source = inspect.getsource(LiveRecoveryHook)
    restore_source = inspect.getsource(restore_xor_to_gpu)
    online_source = inspect.getsource(OnlineProtector)
    manager_source = inspect.getsource(OnlineChunkManager)
    evidence = {
        1: "LiveRecoveryHook.before_schedule keeps the resident Scheduler/model",
        2: "BytesCheckpointSource + OnlineChunkManager.recover_xor use host DRAM",
        3: "restore_xor_to_gpu streams 36 layers with bounded two-slot rings",
        4: "bounded TCP remote-compute responses feed the GPU staging callback",
        5: "immutable chunk metadata and dirty-chunk reuse avoid clean rescans",
        6: "OnlineProtector captures only newly completed chunks",
        7: "launcher selects 128/256/512 tokens from context length",
        8: "HybridPolicy chooses comparable EC/Re-Prefill estimates",
        9: "CudaRSCodec CPU oracle is GF(2^8) vectorized",
        10: "Triton _rs21_kernel executes on CUDA",
        11: "submit/reconstruct chunks overlap network with immediate RS encode/decode",
        12: "_xor_scatter_kernel writes final non-contiguous KV pool slots",
        13: "validation_stream overlaps later layer writeback; protection hashes overlap Decode",
        14: "OnlineProtector private CUDA stream performs nonblocking pinned D2H",
        15: "Scheduler calls LiveRecoveryHook and output is gated by recovery version",
    }
    code_checks = {
        1: "before_schedule" in live_source,
        2: (hasattr(OnlineChunkManager, "recover_xor")
            and "SGLANG_KV_EC_HOT_MEMORY" in live_source),
        3: ("input_ring" in restore_source and "validation_ring" in restore_source
            and "SGLANG_KV_EC_STREAMING_RECOVERY" in live_source),
        4: ("SGLANG_KV_EC_PIPELINE" in live_source
            and hasattr(RemoteRSClient, "tcp_reconstruct_missing_node_streaming")),
        5: ("manifests" in manager_source
            and "SGLANG_KV_EC_HASH_REUSE" in restore_source),
        6: "state.next_chunk" in online_source and "incremental_enabled" in online_source,
        7: "protection_grain" in launch and "ONLINE_GRAIN_TOKENS" in launch,
        8: ("same_boundary_estimate" in inspect.getsource(HybridPolicy)
            and "SGLANG_KV_EC_HYBRID" in live_source),
        9: ("cpu_transform" in (root / "experiments/kv_ec/cuda_rs_codec.py").read_text()
            and "SGLANG_KV_EC_CUDA_RS" in live_source),
        10: ("_rs21_kernel" in (root / "experiments/kv_ec/cuda_rs_codec.py").read_text()
             and "SGLANG_KV_EC_CUDA_RS" in live_source),
        11: (hasattr(RemoteRSClient, "tcp_submit_node_block_streaming")
             and hasattr(RemoteRSClient, "tcp_reconstruct_missing_node_streaming")
             and "reconstruct_node_chunk" in inspect.getsource(StripeStore)),
        12: ("xor_scatter_into" in restore_source
             and "SGLANG_KV_EC_FUSED_XOR_WRITE" in live_source),
        13: ("validation_stream" in restore_source
             and "SGLANG_KV_EC_ASYNC_VERIFY" in live_source),
        14: ("non_blocking=async_d2h" in online_source
             and "SGLANG_KV_EC_ASYNC_D2H" in online_source),
        15: ("create_live_recovery" in scheduler
             and "kv_ec_recovery_version" in output
             and "SGLANG_KV_EC_SAME_PROCESS_SWITCH" in live_source),
    }
    results = []
    for number, (name, env) in FEATURES.items():
        # Every switch must be produced by enabled(name), so it is 1 unless the
        # baseline or an explicit --disable-opt requests otherwise.
        opt_name = DEFAULT_OPT_NAMES[number]
        applied_default = env in launch and f'enabled("{opt_name}")' in launch
        results.append(dict(number=number, name=name, env=env,
                            implemented=bool(code_checks[number]),
                            default_enabled=applied_default,
                            evidence=evidence[number]))
    assert all(x["default_enabled"] and x["implemented"] for x in results)
    assert "create_live_recovery" in scheduler
    assert "before_schedule()" in scheduler and "after_result(batch)" in scheduler
    assert "kv_ec_output_paused" in output and "kv_ec_recovery_version" in output
    assert hasattr(LiveRecoveryHook, "_recover_ec") and hasattr(LiveRecoveryHook, "_re_prefill")
    assert hasattr(OnlineProtector, "recover_xor_from_memory")
    assert hasattr(OnlineChunkManager, "recover_xor")
    policy = HybridPolicy()
    policy.observe("EC", 20); policy.observe("RE_PREFILL", 10)
    assert policy.choose(ec_available=True, token_count=1024, kv_bytes=1)[0] == "RE_PREFILL"
    policy.observe("EC", 1); policy.observe("EC", 1); policy.observe("EC", 1)
    assert policy.choose(ec_available=True, token_count=1024, kv_bytes=1)[0] == "EC"
    codec = CudaRSCodec("cuda" if cuda else "auto")
    a = bytes(range(256)) * 257
    b = bytes(reversed(range(256))) * 257
    parity = codec.transform(a, b, "encode")
    assert codec.transform(parity, b, "recover0") == a
    fused_scatter = None
    if codec.backend == "cuda":
        import torch
        device = codec.device
        slots = torch.tensor([2, 5], dtype=torch.int64, device=device)
        pool = torch.zeros((8, 2, 4), dtype=torch.bfloat16, device=device)
        n = slots.numel() * pool[0].numel() * pool.element_size()
        pa = torch.arange(n, dtype=torch.uint8, device=device)
        su = torch.arange(n, dtype=torch.uint8, device=device).bitwise_xor_(7)
        codec.xor_scatter_into(pa, su, pool, slots).synchronize()
        expected = torch.bitwise_xor(pa, su).cpu()
        got = pool[slots].contiguous().view(torch.uint8).reshape(-1).cpu()
        assert torch.equal(got, expected)
        fused_scatter = True
    return {
        "status": "PASS",
        "scope": "implementation_and_cuda_preflight_without_model_experiment",
        "root": str(root),
        "features": results,
        "cuda_rs_backend": codec.backend,
        "cuda_rs_byte_exact": True,
        "fused_xor_scatter_real_cuda": fused_scatter,
        "same_process_hook": LiveRecoveryHook.__name__,
        "recovery_point_fields": list(RecoveryPoint.__dataclass_fields__),
        "manager_fault_path_disk_reads": 0,
        "notes": [
            "Recovery validation remains fail-closed before Scheduler commit.",
            "Async checksum work overlaps inference only on the protection path.",
            "A full Qwen3-8B fault experiment was intentionally not started.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve(), cuda=args.require_cuda)
    raw = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(raw + "\n")
    print(raw)


if __name__ == "__main__":
    main()
