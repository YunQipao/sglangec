"""Fast contract tests; no model process or experiment trial is started."""

from types import SimpleNamespace

import torch

from .cuda_rs_codec import CudaRSCodec, cpu_transform
from .live_recovery import HybridPolicy, LiveRecoveryHook, RecoveryPoint
from .gpu_kv_restore import RestoreRejected
from .online_chunk import completed_chunks


def test_recovery_clock_distinguishes_pending_delivery_from_new_token(caplog, monkeypatch):
    from sglang.srt.managers.scheduler_components.output_streamer import (
        _record_kv_ec_client_delivery,
    )

    req = SimpleNamespace(
        rid="clock-rid", kv_ec_fault_monotonic_ns=1_000_000_000,
        kv_ec_recovery_version=3, kv_ec_active_version=3,
        kv_ec_fault_output_count=4, kv_ec_first_delivery_logged=False,
        kv_ec_first_new_token_logged=False,
        kv_ec_response_complete_logged=False,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.output_streamer.time.monotonic_ns",
        lambda: 1_250_000_000,
    )
    with caplog.at_level("INFO"):
        _record_kv_ec_client_delivery(
            req, [1, 2, 3, 4], previous_send_offset=3, finished=False
        )
        assert req.kv_ec_first_delivery_logged is True
        assert req.kv_ec_first_new_token_logged is False
        _record_kv_ec_client_delivery(
            req, [1, 2, 3, 4, 5], previous_send_offset=4, finished=True
        )
    assert req.kv_ec_first_new_token_logged is True
    assert req.kv_ec_response_complete_logged is True
    messages = "\n".join(record.message for record in caplog.records)
    assert messages.count('"kind": "fault_to_first_delivery"') == 1
    assert messages.count('"kind": "fault_to_first_new_token"') == 1
    assert messages.count('"kind": "fault_to_response_complete"') == 1
    assert '"elapsed_ms": 250.0' in messages


def test_recovery_clock_rejects_stale_version(caplog):
    from sglang.srt.managers.scheduler_components.output_streamer import (
        _record_kv_ec_client_delivery,
    )

    req = SimpleNamespace(
        rid="stale-rid", kv_ec_fault_monotonic_ns=1,
        kv_ec_recovery_version=2, kv_ec_active_version=1,
    )
    with caplog.at_level("INFO"):
        _record_kv_ec_client_delivery(
            req, [1], previous_send_offset=0, finished=True
        )
    assert not caplog.records


def test_rs21_cpu_oracle_roundtrip_all_tails():
    for n in (1, 7, 8, 255, 256, 1003):
        a = bytes((i * 7 + 3) % 256 for i in range(n))
        b = bytes((i * 11 + 5) % 256 for i in range(n))
        parity = cpu_transform(a, b, "encode")
        assert cpu_transform(parity, b, "recover0") == a
        assert cpu_transform(parity, a, "recover1") == b


def test_cuda_rs_kernel_roundtrip_when_available():
    if not torch.cuda.is_available():
        return
    codec = CudaRSCodec("cuda")
    a = bytes((i * 7 + 3) % 256 for i in range(131071))
    b = bytes((i * 11 + 5) % 256 for i in range(131071))
    parity = codec.transform(a, b, "encode")
    assert codec.transform(parity, b, "recover0") == a
    assert codec.transform(parity, a, "recover1") == b
    assert codec.last_stats.backend == "triton-cuda"
    assert codec.last_stats.fused_transform_write is True


def test_fused_xor_scatter_writes_real_noncontiguous_pool_layout():
    if not torch.cuda.is_available():
        return
    codec = CudaRSCodec("cuda")
    device = torch.device("cuda:0")
    slots = torch.tensor([2, 5, 9], dtype=torch.int64, device=device)
    pool = torch.zeros((12, 2, 4), dtype=torch.bfloat16, device=device)
    n = slots.numel() * pool[0].numel() * pool.element_size()
    parity_raw = bytes((i * 3 + 1) % 256 for i in range(n))
    survivor_raw = bytes((i * 5 + 7) % 256 for i in range(n))
    parity = torch.tensor(list(parity_raw), dtype=torch.uint8, device=device)
    survivor = torch.tensor(list(survivor_raw), dtype=torch.uint8, device=device)
    done = codec.xor_scatter_into(parity, survivor, pool, slots)
    done.synchronize()
    got = pool[slots].contiguous().view(torch.uint8).cpu().numpy().tobytes()
    assert got == bytes(a ^ b for a, b in zip(parity_raw, survivor_raw))
    untouched = [i for i in range(pool.shape[0]) if i not in slots.cpu().tolist()]
    assert not pool[untouched].view(torch.uint8).any().item()


def test_recovery_point_uses_delivered_boundary_not_kv_length():
    mapping = torch.tensor([[0] * 8, [10, 11, 12, 13, 14, 15, 16, 17]])
    rp = SimpleNamespace(req_to_token=mapping, req_generation=[0, 9])
    scheduler = SimpleNamespace(
        req_to_token_pool=rp,
        model_worker=SimpleNamespace(ps=SimpleNamespace(tp_rank=0, tp_size=2)),
    )
    req = SimpleNamespace(
        rid="r", req_pool_idx=1, kv_committed_len=5,
        send_token_offset=2, output_ids=[101, 102, 103, 104],
    )
    point = RecoveryPoint.capture(scheduler, req, 7)
    assert point.delivered_tokens == 2
    assert point.computed_unsent_tokens == (103, 104)
    assert point.kv_token_boundary == 5
    assert point.kv_slots == (10, 11, 12, 13, 14)
    assert point.recovery_version == 7


def test_post_commit_validation_accepts_only_the_voted_standby_mapping():
    mapping = torch.tensor([[0] * 8, [10, 11, 12, 13, 14, 15, 16, 17]])
    rp = SimpleNamespace(req_to_token=mapping, req_generation=[0, 9])
    scheduler = SimpleNamespace(
        req_to_token_pool=rp,
        model_worker=SimpleNamespace(ps=SimpleNamespace(tp_rank=0, tp_size=2)),
    )
    req = SimpleNamespace(
        rid="r", req_pool_idx=1, kv_committed_len=5,
        send_token_offset=2, output_ids=[101, 102, 103, 104],
        kv_ec_recovery_version=7, kv_ec_output_paused=True,
    )
    point = RecoveryPoint.capture(scheduler, req, 7)
    mapping[1, :5] = torch.tensor([20, 21, 22, 23, 24])
    hook = LiveRecoveryHook.__new__(LiveRecoveryHook); hook.scheduler = scheduler
    hook._validate_point(req, point, expected_slots=(20, 21, 22, 23, 24))
    with __import__("pytest").raises(RestoreRejected):
        hook._validate_point(req, point, expected_slots=(30, 31, 32, 33, 34))


def test_hybrid_selects_the_measured_faster_same_boundary_path():
    policy = HybridPolicy()
    policy.observe("EC", 20)
    policy.observe("RE_PREFILL", 10)
    assert policy.choose(ec_available=True, token_count=1024, kv_bytes=1)[0] == "RE_PREFILL"
    policy.observe("EC", 1)
    policy.observe("EC", 1)
    policy.observe("EC", 1)
    assert policy.choose(ec_available=True, token_count=1024, kv_bytes=1)[0] == "EC"


def test_live_path_switches_are_real_not_launcher_only_labels():
    hook = LiveRecoveryHook.__new__(LiveRecoveryHook)
    hook.policy = HybridPolicy()
    hook.source = object()
    point = SimpleNamespace(kv_token_boundary=1024)
    hook.same_process_switch = False; hook.hybrid_enabled = True
    assert hook._choose_path(point) == ("RE_PREFILL", "same_process_switch_disabled")
    hook.same_process_switch = True; hook.hybrid_enabled = False
    assert hook._choose_path(point) == ("EC", "hybrid_disabled_ec")
    hook.source = None
    assert hook._choose_path(point) == ("RE_PREFILL", "hybrid_disabled_no_ec")


def test_reprefill_baseline_does_not_require_an_online_checkpoint():
    hook = LiveRecoveryHook.__new__(LiveRecoveryHook)
    hook.same_process_switch = False
    hook.hybrid_enabled = True
    hook.source = None
    hook.policy = HybridPolicy()
    point = SimpleNamespace(kv_token_boundary=4096)
    assert hook._choose_path(point) == (
        "RE_PREFILL", "same_process_switch_disabled"
    )


def test_calibrated_hybrid_can_choose_reprefill_before_ec_materialization(monkeypatch):
    hf = SimpleNamespace(
        num_hidden_layers=28, num_key_value_heads=4, head_dim=128,
    )
    hook = LiveRecoveryHook.__new__(LiveRecoveryHook)
    hook.scheduler = SimpleNamespace(
        _kv_ec_online=object(),
        model_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=hf),
            ),
        ),
    )
    hook.same_process_switch = True
    hook.hybrid_enabled = True
    hook.policy = HybridPolicy()
    monkeypatch.setenv("SGLANG_KV_EC_PREFILL_TOKENS_PER_S", "5000")
    monkeypatch.setenv("SGLANG_KV_EC_RESTORE_BYTES_PER_S", "10000000")
    point = SimpleNamespace(kv_token_boundary=4096)
    assert hook._choose_before_materializing_ec(point) == (
        "RE_PREFILL", "same_boundary_estimate"
    )


def test_uncalibrated_hybrid_does_not_guess_before_materialization(monkeypatch):
    hook = LiveRecoveryHook.__new__(LiveRecoveryHook)
    hook.scheduler = SimpleNamespace(_kv_ec_online=object())
    hook.same_process_switch = True
    hook.hybrid_enabled = True
    hook.policy = HybridPolicy()
    monkeypatch.delenv("SGLANG_KV_EC_PREFILL_TOKENS_PER_S", raising=False)
    monkeypatch.delenv("SGLANG_KV_EC_RESTORE_BYTES_PER_S", raising=False)
    assert hook._choose_before_materializing_ec(
        SimpleNamespace(kv_token_boundary=4096)
    ) is None


def test_reprefill_clears_stale_chunked_request_cursor(monkeypatch):
    req = SimpleNamespace(
        kv_ec_output_paused=True, kv_ec_forward_blocked=True,
    )
    batch = SimpleNamespace(
        reqs=[req], filter_batch=lambda keep_indices: None,
    )
    scheduler = SimpleNamespace(
        running_batch=None, last_batch=batch, chunked_req=req,
        server_args=object(), req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(), tree_cache=object(),
        hisparse_coordinator=object(),
        _add_request_to_queue=lambda request, is_retracted: None,
    )
    monkeypatch.setattr(
        "sglang.srt.managers.schedule_batch.retract_all",
        lambda **kwargs: None,
    )
    hook = LiveRecoveryHook.__new__(LiveRecoveryHook)
    hook.scheduler = scheduler
    hook._event = lambda *args, **kwargs: None
    point = SimpleNamespace(recovery_version=1)
    hook._re_prefill(req, point)
    assert scheduler.chunked_req is None


def test_between_forward_recovery_finds_request_in_last_batch():
    req = SimpleNamespace(
        rid="live-rid", kv_ec_output_paused=True, kv_ec_forward_blocked=True
    )
    point = SimpleNamespace(
        request_id="live-rid", recovery_version=1, delivered_tokens=0,
        computed_unsent_tokens=(), kv_token_boundary=1024,
        created_ns=__import__("time").monotonic_ns(),
    )
    hook = LiveRecoveryHook.__new__(LiveRecoveryHook)
    hook.scheduler = SimpleNamespace(
        running_batch=None, last_batch=SimpleNamespace(reqs=[req]), waiting_queue=[]
    )
    hook.requested = True
    hook.completed = False
    hook.point = point
    hook._refresh_from_incremental_protection = lambda live, saved: None
    hook._choose_before_materializing_ec = lambda saved: None
    hook._choose_path = lambda saved: ("EC", "test")
    hook._recover_ec = lambda live, saved: (SimpleNamespace(state="COMMITTED"), 1.0)
    hook._event = lambda *args, **kwargs: None
    assert hook.before_schedule() is True
    assert hook.completed is True and hook.requested is False
    assert req.kv_ec_active_version == 1
    assert req.kv_ec_output_paused is False and req.kv_ec_forward_blocked is False


def test_adaptive_incremental_grains_cover_only_completed_ranges():
    req = SimpleNamespace(
        finished=lambda: False, is_retracted=False, to_finish=None,
        req_pool_idx=1, session_id=None, multimodal_inputs=None,
        kv_committed_len=1024, origin_input_ids=[1] * 1024, output_ids=[],
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(name="EXTEND"),
        seq_lens_cpu=torch.tensor([1024]),
    )
    assert list(completed_chunks(req, batch, 0, 0, 128)) == list(range(8))
    assert list(completed_chunks(req, batch, 0, 0, 256)) == list(range(4))
    assert list(completed_chunks(req, batch, 0, 0, 512)) == list(range(2))


def test_qwen17_incremental_chunks_assemble_exact_28_layer_prefix():
    from .incremental_checkpoint import build_node_checkpoint
    from .node_block_export import unpack
    from .test_debug_capture import rank_fixture

    rank_chunks = {}
    for rank in (0, 1):
        values = []
        for chunk in (0, 1):
            raw, meta = rank_fixture(rank, 2, layers=range(28), count=128, heads=4)
            meta.update(
                snapshot_id=f"chunk-{chunk}", checkpoint_version=f"chunk-{chunk}",
                request_id="same-request", request_version="node-run",
                manager_epoch="node-run", model_version=meta["model_sha256"],
                chunk_id=chunk, capture_grain_tokens=128,
                token_start=chunk * 128, token_range=[chunk * 128, (chunk + 1) * 128],
                layer_range=[0, 28], physical_slots=[
                    rank * 1000 + chunk * 128 + i + 1 for i in range(128)
                ],
            )
            values.append((meta, raw.tobytes()))
        rank_chunks[rank] = values
    checkpoint, blocks = build_node_checkpoint(
        0, rank_chunks, 256, list(range(256)), allow_fixture=True
    )
    meta, unpacked = unpack(checkpoint, allow_fixture=True)
    assert meta["ranks"][0]["token_count"] == 256
    assert meta["ranks"][0]["layers"] == list(range(28))
    assert tuple(blocks) == unpacked


def test_node_budget_admits_qwen17_tp2_bf16_16k_prefix():
    from .node_block_export import MAX_NODE_BYTES

    per_rank = 28 * 2 * 16384 * 4 * 128 * 2
    assert per_rank * 2 < MAX_NODE_BYTES


def test_tcp_chunk_pipeline_encodes_before_full_peer_upload_completes():
    from .remote_rs_client import binding
    from .remote_rs_server import StripeStore
    from .rs_codec import encode
    from .rs_stripe_builder import build
    from .test_remote_rs import node_fixture

    blocks = [node_fixture(i) for i in (0, 1)]
    manifest = build("streaming-live-contract", blocks, allow_fixture=True)
    store = StripeStore(allow_fixture=True, rs_backend="cpu")
    created, _ = store.dispatch(
        dict(op="create_stripe", stripe_id="streaming-live-contract", manifest=manifest), b""
    )
    key = binding(created)
    status = None
    for node, block in enumerate(blocks):
        # Fixtures fit in one chunk. Real Qwen node blocks exercise many 4 MiB
        # chunks through the same transaction and bounded parity buffer.
        status, _ = store.dispatch(dict(
            op="submit_node_chunk", node_id=node, offset=0, **key,
        ), block)
    assert status["state"] == "COMMITTED"
    assert status["streaming_pipeline"] is True
    assert store.stripes["streaming-live-contract"]["parity"] == encode(*blocks)
