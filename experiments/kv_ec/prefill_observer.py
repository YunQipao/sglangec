"""Default-off observations for the single-request TP2/256 eager control only.

No KV creation or mutation. The opt-in CUDA fence is deliberately included in
the measured path; observations are inclusive and must never be summed.
"""
import os
import time

from .perf_observer import emit, enabled


def _profile(s, batch):
    if os.environ.get('SGLANG_KV_EC_PREFILL_OBSERVE') != '1' or not enabled():
        return None
    args = s.server_args
    if (not s.is_generation or s.enable_overlap or len(batch.reqs) != 1 or
            int(s.model_worker.ps.tp_size) != 2 or args.pp_size != 1 or
            args.chunked_prefill_size != -1 or args.max_running_requests != 1 or
            not args.disable_radix_cache or args.dtype != 'bfloat16' or
            args.kv_cache_dtype != 'auto' or not batch.spec_algorithm.is_none() or
            str(args.cuda_graph_config.decode.backend) != 'disabled' or
            str(args.cuda_graph_config.prefill.backend) != 'disabled'):
        return None
    req = batch.reqs[0]
    if (len(req.origin_input_ids) != 256 or req.req_pool_idx is None or
            req.finished() or req.is_retracted or req.inflight_middle_chunks > 0 or
            req.kv_committed_len < 256):
        return None
    return req


def prefill_kv_ready(s, batch, result):
    """Called after stock result handling/cache bookkeeping, before streaming."""
    req = _profile(s, batch)
    if (req is None or not batch.forward_mode.is_extend_without_speculative() or
            result.can_run_cuda_graph or len(req.output_ids) != 1 or
            getattr(batch, 'extend_lens', None) != [256] or
            getattr(batch, 'extend_num_tokens', None) != 256):
        return
    # Confirm sampled token availability and completion of all KV CUDA work.
    # Imported only in the explicit opt-in profile, never on the normal path.
    import torch
    torch.cuda.synchronize(s.req_to_token_pool.device)
    now = time.monotonic_ns()
    emit('prefill_kv_ready', 'recovery', start_ns=now, end_ns=now,
         tp_rank=int(s.model_worker.ps.tp_rank), request_id=req.rid,
         token_count=256, sampled_token_id=int(req.output_ids[0]),
         scheduler_ready=True, cuda_completed=True, profile='tp2-256-eager-stock-prefill')


def before_decode(s, batch):
    """First decode only, before native run_batch launches any model work."""
    req = _profile(s, batch)
    if req is None or not batch.forward_mode.is_decode() or len(req.output_ids) != 1:
        return
    now = time.monotonic_ns()
    emit('kv_first_decode', 'decode', start_ns=now, end_ns=now,
         tp_rank=int(s.model_worker.ps.tp_rank), request_id=req.rid,
         token_count=256, profile='tp2-256-eager')
