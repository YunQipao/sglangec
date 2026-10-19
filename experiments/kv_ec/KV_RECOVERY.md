# Controlled KV recovery status

This document records the boundary of the current prototype. The new
`kv_pool_audit.py` helper is read-only and is disabled unless a caller invokes
it explicitly. It does not import SGLang, call NCCL, call `set_kv_buffer`, or
modify a live KV pool.

## What is proven

- The current capture hook runs before output-token append and before
  `release_kv_cache`/`maybe_cache_unfinished_req` in the non-overlap prefill
  path.
- `ReqToTokenPool` exposes a request-row generation and a logical-token to
  physical-slot mapping.
- `ReqToTokenPool.free()` immediately returns the row to `free_slots`; its
  generation is not a physical-slot generation.
- Prefix caching can share and rewrite physical slots. Therefore a historical
  `COMMITTED` JSON is not proof that those bytes are still live or owned by the
  request.
- A full 768-token prefill can be represented as three 256-token child
  snapshots, but all three must share one request version and must be
  independently `COMMITTED`. `validate_chunk_group()` rejects gaps, overlap,
  mixed request IDs, mixed versions, and duplicate child IDs.

## What is deliberately blocked

There is no physical-slot generation, lease, allocator pin, or scheduler
pause/resume transaction in this checkout. A CUDA event only proves that
previous GPU work completed; it does not prevent another scheduler path from
freeing or reusing a slot. Consequently this phase does not implement:

- destructive KV erasure;
- `set_kv_buffer` writeback;
- request continuation after a simulated rank failure;
- asynchronous capture return before the copy owner releases its lease.

The correct next interface is a scheduler-owned recovery transaction that
pins the request row and every physical slot across all TP ranks, records a
slot generation/allocator epoch, waits for all relevant CUDA streams, performs
the writeback, and validates the same generation before resuming scheduling.
Until that interface exists, the only valid GPU experiment is read-only
mapping/layout inspection in an isolated server.

## CPU checks

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
python -m py_compile experiments/kv_ec/kv_pool_audit.py
python -m unittest experiments.kv_ec.test_kv_pool_audit -v
```
