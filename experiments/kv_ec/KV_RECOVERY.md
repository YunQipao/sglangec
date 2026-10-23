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

The ordinary page-size-one allocator has experimental slot generations and
leases. The Scheduler capture hook can run a non-destructive local or
cross-rank lease validation transaction. This is not a recovery pause/resume
mechanism for arbitrary scheduling modes. Consequently this phase does not implement:

- destructive KV erasure;
- `set_kv_buffer` writeback;
- request continuation after a simulated rank failure;
- asynchronous capture return before the copy owner releases its lease.

The correct next interface is a scheduler-owned recovery transaction that
pins the request row and every physical slot across all TP ranks, records a
slot generation/allocator epoch, waits for all relevant CUDA streams, performs
the writeback, and validates the same generation before resuming scheduling.
Until writeback ownership and rollback are proven, GPU experiments are limited
to isolated capture and non-destructive lease validation.

## Lease prototype

`recovery_lease.py` is a CPU model of that transaction. It is not imported by
SGLang and does not change allocator behavior. Its required integration points
are:

1. `begin_recovery`: scheduler thread after the final prefill/decode forward and
   before `release_kv_cache`, after all TP ranks report the same request and
   slot versions.
2. `before_free`/`before_alloc`: allocator `free`, `free_segment(s)` and
   allocation paths must reject or defer protected slots.
3. `validate_recovery`: after all CUDA copy/write events complete, re-read row
   generation, physical slots, slot generations, checkpoint state and manager
   epoch.
4. `commit_recovery`: release the lease only after validation and writeback.
5. `abort_recovery`: release the lease and leave original KV untouched.

The ordinary page-size-one `TokenToKVPoolAllocator` now exposes an opt-in local
slot-generation registry and allocator-side free/reuse guards. An additional
opt-in Scheduler transaction can pin one request's local slots, record a real
CUDA event, re-read request/slot generations, and release the lease. It is
local to each TP rank and is not a global recovery commit.
The registry requires a worker-created CUDA event; completion booleans are not
accepted. Timeout or registry shutdown never makes an in-flight slot reusable.

## Read-only Scheduler integration

`scheduler_recovery_guard.py` is now called from `debug_capture.capture()` only
when `SGLANG_KV_EC_RECOVERY_GUARD=1`. It reads the real scheduler's:

- `scheduler.req_to_token_pool.req_to_token`
- `scheduler.req_to_token_pool.req_generation`
- `req.req_pool_idx`
- `scheduler.model_worker.ps.tp_rank/tp_size`

It checks the supported mode (TP=4, one request, no overlap, no prefix cache,
no chunked prefill), records a SHA256 of the physical-slot list, and never
stores the raw mapping in the diagnostic output. The guard itself remains
read-only (`state=READ_ONLY_BLOCKED`). The separate transaction is enabled
only with `SGLANG_KV_EC_SCHEDULER_LEASE=1`; it reports `LOCAL_RELEASED` only
after a real CUDA Event completes and the mapping and generations are
revalidated. It never calls `set_kv_buffer`.

For the ordinary page-size-one allocator, `alloc`, `free`, grouped
`free_group_end`, `clear`, and `resize` are guarded through the registry.
`merge_and_sort_free` only moves already-freed pages and cannot bypass the
guard. Prefix-cache insertion/eviction and paged/SWA/HiSparse/unified
allocators, plus `alloc_extend`/`alloc_decode`, are not covered.

## Page-size-one allocator integration

The opt-in `TokenToKVPoolAllocator.slot_lease_registry` now maintains a local
generation counter for every physical token slot. `alloc()` increments the
counter, while `free()` and grouped frees reject currently leased slots.
`begin_slot_lease`, `validate_slot_lease`, and `release_slot_lease` are exposed
only on this allocator. The default is disabled and no KV tensor is modified.

This covers only the ordinary `page_size=1` allocator used by the controlled
TP=4 BF16 NHD experiment. It does **not** cover `PagedTokenToKVPoolAllocator`,
SWA, HiSparse, Unified, NPU, DCP, prefix-cache ownership, or every allocation
kernel path. Therefore `writeback_supported` remains false and this is not
permission to run a destructive experiment.

## Cross-rank coordination and complete prefill

`cross_rank_lease.py` contains a CPU-only TP=4 protocol with explicit states
`PREPARE`, `LOCAL_LEASED`, `ALL_RANKS_READY`, `CUDA_COMPLETED`,
`ALL_RANKS_VALIDATED`, `RELEASE_AUTHORIZED`, and `GLOBAL_RELEASED`. A rank failure or timeout enters
a terminal failure state; all local abort acknowledgements are required before
the coordinator releases its logical ownership. The model performs no NCCL or
GPU access; `cross_rank_worker.py` now connects real local Scheduler leases to
the independent Unix control server. This integration still requires a new
TP4 GPU run; CPU tests are not GPU evidence.

`chunk_protection.py` requires three independently `COMMITTED` 256-token
snapshots for a 768-token request. It rejects metadata-only chunks, gaps,
overlap, mixed request versions, checkpoint versions, manager epochs, or
incomplete TP rank sets. Only `COMPLETE` represents a full prefill protection
point.

The controlled launcher can now request those captures with
`--chunk-size 256` (and `--token-count 256`). It creates
`<base>-chunk-1..3`, and the capture hook reads the three disjoint ranges from
the same live request before normal cache release. Start the Manager with
`--max-pending 3`; this is required because the chunks are independent
checkpoints. Validate them with:

```bash
python -m experiments.kv_ec.complete_prefill \
  --socket "$RUN/manager.sock" \
  --base-snapshot "$SNAP" \
  --request-id kv-ec-artificial-request \
  --request-version request-v1 \
  --checkpoint-version "$SNAP" \
  --manager-epoch "$MANAGER_EPOCH" \
  --output "$OUT/$SNAP-complete.json"
```

This command only aggregates already `COMMITTED` Manager metadata; it does
not create a checkpoint or perform GPU writeback. A missing or failed chunk
causes a nonzero exit and no `COMPLETE` report.

`cross_rank_control.py` provides the separate, metadata-only Unix control
plane for the next integration stage. Its protocol is:

```text
prepare_rank (all ranks) -> local_leased (all ranks)
        -> cuda_completed (all ranks)
        -> validated (all ranks)
        -> authorize_release
        -> actual local allocator release + local_released ACK (all ranks)
        -> GLOBAL_RELEASED
```

The control server owns the CPU coordinator and never calls NCCL or touches
GPU memory. Workers must create and query their own CUDA Events before sending
the corresponding messages. Exact duplicate phase messages are idempotent.
A timeout, coordinator restart, or rank failure cannot produce `GLOBAL_RELEASED`;
failed transactions require all four `abort_ack` messages before coordinated
slot ownership is cleared.

Start it separately from the Parity Manager:

```bash
python -m experiments.kv_ec.cross_rank_control \
  --socket "$RUN/control.sock" \
  --parity-socket "$RUN/manager.sock" \
  --timeout 30
```

Enable the live bridge only with `SGLANG_KV_EC_CROSS_RANK_LEASE=1` in addition
to the existing Guard and Scheduler Lease flags. See [CROSS_RANK_GPU.md](CROSS_RANK_GPU.md)
for a complete isolated run. All switches default off.

Timeout only invalidates an allocator lease; it never automatically removes
slot ownership, even when the Event has completed. Explicit abort cleanup
requires a Worker-owned CUDA fence and matching original Manager epoch.
A failed transaction may have some local leases released, but must never be
reported as globally successful. If CUDA completion cannot be proved, the
remaining lease is retained fail-closed; normal allocator reclamation can then
fail and operator intervention is required. This is not fault-tolerant serving.

## CPU checks

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
python -m py_compile experiments/kv_ec/kv_pool_audit.py
python -m unittest experiments.kv_ec.test_kv_pool_audit -v
```
