# Controlled GPU KV Recovery Design

This document defines the next experimental boundary for KV EC recovery. It
does not enable online KV writeback and does not change the SGLang scheduler.
The only implemented GPU recovery path remains `gpu_scratch_restore.py`, which
allocates temporary tensors owned by the experiment process.

## Current Evidence Gate

The fresh cross-rank run `kv-ec-cross-rank.MgFVa6` passed the preconditions for
offline recovery design:

- three chunks are `COMMITTED`;
- each chunk has four `GLOBAL_RELEASED` rank records;
- all twelve CUDA event records are complete;
- slot generations before and after the lease match;
- the allocator available size is restored on every rank;
- the request response was produced and the control/server logs contain no
  worker or coordinator failure.
- the 768-token response output IDs match the prior baseline evidence exactly
  (same input hash, deterministic sampling, and 16 output IDs).

This is evidence for the control plane and CPU parity ownership. It is not
evidence that an online KV pool can be overwritten safely.

## Existing Performance Evidence

The repository currently contains one cross-rank three-chunk run, not ten
independent A/B runs per configuration. Therefore the following values are
diagnostic only and must not be reported as a controlled overhead result:

| Run | Samples | Scheduler capture median/P95 (ms) | Control median/P95 (ms) | Parity commit median/P95 (ms) |
|---|---:|---:|---:|---:|
| scheduler lease + cross-rank | 12 rank/chunk records | 274.36 / 364.85 | 180.58 / 272.44 | 148.59 / 157.34 |
| scheduler lease only | 4 rank records | 117.66 / 117.93 | N/A | N/A |

The two rows are not matched ten-run experiments: they use different dates,
request IDs, and run configurations. The apparent difference cannot be
attributed solely to cross-rank control. The cross-rank parity commit is
asynchronous with respect to the lease release, so it must not be added to
the synchronous foreground cost without an explicit queue/backpressure
measurement.

## Recovery Transaction

The proposed transaction is per request range and per TP rank. It must have a
unique transaction ID, capture ID, request version, manager epoch, checkpoint
version, request-pool row generation, and exact half-open token range.

```text
PREPARE
  -> PAUSED
  -> LEASED
  -> RECONSTRUCTED
  -> WRITEBACK_VALIDATED
  -> RESUMED

Any state before RESUMED may transition to ABORTED.
RESUMED and ABORTED are terminal.
```

### State contracts

`PREPARE`

- Scheduler owner thread identifies the request by `rid`.
- Verify the supported runtime: single request, ordinary MHA pool, NHD layout,
  BF16/FP16, no quantized KV, no prefix sharing, no chunked prefill,
  no overlap scheduling, no speculative/PD/PP/DP/DCP/HiCache/unified mode.
- Record `req_pool_idx`, request-row generation, token range, physical slots,
  pool class, tensor shapes, strides, dtype, device, and model/source identity.
- Reject before any collective or pool mutation if any rank disagrees.

`PAUSED`

- Stop scheduling this request at a scheduler-owned safe point after its last
  completed forward and before cache/free/reuse can rewrite the request row.
- Do not use an HTTP callback or an arbitrary decode boundary as the pause
  primitive.
- The pause must be acknowledged by every TP rank. Missing acknowledgements
  fail closed.

`LEASED`

- Hold the request row and all referenced physical slots against allocator
  free/reuse for the duration of the transaction.
- A row generation check alone is insufficient: the current allocator does
  not expose a general physical-slot generation API.
- The first implementation must keep the lease and copy in the scheduler
  owner thread, or add an explicit allocator lifetime epoch before allowing
  asynchronous copy.

`RECONSTRUCTED`

- Reconstruct one missing rank only from a complete, same-version parity group.
- Validate all rank metadata, checksums, token range, layout, model identity,
  manager epoch, and source commit before allocating a destination.
- No reconstructed bytes may be written into the online KV pool at this state.

`WRITEBACK_VALIDATED`

- Allocate a separate destination tensor and copy the reconstructed bytes into
  the exact local K/V layout.
- Validate byte equality/checksums and shape/stride/device metadata on the
  destination.
- Record the CUDA stream and event used for the copy. The event must complete
  before the destination becomes visible to attention.
- Re-check request-row generation, physical slots, lease ownership, and manager
  epoch. Any mismatch aborts the transaction.

`RESUMED`

- Resume only after all TP ranks have acknowledged validated destinations and
  the scheduler has installed the corresponding mapping.
- The first resumed forward must use the same request version and token range.
- A response or decode continuation is not proof of byte correctness; retain
  the validation evidence.

`ABORTED`

- Never report successful recovery.
- Do not release or reuse a protected slot until the relevant CUDA event is
  complete or the process has been safely fenced.
- If one rank fails after another rank has written, leave the online pool
  unchanged in this prototype and fail the request/transaction closed.

## SGLang Integration Boundary

The safe candidate point is the scheduler result path after the final prefill
forward has completed and before `update_finish_state`,
`release_kv_cache`, or unfinished-request caching can rewrite/free mappings.
The exact implementation must be runtime-checked rather than inferred from
static class names.

The integration must obtain, on each rank:

- the live `Req` and its `req_pool_idx`;
- the request-row generation;
- `req_to_token` slots for the exact completed range;
- the concrete `TokenToKVPool` class and local K/V tensors;
- K/V shape, dtype, stride, device, layer range, and stream/event dependency.

`get_key_buffer`, `get_value_buffer`, and `get_kv_buffer` are internal accessors,
not lifetime leases. They cannot by themselves make asynchronous snapshot or
writeback safe. `set_kv_buffer` must not be called by the offline path.

## Scratch-Only Experiment

The currently allowed experiment is:

1. Read a real complete rank snapshot.
2. Recover one rank in CPU memory.
3. Validate checksum, K/V records, 36-layer layout, and 256/768-token range.
4. Allocate temporary GPU tensors.
5. Copy bytes to the temporary tensors.
6. Validate every destination byte.
7. Release the temporary tensors.

The output must contain `online_kv_pool_modified=false`. It must never receive a
live pool pointer or request mapping. Existing evidence:

- `kv_ec_runs/tp4-all-256-20261016T142603/recovery.json`: all four erasure
  positions byte-exact, all K/V checksums equal, dtype/shape valid.
- `kv_ec_runs/scratch_all_256.json`: TP4, 36 layers, one 256-token chunk,
  9,437,184 bytes per rank, ten samples per lost rank, byte-exact.
- `kv_ec_runs/scratch_l0_128.json`: one-layer layout smoke test.

The cross-rank run covers three separate 256-token chunks, which is a
768-token request-level protection sequence. It is not a single contiguous
768-token Scratch allocation; that larger single-allocation case still needs a
separate measurement.

Once a new GPU run has produced three real complete directories, the sequence
Scratch verifier is:

For the complete capture/request/manifest-preparation/acceptance commands see
[GPU_SCRATCH_CAPTURE.md](GPU_SCRATCH_CAPTURE.md). Capture writes rank files only:
run explicit `prepare_scratch` before the command below. Historical FAST_PARITY
diagnostic directories cannot be used as offline payload snapshots.

```bash
python -m experiments.kv_ec.gpu_scratch_restore \
  "$OUT/$SNAP-chunk-1" "$OUT/$SNAP-chunk-2" "$OUT/$SNAP-chunk-3" \
  --device cuda --warmup 2 --repeat 10 \
  --output "$OUT/$SNAP-scratch-sequence.json"
```

The command refuses missing/metadata-only chunks, mixed request identity,
non-contiguous token ranges, wrong chunk lengths, or a non-CUDA device. It
loads and releases one chunk's payload at a time and reports
`online_kv_pool_modified=false`.

These results authorize more scratch measurements, not online writeback.

## Promotion Criteria for Online Writeback

Online writeback remains blocked until all of the following exist:

1. A scheduler-owned pause/resume API with per-rank acknowledgements.
2. An allocator lease that prevents physical slot free/reuse, including an ABA
   protection mechanism stronger than request-row generation alone.
3. A runtime layout validator covering every supported KV pool variant, with
   unsupported variants rejected before mutation.
4. Stream/event ordering tests proving attention cannot observe a partial copy.
5. Failure-injection tests for rank loss, manager epoch change, stale mapping,
   generation change, CUDA event timeout, and partial writeback.
6. A transaction journal that makes partial local writes fail closed and
   prevents false `RESUMED` or `GLOBAL_RELEASED` results.
7. Ten independent matched A/B GPU experiments with fresh server, manager, and
   snapshot IDs, reporting median/P95 for TTFT, TPOT, request total,
   scheduler capture, control, and parity queue/commit latency.

Until these gates are met, the correct conclusion is:

```text
offline CPU parity: enabled
offline GPU scratch restore: enabled
online GPU KV writeback: blocked
```
