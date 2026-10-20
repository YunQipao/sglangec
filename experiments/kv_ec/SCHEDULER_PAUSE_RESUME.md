# Scheduler Pause/Resume Prototype

This is a default-off, metadata-only prototype. It does not read, erase,
restore, or overwrite online KV data. It is intentionally limited to one
Qwen3-8B request on plain TP4 with BF16 NHD KV, page size 1, no prefix cache,
no overlap, no chunked prefill, and no disaggregation or pipeline parallelism.

## Actual call chain

For the supported non-overlap scheduler:

1. `Scheduler.run_event_loop` creates `schedule_stream` and enters its
   `StreamContext`.
2. `event_loop_normal` receives and processes requests.
3. `get_next_batch_to_run` selects the prefill batch.
4. `run_batch` calls `model_worker.forward_batch_generation`.
5. `ModelRunner.forward` writes the normal KV pool and returns the prefill
   result.
6. `process_batch_result` appends the first output token and performs the
   normal cache bookkeeping.
7. `SchedulerPauseGate.after_result` observes that completed prefill. It reads
   the ReqToToken mapping and slot generations, protects the request row,
   starts the local slot lease, and records a CUDA event.
8. The next event-loop iteration calls `gate.tick` before batch selection.
   While active, selection and model forward are skipped. Four ranks vote on
   the CPU TP process group each iteration.
9. After explicit coordinator authorization, every rank validates the row,
   request version, slot generations, and CUDA event. All four ranks release
   their local lease and row protection before the final resume vote.
10. The normal scheduler selects the request again. The final evidence file is
    written only after the next real decode result is processed.

The event fence records the current producer stream used by the normal
non-overlap loop, then makes `forward_stream` wait for it before publishing the
allocator event. This prevents an event on an unrelated stream from being
treated as proof that KV writes completed.

## State machine

Coordinator states:

`PREPARE -> PAUSED -> RESUME_REQUESTED -> RELEASE_AUTHORIZED ->
ALL_RELEASED -> RESUME_AUTHORIZED`

Terminal state: `FAILED`.

Rank-local states include `EVENT_PENDING`, `PAUSED`, `VALIDATED`, `RELEASED`,
`ARMED`, and `RESUMED`. `RESUMED` means only that scheduling was authorized.
The final JSON evidence additionally requires `decode_observed=true`.

Any request-row generation, slot generation, request-version, coordinator
epoch, CUDA-event, rank vote, authorization, or deadline failure is fail-closed.
The slot lease registry is closed on a local failure so protected slots cannot
be reused by this process.

## Isolated validation

Use a new run directory and a new transaction ID. Prepare a coordinator:

```bash
RUN=/tmp/kv-ec-pause-$(date -u +%Y%m%dT%H%M%S)
mkdir -p "$RUN"
TX="pause-$(date -u +%Y%m%dT%H%M%S)"

python -m experiments.kv_ec.pause_control serve \
  --socket "$RUN/pause.sock" \
  --transaction-id "$TX" \
  --timeout-s 30 \
  --identity-output "$RUN/identity.json"
```

Start the four-rank server with the identity from that server:

```bash
export SGLANG_KV_EC_PAUSE_RESUME=1
export SGLANG_KV_EC_PAUSE_REQUEST_ID=kv-ec-pause-request
export SGLANG_KV_EC_PAUSE_IDENTITY="$(cat "$RUN/identity.json")"
export SGLANG_KV_EC_PAUSE_SOCKET="$RUN/pause.sock"
export SGLANG_KV_EC_PAUSE_OUTPUT="$RUN/ranks"
export SGLANG_KV_EC_PAUSE_TIMEOUT_S=30
export SGLANG_KV_EC_PAUSE_IPC_TIMEOUT_S=0.5
```

Use the existing isolated TP4 Qwen3-8B serving command with overlap,
prefix/radix cache, chunked prefill, PD, PP, DP, and DCP disabled. Send exactly
one 768-token request with request ID `kv-ec-pause-request`. After all four
ranks report `PAUSED`, authorize once:

```bash
python -m experiments.kv_ec.pause_control authorize \
  --socket "$RUN/pause.sock" --identity "$RUN/identity.json"
```

Inspect the four final records:

```bash
find "$RUN/ranks" -name 'pause-rank-*.json' -print
rg '"decode_observed": true|"online_kv_pool_modified": false' "$RUN/ranks"
```

An authorization without a subsequent real decode is incomplete and must not
be accepted as a successful pause/resume run. No command in this prototype
calls `set_kv_buffer`.
