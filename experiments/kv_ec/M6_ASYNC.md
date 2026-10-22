# M6: blocking snapshot, asynchronous CPU protection

Scope: Qwen3-8B, TP4, one request, 768 artificial input tokens, 16 output tokens,
first 256-token chunk only, BF16 NHD, page size 1. Default off. A/B/C/D and
the original manager wire protocol/parity layout remain available unchanged.
This is a prototype protection mode, **not an asynchronous online-recovery mode**.

## Actual dependencies

`SchedulerPauseGate.after_result` runs after Prefill output/cache bookkeeping and
before the next Decode allocation. It protects the request row and all 768
physical slots, fences schedule-stream producers into forward-stream, and records
the allocator's lease CUDA event. Mapping, row/slot generation, request lifetime
and all existing runtime restrictions are checked.

B/C's existing `prepare` captures, submits and waits for COMMITTED **before**
the four-rank PAUSED observation. B rechecks its committed source; C performs the
existing logical KV fault / byte-exact recovery. Their source lease release thus
depends on COMMITTED. D retains its independent Prefill retry handoff.

E uses the same initial event, four-rank PAUSED/readiness votes and validated
release/arm/resume handshake, but its coordinator proof explicitly says
`checkpoint_committed=false`. Its distinct proof means **only copy completion
and immutable Snapshot readiness**, never parity/recoverability.

E capture:

1. Wait on the producer/lease event on forward-stream.
2. Run the original blocking `pack_chunk`, including the existing layout/slot/
   hash checks. No asynchronous GPU-copy refactor.
3. Record and synchronize a capture completion CUDA event.
4. Copy payload into immutable Python `bytes`, serialize metadata into immutable
   JSON, and validate the owned bytes independently. Later caller mutations and
   writable NumPy aliases cannot modify this Snapshot.
5. All four ranks attest Snapshot readiness. Existing generation/event/lease
   validation and final TP votes still authorize source Slot/Row release.
6. Only after release, start one rank-local CPU worker and return to Decode.
   No foreground `join`, commit polling, parity fetch or recovery collective.
7. The worker sends immutable bytes using the original manager protocol. The
   independent manager validates every rank, XORs, and commits. The worker polls
   status with a fixed deadline, validates the exact checkpoint and rank hash,
   and publishes terminal metadata-only evidence.

## State and ownership

| State | Owned memory | Meaning |
|---|---|---|
| UNPROTECTED | none | No accepted task |
| SNAPSHOT_READY | one immutable CPU block + metadata | Copy complete; not EC recoverable |
| COMMITTING | the same CPU block; bounded socket/manager copies | Background submit/poll; not EC recoverable |
| COMMITTED | metadata/status only | Independent manager validated the complete TP set |
| FAILED | in-flight bytes until bounded I/O exits, then none | Error/cancel/timeout; never recoverable |

`AsyncProtection` accepts exactly **one task including executing work**, with a
9,437,184-byte (9 MiB) per-rank Snapshot budget for this scope. A second task is
rejected, never queued behind it. No completed-task/payload history accumulates.
Snapshot identity includes request hash/version, chunk range, checkpoint version,
manager epoch, TP rank, request row/generation, physical slots and slot generations.

The 9 MiB budget is **not a whole-process memory cap**. Blocking capture has
bounded temporary arrays; freeze briefly owns mutable and immutable copies.
The existing 64 MiB protocol limit, bounded TP4 receiver set and manager's
`max_pending=1` bound IPC/checkpoint working sets. Manager XOR/parity/transport
copies and model allocations are separate. The isolated manager exits after the
run; no multi-request/multi-checkpoint service claim is made.

Normal request completion does not block for COMMITTED. Record the state at
request end, retain CPU-only work, and let the **acceptance process** drain off
the request path until the fixed deadline. Request abort/retraction cancels and
invalidates local protection. Already sent manager work may finish later, but
cannot revive FAILED or pass E acceptance. No cancellation thread frees live KV.

`require_recoverable` rejects precommit faults and stale epoch/checkpoint/request/
row/slot bindings. It is **authorization only**: any future E recovery must acquire
the existing live recovery lease and independently fetch/validate parity. E does
not dispatch fault injection, writeback or retry at all; actual unsupported faults
remain fail-closed through the existing scheduler failure path. Do not claim M3
recovery after releasing the capture lease without reacquiring a valid live lease.

## Evidence and timing

`pause-rank-N.json`: actual source CUDA events, Lease/Row release, runtime checks,
Snapshot binding, foreground blocking and first real Decode observation.

`async-rank-N.json`: immutable independent background terminal record, manager
status, state transitions, commit-observed clock, Snapshot residence/high-water.

`async-lifecycle-rank-N.json`: normal end vs cancellation, pending state, drain
policy, and no foreground wait. The auditor requires all four normal lifecycles.

`checkpoints.json` and `parity-export-verification.json`: production manager
COMMITTED status and actual exported parity length/SHA verification. No KV/parity
payload is written to disk.

- Request Total / TTFT / TPOT: original SSE client metrics, not acceptance duration.
- Foreground blocking: the whole E gate interval through background handoff,
  including Snapshot/hash work, four-rank control and Lease release.
- Snapshot time: capture start through immutable ownership/validation.
- Background COMMITTED: Snapshot ready to observed, validated COMMITTED. Includes
  handoff/control time before submission; it is not a foreground wait interval.
- Unprotected window: completed Prefill to observed COMMITTED, conservatively
  including polling/transport delay. Not measured from request arrival.
- CPU: per-rank process lifetime `ru_maxrss`, not current resident memory or sum
  of all ranks plus manager.
- GPU: CUDA allocator high-water **at Snapshot**, not whole-request peak/NVML
  device usage. A/B memory remains uninstrumented. Do not fabricate A/B/E memory
  deltas from these partial measurements.

The diagnostic smoke inserts a deliberate background delay to require
every rank's first Decode **before any rank starts submitting**, and before
`committed_observed`, exercising a request that can
end before commit. Its latency is **not performance evidence**. Performance uses
zero injected delay and unprofiled fresh A/B/E servers, reporting paired E-A,
E-B, and foreground deltas. Early natural commit in a performance run is allowed;
the code still contains no commit wait in Decode.

Reducing foreground work explicitly creates a window during Decode where parity
is not yet committed. Report the measured window alongside any request benefit.
B's window is independently reconstructed from the same completed-Prefill origin
and its original commit-observed timestamp. `E_minus_B_unprotected_window_ms`
explicitly reports the tradeoff; a positive value means a larger window.
No M6 performance threshold or improvement is claimed without real GPU evidence.

## Linux commands

Use a clean committed `research/online-xor-profiling` checkout and the existing
CUDA/PyTorch environment. Do not upgrade packages. The study driver runs the full
CPU suite first. Acceptance checks all four selected GPU UUIDs for active compute
tasks and refuses busy hardware; it signals only processes it created.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
PY=${PY:-python}
SHADOW=/tmp/kv-ec-pause-accept-o1pg2zr8
OUT=$(mktemp -d /tmp/kv-ec-m6-XXXXXX)

# One correctness smoke, with deliberately delayed background commit.
"$PY" -u -m experiments.kv_ec.m6_abe \
  --gpus 0,1,2,3 --shadow-evidence "$SHADOW" \
  --diagnostic-overlap-smoke --async-commit-delay 2 \
  --repeat 1 --startup-timeout 600 --pause-timeout 120 \
  --output "$OUT/smoke"

# Only after smoke PASS: three fresh, unprofiled, ZERO-delay A/B/E repeats.
"$PY" -u -m experiments.kv_ec.m6_abe \
  --gpus 0,1,2,3 --shadow-evidence "$SHADOW" \
  --smoke-evidence "$OUT/smoke" --repeat 3 \
  --startup-timeout 600 --pause-timeout 120 --output "$OUT/three"

# Read-only re-audit: starts no CPU tests, servers or GPU work.
"$PY" -m experiments.kv_ec.m6_abe \
  --verify-study "$OUT/three" --output "$OUT/three-reverified.json"
```

Run each command only after the preceding command succeeds. Do not compare smoke
Request Total with the undelayed study. A dirty source or incompatible smoke is
rejected before the first server starts. Source commit identities are preserved.

Standalone E-only minimal real-GPU acceptance (runs its own real A token oracle):

```bash
"$PY" -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v &&
"$PY" -u -m experiments.kv_ec.accept_pause \
  --mode async_protection_only --gpus 0,1,2,3 \
  --shadow-evidence "$SHADOW" --async-commit-delay 2 --require-async-overlap \
  --startup-timeout 600 --pause-timeout 120 --output "$OUT/e-only"
```

CPU failure cases, without any GPU or server:

```bash
"$PY" -m unittest experiments.kv_ec.test_m6_async experiments.kv_ec.test_m6_abe -v
"$PY" -m unittest \
  experiments.kv_ec.test_m6_async.AsyncTests.test_fault_before_committed_fail_closed_and_rank_timing_skew \
  experiments.kv_ec.test_m6_async.AsyncTests.test_postcommit_stale_epoch_checkpoint_row_slots_and_request_rejected \
  experiments.kv_ec.test_m6_async.AsyncTests.test_cancel_ready_inflight_late_commit_and_after_commit_invalidate \
  experiments.kv_ec.test_m6_async.AsyncTests.test_manager_error_and_timeout_never_become_recoverable -v
```

These are CPU tests, **not** real-GPU fault acceptance. GPU E fault/retry dispatch,
live post-commit E recovery, multi-request cancellation propagation, multi-chunk
queues, manager-wide byte admission accounting, full A/B/E/manager memory
telemetry, asynchronous D2H, and cross-node protection are not implemented.
