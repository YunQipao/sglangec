# M7 Recovery Policy

M7 separates the protection lifecycle from the recovery decision:

| Checkpoint state | Permitted action | Parity usable |
| --- | --- | --- |
| `UNPROTECTED` | `PREFILL_RETRY` only at a safe prefill handoff | no |
| `SNAPSHOT_READY` | `PREFILL_RETRY` only at a safe prefill handoff | no |
| `COMMITTING` | `PREFILL_RETRY` only at a safe prefill handoff | no |
| `COMMITTED` | `EC_RECOVERY` when all bindings and votes match | yes |
| `FAILED` | `PREFILL_RETRY` only at a safe prefill handoff | no |

The policy is implemented by `recovery_policy.decide_recovery()`. It is a
pure function over checkpoint state, fault rank, token range, request
lifecycle, Row/Slot Generation, Manager Epoch, and four-rank readiness. It
does not inspect or retain KV bytes.

The post-Resume hook treats protection readiness and fault intent separately.
`UNPROTECTED`, `SNAPSHOT_READY`, and `COMMITTING` without an explicit fault
request remain nonblocking: Decode continues while commit is pending. If a
fault is explicitly requested before `COMMITTED`, the pure policy recommends
`PREFILL_RETRY`, but this post-Resume lifecycle cannot safely replay the
768-token prefill without crossing the scheduler handoff or duplicating
output. It therefore records the recommendation and fails closed. It never
calls `fetch_parity`, `reconstruct`, or GPU writeback before commit. Once all
ranks are `COMMITTED` and decode has been observed, M6-B2's deliberate
logical-fault test boundary records `fault_requested` and enters the existing
EC path after a four-rank policy vote.

All ranks exchange the policy code at a fixed collective point. A mismatch,
request cancellation, Manager Epoch change, Row/Slot Generation change,
unsupported range, multiple fault ranks, or missing committed parity is
`FAIL_CLOSED`. A failed/cancelled background worker invalidates the pending
protection; it cannot authorize recovery.

## Multi-Chunk Plan

The next runtime extension keeps each 256-token chunk independent:

- 256 tokens: one checkpoint `request-id/chunk-0`, one `SNAPSHOT_READY ->
  COMMITTING -> COMMITTED` lifecycle and one payload/K/V oracle.
- 512 tokens: chunks 0 and 1 are submitted in order, each with its own
  checkpoint version and parity. Chunk 1 may not be used until its own
  commit is observed.
- 768 tokens: chunks 0, 1 and 2 follow the same ordered queue. A request is
  fully EC-recoverable only when every protected chunk is `COMMITTED` and all
  chunk bindings agree.

The queue is bounded to one in-flight chunk per request and a fixed total
byte budget. A queue timeout or Manager failure leaves that chunk
uncommitted and selects Prefill Retry only if the scheduler is still at a
pre-decode replay boundary; otherwise it fails closed. A committed chunk may
be recovered independently only when replaying the remaining chunks is not
needed and the request mapping/lease covers the exact chunk. If any required
chunk is uncommitted, the complete request falls back to one full 768-token
Prefill Retry; partial EC plus partial replay is not supported.

Output correctness requires the original output history and stream offsets
to remain unchanged, fresh Row/Slot generations for a retry, and independent
historical payload and per-layer K/V hashes for every recovered chunk.
