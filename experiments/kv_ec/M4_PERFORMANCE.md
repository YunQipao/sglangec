# M4 Performance Baseline

## Scope and Evidence

The M3 correctness matrix `/tmp/kv-ec-m3-matrix-9h4va7t5` is real TP4 GPU
evidence: 12/12 PASS, independent baseline servers, matching input/sampling,
four-rank authorization, per-chunk byte-exact live KV writeback and decode.
`m4_analysis.py` reruns the existing verifiers, verifies the saved performance
numbers against rank evidence, and records JSON hashes. It does not start a
server, touch CUDA or modify the source evidence.

Reproduce the report in a new directory:

```bash
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
"$PY" -m experiments.kv_ec.m4_analysis \
  --matrix /tmp/kv-ec-m3-matrix-9h4va7t5 \
  --output /tmp/kv-ec-m4-report-new
```

The audited report is `/tmp/kv-ec-m4-audit-20261021-final/`:
`audit.json`, `performance.csv`, `REPORT.md`. All 12 cases are single
observations (n=1). No rank is statistically faster/slower based on these
sequential observations. No stable Median/P95 is inferred.

No measured Scheduler, lease, coordinator, XOR or CUDA path was changed in
this increment. Preserving those source hashes makes the new A/C trials
comparable to the existing correctness-heavy M3 implementation.

## Actual Measurements

Milliseconds, each cell is a single independent run. Protect is the target
rank's sum of per-chunk protection durations; it is not the four-rank critical
path. XOR includes CPU verification; Verify includes live GPU/hash checks.

| Rank | Tokens | Protect | Parity Commit | Fault->Decode | XOR+Verify | Gather | H2D | KV Write | Verify |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 256 | 499.73 | 121.04 | 1019.04 | 190.17 | 77.53 | 15.77 | 9.13 | 239.04 |
| 0 | 512 | 944.97 | 608.88 | 1614.16 | 297.37 | 122.73 | 24.42 | 15.16 | 396.51 |
| 0 | 768 | 1705.38 | 1091.30 | 2381.42 | 460.11 | 173.59 | 38.70 | 23.13 | 598.54 |
| 1 | 256 | 456.31 | 152.64 | 904.30 | 150.17 | 67.97 | 12.18 | 7.12 | 203.69 |
| 1 | 512 | 1272.37 | 572.71 | 1614.20 | 299.68 | 127.36 | 24.25 | 15.33 | 398.43 |
| 1 | 768 | 1326.41 | 964.20 | 2329.47 | 450.62 | 172.56 | 36.24 | 22.74 | 586.63 |
| 2 | 256 | 495.76 | 109.85 | 1003.10 | 190.36 | 65.64 | 15.51 | 8.74 | 242.64 |
| 2 | 512 | 1235.73 | 542.60 | 1620.64 | 299.52 | 107.69 | 24.19 | 14.95 | 396.38 |
| 2 | 768 | 1683.26 | 1029.24 | 2368.70 | 452.89 | 166.89 | 36.94 | 22.89 | 589.88 |
| 3 | 256 | 484.49 | 106.21 | 918.81 | 146.54 | 82.78 | 12.15 | 7.10 | 201.87 |
| 3 | 512 | 1265.55 | 577.15 | 1573.59 | 287.47 | 137.51 | 23.29 | 14.75 | 386.40 |
| 3 | 768 | 1683.29 | 1080.58 | 2376.69 | 438.88 | 174.22 | 42.33 | 27.03 | 605.90 |

Across these different configurations, the baseline requests take
873.00-936.29 ms; online-recovery requests take 2743.38-6224.42 ms.
These are ranges of raw observations, not latency distributions or confidence
intervals. The total difference is not pure protection overhead.

## Clock and Interval Contract

Rank/coordinator timestamps use local monotonic clocks on one node.
`perf_counter` durations measure host wall time. Explicit CUDA-event times
measure stream intervals and are nested inside host transfer durations.
Cross-rank wall spans are valid for this single-node workload, not cross-node.

| Measurement | Start -> End | Meaning and overlap |
| --- | --- | --- |
| Client TTFT | before local HTTP open -> first SSE event with output IDs | Includes HTTP, queueing, tokenization/server prefill and first stream emission, not server startup. First token can precede the post-prefill pause. |
| Client TPOT | first output event -> final response consumed, divided by 15 | Amortized client value including pause, recovery, final SSE/HTTP overhead; not steady-state decode kernel latency. |
| Request Total | before HTTP open -> final response consumed | TTFT and 15*TPOT partition this client interval for 16 output tokens. |
| Capture D2H+Hash | before `pack_chunk` -> return | 36 layers x K/V: mapping validation, GPU gather, blocking D2H, NumPy copy, hashes, concatenation, metadata validation. Not pure DMA time. |
| Pure D2H / CPU hash | NOT SEPARATELY MEASURED | Existing JSON cannot separate them retrospectively. Do not infer PCIe bandwidth from the composite duration. |
| Parity encoding | before `encode(blocks)` -> return in manager | CPU array copy plus three NumPy XOR passes; recorded as checkpoint `xor_ms`. |
| Parity oracle/metadata | before four test-oracle reconstructions -> timestamp while building commit metadata | Four recoveries/compares plus parity hash/metadata work; `oracle_validation_ms` is not production encoding time. |
| Checkpoint submit | before `submit_once` -> response received | Synchronous serialization/copies, Unix IPC, manager validation and possibly commit/oracle work when final rank submits. |
| Parity commit | first checkpoint accepted/created -> commit timestamp | Includes wait for other ranks, input validation/copies, XOR, oracle/hash. Overlaps submit and protection. |
| Per-chunk protection | before capture -> COMMITTED observed by that rank | Encloses capture, submit, polling and scheduling delay. Chunk sums omit gaps between calls. |
| All-rank protection span | earliest inferred chunk start -> latest commit observed among all ranks | Captures skew, inter-chunk gaps and pending polls; still not total foreground pause. |
| Scheduler pause | gate event/lease preparation completed -> resume-ready release/votes | `pause_ms`, includes protection, prefault checks, recovery and external authorization; stops before the next decode result. |
| Fault trigger | immediately before clearing selected live slots | Explicit logical injection, not detection of a real failed GPU/process. Detection latency is unmeasured. |
| Fault injection+verify | before zero -> cleared-byte/change verification done | Fault simulation and experimental validation cost, not a production detector. |
| Survivor Gather | CPU collective launch -> bounded Gloo work wait complete | Target contributes zeros; exactly three survivor inputs are consumed. Includes rank rendezvous and host transport, not network-only transfer. Other ranks can wait while target injects. |
| Parity fetch | before `fetch_parity` -> validated response returned | Unix IPC, CPU copies and version/parity validation. |
| XOR+CPU verify | before `reconstruct` -> return | Checkpoint/layout checks, parity+survivor SHA, copy/three XOR passes, reconstructed payload SHA and 72 layer views/copies. Pure recovery XOR not isolated. |
| H2D host sum | per K/V allocation start -> synchronous copy and synchronize done | 72 allocations/copies plus host overhead. Each temporary tensor verification is timed separately. |
| H2D CUDA-event sum | stream event before copy -> event after copy | Nested in H2D host sum; not additive to it. Not entire recovery GPU time. |
| Recovered GPU verify | after H2D -> temporary GPU bytes read back and compared | 72 D2H readbacks; experimental byte verification. |
| KV writeback | before mapped scatter -> final completion fence | Destination/source validation, 72 index copies and CUDA event polling. |
| Writeback GPU+hash verify | before live/rollback comparisons -> final recapture metadata equal | GPU comparisons and another full D2H/hash/metadata pass. Encloses mixed validation/transfer, not a pure kernel interval. |
| Byte validation sum | temporary GPU verify + writeback GPU/hash verify | Distinct target-local validation blocks; does not include the checks inside XOR, fault simulation, protection or prefault stages. |
| Chunk recovery sum | fault trigger -> local final checks complete, summed over requested chunks | Excludes prefault live validation, rollback capture, inter-chunk votes and next chunk preparation. Includes simulated fault clearing/verification. |
| Verified->Decode span | target starts `shadow_verified` IPC -> first post-resume decode result | Includes four-rank proof completion, external authorize, 50ms polling, release/armed handshakes/votes, and decode; not pure RPC overhead. |
| Armed->Decode span | target starts `armed` IPC -> first post-resume decode result | A subset of Verified->Decode, includes waiting for peers and first real decode. |
| Control IPC sum | sum of target `_call` elapsed durations | Across the whole paused transaction; mostly nested in the above intervals. Does not include Gloo votes. |
| Fault->Decode | first chunk fault timestamp -> first actual resumed decode result | Full observed post-fault latency, includes next chunk validation/rollback preparation, voting and automatic external authorization. |

The four-rank authorization is proven by all paused/validated/released/armed
ranks and coordinator state, but its *exclusive algorithmic cost* is not
isolated. There is no timestamp for decode launch, so first-decode compute
time cannot be separated from waiting with the existing records.

Do not add enclosing intervals, four parallel ranks, CUDA-event and host
transfer times, parity commit and protection, per-chunk fault->decode times,
or `pause_ms` and recovery. The earliest fault->decode is measured once.
The multi-chunk fault->decode tail contains later chunks, so tails overlap.

Protection duration is not measured foreground overhead versus B: that
requires a protection-only run. `online_total - baseline_total` mixes
protection, safety/validation, scheduling harness, recovery and runtime noise.

## Bottleneck Findings

For 256-token cases (four separate single observations):

- Capture composite: 233.53-239.17 ms for **9 MiB per rank per chunk**.
  `checked_buffers` repeatedly reads GPU min/max/unique via scalar checks;
  72 blocking per-layer D2H copies and multiple full/aggregate/layer hashes
  are on the Scheduler's critical path. This is not evidence of slow DMA.
- Pure parity encoding: 7.44-9.58 ms. The manager's oracle/metadata interval
  is 64.89-70.76 ms, nested in commit. Commit is 106.21-152.64 ms.
- Recovery XOR+CPU verify: 146.54-190.36 ms versus encoding 7.44-9.58 ms.
  Both use the same three-XOR primitive over equal-size blocks, but the
  recovery timer also covers extensive hash/layout validation. This strongly
  motivates split profiling, not a claim that XOR itself takes 190 ms.
- Gather: 65.64-82.78 ms; parity fetch is additional local transport/checks.
  Rank rendezvous and fault simulation can inflate survivor wait times.
- H2D host: 12.15-15.77 ms; nested CUDA copy events: 5.81-7.21 ms.
  Live KV writeback: 7.10-9.13 ms.
- Explicit byte validation sum: 201.87-242.64 ms, plus checks nested inside
  the capture/XOR stages. It is a large measured foreground cost.
- Target proof-submit to first resumed decode: 332.06-347.68 ms for these
  four cases; individual control IPC totals are only 11.26-16.17 ms.
  They cover different intervals, so subtracting them does not yield pure
  polling time. The gate's 50ms state polling is a concrete next profiling
  target, not proof of the exclusive cause of this latency.

For 768 tokens, all-rank protection spans 1976.94-2111.32 ms, while target
protection sums span 1326.41-1705.38 ms. Reporting only the target sum would
miss meaningful inter-chunk/rank skew. Explicit byte validation sums grow
to 586.63-605.90 ms; fault->decode is 2329.47-2381.42 ms.

Prioritized next changes, **after the pilot**, each with a new measurement
revision and full correctness rerun:

1. Split capture into checked-slot validation, GPU gather, CUDA D2H event,
   host copy/concat, hash and metadata validation. Use per-stage exclusive
   host timers plus nested CUDA events; do not remove any safety checks.
2. Split reconstruction into parity/survivor hash, raw XOR and payload/layer
   validation. Measure `recover()` exclusively before considering optimization.
3. Record readiness-vote duration and each coordinator state wait separately;
   distinguish RPC latency, polling sleep, peer wait and first decode compute.
4. Evaluate batched copies and preallocated/pinned buffers against identical
   byte checks only after the above breakdown. No implementation change yet.
5. Async protection requires immutable copied data, producer events and slot
   lifetime protection until DMA completion, bounded queues/backpressure, and
   COMMITTED-before-fault semantics. The current synchronous gate blocks
   decode; making a worker thread alone does not remove this dependency.
6. Existing parity/checkpoints are already reused after fault; no parity
   generation occurs during reconstruction. Never reuse across a different
   snapshot, request version, model fingerprint, epoch or slot generation.
   Avoiding repeated read/hash validations requires a separate correctness
   argument, not silently disabling them for a better benchmark number.

## A/B/C/D Fairness and Retry Status

| Mode | Actual implementation status |
| --- | --- |
| A: no protection | Real independent baseline per accepted M3/C trial; available. |
| B: protection only | NOT IMPLEMENTED in this acceptance harness; protection timers inside C are not a separate B request. |
| C: online EC recovery | Real Scheduler GPU fault/recovery; available. |
| D: lost KV then real prefill retry | NOT IMPLEMENTED; no recovery latency or speedup reported. |

Existing code examination found that `ScheduleBatch.retract_decode()` keeps
at least one request and therefore will not retract this one-request workload.
`retract_all()`/`release_req()` can release request/KV resources, but invoking
them while the current EC gate owns its row/slot lease must fail. Setting
`SGLANG_TEST_RETRACT` or sending a new HTTP request is not a fair D baseline.
This increment does not bypass the lease or alter Scheduler ownership.

Required D implementation contract before comparing:

1. Same 768-token IDs (hash matched), Qwen3-8B fingerprint, four GPU UUIDs,
   TP4/BF16/NHD/page1, no overlap/prefix reuse, deterministic temperature 0,
   16 output tokens. First fault boundary is after initial prefill and one
   emitted output token, before its first decode, identical to C.
2. Preserve the exact emitted output history and logical request identity.
   Inject the same lost rank/range, fence it, vote across all ranks, and
   start the fault clock at the same point. Do not read original target KV
   or use rollback bytes to rebuild state.
3. Use a coordinator-controlled **retry handoff**: after all in-flight GPU
   work completes and generations are revalidated, transfer the blocked
   request out of its protected lease into a dedicated RETRY_PENDING state.
   Only that explicit state transition may release the old physical mapping.
   The existing EC failure path remains fail-closed, never a resume shortcut.
4. Retract/requeue via the tested resource-release path after the handoff.
   Rebuild full prefix on all TP ranks with no cached-prefix reuse and no
   offloaded/backup KV. Non-target shards cannot compute a missing TP shard
   independently, so full-model prefill is the conservative retry baseline.
5. At the recovery boundary C owns prompt KV and a first output token whose
   KV is written by the next decode. If ordinary retry uses
   `origin_input_ids + output_ids`, it performs a 769-token extend and may
   also emit the next token. Count this explicitly: separately record
   prompt-state reconstruction completion and first new output completion.
   Do not compare a C state-before-decode time with D time-after-extra-token,
   or feed the first token twice. Prove logical history/positions match.
6. Validate rebuilt layer K/V against the original logical prompt bytes (as
   a verification oracle only), but permit a fresh row/slot generation.
   EC C must still keep its original mapping. Retry's changed generations
   are evidence of reallocation, not a reason to weaken C's lease checks.
7. Authorize only after all four ranks rebuild and pass validation; observe
   the next real output and compare all 16 IDs exactly to A. Preserve the
   original HTTP logical request or separately account for reconnection and
   deduplicated already emitted tokens.
8. Record fault->rebuilt-equivalent-state, fault->first-new-output,
   GPU prefill forward, allocation/retraction/queue delay, validation,
   synchronization and original request total separately. A normal TTFT is
   never any of these fault-recovery durations.

For B, reuse the same prefill/protection/COMMITTED interlock without injection
or recovery, then release through the normal four-rank protocol and observe
actual decode. It needs its own admitted mode and negative tests; shadow
recovery is not a protection-only substitute.

## Staged Experiments

The new runner intentionally measures only paired A/C with the unchanged
correctness-heavy implementation. It cannot claim M4 A/B/C/D completion.
Every independent C trial has a fresh baseline server and fresh C server,
not multiple samples from the same recovered request. There is no warmup
request; first-request/JIT effects are part of the current configuration.
Run one study at a time, never parallel on the selected four GPUs.

Minimum one-case GPU performance validation (CPU regression runs first):

```bash
"$PY" -u -m experiments.kv_ec.m4_trials \
  --matrix /tmp/kv-ec-m3-matrix-9h4va7t5 \
  --shadow-evidence /tmp/kv-ec-pause-accept-o1pg2zr8 \
  --gpus 0,1,2,3 --repeat 1 --case 0:256 \
  --baseline-port 30359 --pause-port 30360 \
  --startup-timeout 600 --pause-timeout 120
```

First phase: three independent runs for each of (0,256), (1,256),
(0,768), (3,768), interleaved in repeat-major order:

```bash
"$PY" -u -m experiments.kv_ec.m4_trials \
  --matrix /tmp/kv-ec-m3-matrix-9h4va7t5 \
  --shadow-evidence /tmp/kv-ec-pause-accept-o1pg2zr8 \
  --gpus 0,1,2,3 --repeat 3 --output /tmp/kv-ec-m4-pilot-new
```

Inspect raw samples, outliers, software/UUID consistency and all byte/token
checks. No automatic outlier removal. The pilot produces ranges/raw samples,
**not stable Median/P95**. Promotion requires successful complete 3-run
evidence; human review of variability is still required:

```bash
"$PY" -u -m experiments.kv_ec.m4_trials \
  --matrix /tmp/kv-ec-m3-matrix-9h4va7t5 \
  --shadow-evidence /tmp/kv-ec-pause-accept-o1pg2zr8 \
  --gpus 0,1,2,3 --repeat 10 \
  --pilot-evidence /tmp/kv-ec-m4-pilot-new \
  --output /tmp/kv-ec-m4-ten-new
```

Ten runs are ten *additional independent runs* per configuration. Their
Median/P95 are descriptive statistics; n=10 P95 is still tail-uncertain.
The runner saves `trials.json` incrementally, each raw acceptance/log tree,
and `summary.json`. It prints a heartbeat every 30 seconds and stops at the
first CPU/GPU/correctness/configuration failure. Only owned process groups
are signaled on interruption. Hardware/software/source fingerprints must
match; experiments-only commit IDs may differ only with identical measured
file hashes and a verified empty inference-source diff.

Legacy `m3_performance.py` accepts arbitrary small samples and computes
quantiles; use the M4 summary for claims about independent n=1/3/10 studies.

Second phase: **not enabled**. Local model config allows 40960 positions,
but `launch_debug.py` currently starts servers with context length 4096 and
the recovery gate requires exactly 768 prompt slots. 4K prefix plus output
does not fit that server context, and 8K definitely does not.

Before 2K, inspect real free GPU memory and mapping capacity on all four
devices, count prompt+output KV slots, extend the request/gate/context
configuration together, retain all page1 generations and full prefix lease,
and rerun correctness before measuring. Repeat for 4K, then 8K. At TP4 the
plain prompt KV footprint is 36 KiB/token/rank (72 MiB for 2K, 144 MiB for
4K, 288 MiB for 8K), excluding pool reservation, rollback, scratch and model
memory. Those arithmetic estimates are not an actual free-memory check.

## Privacy and Remaining Gaps

The audited tree contains only JSON/log evidence, hashes, mappings and
artificial response IDs, no raw KV binary artifacts. Captured raw KV stays
in local CPU/GPU memory and is sent only through local Unix/Gloo channels.
No uploader is introduced. Offline artifact/source inspection cannot prove
historical machine-wide absence of network egress; such attestation needs
independent network auditing.

Remaining work: real B/D implementations with validated handoff semantics,
exclusive D2H/hash/XOR/coordinator profiling, independent 3/10-run results,
thermal/clock/system-load telemetry, and safe longer-prefix correctness.

## This Increment's Execution

On 2026-10-21 the complete CPU regression ran **207 tests in 147.572s,
OK (9 skipped)**. The skipped tests require real sockets unavailable in
this sandbox. The new M4 unit suite also passed separately; saved GPU
matrix auditing executes without CUDA or subprocess launch.

The M4 rank0/256 one-case smoke reached the normal GPU preflight after CPU
success, then stopped because NVIDIA-SMI cannot communicate with the driver.
This sandbox exposes no NVIDIA devices and PyTorch reports CUDA unavailable.
No inference server was started and no logical fault was injected. This
environment failure does not invalidate the earlier real M3 GPU records.
There are no new pilot/ten-run measurements in this increment.

```text
/tmp/kv-ec-m4-smoke-20261021/cpu-regression.log
/tmp/kv-ec-m4-smoke-20261021/trials.json
/tmp/kv-ec-m4-smoke-20261021/repeat1-rank0-tokens256.log
/tmp/kv-ec-m4-smoke-20261021/repeat1-rank0-tokens256/acceptance.json
/tmp/kv-ec-m4-unit-final-20261021.log
```
