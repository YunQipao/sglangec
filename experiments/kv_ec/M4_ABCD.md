# M4 A/B/C/D Development

## Status and Scope

Branch: `research/online-xor-profiling`. Changes are confined to experiments.
No inference-source changes, environment upgrades, cross-node work, task
takeovers, raw KV files or uploaders are introduced.

GPU acceptance status: **A/B/C/D single-run smoke PASS and three-repeat
study PASS**, from existing evidence `/tmp/kv-ec-abcd-PYvwwJ/`. The later
GPU run was performed in the user's GPU environment; this current execution
environment has no `/dev/nvidia*`, and `nvidia-smi` cannot communicate with
the driver. The evidence was audited read-only here. Do not infer GPU
correctness from CPU tests alone.

Latest smoke manifest: `/tmp/kv-ec-abcd-PYvwwJ/smoke/trials.json`.
All four acceptance records and performance artifacts are present and
strict `summarize()` audit returns PASS with no measured-source drift.
The existing three-repeat manifest at
`/tmp/kv-ec-abcd-PYvwwJ/three/trials.json` also audits PASS. No additional
three-repeat run was started during this turn.

Current CPU regression: **224 tests in 148.500 seconds, OK (9 skipped)**;
log: `/tmp/kv-ec-m4-first-failure-zWdkIZ/cpu-regression.log`. Skips are
sandbox socket/listener restrictions, not skipped Lease checks. Earlier
regressions had 222 tests before the two performance-record tests were added.

Earlier preflight evidence: `/tmp/kv-ec-abcd-dev-4mds2A/`.
`B-smoke/acceptance.json` and `D-smoke/acceptance.json` both report
`state=FAIL`, `GPU_VALIDATED=false`, `EXPERIMENT_PASS=false`, with the
NVIDIA driver error. No model server was launched or fault injected.
`B-smoke.log`, `D-smoke.log`, `cpu-regression.log` and `pilot-audit.json`
are non-sensitive local records. `unified-smoke/` holds the unified
driver's additional CPU/preflight evidence. No three-repeat study is
claimed or promoted from these failed preflights.
`execution-status.json` summarizes the earlier preflight failures and is
not the acceptance manifest; use the later smoke and three-repeat manifests
above for current GPU results.

### B/D Report Construction Fix (2026-10-21)

The later GPU runs `/tmp/kv-ec-pause-accept-ezr_ooqc` (B),
`/tmp/kv-ec-pause-accept-9uvx3sav` (D), and
`/tmp/kv-ec-abcd-smoke-new/repeat1/protection_only` reached identical-token
validation but failed while constructing their performance JSON:
`TypeError: dict() got multiple values for keyword argument 'note'`.
The streaming client's metrics already contain `note`; the report builder
also supplied that keyword explicitly.

`experiment_performance_record()` now preserves the client `note` and uses
`timing_note` for all-rank protection/recovery clock semantics. Regression
tests cover both B and D, including preservation of client metrics,
four-rank maxima, JSON serialization and unchanged input records. This fix
does not change Scheduler execution, allocator ownership or Lease checks.
The complete CPU regression passed **224 tests in 151.852 seconds,
OK (9 skipped)**; log: `/tmp/kv-ec-note-fix-yCvCYU/cpu-regression.log`.

A read-only re-audit of these three saved runs passed the existing strict
evidence checks and corrected report construction. Its record is
`/tmp/kv-ec-note-fix-yCvCYU/offline-reaudit.json`. This is not a new GPU
trial: original acceptance files remain FAIL and were not rewritten.
The unified smoke correctly stopped at B; the subsequent three-repeat
study correctly rejected it with `smoke_not_pass`.

The later passing smoke and three-repeat study supersede the old failed
runs as current acceptance results; old FAIL artifacts remain unchanged.
For a future source change, first run a fresh single-repeat smoke and only
consider a separately approved repeat study after it passes. Never manually
promote failed acceptance or skip the smoke admission check.

| Mode | Implementation |
| --- | --- |
| A `baseline` | Fresh ordinary TP4 server/request, no EC gate. |
| B `protection_only` | Existing paused Scheduler lease, real prefill capture and COMMITTED parity, no injection/reconstruction/writeback, normal authorized decode. |
| C `online_recovery` | Existing target-rank logical fault, three survivors plus precommitted parity, real writeback and exact-byte checks. |
| D `prefill_retry` | Same original request, first-token boundary and lost range; four-rank handoff, ordinary resource release/reallocation, real 768-token prompt replay, exact-byte validation, normal decode. |

Initial unified study: Qwen3-8B, TP4, one request, 768 input tokens, 16
outputs, BF16 NHD, page size 1, no overlap/prefix reuse/chunked prefill/CUDA
graphs, Triton attention, deterministic greedy sampling. Only rank0/256 is
promoted by the unified driver. Standalone D permits ranks 0..3 but only a
256-token logical loss; larger loss ranges are deliberately rejected.
All experimental paths remain default-off. Normal SGLang is unchanged.

## B and Parity Fairness

B reuses `SchedulerShadowRecovery.prepare()` in `SchedulerPauseGate`; no
separate protection state machine is created. Protection takes place after
the first prefill result, before the four-rank paused observation. Every
rank validates its COMMITTED identity and rechecks unchanged source bytes.
The existing validated/released/armed/readiness-vote protocol ends the
row/slot protection before normal decode.

Neither B's Scheduler nor its parity manager calls XOR recovery. The
manager runs without `--test-oracle`; parity encoding and rank/hash/layout
validation still run. Explicit new `--mode online_recovery` uses exactly
the same production-parity setting. Its actual fault reconstruction,
survivor hashes, recovered hashes, layer bytes, rollback safety checks and
four-rank interlocks remain mandatory.

Legacy `--online-restore`, `m4_trials.py` and the historical A/C Pilot keep
their recovery-based parity test oracle. That legacy measurement revision
must not be mixed with the new production-parity study. B cannot be
measured by subtracting two nested C timers.

## D Ownership and History

1. The existing gate pauses after initial prefill and its first output
   token, before the first decode allocation. It owns all 768 prompt slots
   and their row generation, and fences the original producer stream.
2. All ranks hash the full 768-token K/V state as verification-only
   oracles. Only hashes survive; no original KV payload is used to rebuild.
3. After a four-rank readiness vote, the designated rank zeros the first
   256 prompt slots across all 36 layers, fences and verifies the fault.
   The target's monotonic fault timestamp is broadcast over the CPU TP
   group with a bounded wait. This is not a TTFT measurement.
4. Each rank submits its validated, event-complete handoff proof to the
   existing coordinator. No old lease can end before all four proofs.
   Under `RETRY_PENDING`, release the old slot lease and row protection,
   vote, acknowledge release and wait for all four acknowledgements.
5. Invoke the existing `release_req(..., offload_kv=False)` path. It calls
   normal `release_kv_cache`, frees the ChunkCache-owned slots and request
   row, and resets retraction state. There is no allocator bypass.
6. Reinitialize the same request, require zero prefix reuse, prepare a
   normal `ScheduleBatch` with `set_extend_range(0, 768)`, and allocate
   through the ordinary pool/allocator. Require fresh row/allocation
   identity, protect the new row and begin a new slot lease before forward.
7. Execute `scheduler.run_batch()` on all four ranks. Exactly 768 prompt
   tokens are recomputed, not 769. Verify its replayed first sample equals
   the retained first output. Do not call `process_batch_result` on this
   replay result: doing so would append/stream that first token twice.
   The normal forward relay republishes that same token for next decode.
8. Reestablish `cache_unfinished_req` and `last_batch` using the new batch.
   Verify all three 256-token chunks, all 36 layers, both K/V, on all ranks
   against the pre-fault logical hashes, allowing changed physical mapping.
   Check original `output_ids` and all actual `send_*`, `surr_offset`,
   `read_offset` values are unchanged.
9. Vote after new-state validation, submit four rebuild proofs, then use
   the original external authorization and normal lease-release handshake.
   Only the next actual decode result creates final success evidence.
   All 16 final IDs must equal A, with the original logical HTTP request.

Cancellation, changed epoch/generation/mapping, missing rank, incomplete
event, parity/source drift, wrong replayed sample or byte mismatch remains
fail-stop. Failed handoff/rebuild does not have a resume shortcut. There
is no retry HTTP request, parity source, rollback restore or Lease bypass.

## Clock Contract

All fault clocks are single-node monotonic clocks. Do not add enclosing
intervals, parallel ranks or host and nested CUDA-event measurements.

| Metric | Definition |
| --- | --- |
| Client TTFT/TPOT/total | Same existing streaming client in every mode; startup excluded, no warmup. TPOT includes pause/control/validation, not pure kernel decode latency. |
| B `protection_blocking_ms` | Rank-local gate pause start through normal row/slot release/readiness point. Includes capture, commit, rechecks, coordination and external authorization. Max across four ranks is reported. Not a causal A/B delta. |
| B `parity_commit_ms` | Sum of manager first-submission-to-COMMITTED intervals across protected chunks. Nested in protection; identical values checked against saved rank evidence. |
| D `fault_to_rebuilt_state_ms` | Target fault clock through full prompt byte validation and all-rank completion vote; not just kernel completion. |
| D `fault_to_first_decode_ms` | Same fault clock through first actual normal post-rebuild decode result, including authorization/release. Not SSE arrival or request TTFT. |
| D `prefill_forward_ms` | CUDA-event span around the actual replay `run_batch`, including its normal forward input handling and sampling. Not isolated GEMM time. |
| D `prefill_forward_host_ms` | Host span enclosing that replay, completion fencing and normal Scheduler wrapper. CUDA events are nested. |
| D `normal_release_ms` / `reallocation_ms` | Normal resource release and reallocation/new-lease readiness respectively, excluding prior handoff wait. |
| D `rebuilt_forward_done_monotonic` | Raw forward completion boundary before full byte validation, retained separately from verified rebuilt-state time. |

The max rank-local fault-to-decode is used for both C and D in the unified
summary. C also retains its historical target-local `fault_to_decode_ms`
and detailed component timings. The full D recovery includes CPU hash
validation and control cost, and the original C path retains its extensive
validation/rollback safety work. This is a correctness-heavy comparison,
not a production-overhead estimate.

## Historical Pilot Re-Audit

Source: `/tmp/kv-ec-m4-trials-a2uy7v97`. All 12 acceptance/performance trees
were rechecked through `validate_pilot`/`audit_case`, including byte/token
proofs, independent transaction/baseline identities, all-rank authorization,
configuration and saved timer recomputation.

Report: `/tmp/kv-ec-abcd-dev-4mds2A/pilot-audit.json`.
All values below are raw three-run ranges, milliseconds; no stable P95.

| Lost rank / tokens | A TTFT | A TPOT | A total | C TTFT | C TPOT | C total | C fault->decode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 / 256 | 207.98-211.30 | 38.87-49.00 | 790.99-946.36 | 211.52-212.29 | 169.41-178.36 | 2753.38-2887.39 | 907.14-1009.39 |
| 1 / 256 | 212.54-215.78 | 40.32-44.85 | 818.25-886.85 | 210.92-213.50 | 161.82-177.25 | 2640.86-2869.67 | 891.35-1003.70 |
| 0 / 768 | 211.38-212.34 | 44.25-46.70 | 875.88-912.08 | 208.94-215.29 | 391.76-394.12 | 6085.34-6127.10 | 2307.83-2315.73 |
| 3 / 768 | 209.92-212.54 | 41.83-47.86 | 837.36-928.23 | 210.52-212.65 | 366.58-389.99 | 5711.40-6060.42 | 2240.89-2292.95 |

B/D measurements and the new production-parity C measurements are
**unavailable pending real GPU acceptance**. No speedup is calculated.

## Commands

Run from the repository root, without changing the CUDA/PyTorch environment:

```bash
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
SHADOW=/tmp/kv-ec-pause-accept-o1pg2zr8

# Always CPU first.
"$PY" -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v

# Only when four selected GPUs are idle. Every acceptance run checks them.
# Standalone B and D each own a fresh independent A baseline as well.
"$PY" -u -m experiments.kv_ec.accept_pause \
  --mode protection_only --gpus 0,1,2,3 --shadow-evidence "$SHADOW" \
  --lost-rank 0 --recover-tokens 256 --pause-timeout 120 \
  --output /tmp/kv-ec-B-smoke-new
"$PY" -u -m experiments.kv_ec.accept_pause \
  --mode prefill_retry --gpus 0,1,2,3 --shadow-evidence "$SHADOW" \
  --lost-rank 0 --recover-tokens 256 --pause-timeout 120 \
  --output /tmp/kv-ec-D-smoke-new

# Unified rank0 / 256 lost / 768 prompt / 16 output smoke; CPU runs first.
# Create a fresh parent, leaving both output subdirectories uncreated.
RUNS=$(mktemp -d /tmp/kv-ec-abcd-note-fix-XXXXXX)
"$PY" -u -m experiments.kv_ec.m4_abcd \
  --gpus 0,1,2,3 --shadow-evidence "$SHADOW" --repeat 1 \
  --output "$RUNS/smoke" &&
# Three fresh independent repetitions ONLY after complete smoke success.
"$PY" -u -m experiments.kv_ec.m4_abcd \
  --gpus 0,1,2,3 --shadow-evidence "$SHADOW" --repeat 3 \
  --smoke-evidence "$RUNS/smoke" \
  --output "$RUNS/three"
```

A and C are independently callable with `accept_pause --mode baseline`
and `accept_pause --mode online_recovery`, respectively. C needs
`--shadow-evidence "$SHADOW"`; A does not. Ports default to 30359/30360.
Use a new output directory; evidence is never overwritten.

Each B/C/D acceptance still owns an extra paired A server to prove its
exact output against an independent ordinary request. The unified A run
is also fresh and its IDs/input/sampling/model/GPU UUIDs must match all
three other modes. Therefore one unified repeat uses seven sequential
servers, not four simultaneous servers. First-request/JIT effects are
included equally; no warmup is silently introduced.

The unified driver reruns CPU tests, stops on the first failed mode,
audits saved proofs, compares all modes/repeats, writes `trials.json`
incrementally and writes `summary.json` only for a complete accepted
study. It requires unchanged source/configuration from smoke to repeat=3.
The evidence contains metadata/hashes/artificial output IDs and logs,
never raw KV arrays.

## Remaining Work and Bottlenecks

- Real B/D GPU byte/token/lifecycle acceptance remains mandatory. CPU
  fixtures validate control/ownership semantics, not the Triton/model path.
- Do not start three repeats until single-run A/B/C/D succeeds; no P95
  or speedup claims from absent measurements.
- Profile full-prompt D hash validation separately from replay forward.
  D currently hashes all 768 tokens before and after rebuilding on all ranks.
- B/C synchronous 72-buffer-per-chunk capture, hashes and parity IPC
  remain foreground costs; preserve checks before optimizing copies.
- Existing gate polling, coordinator round trips and four-rank readiness
  votes remain in both recovery tails. Split these timers before attributing
  causality to raw XOR or prefill compute.
- Longer lost ranges/prompts, ten-run statistics and asynchronous
  protection remain future work, not part of this change.
