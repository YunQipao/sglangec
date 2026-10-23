# M3 Online Recovery

## Live integration

`SchedulerShadowRecovery` now accepts `lost_rank=0..3` and
`recover_tokens=256/512/768`. The default remains rank 0 / chunk 0.
This is wired into the existing Scheduler pause gate, not a separate CPU
recovery demonstration.

- Protection captures ordered, independent `transaction_id-chunkN`
  snapshots/checkpoint versions. All checkpoints must be COMMITTED before
  the PAUSED observation and before the first fault. Only one checkpoint
  is pending at a time, preserving the manager's bounded pending capacity.
- The selected rank contributes zeros to CPU TP Gather and receives the
  other three ranks in TP-rank order. Its original CPU payload is discarded;
  neither that payload nor GPU rollback bytes are recovery inputs.
- Only the selected rank injects faults and writes recovered bytes into the
  actual KV pool. Every chunk uses its current 256-token mapping and verifies
  all 36 layers' K/V bytes, CUDA completion, hashes and mapping generations.
- `M3RecoveryPlan` is populated only after the live chunk checks and a bounded
  four-rank completion vote. Coordinator observations bind the entire ordered
  checkpoint plan, target and range. All chunks and all four ranks are required
  before external authorization can release the original full 768-token lease.
- Any failure remains fail-stop. All modified ranges retain local rollback
  buffers; rollback outcomes are recorded per range and never authorize resume.
  Re-entry/duplicate recovery is rejected.
- Existing model/layout/runtime checks remain: Qwen3-8B, plain TP4, single
  request, BF16 NHD, page size 1, no overlap, prefix cache or CUDA graphs.

CPU substitutes for CUDA/IPC in unit tests are **not GPU acceptance**.
Historical M2 records remain readable but never produce `M3_PASS=true`.

## Commands

Run from the repository root. Use the existing environment; do not upgrade
CUDA or PyTorch. Select four independently confirmed idle physical GPUs.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
SHADOW=/tmp/kv-ec-pause-accept-o1pg2zr8
```

Full CPU regression:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m unittest discover \
  -s experiments/kv_ec -p 'test_*.py' -v
```

Minimum GPU smoke (CPU regression first, then only rank 0 / 256):

```bash
"$PY" -u -m experiments.kv_ec.accept_m3 --smoke \
  --gpus 0,1,2,3 --shadow-evidence "$SHADOW" \
  --baseline-port 30359 --pause-port 30360 \
  --startup-timeout 600 --pause-timeout 120
```

Full correctness matrix (not a performance benchmark):

```bash
"$PY" -u -m experiments.kv_ec.accept_m3 \
  --gpus 0,1,2,3 --shadow-evidence "$SHADOW" \
  --baseline-port 30359 --pause-port 30360 \
  --startup-timeout 600 --pause-timeout 120
```

The runner executes full CPU regression, A=(0,256), B=(1,256), C=(0,768),
then the nine remaining combinations. It stops at the first failure;
later combinations remain explicitly NOT_RUN. Every combination owns
separate fresh baseline/server/coordinator/parity processes and evidence.
Only processes created by that acceptance run are cleaned up.

Individual runs retain the requested `accept_pause.py` interface:

```bash
"$PY" -u -m experiments.kv_ec.accept_pause --online-restore \
  --lost-rank 1 --recover-tokens 256 \
  --gpus 0,1,2,3 --shadow-evidence "$SHADOW" \
  --baseline-port 30359 --pause-port 30360 \
  --startup-timeout 600 --pause-timeout 120
```

Run an individual B or C only after the preceding GPU cases have passed.
`--output` optionally specifies a new evidence directory; existing paths
are rejected rather than overwritten.

## Evidence and Timing

Each accepted combination contains:

- `acceptance.json`: verdict, target, token count, chunk count and independent
  target-rank chunk results, including layer-by-layer verification.
- `identity.json`, `checkpoints.json` (all chunks), `checkpoint.json`
  (chunk-0 compatibility), `ranks/pause-rank-{0,1,2,3}.json`, coordinator
  and server logs, separate baseline and pause token responses.
- `performance-online.json`: raw per-chunk protection/parity and recovery
  timings plus client TTFT/TPOT/request duration. Configuration includes
  model/software identities, experiment source hashes, GPU UUIDs, input
  hash, sampling and the specific target/range.

`recovery_ms` sums fault-injection-through-verification chunk intervals,
excluding protection, rollback capture before each fault, external
authorization waits and decode. `fault_to_decode_ms` is a separate observed
end-to-end interval including automatic coordinator/resume/decode latency
and, for multiple chunks, intervening chunk preparation. Do not conflate
it with algorithm-only recovery time. No manual sleep is inserted.

`m3_performance.py` can aggregate accepted raw records with identical configs:

```bash
"$PY" -m experiments.kv_ec.m3_performance /path/to/performance-online.json
```

No raw KV is persisted or uploaded by this acceptance path.

## Execution Status

**M3 real GPU correctness acceptance is complete: 12/12 PASS.**

The authoritative run is `/tmp/kv-ec-m3-matrix-9h4va7t5`, executed on
2026-10-21, approximately 03:35-04:03 UTC. The full CPU regression preceded
the GPU matrix. This supersedes the earlier sandbox-only preflight failure.

| Lost Rank | 256 Tokens | 512 Tokens | 768 Tokens |
| --- | --- | --- | --- |
| 0 | PASS | PASS | PASS |
| 1 | PASS | PASS | PASS |
| 2 | PASS | PASS | PASS |
| 3 | PASS | PASS | PASS |

The M4 read-only audit reruns the existing M1/M3 verifiers on all 12 cases,
checks all four rank records and per-chunk checkpoints, corroborates actual
TP4 CUDA execution with server logs and four GPU UUIDs, and recomputes
the saved performance values from raw rank evidence. All checks pass.
Each case has a distinct transaction, independent baseline response file
and its own baseline server execution. All input hashes, model/software
fingerprints, GPU UUIDs, inference configuration and sampling match.

M2 default rank0/256 shows no regression. M3-A (dynamic rank) and M3-B
(multi-chunk recovery) are hardware-complete for the supported 768-token
single-request workload. CPU fault-injection regressions remain distinct
from successful real GPU acceptance; GPU negative-path tests are not implied.

Evidence and analysis:

```text
/tmp/kv-ec-m3-matrix-9h4va7t5/matrix.json
/tmp/kv-ec-m3-matrix-9h4va7t5/cpu-regression.log
/tmp/kv-ec-m3-matrix-9h4va7t5/rank{0,1,2,3}-tokens{256,512,768}/
/tmp/kv-ec-m4-audit-20261021-final/audit.json
/tmp/kv-ec-m4-audit-20261021-final/performance.csv
/tmp/kv-ec-m4-audit-20261021-final/REPORT.md
```

There are 12 real `performance-online.json` files, **one sample per
configuration**. They are not stable Median/P95 estimates. The audit finds
only JSON/log artifacts, no persisted raw KV. Code uses local Unix/Gloo
and loopback HTTP; this is not a historical network-egress attestation.

M4 can now begin controlled 3-run A/C pilot measurements. Protection-only B
and genuine prefill-retry D are not implemented in the M3 acceptance path.
No complete A/B/C/D comparison or recovery speedup is claimed.
See `M4_PERFORMANCE.md` for metric boundaries, existing measurements,
repeatability commands and the safe retry implementation plan.
