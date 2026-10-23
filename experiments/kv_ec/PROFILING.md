# Online XOR timing v2 and transfer experiments

No scheduler core changes, KV writes, RS or early asynchronous return. Source
slots stay owned by the stopped scheduler until copying and checks complete.
Hashes/metadata validation and manager commit rules are unchanged. This change
measures existing costs; it does not claim a performance improvement.

## What changed

`rank_N.timing.json` uses `timing_schema=kv-ec-exclusive-wall-v2`:

- `wall_regions_ms`: non-nested CPU wall time. Sum ONLY these regions;
  compatibility fields such as `sha256_ms`, `d2h_ms` and `ipc_enqueue_ms` are
  summaries/aliases, not extra regions. `accounted_ms + unaccounted_ms` equals
  `scheduler_capture_ms` (also the legacy `scheduler_blocking_ms`).
- Separate pool getter/layout validation, gather calls, gather fence,
  pinned allocation, D2H copy calls, D2H fence, numpy chunk copies, per-segment
  SHA, K/V aggregate SHA, payload SHA, request/token SHA, layer metadata,
  final request generation check, final slot mapping check, GPU properties,
  client import/lookup, first thread/client initialization, IPC enqueue and
  host/tensor cleanup. Residual covers timing instrumentation and remaining
  Python bookkeeping; it is not guessed or silently folded into a phase.
- `--profile-detail`: per-layer regions plus scheduler-thread CPU time. Compare
  layer 0 with later layers and the first protected request with later ones.
  Wall minus thread CPU can include GPU waits, OS scheduling, locks, and work
  performed by other threads; it is NOT a CPU-contention measurement by itself.
- `--cuda-events`: optional elapsed intervals on the current CUDA stream for
  gather and D2H. They include stream elapsed time and may include idle gaps
  between submissions, especially blocking copies; they are NOT guaranteed
  pure kernel durations and must NEVER be added to wall regions.
- `--cpu-profile`: cProfile only around capture, creates `rank_N.cpu.prof`.
  It contains function statistics, not tensor/request contents. Profiling and
  profile serialization perturb the request: exclude these runs from formal
  latency comparisons. Stats file writing happens after capture timing.
- `rank_N.diagnostic.json` reports the actual timing-JSON mkdir/write time and
  info-log time. `scheduler_through_diagnostics_ms` includes those operations.
  The diagnostic record's own write is explicitly excluded (a timing record
  cannot include its own completed write). `diagnostic_write_ms` in the main
  timing record is null, not a fabricated zero. Timings also exclude hook entry,
  final function-return overhead, and CPU-profiler serialization if enabled;
  no field claims to measure the entire request or exact full hook duration.
- `ipc_enqueue_details` separately measures owned payload copy, metadata
  deepcopy and `queue.put_nowait`; attempted copies count even when queue full.
  JSON encoding happens in the background socket sender, not in enqueue.
  Server logs report background connect/header encode/send/payload send/response
  wait timings. Response wait includes manager work and queued connections;
  these are NOT additional scheduler blocking time.

`--transfer pinned` (existing default) explicitly allocates two pinned buffers
once per capture and reuses them across layers. It still copies each layer to
owned numpy storage. `--transfer blocking` selects the old blocking CPU transfer
and numpy copy path, with the same completion fences. No silent fallback:
pinned allocation failure rejects this capture and inference continues without
protection. This makes A/B results attributable to the selected path.

DEBUG_SAFE still hashes the payload three times (segments, aggregate K/V,
whole payload) in the worker. FAST_PARITY computes one complete payload SHA in
the worker, which is the transmission-integrity commitment, and marks the
segment/K/V aggregate hashes as `checksum_deferred`. The Manager computes and
checks those remaining hashes after it owns its immutable CPU copy, before it
accepts a rank or can COMMIT. This moves two scans out of the scheduler hot
path; it does not remove checksum validation. `sha256_scanned_bytes` reports
worker scans only, and `checksum_location` identifies the split. A corrupt
payload still fails against the worker's payload SHA.
For an explicit A/B replay, set `SGLANG_KV_EC_DEFER_CHECKSUMS=0` to retain the
pre-optimization worker-side three-scan path, or `=1` for the optimized path;
the default is `1`. This switch is experimental and does not disable checks.

`--ec-warmups 2 --ec-repeat 10` enables a bounded explicit request plan:
`RID-ecwarmup-1/2`, then `RID-measure-1..10`; snapshot IDs use the same suffixes
under the supplied base ID. Each snapshot is unique. No wildcard request matching,
no retry after capture failure, no mutable core scheduler fields. Total planned
captures must be <=16 (manager checkpoint cap). Plans are opt-in; defaults
preserve the old single-target request/snapshot behavior. All EC remains off
unless explicitly enabled. A second capture warms client/module paths, but
pinned buffers are still allocated PER capture; distinguish these costs.

## Local CPU regression

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
python -m py_compile experiments/kv_ec/{debug_capture,capture_timing,capture_plan,checkpoint_protocol,parity_manager,launch_debug,profile_requests}.py
python -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
```

New CPU-backed capture tests mock CUDA and verify both transfer paths, both
protection modes, sync dependencies, complete payload validation, source
immutability, accounting balance, no silent pinned failure, explicit plan
matching, separate event metrics and readable cProfile output. They are NOT
real GPU tests. Existing TP=2/4 and real snapshot CPU regressions remain required.

Local verification on 2026-10-19: 62 tests ran successfully, with two Unix
socket subtests skipped because the tool sandbox forbids binding sockets.
Compilation and `git diff --check` passed. CUDA was unavailable; real socket,
GPU correctness and latency results are still pending server verification.
Manager-embedded capture metadata is queued before timing finalization; use
the per-rank timing/diagnostic files as the authoritative capture timings.

## GPU experiments: one case at a time, three terminals

Use CASE=`baseline`, then `pinned`, then `blocking`, then optionally `diagnostic`.
Each case has a NEW run/socket/snapshot. Stop only your own foreground B and A
between cases; wait for their workers to exit. Never run the cases concurrently.
The current tool sandbox has no CUDA and cannot bind Unix sockets, so all real
GPU/IPC results below must be collected on the server.

Start with a fresh study directory in A:

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export BENCH=$(mktemp -d /tmp/kv-ec-study.XXXXXX)
printf 'Keep this study directory for all cases: %s\n' "$BENCH"
```

For each case in A, keep BENCH pointing to that same actual directory. For a
later terminal/session, set `export BENCH=/tmp/kv-ec-study.ACTUAL` first.
Use TP=4 to diagnose the previous four-GPU result; TP=2 is a separate study:
change TP=2 and GPUS=1,2 consistently in a NEW study directory.

```bash
export CASE=baseline
# Later cases: export CASE=pinned / blocking / diagnostic
export TP=4
export GPUS=0,1,2,3
export COUNT=256
export PORT=30327
export RID=kv-ec-artificial-request
export EC_WARMUPS=2
export REPEAT=10
export MODE=parity
export TRANSFER=pinned
if [ "$CASE" = baseline ]; then export MODE=baseline; fi
if [ "$CASE" = blocking ]; then export TRANSFER=blocking; fi
if [ "$CASE" = diagnostic ]; then export REPEAT=1; fi
nvidia-smi
ss -lntp | grep ':30327' || true
export RUN=$(mktemp -d "$BENCH/$CASE.XXXXXX")
export OUT="$RUN/evidence"
export SNAP="$CASE-$(date -u +%Y%m%dT%H%M%S)-${RUN##*.}"
mkdir "$OUT"
declare -p BENCH CASE TP GPUS COUNT PORT RID EC_WARMUPS REPEAT MODE TRANSFER RUN OUT SNAP > "$RUN/session.env"
printf 'In B/C: source %s/session.env\n' "$RUN"
python -m experiments.kv_ec.parity_manager --socket "$RUN/manager.sock" \
  --timeout 60 --test-oracle 2>&1 | tee "$RUN/manager.log"
```

The manager is unused for baseline; keeping the same CLI setup is harmless.
Oracle is enabled for correctness in every protected case. Its cost must be
reported separately; do not mix Oracle-on and Oracle-off latency samples.

In B, activate the same Conda environment and repository, source the ACTUAL
session path printed by A, then:

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
source /tmp/kv-ec-study.ACTUAL/CASE.ACTUAL/session.env
EXTRA=()
if [ "$MODE" = parity ]; then EXTRA+=(--parity-socket "$RUN/manager.sock"); fi
if [ "$CASE" = diagnostic ]; then EXTRA+=(--profile-detail --cuda-events --cpu-profile); fi
python -u -m experiments.kv_ec.launch_debug \
  --mode "$MODE" --protection-mode FAST_PARITY --transfer "$TRANSFER" \
  --tp "$TP" --gpus "$GPUS" --port "$PORT" \
  --model /media/sda1/iic_data/luotao/models/Qwen3-8B \
  --output "$OUT" --snapshot-id "$SNAP" --request-id "$RID" \
  --token-count "$COUNT" --layers all --ec-warmups "$EC_WARMUPS" --ec-repeat "$REPEAT" \
  "${EXTRA[@]}" 2>&1 | tee "$RUN/server.log"
```

In C, after readiness, source the same session. For baseline, save its paths for
subsequent token comparisons. For later cases replace BASE_OUT/BASE_SNAP with
the baseline values printed by this block (do not reuse baseline snapshot IDs).

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
source /tmp/kv-ec-study.ACTUAL/CASE.ACTUAL/session.env
curl --noproxy '*' --fail "http://127.0.0.1:$PORT/health"
EXTRA=()
if [ "$MODE" = parity ]; then
  export BASE_OUT=/tmp/kv-ec-study.ACTUAL/baseline.ACTUAL/evidence
  export BASE_SNAP=baseline-ACTUAL-SNAPSHOT
  EXTRA+=(--socket "$RUN/manager.sock" --compare-dir "$BASE_OUT" --compare-snapshot-id "$BASE_SNAP")
else
  printf 'For later cases: BASE_OUT=%s BASE_SNAP=%s\n' "$OUT" "$SNAP"
fi
python -m experiments.kv_ec.profile_requests \
  --url "http://127.0.0.1:$PORT" --output "$OUT" --snapshot-id "$SNAP" \
  --request-id "$RID" --tp "$TP" --token-count "$COUNT" --transfer "$TRANSFER" \
  --model-warmups 2 --ec-warmups "$EC_WARMUPS" --ec-repeat "$REPEAT" \
  "${EXTRA[@]}" 2>&1 | tee "$RUN/validation.log"
```

The runner uses fixed 768-token artificial input, temperature=0 and 16 outputs.
It separates two MODEL warmups (not protected), then EC warmups (protected),
then measurements (protected); each EC snapshot has a fresh ID. It validates
live COMMITTED, each erasure Oracle, byte sizes/shapes, output IDs against the
baseline, every rank's timing balance, selected transfer/mode, and no rank files.
Failures abort; only a fully validated run produces a success summary. Never
interpret ENQUEUED or an old saved commit file as current protection.

Each case retains up to 12 parity blocks (108 MiB TP=4/256 tokens; 216 MiB TP=2),
not 12 sets of sources. These are process-lifetime checkpoints, not persistent
KV backups; the process RSS also includes Python/torch, temporary work and
allocator retention. Fresh manager per case avoids capacity/retention ambiguity.

For 128/512-token studies use new study/run/snapshot IDs, change COUNT in A,
and rerun all cases with the same 768-token input. Do not overwrite old evidence.

## Inspect results in C

```bash
# Current case's validated summary (contains all first/warm/measurement timings):
python -m json.tool "$OUT/$SNAP-profile-summary.json"
rg 'KV_EC IPC background phases' "$RUN/server.log"

# Optional diagnostic case only; function tottime/cumtime in seconds:
python - <<'PY'
import os, pstats
from pathlib import Path
root = Path(os.environ['OUT'])
for path in sorted(root.glob('*/rank_*.cpu.prof')):
    print('\nPROFILE', path)
    pstats.Stats(str(path)).strip_dirs().sort_stats('cumulative').print_stats(25)
PY
```

Inspect `client_import_lookup_ms`, `client_initialization_ms`,
`layer_getter_validation_ms`, `final_slot_mapping_check_ms`, and the three
separate hash regions first. Look at `host_buffer_allocation_ms`,
`numpy_chunk_copy_ms`, `host_buffer_release_ms` and per-layer wall/thread CPU
next. cProfile does not split a C-extension call into GPU/wait/allocation work;
use Event data and optional external process/GPU monitoring to investigate.
An unexplained residual remains an open issue, not proof of any specific cause.

No-detail pinned and blocking are the latency A/B; diagnostic is explanatory
only. Request helper includes HTTP/SSE overhead. Nearest-rank P95 and median
exclude EC/model warmups; first protected request is retained separately as
ecwarmup-1. Ten same-process measurements are not ten independent process starts.
For independent evidence, repeat complete baseline/pinned/blocking rounds with
new study dirs, rotate order (e.g. B/P/K then K/B/P), confirm other CPU/GPU jobs
are not interfering, and record the environment. Never stop others' jobs.

Next optimize only after the new traces show where time goes. A future packing
buffer or background-hash change should be one separate A/B with unchanged
integrity and Oracle tests. No removal of completion fences is authorized by
this profiling workflow.
