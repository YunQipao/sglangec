# Three-Chunk Debug Capture and Scratch Restore

This workflow is explicitly opt-in and offline. It does not attach to a live
pool, erase KV, install recovered KV, or resume a request. Scratch success is
not online recovery success.

## Data Contract

`launch_debug --mode capture --chunk-size 256 --token-count 256 --layers all`
captures logical ranges `[0,256)`, `[256,512)`, `[512,768)` from one artificial
768-token request. Each TP4 worker writes `rank_N.bin` plus `rank_N.json`,
`rank_N.timing.json`, and `rank_N.diagnostic.json` in each chunk directory.
Writes are exclusive. Sidecars contain actual model/source identity, request
hash, logical token hash, physical-slot metadata, layer layout, and byte hashes.

Capture alone does **not** create `manifest.json`. `prepare_scratch` waits a
bounded time for all files, rejects worker errors, calls `rank_snapshot.assemble`
on actual bytes, checks all three chunks against request/startup evidence, and
then calls `rank_snapshot.verify` to validate all four CPU erasure positions and
write derived manifests/recovery reports. No metadata or checksum is invented.
An interrupted preparation may leave exclusive output files; use a fresh run
rather than deleting/replacing historical evidence.

`FAST_PARITY` intentionally saves diagnostics, not rank payloads. Its historical
COMMITTED/oracle/global-release evidence cannot be converted to offline rank
snapshots. This workflow does not change that path or add disk writes to it.
Debug snapshots are complete offline files, not Parity Manager COMMITTED records.

## Terminal A: Foreground Capture Server

First inspect GPU availability using `nvidia-smi`. Use four GPUs allocated to
this experiment; never stop another user's process. The defaults below use
0,1,2,3 and a candidate port that the launcher checks without taking over.

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
umask 077
export RUN="$(mktemp -d /tmp/kv-ec-scratch.XXXXXXXX)"
export OUT="$RUN/evidence"
export SNAP="scratch-$(date -u +%Y%m%dT%H%M%S)-${RUN##*.}"
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export PORT=30358
export REQUEST_ID="kv-ec-scratch-${RUN##*.}"
mkdir "$OUT"
printf 'export RUN=%q OUT=%q SNAP=%q MODEL=%q PORT=%q REQUEST_ID=%q\n' \
  "$RUN" "$OUT" "$SNAP" "$MODEL" "$PORT" "$REQUEST_ID" > "$RUN/session.env"
printf 'Session: %s/session.env\n' "$RUN"

# Isolated debug capture, not the separate lease/control experiment.
unset SGLANG_KV_EC_PREPARED_IDENTITY SGLANG_KV_EC_PARITY_SOCKET
unset SGLANG_KV_EC_CONTROL_SOCKET
export SGLANG_KV_EC_SCHEDULER_LEASE=0
export SGLANG_KV_EC_CROSS_RANK_LEASE=0
export SGLANG_KV_EC_RECOVERY_GUARD=0
export SGLANG_KV_EC_DEFER_CHECKSUMS=0
export SGLANG_KV_EC_REQUEST_VERSION="$SNAP-request-v1"
export SGLANG_KV_EC_CHECKPOINT_VERSION="$SNAP"
export SGLANG_KV_EC_MANAGER_EPOCH=local-scheduler-epoch
set -o pipefail
python -m experiments.kv_ec.launch_debug \
  --mode capture --protection-mode DEBUG_SAFE --transfer pinned \
  --tp 4 --gpus 0,1,2,3 --port "$PORT" --model "$MODEL" \
  --output "$OUT" --snapshot-id "$SNAP" --request-id "$REQUEST_ID" \
  --token-count 256 --chunk-size 256 --layers all \
  2>&1 | tee "$RUN/server.log"
```

This command stays in the foreground. The model path must contain `config.json`
and the actual `*.safetensors` shards, not just a cache parent directory.

## Terminal B: Request, Prepare, Restore, Accept

Source the exact session path printed by Terminal A, not an old experiment.
Wait for server readiness before sending the request.

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /tmp/kv-ec-scratch.REPLACE_WITH_ACTUAL/session.env
curl --noproxy '*' --fail "http://127.0.0.1:$PORT/health"

python -m experiments.kv_ec.request_debug request \
  --url "http://127.0.0.1:$PORT" --model "$MODEL" \
  --request-id "$REQUEST_ID" --prompt-tokens 768 --stream-metrics \
  --output "$OUT/$SNAP-response.json"

python -m experiments.kv_ec.prepare_scratch \
  "$OUT/$SNAP-chunk-1" "$OUT/$SNAP-chunk-2" "$OUT/$SNAP-chunk-3" \
  --request-evidence "$OUT/$SNAP-response.json" \
  --initialization "$OUT/$SNAP-initialization.json" --wait-s 30
```

After the request and preparation succeed, stop **only this foreground server**
with Ctrl-C in Terminal A to release its GPU allocations. This is not a command
to terminate unrelated GPU jobs. Check `nvidia-smi` again; select an available
GPU for the isolated scratch process.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.kv_ec.gpu_scratch_restore \
  "$OUT/$SNAP-chunk-1" "$OUT/$SNAP-chunk-2" "$OUT/$SNAP-chunk-3" \
  --device cuda --warmup 2 --repeat 10 \
  --output "$OUT/$SNAP-scratch-sequence.json"

python -m experiments.kv_ec.accept_scratch \
  "$OUT/$SNAP-scratch-sequence.json" \
  --request-evidence "$OUT/$SNAP-response.json" \
  --initialization "$OUT/$SNAP-initialization.json"
```

Acceptance requires exactly 12 distinct chunk/lost-rank cases, 10 measurements
each, byte-exact recovery, matching payload digests, full 36-layer TP4 K/V NHD
layout `[256,2,128]`, contiguous coverage starting at zero, matching
model/request/source identity, and `online_kv_pool_modified=false`. It reloads
the actual input bytes; keep the rank files until acceptance finishes.
Request/startup evidence and provenance declarations are cross-checked local
evidence, not independent attestation of an untrusted report or server.

## Measurements and Lifetime

Each `chunks[i].ranks[j]` entry contains `median_ms`, `p95_ms`, and all 10
`samples_ms` for:

| Field | Scope |
|---|---|
| `xor_ms` | CPU parity encoding from this chunk's rank payloads |
| `cpu_recovery_ms` | CPU XOR decoding excluding the missing rank |
| `h2d_restore_ms` | Scratch allocation, raw-byte H2D, synchronization |
| `gpu_verification_ms` | D2H plus exact byte comparison, not a GPU-only kernel |

CPU hash/structural validation and disk loading are outside those timed regions.
Scratch restoration is sequential by layer and erasure case, not an atomic
online write or one contiguous 768-token GPU allocation. All bytes of each
chunk are checked; contiguous range/identity checks establish full 768-token
coverage. These are offline measurements, not serving latency or A/B samples.
CPU parity is regenerated for each case, not loaded from a live Manager.

Only one chunk's original payloads are held at a time. Scratch tensors are
released after each case, and all CPU/GPU owners are process-local. Persisted
raw snapshots deliberately remain for acceptance; retain them only until the
experiment audit ends, then remove **only this new run's** private evidence
under your data-retention policy. Do not upload raw KV; `rank_*.bin` is ignored
by Git for new untracked files. This does not untrack historical artifacts.

## Failures and CPU Regression

Missing/partial payloads, error sidecars, mixed identities, corrupt hashes,
deferred checksums, stale manifest rank metadata, layout mismatch, incomplete
coverage, or existing output files are rejected. CUDA unavailability raises an
error without a CPU fallback. A failed copy/verification raises before a
success report is written; a partial JSON is not accepted. Do not rename timing
JSON to manifest or extend timeouts to bypass validation.

```bash
python -m unittest \
  experiments.kv_ec.test_gpu_scratch_restore \
  experiments.kv_ec.test_scratch_pipeline -v
python -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
git diff --check
```

The new synthetic CPU pipeline test performs real assembly/checksum validation
and all 120 CPU erasure recoveries but mocks GPU copying. It is not GPU evidence.
Actual GPU Median/P95 results must come from the new saved scratch report.
