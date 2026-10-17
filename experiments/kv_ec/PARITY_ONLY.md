# CPU DRAM parity-only checkpoint prototype

Baseline inspected: `49e8268495`, clean `main` at the start of development.
This prototype does not implement Node RS, adaptive RS, worker restart, online
KV writeback, or fault-tolerant inference. All functionality is opt-in.

## Source analysis and design

1. `SchedulerBatchResultProcessor.process_batch_result_prefill()` calls the
   debug bridge after forward results are available, before appending the first
   output token, caching an unfinished request, or calling `release_kv_cache()`.
2. Each TP worker uses `processor.model_worker.model_runner.token_to_kv_pool`
   and `processor.req_to_token_pool`, resolves request-local physical slots,
   gathers selected layers and completes a blocking D2H into independently owned
   CPU arrays. No GPU Tensor or slot reference is enqueued.
3. Existing capture blocks the scheduler for device fences, fingerprinting,
   gather/D2H, source rechecks, hashing, serialization and file writes. The
   parity variant still blocks for the safe copy, checks and bounded local queue
   copy, but socket transfer/XOR happen outside the scheduler thread.
4. Separate Python processes cannot share an ordinary list. A private Unix
   socket transports length-prefixed JSON and raw bytes (no pickle). Compared
   with shared memory this adds copies, but avoids shared-memory ownership,
   unlink and crash-cleanup ambiguity in the first correctness prototype.
5. No NCCL/barrier/all-gather is added. Worker `submit()` never waits for another
   rank or for COMMITTED. Socket requests have timeouts in the background client;
   the manager has a bounded listen backlog and processes one connection at a
   time. Missing peers expire independently of inference.
6. `release_kv_cache`, unfinished-cache handling and later allocator reuse can
   invalidate row/slot ownership after the hook returns. The existing lifetime
   guards, source rechecks and blocking copy are preserved. No asynchronous task
   reads the original pool.
7. Snapshot ID is the protection version; IDs must be fresh for every trial.
   Cross-rank identity includes request hash, model hash, source commit/diff,
   token hash/range, layer IDs, TP size, dtype, byte length and packed layout.
   Row generation and physical slots are rank-local, not compared across ranks.

State transitions: PENDING -> COMMITTED only after four unique validated ranks;
duplicate rank, corruption or incompatible metadata -> FAILED; missing rank ->
EXPIRED. Different snapshot IDs remain isolated, never combined. ENQUEUED is
not protection: query the manager for COMMITTED. A timeout/connection loss on
the sender means unknown delivery; never infer commit from a successful enqueue.

## APIs, ownership and bounds

- `ParityManager.submit(metadata, uint8_array)` returns local status. The manager
  copies arrays/metadata; caller mutation after return cannot alter inputs.
- `status(snapshot_id)` returns PENDING/COMMITTED/FAILED/EXPIRED/UNKNOWN.
- `release(snapshot_id)` ends protection and drops parity (in-process API).
- `AsyncParityClient.submit()` queues an immutable CPU bytes copy with
  `put_nowait`; full -> QUEUE_FULL. `close(drain=True)` has a bounded join.
- `submit_once()` is for background transfer/control, never the scheduler.
- CLI `--status ID` queries the live manager. Socket mode is 0600. Existing
  socket paths are refused, not removed or taken over.
- Default per-worker queue depth is 1 plus 1 in-flight message, payload cap is
  64 MiB, default pending checkpoints is 1, terminal checkpoint cap is 16.
  At capacity submissions are rejected. Restart the isolated manager or use
  `release()` between research batches; no unbounded retention is permitted.
- COMMITTED retains parity, shared layout/identity and rank hashes, not complete
  rank buffers. `--test-oracle` temporarily uses pending inputs to recover all
  four erasures and saves booleans only, then releases inputs. Constructor-only
  `retain_oracle=True` is for unit tests, never enabled by the CLI/launcher.
- A manager process exit loses DRAM protection. No durability guarantee exists.

## GPU server manual experiment

Current development sandbox cannot access CUDA or bind Unix sockets. Socket
integration is skipped there, not claimed verified. Run all tests on the GPU
server before the online experiment. Do not kill other users' processes.

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
python -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

Terminal A: create a fresh private socket directory and start the manager.
Copy the printed directory into terminal B/C. Manager parity remains in DRAM
until this process exits. `--test-oracle` adds correctness overhead; omit it in
performance-only trials after correctness is verified.

```bash
PARITY_RUN_DIR=$(mktemp -d /tmp/kv-ec-parity.XXXXXX)
echo "$PARITY_RUN_DIR"
python -m experiments.kv_ec.parity_manager \
  --socket "$PARITY_RUN_DIR/manager.sock" --timeout 30 --test-oracle
```

Terminal B (same Conda/repository): choose MODE=baseline, capture, or parity,
and COUNT=128,256,512. Restart *only your own foreground server* normally with
Ctrl-C between trials. Use the same model, GPUs, prompt and sampling in all
modes; choose a free port. Non-baseline fingerprints are computed once before
workers start, with initialization cost saved separately.

```bash
export PARITY_RUN_DIR=/tmp/kv-ec-parity.REPLACE_WITH_PRINTED_VALUE
export MODE=parity
export COUNT=256
export TRIAL="${MODE}-${COUNT}-$(date -u +%Y%m%dT%H%M%S)"
export EC_OUT="$PWD/experiments/kv_ec/results/parity-online"
mkdir -p "$EC_OUT"
echo "TRIAL=$TRIAL"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
set -o pipefail
set -o noclobber
python -u experiments/kv_ec/launch_debug.py \
  --mode "$MODE" --tp 4 --gpus 0,1,2,3 --port 30287 \
  --model /media/sda1/iic_data/luotao/models/Qwen3-8B \
  --output "$EC_OUT" --snapshot-id "$TRIAL" --layers all \
  --token-count "$COUNT" --parity-socket "$PARITY_RUN_DIR/manager.sock" \
  > "$EC_OUT/$TRIAL-server.log" 2>&1
```

Terminal C: set `TRIAL`, `EC_OUT`, `PARITY_RUN_DIR` to the values above. Read
the server log until ready, verify health, then send two non-target warmups and
the single target request. All modes use the same 768-token artificial prompt,
large enough for every protected range. Token comparison must use matching
baseline/capture/parity trials of the same prompt and configuration.

```bash
curl --noproxy '*' --fail http://127.0.0.1:30287/health
for WARMUP in 1 2; do
  python experiments/kv_ec/request_debug.py request \
    --url http://127.0.0.1:30287 --prompt-tokens 768 --stream-metrics \
    --request-id "warmup-$WARMUP" --output "$EC_OUT/$TRIAL-warmup-$WARMUP.json"
done
python experiments/kv_ec/request_debug.py request \
  --url http://127.0.0.1:30287 --prompt-tokens 768 --stream-metrics \
  --output "$EC_OUT/$TRIAL-response.json"
python -m experiments.kv_ec.parity_manager \
  --socket "$PARITY_RUN_DIR/manager.sock" --status "$TRIAL"
```

Poll status if still PENDING; accept only COMMITTED. For 36 layers/256 tokens,
`parity_bytes` and `steady_parity_bytes` must be 9437184 (9 MiB). With Oracle,
all four `oracle_checks[].byte_exact` must be true. In parity mode there must be
no `rank_N.bin` or `rank_N.json`; only diagnostic timing/error JSON is allowed.
The 128/512-token all-layer parity sizes are 4.5/18 MiB respectively.
Query/save commit only for parity mode; baseline/debug have no manager checkpoint.

Save the final live status in a fresh metadata-only report:

```bash
python -m experiments.kv_ec.parity_manager \
  --socket "$PARITY_RUN_DIR/manager.sock" --status "$TRIAL" \
  --output "$EC_OUT/$TRIAL-commit.json"
```

Repeat each mode/range at least 10 independent fresh-ID trials; one target is
captured only once per worker, so repeated HTTP requests on the same launch
are NOT independent protection measurements. Warmups target no EC and exclude
kernel initialization from target TTFT. Model loading/startup timing remains
separate. The initial protected request still pays validation/hash setup costs.

```bash
python experiments/kv_ec/request_debug.py compare \
  "$EC_OUT/BASELINE_TRIAL-response.json" "$EC_OUT/PARITY_TRIAL-response.json"
python experiments/kv_ec/request_debug.py compare \
  "$EC_OUT/BASELINE_TRIAL-response.json" "$EC_OUT/CAPTURE_TRIAL-response.json"
```

## Measurements and failure evidence

After collecting independent trials of ONE mode/range, use the summarizer to
obtain median/P95 (rank timing is grouped by rank). Do not mix ranges or modes:

```bash
python experiments/kv_ec/summarize_online.py \
  --responses "$EC_OUT"/parity-256-*-response.json \
  --timings "$EC_OUT"/parity-256-*/rank_*.timing.json \
  --commits "$EC_OUT"/parity-256-*-commit.json
```

- Rank `*.timing.json`: model fingerprint lookup, GPU sync, synchronized KV
  gather wall time, blocking D2H, source recheck, local IPC enqueue and
  scheduler/snapshot time. Gather now completes before D2H timing starts.
- `*-initialization.json`: full model fingerprint initialization, excluded from
  per-snapshot steady time. Workers inherit this identity; do not change model
  weights while server is alive.
- Manager status: XOR time, parity commit time (first accepted rank through
  validation/collection/XOR, including Oracle when enabled), Oracle time,
  parity bytes and process peak RSS. RSS includes Python/torch/runtime and
  allocations before commit; it is not the parity payload size.
- Background client logs: IPC transfer/response time, submission failure.
- Request JSON: client streaming TTFT, amortized TPOT and total latency. This is
  HTTP/SSE client latency, not CUDA kernel timing. Compare exact output IDs.
- Debug files: `disk_write_ms` includes bin+JSON; final values are saved in the
  separate timing file after writes complete. Timing file's own diagnostic
  write is outside the saved measurement but adds small scheduler overhead.
- Protection costs: gather, D2H, IPC and XOR. Experimental checks: source
  rechecks, SHA256, Oracle, full GPU fences/timing instrumentation. Full rank
  disk I/O is debug-only; parity mode has only diagnostic file I/O.
- 256-token nominal steady payload is 9 MiB plus metadata. Pending inputs are
  up to 36 MiB; parity adds 9 MiB. Transport, validation and worker queue copies
  add transient memory; do not report 9 MiB as process peak or total system RSS.

Failures: QUEUE_FULL, PENDING_CAPACITY, DUPLICATE_RANK, METADATA_MISMATCH,
ValueError/checksum failure, EXPIRED/TIMEOUT, and IPC connection errors are not
protection. Check server logs for `KV_EC parity`, manager status, and
`rank_N.error.json`. No socket cleanup command should remove another manager's
path. Manager shutdown ends all protection, even if an old saved status says
COMMITTED.

Next online writeback needs allocator-owned slot epochs/leases, a request pause
protocol, model/version checks, new allocation and mapping transaction, stream
completion events, and all-rank coordinated resume/rollback. A parity checkpoint
alone does not provide these guarantees. No Node RS work is started here.

## Observed CPU-only replay (2026-10-17)

The development session loaded existing rank files read-only with `assemble()`,
then directly submitted their owned arrays to a new `ParityManager(test_oracle=True)`.
This is a CPU replay, NOT evidence of online IPC or GPU capture. For each source
there were two warmups and ten fresh-manager measured trials, with no file I/O
inside the measured XOR stage. All four Oracle recoveries passed, state was
COMMITTED, and pending rank buffers were empty after each commit.

| Existing evidence | Parity bytes | XOR median / P95 ms |
|---|---:|---:|
| `tp4-l0-128-20261016T141855` | 131072 | 0.0396 / 0.0407 |
| `tp4-all-256-20261016T142603` | 9437184 | 7.8008 / 8.0442 |

The process-lifetime peak RSS at the ends of these groups was 526327808 and
631783424 bytes respectively, including interpreter/torch, the external replay
input arrays and validation/oracle temporaries. These are NOT measurements of
the standalone socket service or the total Worker+Manager CPU memory footprint.
GPU/IPC latency, online request TTFT/TPOT and the 512-token online experiment
remain unmeasured here.
