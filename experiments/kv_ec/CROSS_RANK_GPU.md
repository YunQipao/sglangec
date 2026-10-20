# Non-destructive TP4 control-plane validation

This run does not erase or write GPU KV. All four GPUs must be available to
you; do not terminate any existing service. The launcher enforces page_size=1,
one running request, no radix cache, no overlap and no chunked prefill.
The CPU suite must pass first. Real Unix socket tests must not be skipped on
the GPU server:

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
python -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
git diff --check
```

## Terminal A: new Manager

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
export RUN=$(mktemp -d /tmp/kv-ec-cross-rank.XXXXXX)
export OUT="$RUN/evidence"
export SNAP="cross-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"
declare -p RUN OUT SNAP > "$RUN/session.env"
echo "SESSION=$RUN/session.env"
set -o pipefail
python -u -m experiments.kv_ec.parity_manager \
  --socket "$RUN/manager.sock" --timeout 60 --max-pending 3 --test-oracle \
  2>&1 | tee "$RUN/manager.log"
```

## Terminal B: independent Coordinator

Replace the session path below and in every following terminal with the
actual path printed by Terminal A. Do not source a directory.

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /tmp/kv-ec-cross-rank.ACTUAL/session.env
set -o pipefail
python -u -m experiments.kv_ec.cross_rank_control \
  --socket "$RUN/control.sock" --parity-socket "$RUN/manager.sock" \
  --timeout 30 2>&1 | tee "$RUN/control.log"
```

## Terminal C: isolated SGLang

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /tmp/kv-ec-cross-rank.ACTUAL/session.env
nvidia-smi
ss -lntp | rg ':30357' || true
```

Continue only after confirming GPU0–3 and port 30357 are free.

```bash
export SGLANG_KV_EC_RECOVERY_GUARD=1
export SGLANG_KV_EC_SCHEDULER_LEASE=1
export SGLANG_KV_EC_CROSS_RANK_LEASE=1
export SGLANG_KV_EC_CONTROL_SOCKET="$RUN/control.sock"
export SGLANG_KV_EC_CROSS_RANK_TIMEOUT_S=20
export SGLANG_KV_EC_LEASE_TIMEOUT_S=45
export SGLANG_KV_EC_DEFER_CHECKSUMS=1
export SGLANG_KV_EC_REQUEST_VERSION="$SNAP-request-v1"
export SGLANG_KV_EC_CHECKPOINT_VERSION="$SNAP"
set -o pipefail
python -u -m experiments.kv_ec.launch_debug \
  --mode parity --protection-mode FAST_PARITY --transfer pinned \
  --tp 4 --gpus 0,1,2,3 --port 30357 \
  --model /media/sda1/iic_data/luotao/models/Qwen3-8B \
  --output "$OUT" --snapshot-id "$SNAP" \
  --request-id kv-ec-artificial-request \
  --token-count 256 --chunk-size 256 --layers all \
  --parity-socket "$RUN/manager.sock" \
  2>&1 | tee "$RUN/server.log"
```

The launcher queries the live Manager epoch. Never use the historical static
`local-scheduler-epoch` label for a new run.

## Terminal D: request and live validation

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /tmp/kv-ec-cross-rank.ACTUAL/session.env
curl --noproxy '*' --fail http://127.0.0.1:30357/health
for i in 1 2; do
  python -m experiments.kv_ec.request_debug request \
    --url http://127.0.0.1:30357 --request-id "warmup-$i" \
    --prompt-tokens 768 --stream-metrics --output "$OUT/warmup-$i.json"
done
python -m experiments.kv_ec.request_debug request \
  --url http://127.0.0.1:30357 --request-id kv-ec-artificial-request \
  --prompt-tokens 768 --stream-metrics --output "$OUT/$SNAP-response.json"
export MANAGER_EPOCH=$(python -c 'import os; from experiments.kv_ec.parity_manager import manager_hello; print(manager_hello(os.environ["RUN"] + "/manager.sock"))')
python -m experiments.kv_ec.complete_prefill \
  --socket "$RUN/manager.sock" --base-snapshot "$SNAP" \
  --request-id kv-ec-artificial-request \
  --request-version "$SNAP-request-v1" --checkpoint-version "$SNAP" \
  --manager-epoch "$MANAGER_EPOCH" --output "$OUT/$SNAP-complete.json"
```

If Manager says PENDING, wait briefly and retry the last command; never relax
COMMITTED/Oracle checks. ERROR/FAILED/EXPIRED require diagnosis, not retries
with reused Snapshot IDs.

```bash
python - <<'PY'
import json, os
from pathlib import Path
out, snap = Path(os.environ['OUT']), os.environ['SNAP']
report = json.loads((out / f'{snap}-complete.json').read_text())
assert report['state'] == 'COMPLETE'
assert sum(len(c['oracle_checks']) for c in report['chunks']) == 12
transactions = set()
for chunk in range(1, 4):
    ids = set()
    for rank in range(4):
        p = out / f'{snap}-chunk-{chunk}' / f'rank_{rank}.timing.json'
        perf = json.loads(p.read_text())['performance_ms']
        lease = perf['scheduler_lease']
        assert lease['state'] == 'LOCAL_RELEASED'
        assert lease['event_recorded'] and lease['event_complete']
        assert lease['global_release_confirmed'] is True
        assert lease['coordinator_state'] == 'GLOBAL_RELEASED'
        assert lease['coordinator_released_ranks'] == [0, 1, 2, 3]
        assert lease['slot_generations_before'] == lease['slot_generations_after']
        assert lease['slot_count'] == 256
        assert lease['manager_epoch'] == os.environ['MANAGER_EPOCH']
        assert lease['writeback_supported'] is False
        ids.add(lease['transaction_id'])
        print(chunk, rank, lease['state'], lease['control_elapsed_ms'],
              lease['allocator_available_size_after'])
    assert len(ids) == 1
    transactions.update(ids)
assert len(transactions) == 3
print('PASS: 3 chunks, 12 Oracle checks, 3 four-rank release transactions')
PY
```

Compare with an existing **same-model, same 768-token artificial input**
Baseline response; supply its actual file path:

```bash
python -m experiments.kv_ec.request_debug compare \
  /actual/baseline-response.json "$OUT/$SNAP-response.json"
```

Evidence: `manager.log`, `control.log`, `server.log`, `evidence/*-response.json`,
`evidence/*-complete.json`, and each Chunk's four `rank_*.timing.json` files.
Stop only the services started for this run with Ctrl+C, after validation.

## Semantics and limits

```text
Prefill result hook (before token append/cache release)
  -> capture Chunk -> local allocator lease
  -> prepare_rank[4] -> local_leased[4] -> ALL_RANKS_READY
  -> Worker Event synchronize/query + mapping/generation validation
  -> cuda_completed[4] -> validated[4] -> RELEASE_AUTHORIZED
  -> actual local release -> local_released ACK[4] -> GLOBAL_RELEASED
  -> enqueue immutable CPU Snapshot -> repeat next Chunk -> normal Decode
```

The three Chunks have three separate transactions, not one atomic 768-token
recovery transaction. Parity COMMITTED happens asynchronously **after** the
lease verification; GLOBAL_RELEASED does not imply Parity COMMITTED. COMPLETE
means Manager owns all three parity checkpoints, not that live GPU slots are
still valid. Manager loss invalidates its in-memory parity. Request completion
can free all original GPU slots.

Control IPC runs synchronously in the isolated non-overlap result hook, outside
NCCL calls. Socket waits are bounded but add Scheduler/TTFT blocking. CUDA
synchronization itself has no application timeout and can hang on driver failure;
bounded IPC is not a proof of deadlock-free serving under Worker/GPU loss.
If a rank fails, no global success is reported; local non-destructive abort
requires a genuine CUDA fence. An unreachable/failed Worker can retain slots.
Partial local release after authorization is possible; there is no durable
distributed recovery/rollback log. Socket permissions assume trusted processes
under the same Unix account, not Byzantine participants.

Run a fresh, separate three-Chunk case with only `SGLANG_KV_EC_CROSS_RANK_LEASE=0`
to quantify control overhead. Keep local Lease/Guard, input, GPUs and warmups
identical. Use fresh Managers, run dirs and Snapshot IDs in both cases; never
reuse the target request in one process. Collect at least ten independent cases
for Median/P95. Compare response `client_metrics_ms` and each rank's
`scheduler_capture_ms`, `scheduler_lease.control_elapsed_ms`. Oracle timings
remain correctness overhead, not production protection latency. No new GPU
latency result is claimed by this document.
