# TP=2 online XOR verification (research only)

This flow uses artificial input and never writes a running KV pool. CPU Oracle
verification is transient inside the manager; it is not GPU recovery or continued
inference after a failure. All EC remains disabled by default. No core scheduler,
CUDA, PyTorch or Conda changes are required.

## Implementation and capacity

The launcher accepts TP=1/2/4 for baseline/capture, TP=2/4 for parity.
`--protection-mode FAST_PARITY` is the parity default; `DEBUG_SAFE` can be selected
with the same parity socket, without writing source rank files. The latter adds
the full selected-source recheck. Both retain lifetime guards, request generation
and slot mapping checks, synchronous copy completion and full payload checksums.
FAST does not re-read the source to prove immutability; it relies on the read-only
capture operations and the stopped non-overlap scheduler's ownership. No slot
lease or early scheduler return has been implemented.

The manager validates snapshot `format=kv-ec-rank-bin-v1`, payload hashes and
matching identities/layouts. TP=2 requires exactly ranks {0,1}; TP=4 requires
{0,1,2,3}. Mixed TP, versions, layouts, models, requests or ranges fail closed.
After COMMITTED, only one parity block and metadata remain (unless the explicit
in-process `retain_oracle` test option is used; the CLI below does NOT use it).
A fresh manager has no historical checkpoints: status returns UNKNOWN. Saved
commit JSON is evidence, not live protection; query the running manager again.

For 36 layers, BF16, head_dim=128:

| TP | KV heads/rank | Protected tokens | Rank bytes = parity bytes |
|---|---|---|---|
| 2 | 4 | 128 | 9437184 |
| 2 | 4 | 256 | 18874368 |
| 2 | 4 | 512 | 37748736 |
| 4 | 2 | 128 | 4718592 |
| 4 | 2 | 256 | 9437184 |
| 4 | 2 | 512 | 18874368 |

Formula: layers × tokens × (8/TP) × 128 × 2 BF16 bytes × 2 K/V.
These are theoretical until checked against runtime evidence. Capture checks
actual GPU K/V dimensions, dtype, device and contiguity before copying each
layer. The manager independently checks packed shapes `[tokens, 8/TP, 128]`,
canonical offsets, byte counts and hashes. COMMITTED JSON retains the checked
layer records and rank hashes; the commands below assert the actual byte count.

## CPU regression

Local validation on 2026-10-19 using
`/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python`: 51 tests,
`OK (skipped=2)` in 10.376 seconds. The two skipped subtests are TP=2/TP=4
Unix socket binding, forbidden by this sandbox. Existing real TP=4 snapshot
CPU validation passed. PyTorch reports CUDA unavailable and `/dev/nvidia*`
is absent: no real TP=2 online capture, GPU shape measurement, output-token
comparison or GPU scratch execution is claimed. All shell blocks below and
the embedded validation Python were syntax checked.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
python -m py_compile experiments/kv_ec/{launch_debug,debug_capture,parity_manager,rank_snapshot,gpu_scratch_restore,checkpoint_protocol}.py
python -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
```

CPU-generated TP=2/4 fixtures cover 128/256/512 tokens and 36 layers, all
single-rank erasures, exact byte capacities, rejected mixed identities/versions,
missing-rank expiry, duplicates, corruption, release and fresh-manager UNKNOWN.
Scratch tests check TP=2 dispatch with mocked H2D, not GPU execution. The existing
real TP=4 disk snapshot regression is also retained. Unix socket tests may be
skipped by a restricted sandbox; run them on the GPU server too.

## Terminal A — fresh CPU manager

Check GPU availability first. Do not run on occupied GPUs or terminate other
jobs. If ports 30292/30293 are occupied, choose two unused ports consistently.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
nvidia-smi
export RUN=$(mktemp -d /tmp/kv-ec-tp2.XXXXXX)
export OUT="$RUN/evidence"
export SNAP="tp2-fast-$(date -u +%Y%m%dT%H%M%S)-${RUN##*.}"
export RID="kv-ec-artificial-request"
export COUNT=256
export PORT=30292
export BASE_PORT=30293
mkdir "$OUT"
declare -p RUN OUT SNAP RID COUNT PORT BASE_PORT > "$RUN/session.env"
printf 'In B and C run: source %s/session.env\n' "$RUN"
printf 'Socket=%s/manager.sock Snapshot=%s Port=%s\n' "$RUN" "$SNAP" "$PORT"
python -m experiments.kv_ec.parity_manager \
  --socket "$RUN/manager.sock" --timeout 60 --test-oracle \
  2>&1 | tee "$RUN/manager.log"
```

Keep A running while validating protection. `--test-oracle` checks every lost
rank transiently and releases the original blocks; it does not retain sources.
To test 128 or 512 tokens, start a NEW run directory and snapshot with COUNT
changed above. Keep all inference/request parameters the same.

## Terminal B — TP=2 FAST_PARITY

Replace the session path below with the one printed by A.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
source /tmp/kv-ec-tp2.REPLACE/session.env
python -m experiments.kv_ec.launch_debug \
  --mode parity --protection-mode FAST_PARITY --tp 2 --gpus 1,2 \
  --model /media/sda1/iic_data/luotao/models/Qwen3-8B \
  --port "$PORT" --output "$OUT" --snapshot-id "$SNAP" --request-id "$RID" \
  --token-count "$COUNT" --layers all --parity-socket "$RUN/manager.sock" \
  2>&1 | tee "$RUN/server.log"
```

This sets `CUDA_VISIBLE_DEVICES=1,2` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
After server readiness, proceed to C. For DEBUG_SAFE online parity, use a fresh
run/snapshot and replace only `--protection-mode FAST_PARITY` with `DEBUG_SAFE`.

## Terminal C — warmup, actual request, live commit and Oracle validation

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
source /tmp/kv-ec-tp2.REPLACE/session.env
curl --fail "http://127.0.0.1:$PORT/health"
for i in 1 2; do
  python -m experiments.kv_ec.request_debug request \
    --url "http://127.0.0.1:$PORT" --request-id "warmup-$i" \
    --prompt-tokens 768 --stream-metrics --output "$OUT/parity-warmup-$i.json"
done
python -m experiments.kv_ec.request_debug request \
  --url "http://127.0.0.1:$PORT" --request-id "$RID" \
  --prompt-tokens 768 --stream-metrics --output "$OUT/parity-response.json"
python - <<'PY'
import hashlib, json, os, time
from pathlib import Path
from experiments.kv_ec.parity_manager import submit_once
run, out = Path(os.environ['RUN']), Path(os.environ['OUT'])
snap, count = os.environ['SNAP'], int(os.environ['COUNT'])
deadline = time.monotonic() + 60
while True:
    status = submit_once(run/'manager.sock', {'kind':'status', 'snapshot_id':snap}, b'', timeout=5)
    if status['state'] == 'COMMITTED':
        break
    if status['state'] in ('FAILED', 'EXPIRED'):
        raise RuntimeError(status)
    if time.monotonic() >= deadline:
        raise TimeoutError(status)
    time.sleep(0.2)
response = json.loads((out/'parity-response.json').read_text())
expected = 36 * count * 4 * 128 * 2 * 2
assert status['snapshot_id'] == snap and status['tp_size'] == 2
assert status['format'] == 'kv-ec-rank-bin-v1'
assert status['request_id'] == hashlib.sha256(os.environ['RID'].encode()).hexdigest()
assert status['model_identifier'] == response['model_identifier']
assert status['token_start'] == 0 and status['token_count'] == count
assert status['layers'] == list(range(36)) and status['dtype'] == 'torch.bfloat16'
assert status['parity_bytes'] == status['steady_parity_bytes'] == status['packed_byte_count'] == expected
assert set(status['rank_hashes']) == {'0', '1'}
assert status['oracle_checks'] == [{'lost_rank':i, 'byte_exact':True} for i in range(2)]
for record in status['layer_records']:
    for kind in ('k', 'v'):
        assert record['packed_shape'][kind] == [count, 4, 128]
        assert record['original_shape'][kind][1:] == [4, 128]
assert response['input_token_count'] == 768 and len(response['output_ids']) == 16
with (out/'commit.json').open('x') as f:
    json.dump(status, f, indent=2)
timings = sorted((out/snap).glob('rank_*.timing.json'))
assert len(timings) == 2
for path in timings:
    timing = json.loads(path.read_text())
    assert timing['performance_ms']['protection_mode'] == 'FAST_PARITY'
    print(path, json.dumps(timing))
assert {json.loads(p.read_text())['tp_rank'] for p in timings} == {0,1}
assert not list((out/snap).glob('*.bin'))
assert not [(out/snap)/f'rank_{i}.json' for i in range(2) if ((out/snap)/f'rank_{i}.json').exists()]
print(json.dumps(response, indent=2))
print('PASS: live COMMITTED, both erasures byte-exact, parity_bytes=', expected)
PY
```

`ENQUEUED` in rank timing only confirms local queue acceptance. The live manager
COMMITTED response above is required. Inspect `manager.log` and `server.log` for
failures. No rank binary files are written by this flow. JSON saves metadata,
artificial output tokens and diagnostics only. GPU memory stays read-only.

## TP=2 baseline on the same GPUs, run sequentially

After C finishes, Ctrl-C ONLY the foreground server you launched in B, and wait
for its workers to exit. Keep A alive. Check `nvidia-smi` before starting the
baseline; do not run both models concurrently on GPUs 1,2.

In B, with the same session loaded:

```bash
python -m experiments.kv_ec.launch_debug \
  --mode baseline --tp 2 --gpus 1,2 \
  --model /media/sda1/iic_data/luotao/models/Qwen3-8B \
  --port "$BASE_PORT" --output "$OUT" --snapshot-id "$SNAP-baseline" \
  --request-id "$RID" --token-count "$COUNT" --layers all \
  2>&1 | tee "$RUN/baseline-server.log"
```

In C, after baseline readiness:

```bash
curl --fail "http://127.0.0.1:$BASE_PORT/health"
for i in 1 2; do
  python -m experiments.kv_ec.request_debug request \
    --url "http://127.0.0.1:$BASE_PORT" --request-id "warmup-$i" \
    --prompt-tokens 768 --stream-metrics --output "$OUT/baseline-warmup-$i.json"
done
python -m experiments.kv_ec.request_debug request \
  --url "http://127.0.0.1:$BASE_PORT" --request-id "$RID" \
  --prompt-tokens 768 --stream-metrics --output "$OUT/baseline-response.json"
python -m experiments.kv_ec.request_debug compare \
  "$OUT/baseline-response.json" "$OUT/parity-response.json"
```

The helper uses identical fixed input IDs, temperature=0 and max_new_tokens=16.
Launcher inference flags are the same in both modes. Initialization and two
warmups are separate from the measured request. This run is a correctness check,
not a statistically meaningful performance comparison.

## Optional offline GPU Scratch Restore

Online parity intentionally does not save rank blocks. Scratch Restore needs a
SEPARATE artificial debug-capture run; it cannot load timing-only online output.
After stopping your own baseline server, in B start:

```bash
python -m experiments.kv_ec.launch_debug \
  --mode capture --tp 2 --gpus 1,2 \
  --model /media/sda1/iic_data/luotao/models/Qwen3-8B \
  --port "$PORT" --output "$OUT" --snapshot-id "$SNAP-scratch" \
  --request-id "$RID" --token-count "$COUNT" --layers all
```

In C, after readiness:

```bash
python -m experiments.kv_ec.request_debug request \
  --url "http://127.0.0.1:$PORT" --request-id "$RID" --prompt-tokens 768 \
  --stream-metrics --output "$OUT/scratch-response.json"
python -m experiments.kv_ec.rank_snapshot "$OUT/$SNAP-scratch"
```

Stop only your scratch capture server in B before allocating GPU scratch tensors:

```bash
CUDA_VISIBLE_DEVICES=1,2 CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -m experiments.kv_ec.gpu_scratch_restore "$OUT/$SNAP-scratch" \
  --device cuda:0 --warmup 1 --repeat 3 --output "$OUT/scratch-restore.json"
```

This separately saves two full ARTIFICIAL rank snapshots for offline validation;
do not use real user prompts or commit those binaries. Both ranks are recovered
individually into temporary tensors. TP=4 uses the same tool and all four ranks.
It never attaches to a running server or writes a real KV pool.

Next work should first establish a request/slot-generation lease and completion
event protocol, then validate restore into an isolated allocator under paused
scheduling. Real pool fault injection, writes, resumption and cross-node RS are
outside this change.
