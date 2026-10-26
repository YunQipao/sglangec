# Online chunk protection, stage 1

## Review and implementation plan

The existing `debug_capture.maybe_capture` hook is inside the completed-prefill
result processor, before append/cache/free. Its lifetime checks deliberately
exclude decode and intermediate prefill. `rank_snapshot` defines the canonical
ascending layer / K / V / logical token byte layout. `ReqToTokenPool` tracks
request-row generations, but these are rank-local and cannot identify a logical
checkpoint across TP workers. `ScheduleBatch.prepare_for_decode` advances
`kv_committed_len` before forward, so that counter alone is not a completion fence.

The smallest shared prefill/decode trigger is in `event_loop_normal`, immediately
after `run_batch`, before `process_batch_result` can release KV. The opt-in bridge
is initialized at scheduler startup; overlap mode is rejected there. No allocator
or model kernels need changes.

1. Require plain single-node TP2/TP4, Qwen3-8B BF16 MHA, page size 1,
   no overlap, cache sharing, speculation, offload, graphs or sessions.
2. At a completed forward, fence device writes and gather complete 256-token
   chunks into **independent GPU storage**. Fence those reads before returning
   control to the scheduler. Never enqueue a pool tensor or a live request.
3. Copy independent staging storage to bounded pinned Host buffers on a separate
   CUDA stream. A worker waits for the copy event, hashes bytes and sends them to
   a separate local Unix-socket manager. CPU XOR and disk writes happen there.
4. Match request/run/checkpoint/model/token/layer/layout identity across all
   ranks; timeout incomplete groups, reject conflicting duplicates and apply
   capacity limits. Persist only complete protected groups.
5. Exercise CPU fixtures and lifecycle failures first. Real CUDA ordering,
   reference-byte equivalence and serving latency require isolated TP2, then TP4
   runs. Online RDMA/RS is a later consumer of a completed CPU-owned checkpoint.

This stage accepts a conservative GPU synchronization/gather cost on the
inference thread. D2H completion, checksum scans, XOR, IPC and persistence must
remain outside it. It does not claim zero serving overhead or online recovery.

### Validation performed on 2026-10-26

Base: local `feat/physical-two-node-tp4-rs`, HEAD `dad684d6e8`. The targeted
Capture/parity/chunk/physical-RS/remote-RS/RDMA regression suite runs 89 tests:
78 pass and 11 skip. The new online module runs 16 tests: 14 pass and 2 skip.
The new skips are the real Unix-socket bind test (sandbox denies bind) and CUDA
staging (NVIDIA driver unavailable, `torch.cuda.is_available()` is false).
CPU worker-to-manager testing with a mocked transport does execute real hashing,
metadata validation, XOR and persistence. Compile checks, formatting checks and
`git diff --check` pass. No real TP2/TP4 serving or physical two-node online
validation has been performed in this environment.

## Implemented contract and limits

The feature defaults off. `SGLANG_KV_EC_ONLINE=1` explicitly enables it;
`SGLANG_KV_EC_ONLINE_SOCKET` must be an absolute Unix-socket path and
`SGLANG_KV_EC_ONLINE_RUN_ID` must match the local manager. Use a fresh run ID and
output directory for every server lifetime and unique request IDs in that run.
The launcher prepares the model fingerprint before loading the server. Direct
environment-based launch instead fingerprints during scheduler initialization.
Weight-update or memory-release/resume inputs end the protection epoch before
the scheduler dispatches them: queued tasks are cancelled and subsequent online
capture is disabled until a fresh server run. The update API itself is unchanged;
the old model fingerprint is never attached to newly updated weights.

Each rank emits the existing `kv-ec-rank-bin-v1` layout plus `request_id` (SHA256),
zero-based `chunk_id`, half-open `token_range`, half-open `layer_range=[0,36)`,
`tp_rank`, `tp_size`, `model_version` (weight/config SHA256), `checkpoint_version`,
`request_version` and `manager_epoch`. The checkpoint is a deterministic SHA256 of
run/request/chunk; physical slots and request-row generation remain rank-local.
The CPU manager matches the full common identity and layer byte offsets across
all ranks before XOR. It never pads or combines mismatched chunks.

Full chunks in intermediate prefill, final prefill and ordinary decode are
eligible. The final sampled token is excluded until a later forward writes its
KV. Incomplete tails are never read. A semaphore reserves capacity **before**
allocating staging; the default is two queued/active chunks per rank. A chunk
owns 18 MiB per TP2 rank or 9 MiB per TP4 rank, separately in GPU staging and
pinned Host storage (plus one-layer gather temporaries and bounded CPU IPC
copies). Queue pressure retries at the next live forward; request-end diagnostics
report unsubmitted complete chunks. This is best-effort protection under
overload, not a guarantee that every chunk is protected before request completion.

Only the scheduler reads request mappings or KV. It fences device writes,
gathers independent bytes, fences gather completion, rechecks the row generation
and mapping, and then enqueues D2H. Free/reuse after that point cannot affect the
copy. The worker waits for the CUDA event **before** reading Host memory.
Abort/retraction/recycled-row detection cancels queued tasks; cancellation also
waits for D2H before releasing staging. Normal request completion drains detached
snapshots. Already transmitted valid snapshots may still commit after an abort;
cancellation is best-effort and does not revoke a committed checkpoint. Missing
ranks expire at the manager. No online GPU-pool restore is implemented.

Manager pending groups and in-memory terminal history are bounded. Terminal
history retains 256 entries; persisted directories prevent committed checkpoint
reuse after history eviction. Expired/failed IDs can be retried after history
eviction. `manifest.json` is an atomic completion marker after parity is written;
there is no fsync durability guarantee. `--retain-ranks` keeps debug evidence,
otherwise only parity and rank metadata are persisted. Disk retention is not
bounded: select an experiment output directory and manage its lifetime externally.

`rank_performance_ms` records CUDA-event `gpu_to_host_ms`, `queue_wait_ms` and
`scheduler_stage_ms`. The manifest records `xor_ms` and `aggregation_ms`.
`KV_EC_ONLINE` worker logs report IPC outcomes, backpressure and request retirement
with scheduler request latency and unsubmitted chunk counts. The smoke client
records actual HTTP `request_ms`, excluding its later evidence wait. These are
observations, not measured overhead claims; use the matched baseline below.

## CPU tests and CUDA staging test

Run from the repository root with its configured PyTorch Python environment:

```bash
python -m unittest experiments.kv_ec.test_online_chunk -v
python -m unittest \
  experiments.kv_ec.test_debug_capture \
  experiments.kv_ec.test_parity_manager \
  experiments.kv_ec.test_chunk_protection \
  experiments.kv_ec.test_tp4_physical_rs \
  experiments.kv_ec.test_remote_rs \
  experiments.kv_ec.test_rdma_transport -q
```

The new CPU tests compare the **actual** online gather implementation against an
independent offline BF16 tensor selection, check TP2/TP4 aggregation and every
single-rank XOR recovery, exercise boundaries, cancellation, row reuse,
backpressure, duplicate/corrupt/mismatched submissions and timeout. The socket
test skips if the environment forbids Unix-socket bind. The CUDA staging test
skips without CUDA; with CUDA it tests real streams, D2H events and immediate
source-buffer reuse for all TP2/TP4 shard shapes. It uses synthetic tensors on
one GPU, so it does not prove model serving or TP collectives.

## Shared-GPU debugging (TP2/TP4, Online/Baseline)

`launch_online` rejects selected GPUs with compute tasks by default. Explicit
`--allow-shared-gpu-debug` permits those tasks after read-only per-GPU memory
admission. It never stops, pauses or changes another process. `--gpus` names
physical `nvidia-smi` indices in rank order and replaces any inherited
`CUDA_VISIBLE_DEVICES`; unrelated GPUs do not trigger the busy check.

Before starting SGLang, both modes report total/used/free bytes and estimated
**additional** memory for every selected GPU. Admission reuses
`gpu_memory_budget.py`: local safetensors weight size divided by TP, a 20%
weight-loading margin, conservative BF16 context KV for 4096 tokens, existing
CUDA/compute/EC runtime reserves, and `total_bytes * mem_fraction_static`.
The estimate is the larger of the model/runtime estimate and static allocation
plus runtime reserves. Each GPU must pass independently; ample memory on other
GPUs cannot compensate for an insufficient card. A static allocation too small
for weights plus context KV is also rejected. Local `config.json` and
safetensors weights/index are required for this estimate; missing budget data
fails closed. These are conservative estimates, not reservations: another job
can still consume memory after the check.

Use the installed server environment on this host. Set the following in **each
terminal**, choosing a fresh run ID, available port and the physical GPUs for
your experiment:

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PATH="/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export TP=4 GPUS=0,1,2,3 PORT=30381
# For TP2 instead: export TP=2 GPUS=0,1 PORT=30382
export RUN=online-shared-tp4-20261026-trial01
export SOCK="/tmp/${RUN}.sock" OUT="/tmp/${RUN}"
export MEM_FRACTION=0.35
```

Terminal A, Online only (wait for `READY`):

```bash
python -u -m experiments.kv_ec.online_chunk_manager \
  --socket "$SOCK" --output "$OUT" --run-id "$RUN" \
  --max-pending 8 --retain-ranks --test-recovery
```

Terminal B, Online server:

```bash
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
python -u -m experiments.kv_ec.launch_online \
  --model "$MODEL" --gpus "$GPUS" --tp "$TP" --port "$PORT" \
  --socket "$SOCK" --run-id "$RUN" --capacity 4 \
  --prefill-chunk-size 256 --mem-fraction-static "$MEM_FRACTION" \
  --allow-shared-gpu-debug
```

For a separate Baseline run, use a fresh run ID/port and the same server command
with `--baseline` appended; no manager is required (the socket/run-id arguments
remain required by the CLI). The explicit environment assignment is inherited
by SGLang and disables only its TP memory-imbalance check. Neither normal nor
shared mode disables that check automatically. For normal exclusive-GPU runs,
omit both the shared flag and the environment assignment (unset an inherited
`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` if necessary).

`MEM_FRACTION=0.35` is a starting configuration, not a claim about current free
memory. Adjust it using the printed per-card budget; lowering it can reduce
the static pool, but cannot bypass model/runtime minimums. Do not launch if
`nvidia-smi` cannot query the driver. Check readiness in another terminal:

```bash
curl --fail "http://127.0.0.1:${PORT}/health"
```

For the existing two-node RS procedure, keep its manager/RS configuration and
commands, and append `--allow-shared-gpu-debug` to each `launch_online` command
only when sharing GPUs intentionally. The transport/protection path is unchanged.

## Single-machine TP2 acceptance

Use **free, explicitly chosen** GPUs and ports. Commands are foreground and do
not kill any process. Example paths assume `/tmp` is available on the server.
Set the same variables in each terminal; choose fresh names if evidence exists.

```bash
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export RUN=online-tp2-20261026-a
```

Terminal A (local CPU manager; wait for `READY`):

```bash
python -m experiments.kv_ec.online_chunk_manager \
  --socket /tmp/kv-online-tp2-a.sock --output /tmp/kv-online-tp2-a \
  --run-id "$RUN" --retain-ranks --test-recovery
```

Terminal B (server; replace `0,1` with free physical GPU indices):

```bash
python -m experiments.kv_ec.launch_online \
  --model "$MODEL" --gpus 0,1 --tp 2 --port 30381 \
  --socket /tmp/kv-online-tp2-a.sock --run-id "$RUN" \
  --capacity 4 --prefill-chunk-size -1
```

Terminal C (after `/health` is ready):

```bash
python -m experiments.kv_ec.online_chunk_smoke \
  --model "$MODEL" --url http://127.0.0.1:30381 \
  --request-id online-tp2-prefill --prompt-tokens 768 --new-tokens 16 \
  --manager-output /tmp/kv-online-tp2-a --output /tmp/tp2-online-prefill.json
python -m experiments.kv_ec.online_chunk_smoke \
  --model "$MODEL" --url http://127.0.0.1:30381 \
  --request-id online-tp2-decode --prompt-tokens 255 --new-tokens 258 \
  --manager-output /tmp/kv-online-tp2-a --output /tmp/tp2-online-decode.json
```

Acceptance requires three prefill chunks (online IDs 0,1,2), two chunks reaching
their boundary during decode, all ranks, and all XOR erasure checks. Evidence wait
fails on missing chunks, rather than treating an enqueue as successful protection.
For an additional run exercising intermediate-prefill triggers, use a **fresh**
run/socket/output and `--prefill-chunk-size 256`. Run the same smoke requests.

## Single-machine TP4 acceptance

Use the same three-terminal procedure with a fresh run and four free GPUs:

```bash
# Terminal A
export RUN=online-tp4-20261026-a
python -m experiments.kv_ec.online_chunk_manager \
  --socket /tmp/kv-online-tp4-a.sock --output /tmp/kv-online-tp4-a \
  --run-id "$RUN" --retain-ranks --test-recovery

# Terminal B (set MODEL and RUN in this terminal as well)
python -m experiments.kv_ec.launch_online \
  --model "$MODEL" --gpus 0,1,2,3 --tp 4 --port 30382 \
  --socket /tmp/kv-online-tp4-a.sock --run-id "$RUN" \
  --capacity 4 --prefill-chunk-size -1

# Terminal C
python -m experiments.kv_ec.online_chunk_smoke \
  --model "$MODEL" --url http://127.0.0.1:30382 \
  --request-id online-tp4-prefill --prompt-tokens 768 --new-tokens 16 \
  --manager-output /tmp/kv-online-tp4-a --output /tmp/tp4-online-prefill.json
python -m experiments.kv_ec.online_chunk_smoke \
  --model "$MODEL" --url http://127.0.0.1:30382 \
  --request-id online-tp4-decode --prompt-tokens 255 --new-tokens 258 \
  --manager-output /tmp/kv-online-tp4-a --output /tmp/tp4-online-decode.json
```

Repeat with fresh names and `--prefill-chunk-size 256` before multi-server work.

## Independent offline reference comparison

The full-prefill acceptance run (`--prefill-chunk-size -1`) intentionally uses the
same serving flags, artificial prompt, seed and request ID as the existing
offline capture. After the online experiment, release **only your own** server
through your normal process-management workflow, or choose other free GPUs.
The original offline workflow remains unchanged:

```bash
python -m experiments.kv_ec.launch_debug --mode capture \
  --model "$MODEL" --gpus 0,1 --tp 2 --port 30383 \
  --output /tmp/kv-offline-tp2-a --snapshot-id reference-tp2 \
  --request-id online-tp2-prefill --token-count 256 --chunk-size 256 --layers all

# Separate terminal after readiness
python -m experiments.kv_ec.request_debug request \
  --model "$MODEL" --url http://127.0.0.1:30383 \
  --request-id online-tp2-prefill --prompt-tokens 768 \
  --output /tmp/tp2-offline-response.json
```

The smoke report maps chunk IDs to online checkpoint directories. For **each**
chunk, pass its recorded path and the corresponding offline directory:

```bash
python -m experiments.kv_ec.verify_online_chunk \
  /tmp/kv-online-tp2-a/ONLINE_CHECKPOINT_SHA256 \
  /tmp/kv-offline-tp2-a/reference-tp2-chunk-1
# online chunk 0 -> offline chunk-1, 1 -> chunk-2, 2 -> chunk-3
```

Repeat the offline capture and comparison with `--tp 4 --gpus 0,1,2,3`, matching
TP4 request ID and new output names. The comparison checks model/request/token/
layout identity, all rank bytes, stored parity and all single-rank reconstructions.
Do not compare decode-generated KV against replay-prefill KV as a presumed
bit-exact oracle: different attention kernels/shapes can change BF16 rounding.
The CUDA staging test independently compares both reads of the same source.

For latency, rerun `launch_online` with `--baseline` and identical TP, GPU,
prefill, model and serving settings on a free port. Use `online_chunk_smoke
--baseline` with the same artificial request lengths and fresh result filenames.
Compare warm repeated `request_ms` samples and `output_ids_sha256` against online
runs; do not count model hashing/startup or evidence polling as request latency.

## Later online RDMA + RS boundary

The exact CPU-owned handoff is in `OnlineChunkManager.submit`, after all ranks
pass `rank_manifest(entries)`, with ordered `blocks`, `entries` and local `parity`.
Do not extend the scheduler hook with network operations. A later bounded
`on_checkpoint(blocks, entries, parity)` consumer can run here on the manager side:

1. Build existing wire bytes with
   `node_block_export.pack(node_id, [b.tobytes() for b in blocks], entries)`.
   The existing format accepts the online rank layout; no GPU addresses escape.
2. Pass the block to `RemoteRSClient.register_node(stripe_id, node_id, block)`;
   use the returned registration identity with `remote_rs_client.binding`.
3. After both nodes register compatible checkpoints, call
   `RemoteRSClient.rdma_submit_node_block(stripe_id, node_id, block, **binding, ...)`.
   Existing `physical_two_node_rs.run_node` shows the concrete RDMA arguments.
4. Coordinate common model/source profile, request version, logical chunk and
   checkpoint version across 232/233 **before** registration; local row/slot
   generations must remain rank-local. Reuse current remote protocol timeouts and
   RS(2,1) validation, and define acknowledgment/backpressure/retention explicitly.

That callback is an interface proposal, not an installed network hook. Current
code never invokes online RDMA or RS. Default parity-only retention does not
retain data rank blocks; the next stage must hand off ownership before releasing
`blocks`, or explicitly enable debug rank retention and use the existing offline
node export. Dual-server serving interference remains unverified.
