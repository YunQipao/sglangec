# Qwen3-8B real KV hierarchical EC (prototype)

This prototype consumes the existing, completed `rank_snapshot` output. It
reuses the SGLang capture path, rank-bin schema and validation, then runs local
XOR and cross-node RS after the request has completed. It does not hook the
scheduler and does not provide online protection/restore during inference.

## Encoding choice and failure domains

The cross-node RS data blocks are the complete serialized four-rank node
snapshots (方案 A). Each serialized block contains its own four TP rank payloads
and rank-local metadata. Node A and Node B may use different request IDs; their
model fingerprint, BF16/TP4 layout, layer range, 256-token range and per-rank
KV layout must match. TP ranks remain local to their node and do not form a
fictional TP=8 model.

方案 B (RS over only the local XOR parity) is not used. A node's XOR parity
cannot recover its four rank payloads by itself: after recovering that parity,
three original rank blocks are still missing. It could protect parity storage,
but that is not node-level KV protection.

Current placement is D0 on Node A, D1 on Node B, and P in a separate buffer on
Node B. This verifies block erasures only. D1 and P share Node B's physical
failure domain, so the prototype does **not** recover arbitrary whole-node
failure. A third independent failure domain is required for that claim.

## Capture and execution

Run these steps after activating the repository's already working SGLang
Conda environment on both servers. Use the same repository commit, model
weights, tokenizer and inference settings. Each node receives its own request
ID. The launcher only starts its own server; stop that owned server with its
terminal interrupt after its request has completed.

On Node A, start the capture server (replace GPU IDs/model/output path as
needed):

```bash
cd /path/to/sglangec
python -m experiments.kv_ec.launch_debug \
  --mode capture --tp 4 --gpus 0,1,2,3 --port 30281 \
  --model /models/Qwen3-8B --context-length 4096 \
  --mem-fraction-static 0.60 --token-count 256 --chunk-size 256 --layers all \
  --output /data/kv-ec/node-a --snapshot-id node-a-run --request-id node-a-request
```

In another Node A terminal, send its artificial 768-token request:

```bash
python -m experiments.kv_ec.request_debug request \
  --url http://127.0.0.1:30281 --model /models/Qwen3-8B \
  --request-id node-a-request --prompt-tokens 768 --stream-metrics \
  --output /data/kv-ec/node-a/request.json
```

On Node B, start its independent capture server:

```bash
cd /path/to/sglangec
python -m experiments.kv_ec.launch_debug \
  --mode capture --tp 4 --gpus 0,1,2,3 --port 30281 \
  --model /models/Qwen3-8B --context-length 4096 \
  --mem-fraction-static 0.60 --token-count 256 --chunk-size 256 --layers all \
  --output /data/kv-ec/node-b --snapshot-id node-b-run --request-id node-b-request
```

Then issue its own request on Node B:

```bash
python -m experiments.kv_ec.request_debug request \
  --url http://127.0.0.1:30281 --model /models/Qwen3-8B \
  --request-id node-b-request --prompt-tokens 768 --stream-metrics \
  --output /data/kv-ec/node-b/request.json
```

Ensure the two requests' token ranges and
all 36 layer layouts match. Each output should contain `node-{a,b}-run-chunk-1`
through `-chunk-3`; the capture includes three 256-token ranges covering the
768-token prompt.

Check RDMA devices on both nodes before selecting the verbs path:

```bash
ibv_devinfo
ibstat
rdma link show
ls -l /dev/infiniband
```

The native helper requires libibverbs headers/library, a C compiler, a usable
verbs device and device permissions. For RoCE, configure the correct GID index
and port in `KV_EC_RDMA_GID_INDEX` / `KV_EC_RDMA_PORT` for the installed fabric.
The prototype defaults to device auto-selection, port 1 and GID index -1. Do
not select `rdma` if device discovery fails.

Start the receiver first on Node B, then the sender on Node A. Example TCP
stripe for chunk 1 (repeat for chunks 2 and 3 with unique output/stripe IDs):

```bash
# Node B
python -m experiments.kv_ec.multinode.kv_hierarchical receive \
  --snapshot /data/kv-ec/node-b/node-b-run-chunk-1 --node-id node-b \
  --transport tcp --bind-host 0.0.0.0 --data-port 19481 \
  --stripe-id stripe-1 --experiment-id qwen3-two-node \
  --request-metrics /data/kv-ec/node-b/request.json \
  --output /data/kv-ec/results/tcp-chunk-1

# Node A
python -m experiments.kv_ec.multinode.kv_hierarchical send \
  --snapshot /data/kv-ec/node-a/node-a-run-chunk-1 --node-id node-a \
  --transport tcp --peer NODE_B_CONTROL_OR_DATA_IP --data-port 19481 \
  --stripe-id stripe-1 --experiment-id qwen3-two-node \
  --request-metrics /data/kv-ec/node-a/request.json \
  --output /data/kv-ec/results/sender-tcp-chunk-1
```

For real RDMA, repeat with `--transport rdma` on both ends and add
`--rdma-device mlx5_0` (or the actual device name). Set the fabric environment
on both nodes before running, for example:

```bash
export KV_EC_RDMA_PORT=1
export KV_EC_RDMA_GID_INDEX=3  # only if this is the working RoCE GID index
```

Use unique output directories; tools refuse to overwrite evidence. Only the
RDMA verbs helper transmits the node block for `--transport rdma`; the TCP
fallback is labeled `tcp` and must not be reported as RDMA.

## Verify and test

Node B verifies the committed stripe and its local XOR again without starting
SGLang or requiring a GPU:

```bash
python -m experiments.kv_ec.multinode.verify_hierarchical \
  /data/kv-ec/results/tcp-chunk-1 \
  --local-snapshot /data/kv-ec/node-b/node-b-run-chunk-1
```

Repeat with the corresponding RDMA output directory. CPU unit tests:

```bash
python -m unittest experiments.kv_ec.multinode.tests.test_kv_hierarchical \
  experiments.kv_ec.multinode.tests.test_rs_codec
```

The result directory contains `hierarchical-result.json`,
`hierarchical-result.csv`, `D0.bin`, `D1.bin`, `P.bin`, the received serialized
node block, this node's local XOR parity, and the sender's local XOR parity in
its separate output directory. JSON records node/request/chunk/global stripe
identity, model fingerprint, checkpoint version, local rank and KV-head ranges,
checksums, codec parameters, transfer bytes/timings, XOR/RS encode and recovery
timings, protection completion, process CPU time, estimated host buffer bytes,
and client TTFT/TPOT/throughput when `--request-metrics` is supplied. Host DRAM
includes an estimated protected-payload byte count and process peak RSS. GPU memory
is `null` because this offline stage does not query GPU memory. CSV contains the
same scalar timing/traffic fields; the JSON is authoritative for per-node
metadata and checks.

## Stage acceptance

1. CPU unit tests pass, including rank snapshot validation, independent local
   request identities, local single-rank XOR recovery, and RS D0/D1/P erasures.
2. Both live node captures produce all three complete 256-token chunks and
   ordinary Qwen3-8B BF16 TP=4 rank files with checksums.
3. Run TCP first; require Node B result `state=COMMITTED`, verifier `state=PASS`,
   four rank checks per node, byte-exact D0/D1 recovery, and P recomputation.
4. Run the same stripe over RDMA; require `transport=rdma`, successful verbs
   completion on both nodes, matching checksums, and verifier `state=PASS`.
5. Only after correctness passes, compare repeated Baseline, Local-XOR,
   Hierarchical-TCP and Hierarchical-RDMA inference trials. Keep the request
   prompt, model and server configuration fixed; gather the request JSON metrics
   and system GPU/driver/fabric inventory separately. This prototype attaches
   client metrics, but does not automate the four-mode inference benchmark or
   collect GPU-memory/driver/NIC counters.

The snapshot processing happens after request completion, so attached TTFT,
TPOT and throughput are request measurements and do not imply online EC latency
overhead. Shared-GPU performance results must be labeled as externally loaded
and compared only with similarly loaded runs. No GPU fault, online restore,
third failure domain, or whole-node failure recovery is claimed.
