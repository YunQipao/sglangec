# Two logical nodes with a third-machine Gateway (prototype)

This module is deliberately independent of M6/M7's default entry points. The
layout under study is one physical Server A, split into logical Node0=GPU0,1
and Node1=GPU2,3, plus Server G running only the relay. It is **not** a
multi-physical-node SGLang TP run. Gateway data are opaque and volatile; no
payload is written to disk.

## Reused code and current boundaries

`rank_snapshot.py` remains the source of truth for real request KV packing,
mapping metadata and per-layer/K/V hashes. `xor_codec.py` provides bytewise
XOR. `checkpoint_protocol.py` is the existing local manager framing. New
`gateway_protocol.py`, `gateway_server.py` and `gateway_client.py` implement a
separate TCP relay. `gateway_pipeline.py` implements scheme A reference XOR,
scheme C's bounded online chunk aggregator and deterministic CPU comparison.

The current prototype has not wired CUDA gather/D2H into the dual-node runner;
it does not claim pinned-memory or GPU-event integration. The safe integration
point is an already frozen CPU snapshot from the existing rank capture path.
Each rank's captured payload should be striped into chunks, sent with the
checkpoint metadata, and COMMITTED only after all chunk receipts and parity
chunk hashes agree. Do not export raw payloads into evidence files.

## Local CPU tests and parity check

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
python -m unittest experiments.kv_ec.test_gateway_prototype -v
python -m experiments.kv_ec.gateway_pipeline \
  --bytes-per-rank $((9*1024*1024)) --ranks 4 --chunk-bytes $((256*1024)) \
  --output /tmp/kv-ec-gateway-a-c.json
```

The loopback tests exercise real TCP and can be blocked by sandbox policy. Run
them in the target Linux environment if local socket creation is prohibited.
Synthetic data only; the JSON says so explicitly.

## Three-machine application path

Choose a unique epoch string and set `G_IP`, `A_IP`, and ports reachable only
as required by the lab firewall policy. Example below assumes Node1's receiver
and Gateway both run on Server G; the receiver binds loopback so the only
external ingress is the Gateway listener.

Server G, terminal 1 (receiver):

```bash
EPOCH=lab-ec-2026-01
python -m experiments.kv_ec.gateway_server --role receiver \
  --bind 127.0.0.1 --port 46102 --epoch "$EPOCH" --node-id node1 \
  --max-queue 256 --metrics-json /tmp/kv-ec-receiver-metrics.json
```

Server G, terminal 2 (Gateway):

```bash
EPOCH=lab-ec-2026-01
python -m experiments.kv_ec.gateway_server --role gateway \
  --bind 0.0.0.0 --port 46101 --receiver 127.0.0.1:46102 --epoch "$EPOCH" \
  --metrics-json /tmp/kv-ec-gateway-metrics.json
```

Server A, CPU-only synthetic Node0 sender:

```bash
EPOCH=lab-ec-2026-01
python -m experiments.kv_ec.gateway_client --gateway G_IP:46101 \
  --epoch "$EPOCH" --source-node node0 --destination-node node1 \
  --total-bytes $((9*1024*1024)) --chunk-bytes $((256*1024)) \
  --output /tmp/kv-ec-node0-network.json
```

For Node1->Node0 reverse traffic, restart Gateway with `--destination-node`
support in a future direction-aware configuration (not yet implemented); do not
claim bidirectional operation from this one-way prototype. The current strict
route is node0 -> Gateway -> node1.

Connectivity checks (no payload):

```bash
# From A: TCP listener reachability
python -c 'import socket; s=socket.create_connection(("G_IP",46101),3); print(s.getpeername()); s.close()'
# On G: observe actual ingress/egress packets while sending
sudo tcpdump -ni any 'tcp port 46101 or tcp port 46102'
# Or record per-interface byte counters before/after with `ip -s link`.
```

Stop each service with Ctrl-C to atomically emit its metadata-only metrics JSON.
The receiver queue is bounded at 256 volatile chunks (maximum 256 MiB); queue
saturation rejects more traffic instead of growing memory. For longer tests,
drain/process receiver chunks between sends or choose a smaller finite queue
only when the workload fits.

App JSON alone is not proof of physical routing. Save Server G's packet capture
or before/after interface counters, command/filter, interface, timestamps, and
the corresponding client run ID. No raw KV is needed for this evidence.

## Real rank snapshot (manual, not a dual-node SGLang claim)

Use the existing accepted TP4 capture command in `GPU_SCRATCH_CAPTURE.md`,
which invokes the current `launch_debug.py`/capture worker and
`rank_snapshot.py`. It captures a single-machine TP4 run. Before opening the
raw rank files, keep them in a private volatile filesystem and never upload
them. This prototype still needs an adapter to consume the capture's immutable
rank bytes directly and split them into relay chunks; no adapter or end-to-end
GPU acceptance command is claimed yet.

## Optional namespace isolation

No namespace or firewall scripts are provided in this first application-layer
stage. A safe implementation needs explicit per-run namespace names, veth pairs
and addresses, rules limited to those interfaces/namespaces, an ownership
manifest, and an idempotent cleanup script. It must not change the host default
route, existing Docker networks, or global firewall policy. Until then, the
explicit socket endpoint is the only supported mode.

## Metrics and next integration

Gateway keeps volatile counters for payload RX/TX, chunks, duplicate retries,
failures and relay time. Sender records endpoint, chunk RTT distribution and
throughput. Queueing/backpressure is bounded (connection semaphore, receiver
queue); queue saturation fails closed. Scheme C's `ChunkXorAggregator` emits a
parity chunk once all source ranks' corresponding chunks arrive, with a fixed
in-flight chunk cap. The CPU harness compares all parity bytes against scheme
A.

Next steps: (1) add a receiver-side checkpoint assembler with per-chunk digest
manifest and explicit PREPARING/COMMITTED states; (2) stream immutable
`rank_snapshot` output through the relay, never filesystem payloads; (3) add a
CUDA producer adapter using existing capture's D2H completion events and
pinned, double-buffered host chunks; (4) prove Gateway traffic with G-side
packet capture; (5) only then benchmark A/C with at least three matched runs.
TTFT/TPOT and foreground-blocking metrics belong to the existing SGLang request
instrumentation and are not inferred from this CPU prototype.
