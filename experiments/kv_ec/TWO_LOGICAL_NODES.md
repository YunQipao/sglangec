# Single-host two-logical-node TP4 experiment

This is an experiment-only launcher for one physical 4×RTX 4090 host:

| logical node | namespace | visible GPUs | SGLang `node-rank` | global TP ranks |
|---|---|---|---:|---|
| A | `kvectp-a` | 0,1 | 0 | 0,1 |
| B | `kvectp-b` | 2,3 | 1 | 2,3 |

It launches one native SGLang process per namespace with `--tp-size 4
--nnodes 2`; it is not two independent TP2 servers. Both processes use the
same `--dist-init-addr 10.233.77.1:29577`, and the namespace veth pair is the
only explicitly selected NCCL/Gloo interface. The process uses a local
`CUDA_VISIBLE_DEVICES` view, so Node B's local device 0 is physical GPU2.

The scripts never modify the host default route, physical NIC, Docker network,
global firewall, or existing namespaces. Setup and cleanup are plan-only until
the user passes `--apply` and types the confirmation phrase. No namespace is
created if a named namespace or veth already exists. GPU tasks are not probed
or terminated by this prototype; check shared-GPU admission separately before
using `--execute`.

## Source audit conclusions

This checkout supports `--nnodes`, `--node-rank`, and `--dist-init-addr`. In
`engine.py`, with `nnodes=2,pp=1,tp=4`, `_calculate_rank_ranges()` assigns two
TP ranks per node. `bootstrap.py` builds a TCP distributed init method from
`--dist-init-addr`, then initializes the global TP process group. Each worker
loads its own TP shard and allocates its own local KV pool; the existing KV EC
snapshot metadata must therefore record the local physical mapping and cannot
be treated as one shared KV tensor.

The existing EC parity manager and cross-rank control use AF_UNIX sockets and
are single-host control paths. Python multiprocessing Pipes in the SGLang
engine are also local process controls. They are not used by this launcher as
cross-namespace transport. The native SGLang distributed group is separate
from EC protection traffic. No cross-namespace KV protection path is claimed
yet.

## Phase 1: create and verify the private link

From the repository root, first print the plan:

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
sudo bash experiments/kv_ec/setup_network.sh
```

After reviewing the printed commands, execute only the named resources:

```bash
sudo bash experiments/kv_ec/setup_network.sh --apply
sudo bash experiments/kv_ec/verify_network.sh
sudo bash experiments/kv_ec/verify_network.sh --run
```

The first verification command is read-only plan mode. The second asks for a
confirmation phrase and then runs only namespace-local address, ping, and route
queries. Cleanup is similarly guarded:

```bash
sudo bash experiments/kv_ec/cleanup_network.sh
sudo bash experiments/kv_ec/cleanup_network.sh --apply
```

## Phase 2: native global TP4 smoke

Both node processes must run concurrently. Start Node A first, then Node B;
Node A owns the TCP rendezvous address. The launcher defaults to plan-only:

```bash
export SGLANG_MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export SGLANG_EC_RUN_ID=logical-tp4-smoke-$(date -u +%Y%m%dT%H%M%SZ)
export SGLANG_PORT=30000
export SGLANG_DIST_PORT=29577

bash experiments/kv_ec/run_logical_node.sh A --no-cuda-graph
bash experiments/kv_ec/run_logical_node.sh B --no-cuda-graph
```

If the plans are correct, run each in separate terminals and type the required
per-node confirmation:

```bash
bash experiments/kv_ec/run_logical_node.sh A --no-cuda-graph --execute
bash experiments/kv_ec/run_logical_node.sh B --no-cuda-graph --execute
```

For the first diagnostic attempt, force NCCL away from P2P and SHM only inside
the child processes:

```bash
bash experiments/kv_ec/run_logical_node.sh A --diagnostic-socket --no-cuda-graph --execute
bash experiments/kv_ec/run_logical_node.sh B --diagnostic-socket --no-cuda-graph --execute
```

This is a diagnostic transport run, not the comparable baseline. The normal
run does not set `NCCL_P2P_DISABLE` or `NCCL_SHM_DISABLE`. Inspect each saved
`server.log` for NCCL `NET/Socket` lines and save the exact command and
environment. Do not infer that every NCCL operation used veth from an
application log alone.

The default service port is inside each namespace. A request must therefore be
issued from Node A's namespace, for example:

```bash
sudo ip netns exec kvectp-a curl -sS http://10.233.77.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/media/sda1/iic_data/luotao/models/Qwen3-8B","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}'
```

The model name must match the server's configured model/API expectations.
Capture the response separately and compare token IDs with a previously
validated single-machine TP4 baseline. The launcher does not declare success
from process startup alone. Use:

```bash
python -m experiments.kv_ec.verify_logical_rank_logs \
  experiments/kv_ec/results/$SGLANG_EC_RUN_ID-node0/server.log \
  experiments/kv_ec/results/$SGLANG_EC_RUN_ID-node1/server.log \
  --output /tmp/kv-ec-logical-rank-map.json
```

That verifier requires four distinct runtime TP ranks and checks the expected
node assignment. It is intentionally evidence-based; missing model-runner
rank context is a failure.

## Phase 3+ boundary

Real KV snapshots must continue to use `rank_snapshot.py`'s existing capture,
layout, slot-generation and independent hash checks. The current launcher does
not wire those bytes into a cross-namespace TCP transport and does not claim
real KV protection or parity COMMITTED across the two logical nodes. The
Gateway prototype on the other branch is not imported here. Scheme A/C and
1/3/9 MiB chunk transport should be added only after native TP4 smoke has
produced rank and NCCL evidence; existing Unix-socket manager code must not be
assumed to work across namespaces.

There is no namespace network emulation of a physical node failure. A failed
logical namespace is only a process/network-isolation test, not a physical GPU
or server failure.
