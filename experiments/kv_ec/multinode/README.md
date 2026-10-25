# Two node synthetic RS(2,1) prototype

This is a CPU Host DRAM prototype. It does not start SGLang, use a GPU, capture
a live KV cache, or claim online/GPU failure recovery.

## Existing code and missing pieces

| Existing implementation | What it provides / what this prototype adds |
|---|---|
| `../xor_codec.py` | Same-length XOR and one data erasure. New `rs_codec.py` supplies GF(256) coefficients and RS(2,1) erasure recovery. |
| `../parity_manager.py` | Fail-closed identity/checksum checks and commit only after inputs arrive. This prototype similarly marks COMMITTED only after both data blocks, parity, checksum, erasure, and transfer checks complete. |
| `../snapshot.py` | SHA-256 and raw CPU-byte preservation. This prototype uses synthetic Host DRAM buffers, without live pool access. |
| `../chunk_protection.py` | Contiguous committed chunks for one local request. New global `experiment_id/stripe_id/chunk_id` is separate from each node's local request identity. |
| Existing tests | `tests/` adds GF codec, metadata/corruption, and TCP integration tests. |

Initial placement is D0 on Node A, D1 on Node B, and P in an independent
buffer on Node B. Losing D1 or P can be demonstrated at block level. Losing
Node B removes both D1 and P. **With two physical nodes this does not provide
arbitrary single-node failure recovery.** A future placement needs a third
failure domain or explicit tradeoffs.

## Requirements and RDMA setup

CPU mode requires Python 3.10+ and NumPy. TCP mode uses ordinary TCP sockets
for payload in both directions and reports `transport=tcp`.

RDMA mode compiles `verbs_helper.c` into a per-process path under the system
temporary directory using `cc`, `libibverbs`, and verbs headers. It uses host
buffers, MR registration, PD, CQ, RC QP, TCP exchange of QP/MR metadata,
`RDMA_WRITE_WITH_IMM`, completion checks, and orderly resource release. TCP
carries control, lengths, connection metadata, and acknowledgements; data
payload bytes use verbs. No CUDA, GPUDirect RDMA, NCCL, or pyverbs is used.
RDMA mode never silently falls back to TCP.

InfiniBand LID routing is the default. For RoCE/GID routing, export matching
`KV_EC_RDMA_PORT` (default `1`) and `KV_EC_RDMA_GID_INDEX` on both nodes; choose
a valid local GID index on each host. The helper is a first prototype and is
not hardened for untrusted networks or concurrent stripes.

Inspect each server:

```bash
ibv_devinfo
rdma link
ibstat
ls -l /dev/infiniband
```

The current development container does not provide the verbs headers or an
active HCA (`/dev/infiniband` is absent), so the native path is **NOT VERIFIED**
here. Check firewall and route access
for TCP ports 19470, 19471, and 19472 with the cluster administrator.

## Paired two-node commands

Use the same checkout/revision and Python environment on both nodes. Start
Node B first. Its output directory must be new for each run:

```bash
cd /path/to/sglangec
PYTHONPATH="$PWD" python -m experiments.kv_ec.multinode.worker \
  --transport rdma --control-host 0.0.0.0 --control-port 19470 \
  --data-host 0.0.0.0 --data-port 19471 \
  --return-host NODE_A_IP --return-port 19472 --rdma-device mlx5_0 \
  --rdma-gid-index 3 --rdma-port-num 1 \
  --d0-bytes 1048576 --d1-bytes 1048576 \
  --experiment-id rs21-exp-001 --stripe-id stripe-0001 \
  --checkpoint-version cp-001 --output /data/rs21/node-b/run-001
```

Then on Node A, use the routable Node B address and a new output directory:

```bash
cd /path/to/sglangec
PYTHONPATH="$PWD" python -m experiments.kv_ec.multinode.coordinator \
  --transport rdma --node-b NODE_B_IP --control-port 19470 --data-port 19471 \
  --return-bind-host 0.0.0.0 --return-port 19472 --rdma-device mlx5_0 \
  --rdma-gid-index 5 --rdma-port-num 1 \
  --d0-bytes 1048576 --d1-bytes 1048576 \
  --experiment-id rs21-exp-001 --stripe-id stripe-0001 \
  --checkpoint-version cp-001 --output /data/rs21/node-a/run-001
```

The payload travels Node A→B as D0 and Node B→A as D1. TCP mode is an explicit
comparison: use `--transport tcp` in both commands. Do not report TCP rows as
RDMA measurements. Synthetic D0/D1 are reproducible from the common seed
(default `20261025`). The nodes have distinct local synthetic request IDs;
global stripe identity is independent of those IDs.

Unequal lengths are supported using zero padding. Set both lengths identically
on both commands. Each length is limited to 1..256 MiB. For custom placement,
stripe IDs, chunk IDs, or checkpoint versions, pass matching values to both
commands.

## Verification and benchmarks

CPU unit tests:

```bash
python -m unittest discover -s experiments/kv_ec/multinode/tests -v
```

Standalone synthetic RS verification, including unequal lengths and all three
block erasures:

```bash
PYTHONPATH="$PWD" python -m experiments.kv_ec.multinode.verify \
  --d0-bytes 1048576 --d1-bytes 786432 --seed 20261025
```

CPU-only codec benchmark:

```bash
PYTHONPATH="$PWD" python -m experiments.kv_ec.multinode.benchmark \
  --sizes-mib 1 4 16 64 --iterations 3 --output /data/rs21/codec-bench-001
```

For network benchmarks, start a fresh worker/coordinator pair per row and run
each size once with TCP and once with RDMA. Set both `--d0-bytes` and
`--d1-bytes` to the following value:

| Block | Bytes |
|---:|---:|
| 1 MiB | 1048576 |
| 4 MiB | 4194304 |
| 16 MiB | 16777216 |
| 32 MiB | 33554432 |
| 64 MiB | 67108864 |

Output directories must be unique. `verify_stripe` rejects missing data,
checksum failures, identity/codec/placement mismatches, pending protection
state, and nonzero recovered padding before the worker emits COMMITTED.

## Output fields and definitions

Both sides write `result.json` and `result.csv`. Node B also writes the received
raw `D0.bin`, padded RS buffers `D0.padded.bin` and `D1.padded.bin`, and `P.bin`.
It fsyncs the three padded blocks before publishing the committed JSON. JSON includes transport,
global stripe identity, per-block source/role/length/padding/codec/checksum,
checkpoint state, placement, erasure proofs, latency, transferred bytes,
process CPU, and Host DRAM/RSS. RDMA rows additionally include the selected
device/GID/port, local and remote QP numbers, CQ completion status, device/QP/MR
setup timings, exchange/completion timing, and helper-reported payload bytes;
TCP control bytes are not counted as RDMA payload.

- `rs_encode_ms`: Node B CPU RS encode wall time.
- `rs_decode_ms`: maximum of the three individual CPU erasure check intervals.
- `transfer_ms` and `reverse_transfer_ms`: sender interval through remote
  completion/acknowledgement in each direction.
- `effective_network_gbps`: both payload directions' bytes divided by the sum
  of those sender intervals; excludes RS encode/decode.
- `protection_completion_ms`: Node A wall time from initial D0 send through
  result commit and verification. It encloses transfer and encoding intervals;
  do not add overlapping components to reconstruct it.
- `recovery_ms`: maximum single-erasure CPU decode interval.
- `total_transferred_bytes`: D0 plus D1 payload bytes; excludes metadata.
- `cpu_percent`: Node B process CPU seconds divided by its protection wall time.
- `host_dram_bytes`: D0+D1+P live byte buffers; `max_rss_bytes` is process peak
  RSS, not node-wide DRAM usage.

## Limitations

- Synthetic CPU buffers only: no GPU inference, real KV capture, scheduler
  integration, GPUDirect, NCCL, persistent checkpoint store, or live restore.
- The tested failures are individual RS block erasures, not arbitrary node
  failures. Node B stores both D1 and P in this placement.
- One stripe per worker invocation; start a fresh worker and use fresh output
  paths for every size/repetition.
- One-shot RC QPs and TCP metadata exchange are prototype quality. RDMA must be
  verified on the target server pair, including selected device/port/GID.
