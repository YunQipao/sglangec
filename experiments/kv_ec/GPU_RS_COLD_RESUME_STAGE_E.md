# Stage E: Native RDMA/RS To New-Instance TP2 Cold Resume

TP2/TP4 are now supported through the same real capture/restore path. For the
complete TP4 shared configuration and two-host commands, see
[TP4_SHARED_STAGE_E.md](TP4_SHARED_STAGE_E.md). The historical TP2 procedure below remains valid.

Stage E is opt-in and reuses Stage C's `RemoteRSClient` reconstruction and
Stage D's request injection, one-shot ledger, GPU transaction, TP voting,
attention observations, logits comparison and cleanup checker. It adds no
Scheduler hook and changes no allocator, CUDA fence, codec or GPU transaction.
Its CPU tests are **not** CUDA/native RDMA acceptance. Run the procedure below
on 232/233 before claiming `pass_stage_e=true`.

## Compatibility And Protection Protocol

| Component | Existing contract | Stage E adapter |
|---|---|---|
| Stage C | Two independent complete TP2 Node Blocks; online identity in rank metadata; immutable RS handle/manifest; native upload and reconstruction | Protect two fresh Stage D captures through existing OnlineChunkManager/OnlineRSExporter |
| Stage D | Exact KV file hash bound to sealed request state and two capture proofs; one pending sampled token without KV | Preserve the entire original state, seal, metadata and original KV hash |
| Export envelope | OnlineRSExporter adds `global_protection`, changing serialized Node Block bytes/hash | Strip **only** that verified extension after RS reconstruction; repack original metadata and require exact original descriptor/SHA256 |
| Process lifecycle | Stage C resumes same instance; Stage D requires all origin processes stopped and a new process on the same host/boot | D1 originates on 233 and cold-resumes on 233 after native recovery on 232 and native delivery back to 233 |
| GPU ownership | Stage D initial install allocates new row/slots and commits on both ranks | Identical Stage D initial install; original row/slots are never reused |

Historical Stage C KV and a new Stage D request state cannot be combined.
Capture **both nodes anew**: Node 0 is an independent real request on 232,
Node 1 is the real request to resume on 233. Each Node Block contains both TP
ranks, not one rank per physical host. The captures may have different input
tokens but must have compatible model fingerprints, source/core identity,
BF16 layout, TP2, 36 layers and `[0,256)` range. Each request/snapshot ID must
be distinct. No duplicated or synthetic second node is accepted.

`claim` fully validates an existing sealed Stage D Capture. It records exact
state/seal/proof documents and their hashes, original Node Block descriptor,
model/source fingerprints, request ID/token digest, TP rank layouts, pending
token, token range and capture ID. `plan` binds both claims. The SHA256 of the
canonical complete plan becomes the online protection ID/checkpoint version;
therefore the existing RS manifest binds both captures and their request states.

`export` feeds validated captured rank bytes to the existing online manager
(local XOR and both loss checks), then existing native RDMA exporter. It does
not re-execute Prefill. `protect` consumes **two actual exporter reports** and
queries the live RS server with their stripe ID, handle and manifest SHA256.
It accepts only two GLOBAL_COMMITTED receipts, native client/server CQ evidence,
matching metadata, local XOR reports and zero TCP control payloads.

The enforced causal order is:

1. Both real CUDA Captures seal their KV/request-state consistency boundary.
2. Both native uploads finish and the RS server commits parity.
3. `protect` verifies both receipts against live GLOBAL_COMMITTED state.
4. `recover` withholds D1 as an input; its only local KV input is D0. The RS
   server retains parity and discards source blocks at commit.
5. Native RDMA uploads D0 and returns reconstructed D1. The adapter reproduces
   the original sealed Capture hash **before** any target Scheduler runs.
6. Native RDMA delivers the hash-bound recovered bundle from 232 to 233.
7. Independent `check-source` compares to the original Capture and validates
   matching sender/receiver CQ statistics, archive SHA256 and every member.
8. A new process on 233 uses the sealed state and recovered KV, with zero prefix
   Prefill, followed by exactly 16 native Decode forwards.

This is causal API ordering, not a cross-host wall-clock ordering assertion.
Do not simulate loss before `protect` succeeds. Receipt timestamps are retained
as observations, not trusted synchronized clocks. A failed/expired/restarted
stripe requires fresh protection; `launch` and final `check` query its live
handle again. The ledger is claimed only after Stage E launch admission.

Runtime `load_checkpoint(capture, rs_source)` reads request state, seal, metadata
and capture proofs from the original bundle, but **no original KV payload file**.
Both ranks unpack the same validated reconstructed Node Block. The original
`global_node.bin`/rank payloads are used only for Capture/reference validation
and independent source comparison. No original target path is an argument to
`recover`. No TCP payload fallback or fixture option exists in the Stage E CLI.

## Ports And Directories

Example settings (verify NIC/GIDs and free ports on site):

| Host | Function | Address/ports | Directory |
|---|---|---|---|
| 232 | origin Node 0 TP2 | HTTP 30984 | `$RUN/capture`, `$RUN/node0-reference` |
| 233 | origin Node 1 TP2 | HTTP 30985 | `$RUN/capture`, `$RUN/node1-reference` |
| 233 | RS server, kept alive through final check | 192.168.2.245:30700; native listeners from 30720 | `$RUN/rs-server` |
| 232 | RS recovery client/return | 192.168.2.243:30890 | `$RUN/recovered-source` |
| 233 | recovered bundle native receiver | port 30892 | `$RUN/rs-source` |
| 233 | fresh target TP2 | HTTP 30986 | `$RUN/cold-new`, `$RUN/restore-ledger` |

Historical HCA settings: `mlx5_0`, port 1; 232 GID 5, 233 GID 3. Management
SCP uses 10.102.0.232/233. SCP below transfers **JSON metadata only**, never
target KV. If native self-host upload from 233 to its parity server is not
supported by the installed HCA configuration, export must fail; do not replace
it with TCP and do not claim PASS.

## 1. Fresh Environment On Both Hosts

Run once on 232 to choose the unique common experiment ID:

```bash
export RUN="$(mktemp -d /tmp/kv-rs-cold-e-$(date +%Y%m%d).XXXXXX)"
export EID="${RUN##*/}"
printf 'Copy this EID to 233 and every additional terminal: %s\n' "$EID"
```

On 233 set the exact printed EID, then create its independent directory:

```bash
export EID=REPLACE_WITH_PRINTED_EID
export RUN="/tmp/$EID"
mkdir "$RUN"  # must fail if already present; do not reuse an old run
```

In **every** additional terminal set the same EID/RUN without repeating mkdir.
On both hosts initialize the environment:

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
set -o pipefail
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export EID=REPLACE_WITH_PRINTED_EID
export RUN="/tmp/$EID"
git branch --show-current
nvidia-smi
```

Use the local model installation path on 232 if different. Both content/source
fingerprints must match. Select idle GPUs only; no command here stops another
service. `auto` chooses idle GPUs or refuses to launch.

## 2. Start An Empty RS Server On 233

In a dedicated foreground terminal on 233, with the fresh RUN environment:

```bash
python -u -m experiments.kv_ec.remote_rs_server \
  --host 192.168.2.245 --port 30700 --ttl 7200 --timeout 180 \
  --max-stripes 4 --max-bytes 536870912 \
  --rdma-device mlx5_0 --rdma-gid-index 3 --rdma-port-num 1 \
  --rdma-port-base 30720 --rdma-timeout 180 \
  --rdma-directory "$RUN/rs-server" 2>&1 | tee "$RUN/rs-server.log"
```

Leave it running until final Stage E check. If the port is occupied, use a new
free port consistently; do not terminate its owner. No historical receipt or
stripe survives a server restart: both exports below must create a new stripe.

## 3. Fresh Capture + Reference On Each Host

On **232**, in an origin-server terminal:

```bash
export NODE=0 PORT=30984
export TEMPLATE_ROOT=/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node0/local
export REF="$RUN/node0-reference"
```

On **233**, in a separate origin-server terminal:

```bash
export NODE=1 PORT=30985
export TEMPLATE_ROOT=/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local
export REF="$RUN/node1-reference"
```

On each host run this identical block. These historical files are **admission
templates only**; actual checkpoint KV and pending token come from the new
real CUDA Capture. If those paths are absent, point TEMPLATE_ROOT at a real
complete Node 0/1 template of the same layout. Do not manufacture node metadata.
The `public-test-inputs` helper only supports the original artificial public
prompt. For another real template, supply its original 256 input IDs instead;
do not bypass a token-digest mismatch.

```bash
export TEMPLATE="$(python - "$TEMPLATE_ROOT" "$NODE" <<'PY'
import sys
from pathlib import Path
from experiments.kv_ec.gpu_restore_hook import validate_checkpoint
found = []
for p in sorted(Path(sys.argv[1]).rglob('global_node.bin')):
    try:
        m = validate_checkpoint(p.read_bytes())
        if (m['node_id'] == int(sys.argv[2]) and len(m['ranks']) == 2
                and m['ranks'][0]['token_start'] == 0
                and m['ranks'][0]['token_count'] == 256):
            found.append(p)
    except (ValueError, KeyError):
        continue
if not found:
    raise SystemExit('Need a real matching Node/TP2/256 admission template')
print(found[0])
PY
)"
test -n "$TEMPLATE" && test -f "$TEMPLATE" || exit 1
python -m experiments.kv_ec.verify_gpu_restore public-test-inputs \
  --model "$MODEL" --checkpoint "$TEMPLATE" --output "$RUN/input_ids.json"
python -u -m experiments.kv_ec.verify_gpu_cold_resume capture \
  --model "$MODEL" --template-checkpoint "$TEMPLATE" \
  --input-ids "$RUN/input_ids.json" --bundle "$RUN/capture" \
  --output "$REF" --gpus auto --port "$PORT" \
  --mem-fraction-static 0.60 --timeout 120 2>&1 | tee "$RUN/capture-server.log"
```

In an additional terminal on each host, initialize the same environment and
set NODE/PORT/REF as above. Note the distinct `node0-reference` and
`node1-reference` output names: Stage D derives request IDs from these names.

```bash
python -m experiments.kv_ec.verify_gpu_cold_resume submit \
  --output "$REF" --url "http://127.0.0.1:$PORT" --wait-ready 600 --timeout 600
python -m experiments.kv_ec.verify_gpu_cold_resume reference --output "$REF"
echo "REFERENCE_EXIT=$?"
python -m experiments.kv_ec.verify_rs_cold_resume claim \
  --capture "$RUN/capture" --output "$RUN/claim$NODE.json"
```

Require `pass_reference=true` and exit 0 on both hosts. Then manually Ctrl-C
**only each experiment's own origin server**. Stage D refuses cold launch
until the sealed origin Scheduler and parent server processes have exited.
Keep the independent RS server alive.

## 4. Bind The Two Real Captures And Protect Them

On 232, copy only the new Node 1 claim metadata and create the plan:

```bash
scp "iic@10.102.0.233:$RUN/claim1.json" "$RUN/claim1.json"
python -m experiments.kv_ec.verify_rs_cold_resume plan \
  --node0-claim "$RUN/claim0.json" --node1-claim "$RUN/claim1.json" \
  --run-id "$EID" --output "$RUN/plan.json"
scp "$RUN/plan.json" "iic@10.102.0.233:$RUN/plan.json"
```

Run the following two exports **concurrently** (the first waits for its peer).
232:

```bash
python -u -m experiments.kv_ec.verify_rs_cold_resume export \
  --plan "$RUN/plan.json" --capture "$RUN/capture" --node-id 0 \
  --remote-host 192.168.2.245 --remote-port 30700 \
  --rdma-device mlx5_0 --rdma-gid-index 5 --rdma-port-num 1 \
  --timeout 600 --output "$RUN/export0" 2>&1 | tee "$RUN/export0.log"
```

233:

```bash
python -u -m experiments.kv_ec.verify_rs_cold_resume export \
  --plan "$RUN/plan.json" --capture "$RUN/capture" --node-id 1 \
  --remote-host 192.168.2.245 --remote-port 30700 \
  --rdma-device mlx5_0 --rdma-gid-index 3 --rdma-port-num 1 \
  --timeout 600 --output "$RUN/export1" 2>&1 | tee "$RUN/export1.log"
```

After **both** exports exit 0, on 232:

```bash
scp "iic@10.102.0.233:$RUN/export1/export.json" "$RUN/export1.json"
python -m experiments.kv_ec.verify_rs_cold_resume protect \
  --plan "$RUN/plan.json" --node0-export "$RUN/export0/export.json" \
  --node1-export "$RUN/export1.json" --remote-host 192.168.2.245 \
  --remote-port 30700 --timeout 120 --output "$RUN/protection.json"
```

No target KV is copied to 232. `export.json` contains actual exporter receipt,
local XOR hash/recovery checks and the plan hash, not the rank payload bytes.

## 5. Controlled D1 Loss And Native Recovery On 232

After protection succeeds, the recovery invocation deliberately has **no D1
input**, even though the original remains for independent acceptance on 233.
This models a withheld/lost Node Block, not physical destruction of the host.

```bash
export RESTORE0="$(python - "$RUN/capture/request_state.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['restore_id'])
PY
)"
python -u -m experiments.kv_ec.verify_rs_cold_resume recover \
  --protection "$RUN/protection.json" \
  --survivor "$RUN/export0/$RESTORE0/global_node.bin" \
  --remote-host 192.168.2.245 --remote-port 30700 \
  --rdma-device mlx5_0 --rdma-gid-index 5 --rdma-port-num 1 \
  --rdma-return-host 192.168.2.243 --rdma-return-port 30890 \
  --timeout 180 --output "$RUN/recovered-source" 2>&1 | tee "$RUN/recover.log"
python -m experiments.kv_ec.verify_rs_cold_resume inspect --bundle "$RUN/recovered-source"
```

## 6. Native Delivery To 233 And Independent Byte Acceptance

First start the native receiver on 233:

```bash
python -u -m experiments.kv_ec.verify_rs_cold_resume receive-source \
  --remote-host 192.168.2.243 --remote-port 30892 \
  --rdma-device mlx5_0 --rdma-gid-index 3 --rdma-port-num 1 \
  --timeout 180 --output "$RUN/rs-source" 2>&1 | tee "$RUN/receive.log"
```

`receive-source` listens locally; its `--remote-host` is an interface-compatible
CLI parameter, not a TCP payload download. Then send from 232:

```bash
python -u -m experiments.kv_ec.verify_rs_cold_resume send-source \
  --bundle "$RUN/recovered-source" --remote-host 192.168.2.245 --remote-port 30892 \
  --rdma-device mlx5_0 --rdma-gid-index 5 --rdma-port-num 1 \
  --timeout 180 --output "$RUN/delivery-send.json" 2>&1 | tee "$RUN/send.log"
scp "$RUN/delivery-send.json" "iic@10.102.0.233:$RUN/delivery-send.json"
```

After both complete, independently compare on 233:

```bash
python -m experiments.kv_ec.verify_rs_cold_resume check-source \
  --bundle "$RUN/rs-source" --capture "$RUN/capture" \
  --send-receipt "$RUN/delivery-send.json"
```

This requires exact byte equality with the original sealed target **and**
matching native sender/receiver archive SHA/length/CQ stats. Payload files in
the ZIP must equal all validated recovery artifacts. Copying an original KV
file into rs-source cannot satisfy these checks. The experiment trusts its
writers and native helper logs; these are integrity records, not cryptographic
attestations against a malicious host administrator.

## 7. Fresh Cold Target On 233

Use a fresh ledger. Leave RS server 30700 alive. In a foreground terminal:

```bash
mkdir "$RUN/restore-ledger"
python -u -m experiments.kv_ec.verify_rs_cold_resume launch \
  --model "$MODEL" --bundle "$RUN/capture" --rs-source "$RUN/rs-source" \
  --reference "$RUN/node1-reference" --ledger "$RUN/restore-ledger" \
  --output "$RUN/cold-new" --gpus 1,3 --port 30986 \
  --mem-fraction-static 0.60 --timeout 120 2>&1 | tee "$RUN/cold-server.log"
```

Use idle GPUs only; change `1,3` to `auto` if needed. In another 233 terminal:

```bash
python -m experiments.kv_ec.verify_rs_cold_resume restore \
  --output "$RUN/cold-new" --url http://127.0.0.1:30986 \
  --wait-ready 600 --timeout 600
python -m experiments.kv_ec.verify_rs_cold_resume check --output "$RUN/cold-new"
echo "STAGE_E_EXIT=$?"
```

Do not reuse any claimed capture/ledger, even after a failed attempt. A retry
requires a new RUN, both fresh Captures, new stripe and fresh ledger. All CLI
output creation is exclusive; no command overwrites an earlier experiment.

## PASS And Evidence

`check` exits 0 only if strict Stage D acceptance passes **and** the Stage E
bindings/native transport/live RS checks pass. Required values include:

- `pass_stage_e=true`, `pass_stage_d=true`, `real_rdma=true`;
- `rs_recovery_byte_exact=true`, `gpu_restore_committed=true`;
- `cuda_executed=true`, `new_instance=true`, `restored_kv_tokens=256`;
- both ranks COMMITTED, same RS source binding, no original KV restoration input;
- `restored_slots_used_by_decode=true`, `resume_decode_forward_count=16`;
- `recomputed_prefix_tokens=0`, `original_prefix_prefill_tokens=0`;
- `output_token_ids_match=true`, `no_slot_leak=true`, `no_request_row_leak=true`.

The 17 outputs include the sealed pending `y0` and 16 newly sampled tokens.
Stage D verifies each Forward's actual backend input, position and attention
indices, full logits files/hashes, exact IDs, first difference/max numerical
errors, generation, row/slot cleanup and TP phase sequence. An ID difference
never becomes PASS merely because logits are numerically close.

Inspect `plan.json`, both `export.json`/`global_status.json`, `protection.json`,
`rs-source/stage_e.json`, `protected/recovery.json`, `delivery.zip`, both delivery
receipts, `source-check.json`, `cold-new/run.json`, both rank JSON/event logs,
logits arrays and final `check.json`. The verdict retains the live RS status,
stripe ID/handle/manifest hash, reconstructed byte count/hash and comparisons.

## CPU Regression And Limits

```bash
PYTHONPATH=python:. python -m unittest \
  experiments.kv_ec.test_gpu_cold_resume experiments.kv_ec.test_gpu_resume \
  experiments.kv_ec.test_gpu_restore_hook experiments.kv_ec.test_gpu_kv_restore \
  experiments.kv_ec.test_scheduler_pause experiments.kv_ec.test_rdma_upload_control \
  experiments.kv_ec.test_rs_recovery_source experiments.kv_ec.test_rs_cold_resume \
  experiments.kv_ec.test_online_rs experiments.kv_ec.test_online_chunk \
  experiments.kv_ec.test_remote_rs experiments.kv_ec.test_rs_experiments \
  experiments.kv_ec.test_rdma_transport -q
git diff --check
```

The Stage E CPU tests use explicit fixture/transport mocks, the actual online
manager/exporter/StripeStore/RS codec, real state/seal validators and negative
integrity checks. They prove original KV files are unnecessary for runtime RS
loading, but cannot prove CUDA execution or physical RDMA connectivity.
Local result: 181 tests run, 173 passed and 8 skipped; all 14 Stage E tests
passed. Skips cover sandbox-denied Unix/TCP sockets, unavailable CUDA, the
explicit opt-in native RDMA test, and prerequisite-dependent negative tests.
`git diff --check` passed. All 19 Bash blocks passed `bash -n` (syntax only).

Scope remains one Qwen3-8B BF16 TP2 request, 256 tokens, greedy sampling,
Triton, no graphs/overlap/radix/speculation, same-host new-process resume.
Capture export is a post-seal CPU adapter into the online protection machinery,
not background protection of every serving request. Parity is volatile/TTL
bound. D1 and P are co-located on 233; this topology cannot survive loss of the
entire 233 host. The same-host boot identity, explicit ledger and manual
ordering do not provide distributed ownership fencing, automated detection,
service availability or complete whole-node failover.
