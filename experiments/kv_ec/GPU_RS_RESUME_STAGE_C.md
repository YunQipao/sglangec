# Stage C: RS Recovery to GPU Decode

Stage C adds a controlled single-Node-Block erasure path to Stage B. The client
reconstructs a missing real `global_node.bin` from the other Node Block and a
GLOBAL_COMMITTED remote parity stripe with `RemoteRSClient`'s native verbs
receive and send. It validates the online RS identity, immutable stripe handle,
manifest hash, target descriptor, model/layout/token identity, transfer CQ
statistics, and exact bytes before sealing the result. Both Scheduler ranks
read the same hash-bound artifact as immutable bytes, then reuse the existing
Stage B verified-writeback, two-rank commit, and Decode lifecycle.

This is **controlled single-block loss recovery followed by same-instance
request continuation**. It is not a cold instance restart or whole-node
failover. Stage C makes no whole-node failover guarantee. The original target
block is not an input to `recover`; it is read only by the independent
`check-source` acceptance step or by a separate reference run. TCP carries
control metadata only; native RDMA transfers survivor and reconstructed block.

## Preconditions

- Use the exact working tree and model fingerprint used by the real Stage B
  reference. Re-run the reference on the intended GPU host if its model path,
  argv, or runtime identity differs.
- The existing online exporter must have emitted both nodes' current
  `GLOBAL_COMMITTED` `global_status.json` receipts. The RS server process,
  handle, and manifest must still be live. Stage C will not mint a global
  receipt or recreate protection from a local target file. If the handle
  expired or server restarted, rerun the existing online two-node protection
  procedure first, in a new run directory.
- Use the same original request tokens on both nodes and select the receipt for
  that request/chunk. Do not stop other processes. Pick currently idle GPUs.
- Example topology below: TP2 resumes on **232** using target Node 1; Node 0
  survivor is local there. Remote parity/RS server is **233**. This lets the
  survivor be uploaded and the reconstructed block returned through native
  RDMA to the host running the two Scheduler ranks. Reverse node roles if the
  actual stripe layout differs; the GPU host must have the survivor file.
- Historical link-layer settings: 232 RDMA IP `192.168.2.243`, `mlx5_0`, GID
  5; 233 `192.168.2.245`, `mlx5_0`, GID 3. Confirm these on site before use.

## 232: GPU / Recovery Client Host

Run from a new shell. Point `TARGET_REF` at the original Node 1 block only for
independent byte comparison/reference; it is not passed to `recover` or to the
Scheduler hook. Point `REFERENCE_RUN` at a matching Stage B reference run made
on this host and using `TARGET_REF`.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export GLOBAL_RUN=kv-rs-resume-c-$(date +%Y%m%d-%H%M%S)
export RUN=/tmp/$GLOBAL_RUN
mkdir "$RUN"
export NODE0_ROOT=/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node0/local
export NODE1_TARGET_DIR=/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local/5695c8b23b7b66cd11158fd11cb01b970185b44ad57436f99929f294e40a0c3b
export NODE1_RECEIPT="$RUN/node1-receipt.json"
export TARGET_REF="$RUN/original-target-reference.bin"
```

Use authentic receipts emitted by the online exporter, and bind both nodes to
the target receipt. Copy the target file to a clearly separate reference-only
path; it is never passed to `recover` or to the Scheduler hook. `USER233` is
the site login name.

```bash
export USER233=iic
mkdir -p "$(dirname "$NODE1_RECEIPT")"
scp "$USER233@10.102.0.233:/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local/5695c8b23b7b66cd11158fd11cb01b970185b44ad57436f99929f294e40a0c3b/global_node.bin" "$TARGET_REF"
scp "$USER233@10.102.0.233:/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local/5695c8b23b7b66cd11158fd11cb01b970185b44ad57436f99929f294e40a0c3b/global_status.json" "$NODE1_RECEIPT"
export NODE0_RECEIPT=$(python - "$NODE1_RECEIPT" "$NODE0_ROOT" <<'PY'
import json, sys
from pathlib import Path
stripe = json.loads(Path(sys.argv[1]).read_text())["stripe_id"]
matches = [p for p in Path(sys.argv[2]).rglob("global_status.json")
           if (lambda d: d.get("stripe_id") == stripe and d.get("node_id") == 0
               and d.get("global_state") == "GLOBAL_COMMITTED")(
                   json.loads(p.read_text()))]
if len(matches) != 1:
    raise SystemExit(f"expected one Node 0 GLOBAL_COMMITTED receipt for {stripe}, got {len(matches)}")
print(matches[0])
PY
)
export NODE0="$(dirname "$NODE0_RECEIPT")"
python -m experiments.kv_ec.verify_rs_resume protect \
  --node0-receipt "$NODE0/global_status.json" \
  --node1-receipt "$NODE1_RECEIPT" \
  --target-node 1 --output "$RUN/receipt"
```

For a new input chunk, locate the matching hash-named directories and receipt
files instead of reusing the example hash. Confirm remote RS is healthy without
restarting it. `recover` receives only the Node 0 survivor; no target-file
argument exists.

```bash
export RECEIPT="$RUN/receipt/receipt.json"
export SURVIVOR="$NODE0/global_node.bin"
export REMOTE_RS=192.168.2.245
mkdir "$RUN/rs-source"
python -m experiments.kv_ec.verify_rs_resume recover \
  --remote-host "$REMOTE_RS" --remote-port 30300 --rdma-device mlx5_0 \
  --rdma-gid-index 5 --rdma-port-num 1 \
  --rdma-return-host 192.168.2.243 --rdma-return-port 30890 --timeout 120 \
  --receipt "$RECEIPT" --survivor "$SURVIVOR" --output "$RUN/rs-source"
python -m experiments.kv_ec.verify_rs_resume inspect --bundle "$RUN/rs-source"
python -m experiments.kv_ec.verify_rs_resume check-source \
  --bundle "$RUN/rs-source" --original "$TARGET_REF"
```

`check-source` is an independent verifier. Set `INPUT_IDS` to the same plain
256-token JSON input used by the Stage B reference. If needed, create a Stage B
reference from the original target on the same host using the documented
`experiments.kv_ec.verify_gpu_resume launch --mode reference` and `submit`
commands. This reference is fresh, separate from recovery, and uses the original
target file solely to establish comparison output.

```bash
export REFERENCE_RUN="$RUN/reference"
export INPUT_IDS="$RUN/input_ids.json"
python -m experiments.kv_ec.verify_gpu_restore public-test-inputs \
  --checkpoint "$TARGET_REF" --model "$MODEL" --output "$INPUT_IDS"
```

Terminal A, foreground reference server:

```bash
PORT=30981
python -u -m experiments.kv_ec.verify_gpu_resume launch --mode reference \
  --model "$MODEL" --checkpoint "$TARGET_REF" --output "$RUN/reference" \
  --gpus auto --port "$PORT" --mem-fraction-static 0.60 --decode-steps 16 \
  --timeout 120 2>&1 | tee "$RUN/reference.server.log"
```

Terminal B (set `RUN` to the exact path printed in Terminal A):

```bash
export RUN=/tmp/kv-rs-resume-c-REPLACE_WITH_PRINTED_RUN_ID
PORT=30981
python -m experiments.kv_ec.verify_gpu_resume submit --output "$RUN/reference" \
  --input-ids "$RUN/input_ids.json" --url "http://127.0.0.1:$PORT" --timeout 300
python -m experiments.kv_ec.verify_gpu_resume check --output "$RUN/reference"
```

After reference PASS, manually stop only its foreground server. Terminal A:

```bash
export REFERENCE_RUN="$RUN/reference"
export INPUT_IDS="$RUN/input_ids.json"
export CHECKPOINT="$RUN/rs-source/recovered_node.bin"
export PORT=30982
python -m experiments.kv_ec.verify_rs_resume launch \
  --model "$MODEL" --bundle "$RUN/rs-source" --reference "$REFERENCE_RUN" \
  --decode-steps 16 --output "$RUN/resume" --gpus auto --port "$PORT" \
  --mem-fraction-static 0.60 --timeout 120 2>&1 | tee "$RUN/server.log"
```

Terminal B, with the same explicit RUN path:

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RUN=/tmp/kv-rs-resume-c-REPLACE_WITH_PRINTED_RUN_ID
export REFERENCE_RUN="$RUN/reference"
export INPUT_IDS="$RUN/input_ids.json"
export PORT=30982
python -m experiments.kv_ec.verify_rs_resume submit \
  --output "$RUN/resume" --input-ids "$INPUT_IDS" \
  --url "http://127.0.0.1:$PORT" --wait-ready 600 --timeout 300
python -m experiments.kv_ec.verify_rs_resume check \
  --output "$RUN/resume" --reference "$REFERENCE_RUN"
```

Stage C check requires `pass_stage_c=true`, `real_rdma=true`,
`rs_recovery_byte_exact=true`, `gpu_restore_committed=true`,
`restored_slots_used_by_decode=true`, `resume_decode_forward_count=16`,
`recomputed_prefix_tokens=0`, `output_token_ids_match=true`, and
`no_slot_leak=true`. Inspect both `rank_0.json` and `rank_1.json`; both must
report the same RS source binding and COMMITTED transaction. `recovery.json`
contains stripe ID, handle, manifest SHA256, target Node ID, byte count/hash,
all four native verbs directions/CQ stats, and zero TCP payload counters.

## 233: RS Server Host

The preferred case reuses the live server that owns the receipts' immutable
handle and manifest. Do not restart it. Check it from 232:

```bash
python - <<'PY'
from experiments.kv_ec.remote_rs_client import RemoteRSClient
s, _ = RemoteRSClient("192.168.2.245", 30300, timeout=5).call("health")
print(s)
PY
```

If no live server/handle remains, first run the existing two-node online RS
protection procedure in [ONLINE_RS_STAGE2.md](ONLINE_RS_STAGE2.md), using fresh
run/output directories and the current `global_status.json` receipts. Do not
substitute a hand-created receipt or local RS mock. The native recovery traffic
is initiated by 232 and returned from 233; 233 need not access the missing
Node 1 file during recovery. Do not stop this server; the existing handle and
manifest must remain live for the duration of reconstruction.
