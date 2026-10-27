# Stage D: Same-host New-Instance Cold Resume

This is an opt-in controlled experiment for one Qwen3-8B BF16 request on TP2.
It does **not** add automatic failover, detect process death, or make the SGLang
engine generally restartable. It restores only a real, sealed 256-token text
prefix, greedy sampling state, and the one pending sampled token.

## Boundary And State

For prefix `P[0:256]`, Prefill produces KV for all 256 tokens and samples `y0`.
At the state checkpoint boundary, the native result processor has appended
`y0`, the non-overlap `FutureMap` has published it, and `ChunkCache` has copied
the live row mapping. `y0` is still pending: the next Decode consumes it at
position 256 and only then writes its KV. The Stage D hook captures after the
Prefill result processor returns, before scheduler admission of another batch.

| State | `output_ids` | materialized KV | next native input | Position | Row and slots |
|---|---|---:|---|---:|---|
| origin after Prefill result | `[y0]` | 256 | `y0` in FutureMap | 256 | live origin row, generation, 256 owned slots |
| sealed request file | `[y0]` | 256 | explicit `pending_token_id=y0` | 256 | origin row/generation recorded as identity, not reused |
| fresh target before install | `[y0]` loaded from state | 0 | `y0` staged in target FutureMap | 256 | new native Req row and target generation, no KV owner |
| target after initial install | `[y0]` | 256 | `y0` in target FutureMap | 256 | same new row, 256 newly allocated slots |
| first target `prepare_for_decode` | `[y0]` | 257 scheduled | `y0` | 256 | new slots plus native output slot |
| first target Decode result | `[y0,y1]` | 257 materialized | `y1` | 257 | native result processor owns subsequent lifecycle |

The state includes schema/checkpoint/request versions, request and restore IDs,
model/source identity, original IDs and digest, sampled IDs, explicit materialized
KV and allocation lengths, pending token and its `has_kv=false` bit, next input,
position and sequence length, the complete supported greedy/stop policy, the
no-RNG policy, source node/range, origin Row/Generation, KV and state SHA256.
Both ranks must seal the same state digest against their own capture proof and
payload. Replay is rejected by an exclusive, persistent one-shot ledger claim.
The restored native Req keeps the source logical `extend_batch_idx=1` and
`decode_batch_idx=0`; this records the completed origin Prefill without claiming
that the target executed Prefill. The pending token is submitted under the
sealed request ID, and the scheduler checks that ID again after native admission.

The origin capture is itself the reference run. It uses `--template-checkpoint`
only to verify the expected model/layout/input IDs; the newly packed
`global_node.bin` is gathered byte-for-byte from that run's actual completed
CUDA Prefill. The capture then completes its normal 16 Decode forwards and
records Reference IDs/logits. It is not valid to pair an old KV block with new
or invented request state.

The fresh scheduler creates its request through the ordinary HTTP/tokenized
request path. The experiment-only `after_inputs` hook runs after normal request
validation but before prefill planning. It requires an otherwise empty native
scheduler, removes exactly that one Req from the waiting queue, allocates a real
Req row, seeds the native non-spec FutureMap, constructs the single-request
ScheduleBatch with native `init_new`, and only then calls the existing Stage A
GPU transaction in `initial_install` mode. Initial install records the stale
empty row bytes as an undo image; it claims no old Slot and frees none. After
both ranks vote through VERIFIED/MAPPED/COMMITTED/FINISHED, the native running
batch goes through ordinary `update_running_batch`, `prepare_for_decode`,
Triton attention, sampling, and BatchResultProcessor. Any failed rank aborts
both journals; no Decode may start. A completed Decode uses the target's
actual Triton `kv_indices` and new row mapping, and final ChunkCache / Req Pool
cleanup is checked on both ranks.

## 233 Procedure

Use a new shell on 233. Check GPU availability first; the launcher only chooses
currently idle devices and never stops other processes. The example assumes
the previous public test prompt's IDs are represented by `TEMPLATE`; change it
only if its identity/input digest matches the supplied `INPUT_IDS`.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export TEMPLATE=/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local/5695c8b23b7b66cd11158fd11cb01b970185b44ad57436f99929f294e40a0c3b/global_node.bin
export RUN=/tmp/kv-cold-d-20261027-01
mkdir -p "$RUN"
export INPUT_IDS="$RUN/input_ids.json"
export BUNDLE="$RUN/capture_bundle"
export REF="$RUN/capture-reference"
export LEDGER="$RUN/restore-ledger"
mkdir "$LEDGER"
python -m experiments.kv_ec.verify_gpu_restore public-test-inputs \
  --model "$MODEL" --checkpoint "$TEMPLATE" --output "$INPUT_IDS"
```

Run capture in the first terminal; it launches the original instance and leaves
the server in the foreground:

```bash
export PORT=30485
python -u -m experiments.kv_ec.verify_gpu_cold_resume capture \
  --model "$MODEL" --template-checkpoint "$TEMPLATE" --input-ids "$INPUT_IDS" \
  --bundle "$BUNDLE" --output "$REF/evidence" --gpus 1,3 --port "$PORT" \
  --mem-fraction-static 0.60 --timeout 120
```

In a second terminal, submit the normal 17-token request and inspect the
Reference/capture evidence. HTTP completion alone is not acceptance.

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RUN=/tmp/kv-cold-d-20261027-01
python -m experiments.kv_ec.verify_gpu_cold_resume submit \
  --output "$RUN/capture-reference/evidence" --url http://127.0.0.1:30485 \
  --wait-ready 600 --timeout 600
python -m experiments.kv_ec.verify_gpu_cold_resume reference \
  --output "$RUN/capture-reference/evidence"
```

After capture/reference has completed, manually stop **only this experiment's
own original server** with Ctrl-C. The restore command checks that both recorded
origin Scheduler PIDs and the `exec`-preserved server parent PID have exited,
and that host/boot IDs match. It will refuse
to launch a second instance while the origin is still alive. Choose currently
idle GPUs again; use a different output directory and port:

```bash
export RUN=/tmp/kv-cold-d-20261027-01
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
python -u -m experiments.kv_ec.verify_gpu_cold_resume launch \
  --model "$MODEL" --bundle "$RUN/capture_bundle" \
  --reference "$RUN/capture-reference/evidence" --ledger "$RUN/restore-ledger" \
  --output "$RUN/cold-resume/evidence" --gpus 1,3 --port 30486 \
  --mem-fraction-static 0.60 --timeout 120
```

Submit and check from another terminal:

```bash
export RUN=/tmp/kv-cold-d-20261027-01
python -m experiments.kv_ec.verify_gpu_cold_resume restore \
  --output "$RUN/cold-resume/evidence" --url http://127.0.0.1:30486 \
  --wait-ready 600 --timeout 600
python -m experiments.kv_ec.verify_gpu_cold_resume check \
  --output "$RUN/cold-resume/evidence"
```

The one-shot ledger makes every restore attempt consume that checkpoint, including
a failed attempt. To run failure injection, make a separate capture/reference
bundle and separate ledger first; then launch with `--inject-rank-failure 1`,
submit the request (the scheduler is expected to fail closed), and run
`check-failure`. Both rank reports must show an ABORTED transaction, no Decode,
and full Slot/Row reclamation. Capture a fresh bundle before subsequently trying
a normal restore.

Stage C input can be selected by adding `--rs-source <sealed-C-recovery-bundle>`
to the restore `launch` command. It is accepted only if its reconstructed bytes
are exactly the same hash as the newly sealed D capture. Stage C's older
historical data cannot be paired with a newly invented request state.

## PASS And Limits

`check` returns `pass_stage_d=true` only when the source state/KV seals validate,
the origin is stopped and the target processes are distinct, both ranks commit
the actual 256-token write, target Prefix Prefill count is zero, the saved
pending token/position and Decode sequence are correct, each eager Triton
observation indexes the restored Slots, exactly 16 post-restore CUDA forwards
occur, both ranks match all 16 Reference tokens, logits are saved and compared,
and target Req Rows/Slots fully return to their pools. The JSON includes per-step
max-absolute/RMS Logit error and any first token difference. A difference is a
failure, not rounded into a pass.

`check-failure` is a separate negative test. CPU tests validate state rejection,
version/token/KV-length corruption, generation disagreement, one-shot replay,
initial-install commit and rollback. Neither CPU tests nor `--help` establish
real CUDA PASS. This checkout/environment must complete the commands above on
233 before Stage D can be called GPU-verified. Only a controlled single-request
same-host cold start is in scope; generic queue restoration, arbitrary sampling
state, multi-request restart, cross-node failover, and whole-node recovery are
not implemented.
