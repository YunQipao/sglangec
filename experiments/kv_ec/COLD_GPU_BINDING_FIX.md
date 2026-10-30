# TP2 shared Stage E cold capture binding failure (2026-10-30)

## Evidence and scope

Failure source: commit `1ce9a343a9ad4f53b21cd0354ac49378abdac11e`, trial
`trial-000-16593f3e8c5d4bbd9a771d530ca93f69` under
`/media/sda1/iic/luotao/kv-ec-perf-tp2-shared/run.CKe384hS/stage-e-smoke-r1`.
Node0 files were read directly. Node1 files were read from the trial's
`json-evidence-233.zip`, including `origin/server.log`, both Rank reports,
`run.json`, `check.json`, and both `capture/rank_*.meta.json`.
Direct SSH was attempted and rejected by this execution sandbox (`socket:
Operation not permitted`). Node1's live filesystem and binary checkpoint have
not been independently reread; the archive is the evidence source for that node.

| Node | TP Rank | physical index | local CUDA | admitted UUID (GPU- prefix omitted) | reported runtime UUID |
|---|---|---|---|---|---|
| 232 | 0 | 1 | cuda:0 | b5c4bcac-9041-35b8-ea26-939c462bc740 | same |
| 232 | 1 | 2 | cuda:1 | 7d7259f5-2df4-e6e4-9cdc-5bae71101612 | same |
| 233 | 0 | 2 | cuda:0 | a6984ac4-fa44-c087-7069-734b7e0482a0 | same |
| 233 | 1 | 3 | cuda:1 | dc33e507-c67b-85a8-4ed9-37c7e92f7c14 | same |

All four reports record `cuda_executed=true`, `pass_reference=true` and
16 resume Decode forwards. Both independent reference checks are FAIL with
`RestoreRejected:cold_capture_gpu_binding_failed`. Reported reference completion
is not independent acceptance. Origin logs record successful weight loads, KV
allocation and HTTP `/generate` 200; the KeyboardInterrupt tracebacks follow
the experiment's owned-process cleanup and are not the first functional failure.

The old check in `verify_gpu_cold_resume.read_evidence()` tested three fields:

| comparison | actual in all four capture metadata records | expected | result |
|---|---|---|---|
| gpu_id == rank | missing / None | 0 or 1 (CUDA ordinal, NOT physical index) | false |
| cuda_visible_devices == config.gpus | missing / None | 1,2 on 232; 2,3 on 233 | false |
| normalized gpu_uuid == admitted UUID | missing / None | the UUID for that node and Rank | false |

The loop fails first at Rank0; evaluating the same comparisons independently
shows Rank1 would also fail on each node. Node0's actual `global_node.bin` was
unpacked and has the same missing fields as its sidecars.

Reproduce without launching services or modifying the failed trial:

```bash
cd /media/sda1/iic/luotao/sglangec
export PYTHONPATH="$PWD/python:$PWD"
/media/sda1/iic/miniconda3/envs/sglang-ec/bin/python - <<'PY'
import json, pathlib, zipfile
from experiments.kv_ec.gpu_admission import normalized_uuid, validate_rank_binding
from experiments.kv_ec.node_block_export import unpack
p = pathlib.Path('/media/sda1/iic/luotao/kv-ec-perf-tp2-shared/run.CKe384hS/stage-e-smoke-r1/trial-000-16593f3e8c5d4bbd9a771d530ca93f69')
with zipfile.ZipFile(p / 'json-evidence-233.zip') as z:
    for node in (0, 1):
        def read(name):
            return json.loads((p / name).read_text() if node == 0 else z.read(name))
        root = f'node{node}-reference'
        cfg = read(root + '/run.json')
        print('node', node, 'check', read(root + '/check.json')['failure_reason'])
        for rank in (0, 1):
            binding = read(f'{root}/rank_{rank}.json')['rank_gpu_binding']
            validate_rank_binding(binding, cfg, rank)
            captured = read(f'capture/rank_{rank}.meta.json')
            if node == 0:
                packed = unpack((p / 'capture/global_node.bin').read_bytes())[0]['ranks'][rank]
                assert packed == captured
            checks = dict(gpu_id=captured.get('gpu_id') == rank,
                cuda_visible_devices=captured.get('cuda_visible_devices') == cfg['gpus'],
                gpu_uuid=normalized_uuid(captured.get('gpu_uuid')) == normalized_uuid(binding['gpu_uuid']))
            print(rank, binding, 'old_capture_comparisons=', checks)
PY
```

Root cause: `GPUColdResumeHook.after_forward()` uses
`OnlineProtector._stage()`/`finish_metadata()`, which never collected these device
fields. The checker assumed the richer debug-capture metadata schema. This is
missing runtime identity evidence caused by a producer/consumer mismatch, not
evidence of an incorrect actual CUDA binding. Neither shared admission nor a
physical index different from Rank is itself invalid. `1,2` and `2,3` are
consecutive physical selections with a nonzero offset; `1,3` is also tested as
a genuinely non-contiguous selection, and `3,1` tests selected order.

## Minimal implementation

After the existing D2H completion fence, the cold hook re-observes the actual
KV pool CUDA device and its UUID via `torch.cuda.get_device_properties(device)`.
The existing runtime binding check verifies device ordinal, scheduler gpu_id,
Rank, ordered CUDA_VISIBLE_DEVICES and admitted physical UUID before any rank
artifact is written. The observed identity is then sealed in that Rank's
metadata and thus in the canonical Node Block hash. The independent checker
requires all the old fields plus tp_rank/tp_size/cuda_device/physical_gpu_index,
and reports every failing field. UUIDs must be present and complete 128-bit GPU
UUIDs; empty/None/unavailable identities cannot match each other.

No RS framing, RDMA protocol, KV layout/hash/readback, or Scheduler restore
semantics change. The old trial remains FAIL; no metadata may be backfilled
from startup reports. A new capture is mandatory.

Validation performed locally on 232 with the existing `sglang-ec` Python:

- New regression module: 9 tests, OK, including the actual cold `after_forward`
  producer path with CPU data/mocked CUDA and rejection before artifact writes.
- The 13-module CPU command below: 297 tests in 42.486 seconds, OK, 7 skipped.
  Skips cover forbidden socket binds, unavailable CUDA, and hardware-only/native
  RDMA prerequisites or explicit opt-in. They are not acceptance successes.
- `test_launch_online`, `test_shared_gpu_debug`, `test_capture_profiling`:
  27 tests in 1.480 seconds, OK.
- `git diff --check` and `compileall`: successful. All 10 runbook Bash blocks
  pass `bash -n`; the generated jq configuration passes `validate_config` in a
  read-only dry run. These checks do not execute deployment or the smoke run.
- Read-only audit of the unchanged Node0 capture still raises
  `cold_capture_gpu_binding_failed`, now naming each missing device field.

No live SSH deployment, new GPU capture, two-host preflight, or real GPU/RDMA
smoke acceptance was performed in this sandbox.

## Deployment (commands for the operator, not executed by the diagnosis)

All commands below stop on failure. Preserve the old trial. Do not stop unknown
GPU jobs or old RS services. Fresh ports below are examples that must be free;
if occupied choose other ports and update the new configuration consistently.

On 232, deploy only the repair files to a clean checkout of the same base commit
on 233. The remote clean-tree guard intentionally prevents overwriting local work.

```bash
set -euo pipefail
cd /media/sda1/iic/luotao/sglangec
test "$(git rev-parse HEAD)" = 1ce9a343a9ad4f53b21cd0354ac49378abdac11e
ssh -o BatchMode=yes -o ConnectTimeout=10 iic@10.102.0.233 '
  set -eu
  cd /media/sda1/iic_data/luotao/sglangec/sglang
  test "$(git rev-parse HEAD)" = 1ce9a343a9ad4f53b21cd0354ac49378abdac11e
  test -z "$(git status --porcelain)"
'
rsync -avR \
  ./experiments/kv_ec/gpu_admission.py \
  ./experiments/kv_ec/gpu_cold_resume_hook.py \
  ./experiments/kv_ec/verify_gpu_cold_resume.py \
  ./experiments/kv_ec/test_cold_gpu_binding.py \
  ./experiments/kv_ec/COLD_GPU_BINDING_FIX.md \
  iic@10.102.0.233:/media/sda1/iic_data/luotao/sglangec/sglang/
diff <(sha256sum experiments/kv_ec/{gpu_admission,gpu_cold_resume_hook,verify_gpu_cold_resume,test_cold_gpu_binding}.py) \
     <(ssh iic@10.102.0.233 'cd /media/sda1/iic_data/luotao/sglangec/sglang && sha256sum experiments/kv_ec/{gpu_admission,gpu_cold_resume_hook,verify_gpu_cold_resume,test_cold_gpu_binding}.py')
```

On **each node**, initialize its correct environment (also use this initialization
in each new terminal used for bootstrap). The expected host names below come
from the failed run's recorded process identities.

```bash
set -euo pipefail
case "$(hostname)" in
  sdu-232)
    export NODE=0 GPUS=1,2
    export REPO=/media/sda1/iic/luotao/sglangec
    export PY=/media/sda1/iic/miniconda3/envs/sglang-ec/bin/python
    export MODEL=/media/sda1/iic/luotao/models/Qwen3-8B
    export CUDA_ROOT=/usr/local/cuda-13.3 ;;
  disai-4090-3)
    export NODE=1 GPUS=2,3
    export REPO=/media/sda1/iic_data/luotao/sglangec/sglang
    export PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
    export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
    export CUDA_ROOT=/media/sda1/iic_data/anaconda3/envs/sglang-ec ;;
  *) echo 'Unexpected hostname; set node-specific paths explicitly'; exit 1 ;;
esac
cd "$REPO"
export PYTHONPATH="$PWD/python:$PWD"
export CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$CUDA_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CUDA_ROOT/targets/x86_64-linux/include${CPATH:+:$CPATH}"
git diff --check
"$PY" -m compileall -q experiments/kv_ec
"$PY" -m unittest -v \
  experiments.kv_ec.test_cold_gpu_binding experiments.kv_ec.test_tp4_stage_e \
  experiments.kv_ec.test_gpu_kv_restore experiments.kv_ec.test_gpu_restore_hook \
  experiments.kv_ec.test_gpu_resume experiments.kv_ec.test_gpu_cold_resume \
  experiments.kv_ec.test_rs_cold_resume experiments.kv_ec.test_perf_stage_e \
  experiments.kv_ec.test_online_chunk experiments.kv_ec.test_online_rs \
  experiments.kv_ec.test_remote_rs experiments.kv_ec.test_rdma_transport \
  experiments.kv_ec.test_rdma_upload_control
```

These are CPU regressions. Socket tests may SKIP when the sandbox forbids binds;
neither CPU PASS nor mocked RDMA tests establish real GPU/RDMA acceptance.

## Fresh templates on both nodes (no old capture/checkpoint reuse)

On 232 generate one unique tag and copy its **printed literal export** into all
bootstrap/RS/coordinator terminals on both machines:

```bash
export TAG=cold-binding-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)
printf 'export TAG=%s\n' "$TAG"
```

After the environment initialization above and setting the same TAG, run this
in every bootstrap terminal on each node:

```bash
export BOOT=/tmp/kv-ec-binding-$TAG-node$NODE
export RID=$TAG-node$NODE
export BPORT=$((32980 + NODE))
```

Terminal A on each node (new local manager, foreground):

```bash
mkdir "$BOOT"  # must not already exist
"$PY" -u -m experiments.kv_ec.online_chunk_manager \
  --socket "$BOOT/manager.sock" --output "$BOOT/local" --run-id "$RID" \
  --retain-ranks --test-recovery 2>&1 | tee "$BOOT/manager.log"
```

Terminal B on each node (new owned GPU server, foreground):

```bash
"$PY" -u -m experiments.kv_ec.launch_online \
  --model "$MODEL" --tp 2 --gpus "$GPUS" --gpu-admission-mode shared \
  --mem-fraction-static .5 --prefill-chunk-size -1 \
  --socket "$BOOT/manager.sock" --run-id "$RID" --port "$BPORT" \
  2>&1 | tee "$BOOT/server.log"
```

Terminal C on each node, after its server is ready:

```bash
curl --fail --silent --show-error "http://127.0.0.1:$BPORT/health"
"$PY" -m experiments.kv_ec.request_debug request \
  --model "$MODEL" --url "http://127.0.0.1:$BPORT" --prompt-tokens 384 \
  --request-id "$RID" --output "$BOOT/request.json"
SNAP=$("$PY" -c 'import hashlib,sys; r=sys.argv[1]; h=hashlib.sha256(r.encode()).hexdigest(); print(hashlib.sha256(f"{r}:{h}:0".encode()).hexdigest())' "$RID")
"$PY" -m experiments.kv_ec.node_block_export --snapshot "$BOOT/local/$SNAP" \
  --node-id "$NODE" --output "$BOOT/template"
"$PY" -m experiments.kv_ec.verify_gpu_restore public-test-inputs \
  --checkpoint "$BOOT/template/node.bin" --model "$MODEL" --output "$BOOT/input_ids.json"
"$PY" -m experiments.kv_ec.verify_gpu_restore inspect --checkpoint "$BOOT/template/node.bin"
```

Manager persistence is asynchronous: if the snapshot is not yet committed,
wait for its `manifest.json` state COMMITTED and retry export only if the new
export directory does not exist. Do not overwrite artifacts. Stop only the two
foreground processes you started, using Ctrl-C in their own terminals, and wait
for their GPU workers to exit before preflight. Do not use pkill/killall/reset.
Bootstrap's new template is only layout/model/input admission; the smoke driver
will still produce new cold captures and checkpoints for this new trial.

## Fresh RS service on 233

Use the 233 environment and same TAG. Confirm TCP 32700 and RDMA port range
32720–32847 are free (`ss -ltn`). Do not kill listeners. Start a new foreground
service in a new directory, keeping existing services untouched:

```bash
RS_RUN=$(mktemp -d /tmp/kv-rs-binding-$TAG.XXXXXXXX)
"$PY" -u -m experiments.kv_ec.remote_rs_server \
  --host 192.168.2.245 --port 32700 --ttl 14400 --timeout 180 \
  --max-stripes 32 --max-bytes 17179869184 \
  --rdma-device mlx5_0 --rdma-gid-index 3 --rdma-port-num 1 \
  --rdma-port-base 32720 --rdma-timeout 180 \
  --rdma-directory "$RS_RUN/rs-server" 2>&1 | tee "$RS_RUN/server.log"
```

## New configuration, preflight and Stage E smoke on 232

Use the 232 environment and same TAG. The old JSON below supplies only the
node-local installation/CUDA/HCA settings; all templates, input files, output
roots, service ports and Stripe/ledger/capture lifetimes are new.

```bash
RUN=/media/sda1/iic/luotao/kv-ec-binding-$TAG-run
mkdir "$RUN"
BASE=/media/sda1/iic/luotao/kv-ec-perf-tp2-shared/run.CKe384hS/config.json
jq --arg tag "$TAG" '
  .tp_size=2 | .prefix_tokens=256 | .rs=[2,1] |
  .gpu_admission={mode:"shared"} | .mem_fraction_static=0.5 |
  .nodes[0].gpus="1,2" | .nodes[1].gpus="2,3" |
  .nodes[0].template=("/tmp/kv-ec-binding-"+$tag+"-node0/template/node.bin") |
  .nodes[1].template=("/tmp/kv-ec-binding-"+$tag+"-node1/template/node.bin") |
  .nodes[0].input_ids=("/tmp/kv-ec-binding-"+$tag+"-node0/input_ids.json") |
  .nodes[1].input_ids=("/tmp/kv-ec-binding-"+$tag+"-node1/input_ids.json") |
  .remote_root=("/media/sda1/iic_data/luotao/kv-ec-binding-"+$tag+"-runs") |
  .rs_server.port=32700 | .return_port=32890 | .delivery_port=32892 |
  .target_port=32986 | .nodes[0].origin_port=32984 | .nodes[1].origin_port=32985
' "$BASE" > "$RUN/config.json"
CONFIG=$RUN/config.json
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$CONFIG" --output "$RUN/preflight.json" --repeat 1
jq -e '.status=="PASS" and .tp_size==2 and .gpu_admission_mode=="shared"' "$RUN/preflight.json"
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/stage-e-smoke" --repeat 1
mapfile -t TRIALS < <(find "$RUN/stage-e-smoke" -mindepth 1 -maxdepth 1 -type d -name 'trial-*')
test "${#TRIALS[@]}" -eq 1
T=${TRIALS[0]}
jq -e '.pass_fail=="PASS" and .functional_correctness_pass==true and
  .tp_size==2 and .gpu_admission_mode=="shared" and .isolated_performance_eligible==false' "$T/trial.json"
jq -e '.pass_stage_e==true and .pass_stage_d==true and .real_rdma==true and
  .gpu_restore_committed==true and .new_instance==true and
  .resume_decode_forward_count==16 and .recomputed_prefix_tokens==0 and
  .output_token_ids_match==true and ([.ranks[].tp_rank]==[0,1]) and
  (.comparisons|length)==2 and all(.comparisons[]; .output_token_ids_match==true) and
  all(.ranks[]; .cuda_executed==true and .committed==true and .verified==true and
    .scheduler_injected==true and .prefix_prefill_forward_count==0 and
    (.stats.layers_verified|length)==36 and
    all(.stats.layers_verified[]; .k_byte_exact==true and .v_byte_exact==true)) and
  ([.rank_gpu_kv_mapping[].tp_rank]==[0,1]) and
  all(.rank_gpu_kv_mapping[]; .packed_byte_count==18874368 and
    .gpu_binding.tp_rank==.tp_rank and .gpu_binding.runtime_gpu_uuid!=null)' "$T/stage-e-check.json"
jq '{tp_size,gpu_admission_mode,rank_gpu_kv_mapping}' "$T/stage-e-check.json"
```

The driver independently checks both new references before protection/recovery.
Any failed preflight or gate stops the run; inspect newly generated logs rather
than editing PASS flags or reusing its failed capture/ledger/Stripe. Successful
shared correctness still does not imply isolated performance eligibility.
