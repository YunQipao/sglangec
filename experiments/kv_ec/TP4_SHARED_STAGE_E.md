# Stage E：每机 TP=4、shared GPU、真实两机验收

完整配置：[perf_stage_e_tp4_shared.example.json](perf_stage_e_tp4_shared.example.json)。
四种组合的写死逻辑审计、自动模板选择、paired-ready、源码校验及手动运行命令见
[TP_SELECTABLE_STAGE_E.md](TP_SELECTABLE_STAGE_E.md)。
旧 TP2 配置、旧结果目录和序列化格式保留。配置 `tp_size` 为 2 或 4，
每个节点的 `gpus` 必须恰好有这么多不同的物理编号。省略 `gpu_admission`
仍为 strict；切换为 `{"mode":"strict"}` 即启用严格检查。

## 实现与数据语义

- 每台机器是一个完整逻辑节点；不是把四个 Rank 分配到四台机器。
  Qwen3-8B BF16、36 层、page size 1、NHD、Prefix 256：TP4 每个 Rank
  每层 K/V shape 均为 `[256,2,128]`，原始 Rank KV 为 9,437,184 bytes。
  四个独立 Rank payload 合计 37,748,736 bytes。TP2 每 Rank 为两倍大，
  所以 TP4 **整节点 KV 不是简单翻倍**；元数据、四份证明和封装会增大。
- Node Block 为原 `KVNODE01`：12-byte framing + canonical JSON +
  Rank 0、1、2、3 的原始字节；每个 Rank 都独立校验分段、shape、hash。
  不复制报告、不复制 KV 分片来凑四份。四个 Rank 必须具有同一模型、请求、
  token digest、snapshot、request/checkpoint version；封存状态还绑定同一
  request row、generation、pending token、源进程和源代码身份。
- 节点内 XOR 为 `P=R0 XOR R1 XOR R2 XOR R3`；单 Rank 丢失时用 P 与
  其余三个 Rank 重建。线上 manager 验证所有四种单 Rank 丢失。
  节点间仍为 RS(2,1)：D0 和 D1 各包含本机的四 Rank，RS Server 保存全节点校验。
- 保留原生上传、RS 重建、RDMA 返回及 RDMA delivery、Cold Restore、
  Scheduler request rebuild、TP collective votes、逐层 GPU 读回、独立原始
  Capture/reference 与 16 次真实 Decode 的逐 Token 对照。所有四 Rank 都验收。
- shared 不跳过 `check_shared_gpu_budget()`。逐卡核对权重 +20%、CUDA context、
  计算临时量、KV、capture staging、restore writeback；共享静态池按本仓库
  SGLang 的 MIN-rank free memory 规则估算，而不是大卡总容量或平均剩余显存。
  四卡总容量可不同，运行时仍须支持所选硬件、BF16 和 TP collectives。
  预算不是显存锁；哈希前后均重查，后续 OOM/进程退出/不完整报告必须 FAIL。
- strict 保留空闲要求、逐卡预算、SGLang TP 不均衡检查及旧 TP2 保守预算。
  仅传给子进程的环境中显式设置 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK`：
  strict=1、shared=0；不在 shell、系统或 Conda 中全局 export 此开关。
- `max_node_bytes` 示例为 96 MiB。预检检查真实 TP4 模板大小 + 1 MiB，
  每轮上传前再次检查**新 Capture 的真实大小** + 1 MiB；RS 按全节点长度预留，
  源归档还受现有 `MAX_ARCHIVE=160 MiB` 约束，任何超限失败，不截断。

## 0. 两机部署及 CPU 回归

当前实验分支为 `feat/kv-gpu-restore-resume-20261027`。两端部署相同修改，
包括新增文件；不用 commit/push 即可运行。不能只复制配置而漏掉新 Python 文件。
协调器比对 Commit、tracked diff 和全部实验 Python 文件哈希，并向两机发送
同一配置、验证两端回报的配置 SHA256。现场 Python、CUDA、模型路径均须确认。

两台机器各运行：

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
git branch --show-current
git rev-parse HEAD
git diff --stat
"$PY" -m unittest \
  experiments.kv_ec.test_tp4_stage_e \
  experiments.kv_ec.test_tp4_stage_e_live \
  experiments.kv_ec.test_tp4_physical_rs \
  experiments.kv_ec.test_gpu_memory_budget \
  experiments.kv_ec.test_perf_stage_e \
  experiments.kv_ec.test_gpu_kv_restore \
  experiments.kv_ec.test_gpu_restore_hook \
  experiments.kv_ec.test_gpu_resume \
  experiments.kv_ec.test_gpu_cold_resume \
  experiments.kv_ec.test_rs_cold_resume
"$PY" -m unittest \
  experiments.kv_ec.test_online_chunk experiments.kv_ec.test_online_rs \
  experiments.kv_ec.test_remote_rs experiments.kv_ec.test_rs_recovery_source \
  experiments.kv_ec.test_rs_experiments experiments.kv_ec.test_rdma_transport \
  experiments.kv_ec.test_rdma_upload_control experiments.kv_ec.test_launch_online \
  experiments.kv_ec.test_shared_gpu_debug experiments.kv_ec.test_scheduler_pause \
  experiments.kv_ec.test_scheduler_recovery_guard experiments.kv_ec.test_debug_capture \
  experiments.kv_ec.test_kv_pool_audit
```

不设置 `KV_EC_TP4_LIVE_CONFIG` 时，两个真实硬件测试明确 SKIP；CPU PASS
不代表硬件测试 PASS。源码内 synthetic fixture 仍被生产验证器拒绝。

## 1. 环境检查（232、233 分别执行）

初始化变量后，每台机器执行；232 上 NODE=0/GID=5，233 上 NODE=1/GID=3：

```bash
hostname
test -c /dev/nvidiactl
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
ip -br -4 addr
cat /sys/class/infiniband/mlx5_0/ports/1/state
# 232 用 5；233 用 3。还要确认该 GID 对应下面的 RDMA IPv4。
cat /sys/class/infiniband/mlx5_0/ports/1/gids/5    # 仅 232
# cat /sys/class/infiniband/mlx5_0/ports/1/gids/3  # 仅 233
ss -ltn
df -h /tmp
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
"$PY" -m experiments.kv_ec.perf_benchmark inventory --output /tmp/tp4-inventory-$(date +%s%N).json
```

232 的 RDMA 地址必须为 192.168.2.243，233 为 192.168.2.245；管理地址分别
为 10.102.0.232 / 10.102.0.233。在 **232** 检查免交互 SSH：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 iic@10.102.0.233 hostname
```

示例选物理卡 0,1,2,3，不代表这些卡目前预算足够。按显存报告调整配置；
不清理进程、不重置 GPU、不停止原 RS。异构架构/驱动不支持 TP 时由真实启动
失败关闭，shared 不是绕过运行时兼容检查。

## 2. 若没有真实 TP4 模板：先生成（两机各一次）

旧 TP2 checkpoint 不能用作 TP4 模板，也不能拆分、复制或重写旧 KV 为 TP4。
用现有在线 Capture 生产一个真实模板。模板只用于准入；Stage E 每轮仍重新
Capture 同一请求的四 Rank，并封存真正的待恢复边界。

在两台机器分别创建独立工作目录。每个新终端先执行第 0 节的环境初始化，
再设置打印出的 BOOT/RID/NODE 以及相同 MODEL：

```bash
# 232 设置 NODE=0；233 设置 NODE=1。
NODE=0
BOOT=$(mktemp -d /tmp/kv-tp4-bootstrap.XXXXXXXX)
RID=${BOOT##*/}
MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
printf 'NODE=%s BOOT=%s RID=%s\n' "$NODE" "$BOOT" "$RID"
```

终端 A（manager，前台）：

```bash
"$PY" -u -m experiments.kv_ec.online_chunk_manager \
  --socket "$BOOT/manager.sock" --output "$BOOT/local" --run-id "$RID" \
  --retain-ranks --test-recovery
```

终端 B（GPU 服务，前台；按实际安装设置 CUDA 的 bin/lib/include）：

```bash
CUDA_ROOT=/media/sda1/iic_data/anaconda3/envs/sglang-ec
env CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  PATH="$CUDA_ROOT/bin:$PATH" \
  LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CPATH="$CUDA_ROOT/targets/x86_64-linux/include${CPATH:+:$CPATH}" \
  "$PY" -u -m experiments.kv_ec.launch_online \
  --model "$MODEL" --tp 4 --gpus 0,1,2,3 --gpu-admission-mode shared \
  --mem-fraction-static .5 --prefill-chunk-size -1 \
  --socket "$BOOT/manager.sock" --run-id "$RID" --port 31980
```

终端 C（确认本次服务启动完成后请求；384-token 公共输入的首个完整 256-token
块，匹配本仓库公共模板规则；不把模板 Prefill 当作恢复执行）：

```bash
"$PY" -m experiments.kv_ec.request_debug request \
  --model "$MODEL" --url http://127.0.0.1:31980 --prompt-tokens 384 \
  --request-id "$RID" --output "$BOOT/request.json"
SNAP=$("$PY" -c 'import hashlib,sys; r=sys.argv[1]; h=hashlib.sha256(r.encode()).hexdigest(); print(hashlib.sha256(f"{r}:{h}:0".encode()).hexdigest())' "$RID")
"$PY" -m experiments.kv_ec.node_block_export --snapshot "$BOOT/local/$SNAP" \
  --node-id "$NODE" --output "/tmp/kv-ec-tp4-template-node$NODE"
"$PY" -m experiments.kv_ec.verify_gpu_restore public-test-inputs \
  --checkpoint "/tmp/kv-ec-tp4-template-node$NODE/node.bin" \
  --model "$MODEL" --output /tmp/kv-ec-tp4-input_ids.json
"$PY" -m experiments.kv_ec.verify_gpu_restore inspect \
  --checkpoint "/tmp/kv-ec-tp4-template-node$NODE/node.bin"
```

inspect 应显示 tp_size=4、四个不同来源 Rank 的字节校验信息、每 Rank 9,437,184
bytes。它仍是 CPU 检查，gpu_acceptance=false。若文件已存在，命令拒绝覆盖，
换新目录并同步改配置。完成后只在 A/B 各自终端 Ctrl-C 停止这两个自建前台
进程；确认退出且 31980 释放，再做 Stage E。不要用 pkill 或 killall。

## 3. 233：独立 RS Server 准确启动命令

新端口与旧 TP2 的 30700 分开，原服务保持不动。确认 31700 和 native 端口
31720–31847 空闲。下面使用 32 个 Stripe、16 GiB 上限、4 小时 TTL；容量仍
由预检核验，过期 tombstone 不能假设自动回收。

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
RS_RUN=$(mktemp -d /tmp/kv-rs-tp4-shared-233.XXXXXXXX)
set -o pipefail
"$PY" -u -m experiments.kv_ec.remote_rs_server \
  --host 192.168.2.245 --port 31700 --ttl 14400 --timeout 180 \
  --max-stripes 32 --max-bytes 17179869184 \
  --rdma-device mlx5_0 --rdma-gid-index 3 --rdma-port-num 1 \
  --rdma-port-base 31720 --rdma-timeout 180 \
  --rdma-directory "$RS_RUN/rs-server" 2>&1 | tee "$RS_RUN/rs-server.log"
```

保持前台 RS 运行至最后一次 live check。占用端口时选新端口并修改配置，
不接管监听者、不停止旧服务。RS 无需 GPU；self-host RDMA 上传也必须真实成功。

## 4. 232：完整配置、Preflight 和单轮真实 Stage E

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
RUN=$(mktemp -d /tmp/kv-tp4-shared-232.XXXXXXXX)
cp experiments/kv_ec/perf_stage_e_tp4_shared.example.json "$RUN/config.json"
# 按两台安装检查路径、GPU 编号、HCA/GID、CUDA Toolkit、tmpdir、模板和输入。
# 若 /tmp 容量不足，用专用数据盘改两端 remote_root/tmpdir 和本地 RUN。
CONFIG="$RUN/config.json"
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$CONFIG" --output "$RUN/preflight.json" --repeat 1
jq -e '.status == "PASS" and .tp_size == 4 and .gpu_admission_mode == "shared"' "$RUN/preflight.json"
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/stage-e-smoke" --repeat 1
```

Preflight 不创建 GPU 服务、Stripe 或 GPU context；只在两端构建/清理自己的
工具链临时探针并检查资源。Preflight PASS 的 functional_correctness_pass=false：
只有真实实验通过全部 Gate 后才置 true。

## 5. 232：单轮冷启动 Re-Prefill

```bash
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/reprefill-smoke" --repeat 1 \
  --recovery-mode reprefill
```

该控制组同样生成并验证两个全 Rank 原始 Capture/reference，完成真实 RDMA
保护后新建 TP4 普通推理实例，核对模型、输入和原始 17 个输出 ID（pending y0
+ 16 Decode）。它是 Re-Prefill 正确性，不声称 Stage E Restore PASS。

也可选择运行独立硬件验收测试（再消耗两条全新 Stripe；证据目录会打印并保留）：

```bash
KV_EC_TP4_LIVE_CONFIG="$CONFIG" "$PY" -m unittest -v experiments.kv_ec.test_tp4_stage_e_live
```

## 6. 功能 PASS/FAIL 检查

任何非零退出、failure.json、非 accepted trial、缺少 Rank 2/3 或 Token 不一致都
不能算 PASS。下面只读取本轮证据；Stage E 在线 `check` 已由 repeat 在 233
执行并要求 RS 仍有效。功能成功和共享性能隔离资格分别检查：

```bash
T=$(find "$RUN/stage-e-smoke" -mindepth 1 -maxdepth 1 -type d -name 'trial-*')
jq -e '.pass_fail == "PASS" and .functional_correctness_pass == true and
  .tp_size == 4 and .gpu_admission_mode == "shared" and
  .isolated_performance_eligible == false' "$T/trial.json"
jq -e '.pass_stage_e == true and .functional_correctness_pass == true and
  .real_rdma == true and .new_instance == true and .tp_size == 4 and
  .resume_decode_forward_count == 16 and .recomputed_prefix_tokens == 0 and
  .output_token_ids_match == true and .isolated_performance_eligible == false and
  ([.ranks[].tp_rank] == [0,1,2,3]) and (.comparisons|length) == 4 and
  all(.comparisons[]; .output_token_ids_match == true) and
  all(.ranks[]; .cuda_executed == true and .committed == true)' "$T/stage-e-check.json"
jq '{tp_size,gpu_admission_mode,mem_fraction_static,gpu_resources_before_start,
  model_tp_prefix_identity,rank_correctness}' "$T/stage-e-check.json"
R=$(find "$RUN/reprefill-smoke" -mindepth 1 -maxdepth 1 -type d -name 'trial-*')
jq -e '.functional_correctness_pass == true and .isolated_performance_eligible == false' "$R/trial.json"
jq -e '.pass_reprefill == true and .tp_size == 4 and .output_token_ids_match == true' "$R/reprefill.json"
rg -n 'INSUFFICIENT_GPU_MEMORY|GPU_STATIC_KV_POOL_INSUFFICIENT|GPU_RUNTIME_OOM|out of memory|IDENTITY_MISMATCH' "$RUN" || true
```

失败保留所有证据。用新输出目录和新 Capture 重试，不删除 Ledger claim 重新运行。

## 7. 后续五轮与汇总（首轮功能通过后再运行）

```bash
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$CONFIG" --output "$RUN/preflight-r5.json" --repeat 5
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/stage-e-r5" --repeat 5
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/reprefill-r5" --repeat 5 --recovery-mode reprefill
"$PY" -m experiments.kv_ec.perf_benchmark collect \
  --roots "$RUN/stage-e-r5" "$RUN/reprefill-r5" --output "$RUN/summary.json"
jq '{accepted_trials,rejected,gpu_admission_context,rows}' "$RUN/summary.json"
jq -e '(.rejected|length) == 0 and (.accepted_trials|length) == 10 and
  all(.rows[]; .tp_size == 4 and .gpu_admission_mode == "shared" and
    .isolated_performance_eligible == false)' "$RUN/summary.json"
```

TP2/TP4、strict/shared、不同配置和源码身份分组隔离，不混合均值。
shared 可以真实功能 PASS，但所有性能结果均标记 isolated_performance_eligible=false。

## 修改范围与测试映射

| 文件 | 原因 |
|---|---|
| perf_preflight.py | TP2/4 配置、准确 GPU 数、真实模板预算、两端配置身份、四卡证据、安全的 TP 模板路径展开 |
| perf_stage_e.py | 四 Rank target 绑定、TP4 Re-Prefill 参数、报告身份 |
| perf_benchmark.py | TP4 推理/控制组/汇总、递归源码完整性、四卡启动/分片证据，防止不同模型/输入/TP/模式混合统计 |
| gpu_admission.py | 按 TP 校验 GPU UUID/index 数量及资源证明、实际 CUDA KV pool 与物理 GPU 绑定 |
| gpu_memory_budget.py | TP4 实际 KV 分片、Capture/Restore 字节预算、共享 MIN-free 静态池 |
| launch_online.py | 自动选卡诊断从 pair 改成 rank set；原逐卡检查保留 |
| gpu_kv_restore.py | 允许真实 TP4，保留模型/layout/分段/写回检查 |
| gpu_restore_hook.py | TP4 checkpoint、world size、全 Rank 一致性投票 |
| gpu_resume_hook.py | 原生前向身份检查使用实际 TP |
| gpu_cold_resume_hook.py | 四份真实 Capture 封存、状态 hash/进程证明、四 Rank commit marker |
| cold_resume_state.py | 四 Rank KV/metadata/proof/state 校验与独立源进程身份 |
| verify_gpu_restore.py | 从真实 checkpoint 确定 TP，逐 Rank 验证 |
| verify_gpu_resume.py | 全 Rank 引用 hash、恢复统计、Token/logits 对照与完成等待 |
| verify_gpu_cold_resume.py | 四 Rank 独立原始对照、Scheduler 证据、报告资源身份与正确性 |
| rs_cold_source.py | 四 Capture proof、四种局部 XOR 恢复证明，不变 RS(2,1) |
| verify_rs_cold_resume.py | 四 Rank 真实 exporter 入口与冷恢复最终 Gate |
| reprefill_server.py | 普通新实例 TP4，启动报告记录实际 TP |
| node_block_export.py | 拒绝 rank_lengths 数量与真实 ranks 数量不符 |
| test_perf_stage_e.py / test_gpu_restore_hook.py | 更新真实 TP 及子进程启动契约的 CPU fixtures |
| test_launch_online.py | 修复旧 execv mock，使 CPU 测试拦截当前 execve，验证显式子进程环境 |
| dual_node_check.py / test_dual_node_launch.py | 独立两机健康检查支持 TP2/4，任一 Rank hash 缺失拒绝 |
| test_rs_cold_resume.py | 同一生产 adapter/codec 的 TP2、TP4 故障/封装/Delivery 回归 |
| test_tp4_stage_e.py | 四组合配置、模板选择、显存、真实设备绑定的 CPU 合约、来源、写回/模型/rank/layout、Scheduler、Token/源码及汇总负例 |
| test_tp4_stage_e_live.py | 两个无 mock 的显式两机硬件测试：新实例 16 Decode 和冷 Re-Prefill |
| perf_stage_e_tp4_shared.example.json | 独立 TP4 shared 两机配置，不覆盖 TP2 配置 |
| perf_stage_e_232.example.json | 增加 TP2/TP4 配置说明，保留旧默认运行参数和模板路径 |
| TP_SELECTABLE_STAGE_E.md | 本轮完整审计、四模式切换、两机源码一致性、冷/paired-ready 边界及手动命令 |
| TP4_SHARED_STAGE_E.md / GPU_ADMISSION_STAGE_E.md / GPU_RS_COLD_RESUME_STAGE_E.md / PERF_STAGE_E_256.md | 运行命令、数据语义、兼容性与限制 |

`xor_codec.py` 已原生支持 N>=2 的逐字节 XOR，新增四 Rank 全单丢失回归，
无需修改算法。原生 RDMA/RS codec、Scheduler/allocator 核心无需改变数据语义；
适配在其现有 Stage A/B/D Hook 层完成，不绕过这些生产路径。

15 类要求的覆盖分别见：旧 TP2 回归；TP4 两模式配置/数量负例；四卡显存差异与
逐卡不足；四 Rank XOR；来源绑定；错误 rank/layout/model；全四票 identity；
CPU native Scheduler 状态及硬件新实例对照；硬件 Re-Prefill；OOM/动态显存变化；
strict/shared 子进程环境覆盖；两机代码及配置不一致负例。

## 2026-10-30 本次兼容性验证记录

- 后续四组合补齐后的最终联合CPU回归为722项：699通过、23跳过、无失败；
  另有2项socket权限受限的TCP集成测试未验收。完整命令和限制见
  `TP_SELECTABLE_STAGE_E.md` 第9节；以下363项是前一阶段的回归记录。
- 当前分支 `feat/kv-gpu-restore-resume-20261027`，基于 `369553eee0` 修改；
  未提交、未推送。未修改旧配置或历史实验结果。
- 核心及依赖的 23 个测试模块联合回归共 363 项：353 项通过、10 项跳过，
  无失败，用时 47.417 秒；最终结果见本次执行日志 `/tmp/kv-tp4-final-cpu.log`。
  真实硬件测试保持显式 opt-in，不能以跳过充当 PASS。
- 历史目录 `kv-rs-cold-e-20261029.FYgjKW` 的原报告为 TP2 Stage E PASS。
  用当前生产代码只读验证其恢复源、封存状态、两个 Rank 逐层字节证明及原始
  reference 对照：37,777,310-byte Node Block 可解析，两个 Rank 的 16 次 Decode
  Token 完全一致。此为**历史证据离线复核**，不是新 CUDA/RDMA 实验。
- 历史 run.json 记录的 `/tmp/kv-rs-cold-e-20261029.FYgjKW/...` 绝对路径在当前
  沙箱不存在，不能直接执行其完整历史 CLI check；未改写旧 run.json 或 hash。
  上述复核显式读取仓库保留的相应原始文件。
- 当前执行环境 hostname 为 `disai-4090-3`，但无 `/dev/nvidia*`，nvidia-smi
  无法访问驱动，netlink 与 SSH socket 被沙箱拒绝。没有启动 GPU 服务、RS 服务，
  没有新的 TP4 硬件 PASS、性能数字或现场 UUID。需在真正可访问 GPU/RDMA/SSH
  的 232/233 shell 按第 1–6 节执行单轮；五轮命令只作为后续交付。

| 模式 | 配置/CPU/封装及 Gate | 新硬件验收 |
|---|---|---|
| TP2 strict（默认） | CPU 回归通过；旧真实 checkpoint/恢复证明/原始 Token 离线复核通过 | 本次未重跑，保留历史 PASS |
| TP4 strict | 配置与四 Rank CPU 路径覆盖；空闲/显存/TP 不均衡检查保留 | 待现场执行 |
| TP4 shared | 四 Rank Capture/恢复、逐卡预算、故障负例及模式隔离覆盖 | 待现场执行；即使功能 PASS 性能隔离资格仍为 false |
