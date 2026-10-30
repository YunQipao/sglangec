# Stage E GPU 资源准入：strict / shared

TP=4 的完整配置、真实模板准备、两机 Preflight/单轮/五轮命令与兼容性说明见
[TP4_SHARED_STAGE_E.md](TP4_SHARED_STAGE_E.md)。下列旧 TP2 示例继续可用。

两台机器必须部署相同代码（包括 `gpu_admission.py`）；不自动 commit、push、
同步代码、启动 GPU 实验、重启 RS 或终止他人进程。以下真实实验命令由操作者执行。

## 配置

完整配置基于 `perf_stage_e_232.example.json`，修改实际模型、模板、输入、SSH、
端口、HCA/GID、CUDA Toolkit、输出目录后使用。模式只能在顶层配置：

```json
{
  "gpu_admission": {"mode": "shared"},
  "mem_fraction_static": 0.35
}
```

以上仅为完整配置中的两个字段，不是独立可运行配置。缺少 `gpu_admission` 默认
strict；null、字符串、缺少 mode、未知字段和非法 mode 都拒绝。
`mem_fraction_static` 仍为顶层标量，不接受数组或节点覆盖。

- strict：逐卡预算 + 禁止所选卡存在计算进程；强制本次子进程的
  `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=1`。
- shared：逐卡预算仍必须通过，允许已有计算进程和可用显存/总容量不同；
  仅为本次子进程设置该变量为 0。旧的反向环境变量也显式设置，防止覆盖新变量。
- 不能获得显存、任务清单、模型配置、权重大小时失败关闭，不默认放行。
- Preflight 后，每轮再次预检；Capture、Restore、Re-Prefill 在模型身份哈希前后
  再次做准入。之后仍可能发生竞争/OOM，只能记录失败，不能保证独占或自动抢资源。

### 当前 KV Pool 规则（不是修改推理核心）

`distributed/bootstrap.py` 取得各 rank 最小可用显存 `F`；
`mem_cache/kv_cache_configurator.py::_profile_available_bytes` 的预算为：

```text
KV budget = min(post-weight free memory) - F * (1 - mem_fraction_static)
```

不是简单按显卡总容量分配。shared 除保留原有保守的逐卡预算（权重+20%、CUDA
context 2 GiB、临时计算 2 GiB、EC staging 512 MiB、KV）外，还检查
`F * fraction >= weight_per_rank * 1.2 + context_KV`，此处准入估算的 F 会先减去
CUDA context 保留量，实际 F 由 SGLang 初始化后测量。预算是保守估计而非资源预留。
`.35` 不保证能在 24 GiB 卡上运行 Qwen3-8B TP2；盲目降低比例可能导致
`GPU_STATIC_KV_POOL_INSUFFICIENT`，提高比例又可能触发逐卡 free 不足。
如果没有可行比例，请换 GPU 或等待资源，不能跳过检查。

### 自动选卡

Stage E 批量配置要求每个 node 的物理编号数量严格等于 tp_size（2 或 4），如 `"gpus":"0,1,2,3"`，
便于可重复实验；不接受配置中的 auto。独立 Capture/Restore/Re-Prefill 启动 CLI
支持 `--gpus auto`：strict 只看空闲卡；shared 可选忙卡。先逐组验证预算，再最大化
所有 Rank 中最小的 `free-required` 安全余量，不使用平均值掩盖单卡不足。

## CPU 验证（232 和 233 各执行一次）

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
mkdir -p /media/sda1/iic_data/luotao/kv_ec_artifacts/stage-e-tmp
git branch --show-current
git rev-parse HEAD
git diff --stat
"$PY" -m unittest experiments.kv_ec.test_perf_stage_e \
  experiments.kv_ec.test_gpu_memory_budget -q
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

CPU Mock 不构成真实 GPU/RDMA PASS。233 保留人工启动的现有 RS；两模式不要求更换
RS。若未启动，按 `PERF_STAGE_E_256.md` 的 RS 启动步骤操作，勿重复绑定已用端口。
RS 跨批次仍需要足够 max-stripes、TTL 和容量；预检不足时拒绝，不假设 TTL 回收名额。

## 232：创建本次配置及输出根

使用已验证配置，不能直接使用模板中的占位路径。只读取旧配置，不修改历史实验。
选择数据盘，避免系统 `/tmp` 再次占满；两台机器的 tmpdir 必须实际存在且可写。

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
BASE=/media/sda1/iic_data/luotao/kv_ec_artifacts/gpu-admission
mkdir -p "$BASE"
RUN="$(mktemp -d "$BASE/batch.XXXXXXXX")"
CONFIG_SOURCE=/absolute/path/to/your-verified-stage-e-config.json

# 生成新配置，不覆盖已有配置；需要 jq。
jq '.gpu_admission={"mode":"strict"} | .remote_root="/media/sda1/iic_data/luotao/kv_ec_artifacts/perf-233" | .nodes[].tmpdir="/media/sda1/iic_data/luotao/kv_ec_artifacts/stage-e-tmp"' "$CONFIG_SOURCE" > "$RUN/strict.json"
jq '.gpu_admission={"mode":"shared"}' "$RUN/strict.json" > "$RUN/shared.json"
```

编辑新配置的 GPU index、比例和每节点路径；不要复用旧 Ledger/恢复点。
若需指定 `.35`，只在新配置顶层修改，先让预检确认是否可行。

## 232：双机预检及 Smoke（手工执行）

Preflight 自动 SSH 到 233，在与 Trial 相同的 Host 环境检查 CUDA/Ninja、模板、
模型、GPU、Git、磁盘、RDMA、RS。只编译自身临时 CUDA 探针，不加载模型、
不建立保护、不创建 Stripe、不启动 GPU 服务。

```bash
# 严格模式
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$RUN/strict.json" --output "$RUN/preflight-strict.json" --repeat 1

# 共享模式
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$RUN/shared.json" --output "$RUN/preflight-shared.json" --repeat 1

# 选择模式，下面以 shared 为例；strict 改成 strict.json。
CONFIG="$RUN/shared.json"

# 真实 Stage E + 同一恢复实例 paired-ready Re-Prefill 单轮
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/stage-e-smoke" --repeat 1 \
  --paired-ready-reprefill

# 独立冷启动 Re-Prefill 单轮；它不会声称 pass_stage_e
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/reprefill-smoke" --repeat 1 \
  --recovery-mode reprefill

# 五轮真实 Stage E（可去掉 paired-ready 标志）
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/stage-e-r5" --repeat 5 \
  --paired-ready-reprefill

# 五轮冷启动 Re-Prefill
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/reprefill-r5" --repeat 5 \
  --recovery-mode reprefill

"$PY" -m experiments.kv_ec.perf_benchmark collect \
  --roots "$RUN/stage-e-r5" "$RUN/reprefill-r5" --output "$RUN/summary.json"
```

每次 repeat 的输出路径必须不存在；重试使用新路径、新 Trial、一次性 Ledger。
常驻 RS 容量按所有批次累计检查；上述命令不意味着现有容量足够。

## 结果与诊断

`run.json` 记录模式、比例、TP 开关、启动前逐卡 UUID/index/total/free/required
bytes、安全余量、已有任务 PID；`preflight.json` 保留两节点报告。
批次 metadata、trial、两种 Re-Prefill 结果和 summary 均有资源模式标记。
启动前模式或比例与 coordinator 不一致时拒绝；collector 也重新核对启动证据。

- `functional_correctness_pass=true`：仅在现有真实功能 Gate 全部通过后标记。
- `isolated_performance_eligible=true`：仅已验收 strict trial 可标记；表示启动前
  检查未发现所选卡的其他计算任务，不是硬件资源锁，也不保证之后没人抢卡。
- shared 即使真实 `pass_stage_e=true`，隔离性能资格也始终 false。
- collector 不合并模式，并继续按配置、代码身份和方案分组；失败、缺失指标、
  无效验收证据不进入成功统计。历史未标记记录不自动赋予独占性能资格。
- 所有 XOR/RS/RDMA/GPU Restore/Scheduler/16 Decode/Token 一致性 Gate 保持不变。

```bash
jq '{status,error,gpu_admission_mode,nodes}' "$RUN/preflight-shared.json"
find "$RUN" -name trial.json -exec jq '{run_id,pass_fail,functional_correctness_pass,isolated_performance_eligible,gpu_admission_mode,failure_reason}' {} \;
rg -n 'INSUFFICIENT_GPU_MEMORY|GPU_STATIC_KV_POOL_INSUFFICIENT|GPU_RUNTIME_OOM|GPU_ADMISSION_MODE_MISMATCH|out of memory|Loaded weights leave no' "$RUN"
```

服务器详细日志在各 trial 的 origin/target/server.log；233 原始日志保留在
配置的 remote_root/TRIAL_ID，232 也收集 `json-evidence-233.zip` 中的退出收据。
OOM/显存变化会导致 FAIL，失败现场保留；不得把旧恢复点重新当成新 Trial。
若 remote_root 没有 raw 日志归档，可在 233 用相同 rg 命令检查对应 Trial。
