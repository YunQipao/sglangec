# Stage E 256-token TP2 性能基线

本方案只接受两台物理机上的真实 Stage E PASS。默认观测关闭；启用时只
设置 `KV_EC_PERF_DIR`、`KV_EC_PERF_RUN_ID` 和 `KV_EC_PERF_ROLE`，不会改变
Scheduler、GPU KV 写回、RS 或 RDMA 数据通路。每个进程使用自己的
`monotonic_ns`；GPU H2D/读回和 native verbs 使用已有 CUDA Event/CQ 统计。
不同主机的时间戳不相减，`inclusive` 区域不相加。

## 计时字段

`perf-*.jsonl` 中的 `stage/category/source` 对应：

| 指标 | 进程/主机 | 计时来源 |
|---|---|---|
| `gpu_kv_snapshot`, `gpu_snapshot_gather` | origin Scheduler，232/233 | 本进程 monotonic；已有 CUDA stream fence |
| `gpu_to_host` | origin TP rank，232/233 | CUDA Event |
| `local_xor_encode`, `local_xor_validation` | OnlineChunkManager，源节点 | monotonic；validation 单列 |
| `rs_upload_inclusive`, native `rdma_native` | exporter client，232/233 | monotonic 外围；verbs CQ 原生字段 |
| `rs_parity_encode`, `global_committed` | RS server，233 | server monotonic；GLOBAL_COMMITTED 是状态标记 |
| `rs_recovery_rpc_inclusive`, native recovery legs | 232 client/233 server | monotonic 外围；四条 CQ 记录 |
| `delivery_send_inclusive`, `delivery_receive_inclusive` | 232/233 | monotonic 外围；verbs CQ |
| `gpu_kv_writeback` | target TP rank，233 | monotonic 外围；已有 CUDA Event |
| `gpu_layer_readback_validation` | target TP rank，233 | monotonic；实验校验单列 |
| `scheduler_request_rebuild` | target Scheduler，233 | monotonic |
| `first_restored_decode_cuda` | target TP rank，233 | CUDA Event |
| `model_weight_load` | target ModelRunner，233 | 已有 `weight_load_time` |

观测异常只写日志并被忽略；不会让正常服务失败。现有
`stats.h2d_ms/verify_ms`、`rank_performance_ms`、`server_rs_encode_ms` 和
`native_stats` 是原始数据，报告不得把校验耗时伪装成必要开销。

## 对照组

Stage E `repeat/trial` 的服务就绪探针仅访问 `/health`，要求 HTTP 200，
不解析响应体、不跟随重定向，并在请求前后核验本轮进程身份与退出收据。
原有 Stage A launcher 已设置 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`；
此时 `/health` 在 HTTP 层直接返回，不生成请求，也不调用内部状态 RPC。
不要使用 `/server_info` 或 `/health_generate` 判断 Restore Hook 服务就绪。

`perf_benchmark inference --scheme no-ec` 使用同一 Qwen3-8B、256 token、TP2、
采样参数和 17 个输出；服务必须关闭 radix cache、overlap、chunked prefill，
CUDA graph disabled。`reprefill` 是无 EC 的故障后重新 Prefill 控制，启动另一个
同配置服务并重复提交同一输入；其 CLI 明确标注不包含故障发现时间。
该独立 `inference` 命令仍使用 `/server_info` 严格核对配置，只能指向单独的
普通推理服务，不能指向实验 Restore Hook 服务；`repeat/trial` 不调用它。

`xor-only` 接受真实 GPU Capture，测量受控单 TP rank 丢失的局部 XOR；它不声称
GPU/Scheduler 恢复。`rs-only` 应使用现有 Stage C/`physical_two_node_rs.py` 的
Node Block 故障入口；若没有真实 Stage C PASS，结果必须保持 NOT IMPLEMENTED。
`two-level` 仅由 Stage E `check` 通过的真实运行产生。

## 从 0 到 1：配置与前置材料

完整的、可复制的配置文件随代码提供：

- 232：`experiments/kv_ec/perf_stage_e_232.example.json`，包含本地 Node 0 与 SSH Node 1。
- 233：`experiments/kv_ec/perf_stage_e_233.env.example`，只配置人工启动的常驻 RS。

两台机器先切到 `feat/kv-gpu-restore-resume-20261027` 的相同 Commit，部署本次
修改（包括新增 Python 文件），保持相同 tracked diff 和 `perf_*.py` 文件哈希。
这里不提供自动 checkout、commit、push 或覆盖远端仓库的操作。
两台机器需要可用的 `sglang-ec` Conda 环境、`cc`、libibverbs 与 verbs.h。
协调器将每个节点配置的 CUDA bin、`conda_prefix/bin` 放在继承 PATH 前面；
SSH 保留远端原有 PATH 后缀，不带入 232 的 PATH。preflight 使用与 Trial 相同的
`Host.argv` 环境检查 Ninja 路径及 `ninja --version`，报告写入 `checks.ninja`。
缺失、无法执行或超时会拒绝启动；不修改系统配置或 Conda 环境。
232 的 SSH agent/公钥须能通过 `ssh -o BatchMode=yes iic@10.102.0.233 true`。
管理网络用于 SSH/SCP；192.168.2.243/245 是历史 RDMA 网地址，须按现场确认。

模板不是恢复点：它只提供 Stage A 模型/布局/输入准入，不能用其他请求的模板
拼接历史请求状态。先按 `GPU_RS_COLD_RESUME_STAGE_E.md` 准备两台本地有效的
`global_node.bin` 模板和匹配的 256 个 input IDs。若已具备历史有效模板可直接
使用；每轮仍重新 Capture 两个独立请求，并产生新的 snapshot/restore/Stripe ID。
任意缺失材料、模型字节指纹不匹配、显存不足都会拒绝，不会生成示例 KV 替代。

示例使用每机空闲 GPU `0,1`，预检和原有 launch 准入都拒绝其上已有 compute
进程。请调整为现场空闲卡；不支持自动清卡。JSON 内的 Python、Conda、模型、
模板和 input IDs 路径都是可编辑的现场参数，不保证每台安装一致。

### 1. 233：人工启动一次 RS

在独立终端执行并保持前台运行：

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source experiments/kv_ec/perf_stage_e_233.env.example
export PATH="$CONDA_PREFIX_233/bin:$PATH"
export PYTHONPATH="$REPO/python:$REPO"
unset KV_EC_PERF_DIR KV_EC_PERF_RUN_ID KV_EC_PERF_ROLE
export RS_RUN="$(mktemp -d "$RS_RUN_PARENT/kv-rs-perf-233.XXXXXXXX")"
set -o pipefail
python -u -m experiments.kv_ec.remote_rs_server \
  --host "$RS_HOST" --port "$RS_PORT" --ttl "$RS_TTL" --timeout "$RS_TIMEOUT" \
  --max-stripes "$RS_MAX_STRIPES" --max-bytes "$RS_MAX_BYTES" \
  --rdma-device "$RS_RDMA_DEVICE" --rdma-gid-index "$RS_GID_INDEX" --rdma-port-num 1 \
  --rdma-port-base "$RS_RDMA_PORT_BASE" --rdma-timeout "$RS_TIMEOUT" \
  --rdma-directory "$RS_RUN/rs-server" 2>&1 | tee "$RS_RUN/rs-server.log"
```

这是原来的 native RS Server CLI。不要省略显式预算/HCA参数：当前协议没有预算
查询 RPC，预检需要通过 233 `/proc` 验证实际 LISTEN socket 对应的进程与 argv，
再用已有 `health/get_metrics` 查询 READY、epoch、Stripe 数、resident bytes。
若已有 RS 服务占端口，命令必须失败；不要停止它，另选端口并同步配置。

### 2. 232：准备并执行 Preflight

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD"
export RUN="$(mktemp -d /tmp/kv-perf-232.XXXXXXXX)"
cp experiments/kv_ec/perf_stage_e_232.example.json "$RUN/config.json"
# 编辑所有现场路径、空闲 GPU、HCA/GID 和端口，尤其是两台 model/template/input_ids。
${EDITOR:-vi} "$RUN/config.json"
python -m experiments.kv_ec.perf_benchmark preflight \
  --config "$RUN/config.json" --output "$RUN/preflight-r5.json" --repeat 5
```

`preflight` 写指定的本地报告，并在两端创建自建自清理的 CUDA 工具链临时目录；
不创建远端 Trial 目录、保护、Stripe、恢复 Claim 或 GPU 服务。
验证两端 Commit、diff、脚本、完整模型字节指纹、模板/输入、Conda、
两张空闲卡的可用显存与原有预算估计、HCA ACTIVE/GID、HTTP/native监听端口、
SSH、磁盘与 RS。不会运行 native RDMA 数据传输，故预检 PASS 不代表 RDMA PASS。
旧 RS 的 read-only 查询会执行其原有 TTL 维护；协调器不主动发出任何状态修改。

### CUDA Toolkit 配置与无 GPU 工具链预检

每个 `nodes[i]` 都必须配置自己的 `cuda_home`，两端可使用不同的 Toolkit。
它必须是**单个绝对根目录**，不能是 `:/usr/local/cuda`、冒号分隔的路径列表
或 `nvcc` 文件本身。缺少此配置的旧 JSON 会被拒绝，不读取交互式 Shell 的
`CUDA_HOME` 作为回退。示例 JSON 使用 Conda Toolkit 的布局：

```json
{
  "cuda_home": "/media/sda1/iic_data/anaconda3/envs/sglang-ec",
  "cuda_include_dirs": ["/media/sda1/iic_data/anaconda3/envs/sglang-ec/targets/x86_64-linux/include"],
  "cuda_library_dirs": ["/media/sda1/iic_data/anaconda3/envs/sglang-ec/lib", "/media/sda1/iic_data/anaconda3/envs/sglang-ec/targets/x86_64-linux/lib"]
}
```

这只是候选配置，两端均须按实际安装验证。标准 Toolkit 可以配置其真实根目录，
例如 `/usr/local/cuda-13.0`，配套 `include` 与 `lib64`；不支持把 `stubs` 加入
运行时库目录。省略两个目录列表时会检查根目录的 `include`、`lib64`、`lib` 和
`targets/x86_64-linux/{include,lib}` 常见布局。

`Host.argv` 同时用于 Preflight、origin、target、submit 和 service supervisor。
它覆盖 `CUDA_HOME/CUDA_PATH`，令 `CUDACXX/NVCC` 指向该 Toolkit 的 `bin/nvcc`，
并在本节点已有搜索路径前加入 CUDA bin、Conda bin、CUDA include 和 CUDA library：
`PATH`、`CPATH`、`CPLUS_INCLUDE_PATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH`。
SSH 后缀在远端展开，不传入 232 的路径；不依赖 `.bashrc` 或 `.profile`。
CUDA bin 优先于 Conda bin，防止 Conda 中另一份 nvcc 遮蔽所选 Toolkit。

完整 Preflight 会增加 `checks.cuda_toolchain`：检查 header、可执行的 nvcc、
**可供 `-lcudart` 链接的 `libcudart.so`**，只有 `libcudart.so.13` 不足以通过。
然后用 nvcc 编译包含未启动 kernel 的对象，用 `c++ -shared ... -lcudart` 链接，
最后由新 Python 进程动态加载并核对 `/proc/self/maps` 中真实加载的 cudart。
链接不额外传 `-L`，因此实际验证 `LIBRARY_PATH`，加载验证 `LD_LIBRARY_PATH`。
探针子进程强制 `CUDA_VISIBLE_DEVICES=''`，不调用 CUDA API、不加载模型。
它只在输出路径最近的已有父目录下创建并移除自己的临时文件，不清除 JIT 缓存。
因此报告的 `read_only=false` 表示有临时编译写入，`read_only_services=true`
仍表示不修改 RS 等外部服务。与 PyTorch CUDA 主版本不一致会拒绝，次版本不一致
按 PyTorch 的检查方式记录警告；工具链 PASS 不等于驱动、RoPE 或 Stage E PASS。

可以先只跑工具链探针，完全跳过模型身份、GPU 资源与 RS 检查。在 232 设置
`CONFIG` 为更新后的完整 JSON 路径，再执行：

```bash
python - "$CONFIG" <<'PY'
import base64, json, subprocess, sys
from pathlib import Path
from experiments.kv_ec.perf_stage_e import Host
cfg = json.load(open(sys.argv[1]))
encoded = base64.urlsafe_b64encode(json.dumps(cfg).encode()).decode()
for i, node in enumerate(cfg['nodes']):
    root = str(Path(sys.argv[1]).resolve().parent) if i == 0 else cfg['remote_root']
    host = Host(node, root, 'cuda-preflight', i)
    subprocess.run(host.argv('experiments.kv_ec.perf_preflight', 'cuda-probe',
        '--config-base64', encoded, '--node', i, '--output-root', root),
        cwd=node['repo'] if not node.get('ssh') else None, check=True)
PY
```

每端输出一行 `KV_EC_PREFLIGHT`，任一错误均返回非零退出码。常见错误为
`CUDA_HOME_INVALID`、`CUDA_ENV_MISMATCH`、`CUDA_HEADER_NOT_FOUND`、
`CUDA_NVCC_INVALID`、`CUDA_RUNTIME_NOT_FOUND`、`CUDA_JIT_COMPILE_FAILED`、
`CUDA_JIT_LINK_FAILED`、`CUDA_TORCH_VERSION_MISMATCH`。修正配置后使用新的
Smoke OUTPUT，不删除或复用历史 Ledger。

### 3. 232：单轮自动化试跑

```bash
python -m experiments.kv_ec.perf_benchmark repeat \
  --config "$RUN/config.json" --output "$RUN/smoke-r1" --repeat 1
```

退出码 0 且 `trial-*/trial.json` 为 `pass_fail=PASS` 才表示整条自动化通过。
仅看到 Stage E checker 的 PASS、但最后 SCP/清理/证据核验失败，整轮仍为 FAIL。
原始 checker JSON 原样保留，不添加顶层 `run_id`：身份由唯一 Trial ID 绑定的
plan/Stripe、`cold-new/run.json`、两条 TP `ranks[].run_id`、restore ID 与 source
binding 共同验证。验收收据另存文件哈希；它不是新的 Stage E verdict。

### 4. 232：五轮测量（或十轮）

```bash
python -m experiments.kv_ec.perf_benchmark repeat \
  --config "$RUN/config.json" --output "$RUN/baseline-r5" --repeat 5
# 十轮使用另一新目录：
# python -m experiments.kv_ec.perf_benchmark repeat \
#   --config "$RUN/config.json" --output "$RUN/baseline-r10" --repeat 10
```

`repeat` 自动做批次预检、逐轮再预检与 epoch 校验，配置快照写入 batch。
不要并发启动占用同一组 GPU/端口的 batch；检查与启动之间存在外部资源竞争，
最终仍由既有 launch/RDMA bind 准入拒绝冲突，不会接管已有进程。

每轮独立目录与一次性 Ledger，目标启动之前创建全新空 `restore-ledger`。
失败立即停止后续轮次并保留所有本地/远端文件，不删除 Claim，不复用恢复点。
重新执行必须指定新 OUTPUT，重新 Capture/保护；不提供 resume-old-trial。

RS 的关键限制：过期 Stripe **不会从计数字典移除**。TTL 不能回收槽位，原来
`--max-stripes 4` 不能安全启动五轮。预算按现有 Store 的 `3 * max(node_bytes)`
reservation 计算，不能仅看 resident parity。示例 32 slots / 8GiB / TTL 7200s，
`max_node_bytes=64MiB`，可容纳一次 smoke、五轮和十轮；历史未知 Stripe 则按
协议最大 160MiB/Node 保守估计。空间不足在本轮保护之前拒绝；人工为新的测量
任务选择另一空 RS 实例/端口，协调器绝不重启或清空现有 RS。
`protection_timeout=1800s` 必须不大于 RS TTL；超期即 FAIL。原有实时 checker
仍会检测中途过期、handle 变化和 RS 重启，不允许历史 PASS 替代。

### 5. 232：统计和诊断

```bash
python -m experiments.kv_ec.perf_benchmark collect \
  --roots "$RUN/baseline-r5" --output "$RUN/summary-r5.json"
python -m json.tool "$RUN/summary-r5.json"
find "$RUN/baseline-r5" -name trial.json -exec python -m json.tool {} \;
find "$RUN/baseline-r5" -name failure.json -exec python -m json.tool {} \;
find "$RUN/baseline-r5" -name cleanup.json -exec python -m json.tool {} \;
# 每个失败 trial 的 driver.log / commands.jsonl，以及：
# unzip -l "$RUN/baseline-r5/TRIAL_ID/json-evidence-233.zip"
# unzip -p "$RUN/baseline-r5/TRIAL_ID/json-evidence-233.zip" target/server.log
# 若 SSH 已断，直接去 233 对应 remote_root/TRIAL_ID 查看原始日志。
```

`collect` 允许分析失败/未完整结束的 batch，列出每个 rejected 原因；它不连接
RS，不重跑恢复。只把当前流程中实时 checker 成功且完整验证的真实 PASS 纳入
two-level 统计；缺少必需性能观测/CQ、被修改、重复或失败记录整轮拒绝。摘要
按配置与 Commit/diff/脚本身份分组，不混合配置。无成功样本时 rows 为空。
仅在 n>=5/20/100 时报告 P50/P95/P99，否则为 null；均值与样本标准差另列。
`smoke-r1` 与 `baseline-r5` 默认分开统计，避免把试跑当测量。

## 输出与进程安全

每个 trial 有唯一 ID、Commit、`preflight.json`、逐命令 `commands.jsonl`、
`trial.json` PASS/FAIL 与原因；原始 `stage-e-check.json`、`target-run.json`、
`protection.json`、`acceptance.json`、`cq-evidence.json`、`native-metrics.json`、
`resources*.jsonl` 和 233 原始 JSON/JSONL/日志压缩包保留。SCP 仅传元数据/日志，
Capture、重建与恢复交付均不通过 SCP/TCP payload。

origin、target 和一次性 native receiver 全由本轮 supervisor 管理；每个服务
创建新 session/process group，保留 PID、start_ticks、boot ID、UID、session。
停止时先验证本轮 owner marker，并验证 PID 没有复用且组成员属于自己的 session，
只向该进程组依次发 SIGINT/SIGTERM/SIGKILL；从不按端口、用户名或 GPU 批量 kill。
232 每 15s 刷新本轮 90s lease，SSH/driver 断连后 supervisor 限时回收本轮组。
远端不可达时 cleanup 明确 FAIL，保留远端路径供人工诊断。supervisor 自身遭
SIGKILL、机器断电、子进程主动逃逸 session、远端磁盘损坏等仍需人工诊断，
不承诺分布式 fencing。RS 无 supervisor stop 入口，永远不在回收列表里。

常见错误：`PORT_OCCUPIED`、`CONDA_PREFIX_MISMATCH`、`TEMPLATE_MODEL_IDENTITY_MISMATCH`、
`INSUFFICIENT_GPU_MEMORY`、`SOURCE_IDENTITY_MISMATCH`、`HCA_INACTIVE_OR_ZERO_GID`、
`RS_STRIPE_BUDGET`、`RS_BYTE_BUDGET`、`RS_TTL_TOO_SHORT`、SSH/SCP exit/timeout。
启动错误立即查对应 `origin/server.log` 或 `target/server.log`；native RDMA CQ
失败、Stripe expired、Ledger already claimed 原始异常和完整 traceback 留在
driver/服务日志。修正原因后新建 OUTPUT，不删除 failed ledger 后重试旧 Capture。

## 性能口径与限制

表中各指标独立报告，origin/target 的角色分别标记，TP rank 单列。每个阶段的
重复事件为该进程/角色/rank累计时间，不是跨 TP 的关键路径；inclusive 不能求和。
`gpu_kv_snapshot`、D2H、CPU XOR、RS encode/native upload/reconstruct/delivery、
GPU writeback/readback、request rebuild、权重加载与 first decode 分列。
额外保留 `capture_artifact_inclusive`、validation、`harness_command_*`、
`harness_scp_metadata`，后者包含 SSH 与命令工作；不将它们伪装成纯网络耗时。

`failure_to_decode_complete_harness` 是 232 本机的受控 loss→16 Decode 完成
计时，包含 SSH、实验验证、启动；不含故障发现。`new_instance_launch_to_ready_harness`
包含模型/source 全量哈希、admission、权重加载、engine init 与健康轮询。
原有 H2D/writeback 的值包括 host prepare/scatter/fence，不能称为纯 PCIe 时间。
当前没有独立的“启动时全量哈希纯耗时”或“SSH纯耗时”指标，不能通过减法虚构；
缺失时按未测量说明。模型全量哈希在每轮 preflight 中也会发生，预检属于 harness，
不计入 engine 指标，可能预热主机 page cache；当前是 warm filesystem 基线，
不执行系统 drop_caches。五轮不应宣称 P95/P99，亦不宣称 steady-state EC 开销比例。

所有 duration 都来自单一进程/主机的 monotonic 或已有 CUDA/native CQ；不进行
跨机器 monotonic 相减。CPU Mock 测试不产生 Stage E GPU/RDMA PASS，历史成功
`kv-rs-cold-e-20261029.FYgjKW/stage-e-acceptance.json` 也不会自动进入新测量统计。

CPU 回归（在具备 numpy/torch 的 Conda 环境运行，不需要 GPU 或 HCA）：

```bash
python -m unittest experiments.kv_ec.test_perf_stage_e \
  experiments.kv_ec.test_gpu_cold_resume experiments.kv_ec.test_rs_cold_resume \
  experiments.kv_ec.test_rs_recovery_source experiments.kv_ec.test_rdma_transport -v
```
