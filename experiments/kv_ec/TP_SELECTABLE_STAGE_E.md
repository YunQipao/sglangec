# Stage E：同一框架选择 TP2/TP4 × strict/shared

本文件是 `TP4_SHARED_STAGE_E.md` 的配置切换与审计补充。所有现场命令由操作者
手动执行；开发回归不启动 GPU 服务或 RS Server。不能把 CPU fixture 的断言
当作 CUDA/RDMA 验收。原 TP2 五轮结果与旧 JSON 的字面模板路径保持不动。

## 1. 写死逻辑审计

| 路径 | 审计结果与处理 |
|---|---|
| perf_preflight.py | 配置仅允许整数 TP2/4；两端各有 TP 个不同 GPU；标量 fraction；模板 Rank 数、模型、输入、真实长度与容量校验。新增安全的 `{tp_size}` 模板路径解析 |
| perf_stage_e.py | Origin 从所选真实模板获得 TP；两机仍各一个 Node Block；target 从重建 checkpoint 获得 TP；冷/paired-ready Re-Prefill 均传配置 TP |
| perf_benchmark.py | 四模式完整流程；普通 inference 的 `--tp` 默认2兼容旧命令，可显式4；汇总按 TP/模式/配置/源码分组，普通 inference 还按模型、输入、采样、GPU身份分组 |
| verify_gpu_restore.py / verify_gpu_resume.py / verify_gpu_cold_resume.py / verify_rs_cold_resume.py | 逐 Rank 等待、读取、校验、引用哈希和 Token 比对使用实际 TP；历史缺省 TP2 不改变；新报告必须验证真实 CUDA 设备绑定 |
| launch_online.py / reprefill_server.py / gpu_admission.py | TP2/4 launch；strict/shared 子进程环境显式覆盖；Origin/Restore/Re-Prefill 均重做资源准入 |
| gpu_memory_budget.py | 本地 KV heads = 8/TP；逐卡预算，不平均显存；保留旧 TP2 strict 保守阈值，共享使用 MIN-free 静态池规则 |
| gpu_restore_hook.py / gpu_resume_hook.py / gpu_cold_resume_hook.py | TP group world size 必须等于实际 TP；所有 Rank 参与四阶段 commit vote；状态封存、读回、回滚、完成标记和 Decode 验收覆盖完整 Rank 集 |
| gpu_kv_restore.py | Qwen3-8B 的36层、32 attention heads、8 KV heads、128 head dim；TP4 每层 K/V `[256,2,128]`，TP2 `[256,4,128]`；恢复按本 Rank metadata 写入，错误模型/shape/hash 拒绝 |
| cold_resume_state.py / rs_cold_source.py | 封存状态、Capture proof、rank payload hash 和源进程均按实际 TP 校验；两个逻辑节点关系保持不变 |
| rank_snapshot.py / online_chunk.py / online_chunk_manager.py / online_rs_exporter.py | 既有路径已支持完整可变 Rank 集；使用原实现，不复制 TP2 报告；局部恢复测试枚举所有单 Rank 丢失 |
| xor_codec.py / node_block_export.py | XOR 原本支持 N≥2；Node Block 已支持 TP2/4；补充 lengths 数目和 ranks 数目一致检查，不更换编码格式 |
| dual_node_check.py | 发现早期独立诊断工具仍固定两 Rank；补充 `--tp 2/4`，默认2，要求完整 Rank hash 集。其 `range(2)` 仍表示两台物理机 |
| python/sglang/srt/debug_utils/kv_ec.py、Scheduler、tp_worker、memory_pool | 现有 bridge 按每个 Scheduler 实例加载 hook；ps.tp_rank/ps.tp_size 与实际 KV pool 由适配层验证。保留原 Scheduler/allocator 的请求生命周期和恢复调用点，无需更改核心布局 |
| gpu_scratch_restore.py 的 range(2) | 是 CUDA 起止事件，不是两个 TP Rank，不修改 |
| RS submitted_nodes=[0,1]、node-id choices=(0,1) | 是两逻辑节点，继续 RS(2,1)，不改为 RS(4,1) |

### 数据格式、XOR 与容量

原 `KVNODE01` / `kv-node-v1` 不升级：framing + canonical JSON + 按 TP Rank
升序拼接的原始 payload；JSON 保存 `rank_lengths`、逐层 shape/offset/hash。
Qwen3-8B 同一 Prefix 的 TP2/4 Rank 分片长度都分别相等，因此本路径**不使用
padding，也不截断**。不等长输入在 XOR 前明确拒绝；不能把一种不支持的
异长布局静默截断成相同长度。跨节点 RS 的 Node Block 可以不同长度，由原
RS manifest 记录原长度，编码 padding 后按 manifest 恢复原始字节。

TP4 每 Rank 原始 KV 9,437,184 bytes，四 Rank 合计37,748,736 bytes；TP2
每 Rank 是其两倍，整节点 KV 总量相同，不能机械地把 `max_node_bytes` 翻倍。
Preflight 按真实模板 +1MiB、每轮按新 Capture +1MiB检查，仍受 Node Block、
传输归档和 RS 容量上限约束。超限拒绝，不复制或重解释 TP2 数据。

## 2. 四种配置与真实模板

保留 [旧默认配置](perf_stage_e_232.example.json) 的 TP2 literal 模板路径和
RS 30700，不修改旧实验数据。JSON 的 `_configuration_notes` 说明切换方法。
[可切换完整配置](perf_stage_e_tp4_shared.example.json) 保留全部两端模型/SSH/
RDMA/端口设置，使用独立 RS 31700，template 支持唯一占位符 `{tp_size}`。

| 组合 | tp_size | gpu_admission.mode | node0.gpus 示例 | node1.gpus 示例 |
|---|---:|---|---|---|
| TP2 strict | 2 | strict 或省略 gpu_admission | 0,1 | 4,5 |
| TP2 shared | 2 | shared | 0,1 | 4,5 |
| TP4 strict | 4 | strict 或省略 gpu_admission | 0,1,2,3 | 4,5,6,7 |
| TP4 shared | 4 | shared | 0,1,2,3 | 4,5,6,7 |

两机编号可不同，上表不是现场可用性声明。`mem_fraction_static: 0.35` 是合法
标量，但是否可启动取决于每卡预算；降低 fraction 可能令权重+KV静态池不足。
示例 .5 也不保证所有现场 GPU 可用。shared 允许其他计算任务和显存不均衡，
仍检查权重余量、CUDA context、KV、通信/计算暂存、Capture/Restore 预算。

**先各自准备真实 TP2 和 TP4 模板**，例如每台机器的：

```
/tmp/kv-ec-tp2-template-node0/node.bin  # 232、真实 TP2
/tmp/kv-ec-tp4-template-node0/node.bin  # 232、真实 TP4
/tmp/kv-ec-tp2-template-node1/node.bin  # 233、真实 TP2
/tmp/kv-ec-tp4-template-node1/node.bin  # 233、真实 TP4
```

生产方法见 `TP4_SHARED_STAGE_E.md` 第2节；生成 TP2 时同步用 `--tp 2`、两卡
和对应独立目录。禁止将旧 TP2 block 拆分或复制为 TP4。准备完成后，只需修改
`tp_size`、`gpu_admission.mode`、`nodes[*].gpus`、`mem_fraction_static` 四处；
模板路径自动展开。Literal 路径照旧可用，但选错 TP 时明确 FAIL。缺模板时
Preflight 拒绝，不在只读预检中偷偷启动 GPU 生成模板。

## 3. 两机代码一致性（手动，在232运行）

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
AUDIT=$(mktemp -d /tmp/kv-ec-source-audit.XXXXXXXX)
"$PY" -m experiments.kv_ec.perf_benchmark inventory --output "$AUDIT/node0.json"
ssh -o BatchMode=yes -o ConnectTimeout=10 iic@10.102.0.233 \
  'cd /media/sda1/iic_data/luotao/sglangec/sglang && env PYTHONPATH=python:. /media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python -c "import json; from experiments.kv_ec.perf_benchmark import identity; print(json.dumps(identity()))"' \
  > "$AUDIT/node1.json"
jq -s -e '.[0].commit_sha == .[1].commit_sha and
  .[0].tracked_diff_sha256 == .[1].tracked_diff_sha256 and
  .[0].benchmark_files == .[1].benchmark_files and
  .[0].host != .[1].host and .[0].boot_id != .[1].boot_id' \
  "$AUDIT/node0.json" "$AUDIT/node1.json"
```

哈希覆盖未提交/未跟踪的实验 Python、native源码，以及 SGLang runtime Python。
Preflight 还自动向两机发送同一配置，并比较两端回报的 canonical 配置 SHA256、
模型身份及现场资源。不能只比 commit SHA 而漏掉工作树改动和新增文件。

## 4. 233 RS Server 完整命令（手动、新独立端口）

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
RS_RUN=$(mktemp -d /tmp/kv-rs-selectable-233.XXXXXXXX)
set -o pipefail
"$PY" -u -m experiments.kv_ec.remote_rs_server \
  --host 192.168.2.245 --port 31700 --ttl 14400 --timeout 180 \
  --max-stripes 32 --max-bytes 17179869184 \
  --rdma-device mlx5_0 --rdma-gid-index 3 --rdma-port-num 1 \
  --rdma-port-base 31720 --rdma-timeout 180 \
  --rdma-directory "$RS_RUN/rs-server" 2>&1 | tee "$RS_RUN/rs-server.log"
```

保持前台运行。旧30700服务及传输协议不变；新31700只用于隔离本轮实验。
先确认31700、31720–31847空闲；有监听者就重新选择专用端口并同步配置，
不停止未知进程。32个slot/16GiB供下面手动实验使用，实际剩余容量以预检为准。

## 5. 232 Preflight、TP4 单轮和 paired-ready（手动）

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
RUN=$(mktemp -d /tmp/kv-ec-selectable-232.XXXXXXXX)
cp experiments/kv_ec/perf_stage_e_tp4_shared.example.json "$RUN/tp4-shared.json"
CONFIG="$RUN/tp4-shared.json"
# 先核实所有模型/模板/输入/Conda/CUDA/HCA/GID路径及实际选卡预算。
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$CONFIG" --output "$RUN/preflight.json" --repeat 1
jq -e '.status == "PASS" and .tp_size == 4 and .gpu_admission_mode == "shared"' "$RUN/preflight.json"
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/tp4-two-level-smoke" --repeat 1
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/tp4-reprefill-smoke" --repeat 1 --recovery-mode reprefill
# 可选：同一新实例恢复成功后，再发一次无缓存的完整 Re-Prefill 请求。
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/tp4-paired-smoke" --repeat 1 --paired-ready-reprefill
```

paired-ready 是同一个已接受 Stage E target 上的请求控制组，使用同模型、
Prefix、TP、物理GPU和17个输出Token（pending y0 +16 Decode）。其
`request_ms`/`prefill_ttft_ms` **不包括冷启动**，不能与 Stage E 总恢复时延
直接比较。冷 Re-Prefill 的 `fault_boundary_to_decode_complete_ms` 才含新实例
启动与明确的故障边界。采样统一为 greedy、ignore_eos、17个输出，禁用缓存/
overlap/CUDA graph；独立原始参考必须一致。

## 6. 检查四 Rank 真实 GPU Restore 证据（手动）

```bash
T=$(find "$RUN/tp4-two-level-smoke" -mindepth 1 -maxdepth 1 -type d -name 'trial-*')
jq -e '.pass_fail == "PASS" and .functional_correctness_pass == true and
  .tp_size == 4 and .gpu_admission_mode == "shared" and
  .isolated_performance_eligible == false' "$T/trial.json"
jq -e '.pass_stage_e == true and .pass_stage_d == true and .real_rdma == true and
  .gpu_restore_committed == true and .new_instance == true and
  .resume_decode_forward_count == 16 and .recomputed_prefix_tokens == 0 and
  .output_token_ids_match == true and ([.ranks[].tp_rank] == [0,1,2,3]) and
  (.comparisons|length) == 4 and all(.comparisons[]; .output_token_ids_match == true) and
  all(.ranks[]; .cuda_executed == true and .committed == true and .verified == true and
    .scheduler_injected == true and .prefix_prefill_forward_count == 0 and
    (.stats.layers_verified|length) == 36 and
    all(.stats.layers_verified[]; .k_byte_exact == true and .v_byte_exact == true)) and
  ([.rank_gpu_kv_mapping[].tp_rank] == [0,1,2,3]) and
  all(.rank_gpu_kv_mapping[]; .packed_byte_count == 9437184 and
    .gpu_binding.tp_rank == .tp_rank and .gpu_binding.runtime_gpu_uuid != null)' "$T/stage-e-check.json"
jq '{tp_size,gpu_admission_mode,model_tp_prefix_identity,rank_correctness,rank_gpu_kv_mapping}' \
  "$T/stage-e-check.json"
R=$(find "$RUN/tp4-reprefill-smoke" -mindepth 1 -maxdepth 1 -type d -name 'trial-*')
jq -e '.functional_correctness_pass == true and .isolated_performance_eligible == false' "$R/trial.json"
jq -e '.pass_reprefill == true and .tp_size == 4 and .output_token_ids_match == true' "$R/reprefill.json"
P=$(find "$RUN/tp4-paired-smoke" -mindepth 1 -maxdepth 1 -type d -name 'trial-*')
jq -e '.pass_reprefill == true and .tp_size == 4 and .output_token_ids_match == true' \
  "$P/paired-ready-reprefill.json"
```

`check.json` 中的逐层K/V哈希已由生产 checker 与重建checkpoint及原始参考比较，
不是只读取自报布尔值。新报告的 Rank→CUDA device→物理index/UUID 会与
启动准入逐卡记录核对；Capture 还与真实快照GPU身份核对。`rank_gpu_kv_mapping`
绑定对应分片 SHA256、字节数和每层布局，Trial/Summary保留此映射。
缺Rank、错误UUID、错误模型/布局/hash、GPU写回错误、OOM或任意集体投票失败
均不能接受。失败保留日志和claim，不清理他人任务，也不复用失败恢复ledger。

## 7. TP2/TP4 重复与汇总（手动，先单轮功能通过）

```bash
# 使用同一基础配置，仅改四类允许字段。下面两机都选0开始的卡仅作配置示例。
# 启动前按各自现场选卡；TP2/TP4对应的真实模板必须已经准备好。
for TP in 2 4; do
  for MODE in strict shared; do
    if [ "$TP" = 2 ]; then G0=0,1; G1=0,1; else G0=0,1,2,3; G1=0,1,2,3; fi
    jq --argjson tp "$TP" --arg mode "$MODE" --arg g0 "$G0" --arg g1 "$G1" \
      '.tp_size=$tp | .gpu_admission.mode=$mode | .nodes[0].gpus=$g0 |
       .nodes[1].gpus=$g1 | .mem_fraction_static=0.5' "$CONFIG" > "$RUN/tp${TP}-${MODE}-r5.json"
  done
done
# 一次选择一种组合；不要把四组任务并发投放到同一组GPU。
TP=4
MODE=shared
C="$RUN/tp${TP}-${MODE}-r5.json"
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$C" --output "$RUN/tp${TP}-${MODE}-preflight-r5.json" --repeat 5
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$C" --output "$RUN/tp${TP}-${MODE}-two-level-r5" --repeat 5
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$C" --output "$RUN/tp${TP}-${MODE}-reprefill-r5" --repeat 5 --recovery-mode reprefill
"$PY" -m experiments.kv_ec.perf_benchmark collect \
  --roots "$RUN/tp${TP}-${MODE}-two-level-r5" "$RUN/tp${TP}-${MODE}-reprefill-r5" \
  --output "$RUN/tp${TP}-${MODE}-summary.json"
jq -e '(.rejected|length)==0 and (.accepted_trials|length)==10' "$RUN/tp${TP}-${MODE}-summary.json"
jq '{accepted_trials,rejected,gpu_admission_context,rows}' "$RUN/tp${TP}-${MODE}-summary.json"
```

TP=2用同一命令改TP变量即可；strict需无其他计算任务并满足原不均衡检查。
每次换组合需新输出目录。一个32-slot RS服务不足以容纳全部四组合40轮加smoke，
且过期tombstone占slot；在不同组间另开专用RS服务/端口并更新配置，或操作者在
首次启动时设足够的max-stripes/max-bytes/native端口范围。预检拒绝超容量，
不重启旧服务以凑slot。不同TP/模式不混合统计；shared结果始终无独占性能资格。

## 8. 修改清单与验证状态

累积文件及原因见 `TP4_SHARED_STAGE_E.md` 的“修改范围与测试映射”；本轮额外
扩展 `dual_node_check.py` / `test_dual_node_launch.py`、可切换模板、实际CUDA设备
绑定、普通inference身份分组、源码递归hash、四组合冷/paired-ready编排测试。
`perf_stage_e_232.example.json` 只增加说明，未改旧TP2运行参数。

已实现的是生产路径和失败闭合校验；CPU测试使用显式fixture/模拟I/O验证协议、
字节编码、原生Scheduler计数器和负例，不是RDMA/CUDA性能或正确性成功证据。
本轮不执行任何真实GPU或RDMA验收；现场四组合均由操作者按上述命令验证。
旧TP2历史PASS保持原样，未提交、未推送、未覆盖历史结果。

## 9. CPU 回归入口与环境限制

先编译，再执行 unittest；清除本轮硬件 opt-in，保证不自动投放两机实验。
旧 `test_kv_ec.py` / `benchmark.py` 使用同目录绝对导入，测试入口必须包含
`experiments/kv_ec`，否则会导入环境里同名的 `benchmark` 包。

在允许本地TCP socket的两台服务器上可执行完整发现：

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
env -u KV_EC_TP4_LIVE_CONFIG PYTHONPATH=experiments/kv_ec:python:. \
  "$PY" -m compileall -q experiments/kv_ec python/sglang/srt
env -u KV_EC_TP4_LIVE_CONFIG PYTHONPATH=experiments/kv_ec:python:. \
  "$PY" -m unittest discover -s experiments/kv_ec -t .
```

当前沙箱禁止创建AF_INET socket。完整发现时
`multinode.tests.test_tcp_integration` 的两项测试在 `socket.socket()` 处收到
`PermissionError: Operation not permitted`，不能宣称通过，也不为变绿而修改
其断言或生产传输。其余模块使用以下明确排除入口执行；需要真实socket的其他
既有测试按其自身环境探测明确SKIP，真实两机硬件测试也明确SKIP：

```bash
mapfile -t CPU_TESTS < <(rg --files experiments/kv_ec -g 'test*.py' | sort |
  rg -v '/multinode/tests/test_tcp_integration.py$' |
  sed -e 's|/|.|g' -e 's|\.py$||')
env -u KV_EC_TP4_LIVE_CONFIG PYTHONPATH=experiments/kv_ec:python:. \
  "$PY" -m unittest "${CPU_TESTS[@]}"
```

本轮原始发现日志 `/tmp/kv-tp-selectable-all-cpu.log`；最终CPU联合回归日志
`/tmp/kv-tp-selectable-cpu-verified.log`；相关201项定向回归日志
`/tmp/kv-tp-selectable-targeted-final.log`。定向回归无失败。CPU编排中的mock
明示为fixture，生产入口继续拒绝fixture；这些输出不具有硬件验收资格。

2026-10-30 最终结果：`compileall` 通过；联合CPU套件 **722项，699通过、
23跳过、0失败，用时196.444秒**。另有上述2项TCP集成测试因沙箱socket权限
受限，单列为未验收，不计入通过项。相关201项定向回归全部通过，用时8.272秒。
两份运行文档合计20个bash代码块经 `bash -n` 检查通过，未执行其中现场命令。
历史TP2归档再次只读验证：37,777,310-byte恢复Node Block、两个Rank的GPU写回
证明及各自16次Decode原始Token对照均通过；仍是离线证据复核，不是新硬件实验。
