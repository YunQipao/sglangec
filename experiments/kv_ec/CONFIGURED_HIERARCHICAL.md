# 配置驱动的 TP=2/4 STRICT/SHARED 验证

## 审计与能力边界（2026-10-30）

审计起点：`feat/kv-gpu-restore-resume-20261027`，HEAD `0ffbf81b1c`
（最近提交 `optimising`、`de74b9cdc3 add tp mode`、`369553eee0 add shared mode`），
起始工作区干净。origin 指向 `git@github.com:YunQipao/sglangec.git`。
当前执行环境无法解析 github.com；未获取或声称核实 GitHub 最新版本。

| 入口 | TP | 数据与恢复含义 |
|---|---|---|
| `launch_debug baseline/capture` | 1/2/4 | 独立真实 SGLang；capture 从真实 GPU KV 只读复制 |
| `launch_debug parity` + `parity_manager` | 2/4 | 实时 Capture、本地 XOR COMMITTED；`--test-oracle` 为瞬时 CPU 字节恢复，不写回 KV Pool |
| `accept_pause` | 4 | 四 Rank Scheduler 暂停、租约、shadow/online 事务；不能当作 TP2 在线恢复入口 |
| `gpu_scratch_restore` | 按快照 Rank | CUDA scratch 分配/H2D 对照；不等于 SGLang KV Pool 恢复 |
| `gpu_kv_restore` / `verify_gpu_cold_resume` / Stage E | 2/4 | 另有真实 Pool 写回、Scheduler 重建与原始 16 Decode 对照；必须单独运行其完整验收 |
| `multinode.worker/coordinator` | 与 TP 无关 | synthetic D0/D1，真实 TCP 或显式 verbs RDMA；不捕获模型 KV |
| `multinode.kv_hierarchical` | 本次从 4 扩展为 2/4 | 请求完成后真实磁盘快照 → 本地 XOR → 两 Node Block RS；无在线 Scheduler 接入/Pool 写回 |
| `remote_rs_server.py` | Node Block 服务 | 确有独立 RS 服务，属于 Stage E 等另一协议；本流程不启动或停止该服务 |

`shared_gpu_debug` 是调试准入和性能标识，不表示内存隔离。直接 launcher 现已统一 STRICT/SHARED：
逐 GPU 检查进程、权重、KV、上下文、计算/NCCL 与捕获/恢复暂存预算，STRICT 禁止已有计算任务。
SHARED 允许其他任务；两种模式都不结束其他进程。子进程显式启用/关闭 TP 不均衡检查，覆盖父 shell
及旧别名。`accept_pause` 保留原四 Rank 约束。

底层 `distributed/bootstrap.py` 取跨 Rank 最小空闲显存；`kv_cache_configurator.py` 再从加载权重后的
最小空闲量减去静态比例对应的 runtime slack。它支持在预算允许时使用不均衡空闲显存，但不提供
每 Rank 独立 KV Pool 大小。预算按 MIN，不是平均值；较空的卡不会补偿较满的卡。
`mem_fraction_static` 不是进程显存硬上限；调低比例甚至可能导致模型权重后没有 KV 空间。
当前不保证异构 GPU 型号的 NCCL/内核支持，现场初始化失败必须 FAIL。
其他任务动态分配仍可能导致 OOM；采样只能观察，不是预留或隔离机制。

Node Block 沿用 `KVECNODE1` / `kv-ec-node-block-v1`，有显式 TP、Rank 顺序、长度、布局、哈希。
TP2 `[256,4,128]`、TP4 `[256,2,128]`，均为 36 层 BF16 K/V；单 Rank 分别
18,874,368 / 9,437,184 bytes，整 Node 原始数据均 37,748,736 bytes，封装还含 JSON 元数据。
跨节点仍 RS(2,1)，不同 TP 不可混合；布局、来源版本、模型和长度均校验，不静默 padding 本地 XOR。
跨节点 RS 采用记录原长度、零 padding 到公共宽度，恢复后截断并校验。
旧 TP4 v1 格式仍可解析；本次未转换或修改历史快照。

D0 位于 Node0，D1 与 P 都位于 Node1。只验证 D0、D1、P 单块擦除，**不容忍任意物理节点故障**。

## 统一配置

完整可执行示例为同目录 `config-example.json`，默认 TP2 SHARED TCP。
`min_free_gpu_gb=16`、`mem_fraction_static=.75` 是候选值，必须通过现场逐卡 preflight，不能当作可用保证。
统一配置是新快照流程的入口，不替换已有 Stage E JSON，不更改其历史数据。

- TP2：`"tp_size":2,"gpu_ids":[0,1]`。
- TP4：`"tp_size":4,"gpu_ids":[0,1,2,3]`。
- `gpu_mode` 取 `STRICT` 或 `SHARED`；物理 GPU 编号允许两机不同。
- `mem_fraction_static` 必须为一个有限数值，不接受数组。
- 两机代码、TP、模式、上下文、比例和网络配置相同。模型/Tokenizer 路径、GPU 编号、GID 可因主机不同，
  但模型内容和 Tokenizer 文件哈希必须一致。配置缺字段、未知字段或不支持组合均拒绝。
- 本配置要求 `rs_host == node1_host`，worker 与真实快照 receiver 是不同入口/不同端口。
- `remote_root` 用于人工进入远端仓库；工具不自动 SSH 或部署。
- RDMA 必须显式设 `transport=rdma` 和各机实测 `rdma_gid_index`；未配置拒绝，不回退 TCP。

## 两机共同准备：CPU 先行

232 和 233 都运行下面代码。不要在两机代码不一致时开始捕获；不需要 sudo。

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -o pipefail
export CFG="$PWD/experiments/kv_ec/config-example.json"
git branch --show-current
git status --short
git log -5 --oneline
python -m compileall -q experiments/kv_ec python/sglang/srt
CPU_LOG=$(mktemp /tmp/kv-ec-cpu-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log)
env -u KV_EC_TP4_LIVE_CONFIG -u KV_EC_RUN_REAL_RDMA \
  python -m unittest discover -s experiments/kv_ec -t . -p 'test_*.py' -v \
  2>&1 | tee "$CPU_LOG"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
```

正常服务器 CPU 测试预期 `OK`，硬件 opt-in 用例可跳过。若 TCP socket 权限被执行沙箱禁止，
应报告环境限制；不要删测试或将错误改成 PASS。

以下工具均为独立前台步骤，日志写入 `--run` 下，不自动启动后续实验。
端口冲突时选择新端口，不停止原服务。每次实验 `init` 创建 UTC 时间戳+随机后缀目录，
日志 exclusive-create，重复步骤会拒绝覆盖。新终端重新加载 CFG/RUN。

## 第一阶段：232 TP2 SHARED 实时 Capture + 本地 XOR

232 终端 A：

```bash
export RUN=$(python -m experiments.kv_ec.configured_debug --config "$CFG" init)
printf 'RUN=%s\n' "$RUN"
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 preflight
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 manager
```

232 终端 B（RUN 替换为 A 的输出，配置必须相同）：

```bash
export CFG="$PWD/experiments/kv_ec/config-example.json"
export RUN=/tmp/kv-ec-config-runs/REPLACE_WITH_PRINTED_RUN
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 parity
```

232 终端 C：

```bash
export CFG="$PWD/experiments/kv_ec/config-example.json"
export RUN=/tmp/kv-ec-config-runs/REPLACE_WITH_PRINTED_RUN
tail -f "$RUN/parity-node0.log"
# 看到 Server is ready 后，在另一终端或 Ctrl-C 退出 tail（不会停止模型）：
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 request
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 check-parity
```

预期：`GPU_PREFLIGHT_PASS=true`，推理响应保存 768 输入/16 输出 Token，最后：
`LOCAL_XOR_COMMITTED=true; ALL_RANK_ORACLES_BYTE_EXACT=true; GPU_RESTORE_VERIFIED=false`。
这里只保护首 256 Token，人工输入请求总长仍为既有 768 Token；不是整请求在线故障恢复。
两个 Rank 的 Oracle 都由 manager 用真实 Capture 数据瞬时执行，随后释放源块，不保留恢复用 Oracle。
`ENQUEUED` 或 HTTP 成功均不等于 COMMITTED。

日志和显存：`parity-node0.log`、`request-node0.log`、对应 `*-memory.jsonl`；含 GPU UUID、
物理编号、总/空闲显存、其他任务、before/during/after_server_ready/after_child 阶段。
进程采样每 2 秒；精确加载权重前后也看 SGLang 原生 `Load weight begin/end` 日志。
小于采样间隔的变化可能漏采，不声称捕获所有峰值。预算或观测失败保留日志并返回非零。
所有 launch/exit 元数据标记 `shared_gpu=true` 和 `isolated_performance_eligible=false`。

## 第二阶段：233 Worker / 232 Coordinator synthetic RS(2,1)

保留第一阶段日志，以下不占用 GPU。两机使用相同 RUN 字符串，可直接用第一阶段 RUN。
233 终端（共同准备已执行；RUN 手动填写）：

```bash
export RUN=/tmp/kv-ec-config-runs/REPLACE_WITH_PRINTED_RUN
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 1 worker
```

232 终端：

```bash
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 coordinator
tail -n 50 "$RUN/coordinator-node0.log"
```

上述命令准确调用以下真实入口（若选择直接运行，必须使用另一个全新输出目录）：

```bash
# 233：单 stripe worker，会在绑定身份的 commit ACK 后退出，不是长期在线 RS 服务。
python -m experiments.kv_ec.multinode.worker \
  --transport tcp --control-host 0.0.0.0 --control-port 19470 \
  --data-host 0.0.0.0 --data-port 19471 \
  --return-host 10.102.0.232 --return-port 19472 \
  --d0-bytes 1048576 --d1-bytes 786432 \
  --experiment-id "$RUN" --stripe-id "$RUN-synthetic-direct" \
  --output "$RUN/synthetic-worker-direct"
# 232：先启动上面的 233 worker。
python -m experiments.kv_ec.multinode.coordinator \
  --transport tcp --node-b 10.102.0.233 --control-port 19470 --data-port 19471 \
  --return-bind-host 0.0.0.0 --return-port 19472 \
  --d0-bytes 1048576 --d1-bytes 786432 \
  --experiment-id "$RUN" --stripe-id "$RUN-synthetic-direct" \
  --output "$RUN/synthetic-coordinator-direct"
```

Worker/Coordinator 生成 synthetic payload，传输是真实 TCP；结果含 checksum/时延/流量/COMMITTED。
独立三种单块擦除检查（CPU synthetic；不是网络/GPU 验收）：

```bash
python -m experiments.kv_ec.multinode.verify --d0-bytes 1048576 --d1-bytes 786432
```

## 第三阶段：两机真实 TP2 KV 快照 + TCP RS

第一阶段与快照阶段使用不同 RUN；仅在前台终端 Ctrl-C 停止**自己启动的模型**后再加载新模型，
不运行 killall/pkill，不操作其他任务。parity manager 可保持，快照流程不依赖它。

232 创建新 RUN，记下路径并在 233 使用同一个字符串：

```bash
export RUN=$(python -m experiments.kv_ec.configured_debug --config "$CFG" init)
printf 'RUN=%s\n' "$RUN"
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 preflight
```

233：

```bash
export RUN=/tmp/kv-ec-config-runs/REPLACE_WITH_NEW_RUN
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 1 preflight
```

232 手动复制双方 preflight 内容进行代码/配置/模型/Tokenizer 一致性 Gate；只新建当前 RUN 文件：

```bash
scp "$RUN/preflight-node0.json" "iic@10.102.0.233:$RUN/peer-node0.json"
scp "iic@10.102.0.233:$RUN/preflight-node1.json" "$RUN/peer-node1.json"
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 \
  --peer-file "$RUN/peer-node1.json" compare
```

233：

```bash
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 1 \
  --peer-file "$RUN/peer-node0.json" compare
```

两边均要求 `TWO_NODE_IDENTITY_MATCH=true`。源文件哈希包括未提交源码与递归 multinode/native/runtime。
不同物理 GPU 编号不影响身份匹配。执行中改代码/配置将拒绝传输，应重新创建 RUN 并捕获。

232 前台启动 capture，233 前台也启动各自 capture：

```bash
# 232 模型终端
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 capture
# 233 模型终端
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 1 capture
```

各服务器另开终端，等待其模型 ready 后分别请求：

```bash
# 232 请求终端
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 request
# 233 请求终端
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 1 request
```

会产生三个连续 256-token chunk，每 chunk TP2 有 `rank_0/1.bin/.json`。
本最小验收传输 chunk-1；chunk-2/3 同理，但必须选择新的 stripe/output，并对应相同 token range。
发送前工具验证真实 provenance、两 Rank、布局、请求/Token hash、原 Capture 模型指纹与 16 输出长度。
这两个节点的请求身份可以不同；同节点 Rank 必须同一次请求/版本。

233 先接收（真实快照 receiver，非 synthetic worker）：

```bash
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 1 receive
```

232 发送：

```bash
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 0 send
```

233 接收完成后自动化复验：

```bash
python -m experiments.kv_ec.configured_debug --config "$CFG" --run "$RUN" --node 1 verify-rs
cat "$RUN/verify-rs-node1.log"
python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
r = json.loads((root/'snapshot-receive/hierarchical-result.json').read_text())
assert r['state'] == 'COMMITTED' and r['tp_size'] in (2,4)
assert r['stripe']['rs_parameters']['k'] == 2 and r['stripe']['rs_parameters']['m'] == 1
assert r['shared_node_failure_recovery'] is False
assert all(r['verification'].values())
assert len(r['local_xor']['rank_recovery_checks']) == r['tp_size']
exit_record = json.loads((root/'verify-rs-node1-exit.json').read_text())
assert exit_record['exit_code'] == 0
assert exit_record['shared_gpu'] == (exit_record['gpu_mode'] == 'SHARED')
assert exit_record['isolated_performance_eligible'] is False
print('REAL_SNAPSHOT_RS_PASS=true; ONLINE_GPU_RESTORE_VERIFIED=false')
PY
```

预期 verifier JSON `state=PASS`、`tp_size=2`，D0/D1 byte-exact、P recomputation、每 Rank checksum 全部 true。
快照结果保留 `D0.bin/D1.bin/P.bin`、原长度、padding、hash、encode/recovery/transfer ms、cross_node_bytes。
TCP 结果不能报告为 RDMA。RDMA 另用新 RUN、现场 GID、相同命令，并核查 verbs completion 原始证据。

## 修改文件与验证范围

| 文件 | 原因 |
|---|---|
| `config-example.json` | 完整 TP2 SHARED TCP 配置 |
| `configured_debug.py` | 串接已有 CLI、明确节点角色/数据类型、逐步手动运行、新日志、显存采样、双机来源 Gate |
| `debug_resources.py` | 统一逐卡 STRICT/SHARED 准入，复用预算函数 |
| `launch_debug.py` | 准入覆盖所有模式；Tokenizer/min-free/GPU mode 参数；显式子进程策略 |
| `debug_capture.py` | 新 Capture 绑定实际 GPU UUID/物理编号与资源模式；保留旧快照解析 |
| `accept_pause.py` | 保留 TP4；显式 strict/shared 子进程环境；shared 采用真实 MIN-free 预算 |
| `multinode/kv_hierarchical.py` | TP2/4 布局、头范围、完整 Rank 封装/解码与兼容检查 |
| `multinode/verify_hierarchical.py` | TP2/4 验证；重新验证双节点兼容性与本地快照精确绑定 D1 |
| `multinode/tests/test_kv_hierarchical.py` | 现有 fixture 支持 TP2，保留 TP4 |
| `test_configured_debug.py` | 配置矩阵/错误/协议区分、逐卡显存/strict/策略、TP2 Node Block XOR/RS 生产路径 |
| `test_debug_capture.py`, `test_shared_gpu_debug.py` | launcher 测试显式替代 GPU 准入，不探测用户 GPU，不替代 Capture/恢复 Gate |
| `test_capture_profiling.py` | CPU-backed 真实 Capture 路径检验 UUID 绑定与错误设备拒绝，非硬件证据 |
| 本文、`TP2_ONLINE.md`、`multinode/REAL_KV_HIERARCHICAL.md` | 审计、运行命令、边界与新入口 |

当前执行环境无 `/dev/nvidia*`，未运行真实 GPU、两机 TCP/RDMA 或新在线 Restore。
CPU synthetic fixtures + mock transport 只验证生产 Node Block/XOR/RS 编解码，不证明物理网络/GPU。
本次保留已有 TP2/4 Stage E 路径，但该路径的新实例 Pool 写回/16 Decode 必须按
`TP_SELECTABLE_STAGE_E.md` 单独运行，不能用本 runbook 的 Oracle/快照 PASS 替代。
下一步优先现场完成 TP2 SHARED 本地 XOR与 TCP 快照验收，再跑真实 RDMA和已有 Stage E TP2 cold restore；
四 Rank online `accept_pause` 泛化 TP2 需另行实现/验收完整租约与跨 Rank 状态机，不机械减少 Rank。

本次验证记录：`compileall`、`git diff --check`、完整 JSON 及现场可读模型/Tokenizer 结构检查通过；
本文 17 个 bash block 均通过 `bash -n`。全量 discovery 执行 742 项：717 通过、23 跳过、2 错误，
两错误均为 sandbox 在 `socket.socket()` 拒绝权限，并非将其改为跳过/通过。
全量之后补强旧封装与 UUID 诊断的最终针对性回归 75 项全部通过。
日志 `/tmp/kv-config-full-final.log`、`/tmp/kv-config-regression-final.log`。
没有运行任何真实 GPU/RDMA/双机服务，没有 commit/push、清理数据或终止既有 GPU/RS 任务。
