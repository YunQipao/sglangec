# 在线跨节点最小集成：本地 XOR → 后台 RDMA → RS(2,1)

适用分支 `feat/physical-two-node-tp4-rs`。本轮不修改 SGLang Scheduler、
KV allocator、CUDA capture、已有离线物理节点入口及 RDMA 实现。

## 审查结论与实施计划

1. `online_chunk.py` 已将完成且通过 lifetime/fence 验证的 Chunk 变成私有
   CPU 数组，再发给 `OnlineChunkManager.submit()`。接入 GPU pool 会重复
   生命周期管理；最小安全接入点是 Manager 写完本地 XOR `manifest.json`
   之后、释放完整 rank 数组之前。
2. Manager 新增可选 `global_exporter.enqueue(directory, entries, blocks)`。
   准入只保留私有 CPU 数组引用、复制元数据和写状态；打包、哈希、源节点
   文件写入、远端注册、RDMA、远端提交轮询全部由独立有界 worker 执行。
   不延长 GPU KV/slot 生命周期。请求结束或 KV 回收不影响已拥有的 CPU 数据。
3. `node_block_export.pack(node, blocks, ranks)` 打包完整 rank 数据，而不是
   本地 XOR P。沿用它与 `RemoteRSClient.rdma_submit_node_block()`，不新增
   二进制协议、不在失败后回退 TCP。接收端仍是现有 `StripeStore` 的原生
   RDMA receive worker → `_commit_block_locked()` → `rs_codec.encode()`。
4. 用显式 request SHA256 → protection ID 配置建立全局配对；不按请求到达
   顺序或本地 snapshot ID 猜配对。每节点请求、局部 checkpoint/run/generation
   可以不同，全局 run/protection/checkpoint epoch/chunk/range 必须相同。
   导出元数据新增 `global_protection`，本地 manifest 元数据不改动。
5. 分别记录 local/global 状态。CPU Mock 先覆盖真实 Store/codec、三 Chunk、
   TP2/TP4、超时/取消/限流/损坏与三块独立恢复；最后才在两机做真实 TP4 验收。

## 状态、身份与限制

- 原有 `manifest.json.state=COMMITTED` **仅表示本地 XOR 完成**；该文件和
  原有 API 返回值保持兼容。
- 新增 `global_status.json`，`local_state=LOCAL_COMMITTED`；全局依次经过
  `QUEUED → PACKING → REGISTERING → WAITING_PEER → UPLOADING →
  WAITING_COMMIT → GLOBAL_COMMITTED`。远端 `PENDING` 绝不是全局完成。
- `GLOBAL_COMMITTED` 必须取得绑定同一 handle/manifest SHA 的服务端
  `COMMITTED`、`submitted_nodes=[0,1]` 和 parity SHA。它是**当时的提交回执**，
  不是永久保护承诺，也不等价于已完成恢复验收；默认
  `recovery_byte_exact=NOT_YET_VERIFIED`，由独立验收报告证明恢复正确性。
- 终态包括 `SKIPPED_UNMAPPED`、`DROPPED_CAPACITY`、`EXPIRED`、`FAILED`、
  `FAILED_IDENTITY_CONFLICT`、`CANCELLED`。上传开始后失败/超时的
  `remote_outcome=UNKNOWN`，不能声称远端一定未提交，也不能自动换全局身份
  重传。可用绑定的只读 status 查询核实；本轮不做自动重放/重启恢复。
- `max_pending` 包含排队和执行中的任务；`max_bytes` 限制这些任务原始
  rank 数组总字节，**不是进程 RSS 上限**。pack、descriptor 验证、native
  helper staging 还有临时副本，每个 worker 按 `MAX_NODE_BYTES=160MiB`
  限制 Node Block；应另留数倍节点块的 Host RAM 余量。
- `max_checkpoints` 限制一次 exporter 生命周期的累计准入，配合 Store
  `max_stripes/max_bytes`，适合有限轮实验。输出文件不会自动清理；Store
  tombstone 占用 stripe quota，TTL 不会重新开放相同身份。持续服务的 GC、
  崩溃恢复、重试协议不在本轮范围；新实验使用新 global run 与新目录。
- deadline 从入队开始，超时任务不会再发起新的网络操作。正在执行的
  TCP/native verbs 操作只能等其有限 timeout 返回，可能晚于任务 deadline；
  本轮没有强制中断 native helper。shutdown 取消排队/等待中的任务，最多
  join 5 秒，不停止任何 GPU 进程；强制终止进程可能留下非终态回执。
- workers 可并行等待不同 Chunk；全部 worker 都等不到对端时，后续 Chunk
  会排队或被拒绝，而不是阻塞推理；暂不实现公平调度/优先级。
- 各 rank、各节点必须匹配模型指纹、源码版本、KV layout、token 范围、
  36 层 BF16、TP2/TP4。当前沿用离线兼容规则，`model_identifier` 绝对路径
  也要求相同。token 内容和 request ID 可以不同（当前离线规则要求两节点
  request ID 与 snapshot ID 不同）。
- D0/D1 是各节点的完整序列化 `global_node.bin`，在对应源节点文件系统保留。
  本地 `parity.bin` 是 TP rank XOR，远端 P 是 D0/D1 的 RS(2,1) parity，二者
  不可互换。文件不做 fsync，不承诺掉电持久性。
- 两个物理故障域：D0 在 232，D1 和远端 P 在 233。**不能保证任意整节点
  故障恢复**：失去 233 会同时失去 D1 与 P。仅验证 D0、D1 或 P 单块擦除；
  不包含故障检测、自动 failover、GPU KV 回写或推理恢复。

## 性能口径

已有 local manifest 保留每 rank 的 `gpu_to_host_ms`、capture queue wait、
`xor_ms`、`aggregation_ms`；`online_chunk_smoke` 的 `request_ms` 测请求延迟。
新全局回执 `metrics` 记录：

| 指标 | 含义 |
| --- | --- |
| `queue_wait_ms` | 全局保护入队至 worker 执行 |
| `pack_validate_ms` / `node_persist_ms` | 节点块打包校验 / 源节点文件写入 |
| `register_ms` / `peer_wait_ms` | 远端注册 / 等待另一描述符 |
| `rdma_native` | 原有 verbs helper 的 setup、MR、QP、exchange、completion 等原始统计 |
| `rdma_upload_wall_ms` | 整个 upload API，包含控制/远端校验/轮询，可能包含 RS 编码 |
| `global_commit_wait_ms` | upload 返回后额外等待另一数据块提交 |
| `server_rs_encode_ms` / `server_commit_ms` | 接收端独立 RS 编码 / stripe 总提交耗时 |

不能把 upload wall 当作纯 RDMA 传输时间，也不能把这些重叠计时直接相加。
当前 debug rank 持久化、完整恢复验算和 D0/D1 本地保留有明显额外开销；本轮
先测正确性，不声称性能达到上线要求。失败回执保留已完成阶段的指标。

## CPU 单元与回归命令

从 `sglang/` 执行，选已有 torch/numpy 的环境，无需安装新依赖：

```bash
python -m unittest experiments.kv_ec.test_online_rs -v
python -m unittest \
  experiments.kv_ec.test_online_rs \
  experiments.kv_ec.test_online_chunk \
  experiments.kv_ec.test_debug_capture \
  experiments.kv_ec.test_parity_manager \
  experiments.kv_ec.test_chunk_protection \
  experiments.kv_ec.test_tp4_physical_rs \
  experiments.kv_ec.test_remote_rs \
  experiments.kv_ec.test_rdma_transport -q
```

CPU Mock 使用真实 `StripeStore.dispatch()` 与真实 RS 编码/恢复，只替换 native
verbs 数据搬运；Mock 回执明确 `transport=CPU_MOCK`。生产 JSON 配置禁止注入
`allow_fixture/client_factory`，没有 mock 或 TCP payload 开关。

2026-10-26 本地验证：新增 15 项 CPU 测试全部通过；与上述既有回归合计
104 项，93 项通过、11 项因 CUDA/socket 环境限制跳过；编译、CLI help 和
文档 Bash 语法检查通过。没有执行真实 GPU/RDMA 双机测试。

## 双机 TP4 启动

以下命令供人工执行；本轮不自动启动 GPU 服务、不停止已有进程。先确保两边
代码相同（包括本轮未提交文件）和相同模型文件，并对比
`git rev-parse HEAD`、`git diff HEAD -- python`、本轮文件 SHA256。不要在配置
文件中放真实用户提示词。请求使用公开合成 token。

模型安装绝对路径可以不同，例如 232 的
`/media/sda1/iic/luotao/models/Qwen3-8B` 与 233 的
`/media/sda1/iic_data/luotao/models/Qwen3-8B`。各节点必须传入自己真实的
`--model` 路径；不要重写已有快照或统一伪造 `model_identifier`。
跨节点继续严格比较 `model_sha256`、`source_commit`、`core_diff_sha256`、
`dtype`、`tp_size`、`token_start`、`token_count`、`layers` 和逐 Rank KV layout。
节点描述符仍包含原始路径，完整节点块及其 manifest/SHA 校验不变。
不兼容错误现在附带字段、两节点值；layout 错误附带 Rank 和 record 索引。

复验必须让 RS 接收服务、两边 Manager 和 SGLang 使用修改后的代码启动新实例。
选择新的全局 run、配对配置、socket、输出目录和可用端口，按下面流程重新请求
chunk 0/1/2，再运行 `wait` 和完整 `verify`。已有 `FAILED/EXPIRED` 回执不能
通过修改文件变为成功；历史快照、状态和哈希均应保留。不要停止其他用户服务。

仓库历史物理网络示例为 232 侧 `192.168.2.243 / mlx5_0 / GID 5`，233 侧
`192.168.2.245 / mlx5_0 / GID 3`；管理 IP 示例是 `10.102.0.232/233`。
**执行前必须通过 `ip -br address`、`ibv_devinfo -d mlx5_0`、`rdma link`、
`ip route get ...` 核对并按现场修改**，不能把历史配置当作当前检测结果。
检查 GPU 仅用 `nvidia-smi`；launcher 会拒绝选中的忙卡，不会杀进程。

### 两机公共环境（每个终端都设置）

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
# 232 使用此路径；233 改为 /media/sda1/iic_data/luotao/models/Qwen3-8B
export MODEL=/media/sda1/iic/luotao/models/Qwen3-8B
export GLOBAL_RUN=online-rs-pathfix-20261026-trial01  # 两机相同，每轮换新值
export RS_HOST=192.168.2.245
export RS_PORT=30300
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

# 232 设置：
export NODE_ID=0 GID=5 NODE_RDMA_HOST=192.168.2.243
# 233 改成：export NODE_ID=1 GID=3 NODE_RDMA_HOST=192.168.2.245

export LOCAL_RUN="${GLOBAL_RUN}-node${NODE_ID}"
export RID="${GLOBAL_RUN}-request-node${NODE_ID}"
export RUN="/tmp/kv-online-rs-${GLOBAL_RUN}/node${NODE_ID}"
export GPUS=0,1,2,3  # 只选择自己确认空闲的四张卡
mkdir -p "$RUN"
```

### 233：先启动独立 RS 接收服务（终端 1）

使用空闲 TCP 控制端口及 rendezvous 端口池 30400..30431；验收返回端口
30490 同样必须可达。233 上传 D1 也经过原生 RDMA，涉及同一主机 HCA 路径，
其可用性需要现场验证，不能用 TCP 替代冒充通过。

```bash
python -u -m experiments.kv_ec.remote_rs_server \
  --host "$RS_HOST" --port "$RS_PORT" --timeout 30 --ttl 1800 \
  --max-stripes 8 --max-bytes 536870912 \
  --rdma-device mlx5_0 --rdma-gid-index 3 --rdma-port-num 1 \
  --rdma-port-base 30400 --rdma-timeout 120 \
  --rdma-directory "/tmp/kv-online-rs-${GLOBAL_RUN}/rs-rdma"
```

### 两机：各生成自己的配对配置并启动 CPU Manager（终端 2）

下面生成文件使用 `x`，不会覆盖旧配置。两机不同 request hash 映射到同一个
`pair-A`，chunk 0/1/2 自动派生不同全局 stripe ID。`run_id/checkpoint_version`
是**全局**身份，和 Manager 的 `--run-id`（节点本地身份）分开。

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
e = os.environ
config = dict(
    node_id=int(e["NODE_ID"]), run_id=e["GLOBAL_RUN"],
    checkpoint_version="global-v1",
    requests={hashlib.sha256(e["RID"].encode()).hexdigest(): "pair-A"},
    remote_host=e["RS_HOST"], remote_port=int(e["RS_PORT"]),
    rdma_device="mlx5_0", rdma_gid_index=int(e["GID"]), rdma_port_num=1,
    timeout=240, control_timeout=10, rdma_timeout=90,
    workers=2, max_pending=4, max_bytes=268435456, max_checkpoints=8,
)
with (Path(e["RUN"]) / "global-rs.json").open("x") as f:
    json.dump(config, f, indent=2)
PY
python -u -m experiments.kv_ec.online_chunk_manager \
  --socket "$RUN/manager.sock" --output "$RUN/local" --run-id "$LOCAL_RUN" \
  --timeout 60 --max-pending 8 --retain-ranks --test-recovery \
  --global-rs-config "$RUN/global-rs.json"
```

省略 `--global-rs-config` 就是原有单节点在线 XOR，默认无跨节点行为。

### 两机：分别启动独立 TP4 SGLang（终端 3）

```bash
python -u -m experiments.kv_ec.launch_online \
  --model "$MODEL" --gpus "$GPUS" --tp 4 --port 30381 \
  --socket "$RUN/manager.sock" --run-id "$LOCAL_RUN" \
  --capacity 4 --prefill-chunk-size 256 --mem-fraction-static 0.35
```

launcher 只监听各自 `127.0.0.1:30381`，两机不是同一分布式推理作业。
单机 TP2 开发将 `--tp 4 --gpus 0,1,2,3` 改为 `--tp 2 --gpus 0,1`；两节点
配对时两边 TP 必须相同。仅测试原有单节点 XOR 时省略全局配置。

### 两机：都健康后，各发送请求并等待独立全局回执（终端 4）

```bash
curl --fail http://127.0.0.1:30381/health
python -m experiments.kv_ec.online_chunk_smoke \
  --url http://127.0.0.1:30381 --model "$MODEL" --request-id "$RID" \
  --prompt-tokens 768 --new-tokens 16 --manager-output "$RUN/local" \
  --output "$RUN/request.json" --timeout 240
python -m experiments.kv_ec.verify_online_rs wait \
  --root "$RUN/local" --expected-chunks 3 --timeout 300
```

两个请求内容可以不同；此 smoke 命令用相同公开 token 只是方便比较。
`online_chunk_smoke` 只证明本地完成，必须同时检查 `verify_online_rs wait`。
不要将两个不同 protection ID 的任务按到达先后强行合并。

## 双机验收（在 232 运行，先等两边 wait 成功）

人工通过管理网络将 233 证据复制到 232 的**新目录**；这是测试参考数据搬运，
不是在线传输路径。SSH 用户/地址使用现场已授权配置，例如：

```bash
# REMOTE_SSH 自行设为已授权的 SSH 目标；不在脚本中假定用户名。
scp -r "${REMOTE_SSH}:/tmp/kv-online-rs-${GLOBAL_RUN}/node1/local" \
  "$RUN/node1-evidence"
python -m experiments.kv_ec.verify_online_rs verify \
  --node0-root "$RUN/local" --node1-root "$RUN/node1-evidence" \
  --expected-chunks 3 --remote-host "$RS_HOST" --remote-port "$RS_PORT" \
  --rdma-device mlx5_0 --rdma-gid-index "$GID" --rdma-port-num 1 \
  --rdma-return-host "$NODE_RDMA_HOST" --rdma-return-port 30490 \
  --output "$RUN/acceptance" --timeout 120
```

验收成功条件：

1. 两边恰有 chunk 0/1/2，相同全局身份、同一 remote handle/manifest SHA；
   保留本地 manifest 的独立 `COMMITTED`。
2. 每个源节点 `global_node.bin` 完整解析、rank checksum、local XOR checksum
   和所有 local 单 rank 恢复通过，导出元数据去掉全局 marker 后和本地 manifest
   完全一致。
3. 真实 RDMA 下载远端 P，独立验证 D0/D1 描述符及 P 的 SHA；本地分别重建
   D0/D1/P 并逐字节比较完整序列化块（包括元数据、原始长度和 padding）。
4. 再调用远端 native RDMA recovery API，分别上传 survivor、下载恢复 D0/D1，
   与源块逐字节比较；`acceptance/result.json.state=PASS` 且三个 Chunk 全通过。
   这只模拟单块缺失，不删除原件、不注入整节点故障、不停止推理进程。
5. 分开检查回执中的 queue/RDMA/RS 时间与各自 `request.json.request_ms`。
   需要性能对照时用原有 `launch_online --baseline` 在新的独立实验运行。

当前 server recovery key 每 stripe/方向只适合验收一次；不要在同一 stripe 上
无条件重跑 recovery。验收失败保留目录及回执，换新全局 run 调查和复测。

## 尚需真实环境验证与后续接口

CPU Mock 不能证明真实 GPU fence、Host D2H 重叠、HCA/路由/GID、RDMA completion、
两机同时推理的干扰和请求延迟。当前运行环境没有可用 CUDA，netlink/socket
权限受限；本轮不宣称真实 TP4 双机实验通过。

后续在线控制器对接点是 `OnlineRSExporter.enqueue()` 的局部提交边界；
可用更完整的外部配对计划替换 `requests` 映射，但必须保留全局版本和 range
绑定。网络后端沿用 `RemoteRSClient` 的描述符注册、immutable binding、
`rdma_submit_node_block()`、`get_stripe_status`、`rdma_fetch_parity()` 与
`rdma_reconstruct_missing_node()`。新 worker 没有调用离线 CLI，也没有改变
其参数；`binding(status)` 含 `stripe_id`，使用 positional stripe ID 时只传
`handle/manifest_sha256`，避免重复参数。

若要声明整节点容错，后续必须引入第三个独立物理故障域/重新设计放置策略，
并单独实现持久化、失效检测、恢复调度、GPU pool 安全回写；不能仅把状态名
改成 GLOBAL_COMMITTED。
