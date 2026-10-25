# 双逻辑 TP2 Node + Host XOR + 远端 RS(2,1)

## 状态与边界

这是默认关闭的独立实验工具。未修改 SGLang 核心调度、CUDA、PyTorch 或
Conda 环境。A 的 GPU 0,1 是 Node 0，2,3 是 Node 1；两个独立 TP2 服务，
不是已有 `run_logical_node.sh` 的跨命名空间 TP4 集群。

M1 是现有在线本地 XOR；M2/M3 首先使用真实推理产生的**离线 CPU KV
Snapshot**。远端编码、整节点逻辑擦除恢复不写运行中的 KV Pool，也不证明
恢复后能继续推理。M4 第三组明确标记为 `offline-remote-rs`：在线 Capture
后才执行本地 XOR 和远端 RS，不能解释成在线 Local XOR + Remote RS 干扰。

开发环境记录：基线 Git `cf7a3ee35eb3c02349b59d9003b274e4cdd13de1`。
`git ls-remote origin HEAD` 因 GitHub DNS 不可用失败，无法声称它是远端最新
提交；运行时每份结果都记录实际 HEAD、工作区状态和实验 Python 源码 SHA256。
本环境 GPU 驱动不可用，TCP/Unix bind 被沙箱拒绝。真实 TP2、物理网卡、
真实 KV 端到端和真实 Serving 指标均为 **NOT VERIFIED**。
CPU Fixture 仅用于单测，不是最终验收数据。

## 审计结果（先读此节）

已逐项审阅 `launch_debug.py`、`parity_manager.py`、`debug_capture.py`、
`rank_snapshot.py`、`checkpoint_protocol.py`、`async_protection.py`、
`chunk_protection.py`、`recovery_policy.py`、`TP2_ONLINE.md`、
`M7_RECOVERY_POLICY.md`。关键结论如下：

1. `ParityManager.submit()` 在完整 TP 集合校验后生成 XOR、写入完整提交
   Metadata/rank hashes，随后 `checkpoint.ranks.clear()`。CLI
   `--test-oracle` 只在提交前临时重建验证，并不保留源数据；只有显式 Python
   测试 API `retain_oracle=True` 才保留副本。不能用 L0/L1 恢复完整节点。
2. Capture 在非 overlap、无 prefix cache、单运行请求、完整 prefill 的停止
   调度线程内执行。先 device synchronize，再逐层 Gather/D2H，同步后将复用
   的 pinned staging `.numpy().copy()`，最终 concatenate 独立 CPU payload。
   返回前复查请求 generation、slot mapping。DEBUG_SAFE 还重新读取源验证；
   FAST_PARITY 不做这次重复读取。`AsyncParityClient` 将 payload 转为不可变
   bytes、deepcopy Metadata，队列有界；完成后删除最后一个局部 payload。
3. Capture 已包含 hashed request identity + encoding、token range/token hash、
   层 layout、rank-local row generation、request/checkpoint version、manager
   epoch。普通 Capture 的 slot_generation 可为 null，**不能伪造补齐**；离线
   Pack 原样保存，不将其当作在线恢复授权。`rank_snapshot.assemble()` 的
   COMMON 不包含全部新 version 字段，本新增 Pack 额外逐字段校验它们。
4. `--mem-fraction-static` 已存在，默认 0.60，不重复增加/改变；仅新增
   `--host` 默认 127.0.0.1 并用于 probe 和 launch。HTTP 用 30292/30294，
   Manager socket、输出目录、request/snapshot ID 各自独立。现有 PortArgs
   使用 NamedTemporaryFile 构造独立 IPC，默认 NCCL 端口动态选择；没有发现
   阻止独立实例运行的固定 IPC 冲突，但端口选择与实际 TP2 共存须硬件验证。
   模型/编译缓存共享不等同于本实验状态共享。不要继承其他恢复实验环境变量。
5. TP2 parity CLI 只有提交/Oracle，没有注入故障后写回和继续推理。
   离线 rank_snapshot 支持 CPU XOR、逐字节/segment hash 和 tensor layout
   验证；gpu_scratch_restore 是独立临时 GPU Tensor，不是 KV Pool。
   仓库另有 TP4、256-token 的 M6/M7 lease/暂停投票/post-resume 逻辑故障
   注入、survivor gather、真实 mapped-slot writeback 与继续执行代码
   (`post_resume_recovery.py`)；它不适用于本次双 TP2 远端整节点恢复，且本环境
   没有验证其 GPU 执行。纯 `recovery_policy` 的返回值不是已执行的恢复。

## 数据格式、编码与生命周期

- `D0=Pack(A0,A1)`、`D1=Pack(B0,B1)`，不是 Pack(L0,L1)。本地 L 是两个
  原始 Rank 的 XOR。Node Block 顺序：8-byte magic `KVNODE01`、uint32
  big-endian JSON length、canonical UTF-8 JSON（sorted keys，无空白）、
  rank0 原始 uint8、rank1 原始 uint8。每 Rank 继续遵循升序 layer→K→V→逻辑
  token 顺序，BF16 bytes 原样保存，不做浮点转换。拆包验证所有 offsets、
  shapes、rank/K/V/segment SHA256，保留原始 request/version/generation。
- RS 域 GF(2^8)，不可约多项式 0x11d，primitive element=2，系统生成矩阵
  `[[1,0],[0,1],[1,1]]`。**这个 RS(2,1) 就是普通 XOR 特例，无算法优势**。
  P=D0 XOR D1，短 Node Block 补零至 max length；保留各自原始长度，恢复
  后检查零 padding 再截断。逻辑 chunk/model/layout 必须相同；不同独立
  request ID/checkpoint version 可不同，但各自完整绑定到 immutable manifest。
- 只对 `provenance=sglang_request`、complete、checksum 非 deferred 的真实
  Capture 文件导出，现有 rank_snapshot 校验后再 Pack。证明来源仍需同时保存
  服务日志与 request JSON；Metadata 自我声明不是独立远程证明。禁止将
  Fixture 改标成真实 Capture。生产 CLI 没有 allow-fixture 选项。
- 离线 export 一次一个节点、最多 160 MiB Node Block，无 GPU 引用和在线
  队列。源盘文件属于专用实验快照，不属于正常服务路径。后续若做在线导出，
  应在 D2H 完成且独立 CPU bytes 拥有后接容量/字节预算明确的后台队列，队满
  拒绝保护，完成/超时释放；本次没有安装这个 Hook，也不延长在线 Manager
  的 Rank 生命周期。不要从已释放 GPU slot 再捕获。
- B 使用串行、有超时的 TCP handler（没有无界线程队列），消息上限
  160 MiB，header 上限 2 MiB，store 默认16 stripes、512 MiB 数据预留。
  每 stripe 保守预留3×max node length，另有 JSON、校验/NumPy/收发临时
  副本和 socket/kernel buffer 开销，`max-bytes` **不是整个进程 RSS 上限**。
  两份数据校验完成才 COMMITTED，之后释放 D0/D1，只留 P 和完整 Metadata。
  整节点恢复必须由 A 真实发送唯一幸存 D，B 不会偷偷用缓存原始 D。
- 每次 create 返回随机 handle + manifest SHA；后续所有 stripe API 必须携带
  两者。相同 create/相同 node bytes 重试幂等，冲突拒绝而不污染已有状态。
  所有数据/恢复应答验 SHA；损坏 P、过期、未提交、双丢失均拒绝。
  TTL 从 create 开始、不会因重试续期；过期释放 bytes、保留 ID tombstone。
  达到16个 ID 后使用新 B 进程；重启后旧 handle 不可重用，无持久化承诺。
  一帧使用绝对接收 deadline，半帧和超时不会提交数据。丢 ACK 可按相同
  handle 重试。TCP SHA256 是完整性检查，不是身份认证/TLS；仅用于受控 LAN。

## A. 服务器 A 环境检查

以下命令从仓库根目录运行，先确认 `git status`，如需更新远端请先人工审查，
不要自动 reset/pull 覆盖修改。使用已具备依赖的 Python，不安装/修改环境。

```bash
cd /path/to/sglangec
export PY=/path/to/existing/sglang-env/bin/python
export MODEL=/path/to/Qwen3-8B
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD
git status --short
git ls-remote origin HEAD
"$PY" -c 'import torch,numpy,transformers; print(torch.__version__,torch.cuda.is_available())'
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.used --format=csv
nvidia-smi topo -m
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv
ss -ltnp
ip -br address
ip route
env | sort | grep -E '^(SGLANG_KV_EC_|NCCL_|GLOO_)' || true
```

在干净终端操作；如存在其他实验 `SGLANG_KV_EC_*` 设置，先移除这些设置。
尤其不要继承 scheduler lease/pause/recovery/fault 环境。不要停止其他 GPU 作业；
GPU 不空闲则等待或人工协调。以下使用新的目录，不覆盖已有结果。

```bash
export RUN=$(mktemp -d /tmp/dual-kv-rs.XXXXXX)
export SNAP0="n0-${RUN##*.}" SNAP1="n1-${RUN##*.}"
export RID0="node0-${RUN##*.}" RID1="node1-${RUN##*.}"
export REMOTE_HOST=192.0.2.20  # 必须替换成 B 的真实物理 LAN IP
export REMOTE_PORT=30300
export STRIPE="rs-${RUN##*.}"
mkdir "$RUN/node0" "$RUN/node1"
declare -p PY MODEL PYTHONPATH RUN SNAP0 SNAP1 RID0 RID1 REMOTE_HOST REMOTE_PORT STRIPE > "$RUN/session.env"
ip route get "$REMOTE_HOST" | tee "$RUN/route-A.txt"
```

其他终端 `source /tmp/dual-kv-rs.实际后缀/session.env`，所有终端 cwd 均为
仓库根。`RUN` 同一轮共享，两个 Node 各自子目录/日志/socket 独立。

## B/C. 服务器 B 环境检查与启动

将本轮同一代码版本复制/检出到 B，Python 只需标准库和 NumPy，无 Torch/GPU。

```bash
cd /path/to/sglangec
export PY=/path/to/existing/numpy-env/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD
git status --short
"$PY" -c 'import numpy; print(numpy.__version__)'
uname -a
ip -br address
ip route
ss -ltnp
free -b
export B_RUN=$(mktemp -d /tmp/remote-rs.XXXXXX)
"$PY" -u -m experiments.kv_ec.remote_rs_server --host 0.0.0.0 --port 30300 \
  --timeout 60 --ttl 1800 --max-stripes 16 --max-bytes 536870912 \
  2>&1 | tee "$B_RUN/server.log"
```

保持前台运行。仅在受控网络开放 A→B TCP/30300。另一个 B 终端保存
`ip route get A的真实IP` 和 `ip -s link` 证据；确认不是 lo/veth/容器内部路径。
`create_stripe` 的 TTL 要覆盖整轮验证；超过 TTL 重新使用新 stripe ID。

## D. A 两个独立 TP2 Manager

分别在两个终端执行，不使用 `--test-oracle` 也可提交；此处保留 Oracle
仅帮助检查本地 XOR，**Oracle 成功不代表实际故障恢复**。

```bash
"$PY" -u -m experiments.kv_ec.parity_manager --socket "$RUN/node0/manager.sock" \
  --timeout 120 --max-pending 3 --test-oracle 2>&1 | tee "$RUN/node0/manager.log"
```

```bash
"$PY" -u -m experiments.kv_ec.parity_manager --socket "$RUN/node1/manager.sock" \
  --timeout 120 --max-pending 3 --test-oracle 2>&1 | tee "$RUN/node1/manager.log"
```

## E. A 两个独立 TP2 推理实例（M1）

两个新终端同时运行。`--token-count 256` 只保护真实768-token prompt 的第一
个256-token chunk，不宣称整个请求所有 KV 都受保护。如需全部3个 chunk，
加 `--chunk-size 256`，后续分别验证 `SNAP-chunk-1/2/3`，勿混用 snapshot ID。

```bash
"$PY" -u -m experiments.kv_ec.launch_debug --mode parity --tp 2 --gpus 0,1 \
  --host 127.0.0.1 --port 30292 --mem-fraction-static 0.60 --model "$MODEL" \
  --output "$RUN/node0" --snapshot-id "$SNAP0" --request-id "$RID0" \
  --token-count 256 --layers all --parity-socket "$RUN/node0/manager.sock" \
  2>&1 | tee "$RUN/node0/server.log"
```

```bash
"$PY" -u -m experiments.kv_ec.launch_debug --mode parity --tp 2 --gpus 2,3 \
  --host 127.0.0.1 --port 30294 --mem-fraction-static 0.60 --model "$MODEL" \
  --output "$RUN/node1" --snapshot-id "$SNAP1" --request-id "$RID1" \
  --token-count 256 --layers all --parity-socket "$RUN/node1/manager.sock" \
  2>&1 | tee "$RUN/node1/server.log"
```

## F. 双健康检查、真实测试请求、提交/日志收集

人工构造公开非敏感 prompt，数据来自实际 Qwen3 Forward 的 KV，不是随机
uint8。两个服务各自生成16个 token，固定 temperature=0；需要预热时使用
其他 rid，避免消耗一次性 Capture 计划。

```bash
curl --fail http://127.0.0.1:30292/health
curl --fail http://127.0.0.1:30294/health
"$PY" -m experiments.kv_ec.request_debug request --url http://127.0.0.1:30292 \
  --model "$MODEL" --request-id "$RID0" --prompt-tokens 768 --stream-metrics \
  --output "$RUN/node0/response.json" &
REQ0=$!
"$PY" -m experiments.kv_ec.request_debug request --url http://127.0.0.1:30294 \
  --model "$MODEL" --request-id "$RID1" --prompt-tokens 768 --stream-metrics \
  --output "$RUN/node1/response.json" &
REQ1=$!
wait "$REQ0"
wait "$REQ1"
"$PY" -m experiments.kv_ec.dual_node_check \
  --node0-socket "$RUN/node0/manager.sock" --node0-snapshot "$SNAP0" --node0-request "$RID0" \
  --node1-socket "$RUN/node1/manager.sock" --node1-snapshot "$SNAP1" --node1-request "$RID1" \
  --node0-logs "$RUN/node0/manager.log" "$RUN/node0/server.log" \
  --node1-logs "$RUN/node1/manager.log" "$RUN/node1/server.log" \
  --timeout 120 --output "$RUN/m1-checks"
```

检查两个独立 manager_epoch、request SHA、COMMITTED、rank_hashes 和 parity
大小。36层、TP2、256-token、BF16：每 Rank=18,874,368 bytes，本地 parity
同样大小。Node 原始数据=37,748,736 bytes，加上 canonical header 是 D 长度。
这些理论值必须与真实报告相符才通过 GPU/KV 验收。

## G. 专用真实 KV Capture 与导出（与 M1 分开一轮）

先 Ctrl-C **仅自己启动的两个推理前台终端**，等其 workers 退出，用
`nvidia-smi` 确认。在线 Manager 已释放源 Rank，不能从 M1 timing JSON 导出。
下面在新目录启动 Capture 模式，不连接 Manager，不访问旧 GPU slots。

```bash
mkdir "$RUN/capture0" "$RUN/capture1"
```

两个新终端分别启动：

```bash
"$PY" -u -m experiments.kv_ec.launch_debug --mode capture --tp 2 --gpus 0,1 \
  --host 127.0.0.1 --port 30292 --mem-fraction-static 0.60 --model "$MODEL" \
  --output "$RUN/capture0" --snapshot-id "$SNAP0-cap" --request-id "$RID0" \
  --token-count 256 --layers all 2>&1 | tee "$RUN/capture0/server.log"
```

```bash
"$PY" -u -m experiments.kv_ec.launch_debug --mode capture --tp 2 --gpus 2,3 \
  --host 127.0.0.1 --port 30294 --mem-fraction-static 0.60 --model "$MODEL" \
  --output "$RUN/capture1" --snapshot-id "$SNAP1-cap" --request-id "$RID1" \
  --token-count 256 --layers all 2>&1 | tee "$RUN/capture1/server.log"
```

在请求终端：

```bash
curl --fail http://127.0.0.1:30292/health
curl --fail http://127.0.0.1:30294/health
"$PY" -m experiments.kv_ec.request_debug request --url http://127.0.0.1:30292 \
  --model "$MODEL" --request-id "$RID0" --prompt-tokens 768 --stream-metrics \
  --output "$RUN/capture0/response.json"
"$PY" -m experiments.kv_ec.request_debug request --url http://127.0.0.1:30294 \
  --model "$MODEL" --request-id "$RID1" --prompt-tokens 768 --stream-metrics \
  --output "$RUN/capture1/response.json"
"$PY" -m experiments.kv_ec.rank_snapshot "$RUN/capture0/$SNAP0-cap"
"$PY" -m experiments.kv_ec.rank_snapshot "$RUN/capture1/$SNAP1-cap"
"$PY" -m experiments.kv_ec.node_block_export --snapshot "$RUN/capture0/$SNAP0-cap" \
  --node-id 0 --output "$RUN/export0"
"$PY" -m experiments.kv_ec.node_block_export --snapshot "$RUN/capture1/$SNAP1-cap" \
  --node-id 1 --output "$RUN/export1"
```

再次验证既有 rank_snapshot 需 `--check-only`；不要重复覆盖 manifest/recovery。
每次 export 写 node.bin 与 export.json/export.csv。导出不需要 GPU。

## H/I. 真实远端编码提交与逻辑故障验证

```bash
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op health --output "$RUN/remote-health"
"$PY" -m experiments.kv_ec.dual_node_rs --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --chunk-tokens 256 --network-profile physical \
  --stripe-id "$STRIPE" --node0-block "$RUN/export0/node.bin" \
  --node1-block "$RUN/export1/node.bin" --timeout 120 --output "$RUN/rs-physical"
```

一次完整流程：create→未提交恢复拒绝→D0/D1 TCP 上传→COMMITTED→fetch P→
两个节点各自两个 Rank 擦除（共4项）→整 D0/D1 各自擦除恢复→双 D 拒绝→
缺 P 拒绝。重构的字节必须等于外部原件，同时验证全部 Rank/segment hashes。
所有原件只作比较 Oracle，重构 API 不接收丢失 D。测试不模拟机器断电/进程
退出，也不进行 GPU Pool 写回。缺 P 的 store 拒绝路径还由单元测试验证。

可单独查询、fetch、再次恢复。`--binding` 指向 create/status JSON，或者本流程
的 result.json（使用其中 commit 字段创建一个专用 binding 文件见下方）。

```bash
"$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
r = json.loads((run/'rs-physical/result.json').read_text())
with (run/'binding.json').open('x') as f:
    json.dump(r['commit'], f, indent=2)
PY
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op get_stripe_status --stripe-id "$STRIPE" \
  --binding "$RUN/binding.json" --output "$RUN/status"
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op fetch_parity --stripe-id "$STRIPE" \
  --binding "$RUN/binding.json" --output "$RUN/parity"
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op reconstruct_missing_node --stripe-id "$STRIPE" \
  --binding "$RUN/binding.json" --missing-nodes 0 --payload "$RUN/export1/node.bin" \
  --output "$RUN/recovered0"
cmp "$RUN/recovered0/payload.bin" "$RUN/export0/node.bin"
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op reconstruct_missing_node --stripe-id "$STRIPE" \
  --binding "$RUN/binding.json" --missing-nodes 1 --payload "$RUN/export0/node.bin" \
  --output "$RUN/recovered1"
cmp "$RUN/recovered1/payload.bin" "$RUN/export1/node.bin"
```

双丢失请求应该非零退出，独立输出目录会保存 FAIL JSON/CSV：

```bash
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op reconstruct_missing_node --stripe-id "$STRIPE" \
  --binding "$RUN/binding.json" --missing-nodes 0 1 --payload "$RUN/export0/node.bin" \
  --output "$RUN/expected-two-loss-rejection"
```

低层 create/submit 也有实际 CLI。先写 manifest 再按顺序执行；每条命令的
result.json 都包含完整 response，create 重试原 manifest 不会生成另一个版本。

```bash
export MANUAL_STRIPE="$STRIPE-manual"
"$PY" - "$RUN" "$MANUAL_STRIPE" <<'PY'
import json, sys
from pathlib import Path
from experiments.kv_ec.rs_stripe_builder import build
run = Path(sys.argv[1])
m = build(sys.argv[2], [(run/f'export{i}/node.bin').read_bytes() for i in (0,1)])
with (run/'manifest.json').open('x') as f:
    json.dump(m, f, indent=2)
PY
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op create_stripe --stripe-id "$MANUAL_STRIPE" \
  --manifest "$RUN/manifest.json" --output "$RUN/manual-create"
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op submit_node_block --stripe-id "$MANUAL_STRIPE" \
  --binding "$RUN/manual-create/result.json" --node-id 0 --payload "$RUN/export0/node.bin" \
  --output "$RUN/manual-submit0"
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op submit_node_block --stripe-id "$MANUAL_STRIPE" \
  --binding "$RUN/manual-create/result.json" --node-id 1 --payload "$RUN/export1/node.bin" \
  --output "$RUN/manual-submit1"
```

## J. tc 带宽与 RTT 模拟

先记录实际网卡/IP、链路速率、现有 qdisc，**不要替换现有业务 qdisc**。仅在
获准使用的实验接口且没有自定义 root qdisc 时添加；已有 root 会拒绝 add。
以下只对发往 B 的30300端口数据限速100 Mbit/s、增加20ms单向时延，其余
流量走高带宽 class；不能声称这不影响共享物理接口。先人工协调。

```bash
export IFACE=enp1s0  # 替换成 ip route get 所显示的真实物理网卡
ip route get "$REMOTE_HOST"
sudo ethtool "$IFACE"
tc -s qdisc show dev "$IFACE"
sudo tc qdisc add dev "$IFACE" root handle 1: htb default 20
sudo tc class add dev "$IFACE" parent 1: classid 1:10 htb rate 100mbit ceil 100mbit
sudo tc class add dev "$IFACE" parent 1: classid 1:20 htb rate 10000mbit ceil 10000mbit
sudo tc qdisc add dev "$IFACE" parent 1:10 handle 10: netem delay 20ms
sudo tc filter add dev "$IFACE" protocol ip parent 1: prio 1 u32 \
  match ip dst "$REMOTE_HOST/32" match ip dport 30300 0xffff flowid 1:10
tc -s -j qdisc show dev "$IFACE" > "$RUN/tc-A.json"
ip -s -j link show dev "$IFACE" > "$RUN/link-A-before.json"
```

20ms 不是完整40ms RTT 的声明：若需对称40ms，在 B 的实验接口也为 B→A
返回流量配置20ms（按 `sport 30300` filter、A 的目标IP），保留B的证据。
qdisc rate 是配置的**模拟带宽**，ethtool 是实际链路标称速率，TCP goodput
是本实验测得的应用有效带宽；三者不能混用。MTU/网卡 offload/系统负载也应保存。

## K. 网络性能与推理影响

用相同真实 Capture bytes、新 stripe ID 和新 output 重复物理/tc 测试：

```bash
"$PY" -m experiments.kv_ec.dual_node_rs --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --chunk-tokens 256 --network-profile tc \
  --stripe-id "$STRIPE-tc" --node0-block "$RUN/export0/node.bin" \
  --node1-block "$RUN/export1/node.bin" --timeout 120 --output "$RUN/rs-tc"
ip -s -j link show dev "$IFACE" > "$RUN/link-A-after.json"
"$PY" -m experiments.kv_ec.remote_rs_client --remote-host "$REMOTE_HOST" \
  --remote-port "$REMOTE_PORT" --op get_metrics --output "$RUN/remote-metrics"
```

报告包含每 Rank bytes、local XOR、Pack、RS encode、connect/send/receive/
roundtrip、远端 first-create-to-commit、A observed commit、重构/解析验证耗时、
SHA/byte-exact、A/B CPU时间差/一核百分比、当前/峰值RSS。传输计数是完整有效
帧的应用 TCP stream bytes（payload +4-byte length+JSON），不包含 Ethernet/IP/
TCP headers、ACK、重传；B 逐次 recv/send 计数，也覆盖损坏/超时的半帧，errors 会增加。
报告自动读取 A/B 路由所选接口的 sysfs tx/rx bytes、packets 和链路标称速率，
并计算 NIC counter delta；仅当双方 route interface 存在物理 device 且非
loopback 时标记 `PHYSICAL_INTERFACE_ROUTES_OBSERVED`。这仍不是抓包证明。
实际物理 wire bytes 需另取专用网卡计数或抓包并隔离其他流量。不可把协议
计数称作以太网实际总字节数。RX includes wait time，sendall 耗时可能只是
内核缓存接收耗时；`effective_payload_mbps` 包含网络往返/服务端校验和编码，
不是纯链路带宽。profile 参数只是标签，不会偷偷用 sleep 模拟网络。
模拟带宽默认 null，标称速率在 sysfs 可读时记录（否则 null），须与保存的
tc/ethtool 证据一起分析。

M4 需三轮**重新启动**模型，每轮新 Capture 目录/snapshot；同样的模型、
人工输入、temperature=0、16-token输出和 GPU 映射。各轮先用其他 RID 做
同数量预热，再并发运行目标请求。建议至少5轮，保留全部 JSON/CSV；不在
没有样本时生成统计结论。`--pids` 只传本轮自己启动的两个服务/Manager PID，
程序会只读采样这些 PID 及子进程，不终止任何进程。

第一轮：按 E 启动但将 `--mode parity` 改为 `--mode baseline` 并删去
`--parity-socket`，使用新 output/snapshot，启动后记录两个 PID：

```bash
export SERVER0_PID=12345 SERVER1_PID=12346  # 替换成自己本轮进程的实际 PID
"$PY" -m experiments.kv_ec.serving_impact run --mode no-ec --model "$MODEL" \
  --node0-request "$RID0" --node1-request "$RID1" \
  --pids "$SERVER0_PID" "$SERVER1_PID" --output "$RUN/m4-noec"
```

第二轮：全新 M1 parity 实例，snapshot 是 E 中的变量（若 F 已消耗该请求，
必须使用新 snapshot/restart，不能重复旧 capture plan）。记录新服务/Manager PID：

```bash
export MANAGER0_PID=12347 MANAGER1_PID=12348  # 同样替换为本轮真实 PID
"$PY" -m experiments.kv_ec.serving_impact run --mode local-xor --model "$MODEL" \
  --node0-request "$RID0" --node1-request "$RID1" \
  --node0-socket "$RUN/node0/manager.sock" --node0-snapshot "$SNAP0" \
  --node1-socket "$RUN/node1/manager.sock" --node1-snapshot "$SNAP1" \
  --pids "$SERVER0_PID" "$SERVER1_PID" "$MANAGER0_PID" "$MANAGER1_PID" \
  --output "$RUN/m4-local"
```

第三轮：全新 G Capture 实例，**不要先运行 G 的请求命令**，由驱动生成请求
并在完成后导出/远端编码。若 G 已执行过，换新 RUN/快照并重新启动。

```bash
"$PY" -m experiments.kv_ec.serving_impact run --mode offline-remote-rs --model "$MODEL" \
  --node0-request "$RID0" --node1-request "$RID1" \
  --node0-capture "$RUN/capture0/$SNAP0-cap" --node1-capture "$RUN/capture1/$SNAP1-cap" \
  --remote-host "$REMOTE_HOST" --remote-port "$REMOTE_PORT" --chunk-tokens 256 \
  --network-profile physical --stripe-id "$STRIPE-m4" \
  --pids "$SERVER0_PID" "$SERVER1_PID" --output "$RUN/m4-offline"
"$PY" -m experiments.kv_ec.serving_impact compare \
  "$RUN/m4-noec" "$RUN/m4-local" "$RUN/m4-offline" --output "$RUN/m4-comparison"
```

记录 SSE 客户端 TTFT、amortized TPOT、输出 token/s、并发总吞吐、采样 CPU/
GPU利用率和RSS；CPU可能超过100%（多核）。RSS sum 含共享页重复计数，
`nvidia-smi` 采样窗口对短请求可能太粗，不能用缺失数据填零。Manager 的
`parity_commit_ms` 是首 Rank 到 commit，**不是 Capture 起点到 commit 的完整
未保护窗口**，后者当前 null/NOT VERIFIED。离线组保护提交单独报告；真实
online Local XOR + Remote RS 的 TTFT/TPOT 与未保护窗口仍 NOT VERIFIED。
compare 逐 Node 检查 input hash/model/sampling/output token sequence/finish
reason 一致；它不会把第三组升级成在线保护实验。

## L. 测试、验收和停止

```bash
"$PY" -m unittest experiments.kv_ec.test_dual_node_launch experiments.kv_ec.test_parity_manager -v
"$PY" -m unittest experiments.kv_ec.test_remote_rs -v
"$PY" -m unittest experiments.kv_ec.test_rs_experiments -v
"$PY" -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
"$PY" -m compileall -q experiments/kv_ec
"$PY" -m experiments.kv_ec.validate_remote_rs --full-regression --output "$RUN/cpu-validation"
```

CPU单测含 Fixture 只用于可复现算法/协议验证；TCPIntegrationTests 用真正
AF_INET loopback，不是 Fake Socket，但 bind 被禁止时显式 skip。
真实 LAN 验收必须另在两台服务器运行 H/I，同时保存 A/B 路由、NIC计数、
Remote Server 日志、两个真实TP2 Capture/响应/Manager日志，确认全部
byte-exact。不要用 loopback-test 或 fixture 结果替代。

停止：各自前台 Ctrl-C，仅停止本实验启动的服务和 Manager；B 同理。
不使用 pkill/killall/fuser，不清理其他 GPU 进程，不自动删除 sockets/快照。
实验结果留在 RUN/B_RUN，不删除。tc 清理只针对自己确实创建的 root handle：

```bash
tc -s qdisc show dev "$IFACE"
# 人工确认 handle 1: 是上面自己创建的实验 HTB 后才执行：
sudo tc qdisc del dev "$IFACE" root handle 1:
nvidia-smi
ss -ltnp
```

异常退出残留 socket 不自动覆盖；换新 RUN 即可。手动删除前核实该 socket
确属已停止的本实验 Manager。结果文件 open(x)/目录 exclusive create，所有
新增实验主要输出均有独立 JSON 和 CSV；不要将真实 KV binary 提交到 Git。
