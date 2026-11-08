# EC 实验设计：隔离实现与证据报告

本目录是 `/media/sda1/iic_data/luotao/ec-design-20261103.txt` 的有限实现，
默认关闭，没有修改 SGLang 核心或接入自动启动。新增代码只在本 worktree；
此前已审阅但未提交的 restore 修改保持原样。未启动 GPU 服务、未运行 GPU
故障切换、未 SSH 部署、未重启任何 RS 服务、未 commit/push。

已阅读仓库根 README、KV EC README、multinode README、三域相关旧
`DUAL_NODE_REMOTE_RS.md`、M7、GPU admission/TP selectable 文档及现有
export、codec、RS server/client、cold SSE/Stage E 路径。本目录没有适用的
AGENTS.md（仓库中发现的是 `docs/AGENTS.md`，其作用域不包含本目录）。

## 实现、未测、阻塞

| 项目 | 实现与证据 | 尚未证明 |
|---|---|---|
| 三故障域 | `placement.json` + `admission.py` 固定 D0=232、D1=233、P=物理243；检查域、地址、专用端口和预算 | 三机实际连通性、故障域硬件身份、节点失效后的 serving |
| 远程 parity-only | `remote.py` 复用现有 `StripeStore`；COMMITTED 后清除输入，只保留 parity/manifest；现有 handle/hash/TTL 约束保留 | 243 上实际部署和 TCP 实连，本次没有远程写入 |
| CPU/远程副本 | `storage.py` 完整 NodeBlock 副本、独立持有 bytes、SHA/descriptor/epoch/handle 绑定、预算与过期 tombstone | 在线 GPU 写回、跨机实测；CPU 副本不是节点故障备份 |
| 固定 trace/SSE | `workload.py` 预生成 ID、保存哈希、检查实际 prompt/output 长度和累计输出；可接受监督逻辑擦除 callback | 各 baseline 自动激活、并发 batch runner、服务端 prefill/decode 独立计时 |
| Fault-to-SSE | 同客户端 monotonic clock；故障前或指定已观察 completion 位置调用 callback；只取边界后新增 SSE 事件 | callback 自证不是独立故障证明；缓冲/保存的 pending token 不等于新的 decode；没有检测实际 GPU/进程失效 |
| E1–E6 | `matrix.json` 逐项声明实现/未运行/阻塞；统一 fallback admission API | TP4/64K/4K 输出没有完整实现或验收 |
| E4 codec | 历史真实 TP1 capture 切出四段不重叠 128 KiB；CPU XOR、两种 RS，各重复5并 bytecheck；raw JSON+manifest | TP2/TP4 分片实测、1/4/16/64 MiB、网络、GPU pipeline |
| 通信微基准 | 可手动运行受限 TCP echo，校验往返 bytes、记录 framing/payload 和独立端到端 pipeline 时间 | 当前 sandbox 禁止 bind；没有 TCP/RDMA 性能数字 |
| 策略 | `policy.py` 精确上下文、每路线至少5独立成功测量的中位数选择；禁止外推和静默丢弃失败 | 没有两条 serving 路径的真实校准输入，所以没有生产校准结果，不使用 codec 时间冒充 prefill/recovery |
| 内存成本 | `memory_ledger` 分别统计 stable parity/replica、snapshot、temporary；不相加两级冗余比例 | ledger 不是实测 RSS/显存峰值；需要监督运行时 allocation/RSS 采样 |

用户给出的历史状态单独保留为背景，不作为本次新证据：attempt15 有30次
onlineSmoke PASS、4096/1024/256 recovery pairSmoke PASS；512由用户明确中断，
formal5未开始。旧结果没有修改，本次没有补写或延续它们的编号。

## 放置与安全准入

| 角色 | 物理机器 / TCP 地址 | 项目根 | 持有数据 |
|---|---|---|---|
| D0 | 232 / 10.102.0.232 | /media/sda1/iic/luotao | 原始 node0 KV、该节点 Host DRAM 的 local XOR |
| D1 | 233 / 10.102.0.233 | /media/sda1/iic_data/luotao | 原始 node1 KV、该节点 Host DRAM 的 local XOR |
| P | 243 / 10.102.0.243（用户 iic） | /media/home/iic/luotao | parity 或独立 remote replication baseline 的完整副本 |

`192.168.2.243` 属于物理232，`192.168.2.245` 属于物理233，两者均不能
作为第三故障域。用户确认物理243的 `/sys/class/infiniband` 为空、无 RDMA
网络接口；本次没有远程探测，本实现明确使用 TCP。
配置要求 RDMA 时 fail closed；TCP-only store 也在启动 helper 前拒绝 RDMA
请求。`require_host_rdma` 仅检查本机设备存在，不能替代真实传输测试。
现有 verbs 原型仅能称为 **HostRDMA**，没有 GPUDirect RDMA 证据。

默认使用新端口32951(parity)、32952(replica)、32953(echo)。准入拒绝
30701/31700/31702/31704，不探测或改变这四个健康 RS 服务。新端口被占用时
bind 失败，不杀进程、不重启服务。协议沿用实验 LAN 的无认证边界，只用于
隔离可信网络。配置不含密钥。机器身份准入同时要求显式 physical-id 与本机
hostname 解析地址匹配；多网卡环境若 hostname 不解析该地址，应先由操作者
核实机器与 DNS，工具保守拒绝，不自动修改网络。

下列命令仅供 parent 审阅后监督执行；本次没有执行 serve/run 的服务命令。
243 的 Python 路径需使用其已存在的环境，不能照搬232/233的 Conda 路径。
所有命令从独立部署 checkout 中运行，工作目录及日志必须属于本实验，勿写旧结果。

```bash
export PYTHONPATH=python:.
export CUDA_VISIBLE_DEVICES=
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
CFG=experiments/kv_ec/design_study/placement.json
$PY -m experiments.kv_ec.design_study.remote --config "$CFG" admit

# 仅在物理243、专用工作目录，由 parent 监督启动，使用243实际 Python：
python -m experiments.kv_ec.design_study.remote --config "$CFG" serve --physical-id 243 --kind parity
# replica 是另一条 baseline，按需要单独监督启动：
python -m experiments.kv_ec.design_study.remote --config "$CFG" serve --physical-id 243 --kind replica
```

### 有界三主机 TCP 验证（parent 监督）

`three_host.py` 只适合 parent 已经拿到两份真实 D0/D1 NodeBlock 的情况；它不
启动 P 服务。它会先绑定两个完整 descriptor，再提交两块数据；COMMITTED 后
要求 P 报告 `retained_node_block_bytes=0`，随后 parity 读取 5 次、D0/D1
单擦除恢复各 5 次。恢复请求只发送 surviving NodeBlock，缺失块的原始文件
只在调用方本地用于最终 bytecheck。P 的返回是 TCP application framing，不能
当作磁盘、RDMA、GPU、物理节点失效或 RTO 证据。

如果 parent 希望两台数据机各自只接触本机文件，应在 D0、D1 分别运行下面的
`source_trial.py`（同一新 stripe ID；两个进程可并行）。两边都需要 parent 先
确认 P 的服务已由监督启动，且输出目录是 artifacts 下的新目录：

```bash
export PYTHONPATH=python:.
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
CFG=experiments/kv_ec/design_study/placement.json
ID=parent-$(date +%s)-fresh
# D0 物理232：只读本机 node0.bin
$PY -m experiments.kv_ec.design_study.source_trial --config "$CFG" \
  --role D0 --physical-id 232 --node-block /path/to/node0.bin --stripe-id "$ID" \
  --output experiments/kv_ec/design_study/artifacts/${ID}-D0
# D1 物理233：只读本机 node1.bin
$PY -m experiments.kv_ec.design_study.source_trial --config "$CFG" \
  --role D1 --physical-id 233 --node-block /path/to/node1.bin --stripe-id "$ID" \
  --output experiments/kv_ec/design_study/artifacts/${ID}-D1
```

若 parent 在一个有两份输入文件的监督环境运行集中式协议检查，则使用：

```bash
$PY -m experiments.kv_ec.design_study.three_host --config "$CFG" \
  --nodes /path/to/node0.bin /path/to/node1.bin --stripe-id "$ID" \
  --repetitions 5 --output experiments/kv_ec/design_study/artifacts/${ID}-parent
```

任一命令失败都会保留 `raw.json`（若已创建输出目录）；不要复用 stripe ID 或
覆盖 artifacts。transport 必须显式为 TCP，RDMA 请求 fail-closed。

在232、233各自导出完整真实 TP2/TP4 NodeBlock（复用
`experiments.kv_ec.node_block_export`），各自执行 `remote register`，只上传
descriptor。两个注册完成后，通过 `remote status` 保存 PENDING/COMMITTED
status；在每台源机器使用该 status 执行 `remote submit`。每个输出都必须是
新文件，不覆盖。注册顺序无关，首次返回 REGISTERING 不能拿去 submit。

```bash
# 示例：在232，node0.bin 为本机真实 export
$PY -m experiments.kv_ec.design_study.remote --config "$CFG" register \
  --role D0 --physical-id 232 --node-block node0.bin --stripe-id study-unique-001 --output register0.json
# 在233同样使用 --role D1 --physical-id 233 --node-block node1.bin
$PY -m experiments.kv_ec.design_study.remote --config "$CFG" status \
  --stripe-id study-unique-001 --output ready-status.json
$PY -m experiments.kv_ec.design_study.remote --config "$CFG" submit \
  --role D0 --physical-id 232 --node-block node0.bin --stripe-id study-unique-001 \
  --status ready-status.json --output submitted0.json
# 233也 submit 后，重新获取 committed-status.json，必须 state=COMMITTED
$PY -m experiments.kv_ec.design_study.remote --config "$CFG" recover \
  --status committed-status.json --missing-node 0 --survivor surviving-node1.bin --output recovered-node0.bin
```

`recover` 在恢复目标执行，只需 surviving node1 与预故障 status，不访问丢失
node0 的原始 bytes。它验证完整 descriptor（请求/版本/epoch/内容），不是只比较
长度。CLI 不注入任何物理故障，也不写 GPU pool；离线恢复成功不等于 node RTO。
TTL默认900秒，结束后 ID tombstone 保留；达到对象数预算则拒绝新 ID。
不要为了扩容重启用户现有服务。

副本入口 `put-replica --role D0 --physical-id 232 --node-block node0.bin
--object-id request-checkpoint-unique --output replica-binding.json` 在源机执行；
`get-replica --binding replica-binding.json --descriptor prefault-descriptor.json
--output replica-restored.bin` 在目标机执行，后者只需要预故障 descriptor。
`prefault-descriptor.json` 是 `node_block_export.descriptor()` 的完整 JSON。
这些子命令同样需要 `remote --config "$CFG"` 前缀。CPU baseline 可直接使用
本机 `ReplicaStore` + `LocalReplicaClient` + `put_replica/get_replica`；
它的对象持有完整副本，不会谎称只有 parity。两者尚未连接在线 GPU restore。

## 统一资源、trace、故障与策略

`admission.uniform_resources(candidates, evidence)` 按候选顺序选取唯一配置，
要求 base/cpu_replication/remote_replication/rs_only/two_level 都有相同 TP、
输入/输出长度、batch、dtype 和 trace hash 的通过证据，且实测 free_bytes
满足 required_bytes。每个 evidence 需包含 preflight artifact SHA256；调用方
负责验证该文件与服务启动参数。缺少任一 baseline 就拒绝，不把某一方法降成
TP2而保留另一方法 TP4。该 API 是 resource admission，不宣称 runtime 支持。
可复用现有 `gpu_admission.py`/`perf_preflight.py` 采集实际 GPU 占用；本次未运行。

当前候选 fallback 是 TP2/4096输入/17输出/batch1，不代表已通过新 baseline
统一验收。现有 Stage E 的 prefix 上限、模板及17输出语义保持原样；没有修改
其配置校验来假装支持64K或4K输出。长上下文应先准备真实模板、明确 KV 预算、
验证每个方法同配置可运行，失败时整组降级或明确标记不支持。

```bash
$PY -m experiments.kv_ec.design_study.workload trace --seed 11 --requests 10 \
  --input-tokens 4096 --output-tokens 17 --vocab-size 1000 \
  --model-sha256 ACTUAL_64_HEX_MODEL_TOKENIZER_FINGERPRINT --output fixed-trace.json
# parent 已监督启动并确认 baseline 后才运行，工具自己不启动服务：
$PY -m experiments.kv_ec.design_study.workload run --trace fixed-trace.json \
  --url http://127.0.0.1:SUPERVISED_PORT --baseline base --output new-trace-result
```

Trace 输入长度精确，ID 从给定词表范围采样，不自动过滤特殊 ID；输出
`ignore_eos=True`，仍以响应实际 token 数验收。提供的 model/tokenizer fingerprint
必须由操作者从真实模型资料计算。baseline 参数只是标签，输出明确保存
`baseline_activation_verified=false`，不能靠标签声称启用了保护。

可编程 `measure(..., fault=callback, fault_after_tokens=N, expected_output=reference)`
用于与监督的现有 hook 集成。callback 收到请求和已观察 completion 位置，返回
`kind=kv_block_erasure`、同 request_id、checkpoint_binding、同
observed_completion_tokens、verified=True。没有 callback 时 fault-to-SSE 为 null。
SSE批量输出可能跳过精确 decode 位置；工具记录实际观察值，若直到完成才满足
故障位置则拒绝。callback 必须调用并验证真实逻辑擦除 hook，而不能只写 receipt；
本次只用 mock 验证测量逻辑，没有安装 GPU 注入 adapter。

节点/进程故障、TP 通信组重建、检测与目标资源准备不在 callback 的支持范围。
恢复时间（KV ready）与 fault-to-first-SSE/RTO 分开。每个 failed/timeout 请求
保留在 raw results 中；`policy.summarize` 同时报完成率、恢复成功率、超时率，
MTTR 字段明确只针对成功恢复，无全部恢复时间时 unconditional MTTR 为 null。
没有独立有效推理时间测量，EITR 为 null，不能用 wall time 编造。

`request_fault_schedule` 固定 seed 的5%/10%/15%表示独立 Bernoulli **请求级**
故障概率，实际选中数未必等于比例；不表示每秒设备故障率。E6按一个变量
变化，不自动做全笛卡尔积。

`policy.calibrate` 只接受同 model/trace/TP/长度/batch/fault/transport/hardware/
measurement_profile 的 EC 与 prefill 两组独立 trial，边界必须均为
fault_to_kv_ready，至少各5次，有 trial ID 与 artifact hash，失败不允许隐藏。
它是测量中位数规则，没有拟合模型。codec 微基准、client SSE、KV ready
三类时间不可混用。`choose` 还需要 committed/binding/replay_safe 的显式布尔
安全票；没有安全路径返回 fail_closed。它不替代已有 `recovery_policy.py` 的
row/slot generation、manager epoch、TP vote 与 request lifecycle 校验，
更不会自动在 post-decode 生命周期执行不安全的重新 prefill。

## 微基准和本次原始证据

`microbench.run` 默认要求两台机器独立导出的真实 NodeBlock，并用现有
manifest 校验跨节点 model/dtype/layout。CPU local XOR 使用同一节点真实 rank
payload。每点固定重复5，逐一恢复每个 rank 和 D0/D1，校验原始 bytes。
`multinode/rs_codec.py` 的 GF256(1,2) 与远程协议 `rs_codec.py` 的 XOR 特例
分开命名。吞吐量分子保存 useful_bytes：encode 为输入数据总量，decode 为
丢失块长度；不是 DRAM 总读写流量。

默认1/4/16/64 MiB；数据不足记录 unavailable，不扩增、不平铺、不补假数据。
可选 `--echo-host` 只对已监督启动的 echo service 测试；往返包括 connect、
SHA、upload、download，不叫单向带宽。Host pipeline 单独测总墙钟，同时保留
transfer/decode 子阶段，不把子阶段相加当成异步端到端恢复。没有测到的
Gather/D2H/H2D/writeback、HostRDMA、GPUDirect 均明确 unavailable。

本次唯一可用历史真实 KV 文件为
`../results/real/tp1-capture-20261016T133950/rank_0.bin`，512 KiB。校验其 sidecar、
总体/逐层 K/V 哈希后，切成四段128 KiB互不重叠区间，分别输入 codec。
这证明处理真实 capture bytes 的 CPU codec 正确性，不证明真实 TP2/TP4
分片或三节点执行；provenance 来自历史 metadata，并非本次重新 attest GPU capture。

```bash
$PY -m experiments.kv_ec.design_study.microbench run \
  --rank-capture experiments/kv_ec/results/real/tp1-capture-20261016T133950/rank_0.bin \
  --sizes-mib 0.125 1 4 16 64 --output experiments/kv_ec/design_study/artifacts/NEW_UNIQUE_RUN
# 未来真实大块模式：--nodes exported-node0.bin exported-node1.bin
# TCP echo仅由parent在专用端口监督启动，不使用RS健康端口：
# $PY -m experiments.kv_ec.design_study.microbench echo-server --host VERIFIED_ADDRESS --port 32953
```

`artifacts/historical-real-kv-codec-r2/` 保存最终代码的45个 timed codec records
（每轮9项×5），所有恢复 byte-exact；四个大尺寸点 unavailable。
manifest 固定输入 SHA256、metadata、执行时 codec 源码哈希与 raw JSON 哈希。
这些数值是本机 CPU 重测值，不用于宣称 serving 加速、网络带宽或校准恢复策略。
首次执行 `r1` 后补充 manifest 环境字段，r1保持不变；最终 `r2` 记录
NumPy、主机、CPU affinity 与 warmup=0。无 warmup/CPU隔离，不能用这份
小规模结果作稳定性能结论。

## 验证与后续监督顺序

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=python:. \
  /media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python -m unittest \
  experiments.kv_ec.design_study.test_design_study -v
```

测试覆盖：三域/错误别名/健康端口/RDMA fail closed，统一 fallback，副本
immutable binding/容量/过期/重启/损坏，现有 parity-only 的两方向 erasure
恢复，固定 trace 篡改、SSE 输出历史/截断/reference/故障位置、保留失败分母，
策略安全门与不混边界，真实 codec 路径重复5及 manifest/fixture gate。
Loopback bind 被 sandbox 返回 EPERM，socket 实连测试应 skip，不能改记 PASS。
首次测试发现测试 fixture 缺少版本字段，已补齐；失败日志保留在 artifacts，
后续通过日志另存。最终聚焦回归50项：47通过、3因TCP bind权限跳过，
见 `artifacts/focused-tests-final-r2.log`，其中新模块19项：18通过、1跳过。

下一步由 parent 监督：审阅三域配置并验证机器身份/专用端口 → 在243部署本
checkout的 TCP-only parity/replica服务 → 三域真实 capture/register/commit/
离线 bytecheck → 完整 CPU/remote replica GPU adapter 与统一 resource admission →
固定 trace 运行所有 baseline → 分别校准 fault-to-KV-ready → 监督 GPU逻辑
擦除与SSE → 最后单独审批和设计物理节点/进程失效恢复。未完成这些步骤前，
E1/E2/E5/E6 保持 NOT RUN，物理节点 RTO 与 GPU failover 不得声称已完成。
