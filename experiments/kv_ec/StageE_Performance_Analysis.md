# Stage E 五轮真实性能分析（r5）

> 本报告只读分析 `/tmp/kv-perf-232.u9wlhC8o/baseline-r5` 与 `summary-r5.json`，未运行 GPU/RDMA 实验，未修改 XOR、RS、Scheduler 或 GPU KV Cache 恢复算法，也未停止任何进程。

## 1. 数据范围与结论

- 统计对象：5 个 accepted trials，0 rejected，0 missing metrics，130 组序列；方案 `two-level`，Qwen3-8B，TP=2，Prefix=256 token，受控单块/单节点损失后恢复并完成 16 次 Decode。
- 五轮 trial 命令耗时：279.279、276.984、275.826、277.536、279.858 s；均值 277.897 s，标准差约 1.659 s，P50 277.536 s，范围 4.032 s（1.45%）。
- 正确性：5/5 `PASS`，5/5 `decode_correctness=true`；Stage E 检查同时要求新实例、真实 RDMA、GPU restore committed、16 次恢复 Decode、输出 token 一致和无资源泄漏。
- 性能结论：当前最大开销是 Harness 等待/启动以及恢复前后的完整流程；GPU KV 写回、XOR/RS 计算和 verbs CQ 传输均为毫秒到秒级，不能从现有数据推断算法本身占据 117 秒端到端时间。

## 2. 计时实现、事件边界与包含关系

计时单位规则：`perf_observer.emit()` 将 `monotonic_ns` 差值除以 1e6 写成 `duration_ms`；CUDA Event 使用 `elapsed_time()`，本身为毫秒；Native verbs 的 `elapsed_s` 为秒，`perf_stage_e.native_metrics()` 只在汇总前乘 1000。所有 `inclusive` 指标是独立区间，不能互相求和。

| 指标/指标族 | 代码位置与开始→结束 | 事件/单位 | 包含关系与归类 |
|---|---|---|---|
| `capture_artifact_inclusive` | `gpu_cold_resume_hook.py:361` 装饰的 capture 过程返回前后 | 进程 `monotonic_ns`，ms | GPU/请求 Capture 外围；包含该过程内部工作，不等于纯 GPU 拷贝 |
| `checkpoint_load_validate_inclusive` | `cold_resume_state.py:104` 的 `load_checkpoint()` 入口→校验完成 | `monotonic_ns`，ms | 请求状态/Checkpoint 读取与哈希校验；RS 恢复时可嵌套 RS source 校验 |
| `protection_inclusive` | `verify_rs_cold_resume.py:55` 的 `protect()` 入口→GLOBAL_COMMITTED 返回 | `monotonic_ns`，ms | XOR/RS 保护、上传、提交的外围；不能与上传/编码子项相加 |
| `rs_upload_inclusive`, `node*/rdma_upload_wall_ms` | exporter `online_rs_exporter.py:259-268` 上传开始→完成；外围由 observer 记录 | 进程 `monotonic`，ms | RDMA 上传外围；内部含 peer wait/verbs CQ，和 native elapsed 不可相加 |
| `rs_recovery_rpc_inclusive` | `remote_rs_client.py:140` 装饰的 `rdma_reconstruct_missing_node()` 入口→恢复块校验返回 | 232 客户端 `monotonic`，ms | RS 恢复 RPC 外围，含 TCP 控制、server 状态等待、survivor native send、target native receive、checksum；不是纯 RS decode |
| `rs_source_load_validate_inclusive` | `rs_cold_source.py:121` 的 `load_cold_source()` 入口→source/receipt 校验完成 | 各主机进程 `monotonic`，ms | 恢复源读取、绑定和哈希校验；receiver/target/source 角色分别统计 |
| `delivery_send_inclusive` / `delivery_receive_inclusive` | `verify_rs_cold_resume.py:173`/`receive_source()` 装饰器入口→ZIP 解包与 `load_cold_source` 完成 | 各自主机 `monotonic`，ms | RDMA 交付 + archive 校验；包含关系不能与 native send/receive 相加 |
| `gpu_kv_writeback` | `gpu_kv_restore.py:465-486` `started`→CUDA event 完成 | 进程计时，ms；包含 H2D、host tensor prepare、scatter、fence | GPU 数据操作；不是纯 PCIe/H2D |
| `gpu_layer_readback_validation` | `gpu_kv_restore.py:488-502` verify start→逐层读回/哈希/fence 完成 | 进程计时，ms | 正确性验证，非必要恢复数据搬运 |
| `scheduler_request_rebuild` | target Scheduler 重建请求状态的 observer | 进程 `monotonic_ns`，ms | Scheduler 状态恢复；与 GPU 写回并行/包含关系需按事件看，不能直接相加 |
| `first_restored_decode_cuda` | `gpu_cold_resume_hook.py:318` 首个恢复 forward 前 CUDA Event record→`observe_model_forward` 中结束 Event | CUDA Event，ms | 首个恢复后 Decode forward 的 GPU stream 区间，含首次执行/同步开销；不等于 16 token 总时延 |
| `decode_forward_validation` | cold resume hook 每次 forward 的状态校验 observer | 进程 `monotonic_ns`，ms | 正确性验证；与 first decode 或端到端区间不可相加 |
| `model_weight_load` | ModelRunner 既有 `weight_load_time`（observer 在启动加载完成处采集） | 已有 perf counter，ms | SGLang 模型权重加载；属于 `new_instance_launch_to_ready_harness` 内部 |
| `global_committed`, `global_commit_wait_ms`, `server_commit_ms`, `server_rs_encode_ms` | exporter/server 的 server monotonic 标记与阶段计时 | server monotonic，ms | RS server 控制/编码；`global_committed` 是状态标记，0 ms 不代表无保护工作 |
| Native `*/device_setup_ms`, `memory_registration_ms`, `qp_setup_ms`, `exchange_ms`, `completion_ms`, `elapsed_s` | `multinode/rdma_transport.py` 返回 CQ/native stats | verbs/CQ；前五项 ms，`elapsed_s` 秒 | RDMA 传输子项；同一传输的子项不应与外围 inclusive 相加 |
| `harness_command_*`, `harness_scp_metadata` | `perf_stage_e.py` 的 `Host.run`/`scp` span；wait-file/wait-health 由 232 coordinator 包围远端命令 | coordinator `monotonic_ns`，ms | 实验 Harness/SSH/轮询/验证；不是引擎或网络纯耗时 |
| `new_instance_launch_to_ready_harness` | `perf_stage_e.py:435-461` `target_started` 在 launch 前→wait-file + wait-health 后 `ready` | 232 coordinator `monotonic_ns`，ms | Harness 启动外围：SSH、admission、全量 source/weight hash、模型加载、engine init、health poll；包含 `model_weight_load` |
| `failure_to_decode_complete_harness` | `perf_stage_e.py:414-458` `recovery_start`（受控 loss 后）→16 Decode 完成 `decoded` | 232 coordinator `monotonic_ns`，ms | 包含 native recovery、delivery、independent checks、SSH、target startup、16 decodes、validation；明确不含故障发现 |

## 3. 五轮核心指标统计

| 指标 | 均值 (ms) | 标准差 (ms) | P50 (ms) | 解释 |
|---|---:|---:|---:|---|
| `new_instance_launch_to_ready_harness/coordinator-232/sdu-232/-` | 80568.630 | 820.128 | 80964.377 | 启动到 ready 的 Harness 外围 |
| `failure_to_decode_complete_harness/coordinator-232/sdu-232/-` | 117163.444 | 632.177 | 117491.769 | 受控 loss 到 16 次 Decode 完成的 Harness 外围 |
| `rs_recovery_rpc_inclusive/node0-232/sdu-232/-` | 1994.968 | 87.639 | 1935.100 | RS 恢复 RPC 外围 |
| `recover_adapter_inclusive/node0-232/sdu-232/-` | 9334.668 | 130.599 | 9408.105 | RS cold source adapter 外围 |
| `first_restored_decode_cuda/node1-233/target/disai-4090-3/0` | 2115.827 | 69.648 | 2102.219 | 首个恢复后 CUDA forward |
| `first_restored_decode_cuda/node1-233/target/disai-4090-3/1` | 2115.989 | 69.886 | 2102.280 | 首个恢复后 CUDA forward |
| `delivery_send_inclusive/node0-232/sdu-232/-` | 3267.266 | 60.486 | 3291.683 | RDMA source delivery 发送外围 |
| `delivery_receive_inclusive/node1-233/receiver/disai-4090-3/-` | 8293.652 | 142.000 | 8326.980 | RDMA source delivery 接收外围 |
| `gpu_kv_writeback/node1-233/target/disai-4090-3/0` | 61.245 | 1.707 | 61.286 | GPU KV 写回 |
| `gpu_kv_writeback/node1-233/target/disai-4090-3/1` | 73.644 | 1.072 | 73.289 | GPU KV 写回 |
| `gpu_layer_readback_validation/node1-233/target/disai-4090-3/0` | 58.894 | 0.534 | 59.066 | GPU KV 逐层正确性校验 |
| `gpu_layer_readback_validation/node1-233/target/disai-4090-3/1` | 73.703 | 1.290 | 73.133 | GPU KV 逐层正确性校验 |
| `rs_decode_ms` | 67.716 | 0.894 | 67.386 | RS 重构/解码 |
| `local_xor_encode/node0-232/sdu-232/-` | 14.009 | 5.733 | 11.443 | 本地 XOR 编码 |
| `local_xor_encode/node1-233/disai-4090-3/-` | 11.385 | 0.178 | 11.373 | 本地 XOR 编码 |
| `model_weight_load/node0-232/origin/sdu-232/0` | 4358.209 | 92.219 | 4331.978 | 模型权重加载 |
| `model_weight_load/node0-232/origin/sdu-232/1` | 4343.117 | 52.042 | 4348.129 | 模型权重加载 |
| `model_weight_load/node1-233/origin/disai-4090-3/0` | 4209.913 | 29.660 | 4221.954 | 模型权重加载 |
| `model_weight_load/node1-233/origin/disai-4090-3/1` | 3838.178 | 20.058 | 3830.919 | 模型权重加载 |
| `model_weight_load/node1-233/target/disai-4090-3/0` | 4180.052 | 45.232 | 4154.997 | 模型权重加载 |
| `model_weight_load/node1-233/target/disai-4090-3/1` | 3825.191 | 33.672 | 3833.412 | 模型权重加载 |

## 4. 四个大指标为何较大

### 4.1 `new_instance_launch_to_ready_harness`：均值 80,568.630 ms，标准差 820.128 ms，P50 80,964.377 ms
它从 target 进程启动前开始，直到远端 `run.json` 出现并且 `/health` 返回成功结束。代码明确把 SSH、GPU admission、全量 source/weight hash、权重加载、SGLang engine initialization 和 health polling 放在一个外围区间内。它不是“模型加载时间”：单独的 `model_weight_load` 仅约 3.8--4.36 s；其余时间来自进程启动、全量哈希/文件访问、引擎初始化和健康轮询。不能用 80.57 s 减去若干跨进程指标来构造纯启动时间，因为这些指标的边界和时钟不同且存在嵌套。
### 4.2 `failure_to_decode_complete_harness`：均值 117,163.444 ms，标准差 632.177 ms，P50 117,491.769 ms
该区间从受控 D1 loss 已确定后的 `recovery_start` 开始，到 16 次 Decode 完成的 `decoded` 时间戳结束。它包含 RS 恢复、source delivery、独立 source 校验、SSH/命令开销、target 新实例启动、16 次 Decode 及验证；不包含故障发现。Stage-E wait-file 均值 130.884 s、wait-health 均值 86.404 s 是 Harness 命令序列的累计样本，和该 E2E 区间并非同一层级，不能相加。
### 4.3 `rs_recovery_rpc_inclusive`：均值 1,994.968 ms，标准差 87.639 ms，P50 1,935.100 ms
该指标包住 `rdma_reconstruct_missing_node()`：TCP 准备请求、RDMA survivor 上传、server 状态等待、target 重构块接收以及 checksum/manifest 校验。它不是 `rs_decode_ms`（约 67.716 ms）的同义词；后者只代表 RS 重构/解码子阶段。约 1.995 s 主要反映跨节点控制往返、RDMA 两条腿和等待/校验外围。
### 4.4 `recover_adapter_inclusive`：均值 9,334.668 ms，标准差 130.599 ms，P50 9,408.105 ms
`verify_rs_cold_resume.py:recover()` 先读取并校验 protection、执行 survivor XOR hash 校验，再调用 `recover_node()`，读取 recovered source、解包目标 block、写出 `recovered_node.bin` 并生成 Stage-E source evidence。它是 RS cold-source adapter 的完整外围，不是单纯 RS 算法；可能包含 `rs_recovery_rpc_inclusive` 和文件/哈希/归档操作。
### 4.5 `first_restored_decode_cuda`：均值约 2,115.9 ms，标准差约 69.8 ms，P50 2,102.2 ms
实现是在首个恢复 forward 前记录 CUDA Event，在 `observe_model_forward` 中记录结束 Event。区间包括 native forward stream 上的首次恢复 Decode、首次 kernel/allocator/cache/warm-up 影响及等待；不包括完整 16 Decode，也不是端到端恢复时间。正确性 `decode_forward_validation` 是独立的 CPU/状态验证指标，不能从该值中推断。

## 5. GPU、编解码、RDMA、引擎和验证开销的分层判断

- GPU 数据：`gpu_kv_snapshot` 约 50.9--54.2 ms，`gpu_snapshot_gather` 约 20.4--22.7 ms，`gpu_to_host` 约 1.47--2.22 ms，`gpu_kv_writeback` 约 61.2--73.6 ms，逐层 readback validation 约 58.9--73.7 ms。写回指标含 host prepare/scatter/fence，不能当作纯 H2D。
- XOR/RS：local XOR encode 约 11.4--14.0 ms，local validation 约 34.2--59.5 ms；`rs_decode_ms` 约 67.7 ms；`server_rs_encode_ms` 约 60.3 ms。validation 单列，不能算作算法必要计算。
- RDMA：node0/node1 upload wall 约 1.773/1.721 s；native upload `elapsed_s` 汇总约 99--121 ms/110--114 ms（角色和传输定义不同）；delivery send/receive inclusive 约 3.267/8.294 s，包含外围等待、归档校验和跨进程协同。
- SGLang 初始化/权重：`model_weight_load` 约 3.825--4.358 s；它只是启动外围的一部分。`new_instance_launch_to_ready_harness` 还包含 admission、hash、engine init、health。
- 请求恢复/首次 Decode/验证：`scheduler_request_rebuild` 约 3.5--3.9 ms；`first_restored_decode_cuda` 约 2.116 s；每次 `decode_forward_validation` 是独立 correctness observer，不能累加成恢复关键路径。
- Harness：Stage-E 的 wait-file、wait-health、launch、SCP、独立 source check 和 acceptance check 明确属于实验开销。其高值不能直接归因于 XOR/RS 或 GPU KV 恢复算法。

## 6. Native `elapsed_s` 单位核查

结论：汇总输出的单位转换实现正确。

- `multinode/rdma_transport.py` 返回 `elapsed_s`（秒）；例如 trial-000 的 `delivery/send/elapsed_s` 原始值为 0.222700318787 s。
- `perf_stage_e.py:native_metrics()` 对字段名为 `elapsed_s` 的值乘 1000，写入 `native-metrics.json`/汇总器作为 ms；因此对应汇总均值 211.745 ms，而不是 211.745 s。
- 同一逻辑用于 `node*/upload_native/elapsed_s` 和四条 recovery native legs。汇总值如 `recovery/client_receive_stats/elapsed_s` 均值 1,441.037 ms，等价于约 1.441 s。
- 其它 `*_ms` 字段不乘 1000。代码按字段名转换，不存在重复乘 1000 的证据。

## 7. 能否构造严格恢复关键路径

目前不能从这些日志构造一个严格、单一时钟、无重叠的“故障发生→首个恢复 token”关键路径。可确认的边界是：

1. 232 coordinator 的 `recovery_start → decoded`：定义清楚，但它是 Harness E2E 外围，内部跨主机并包含大量嵌套阶段。
2. target 的 `target_started → ready`：定义清楚，是新实例 ready 外围；`ready → restore → first decode` 的完整同机边界没有单独的 coordinator 指标。
3. 各进程的 observer/CUDA/CQ 区间：定义清楚，但来自不同主机/进程，不能直接用 monotonic 时间戳相减，也不能将 inclusive 与子阶段相加。
4. 缺失事件：故障检测完成、恢复命令发出、RS RPC 开始/结束的统一 coordinator marker、delivery 完成、GPU restore commit、scheduler rebuild 完成、首个 Decode 完成，需要一个统一的逻辑事件 ID/序号；目前只有部分事件文件和外围区间。
5. 最小补充方式：在 232 coordinator 只增加逻辑 marker（wall-clock duration 仍由单一 coordinator monotonic 计时），并在每个远端事件写入 `trial_id/phase_seq/event_name`；补充 `fault_detected`、`recovery_rpc_begin/end`、`delivery_begin/end`、`target_ready`、`restore_committed`、`scheduler_rebuilt`、`first_decode_done`、`decode_complete`。不要读取或相减不同主机的 monotonic_ns；用事件序号建立因果关系，用各自本地 duration 报告耗时。

## 8. 当前最值得优化的三个方面

1. **新实例启动到 ready**：先拆分并优化 full source/weight hashing、模型权重加载、engine initialization、health polling；80.57 s 中只有约 4 s 明确属于权重加载，必须先补齐细分计时再决定缓存或并行化。
2. **Stage-E Harness 等待与跨节点交付**：wait-file/wait-health 和 source delivery 的 3.27/8.29 s 外围应减少轮询、SSH/SCP 和重复校验；保持当前算法不变即可验证收益。
3. **恢复后的首次 Decode 冷启动**：2.116 s 明显高于后续单步 validation/forward 量级，应拆分首次 CUDA graph/allocator/kernel warm-up、scheduler rebuild 和真正 forward；不要直接修改 GPU KV 恢复算法。

## 9. 与 Re-Prefill 公平对照的最小验证方案

- **固定模型与代码**：Qwen3-8B 同一模型哈希、同一 commit/tracked diff、同一 CUDA/PyTorch 环境；两方案均 TP=2，使用相同两台物理机和同样 GPU 绑定。
- **固定输入与采样**：相同 256-token Prefix、相同 input IDs、相同 greedy/no-RNG 采样参数、相同 17 个输出目标（或同一 16 Decode continuation 加终止条件）；关闭 radix cache、overlap、chunked prefill，CUDA graph 设置一致。
- **固定故障起点**：两方案都先完成同一 Prefix 的原始服务和 capture；在相同“D1/node1 输入被撤销且故障已被注入”的逻辑边界开始计时。不要把故障发现时间计入 Re-Prefill 而排除在 Stage E 外。
- **固定新实例条件**：两方案都启动一个全新的 target 服务，使用相同模型、TP、GPU、端口资源和启动命令；Stage E 以 `/health` HTTP 200 且本轮进程身份确认作为 ready，Re-Prefill 也必须使用同一 ready 定义。
- **固定完成条件**：两方案都以最后一个规定 Decode 输出完成并通过相同 token-id/状态正确性检查为终点；同时记录首 token/首 Decode 和完整 16 Decode 两个终点。
- **只比较同口径指标**：主指标为 `fault_boundary → decode_complete` 的 coordinator 单机 monotonic duration；次指标为 `new_instance_launch_to_ready`、首个 Decode、完整 Decode。分别报告 Stage E 的 RS/GPU 恢复子项和 Re-Prefill 的重新 Prefill 时间，不能把 Stage E 的包含阶段求和。
- **重复与统计**：两方案各至少 5 轮，随机化交替顺序，保持文件系统/服务预热策略一致；报告均值、标准差、P50，并保留每轮原始事件和正确性结果。

## 10. 全部 130 组汇总序列

下表保留每个汇总序列的均值、标准差、P50。单位均为 ms；P95/P99 在 n=5 时按汇总器规则为 null。指标的代码/边界归类见第 2 节。

| 指标 | 均值 | 标准差 | P50 |
|---|---:|---:|---:|
| `capture_artifact_inclusive/node0-232/origin/sdu-232/-` | 407.789 | 12.147 | 405.738 |
| `capture_artifact_inclusive/node1-233/origin/disai-4090-3/-` | 397.998 | 18.615 | 391.849 |
| `checkpoint_load_validate_inclusive/node0-232/origin/sdu-232/-` | 1281.561 | 27.789 | 1283.732 |
| `checkpoint_load_validate_inclusive/node0-232/sdu-232/-` | 2692.577 | 40.528 | 2698.197 |
| `checkpoint_load_validate_inclusive/node1-233/disai-4090-3/-` | 11321.894 | 58.903 | 11307.690 |
| `checkpoint_load_validate_inclusive/node1-233/origin/disai-4090-3/-` | 1230.737 | 12.797 | 1224.982 |
| `checkpoint_load_validate_inclusive/node1-233/target/disai-4090-3/-` | 11108.713 | 69.459 | 11076.195 |
| `decode_forward_validation/node0-232/origin/sdu-232/-` | 480.311 | 24.950 | 487.616 |
| `decode_forward_validation/node1-233/origin/disai-4090-3/-` | 418.648 | 15.722 | 419.970 |
| `decode_forward_validation/node1-233/target/disai-4090-3/-` | 423.278 | 49.488 | 440.439 |
| `delivery/send/completion_ms` | 99.081 | 1.085 | 98.950 |
| `delivery/send/device_setup_ms` | 4.092 | 0.140 | 4.030 |
| `delivery/send/elapsed_s` | 211.745 | 13.011 | 203.122 |
| `delivery/send/exchange_ms` | 41.144 | 3.813 | 39.858 |
| `delivery/send/memory_registration_ms` | 3.030 | 1.663 | 1.961 |
| `delivery/send/qp_setup_ms` | 1.045 | 0.073 | 1.041 |
| `delivery_receive_inclusive/node1-233/receiver/disai-4090-3/-` | 8293.652 | 142.000 | 8326.980 |
| `delivery_send_inclusive/node0-232/sdu-232/-` | 3267.266 | 60.486 | 3291.683 |
| `failure_to_decode_complete_harness/coordinator-232/sdu-232/-` | 117163.444 | 632.177 | 117491.769 |
| `first_restored_decode_cuda/node1-233/target/disai-4090-3/0` | 2115.827 | 69.648 | 2102.219 |
| `first_restored_decode_cuda/node1-233/target/disai-4090-3/1` | 2115.989 | 69.886 | 2102.280 |
| `global_committed/node0-232/sdu-232/-` | 0.000 | 0.000 | 0.000 |
| `global_committed/node1-233/disai-4090-3/-` | 0.000 | 0.000 | 0.000 |
| `gpu_kv_snapshot/node0-232/origin/sdu-232/0` | 50.873 | 1.821 | 50.407 |
| `gpu_kv_snapshot/node0-232/origin/sdu-232/1` | 51.191 | 1.875 | 51.418 |
| `gpu_kv_snapshot/node1-233/origin/disai-4090-3/0` | 53.507 | 9.234 | 49.803 |
| `gpu_kv_snapshot/node1-233/origin/disai-4090-3/1` | 54.177 | 9.552 | 49.988 |
| `gpu_kv_writeback/node1-233/target/disai-4090-3/0` | 61.245 | 1.707 | 61.286 |
| `gpu_kv_writeback/node1-233/target/disai-4090-3/1` | 73.644 | 1.072 | 73.289 |
| `gpu_layer_readback_validation/node1-233/target/disai-4090-3/0` | 58.894 | 0.534 | 59.066 |
| `gpu_layer_readback_validation/node1-233/target/disai-4090-3/1` | 73.703 | 1.290 | 73.133 |
| `gpu_snapshot_gather/node0-232/origin/sdu-232/0` | 20.414 | 0.676 | 20.073 |
| `gpu_snapshot_gather/node0-232/origin/sdu-232/1` | 20.406 | 0.767 | 19.986 |
| `gpu_snapshot_gather/node1-233/origin/disai-4090-3/0` | 22.685 | 4.972 | 20.386 |
| `gpu_snapshot_gather/node1-233/origin/disai-4090-3/1` | 22.286 | 5.171 | 20.044 |
| `gpu_to_host/node0-232/origin/sdu-232/0` | 1.468 | 0.003 | 1.466 |
| `gpu_to_host/node0-232/origin/sdu-232/1` | 1.466 | 0.003 | 1.466 |
| `gpu_to_host/node1-233/origin/disai-4090-3/0` | 2.216 | 0.006 | 2.218 |
| `gpu_to_host/node1-233/origin/disai-4090-3/1` | 2.212 | 0.005 | 2.210 |
| `harness_command_perf_benchmark_inventory/coordinator-232/sdu-232/-` | 680.588 | 184.364 | 679.997 |
| `harness_command_perf_stage_e_bundle-json/coordinator-232/sdu-232/-` | 215.173 | 0.234 | 215.263 |
| `harness_command_perf_stage_e_init-trial/coordinator-232/sdu-232/-` | 114.512 | 0.239 | 114.585 |
| `harness_command_perf_stage_e_prepare/coordinator-232/sdu-232/-` | 229.318 | 0.387 | 229.387 |
| `harness_command_perf_stage_e_stop/coordinator-232/sdu-232/-` | 343.770 | 0.254 | 343.821 |
| `harness_command_perf_stage_e_wait-file/coordinator-232/sdu-232/-` | 130884.088 | 828.196 | 130484.880 |
| `harness_command_perf_stage_e_wait-health/coordinator-232/sdu-232/-` | 86403.702 | 1070.123 | 86354.029 |
| `harness_command_verify_gpu_cold_resume_reference/coordinator-232/sdu-232/-` | 6306.133 | 278.639 | 6197.141 |
| `harness_command_verify_gpu_cold_resume_submit/coordinator-232/sdu-232/-` | 15294.717 | 263.761 | 15374.755 |
| `harness_command_verify_rs_cold_resume_check-source/coordinator-232/sdu-232/-` | 5168.359 | 42.890 | 5177.807 |
| `harness_command_verify_rs_cold_resume_check/coordinator-232/sdu-232/-` | 13453.137 | 126.154 | 13453.282 |
| `harness_command_verify_rs_cold_resume_claim/coordinator-232/sdu-232/-` | 5956.901 | 91.018 | 5951.762 |
| `harness_command_verify_rs_cold_resume_export/coordinator-232/sdu-232/-` | 16372.447 | 311.012 | 16267.482 |
| `harness_command_verify_rs_cold_resume_plan/coordinator-232/sdu-232/-` | 1780.983 | 54.100 | 1770.459 |
| `harness_command_verify_rs_cold_resume_protect/coordinator-232/sdu-232/-` | 1800.072 | 27.462 | 1819.076 |
| `harness_command_verify_rs_cold_resume_recover/coordinator-232/sdu-232/-` | 11165.495 | 135.538 | 11195.442 |
| `harness_command_verify_rs_cold_resume_restore/coordinator-232/sdu-232/-` | 9731.710 | 155.151 | 9742.797 |
| `harness_command_verify_rs_cold_resume_send-source/coordinator-232/sdu-232/-` | 5108.284 | 103.362 | 5128.779 |
| `harness_scp_metadata/coordinator-232/sdu-232/-` | 273.076 | 20.947 | 275.782 |
| `independent_source_validation/node1-233/disai-4090-3/-` | 7036.253 | 33.027 | 7020.968 |
| `independent_source_validation/node1-233/target/disai-4090-3/-` | 3435.163 | 30.721 | 3443.613 |
| `local_xor_encode/node0-232/sdu-232/-` | 14.009 | 5.733 | 11.443 |
| `local_xor_encode/node1-233/disai-4090-3/-` | 11.385 | 0.178 | 11.373 |
| `local_xor_validation/node0-232/sdu-232/-` | 59.545 | 56.151 | 34.628 |
| `local_xor_validation/node1-233/disai-4090-3/-` | 34.156 | 0.250 | 34.150 |
| `model_weight_load/node0-232/origin/sdu-232/0` | 4358.209 | 92.219 | 4331.978 |
| `model_weight_load/node0-232/origin/sdu-232/1` | 4343.117 | 52.042 | 4348.129 |
| `model_weight_load/node1-233/origin/disai-4090-3/0` | 4209.913 | 29.660 | 4221.954 |
| `model_weight_load/node1-233/origin/disai-4090-3/1` | 3838.178 | 20.058 | 3830.919 |
| `model_weight_load/node1-233/target/disai-4090-3/0` | 4180.052 | 45.232 | 4154.997 |
| `model_weight_load/node1-233/target/disai-4090-3/1` | 3825.191 | 33.672 | 3833.412 |
| `new_instance_launch_to_ready_harness/coordinator-232/sdu-232/-` | 80568.630 | 820.128 | 80964.377 |
| `node0/global_commit_wait_ms` | 0.002 | 0.000 | 0.002 |
| `node0/rdma_upload_wall_ms` | 1773.411 | 58.467 | 1759.795 |
| `node0/server_commit_ms` | 1923.219 | 193.762 | 1809.414 |
| `node0/server_rs_encode_ms` | 60.339 | 3.095 | 58.906 |
| `node0/upload_native/completion_ms` | 48.273 | 4.699 | 47.856 |
| `node0/upload_native/device_setup_ms` | 3.910 | 0.173 | 3.812 |
| `node0/upload_native/elapsed_s` | 110.781 | 9.620 | 108.357 |
| `node0/upload_native/exchange_ms` | 23.474 | 0.758 | 23.910 |
| `node0/upload_native/memory_registration_ms` | 1.550 | 0.788 | 1.288 |
| `node0/upload_native/qp_setup_ms` | 1.031 | 0.094 | 1.090 |
| `node1/global_commit_wait_ms` | 0.002 | 0.000 | 0.002 |
| `node1/rdma_upload_wall_ms` | 1721.104 | 59.075 | 1733.455 |
| `node1/server_commit_ms` | 1923.219 | 193.762 | 1809.414 |
| `node1/server_rs_encode_ms` | 60.339 | 3.095 | 58.906 |
| `node1/upload_native/completion_ms` | 48.189 | 3.336 | 48.449 |
| `node1/upload_native/device_setup_ms` | 5.095 | 0.509 | 4.962 |
| `node1/upload_native/elapsed_s` | 110.468 | 3.016 | 110.782 |
| `node1/upload_native/exchange_ms` | 23.550 | 0.320 | 23.502 |
| `node1/upload_native/memory_registration_ms` | 1.281 | 0.254 | 1.116 |
| `node1/upload_native/qp_setup_ms` | 0.783 | 0.033 | 0.787 |
| `protection_inclusive/node0-232/sdu-232/-` | 6379.650 | 158.192 | 6291.823 |
| `protection_inclusive/node1-233/disai-4090-3/-` | 6353.345 | 176.624 | 6229.909 |
| `recover_adapter_inclusive/node0-232/sdu-232/-` | 9334.668 | 130.599 | 9408.105 |
| `recovered_block_return/node0-232/sdu-232/-` | 1441.037 | 50.715 | 1439.534 |
| `recovery/client_receive_stats/completion_ms` | 51.591 | 8.125 | 54.687 |
| `recovery/client_receive_stats/device_setup_ms` | 4.444 | 0.321 | 4.628 |
| `recovery/client_receive_stats/elapsed_s` | 1441.037 | 50.715 | 1439.534 |
| `recovery/client_receive_stats/exchange_ms` | 0.631 | 0.039 | 0.637 |
| `recovery/client_receive_stats/memory_registration_ms` | 20.702 | 2.027 | 21.047 |
| `recovery/client_receive_stats/qp_setup_ms` | 0.802 | 0.080 | 0.798 |
| `recovery/client_send_stats/completion_ms` | 51.277 | 4.914 | 52.623 |
| `recovery/client_send_stats/device_setup_ms` | 4.025 | 0.163 | 4.037 |
| `recovery/client_send_stats/elapsed_s` | 114.648 | 5.155 | 115.191 |
| `recovery/client_send_stats/exchange_ms` | 23.368 | 0.452 | 23.496 |
| `recovery/client_send_stats/memory_registration_ms` | 1.348 | 0.352 | 1.204 |
| `recovery/client_send_stats/qp_setup_ms` | 0.982 | 0.084 | 0.941 |
| `recovery/receive_stats/completion_ms` | 51.323 | 4.842 | 52.658 |
| `recovery/receive_stats/device_setup_ms` | 5.338 | 0.138 | 5.330 |
| `recovery/receive_stats/elapsed_s` | 177.909 | 17.380 | 182.635 |
| `recovery/receive_stats/exchange_ms` | 0.649 | 0.054 | 0.638 |
| `recovery/receive_stats/memory_registration_ms` | 16.224 | 0.645 | 16.523 |
| `recovery/receive_stats/qp_setup_ms` | 0.844 | 0.086 | 0.819 |
| `recovery/send_stats/completion_ms` | 51.485 | 8.158 | 54.654 |
| `recovery/send_stats/device_setup_ms` | 4.986 | 0.026 | 4.998 |
| `recovery/send_stats/elapsed_s` | 121.961 | 14.771 | 123.945 |
| `recovery/send_stats/exchange_ms` | 26.926 | 2.421 | 27.171 |
| `recovery/send_stats/memory_registration_ms` | 1.824 | 0.321 | 1.690 |
| `recovery/send_stats/qp_setup_ms` | 1.000 | 0.052 | 0.992 |
| `rs_decode_ms` | 67.716 | 0.894 | 67.386 |
| `rs_recovery_rpc_inclusive/node0-232/sdu-232/-` | 1994.968 | 87.639 | 1935.100 |
| `rs_source_load_validate_inclusive/node0-232/sdu-232/-` | 4915.912 | 92.821 | 4922.110 |
| `rs_source_load_validate_inclusive/node1-233/disai-4090-3/-` | 13874.192 | 112.355 | 13823.353 |
| `rs_source_load_validate_inclusive/node1-233/receiver/disai-4090-3/-` | 4974.055 | 74.863 | 4959.261 |
| `rs_source_load_validate_inclusive/node1-233/target/disai-4090-3/-` | 13444.247 | 96.600 | 13437.863 |
| `rs_upload_inclusive/node0-232/sdu-232/-` | 1772.469 | 58.555 | 1759.136 |
| `rs_upload_inclusive/node1-233/disai-4090-3/-` | 1720.475 | 59.110 | 1732.889 |
| `scheduler_request_rebuild/node1-233/target/disai-4090-3/0` | 3.930 | 1.686 | 3.167 |
| `scheduler_request_rebuild/node1-233/target/disai-4090-3/1` | 3.517 | 1.612 | 2.810 |
| `survivor_upload/node0-232/sdu-232/-` | 114.648 | 5.155 | 115.191 |

## 11. 证据路径

- 汇总：`/tmp/kv-perf-232.u9wlhC8o/summary-r5.json`
- 原始 trial：`/tmp/kv-perf-232.u9wlhC8o/baseline-r5/`
- 计时口径文档：`experiments/kv_ec/PERF_STAGE_E_256.md`
- 本报告：`experiments/kv_ec/StageE真实性能分析-r5.md`

## 12. Re-Prefill 对照：现有接口与可运行方案

### 12.1 现有接口能做什么

`experiments.kv_ec.perf_benchmark inference` 已实现 `--scheme no-ec|reprefill`。它在同一个已经就绪的普通 SGLang 服务上提交 256-token 请求，记录：

- `request_ms`：客户端收到完整 17 个输出 token 的总时延；
- `prefill_ttft_ms`：客户端收到第一个生成 token 的时延；
- `cached_tokens=0`、`prompt_tokens=256`、`completion_tokens=17` 的硬校验；
- `server_info` 对 TP=2、关闭 radix cache、关闭 overlap、`chunked_prefill_size=-1`、单请求和 disabled CUDA graph 的硬校验。

代码还明确记录：`reprefill` 从显式客户端重新提交开始计时，不包含故障发现和新实例启动。因此它不能直接代表冷启动 Re-Prefill 的完整恢复时间。

### 12.2 A 类：目标实例已经就绪

A 类比较“目标服务已经 ready 后，恢复 KV 与重新 Prefill”的请求处理阶段。最小可执行流程如下，`URL_READY` 必须分别指向两个独立、已就绪且配置完全相同的普通服务；不要指向 Stage E Restore Hook 服务。

```bash
cd /media/sda1/iic/luotao/sglangec
export MODEL=/media/sda1/iic/luotao/models/Qwen3-8B
export INPUT=/tmp/kv-stage-e-perf-input-256.json
export URL_READY=http://127.0.0.1:<ready-port>
export OUT=/tmp/kv-reprefill-a-$(date +%s)

# 重新 Prefill 控制（只从显式重新提交开始计时）
python -m experiments.kv_ec.perf_benchmark inference \
  --scheme reprefill --url "$URL_READY" --model "$MODEL" \
  --input-ids "$INPUT" --output "$OUT/reprefill" \
  --warmup 2 --repeat 5

# 同一 ready 实例上的 no-EC 请求基线
python -m experiments.kv_ec.perf_benchmark inference \
  --scheme no-ec --url "$URL_READY" --model "$MODEL" \
  --input-ids "$INPUT" --output "$OUT/no-ec" \
  --warmup 2 --repeat 5
```

A 类的生产相关比较应使用 Stage E target 已完成 restore、`restore committed`、Scheduler 已重建之后的同一 ready 服务，并用同一个客户端边界记录“恢复后的首个 Decode/首 token”和“完整 17 token”。当前 `inference` CLI 不能连接到 Stage E 恢复服务并自动完成这些状态绑定，因此 A 类还需要一个已有 target 实例的人工编排步骤；没有该绑定时，只有普通服务的请求基线，不能宣称已完成 KV 恢复对照。

### 12.3 B 类：目标实例需要冷启动

B 类要求两条路径均包含：故障边界、全新 target 进程启动、相同模型加载/引擎初始化、ready 条件，以及后续 256-token Prefill 或 KV Restore 和 17-token Decode。当前 `inference` 接口不启动服务、不注入故障、不等待新实例 ready，因此 **B 类尚未实现，不能构造模拟 PASS 或直接复用 Stage E 总时长**。

最小公平实现应复用现有 Stage E launcher 的资源准入和 `/health` 身份检查，但只增加实验编排和客户端计时，不修改恢复算法或正确性门槛：

1. 对照组：注入同一受控故障后停止 origin，启动全新 target；等待同一 `run.json` + `/health` ready；用同一 256 输入执行完整 Prefill，再执行 16 Decode。
2. KV-EC 组：故障起点、target 启动、ready 条件完全相同；ready 后执行现有 KV Restore，再执行 16 Decode。
3. 分别记录 `fault_boundary→target_started`、`target_started→ready`、`ready→first_token`、`first_token→decode_complete`；端点都由 232 coordinator 的单一 monotonic 时钟记录，远端只写事件序号，不相减远端 monotonic 时间戳。
4. 两组各至少 5 轮，交替运行，固定 warmup、文件系统预热策略、GPU 绑定、模型哈希和输入哈希。

在该最小编排完成前，报告中的 Stage E `new_instance_launch_to_ready_harness` 只能作为共同冷启动成本参考，不能与 A 类 ready-server `reprefill` 请求时延直接比较。

## 13. 是否具备扩展到 512/1024/2048/4096 Token 的条件

当前不具备“直接扩展并宣称可比”的条件，限制如下：

- Stage E 验收和 `perf_benchmark inference` 都硬编码 256 token：`input_ids` 长度必须为 256，状态校验要求 `token_count=256`、`resume_decode_forward_count=16`；
- capture 元数据、请求状态、KV block 大小、显存 admission 和 `max_node_bytes` 需按长度重新核算；
- 512 及以上会改变 GPU snapshot/D2H、XOR、RS payload、RDMA、写回和首次 Decode 的工作量，不能沿用 256 的内存预算或阈值；
- 需要为每个长度重新生成匹配的 template/input IDs、重新执行两节点 preflight，并重新确认 GPU 空闲、RS stripe/byte budget、RDMA 端口和磁盘空间；
- 4096 token 可能触发更高的模型上下文、KV buffer 和恢复临时内存需求；当前没有该长度的真实 PASS 证据。

可扩展的前置条件是：先把长度作为配置参数贯穿 capture/state/acceptance/checker 和 `inference` 的硬校验，再为每个长度单独跑最小 CPU 回归和至少 5 轮真实 Stage E；在此之前，512/1024/2048/4096 均应标记为 NOT IMPLEMENTED/未验证。

## 14. 下一阶段实验矩阵

| 类别 | 方案 | 目标 | 主要计时 | 正确性门槛 |
|---|---|---|---|---|
| 功能正确性 | Stage E two-level，256，5 轮 | 保持已有基线，防止性能改动破坏功能 | PASS、16 Decode、token IDs、资源无泄漏 | 现有全部 Stage E gate |
| 故障恢复性能 | Stage E two-level，256，>=5 轮 | 保护/恢复真实成本 | protection、RS RPC、delivery、restore/writeback、首 Decode、完整 Decode | 同上，不改 gate |
| 故障恢复性能 | ready target：KV Restore vs ready target Re-Prefill | 隔离 KV 重建与 Prefill 处理成本 | ready→first token、ready→complete | 同输入/TP/采样/输出 |
| 故障恢复性能 | cold target：KV Restore vs cold Re-Prefill | 比较完整冷启动恢复路径 | fault boundary→ready、ready→first token、完整 Decode | 相同新实例和 `/health` 条件 |
| 正常推理开销 | `inference --scheme no-ec` | 无 EC 请求基线 | request_ms、prefill_ttft_ms | cached_tokens=0，17 输出 |
| 正常推理开销 | `inference --scheme reprefill` | 显式重新 Prefill 控制 | request_ms、prefill_ttft_ms | 与 no-EC 完全同配置 |
| 扩展性 | 512/1024/2048/4096 | 长度扩展 | 同上全部分层指标 | 当前均未验证 |

## 15. 测量限制与禁止的推断

- 不把 `new_instance_launch_to_ready_harness` 减去 `model_weight_load` 后称为纯 engine init；边界、进程和文件校验不同，剩余项未被独立计时。
- 不把 `recover_adapter_inclusive` 视为 RS 算法时间；它包含 source 文件、哈希、RPC/传输外围和 evidence 生成，而 `rs_decode_ms` 仅是其中一个子阶段。
- 不把 `first_restored_decode_cuda` 视为完整恢复时延；它是 CUDA stream 上首次 forward 区间。
- 不把不同物理机的 monotonic 时间戳相减；不把 inclusive 外围和其内部 native/validation 子项相加。
- 不把 Stage E Harness 的 SSH、SCP、wait-file、wait-health、完整哈希和验收检查当成生产恢复必要开销；它们必须单独列出。
- 在 B 类冷启动 Re-Prefill 尚未按相同故障起点和 ready 条件实现前，不报告“KV 恢复比 Re-Prefill 快多少”的数值结论。
