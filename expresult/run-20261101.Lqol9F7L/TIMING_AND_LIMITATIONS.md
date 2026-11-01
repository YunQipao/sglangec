# 计时语义：哪些数能比较，哪些不能

## 1. 两条路径并没有相同的完成终点

通俗地说：Two-level 目前测的是“恢复作业及其强诊断走到指定位置”，Re-Prefill 新计时测的是“客户端收到了完整响应”。即使都叫 Cold Recovery，也不能直接当同一把尺。

### Two-level 旧端到端区间

见[历史匹配编排源码](source_snapshot/experiments/kv_ec/perf_stage_e.py.txt)的 `run_trial()`：

- 起点：保护流程成功、进入显式受控 D1 丢失后的恢复分支时，在 232 coordinator 记录 `recovery_start = time.monotonic_ns()`。不是检测到物理服务器宕机的时刻。
- 终点：远端 `restore` 命令返回后，232 coordinator 记录 `decoded`。后面还有独立的 `E check`、证据拷贝和验收绑定；因此也不能把 `decoded` 称为所有验收工作全部结束。
- `failure_to_decode_complete_harness` / `stage-e-timing.json` 的 `fault_boundary_to_decode_complete_ms`：上述起止，包含恢复、交付、流程内检查、SSH、目标启动、恢复后 Decode 和对应诊断。
- `new_instance_launch_to_ready_harness`：发起目标 launch 前的 `target_started` 到启动记录与 health 检查完成后的 `ready`；包括 SSH、资源准入、源码/权重哈希、加载模型、引擎/GPU 初始化和轮询。
- `fault_boundary_to_ready_ms`：从恢复分支起点到 Ready，除启动外还包含启动之前的恢复准备和交付等；它不等于纯启动耗时。
- `ready_to_decode_complete_ms`：同一 coordinator 的 Ready 到 restore 命令返回；仍非普通客户端 response-complete。

三轮均有 `cold-comparison-timing-status.json`，明确 performance_measurement_eligible=false：没有等价 SSE 客户端响应，缺少 first-output、response-complete 和 response-stream 指标。**不要用 Ready+某个 GPU span 拼出缺失的 TTFT 或响应完成值。**

### Re-Prefill 新客户端区间

见[原始计时](evidence/cold-reprefill/trial-000-41869374fcfd40bfad4e4abe565ce38e/cold-comparison-timing.json)、[请求记录](evidence/cold-reprefill/trial-000-41869374fcfd40bfad4e4abe565ce38e/reprefill-inference/request-000.json)、[请求实现](source_snapshot/experiments/kv_ec/perf_benchmark.py.txt)和[计时定义](source_snapshot/experiments/kv_ec/cold_timing.py.txt)。

|事件|含义|
|---|---|
|fault_boundary|同一编排恢复分支边界；不含真实故障检测。|
|target_launch|启动目标实例之前。|
|target_ready|启动记录和健康检查完成之后。|
|request_start|客户端开始发送 HTTP 请求前的 monotonic 时间。|
|first_output_token|解析到第一个 completion_tokens>0 的 SSE 数据项；是客户端观测，可能一次已含多个 Token，不保证首个 Token 的 GPU 生成时刻。|
|full_output|第一次解析到 completion_tokens=17 的数据项。|
|response_complete|客户端流读取/关闭完成；区别于 full_output，也区别于后续正确性校验完成。|
|validation_complete|编排读取请求结果并检查输出 IDs 与参考一致后；不是任意后续 cleanup/归档结束。|

每个区间通过 `events_ns[end]-events_ns[start]` 得到，离线工具已复核全部新计时等式。`response_stream_ms` 从 request_start 开始，不是从 first_output 开始；TTFT 也是 request_start→first_output，不含目标启动。

Re-Prefill 的 performance_measurement_eligible=true 仅表示这一路有上述可解析测量，**不表示具备算法公平对比资格**；algorithm_comparison_eligible=false、production_recovery=false。Two-level 缺少对应文件，所以本包将其新计时资格保持为 false，而非填 0。0 会错误暗示“没有耗时”。

## 2. 组件时间的使用规则

1. inclusive 表示包括子调用和函数内等待。不能把 adapter、RPC、文件加载与其子事件逐个相加得到总恢复。
2. GPU 两个 Rank 并发且有 TP 同步，不应把两 Rank 写回或 Forward 时间相加当作端到端延迟；仅取 max 也不能代替真实关键路径测量。
3. `capture_artifact_inclusive` 包括生成 Capture artifact 的宽流程，不等于只读 GPU；每角色可能出现多次，CSV 统计是每轮同分组累计。它通常属于故障前准备。
4. `checkpoint_load_validate_inclusive` 包括加载/验证函数内工作。逐次与累计均保留，但不是“可从总时间剔除的免费诊断”。
5. RDMA 的 device setup、memory registration、QP setup、exchange、completion 与 elapsed 可能互相包含。elapsed/传输 wall time 含等待和控制流程，不是纯 payload 带宽；包内提供原始 CQ 支持追溯，而不伪造统一裸传输时间。
6. GPU 写回 span 是实现所测区间；如包含同步/校验，以实际代码为准，不要改称纯 PCIe DMA 峰值。
7. 原始 native-metrics 的 `/elapsed_s` 值已转为 ms，但原 CQ 中 elapsed_s 为秒；不要重复换算。

### 首次恢复 Decode CUDA 约 2.17 s，能解释成什么？

Hook 在首个 Forward 前记录 CUDA start event，在进入 Forward 后观察回调、诊断 observer 之前记录 end event，利用既有完成 fence 读取 event elapsed。它不为计时再添加一个 CUDA synchronize。

它反映该原生 Forward stream 的首轮执行区间，可能受 TP 通信、GPU 调度、首轮 kernel/graph/编译相关成本影响。但这个单一指标不能分辨各项，更不能证明“2 秒全部是 CUDA 初始化”。模型/GPU 初始化中很多工作已在 Ready 之前发生。

同样不能把它当稳态每 Token 延迟；还没有本包所需的完整后续 Forward 分布和两种方法对称观测。保留现有证据，不凭猜测扣掉“首次成本”。

## 3. 时钟域与身份

232 coordinator 的旧 E2E 起止来自同一主机 monotonic，可在该区间内相减。Re-Prefill 新事件也按该 coordinator 时间域记录。其他节点的 monotonic、远端进程 wall 时间和 CUDA event 则属于其他时间域，不能跨主机拿绝对时间戳相减；NTP 对时也不自动让 monotonic 原点相同。

`analysis/perf_events.csv`只导出历史记录已有的身份字段；空白代表未记录，不是已验证相同。原始 perf JSONL 保留全部字段（如果有 boot_id 等）。不能追补出历史事件不存在的唯一 operation/span 身份。源码 inventory 哈希对应也不能代替时钟域身份验证。

## 4. 可比性分层

|问题|当前证据能否回答|
|---|---|
|指定场景下两种路径能否输出正确结果|能：Two-level 3 轮、Re-Prefill 1 轮；检查项不完全相同，应分别报告。|
|Two-level 本实现三轮旧 Harness 时间范围|能，注明边界、诊断、Shared 和 n=3。|
|Re-Prefill 本轮客户端故障→响应完成|能，71821.204266 ms；n=1，仅该轮。|
|两种目标启动→Ready 的观察值|可并列描述，但启动工作与检查不同，不能据此作算法速度排名。|
|同一种局部组件在一致 span 定义下的波动|可报告本样本描述统计，不能声称独占或因果归因。|
|Two-level 比 Re-Prefill 快/慢多少|不能：缺共同客户端终点，诊断不对称，轮次不匹配。|
|去掉验证后的生产恢复时间|不能：没有直接测量；不做 inclusive 时间相减推算。|
|共享竞争到底占了多少|不能：快照不等于连续利用率，无受控独占对照。|
|Warm Recovery / 整机故障 / p99|不能：不是本实验场景，样本也不足。|

下一轮应保留所有历史字段，新增真正对称客户端事件与明确性能资格；只改字段名或移动计时起点无法补齐这些缺口。
