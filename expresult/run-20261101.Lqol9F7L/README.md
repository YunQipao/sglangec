# TP=2 Shared Cold Recovery：实验整理报告

## 1. 一句话概括

这批实验说明：**Two-level 可以在指定的双节点、共享 GPU、受控数据丢失场景中，把恢复出的 KV 写进新实例的真实 GPU Cache，恢复调度状态，跳过已恢复 Prefix，继续 Decode，并得到正确结果。Re-Prefill 也完成了一次正确的冷恢复基线。**

但它还没有回答“Two-level 到底比 Re-Prefill 快多少”：两条路径的诊断强度、完成终点和观测方式不对称，不能直接拿现有总时间做算法速度排名。

本报告只分析已有文件。本次整理没有启动新的模型、GPU 或 RDMA 测试，也没有更改实验代码、原始结果、网络或其他用户进程。

## 2. 来源、环境和版本

原始目录：`/media/sda1/iic/luotao/kv-ec-perf-tp2-shared/run-20261101.Lqol9F7L`。

原始文件清单共 928 个、1,415,361,981 bytes（约 1.32 GiB）；包括失败尝试、成功结果、KV、logits、进程采样和日志，不能把整目录全部视为成功实验。

|项目|本目录记录的实际配置|
|---|---|
|模型|Qwen3-8B|
|并行|每个实例 TP=2|
|输入|Prefix=256；保留实际 Input IDs 与哈希|
|输出|17 个输出 Token；恢复后 16 次 Decode Forward|
|保护|节点内 XOR；跨节点 RS(2,1)；two-level|
|资源准入|Shared GPU；mem_fraction_static=0.6|
|Node0 / 232|GPU **1,2**；mlx5_0，Port 1，GID 5|
|Node1 / 233|GPU 2,3；mlx5_0，Port 1，GID 3|
|RS 服务|192.168.2.245:30701|
|记录的 HEAD|`7dbe9ff5c73d919293c8798227b56aaa8a997fd4`|
|记录的 tracked diff SHA256|`7c7953193de437a089e1ddca4bc24333b9ef25b0bb195a5eea87b6a0f27786b0`|

**不要套用此前 Known-good 的 GPU 0,2 或 Commit 9cd1ea1…描述本目录。** 以[本次配置](evidence/config.json)和[成功批次元数据](evidence/two-level-stable/metadata.json)为准。Two-level 和 Re-Prefill 的成功批次记录了相同源码身份；逐轮两节点 inventory 也可在证据目录核对。

[源码身份清单](analysis/source_snapshot_identity.json)列出 22 个选定文件。仅在当前文件哈希与历史元数据相等时才复制到 `source_snapshot/`；本次选定文件全部匹配。它不是完整源码/模型/环境快照，也没有把今天的完整 Git diff 冒充历史补丁。

## 3. 所有批次结果：不隐藏失败

|批次|结果|解释|
|---|---|---|
|two-level-smoke|1 次尝试失败|origin 退出；出现 FOREIGN_OR_UNVERIFIABLE_LISTENER，涉及 30985 端口；cleanup 也未通过。不能据此认定某个其他用户进程有问题。|
|two-level-fixed|Preflight 拒绝；3 次未开始|SOURCE_IDENTITY_MISMATCH：tracked_diff_sha256 不一致。不是 3 次恢复失败。|
|two-level-v2|首轮失败，余下 2 次未开始|Stage E 子检查通过，但后续 paired-ready-reprefill 命令失败；cleanup 未通过。整轮不接受。|
|two-level-final|首轮失败，余下 2 次未开始|Stage E 子检查通过，但源码 diff 在批次准入后变化；整轮拒绝。cleanup 通过。|
|two-level-stable|**3/3 accepted PASS**|三轮完整功能验收与 cleanup 均通过。|
|cold-reprefill|**1/1 accepted PASS**|无缓存重新 Prefill，输出 Token 一致，cleanup 通过。|
|xor-rank-0 / xor-rank-1|各 10 次 byte-exact|同一真实 Capture 上的 CPU codec 微基准；不是额外 20 次双节点 GPU 恢复。|

依据：[逐轮目录](analysis/trial_catalog.json)、[Preflight 索引](analysis/preflight_catalog.json)、[失败摘录](analysis/failures)、各批次 `repeat-summary.json`。失败摘录保存原日志路径、完整日志 SHA256、原始行号和有限上下文；并非完整日志。

历史测试记录：[tests.log](evidence/tests.log)初次为 2 failed、93 passed、94 subtests passed；[tests-fixed.log](evidence/tests-fixed.log)修正运行环境后为 93 passed、96 subtests passed，8.29 秒。不能把它改写为“174 项回归全部通过”，也不是本次整理重新跑的测试。

## 4. 三轮 Two-level 究竟证明了什么

三轮都检查到：

- 真实两节点 Native RDMA：每轮保留 8 段成功 CQ 证据（2 段上传、4 段恢复往返、2 段交付），transport=rdma，completion_status=SUCCESS。
- 使用新实例，两个 TP Rank 的 GPU KV 写回均完成，byte_exact=true。
- 两个 Rank 的 Scheduler 状态与 Token 检查通过，全局恢复 Commit 完成。
- resume_decode_forward_count=16，recomputed_prefix_tokens=0，output_token_ids_match=true。
- no_slot_leak、no_request_row_leak 通过。
- verdict、目标启动证据与 protection 的 SHA256 与 acceptance 对应；两节点和批次源码身份对应。

通俗说：不是只在 CPU 内存里把字节拼回来，也不是“有一个 RS 解码函数跑通”。它检查了“拼回数据 → 装进真正的目标 GPU → 恢复请求 → 接着算 → 输出正确”这条链路。

边界也必须明确：故障模型是 `controlled Node Block loss + same-host new-instance cold resume`，whole_node_failover=false。**不是物理服务器整机宕机后跨机器接管的验证**，不包含真实故障检测时间；也不是 Warm Recovery。

证据：[离线核验](analysis/offline_audit.json)、下方每轮原始检查文件。

|轮次|Trial ID / 原始检查|故障边界→旧完成点 s|故障边界→Ready s|启动→Ready s|Ready→旧完成点 s|
|---|---|---:|---:|---:|---:|
|T0|[trial-000-e6a3c9bfc3724da08cf2ff48c064aed0](evidence/two-level-stable/trial-000-e6a3c9bfc3724da08cf2ff48c064aed0/stage-e-check.json)|120.149|110.356|82.221|9.793|
|T1|[trial-001-f21439fa06e04d3a91ff757ed630eca6](evidence/two-level-stable/trial-001-f21439fa06e04d3a91ff757ed630eca6/stage-e-check.json)|119.225|109.384|81.751|9.841|
|T2|[trial-002-067d8652c8b048109bfa90cb79f151ed](evidence/two-level-stable/trial-002-067d8652c8b048109bfa90cb79f151ed/stage-e-check.json)|117.656|107.718|79.631|9.938|

旧完成点不是客户端响应完成点；详情见[计时说明](TIMING_AND_LIMITATIONS.md)。同一轮 Ready→完成与启动→Ready 是总区间内的阶段，不要再次加到总数上。

三轮旧 `failure_to_decode_complete_harness` 为 **119.010 ± 1.260 s**，`new_instance_launch_to_ready_harness` 为 **81.201 ± 1.380 s**（均值 ± 样本标准差，n=3）。这是当前 Shared 环境、含诊断的 Harness 观测，不是生产 RTO。3 轮不能可靠估计尾延迟或长期稳定性。

每轮完整命令耗时约 291–295 秒，还包括 Capture、保护、参考输出及其他实验控制工作，不能当成故障恢复延迟。

## 5. Re-Prefill 的一轮结果

Trial：`trial-000-41869374fcfd40bfad4e4abe565ce38e`。

它启动新的普通目标实例，再提交相同 Prefix；响应记录 cached_tokens=0、prompt_tokens=256、completion_tokens=17，输出 ID 与参考一致。也就是说，确实重新计算了输入 Prefix，而不是命中了既有 Prefix Cache。

|观测区间|时间|
|---|---:|
|目标启动→Ready|67.114 s|
|故障边界→首次观测到输出 Token|71.215 s|
|故障边界→17 个输出齐全|71.821 s|
|故障边界→客户端流关闭|71.821 s（精确 71821.204266 ms）|
|故障边界→验证完成|71.900 s|
|请求开始→首次输出|2.585 s|
|请求开始→完整输出|3.190 s|
|请求开始→客户端流关闭|3.190 s|

这里只有 n=1，不能报告跨轮方差，也不能和 Two-level 的旧完成点计算公平加速比。

原始证据：[Re-Prefill verdict](evidence/cold-reprefill/trial-000-41869374fcfd40bfad4e4abe565ce38e/reprefill.json)、[客户端请求/响应](evidence/cold-reprefill/trial-000-41869374fcfd40bfad4e4abe565ce38e/reprefill-inference/request-000.json)、[精确计时及事件](evidence/cold-reprefill/trial-000-41869374fcfd40bfad4e4abe565ce38e/cold-comparison-timing.json)。完整命令耗时 229.809 s，也不是恢复时间。

## 6. 组件拆解：真正的计算通常比冷启动短很多

下表来自成功 Two-level 三轮；单位 ms，均值 ± 样本标准差。它们不是可相加的成本清单。

|阶段|观测值|正确理解|
|---|---:|---|
|RS 解码|78.276 ± 1.375|RS 重建计算；不是 RS RPC 全流程，更不是总恢复。|
|节点内 XOR 编码：Node0|26.123 ± 0.324|真实节点块的编码 span。|
|节点内 XOR 编码：Node1|11.106 ± 0.516|处理的数据布局/作用域不同，不宜简单归因于机器速度。|
|RS 恢复 RPC inclusive|1958.558 ± 8.983|含 RPC、传输/等待等，不是裸网络或纯 RS。|
|恢复 adapter inclusive|9429.772 ± 10.219|较宽流程，包含内部子阶段；不能再与其中 RPC 相加。|
|GPU KV 写回 Rank0|67.137 ± 8.870|目标 GPU 写回 span，非整个恢复。|
|GPU KV 写回 Rank1|98.457 ± 37.453|范围 75.383–141.671；波动不等于已证明 Shared GPU 是唯一原因。|
|Scheduler Request 重建 Rank0|3.165 ± 0.073|不等于所有调度器协调/全局 Commit 成本。|
|Scheduler Request 重建 Rank1|2.772 ± 0.068|同上。|
|首次恢复 Decode CUDA Rank0|2172.099 ± 64.969|首个 Forward 的 stream CUDA-event 区间，不是稳态单 Token 延迟。|
|首次恢复 Decode CUDA Rank1|2172.131 ± 65.000|TP 同步/等待影响，不能把两 Rank 相加。|
|Capture artifact：Node0 origin|402.982 ± 13.253|每轮同角色 span 累计，涉及 2 次观测；不是纯 GPU Capture。|
|Capture artifact：Node1 origin|384.038 ± 5.360|同上；保护准备工作通常在受控故障边界之前。|

[完整统计](analysis/metric_statistics.csv)含上传、交付、内存注册等更多细分；[所有事件](analysis/perf_events.csv)保留 host/PID/rank 和起止值等。原始 JSONL 保留原样，包括部分记录缺失的身份字段。不要假设所有历史事件都有 operation_id、span_id、boot_id 或可跨机器直接相减的时间戳。

特别注意 `native-metrics.json`：采集器已将 native 的 `elapsed_s` **数值乘以 1000 转为 ms，却保留原字段后缀**。本包的 metric_statistics 里这些数值按 ms 保留，不再乘 1000；原始 CQ 内的 `elapsed_s` 仍是秒。需要核对[历史匹配源码](source_snapshot/experiments/kv_ec/perf_stage_e.py.txt)中的 `native_metrics()`。

### Checkpoint 不是只加载一次

[逐次调用 CSV](analysis/checkpoint_calls.csv)列出每个 `checkpoint_load_validate_inclusive`，并按每轮/host/role 给出[调用统计](analysis/checkpoint_summary.json)。例如 T0 的 Node1 通用角色 10 次累计约 14.864 s，Node1 target 角色 5 次累计约 11.130 s。应报告次数、单次 min/max/mean 和累计，而不是只取一个最小值说“checkpoint 加载只需这么多”。

这些累计跨调用/进程，含读文件、格式/哈希/状态验证、等待等实际函数内成本。可能嵌套在较宽 span 内，不能当作关键路径总额；**不能把它从旧总时间扣除，得到所谓生产恢复延迟。**

### XOR 微基准

各 rank 丢失场景使用同一真实 GPU Capture 数据做 10 次 CPU codec 测量，均 byte-exact。Rank0 平均 encode=8.263 ms、recovery=8.732 ms、validation=9.777 ms；Rank1 为 8.512、8.529、9.740 ms。详见[xor_statistics.json](analysis/xor_statistics.json)。

它说明这份数据上的 codec 正确且计算量较小，**不证明 GPU 回写、Scheduler、恢复 Decode 或新的端到端轮次成功**，也不能冒充 Stage E 的 xor-only 对照方案。

## 7. Shared GPU 的解释边界

[启动资源快照](analysis/gpu_startup_context.json)记录 Node0 GPU1/GPU2 上已有其他任务，约占用 2.3/3.0 GB 显存；Node1 GPU2/GPU3 的快照约 477 MB 已用，未列出同类其他任务。以原始启动 JSON 的 bytes、UUID 和 other_tasks 为准。

它证明 Shared 准入允许其他进程存在，而不是独占环境。显存占用不等于算力占用，启动时刻快照也不是整个轮次的连续利用率；不能仅凭这些快照认定某一次慢请求是谁造成的。原始高频 resources JSONL 未打包，其路径、大小和哈希在 SOURCE_INVENTORY.csv，可有针对性补取。

## 8. 可以写进报告/论文的结论

1. 指定受控故障模型、TP=2 和 Shared 配置下，Two-level 连续三轮通过真实 GPU/Scheduler/Token 端到端正确性验收；Re-Prefill 一轮通过。
2. 在该小样本中，RS 计算约 78 ms、GPU KV 写回约数十到百余 ms，而 Cold 启动到 Ready 约 80 s；这提示冷启动、验证和编排开销值得优先分析。不能据此称已消除这些成本。
3. 当前 Harness 的阶段耗时可作为“带诊断的实验实现开销”报告，必须注明边界、包含项、Shared 环境与 n。
4. 目前没有合格的 Two-level/Re-Prefill 客户端对照结果，因此不报告算法加速比、生产 RTO、Warm 性能、整机容错覆盖或 p99。

3/3 成功表示三次观测成功，不是对任意故障、规模、负载的数学保证。存在失败尝试也不能隐藏；源码/端口准入拒绝与恢复算法错误应分开归类。

## 9. 证据包能做和不能做什么

- `evidence/`：按原相对目录保存选定原始文件，未更改其字节；[复制清单](COPIED_MANIFEST.json)记录原路径/大小/SHA256。
- `analysis/`：本次导出的派生 CSV/JSON，不伪装成原始程序输出。
- `source_snapshot/`：22 个历史哈希匹配源码，便于审查计时和验收逻辑；文件名追加 `.txt`，内容字节不变，避免被仓库测试自动发现。不要直接当作可运行完整项目。
- [SOURCE_INVENTORY.csv](SOURCE_INVENTORY.csv)：928 个原始文件的完整路径、大小、SHA256、保留/省略原因；包含未打包的大文件。
- [EXPORT_INFO.json](EXPORT_INFO.json)：来源、导出时间、文件数、限制和离线核验结论。

未包含 KV `.bin`、logits `.npy`、重复 zip、全量调试日志和资源采样、模型/环境。保留的 verdict 和摘要能重新检查已存验收的一致性，但无法在本包中重新从原始 KV 独立做 GPU 字节回读或 logits 验证；需要完整原目录和真实运行环境。

下一步请阅读[计时边界](TIMING_AND_LIMITATIONS.md)和[后续动作](NEXT_STEPS.md)。
