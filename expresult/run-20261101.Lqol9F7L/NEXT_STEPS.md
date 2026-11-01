# 证据导航与下一步动作

## 1. 按问题找到证据

|要核对的问题|优先文件|
|---|---|
|最终批次是否全通过|`evidence/two-level-stable/repeat-summary.json`、`evidence/cold-reprefill/repeat-summary.json`|
|每轮是否正式接受|各 trial 的 `trial.json`、`acceptance.json`（Two-level）、`cleanup.json`|
|两个 Rank 的真实恢复|各 Two-level trial 的 `stage-e-check.json`、`target-run.json`|
|真实 RDMA 而非 TCP payload|`cq-evidence.json`、`export0/export.json`、`export1.json`、`delivery-send.json`、`delivery-receive.json`|
|RS 恢复与保护对应|`protection.json`、`recovered-source/protected/recovery.json`|
|目标 GPU 与 Shared 占用|各启动 `run.json`、`target-run.json` / `reprefill-target-run.json`；派生 `analysis/gpu_startup_context.json`|
|源码版本、两台机器是否同源|成功批次 `metadata.json`，各 trial `inventory-232.json` / `inventory-233.json`|
|执行过哪些控制命令|各成功 trial `commands.jsonl`；是历史审计，不是现在可直接安全重跑的脚本。|
|为什么暂时不能比较|Two-level `cold-comparison-timing-status.json` 与 Re-Prefill `cold-comparison-timing.json`|
|精确逐次加载成本|`analysis/checkpoint_calls.csv`、`analysis/checkpoint_summary.json`、原始 `perf*/*.jsonl`|
|全部阶段统计|`analysis/metric_statistics.json` / `.csv`、`analysis/perf_events.csv`|
|被省略的原始大文件在哪|`SOURCE_INVENTORY.csv`，含完整相对路径、大小、SHA256|

注意：`analysis/trial_catalog.json` 的 evidence_relative 是相对原始运行目录的路径；打包副本在本目录 `evidence/` 下。所有结果目录并非都携带完整数据；对未复制的大文件按 SOURCE_INVENTORY 定点补取。

## 2. 下一步最小实验实施建议（尚未实施）

1. 在当前源码身份、diff 和两节点哈希留档后再改代码，沿用受保护的成功恢复路径。不把本包源码快照直接覆盖到服务器。
2. 先设计“相同故障边界、同一个客户端请求/响应语义”：两条真实路径都要产生可验证的 first/full/response-complete 事件。Two-level 必须继续使用恢复后的真实 GPU/Scheduler 状态，不能额外发送一个重新 Prefill 的普通请求充当恢复响应。
3. 保留旧 Harness 指标及其语义；新事件必须直接测量，不由旧 inclusive 时间相减。记录 operation_id、span_id、host、boot_id、PID、rank、clock_domain；缺失事件则性能资格不通过。
4. 诊断与正确性检查需单独标明。保持强检查默认不变，尽量对称观测；不能仅删除 Two-level 校验后与另一条完整路径比较。最终仍将功能资格、测量完整性、算法对比资格、独占性能资格分开。
5. 保留 Checkpoint 调用次数/逐次/累计；可选首轮和后续 Forward 计时默认不添加额外 CUDA 同步。核实新字段有单测、旧格式 Collector 兼容，运行实际存在的回归集；不要沿用未核实的“174 项”数字。
6. 先做不占 GPU 的单机单测及历史证据验收，再报告必须同步到 233 的文件和 SHA256，由用户确认安全同步；新实验始终使用全新目录。
7. 两种模式各运行一轮真实 Smoke，检查所有功能约束与新计时资格。只有两个 Smoke 都实际 PASS，才安排交替五轮；任何未执行验证一律标记 NOT RUN。

本次未实现上述代码修正，未开发 Warm，未处理 `--paired-ready-reprefill` Hook 重复请求，也未声称新增 Smoke PASS。不在此盲目生成会占用端口/GPU的执行命令；先让下一位审查者对照仓库当前 CLI 与证据确认入口。

## 3. 固定约束，交接时请一并保留

- 不改 RS(2,1) 语义、RDMA GID/驱动/CUDA/Conda/网络参数，不允许静默 TCP payload 回退。
- 不停止其他用户进程，不覆写历史成功数据，不自动清理、Git commit 或 push。
- 维持两个 Rank GPU byte-exact、Scheduler/global Commit、16 次恢复后 Decode、零 Prefix 重算及输出 ID 一致；Re-Prefill 保持 cached=0、256 输入、17 输出。
- 下一轮以本次实际配置为基准：Node0 GPU1,2，Node1 GPU2,3。若用户选择回到旧 GPU0,2，应作为显式配置变更记录，不能混在同组结果。
- 当前历史故障模型不是整机宕机；不要扩大为整机故障结论。

## 4. 离线校验与可复现整理

从项目根目录：

```bash
python expresult/tools/verify_evidence.py
```

从任意目录独立校验文件哈希：

```bash
cd /media/sda1/iic/luotao/sglangec/expresult
sha256sum -c SHA256SUMS
```

可重新导出到一个**不存在的新目录**（原目录只读；导出程序拒绝覆盖已有目的地）：

```bash
python expresult/tools/build_evidence.py \
  --source /media/sda1/iic/luotao/kv-ec-perf-tp2-shared/run-20261101.Lqol9F7L \
  --destination /tmp/sglangec-evidence-reexport-20261101 \
  --repo /media/sda1/iic/luotao/sglangec
```

导出程序生成原始副本、分析数据、文件清单与选定源码，不会自动重写这三份人工解释文档或根目录 SHA256SUMS。源码快照只复制仍匹配历史哈希的文件；后续代码变动后重新导出可能跳过部分源码，须检查身份清单。

这里的离线核验重复检查原存证据和哈希，并不是独立复现 GPU 实验。若原始 GPU 数据/日志被删除，单靠哈希不能还原内容。请保留完整原目录作为底稿，本包只用于轻量交流。
