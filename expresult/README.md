# SGLangEC 实验资料入口

这里是可供 GitHub 浏览与后续代码审查使用的轻量证据包，不是完整运行目录，也不是新一次双节点实验。

当前资料来自 `run-20261101.Lqol9F7L`，原始目录约 1.32 GiB。这里保留文本、JSON、CSV、关键源码快照和文件哈希，不保留大体积 KV 数据、logits 或模型权重。

## 推荐阅读顺序

1. [实验报告：结果、逐轮数据和结论](run-20261101.Lqol9F7L/README.md)
2. [计时边界与不可比性说明](run-20261101.Lqol9F7L/TIMING_AND_LIMITATIONS.md)
3. [证据索引与下一步工作](run-20261101.Lqol9F7L/NEXT_STEPS.md)
4. [离线验收结果](run-20261101.Lqol9F7L/analysis/offline_audit.json)
5. [逐轮计时 CSV](run-20261101.Lqol9F7L/analysis/timing_trials.csv)、[完整阶段统计 CSV](run-20261101.Lqol9F7L/analysis/metric_statistics.csv)

**最重要的结论：功能恢复已验证，但这批数据还不能证明 Two-level 比 Re-Prefill 更快。** Two-level 的终点是恢复 Hook/诊断路径完成，Re-Prefill 有客户端流式响应终点，两者不等价。Two-level 三轮成功，Re-Prefill 一轮成功；不是五组公平交替实验。

## 上传范围和隐私

上传这个 `expresult/` 目录即可，不要同时上传原始大目录。这里没有自动提交或推送。包内保留了原始证据字节，包含内网 IP、服务器名、用户路径、进程 ID 和 Token ID，**公开上传前应自行审核**。若需要脱敏，请在单独副本中做，并更新校验清单；不要把脱敏文件仍标记为逐字节原始副本。

单个文件均较小，直接文本比压缩包更适合代码审查。实际总大小可运行 `du -sh expresult` 查看。完整性校验：

```bash
python expresult/tools/verify_evidence.py
cd expresult
sha256sum -c SHA256SUMS
```

校验只读取本地文件，不访问服务器、不启动 GPU、不证明当前远端进程状态。哈希清单是完整性检查，不是第三方签名或对实验真实性的独立背书。

## 可以交给下一位分析者的提示词

> 请先阅读 expresult/run-20261101.Lqol9F7L/README.md、TIMING_AND_LIMITATIONS.md 和 NEXT_STEPS.md，再检查 analysis/ 与 evidence/。区分原始证据、派生统计和建议。不要把功能 PASS 当成性能对比合格，不要用约 119 秒除以约 72 秒宣称加速比。Two-level 没有等价客户端 SSE 完成时间；不要用旧 inclusive 时间相减补造。仅审查 Cold 计时的最小修改方案，保持现有真实 TP=2 Shared 恢复路径、RS(2,1) 和 Native RDMA 不变，不开发 Warm，不修复 paired-ready-reprefill，不提交推送，不覆盖历史实验。给出修改计划后再安排新的对称双节点验证。

私有仓库或未提供具体文件内容时，不应假设分析工具已经取得全部资料。先提供上面三份 Markdown 和关键 JSON/CSV；需要时再补相应源码或原始证据文件。
