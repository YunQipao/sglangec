# EC E1–E6 extension v2: parent review and execution

2026-11-03 targeted update（客户端日期）：优先使用
`PARENT_RUNBOOK_20261103.md`。新增 `gpu_fault_recovery.py` 真GPU restore/resume
parent adapter、`transport_microbench.py` 单向真实KV TCP/HostRDMA adapter，
CPU shared production gate、online smoke gate及分离response/drain窗口。
本轮没有执行硬件实验；旧runtime七组五轮保留原身份，不重跑。

这次新增代码都在本目录，唯一按要求放在上级的新入口为
`design_study/online_compact.py`。现有 manager/exporter/launcher/capture 只做局部
扩展；所有原有改动、旧实验与历史结果保留，未 commit/push。
本次没有启动 GPU/SSH/网络服务，没有停止任何已有进程，也没有实测新的 GPU
baseline、真实 CPU capture 消融、物理节点故障或 Node RTO。

## 先审核心 online modes

`online_chunk_manager.py` 的 `protection_mode` 默认 XOR。旧的 XOR + exporter
调用仍工作，但 manifest 如实标为 `xor-rs`；launcher 未显式指定 mode 时继承
manager 的真实 mode，显式指定则必须匹配。每个新 rank 的 mode 也会校验。

| mode | 实际稳定保护对象 | COMMITTED 门槛 |
|---|---|---|
| replication | 独立、完整、可校验读取的每 rank CPU 数组；默认不写额外 rank dump | 全 rank 通过原 layout/hash/request/epoch 校验并复制，manifest提交；完全不调用 XOR |
| rs-only | global exporter，无本地 XOR/parity | 本地仅 LOCAL_SNAPSHOT；绑定的远端 GLOBAL_COMMITTED 后才 COMMITTED |
| xor | 实际持有本地 Host DRAM parity，另存 parity.bin | 本地 parity 提交 |
| xor-rs | 本地 XOR + 原后台 exporter | 本地 COMMITTED 与远端 GLOBAL_COMMITTED 分开；runner 两者均验 |

`retain_ranks` 仍只是 debug，不再当 replication 的实现。replica 从 manager 的
真实 DRAM 数组读取并重新检查整块/逐层 K/V/hash；不从 debug 原文件偷读。
`load_replica` 协议返回真实 bytes。manifest 分开记录 stable local parity、stable
replica、remote parity、source snapshot、临时 rank copies、persisted node snapshot、
capture/encode/copy/local commit/global commit；未知 RSS 保持 null。
remote parity 是每 stripe 的远端对象，同一 stripe 的两个源端 receipt 不能相加。
时间为各自阶段，可能重叠，不能相加冒充端到端。

实际本地 redundancy DRAM 有预算（`--max-replica-bytes`，默认2GiB，适用于 replica
或 parity）；不会静默驱逐。`release_protection` 是显式生命周期结束，会标记
RELEASED，之后不再假装有 live DRAM 保护。compact runner 默认在每 chunk
提交验证/replica 读取验证后释放本地 DRAM；保留文件证据和提交时 ledger。
这是有限在线开销实验，不是无限保留策略；pending/capacity/超时失败均保留。

## 测试（CPU only）

```bash
export PYTHONPATH=python:.
export CUDA_VISIBLE_DEVICES=
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
$PY -m unittest \
  experiments.kv_ec.design_study.extension_v2.test_modes \
  experiments.kv_ec.design_study.extension_v2.test_pipeline \
  experiments.kv_ec.test_online_chunk experiments.kv_ec.test_online_rs \
  experiments.kv_ec.test_launch_online -v
```

新测试使用明确标记的 synthetic fixtures 校验真实复制/编码/擦除/重建代码，
mock 的是传输和外部服务，不是 XOR/RS/bytes；它们不是新的性能或 GPU 实验证据。

## E1 / E3 / 并发 E6：一个服务生命周期跑多个点

本地结果目录存在，但 `../kv-ec-next-results-20261101/scripts/online_benchmark.py`
并不存在；本实现没有猜测或复制另一份 driver。`online_compact.py` 复用已有
`workload.measure` 和固定 trace 校验。parent 后续如提供原 driver，可对接新
plan/runtime/protection 验证函数，不必重新实现 SSE。

先用已有真实 tokenizer/model fingerprint 生成或提供一份固定4096-input trace，
固定4个请求即可覆盖C1/C4，输出17。每个点固定总请求数4，C1串行发4个、C4最多并发4个；
不是C1只发1个。所有方法使用同一 plan（包括相同 prefix、请求 ID、并发、
轮次和精度），TP2、BF16、shared。不要把 model SHA 与 tokenizer SHA 混用：
当前 runtime 准入检查的是 `debug_capture.source_identity` 的 model_sha256；
trace 中的 ID 词表和 tokenizer 来源需要由 parent 的 trace 证据另行保留。

以下命令仅供 parent 审阅后执行；本 agent 没有运行 launch/run。

```bash
STUDY=experiments/kv_ec/design_study
# TRACE 是 parent 确认的现有4096-input/17-output trace JSON。
$PY -m experiments.kv_ec.design_study.online_compact plan \
  --trace "$TRACE" --prefixes 256 --concurrencies 1 --rounds 1 --output "$NEW/smoke-plan.json"
# 正式：prefix256/1024/4096、并发1/4，每个点5轮；每种方法只启动一次。
$PY -m experiments.kv_ec.design_study.online_compact plan \
  --trace "$TRACE" --prefixes 256 1024 4096 --concurrencies 1 4 --rounds 5 --output "$NEW/formal-plan.json"
```

plan 同时写 `formal-plan.rs-map.json`，只打印路径/计数，不把整份 trace/map 打到
CLI 输出。两端 exporter 的 requests 都使用这份映射，global run/checkpoint
identity 一致；local run_id 由每个服务唯一生成。

推荐 parent 用新 wrapper，它复用现有 `perf_stage_e service` 的确切
PID/start_ticks/boot_id/UID/session/run-id/pidfd cleanup、PortLease 固定端口预检、
lease heartbeat 和 readiness。CPU manager 与 GPU 服务均只清理本 wrapper
新启动的进程；不会启动、停止、重启任何 RS 服务。不要自己用 kill/pkill 清理。

```bash
# MODEL 和 GPUS 是 parent 经过资源审核的真实路径与物理 GPU 索引；不自动挑空闲卡。
$PY -m experiments.kv_ec.design_study.extension_v2.supervised_online \
  --mode noec --model "$MODEL" --gpus "$GPUS" --port 32961 \
  --plan "$NEW/smoke-plan.json" --output "$NEW/noec-smoke"
$PY -m experiments.kv_ec.design_study.extension_v2.supervised_online \
  --mode replication --model "$MODEL" --gpus "$GPUS" --port 32961 \
  --plan "$NEW/smoke-plan.json" --reference "$NEW/noec-smoke/measurements/raw.json" \
  --output "$NEW/replica-smoke"
# XOR 同上改 --mode xor。正式 plan 必须先生成自己的完整 NoEC reference。
# rs-only/xor-rs 还需 --rs-config parent-verified-config.json；两源端同时跑同一 plan。
```

或者 attach parent 已监督的健康在线服务：

```bash
$PY -m experiments.kv_ec.design_study.online_compact run --service service.json \
  --plan "$NEW/formal-plan.json" --reference "$NEW/noec-formal/measurements/raw.json" \
  --smoke "$NEW/this-mode-smoke/measurements/raw.json" \
  --output "$NEW/attached-measurements" --total-timeout 7200
```

service.json 必需字段：`mode`、真实 `pid`、本机 `url`。非 NoEC 还必须有真实
`manager_pid`、`manager_output`、`manager_socket`。可选
`release_after_validation=false` 会保留 DRAM，需 parent 预留整个 plan 的内存。
wrapper 会自动生成此文件。attach 不启动、不停止进程；若外部 supervisor 有 lease，
由原 coordinator 继续维护，不能只留下一个服务等本 runner 接管其租约。

runner 读取实际 /proc argv/env 和 listener owner，验证真实 mode/runtime flag、
TP2/BF16/256-prefill/shared/model identity，检查每个 request 的每个256 chunk
的请求哈希、token-ID 哈希、epoch、范围、mode与提交状态，并实际读取验证 replica。
每个保护模式输出必须完全匹配同 plan 的 NoEC token IDs；错误与未尝试请求都保留。
真实 GPU device/process evidence、HostRDMA 全局计数、每 chunk native verbs 统计、
Host RSS samples/HWM 都独立保存；HostRDMA 计数可含他人流量，不称 GPUDirect。
当前 `max-running-requests=1`，并发1/4/8是 **queue concurrency，不是 true batch**。
资源共享性能只能称 shared 环境结果；不能声称没有其他进程的干扰。
客户端 TTFT/响应完成不冒充服务端独立 prefill/decode 时间，后两者为 null。

RS 配置必须由 parent 按 plan 扩容 `max_checkpoints`、`max_pending`、`max_bytes`，
检查远端已有服务 TTL/容量与所需 stripe 数。默认16个 checkpoint不够完整 plan，
不允许为此重启健康 RS。wrapper 不扩容远端服务、不改其参数。
既有 online exporter 仍是 HostRDMA；物理243没有 RDMA，这份 wrapper 不把它
改贴成 RDMA 三域实验。P若仍同233，就没有整机容错证明；三域243仅TCP另用已有
`design_study.remote/three_host` 路径。wrapper port 还需 parent 核对 RS native端口范围。

## E5：真实 KV CPU 分片五方案

显式 `--cpu-host-mode shared` 允许其他项目/用户活动，但拒绝可核验本项目活动；
foreign环境无权限时只记unknown。必须`CUDA_VISIBLE_DEVICES=`；所有CPU结果
isolated_performance_eligible=false。`strict_idle`可选保留整机idle检查。
真实来源校验不变，没有fixture/idle-bypass CLI。
需要 parent 提供已 accepted 的256/1024/4096各两份完整、独立 D0/D1 NodeBlock。
不假设这些路径存在，不从一个 rank 平铺伪造多个节点；输入通过现有 descriptor、
跨节点模型/layout校验并保存完整 source hashes。

```bash
$PY -m experiments.kv_ec.design_study.extension_v2.cpu_ablation \
  --nodes "$ACCEPTED_D0" "$ACCEPTED_D1" --cpu-host-mode shared --rounds 1 --max-seconds 300 --output "$NEW/cpu-smoke"
$PY -m experiments.kv_ec.design_study.extension_v2.cpu_ablation \
  --nodes "$ACCEPTED_D0" "$ACCEPTED_D1" --cpu-host-mode shared --rounds 5 --max-seconds 1800 --output "$NEW/cpu-prefix256-five"
# 对另两组实际 prefix capture 分别运行，新目录；不生成笛卡尔积。
```

五方案 NoEC、CPU replication、local XOR、RS-only、XOR+RS；每个rank失效、每个
nodeKV不可用各5轮。实际分配完整 CPU copies，实际把目标 active bytes 清零后移除
引用，恢复方法拿不到丢失原 payload，使用 prefault hash 验证。nodeKV 不可用在
adapter 中同时撤去该节点的本地副本/local parity；RS 只读取另一个逻辑域的存活
数据和全局 parity。这是同一CPU进程里的访问隔离，不是 GPU、进程或物理节点故障。
NoEC/replica/XOR 的预期不可恢复保持 `recovery_success=false`，只有验收该预期
的 protocol check 可成功；不计成恢复成功。每轮 ledger 和实际 RSS samples 与
lifetime HWM分开；RSS采样可能漏短峰，HWM不是单轮增量，也不是GPU显存。

## E6：实际请求单元故障与 GPU blocker

```bash
$PY -m experiments.kv_ec.design_study.extension_v2.failure_sensitivity \
  --nodes "$ACCEPTED_D0" "$ACCEPTED_D1" --cpu-host-mode shared --requests 100 --rounds 5 \
  --max-seconds 1800 --output "$NEW/cpu-request-fault-five"
```

5/10/15% 是独立 Bernoulli **CPU checkpoint 请求单元**概率，同seed用于五方案；
被选中的每个单元都实际执行清零/丢弃/重建，不是 schedule 数量统计。每请求单元
复用相同的真实 capture pair，不能叫成新的在线 inference capture；失败、超时、
实际 fault count、成功率、条件 MTTR 保留，全事件 MTTR在存在未恢复时为null。
请求概率实验尚未绑定正在进行的live GPU request，故真实GPU故障率serving实验仍
**BLOCKED**。新增E2-A `gpu_fault_recovery.py`只覆盖fresh sealed boundary下的
受控KV失效+新GPU实例restore/resume，不把它叫成请求中途故障率实验。
硬件执行尚待parent，EITR/物理Node RTO不编造。

最小 serving adapter 是 `failure_sensitivity.cold_reprefill(url, checkpoint, row,
selected=..., expected_output=..., timeout=120)`：复用现有 cold `load_checkpoint`
seal/state/proof验证，要求 exact input，实际清零独立Host KV copy，然后发真实
HTTP完整重新prefill并比对NoEC输出。它不是空callback，不返回假GPU erasure证明。
需 parent 先用 `online_compact.verify_runtime` 验证监督的 NoEC服务；此 adapter
只证明 **Host checkpoint loss → ordinary cold re-prefill**，不声称已写回GPU KV、
也不把历史冷恢复smoke当成本次请求级GPU故障实验。物理节点关机未授权，不执行。

## E1–E6 collector：有多少证据就收多少，保留 PARTIAL

```bash
$PY -m experiments.kv_ec.design_study.extension_v2.collector \
  --index "$STUDY/extension_v2/results-20261102/history_index.json" \
  --output "$NEW/collected-report"
```

输出 `report.json/report.md/records.json/records.csv/coverage.svg`，无需安装包。
随附index仅引用本仓库确实可读且有manifest hash的旧128KiB CPU记录，不是 parent
报告的180条1/4/16/64MiB，也不是新的GPU结果。parent拿到其他有效历史records后，
在新的index中追加：`path/sha256/experiment/evidence_class/config`。可选
`manifest/manifest_sha256`会验证manifest.raw_sha256；code hashes与原始source都保留。
已有不同JSON格式可显式指定 `records_key`、`success_key`、`metrics`（dotted path）；
未知格式/测量边界应先parent审阅映射，不能把codec elapsed映射成Serving RTO。
不默认成功，不丢失败。historical_smoke、historical_five_round、current_smoke、
current_five_round、synthetic_fixture分别分组；五个smoke不会合并成正式五轮。
未跑前缀不阻止汇总，未知值null，E1–E6全局始终保守PARTIAL。
