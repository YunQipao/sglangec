# Parent execution handoff — client date 2026-11-03

本轮只修改 design worktree 并跑离线测试。未部署、未启动服务/SSH/GPU/网络、
未发进程信号、未 commit/push，未改 live checkout。旧 runtime 的七组真实五轮
由 parent 独立检查，保持原证据身份；这里不重跑或计作新版本结果。
232 GPU0（旧 GPU2320）不在任何命令的选择内。新部署由 parent 在独立新 worktree
完成：232 使用 `1,3`，233 使用 `0,2`。健康 RS 不得重启。

## 优先 E2-A：233 本地两卡，真正 GPU restore/resume

入口 `gpu_fault_recovery.py` 导入无副作用，只有 parent 显式 CLI 才运行。
第一版只支持一个本地物理节点，TP2 / prefix256 / BF16 / 17 outputs，
三个 baseline：`reprefill`、`replication`、`local-xor`。不使用 SSH，也不触碰 RS。

每个 baseline 每轮：

1. `verify_gpu_cold_resume capture` + `submit` + `reference`，产生本轮真实 CUDA
   post-prefill / pending-token boundary；17 个完整 reference 输出和 logits 保留。
   template 只作模型/layout 准入，绝非实际恢复来源。capture 输出目录 basename
   包含独立 run ID / baseline / repeat，因此 capture run ID 和 restore ID 也不同。
2. 在相同故障边界（发起 exact-owned 原实例正常停止之前）记时；原实例及两个
   rank 必须通过现有 `require_origins_stopped`。读取本轮 seal 通过
   `load_checkpoint` 后，分配独立完整 CPU replica，或真实 XOR parity+survivor。
3. 清零并移除 adapter 拥有的 lost-rank 数组，清零并 unlink 本轮 lost-rank 文件，
   移除本轮原 global_node 和其余 rank payload 的访问路径；不会处理模板/旧实验。
   原 seal/state/meta/proofs 留作审计。拒绝 symlink、hardlink、非本用户文件、
   失效前 hash 变化。`fault-started.json` / `fault-erasure.json` 保留失败现场。
4. replica/XOR 仅从独立保护数据重建所有 rank 和完整 `global_node.bin` 到新目录。
   state、COMMITTED、rank metadata、capture proof 原 bytes 全部复制不改；再走
   原 `load_checkpoint` 全部 hash/layout/state/proof 检查。
5. 新 reference-view 只把原 reference `run.json.capture_bundle` 路径指向重建目录，
   记录原/新 run.json hash 和唯一修改字段；rank reports/logits 不改。
   原 payload 已不存在，restore launch admission、worker 和 reference checker
   的有效数据源均为重建目录。没有 `rs_source` 或原目录 fallback。
6. 新 GPU `verify_gpu_cold_resume launch`，共享原 ledger 一次性 claim；同一个
   `cold_timing.stream_request` 客户端及 `cold_sse_bridge prepare/finish` 送入请求，
   原 cold `check` 验证实际 GPU KV install/readback、mapping、16 Decode、逐 rank
   logits、17 个输出、allocator cleanup、ledger commit。Host copy 本身不能 PASS。
7. reprefill 则用原 `reprefill_server` 完整 uncached prefill + 16 Decode；同输入、
   同 17 输出、同 server argv、同 SSE endpoints。不会贴成 GPU restore。

这是**受控 KV 失效 + 原实例正常停止/新实例启动**，不是物理 GPU 损坏、整机失败，
也不是正在进行的在线请求中途抢断。故障从已封存的真实 boundary 恢复。
每个 baseline 都 fresh capture；正式五轮共 15 次 capture，不能 replay 同一 restore ID。
所有实验使用 parent 提供的同一共享 ledger，不能删除旧 claim/commit 文件。
run ID 还会在 ledger 写独占 `.experiment.json` 记录，禁止换输出父目录重用。

从 `gpu_fault_recovery_233.example.json` 复制到 parent 私有配置，填写
`repo/input_ids/template/ledger` 四个实际路径。Qwen3-8B、Python、Conda、CUDA
路径均可配置；例子来自现有 env example，parent 需按 233 实际安装核对。
`ledger` 必须由 parent 预先创建，并与其他 cold 实验共享；module 不重新初始化它。
配置不接受 SSH、auto GPU 或换端口；固定 `32962`，capture/restore 共用同一端口。

在 parent 新 233 worktree 中，最短命令如下（本轮未执行）：

```bash
export PYTHONPATH="$REPO/python:$REPO"
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
M=experiments.kv_ec.design_study.extension_v2.gpu_fault_recovery
# CFG 为已填完的配置；NEW 为新的结果父目录，不能指向旧结果。
"$PY" -m "$M" --config "$CFG" --run-id e2a233-20261103-smoke01 \
  --output "$NEW/e2a233-20261103-smoke01" --rounds 1 --max-seconds 7200
# 仅当前 smoke 三方案全部 PASS 后；本命令会核验 code/config/input hash。
"$PY" -m "$M" --config "$CFG" --run-id e2a233-20261103-five01 \
  --output "$NEW/e2a233-20261103-five01" --rounds 5 --max-seconds 36000 \
  --smoke "$NEW/e2a233-20261103-smoke01/raw.json"
```

phase timeout：startup 默认900s，request默认300s；cold rank gate120s；总预算由
`--max-seconds` 控制。到期/失败停止后续 baseline，不重试、不换端口、不复用 claim。
cleanup 另留每个 owned service 最多30s marker command +90s supervisor wait。
只通过原 Host/SELF supervisor 做 run-id/PID/start_ticks/UID/boot/session/pidfd
验证后的清理。coordinator 信号进入相同 cleanup；coordinator意外失联时由原
lease 90s 到期策略清理其拥有的 session。不能用 `kill/pkill` 或外部 timeout
强杀一组未经归属核验的进程来替代。

交付结果：顶层 `raw.json`、每轮 `result.json/fault-proof.json/timing.json`、
`capture` 的原 seal/proofs、`restored`、`reference-view`、完整 cold checker
`*-restore/check.json`、`client-sse.json`、origin/target `resources.jsonl`、
owned-process/cleanup/exit receipts，以及整个子进程树的 RSS/CPU ticks samples。
Ready / first SSE token / full17 / response-complete / validation-complete 分别报告。
first SSE token 可能就是保存的 pending token，不能称第一个新 Decode token。
`h2d_per_rank` 原 stats 包含 host tensor prepare、H2D、scatter、已有同步；不是纯链路带宽。

server flags/profile 相同，但 restore 的强制 readback/TP votes/logits 与普通 reprefill
诊断不对称，因此保持 `asymmetric_diagnostics=true`、
`algorithm_comparison_eligible=false`、`isolated_performance_eligible=false`。
报告诊断路径的实测时间，不能据此宣称生产算法的公平 RTO 排名。物理 Node RTO 为 null。

## 两项针对修正：CPU shared 与 online

CPU production 参数为 `--cpu-host-mode shared`，不是 fixture idle_check override。
必须 `CUDA_VISIBLE_DEVICES=`；只读 `/proc` 检查本项目 bench/controller/owned serving。
可核验的本项目活动或其状态读取不明会拒绝；整张进程表读不出也 fail closed。
其他用户/其他项目只记 `shared_host_activity`，无权读取 foreign environ 时为 unknown，
不会因它们用 GPU 而 block。shared 不调用 nvidia-smi；strict 可选
`--cpu-host-mode strict_idle` 保留原整机 idle 检查。所有 CPU record 均
`isolated_performance_eligible=false`。真实 provenance/layout/source 校验不变，
CLI 仍没有 fixture 开关。243 没有本项目 GPU 实验时可跑这两个真实小块命令：

```bash
CUDA_VISIBLE_DEVICES= "$PY" -m experiments.kv_ec.design_study.extension_v2.cpu_ablation \
  --nodes "$ACCEPTED_D0" "$ACCEPTED_D1" --cpu-host-mode shared \
  --rounds 1 --max-seconds 300 --output "$NEW/cpu-smoke"
CUDA_VISIBLE_DEVICES= "$PY" -m experiments.kv_ec.design_study.extension_v2.failure_sensitivity \
  --nodes "$ACCEPTED_D0" "$ACCEPTED_D1" --cpu-host-mode shared \
  --requests 20 --rounds 1 --max-seconds 300 --output "$NEW/cpu-rate-smoke"
```

online 修正与核验：launch_online 的 mode env 与所有其他 hook 开关显式设置；
launcher 清除继承的 prepared identity 并自己生成本次模型/source identity；
wrapper 传 owned manager PID，Unix SO_PEERCRED 验证 peer。attach/runtime gate
也验证 listener 全部 owner、manager argv/epoch/mode、hello，以及每次 replica
load/release 的 peer PID/lifetime。replica reply 的 rank/request/hash/length 全验。
global receipt 绑定真实 exporter 配置、request/chunk/global identity/stripe/node/mode。
rs-only manifest 先 atomically publish，再 atomic rename `global_status.json`，
receipt 最后发布；新增测试在两次 publish 中间读目标，不能看到 halfwrite。
replica 只证明完整独立复制/读取，明确 `xor_recovery_verified=false`，没有 XOR 验证宣称。

trace 默认 `requests=4` 是**每个点固定总请求数4**。C1 依次执行4个，C4 同时最多4个；
不是 C1 只发1个，也不是 true batch（max-running-requests=1，仍是 queue concurrency）。
三 prefix256/1024/4096 × C1/C4 × 五轮 × 每点4请求 = 每 mode 120条。
正常 response window 先结束，再 protection drain；分别给两个 wall window 和吞吐，
原包括 drain 的 lifecycle wall/吞吐也保留。每请求保护错误仍保留。
host CPU/memory/net/RDMA/device gauges 在 `resources.jsonl` 连续采样，整个 serving
子进程树的 CPU ticks/RSS 与各自 lifetime HWM 分开，不把 shared host计数归为单请求。
四请求4096 replication峰值稳定副本约2.25GiB，wrapper 默认本地预算调为4GiB；
这是预算而非预分配，还需 parent 确认可用 Host RAM。

```bash
O=experiments.kv_ec.design_study.online_compact
"$PY" -m "$O" plan --trace "$TRACE" --prefixes 256 --concurrencies 1 \
  --rounds 1 --output "$NEW/online-smoke-plan.json"
"$PY" -m "$O" plan --trace "$TRACE" --prefixes 256 1024 4096 --concurrencies 1 4 \
  --rounds 5 --output "$NEW/online-five-plan.json"
S=experiments.kv_ec.design_study.extension_v2.supervised_online
# GPUS：233为0,2，232为1,3；每 host 单独运行自己的同计划 NoEC reference。
"$PY" -m "$S" --mode noec --model "$MODEL" --gpus "$GPUS" --port 32961 \
  --plan "$NEW/online-smoke-plan.json" --output "$NEW/noec-smoke" --total-timeout 900
"$PY" -m "$S" --mode replication --model "$MODEL" --gpus "$GPUS" --port 32961 \
  --plan "$NEW/online-smoke-plan.json" --reference "$NEW/noec-smoke/measurements/raw.json" \
  --output "$NEW/replication-smoke" --total-timeout 900
"$PY" -m "$S" --mode noec --model "$MODEL" --gpus "$GPUS" --port 32961 \
  --plan "$NEW/online-five-plan.json" --smoke "$NEW/noec-smoke/measurements/raw.json" \
  --output "$NEW/noec-five" --total-timeout 7200
"$PY" -m "$S" --mode replication --model "$MODEL" --gpus "$GPUS" --port 32961 \
  --plan "$NEW/online-five-plan.json" --smoke "$NEW/replication-smoke/measurements/raw.json" \
  --reference "$NEW/noec-five/measurements/raw.json" --output "$NEW/replication-five" \
  --total-timeout 7200 --max-local-bytes 4294967296
# XOR 同样先自己 smoke，再正式，改 --mode xor 与相应输出/--smoke 路径。
```

正式计划必须提供自己的同 mode/code/trace/runtime smoke；旧 runtime 结果不能过此 gate。
global modes 另需 parent 审核的 RS config / requests map，原健康 RS 的容量/TTL须够用，
不允许为补点重启。既有七组不重跑；新版本 smoke 与新工作树结果单独归档。

## E4：真实4096 rank 前 1/4/16/64MiB，TCP 与 HostRDMA各五轮

`transport_microbench.py` 的 prepare/collect 纯离线，send 不启动 receiver。
receiver 必须由 parent 原 `perf_stage_e service` 监督；module验证其自身 PID/start_ticks/
UID/boot/session 与 `receiver/owned-process.json`、trial run ID 一致，之后才可 bind。
parent 提供232真实4096 NodeBlock或rank.bin+metadata；不足64MiB直接拒绝，绝不repeat/pad。

TCP 固定32963，RDMA rendezvous固定32964。每次 bind 前 PortLease + kernel listener
owner扫描，busy/unknown/读取失败即拒绝，无端口轮换，无kill。parent还需核对这两个
固定端口不在健康RS的保留 native范围，不能通过改RS配置解决冲突。
每条真实 payload 单向传送，receiver核验完整 bytes/SHA 后写receipt，collect要求两侧
SHA一致；TCP不是echo，只有1-byte ACK，并在用户态flush后发ACK，与native fflush边界对应。
RDMA必须有native CQ SUCCESS/segments/notify/byte统计，禁止自动fallback为TCP。
sender 的 `acknowledged_unidirectional_ms` 与两端各自 `transport_end_to_end_ms`、
完整 `wall_ms`、CQ/ACK/registration/setup/persist分开；TCP sendall只称Host enqueue。
不做跨主机clock相减，不用TCP echo RTT硬比单向RDMA CQ。这里只是HostRDMA，
GPUDirect=false，H2D=null；GPUadapter的H2D另报。

```bash
T=experiments.kv_ec.design_study.extension_v2.transport_microbench
# 232离线生成；若SOURCE是单rank，用 --metadata "$RANK_METADATA"。
"$PY" -m "$T" prepare --source "$SOURCE_REAL4096" --rank 0 \
  --run-id e4-20261103-smoke01 --rounds 1 --output "$NEW/e4-smoke-plan.json"
```

parent把同一plan带到233。在233已有 parent trial中，`trial-owner.json.run_id` 和
`KV_EC_PERF_RUN_ID` 必须为plan.run_id，parent持续维护lease；运行如下监督命令。
`RXTRIAL`是parent已准备的新trial根，不是健康RS根。`BIND233`为实际HostRDMA地址。

```bash
"$PY" -m experiments.kv_ec.perf_stage_e service --root "$RXTRIAL/receiver" -- \
  "$PY" -m "$T" receive --plan "$PLAN_ON_233" --host "$BIND233" \
  --output "$RXTRIAL/received" --device "$RDMA_DEVICE233" --gid-index "$GID233" \
  --timeout 120 --max-seconds 7200
# 232，仅sender：parent已启动上面的receiver并继续维护其lease。
"$PY" -m "$T" send --plan "$NEW/e4-smoke-plan.json" --host "$RDMA_HOST233" \
  --output "$NEW/e4-smoke-send" --device "$RDMA_DEVICE232" --gid-index "$GID232" \
  --timeout 120 --max-seconds 7200
# parent将received/raw.json带回232，以本地路径指定；module没有SSH能力。
"$PY" -m "$T" collect --plan "$NEW/e4-smoke-plan.json" \
  --sender "$NEW/e4-smoke-send/raw.json" --receiver "$RX_RECEIPT_LOCAL" \
  --output "$NEW/e4-smoke.json"
"$PY" -m "$T" prepare --source "$SOURCE_REAL4096" --rank 0 \
  --run-id e4-20261103-five01 --rounds 5 --smoke "$NEW/e4-smoke.json" \
  --output "$NEW/e4-five-plan.json"
# 正式换新trial/输出路径，同样receive -> send -> collect，形成40条独立transfer记录。
```

每个失败保留PARTIAL记录并停止后续transfer，避免两端序号失步；不重试一条已进入
传输的payload。sender建连可在既有transport的有界窗口内等待receiver。
243无RDMA：HostRDMA必须fail closed，不能把TCP结果重标。物理节点RTO仍BLOCKED。

## 离线 focused tests

```bash
CUDA_VISIBLE_DEVICES= "$PY" -m experiments.kv_ec.design_study.extension_v2.focused_tests \
  --output "$NEW/offline-focused-tests"
```

runner只允许真实子进程`git rev-parse HEAD`与`uname -a`读取fixture元信息；
真实bind/connect/kill/exec/GPU init均禁止，CUDA availability明确mock为false。
不选入会启动真实Unix socket服务或CUDA staging的测试。记录total/PASS/skip/failure/error
各自计数，绝不把skip当PASS。日志和counts.json是CPU synthetic protocol测试证据，
不能当GPU E2-A/E4硬件完成证据。

本轮最终 `artifacts/focused-20261103-r3/counts.json`：**total115，PASS115，skip0，
failure0，error0**；`git diff --check`通过。r1日志保留了过严护栏误拦只读Git元信息的
17个setup失败，r2修正护栏后112项通过；r3再加入归属/flush等回归后115项通过。
各轮独立保留，不能相加冒充额外通过数，也没有把未选入的真实socket/CUDA测试计为PASS。
