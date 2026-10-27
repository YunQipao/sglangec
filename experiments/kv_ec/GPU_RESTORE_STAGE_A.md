# 阶段 A：真实 Scheduler GPU KV 写回（TP=2）

本实验只验证在线 Checkpoint → 新 GPU Slots → 逐层读回 → 请求映射替换 →
正常回收。**不实现故障后请求接续，也不以 HTTP 成功证明恢复成功。** 默认关闭。

## 本轮实际状态（2026-10-27）

- 分支：`feat/kv-gpu-restore-resume-20261027`；无 commit / push。
- 初始 `python -m unittest experiments.kv_ec.test_gpu_kv_restore -v`：6 项通过。
  当前 Conda 没有 pytest，使用仓库的 unittest 入口。
- 事务、异常注入、真实 allocator、真实请求池方法、Scheduler 暂停相关重点回归：
  66 项，65 通过、1 项 Unix socket 因沙箱跳过；跳过项不计作真实 CUDA 验收。
- 扩展目录回归已运行 520 项：499 通过、19 跳过、2 项 TCP 集成用例因沙箱
  禁止 `socket()` 而报错，没有把这些环境阻塞项记作通过。命令需包含
  `PYTHONPATH="$PWD/experiments/kv_ec:$PWD/python:$PWD"`，兼容旧测试的裸模块导入。
- 本地 `torch=2.13.0+cu130`、`torch.cuda.is_available() == False`，没有
  `/dev/nvidia*`；`nvidia-smi` 无法连接驱动。SSH 到 `10.102.0.233` 被执行
  沙箱拒绝（`socket: Operation not permitted`，approval policy=never）。
  **因此本轮尚未执行 233 上的真实 GPU 写回，GPU PASS 未取得。**
- 只读解析了现场历史真实 TP2 `global_node.bin`，逐 Rank/层校验有效：
  `/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local/5695c8b23b7b66cd11158fd11cb01b970185b44ad57436f99929f294e40a0c3b/global_node.bin`。
  SHA256：`beeba53b49d1eb6a751493ce664d94c348829cac1f13d6c0b58e313c40ab82d0`。
  token_start=0，token_count=256，TP=2，每 Rank 为 **18,874,368 bytes**。
  这只是已有真实 Checkpoint 的 CPU 校验，不是本轮 GPU 写回结果。
- 已重新读取实际模型 config / safetensors，model SHA256 为
  `7b9847a1e1530259c3b60505e32f1c8aa99ca20bd26c8de4ac5ede00262cc347`，
  与历史 Checkpoint 相同。公开测试前 256 个 Token 的 digest 也已实际重建核对。
- 本轮本地临时证据使用独立目录 `/tmp/kv-gpu-restore-stage-a-cpu.u9L1C8`。
  不覆盖历史在线 XOR / RDMA / RS 实验文件。

## 本轮修改文件

| 文件（相对于仓库根目录） | 修改 |
|---|---|
| `experiments/kv_ec/gpu_kv_restore.py` | 分配 undo journal、所有权检查、可补偿提交、CUDA fence、逐层证据 |
| `experiments/kv_ec/gpu_restore_hook.py` | 默认关闭的 Scheduler 单请求屏障与双 Rank 协调 |
| `experiments/kv_ec/verify_gpu_restore.py` | inspect / public-test-inputs / launch / submit / check CLI |
| `experiments/kv_ec/test_gpu_kv_restore.py` | 原事务用例改用真实 allocator / journal，保留全部检查 |
| `experiments/kv_ec/test_gpu_restore_hook.py` | 新增异常注入、TP 协议、验收 JSON 反例检查 |
| `experiments/kv_ec/test_scheduler_pause.py` | 补齐真实 Scheduler 循环测试的禁用 Hook 字段 |
| `python/sglang/srt/managers/scheduler.py` | 条件初始化及 before_inputs / after_forward / after_result 调用 |
| `python/sglang/srt/debug_utils/kv_ec.py` | 条件启用时才加载 checkout 中的恢复 Hook |
| `python/sglang/srt/mem_cache/allocator/token.py` | free-list replacement 的异常原子性边界 |
| `python/sglang/srt/mem_cache/slot_lease.py` | 整批校验通过后再修改 generation，拒绝重复分配 |
| `experiments/kv_ec/ONLINE_RS_STAGE2.md` | 修正旧 pre-forward 指引及单 Rank commit 示例 |
| `experiments/kv_ec/GPU_RESTORE_STAGE_A.md` | 本轮生命周期、命令、结果字段、限制及验收状态 |

额外的本地 CPU 报告为
`/tmp/kv-gpu-restore-stage-a-cpu.u9L1C8/cpu_validation.json`，不属于 Git 变更。

## 生命周期与安全边界

1. 请求创建不会直接得到 KV Row。`ScheduleBatch.prepare_for_extend()` 调用
   `alloc_for_extend()`，申请 Req Pool Row 和 KV Slots，并填写 `req_to_token`。
   `ReqToTokenPool.alloc()` 只在首次分配 Row 时递增 generation；分块 Prefill
   复用 Row 不递增。Free 只归还 Row，旧映射数据可能仍留在空闲 Row 中，不能
   根据空闲 Row 的残留数据推断物理 Slot 所有权。
2. `prepare_for_extend()` 将 `kv_committed_len` 更新到目标长度。**这本身不是
   Forward 已完成的证据。** Hook 位于 `event_loop_normal()` 的 `run_batch()`
   返回后、`process_batch_result()` 前，并在读取 / 写入前执行设备同步，确保
   Attention backend 已通过 KV Pool 的 `set_kv_buffer()` 完成原始 KV 写入。
3. 新 Hook 默认不导入实验模块。只有 `SGLANG_KV_EC_GPU_RESTORE=1` 才创建。
   `before_inputs()` 只接受一个 256 输入 Token、`max_new_tokens=1`、无 Session /
   Embedding / 多模态的普通请求，Token digest 必须匹配 Checkpoint。修改权重、
   清池、取消、第二个生成请求等其他 RPC 在本独立实验中拒绝。
4. Hook 限定 Qwen3-8B BF16、36 层、NHD、page-size=1、TP=2、单请求、无 Overlap、
   CUDA Graph、Radix Cache、共享 Prefix、推测解码和动态存储。Chunk 为首个
   256 Token；实验请求正常完整 Prefill 256 Token，禁用 Prefill 分块以便唯一边界。
5. 在申请新 Slots **之前**检查 Row / generation、实际已写 KV 长度与分配长度、
   全行映射唯一性、物理范围和 allocator 空闲集合。必须只有一个活动 Row，且
   所有非空闲物理 Slots 都能由该请求解释；其他 Row 的残留映射不参与判断。
6. Row guard 禁止 Row 分配、释放、清空、池写入。当前非 Overlap Scheduler
   线程同步阻塞在 Hook 内；没有后台实验任务读写 Pool / Slots，在线保护、暂停
   及 Debug Hook 不允许同时启用。因此暂停期间不会运行下一次分配、释放或
   `prepare_for_decode()`，不会发生旧 Slots 被复用的竞争。
7. 写回使用该 Rank 自己的 Scheduler 进程、真实 `global_node.bin` Rank payload、
   真实 `MHATokenToKVPool.set_kv_buffer(None, KVWriteLoc(...), ..., layer_id_override=...)`。
   每层 K/V 在私有 Stream 写回；完成 fence 后通过 `get_kv_buffer()` 和新 Slots
   读回原始字节，逐字节比较，并保存每层 K/V SHA256。
8. `online_chunk.py` 的 `after_forward()` 原本也是在结果释放前获取独立 Staging；
   `before_inputs()` 管理模型 / 存储 epoch，`after_result()` 退休失效请求。新 Hook
   复用相同安全位置，但互斥启用；不会把在线保护后台任务变成 Pool 写入者。
9. 成功提交后才允许 `process_batch_result()` 处理已经采样的一个输出。该请求按
   自己的 `max_new_tokens=1` 正常结束，ChunkCache 通过**新映射**释放 KV，随后
   释放 Row。Hook 的 `after_result()` 再检查 Row / generation / 空闲集合与 TP
   一致性。不会伪造 finished 状态或篡改普通请求的 Prefix Prefill。

`ScheduleBatch.prepare_for_decode()` 会先分配下一 Slot、更新 Row 映射并增加
`kv_committed_len`，不能用这个时刻尚未 Forward 的计数当作真实 KV 长度。本阶段
绝不运行下一次 Decode。HTTP 返回的输出来自恢复前那次正常 Prefill 的采样，
不是“恢复后的推理接续”证据。

## 事务与 TP 一致性

`allocate_restore_slots()` 在调用 allocator 前保存 free/release lists 与 Slot
generation；分配抛错、返回 None / 短数组 / 重复 / 越界 / 旧 Slot 别名以及分配
记账不一致，都回滚该日志。只允许单个 Scheduler 线程持有日志；不声称它支持
任意并发 allocator。原有 lease / KV / dtype / shape / checksum 校验全部保留。

`TokenToKVPoolAllocator.free()` 先创建目标 Tensor，再发布 free list，避免
`torch.cat()` 失败造成部分修改。Lease registry 的 alloc 在修改任何 generation
之前验证整组 Slot 范围和重复项。恢复提交前再次验证 Row、generation、旧映射和
完整物理所有权，拒绝误释放其他使用者的 Slots。

每 Rank 严格按以下顺序进入有界 CPU ProcessGroup 集合通信：

```
本地 VERIFIED → 两 Rank VERIFIED + identity 一致
             → 本地安装 MAPPED → 两 Rank MAPPED
             → 本地释放旧 Slots / COMMITTED → 两 Rank COMMITTED
             → FINISHED（释放 guard / 日志）→ 两 Rank FINISHED
             → 正常结果处理 / 新 Slots 释放 → 两 Rank CLEANUP → PASS
```

每次投票同时比较 phase、run_id、完整 Checkpoint SHA256、Token digest、Row 和
generation 的 digest，不只是比较一个成功标志。任一 Rank 验证失败时，任何 Rank
都不能开始映射替换；映射或旧 Slot free 失败时，已本地提交的另一个 Rank 也保持
日志，可恢复旧映射和 allocator 状态并归还新 Slots。旧 Slot 已 free 后注入抛错
也能补偿，因为 Scheduler 尚未恢复，没有其他分配者。

所有错误 / abort 路径先等 CUDA 写入与读回完成，再撤销映射 / 分配。设备 fence
失败时**隔离资源并终止本次实验**，绝不将可能仍被 CUDA 访问的 Slots 归还。
Generation / guard 已被非法改变且无法证明 Row 身份时，同样 fail closed。
有界集合通信超时、进程崩溃、FINISHED 失败都不进入后续模型 Forward；这是进程内
可补偿协议，**不是**持久化分布式事务或整节点故障恢复。进程退出由操作系统回收
整个 GPU Pool，不能把隔离 / 超时路径记作无泄漏 PASS。

## 233 完整执行命令

以下命令用于解除当前执行沙箱限制后的 233 本机终端；本轮并未执行 GPU 部分。
使用已实际解析过的首个 256-Token 历史参考，只读使用旧文件。`--gpus auto`
读取现场 `nvidia-smi`，选择两张无其他计算任务且通过显存预算的 GPU；不停止任何
进程。不满足两张空闲卡时直接拒绝，人工等待资源即可。

终端 A：准备新的独立目录与请求输入、启动前台服务。

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
git branch --show-current
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv

export CHECKPOINT=/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local/5695c8b23b7b66cd11158fd11cb01b970185b44ad57436f99929f294e40a0c3b/global_node.bin
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export RUN=$(mktemp -d /tmp/kv-gpu-restore-stage-a-20261027.XXXXXX)
export PORT=30481
printf '本次新目录：%s\n' "$RUN"
# 在终端 B / C 手动设置为相同 RUN；evidence 必须还不存在。
python -m experiments.kv_ec.verify_gpu_restore inspect --checkpoint "$CHECKPOINT"
python -m experiments.kv_ec.verify_gpu_restore public-test-inputs \
  --checkpoint "$CHECKPOINT" --model "$MODEL" --output "$RUN/input_ids.json"
# 此工具只重建 request_debug.py 的公开测试输入，并先检查 Token digest。
# 私有 / 其他请求必须使用自己保留的原始前 256 个 input_ids JSON，不能猜测。

set -o pipefail
python -u -m experiments.kv_ec.verify_gpu_restore launch \
  --model "$MODEL" --checkpoint "$CHECKPOINT" --output "$RUN/evidence" \
  --gpus auto --port "$PORT" --mem-fraction-static 0.60 --timeout 120 \
  2>&1 | tee "$RUN/server.log"
```

Launch 在模型启动前计算实际 config + safetensors 的 SHA256，与 Checkpoint
比较；历史代码 commit / diff 与当前修复允许不同，两个 provenance 都保留。
它使用 `launch_online.server_argv()` 的相同普通推理安全参数，但不启用在线
保护：`--tp-size 2 --dtype bfloat16 --page-size 1 --attention-backend triton
--disable-radix-cache --disable-overlap-schedule --max-running-requests 1
--chunked-prefill-size -1 --skip-server-warmup`，Prefill / Decode CUDA Graph
均为 disabled。Health endpoint generation 显式关闭；不要调用 `/health_generate`。
不能使用之前的 `launch_online` 同时开启保护，本轮只消费只读历史 Checkpoint。

终端 B：提交唯一请求；用终端 A 打印的**实际**目录替换 `REPLACE`。

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RUN=/tmp/kv-gpu-restore-stage-a-20261027.REPLACE
python -m experiments.kv_ec.verify_gpu_restore submit \
  --output "$RUN/evidence" --input-ids "$RUN/input_ids.json" \
  --url http://127.0.0.1:30481 --wait-ready 600 --timeout 300
```

终端 C：验收，以此命令退出码 0 和完整双 Rank JSON 为准。

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RUN=/tmp/kv-gpu-restore-stage-a-20261027.REPLACE
set -o pipefail
python -m experiments.kv_ec.verify_gpu_restore check --output "$RUN/evidence" \
  | tee "$RUN/acceptance.json"
```

服务保持前台运行；实验结束后只人工 Ctrl-C 自己启动的该服务。工具不启动 /
停止其他任务，不修改旧 Checkpoint，不调用 Git commit / push。

## 结果字段与 PASS 标准

新目录包括 `run.json`、`submitted.json`、`http_response.json`、每 Rank 独立的
`rank_0.events.jsonl / rank_1.events.jsonl` 与 `rank_0.json / rank_1.json`。
HTTP 文件本身不参与 GPU PASS 判定。`check` 还会重新校验只读 Checkpoint、两份
Rank 结果和完整集合通信事件序列，任何缺失 / 不一致以非零退出。

每 Rank 成功 JSON 必须包含（以下是字段说明，**不是实际 GPU 结果**）：

```json
{
  "schema": "kv-gpu-restore-stage-a-v1",
  "run_id": "<new evidence directory name>",
  "tp_rank": 0,
  "tp_size": 2,
  "checkpoint": "<real readonly global_node.bin>",
  "checkpoint_sha256": "<pinned full file SHA256>",
  "token_ids_sha256": "<matching live request prefix>",
  "cuda_executed": true,
  "verified": true,
  "committed": true,
  "transaction_state": "COMMITTED",
  "row": 1,
  "generation_before": 1,
  "generation_after": 1,
  "generation_unchanged": true,
  "old_slots": ["256 unique positive integers"],
  "new_slots": ["256 disjoint unique positive integers"],
  "slots_disjoint": true,
  "mapping_matches_new": true,
  "original_prefill_tokens": 256,
  "original_prefill_forward_count": 1,
  "original_sampled_output_tokens": 1,
  "restore_forward_count": 0,
  "decode_forward_count": 0,
  "forward_ct_before_restore": 1,
  "forward_ct_after_restore": 1,
  "row_released": true,
  "request_finished": true,
  "allocator_size": "<actual integer>",
  "free_slots_after": "<same actual integer>",
  "no_slot_leak": true,
  "failure_reason": null,
  "runtime_identity": {"model_sha256": "<actual matching weights hash>"},
  "stats": {
    "source": "<checkpoint path>",
    "node_id": 1,
    "tp_rank": 0,
    "tp_size": 2,
    "token_start": 0,
    "token_count": 256,
    "loaded_kv_tokens": 256,
    "written_bytes": 18874368,
    "recomputed_tokens": 0,
    "unexpected_prefill": false,
    "h2d_ms": "<measured number>",
    "verify_ms": "<measured number>",
    "layers_verified": [
      {"layer_id": 0, "k_byte_exact": true, "v_byte_exact": true,
       "k_sha256": "<checkpoint layer K hash>", "v_sha256": "<checkpoint layer V hash>"}
    ]
  },
  "pass_stage_a": true
}
```

`layers_verified` 实际必须覆盖 **0..35 全部 36 层**，上例省略其余层。
Rank 1 必须独立完成相同过程，`tp_rank=1`，哈希对应自己的 Rank payload。
同一个物理 Slot 编号出现在两个 GPU 上是正常的；不重叠检查指各 Rank 本地
old/new 集合，不能错误地要求跨 GPU Slot ID 不重叠。

PASS 同时要求：真实 CUDA 执行；两 Rank VERIFY / 全局 COMMIT / CLEANUP 全部
成功；每层 K/V 字节及哈希完全匹配源 Checkpoint；各加载 256 Token；恢复本身
没有任何模型 Forward / Prefill / Decode；generation 不变；请求结束后所有
物理 Slots 恢复到完整无重复的空闲集合、Row 归还、guard / journal 均已释放。
失败文件保留本地失败原因及 cause chain，不能将隔离、超时或半提交算作 PASS。

## CPU 回归与阶段 B 接口

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
  experiments.kv_ec.test_gpu_kv_restore \
  experiments.kv_ec.test_gpu_restore_hook \
  experiments.kv_ec.test_slot_lease_allocator \
  experiments.kv_ec.test_kv_pool_audit \
  experiments.kv_ec.test_scheduler_pause -v
```

新增异常覆盖 allocator 修改前 / 后抛错，None / 短分配 / 重复 / 越界 / 别名返回，
私有 Stream 创建失败、部分写入失败、GPU 读回校验失败、CUDA fence 失败隔离、
generation 改变、旧映射改变、其他活动 Row、旧 Slot 已空闲、Row guard 防复用、
旧 Slot free 修改前 / 后抛错，以及单 Rank VERIFY / COMMIT 失败与集合通信超时。
CPU 注入测试运行真实 Row / allocator / transaction 代码，只替换 CUDA 原语，
不构成真实 GPU PASS。

阶段 B 可以复用 `restore_to_gpu()`、`install_mapping()`、
`commit(retain_rollback=True)`、`finish()` / `abort()` 和 TP 阶段投票接口；但是
在支持多请求、其他 allocator、持久化故障恢复之前必须重做所有权协议。尚未
实现 Request 状态载入、请求 / Checkpoint 身份授权、完整 Token 序列及采样状态
恢复、最后已采样但未写 KV Token 的处理、Prefix / Decode Bookkeeping 的接续，
以及节点故障后的共识 / 重连。阶段 A 不能由此宣传为阶段 B 推理接续。
