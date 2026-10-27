# Stage B：同实例、单请求 GPU KV 恢复后 Decode

本实验复用阶段 A 的 `restore_to_gpu()` 和双 Rank 可补偿事务。历史
`global_node.bin` 是唯一恢复数据源。不是跨节点 Failover，也不支持从只包含
KV 的文件重建任意消失的请求。阶段 A 的入口及事务不变。

## 代码级边界审查

设输入 P 长 256，Prefill 采样 y0，第 d 次 Decode 输入 y(d-1)。
`kv_committed_len` 是分配/调度长度，只有 Forward 完成并同步后才代表物化 KV。

| 时刻 | origin_input_ids / output_ids | kv_committed_len | Batch / 下一个输入 | Row / 映射 |
|---|---|---|---|---|
| Req 创建 | P / [] | 0 | 尚未准备 | 尚未分配 |
| prepare_for_extend 完成 | P / [] | 256（尚未物化） | EXTEND, seq_len=256, positions=0..255 | alloc_for_extend 分配 Row，generation +1，写 256 Slots |
| Prefill Forward 返回、结果处理前（恢复边界） | P / []；采样 y0 在 result/FutureMap | 256（同步后已物化） | seq_len=256，y0 **没有 KV** | Row 有效，旧 Slots 归本请求；事务写新 Slots、验证、双 Rank 提交 |
| Prefill result 处理后 | P / [y0] | 256 | FutureMap 保存 y0；input_ids 可为 None | ChunkCache.cache_unfinished_req 复制**新映射**到 prefix_indices；请求未结束 |
| prepare_for_decode(d) 完成 | P / [y0..y(d-1)] | 256+d（最后一项尚未物化） | DECODE, seq_len=256+d, position=255+d；resolve_forward_inputs 从 FutureMap 取 y(d-1) | 保留新 Prefix，分配一个输出 Slot，追加行映射 |
| Decode(d) Forward 返回 | 同上 | 256+d（已物化） | Triton kv_indices 覆盖 [0,256+d)，本轮 result 为 yd | Prefix 仍为恢复后的 256 Slots |
| Decode(d) result 处理后 | P / [y0..yd] | 256+d | yd 无 KV；未结束时进入下一 Decode | 最后一轮 ChunkCache 按当前映射释放全部 KV，再释放 Row；free 不增加 generation |

对应源码：`schedule_batch.py::prepare_for_extend/prepare_for_decode`、
`overlap_utils.py::resolve_forward_inputs`、`tp_worker.py::forward_batch_generation`、
`batch_result_processor.py::process_batch_result_prefill/process_batch_result_decode`、
`chunk_cache.py::cache_unfinished_req/cache_finished_req`。

实验正常 Prefill 一次以建立活请求、原生 FutureMap 和首个采样 Token。这个成本
单独报告 `original_prefill_tokens=256`，绝不声称无冷启动 Prefill。恢复后禁止
任何 EXTEND；17 个输出中 y0 来自恢复前 Prefill，后续至少 16 个来自恢复后的
真实 Decode。不把 y0 当作恢复后生成的 Token。

## 实现与 TP 安全语义

- `GPUResumeHook` 继承阶段 A 的输入门禁、checkpoint loader、投票、事件与失败
  记录；直接调用原 `restore_to_gpu()` / `coordinated_restore()`，不改写 GPU
  事务实现。新入口设置 `SGLANG_KV_EC_GPU_RESTORE=1` 和
  `SGLANG_KV_EC_GPU_RESUME=1`；阶段 A 启动器显式将后者设为 0。
- `Scheduler.event_loop_normal()` 在 `run_batch()` 之前调用 `before_forward()`。
  阶段 A 实现为空操作；阶段 B 检查请求身份、generation、独占物理所有权、
  原输入 digest、映射版本和顺序，再执行双 Rank READY 投票。
- 仍在原来的 `after_forward()` 边界执行恢复，顺序为 VERIFIED → MAPPED →
  COMMITTED → FINISHED，全部完成双 Rank 投票后才能回到原生 result processor。
  Prefix 的旧 Slots 正常释放。它们以后可作为 Decode **后缀** Slot 重用；
  禁止的是旧 Slots 继续充当恢复的 256-Token Prefix。
- 事务前后保存活请求状态，包括原输入 digest、空 output_ids、KV 长度、
  seq_lens、Row/generation、extend/decode 计数和 FutureMap 中的待计算 Token。
  必须完全相等。请求状态由同实例暂停保留，不伪造新 Req，不修改 output_ids，
  不把待计算 Token 计入已恢复的 KV。
- `TpModelWorker.forward_batch_generation()` 在模型 Forward 返回、采样前执行
  可选观察器。观察器要求真实 CUDA Tensor，设备同步后保存真实 ForwardBatch
  的 input_ids、positions、req_pool_indices、out_cache_loc，以及真正 Triton
  backend 的 `forward_metadata.kv_indptr/kv_indices`。本仓库的 backend 通过
  ModelRunner 的 ForwardContext 绑定，**不是** ForwardBatch 的属性。
- 每轮验证 Attention 的全部 KV 索引与当前 Row 一致，前 256 个索引必须逐项
  等于已提交的新 Slots。Position 是 256+d-1，输入是上轮采样 Token。
  每层共用此 eager Triton metadata；保存全部 raw logits 为 `.npy`（禁用 pickle）。
- 每轮还有 OBSERVED 和 RESULT 双 Rank 投票；RESULT 的投票身份包括完整输出
  Token digest。单 Rank 校验/采样/结果状态失败，另一 Rank 不得进入下一次
  Forward。集合通信超时、进程崩溃仍按阶段 A 的 fail-closed 语义终止实验；
  不声称支持任意时刻进程死亡后的分布式恢复。
- 单 Scheduler 线程、无 Overlap/Graph/Radix/共享缓存、唯一请求及拒绝其他
  mutation RPC，保证恢复期间不可能并发释放或复用旧 Slots。恢复之后由原生
  Scheduler 分配 Decode 后缀 Slots，最后由 ChunkCache 和 Req Pool 正常清理。

## 三种运行模式

| mode | 行为 | `check` 退出码 0 的含义 |
|---|---|---|
| reference | 相同 256 输入，正常 Prefill + 16 次 Decode，完全不替换 KV | `pass_reference=true`，只是参考运行通过 |
| resume | 相同输入和采样参数，Prefill 后从真实 checkpoint 替换 GPU KV，再执行 16 次 Decode | 双 Rank、逐层字节、真实索引、清理、与参考逐 Token 比较全部通过时，`pass_stage_b=true` |
| corrupt | Rank 0 在新 Slots 尚未安装时翻转第 0 层一个 K 字节，再读回；双 Rank 撤销事务，原生单输出请求清理 | `negative_test_pass=true`；`pass_stage_b=false`，0 次 Decode |

负向模式有意在阶段 A 的完整验证之后再翻转私有 Slot，阶段 B 在全局 VERIFIED
前额外读回全部层。这样验证真实扰动检测和双 Rank abort，而不往原接口添加
绕过校验的开关。被扰动的数据永不映射到请求，原 Slots 未损坏，`max_new_tokens=1`
使请求按原生结束路径释放旧 Slots。其他恢复错误仍属致命失败，不能算作负向通过。

## 233 操作命令

以下为 233 本机终端命令。先确认 GPU 1、3 空闲且满足启动器显存检查；不停止
任何别人的任务。三个模式**依次**运行，各模式使用独立子目录。每个服务以前台
方式启动；该模式检查完成后，只人工 Ctrl-C 自己启动的服务，再进入下个模式。
不要在显存不足时同时启动参考和恢复服务。

初始终端：

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
source /media/sda1/iic_data/anaconda3/etc/profile.d/conda.sh
conda activate sglang-ec
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
git branch --show-current  # feat/kv-gpu-restore-resume-20261027
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
export MODEL=/media/sda1/iic_data/luotao/models/Qwen3-8B
export CHECKPOINT=/tmp/kv-online-rs-online-rs-tp2-shafix-t01/node1/local/5695c8b23b7b66cd11158fd11cb01b970185b44ad57436f99929f294e40a0c3b/global_node.bin
export RUN=$(mktemp -d /tmp/kv-gpu-resume-stage-b-20261027.XXXXXX)
export PORT=30482
printf 'RUN=%s\n' "$RUN"

python -m experiments.kv_ec.verify_gpu_resume inspect --checkpoint "$CHECKPOINT"
python -m experiments.kv_ec.verify_gpu_restore public-test-inputs \
  --checkpoint "$CHECKPOINT" --model "$MODEL" --output "$RUN/input_ids.json"
```

`public-test-inputs` 只允许重建原公开测试 Prompt 并核对 digest。其他 Checkpoint
必须提供自己保存的原始 256 个输入 IDs，不能从 KV 猜测 Token。

参考运行，终端 A：

```bash
set -o pipefail
python -u -m experiments.kv_ec.verify_gpu_resume launch \
  --mode reference --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --output "$RUN/reference" --gpus 1,3 --port "$PORT" \
  --mem-fraction-static 0.60 --decode-steps 16 --timeout 120 \
  2>&1 | tee "$RUN/reference.server.log"
```

终端 B 同样激活环境并设置 PYTHONPATH，将 RUN 替换为终端 A 打印的实际路径：

```bash
export RUN=/tmp/kv-gpu-resume-stage-b-20261027.REPLACE
export PORT=30482
python -m experiments.kv_ec.verify_gpu_resume submit \
  --output "$RUN/reference" --input-ids "$RUN/input_ids.json" \
  --url "http://127.0.0.1:$PORT" --timeout 300
python -m experiments.kv_ec.verify_gpu_resume check --output "$RUN/reference"
echo "REFERENCE_CHECK_EXIT=$?"
```

确认 `pass_reference=true` 后，在终端 A 人工 Ctrl-C 此参考服务。终端 A 保持
MODEL / CHECKPOINT / RUN / PORT 环境变量，再启动恢复实验：

```bash
python -u -m experiments.kv_ec.verify_gpu_resume launch \
  --mode resume --reference "$RUN/reference" \
  --model "$MODEL" --checkpoint "$CHECKPOINT" --output "$RUN/resume" \
  --gpus 1,3 --port "$PORT" --mem-fraction-static 0.60 \
  --decode-steps 16 --timeout 120 2>&1 | tee "$RUN/resume.server.log"
```

终端 B：

```bash
python -m experiments.kv_ec.verify_gpu_resume submit \
  --output "$RUN/resume" --input-ids "$RUN/input_ids.json" \
  --url "http://127.0.0.1:$PORT" --timeout 300
python -m experiments.kv_ec.verify_gpu_resume check \
  --output "$RUN/resume" --reference "$RUN/reference"
echo "RESUME_CHECK_EXIT=$?"
```

确认/保存恢复结果后，只人工停止此恢复服务。终端 A 启动负向实验：

```bash
python -u -m experiments.kv_ec.verify_gpu_resume launch \
  --mode corrupt --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --output "$RUN/corrupt" --gpus 1,3 --port "$PORT" \
  --mem-fraction-static 0.60 --decode-steps 16 --timeout 120 \
  2>&1 | tee "$RUN/corrupt.server.log"
```

终端 B：

```bash
python -m experiments.kv_ec.verify_gpu_resume submit \
  --output "$RUN/corrupt" --input-ids "$RUN/input_ids.json" \
  --url "http://127.0.0.1:$PORT" --timeout 300
python -m experiments.kv_ec.verify_gpu_resume check --output "$RUN/corrupt"
echo "NEGATIVE_CHECK_EXIT=$?"
```

启动器复用阶段 A 的资源检查与固定参数：Qwen3-8B BF16、TP=2、eager Triton、
page-size=1、单请求、context=4096，无 Overlap、CUDA Graph、Radix、chunked
Prefill。`FreezeGCReq` 保留显式允许；`/health` 生成关闭。不要调用
`/health_generate`、取消请求、修改服务状态或提交第二个请求。

## 证据与 PASS

每个目录保留 `run.json`、`submitted.json`、`http_response.json`、两份
`rank_N.json`、事件 JSONL、每 Rank 的 17 份 raw logits（corrupt 仅 1 份）。
`submit` 在 HTTP 返回之后额外等待完整双 Rank 清理报告。HTTP 成功不是 PASS。

`check` 核验只读 checkpoint、两 Rank 身份和 runtime identity、reference
manifest、逐层字节哈希、实际 Decode 索引、Token/Position、Forward 次数、完整
投票事件顺序和资源清理，随后逐项比较 Token IDs 与 raw logits。参考和恢复的
服务参数除端口外必须一致；实验之间不要修改代码、权重或参考证据。

最终验收文件是不可覆盖的 `check.json`；相同证据可重复检查，若结果变化则拒绝
覆盖既有文件。Rank 的原始报告只声明 `local_resume_complete`，其
`pass_stage_b=false` 表示尚未经独立参考比较。最终 `check.json` 的顶层及
`ranks` 中的 `pass_stage_b` 才是组合验收结论，原始 Rank 报告不被改写。

下面只是默认正向模式的**目标结构，不是实测结果**：

```json
{
  "pass_stage_b": true,
  "cuda_executed": true,
  "tp_size": 2,
  "restored_kv_tokens": 256,
  "restored_slots_used_by_decode": true,
  "resume_decode_forward_count": 16,
  "recomputed_prefix_tokens": 0,
  "reference_token_ids_sha256": "<17个输出Token的SHA256>",
  "resumed_token_ids_sha256": "<与reference相同>",
  "output_token_ids_match": true,
  "generation_unchanged": true,
  "no_slot_leak": true,
  "first_difference": null
}
```

必须两 Rank 同时满足条件；36 层 K/V 校验不变。17 个输出中只有后 16 个归于
恢复后 Decode。`recomputed_prefix_tokens=0` 指恢复之后；原来正常执行的
`original_prefill_tokens=256` 在 Rank 报告中单列。

Token 不一致时退出码为 1，保存 `first_difference`、双方 Token、对应两个
Token 的 raw logits，以及逐轮 `max_abs_error/rms_error`。首次 Token 分歧之后
两边已使用不同上下文，之后的 Logits 误差不能解释成相同输入下的数值误差。
即使输出 Token 全同，也保留非零 Logits 误差，不将其伪装为逐位相等。

## CPU 回归与限制

本次 CPU 执行记录：

- 阶段 A 原有定向回归：42 项全部通过。
- 最终阶段 A/B + Scheduler 暂停定向回归：72 项，71 通过、1 项 Unix socket
  测试因沙箱权限跳过；其中阶段 B 新增 18 项全部通过。
- 全量扫描一次运行 547 项：526 通过、19 跳过、2 项既有 TCP 集成测试因
  `socket.socket()` 返回 `PermissionError: Operation not permitted` 受阻。
  该全量扫描之后补入的边界/验收测试包含在上述最终定向回归中。
- `py_compile` 和 `git diff --check` 通过。真实历史 checkpoint 的 CPU-only
  inspect 通过，每 Rank 18,874,368 bytes；这不构成 CUDA 恢复验收。

针对阶段 A/B 和 Scheduler 暂停的回归命令：

```bash
python -m unittest experiments.kv_ec.test_gpu_resume \
  experiments.kv_ec.test_gpu_restore_hook experiments.kv_ec.test_gpu_kv_restore \
  experiments.kv_ec.test_scheduler_pause -q
```

全量旧测试中有顶层 `from benchmark import ...`，需加其原有实验目录：

```bash
PYTHONPATH="$PWD/experiments/kv_ec:$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s experiments/kv_ec -t . -p 'test_*.py' -q
```

本次开发执行环境 `torch.cuda.is_available()==False`、device_count=0，因此这里
没有获得阶段 B 的真实 TP2 CUDA PASS。用户提供的阶段 A 真实验收成功记录保留；
CPU 模型/Logits/投票替身仅验证状态机，不能作为 GPU 验收。沙箱禁止创建 socket，
已有 TCP 集成测试会以 PermissionError 受阻；不因此关闭网络传输校验。

仍不支持：消失 Req 的重新创建、跨节点 Failover、从 KV 推导待计算 Token、
非首块/任意恢复点、共享 Prefix/Radix、Overlap、Graph、推测解码、多请求或
自动故障检测。真正冷恢复需要额外持久化 Token 历史、采样/惩罚器/FutureMap 状态
并定义请求和输出提交协议，本轮没有虚构这些接口已经完成。
