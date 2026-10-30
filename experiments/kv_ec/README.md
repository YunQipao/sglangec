# KV EC 第一阶段：源码分析、CPU XOR 与 Snapshot 格式

双逻辑节点 + 第三机 Gateway 的独立实验原型及限制见
[TWO_NODE_GATEWAY.md](TWO_NODE_GATEWAY.md)。早期 Gateway 章节的范围是 TCP
relay、CPU synthetic 网络与 A/C chunk-XOR；它不代表下面新增的物理 TP4
Snapshot/RDMA 入口，也不能用其结果声称真实在线 Serving 恢复。

当前 TP4 双物理节点、CPU-owned KV Node Block、独立 descriptor 注册、TCP
Remote RS(2,1) 和逻辑恢复原型见
[DUAL_NODE_REMOTE_RS.md](DUAL_NODE_REMOTE_RS.md) 的 M 节；可执行入口为
`experiments.kv_ec.physical_two_node_rs`。该入口只处理已经完成 Capture 的
离线 bytes，默认不改变 SGLang Serving，也不实现 GPU KV Pool 写回。

同一入口现在支持 `--transport rdma`：TCP 继续承载 Remote RS 控制协议，
完整 TP4 Node Block、Parity 与恢复块通过 `experiments.kv_ec.multinode`
中的真实 libibverbs RC QP/RDMA Write 传输。服务器 B 的 RDMA 参数和 Node
A/B 的可复制命令见 [DUAL_NODE_REMOTE_RS.md](DUAL_NODE_REMOTE_RS.md) 的
“真实物理双节点 TP4 RDMA”节。当前 verbs helper 是 one-shot QP；若主机没有
verbs headers/HCA，RDMA 构造会显式失败，不能回退 TCP，真实双机结果必须标记
`NOT VERIFIED`。

单机 4 GPU 模拟 Node0/Node1 的原生 `nnodes=2,tp=4` 实验入口见
[TWO_LOGICAL_NODES.md](TWO_LOGICAL_NODES.md)。

CPU DRAM parity-only 最小在线原型：见 [PARITY_ONLY.md](PARITY_ONLY.md)，
包括状态机、IPC、内存策略、手动 GPU 命令与测量限制。以下早期阶段证据描述
保留作历史记录，不代表新增在线实验已在本会话完成。

## 范围与证据等级

下表记录早期在线 Capture 评估的证据等级；它不把本次新增的 CPU/RDMA
传输原型误报成已完成的 GPU 在线恢复。

分析版本：本地 `v0.5.18`，commit `71de97b264b04dcd514cf904003028aefe9775c8`。
开始时 `git status --branch --short` 为 `HEAD (no branch)`，已有未跟踪
`requirements-installed.txt`，未修改该文件，未切换分支。以下行号均针对这个 commit。
本目录外的 SGLang Python 源码没有修改。没有改安装、依赖或 CUDA，没有清理 GPU 进程。

执行环境：明确调用 `/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python`，
PyTorch `2.13.0+cu130`。当前工具会话 `nvidia-smi` 无法连接驱动，
`torch.cuda.is_available()` 为 False。用户此前报告 TP=1/2/4 推理成功，
但本次不能独立复测，不能用该报告代替本次真实 KV 证据。

| 验收项 | 当前状态 |
|---|---|
| A 正常推理不受影响 | 无核心代码、自动加载或 hook 变更；本次未复测在线推理 |
| B XOR TP=2/4 | CPU 与已有 synthetic CUDA D2H 基准通过；当前会话不能复测 CUDA |
| C 真实请求 KV Snapshot | **未实现/未验证**，只有最小 hook 设计 |
| D 真实 KV byte-exact 恢复 | **未完成**，依赖 C |
| E 代码、命令、CSV | CPU/CUDA synthetic CSV 已提供；没有真实 KV CSV |
| F 默认关闭 | 本目录仅手动执行；SGLang 不会 import 或调用这些实验 |

早期分析章节仍不实现在线增量保护、GPU KV Pool 写回或机器级故障切换；
当前新增的 `physical_two_node_rs --transport {tcp,rdma}` 仅对已完成的真实
TP4 Snapshot 做离线跨节点 RS 和逻辑恢复。
CSV 中的 TP 是合成数据块数，**不是运行中的 SGLang TP 作业**。

## 1. 实际 KVCache 类与访问链路

在给定 Qwen3-8B 普通 CUDA TP 启动配置、无特殊环境变量、无 KV 量化、
无 PP/DP/DCP/speculation/HiCache/HiSparse/自定义内存布局的前提下，构建分支为
`MHATokenToKVPool(KVCache)`，请求池为 `ReqToTokenPool`。
这是本地源码与模型配置推导的选择，**不是本次运行时类名抓取**。
第一份真实证据必须记录 `type(pool).__module__` 和 `type(pool).__name__`；
与推导不符时应拒绝实验，不能静默索引未知布局。

准确链路（路径相对于仓库根目录）：

| 文件:行号 | 类/函数与作用 |
|---|---|
| `python/sglang/srt/managers/scheduler.py:2392` | 构造 `Req(recv_req.rid, ...)` |
| `python/sglang/srt/managers/schedule_batch.py:854` | `Req.__init__` 保存 `self.rid`；`:934` 初始化 `req_pool_idx=None` |
| `python/sglang/srt/managers/scheduler.py:905` | `Scheduler.init_tp_model_worker` |
| `python/sglang/srt/managers/tp_worker.py:125` | `BaseTpWorker.get_memory_pool` 返回本 Rank ModelRunner 的请求池、分配器 |
| `python/sglang/srt/managers/scheduler.py:555` | Scheduler 持有请求池及 token 分配器 |
| `python/sglang/srt/model_executor/model_runner.py:799` | `ModelRunner.alloc_memory_pool` 调用 configurator，保存三个 pool/allocator 引用 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py:924` | `KVCacheConfigurator._build_default_req_pool` 构造普通 `ReqToTokenPool` |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py:948` | `_build_token_to_kv_pool`，`:966` 默认选择 `MHATokenToKVPool`，`:1056` 调用 MHA 构建 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py:1570` | `_build_mha_kv_pool`，按 attn TP/DCP 计算本 Rank KV heads，传 dtype/device/layer range |
| `python/sglang/srt/managers/scheduler.py:3626` | `Scheduler.run_batch` 调用本 Rank worker forward |
| `python/sglang/srt/managers/tp_worker.py:574` | `TpModelWorker.forward_batch_generation` → `ForwardBatch.init_new` → `ModelRunner.forward` |
| `python/sglang/srt/model_executor/model_runner.py:1497` | `ModelRunner.forward`；`:1641` `_forward_raw` 发布 `ForwardContext` |
| `python/sglang/srt/model_executor/forward_context.py:66` | `get_attn_backend`；`:70`/`:74` 从 backend 取 KV/request pool（只在 forward context 有效） |
| `python/sglang/srt/models/qwen3.py:172` | `Qwen3Attention.forward_prepare_native` 做 QKV projection、QK norm、RoPE |
| `python/sglang/srt/models/qwen3.py:268` | `Qwen3Attention.forward`，`:305` 将 Q/K/V 交给 `RadixAttention` |
| `python/sglang/srt/layers/radix_attention.py:150` | `RadixAttention.forward`；`:279` backend forward 路径之一 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py:1286` | 读取 `get_kv_buffer`，`:1292` prefill 非 ragged 分支 `set_kv_buffer`；`:1394`/`:1426` 为其他分支 |
| `python/sglang/srt/layers/attention/triton_backend.py:1260` | `_set_kv_buffer`；`:1336` prefill 调用写池路径 |

Backend 由运行时选择，本次没有证据说一定是 FlashInfer 或 Triton。
保存的是 attention 实际消费的 K（已 QK norm/RoPE）和 V，不是直接截取 QKV 投影输出。

### Unified 不是本模型的默认路径

`python/sglang/srt/mem_cache/unified_memory_pool.py:452` 的
`UnifiedMHATokenToKVPool` 是继承 MHA 的统一字节池 strided view；`:535`
写入接口接收 physical slot，不应据此把普通池也当成虚实翻译池。
`kv_cache_configurator.py:389` 的 unified 分支只支持 hybrid Mamba/hybrid SWA，
`:424` 附近会拒绝普通 Qwen3。不要为本实验开启 `--enable-unified-memory`。
当前没有一个应调用的通用 `TokenToKVPool` 具体类；变量 `token_to_kv_pool`
指向具体 `KVCache` 实例，`TokenToKVPoolAllocator` 是分配器，不是 K/V Tensor 容器。
具体类是 `memory_pool.py:1624` 的 `KVCache` 和 `:1755` 的 MHA 子类。
配置器 `:1701` 起按 page_size/DCP 选择 token 或 paged allocator。

## 2. 每个 TP Rank 的形状、dtype、stride、device

模型文件 `/media/sda1/iic_data/luotao/models/Qwen3-8B/config.json`：
`:9` head_dim=128，`:18` 36 层，`:19` 8 个 KV heads，`:25` bfloat16。
`Qwen3Attention.__init__`（`qwen3.py:91`）按 attn TP 分 heads；TP=2/4 不涉及 heads 复制。
`kv_cache_dtype.py:22` 的 `configure_kv_cache_dtype` 在 `auto` 非量化路径用 model dtype。

普通 NHD 下每个 Layer 的 K、V **分别**是：

| TP | 本 Rank KV heads | K/V shape | 连续 Tensor stride（元素） | dtype |
|---|---|---|---|---|
| 1 | 8 | `[S+P, 8, 128]` | `(1024,128,1)` | 默认 BF16 |
| 2 | 4 | `[S+P, 4, 128]` | `(512,128,1)` | 默认 BF16 |
| 4 | 2 | `[S+P, 2, 128]` | `(256,128,1)` | 默认 BF16 |

`S=pool.size`，`P=pool.page_size`，由启动时预算/后端确定，**不能用 context_length=4096
替代 S**。源码 `memory_pool.py:2045` `_kv_buffer_shapes` 与 `:2058`
`_create_buffers_normal`。36 个 Layer 各有 K/V Tensor（不是一整个请求一个 Tensor）。
device 为对应 Rank 的 CUDA device；逻辑 device 编号与物理 GPU 的映射依赖启动器和
`CUDA_VISIBLE_DEVICES`，必须记录 `tensor.device`、rank、GPU UUID/PCI、可见设备列表，
不能把 `cuda:0` 无条件等同于物理 GPU0。

选择 T 个有效 token 后，打包 K/V 各 `[T,8/TP,128]`，
TP=4 的 K+V 每 token 每层共 1024 bytes，TP=2 为 2048 bytes。
例如 TP=4、单层 64 token 的 K+V 恰为 64KiB；这只是大小计算，不是实测 Snapshot。

上述是**条件性静态推导**。`SGLANG_USE_HND_KVCACHE`、page-major、量化、自定义平台
会改变布局或 stride。真实导出必须保存 `.shape/.dtype/.stride/.device`；当前没有实测值。

## 3–4. Request ID → 请求行 → 真实物理槽位

Request ID 是 `Req.rid`，不是 `req_pool_idx`。在当前 batch 的 `Req` 对象中按 rid
定位请求；本版本没有可以凭 HTTP 请求 ID 在独立客户端直接读 GPU KV 的接口。

`ReqToTokenPool`（`memory_pool.py:256`）在 `:279` 分配 CUDA int32
`req_to_token[(size+1), max_context_len]`，行 0 是 graph padding，
`:283` 创建 **CPU** int64 `req_generation`。
`:291` `alloc` 为请求分配非零行，`:321` 行 generation 加一，`:325` `free`
把行归还并将请求的 `req_pool_idx` 置 None。`clear(:330)` 将 generation 清零，
故还需进程/启动 UUID 与 capture_id，不能只用行 generation 唯一标识跨运行请求。

`allocation.py:281` `alloc_for_extend` 分配请求行和新 token slots；
`:53` `write_cache_indices` 将命中 prefix slots 写到行的前缀，将新 `out_cache_loc`
写到后缀，Triton 或 Python fallback 两条路径等价。
在**已证明安全的生命周期点**，对半开 token 范围 `[start,end)` 获取：

```python
# 设计示意，不是可从外部客户端运行的已实现 API
row = req.req_pool_idx
slots = req_to_token_pool.req_to_token[row, start:end].clone().long()
```

普通 NHD pool 下这些是物理 token slot，可以非连续，不能替换为
`pool_buffer[start:end]`，不能假定一页/一个请求的数据连续。
只允许读取请求已完成 forward 的 token，不能读 padding、未来分配的未写槽位。
应检查 row 非零且未释放、end 在映射宽度内、`0 < slots < buffer.shape[0]`。
若是 paged 布局，padding page 也不得当有效 token；需核实分配器保留页范围。

## 5–8. Prefill 安全提取、各 Rank 独立快照与生命周期

`schedule_batch.py:2440` 在 forward **之前**更新 `kv_committed_len`，
所以它只是长度账本，不是 CUDA completion event。
刚采样出的第一个 output token 也还没有 decode K/V，不能使用
`len(origin_input_ids)+len(output_ids)` 直接作为全部可导出长度。
chunked prefill 还需确认完整 prompt 已计算（`inflight_middle_chunks <= 0`、
本次非 decode/mixed，prompt 长度与本次完成范围一致），拒绝未覆盖范围。

现有读接口：`memory_pool.py:2288` `get_key_buffer`、`:2312` `get_value_buffer`、
`:2324` `get_kv_buffer`。它们会在有 `layer_transfer_counter` 时做层级加载等待，
**不会固定请求所有权、不会阻止 free/reuse，也不保证你的复制与写流同步**。
注释明确说这些 getter 面向 attention backend，不是通用并发 Snapshot API。
量化 getter 可能反量化，不适合宣称原始 storage bytes；第一版必须拒绝量化。
对于无层级加载、普通 BF16 NHD 的已验证安全时机，可通过这些内部 getter 选取
指定 Layer，再用 `index_select(0, slots)` 提取 K/V；不改原池。
`set_kv_buffer(:2327)` 是写接口，不是导出接口，Snapshot 不应调用它。

已有 `MHATokenToKVPool.get_cpu_copy(:2229)` 可复制选定 slots，但会遍历**全部 Layer**
并在前后做平台同步；它不提供所需的单 Layer、请求生命周期保护或 provenance。
不宜默认调用，更不能复制整个 pool。
`allocator/base.py:95` 的 get_cpu_copy 还是 abstract fallback 的 NotImplementedError，
不能假定所有 allocator 都实现了这个接口。

### 最小 hook 设计（尚未安装）

候选位置是 `scheduler_components/batch_result_processor.py:193`
`process_batch_result_prefill` 的每请求处理，**在 `:269` 左右
`req.update_finish_state` / `:277` `release_kv_cache` / `:280`
`maybe_cache_unfinished_req` 之前**，且完整 prefill 的条件已经确定。
`:202` 的 result.copy_done 等待是参考，不可视为对所有模式的生命周期锁。
更稳妥的初版约束：关闭 overlap、chunked prefill、prefix caching；单请求；
无 speculative/PD/PP/DP/DCP/HiCache/unified/page-major/量化/动态迁移。

1. 新实验开关默认 False；显式给目标 rid、layer_id、token range、capture_id，
   所有 Rank 一致启用。先验证模式，不满足时在调度前拒绝，不能在 collective 中途退出。
2. 普通 event loop `scheduler.py:1719` 在 `run_batch` 后处理结果（`:1741`）；
   `:433` 由 `disable_overlap_schedule` 控制，`:4902` dispatch 选择该循环。
   overlap loop `:1754` 会先启动下一批再处理上一批，初版**不支持**。
3. hook 在每 Rank 的 scheduler 所有者线程运行，在写入 forward 完成后，
   本 Rank device synchronize（调试方案）或正确 wait forward completion event。
   停止该调度线程继续释放/重分配，保持现有 prefix refs 不变。
4. 记录 `req.rid`、row、`req_generation[row]`、range、实际 slots；
   使用 `self.tp_worker.model_runner.token_to_kv_pool` 或
   `self.token_to_kv_pool_allocator.get_kvcache()` 取得本 Rank pool。
   不通过 forward-context getter 访问已经退出的 context。
5. 验证精确 pool class、NHD、非量化等；从指定层 gather **选定 token** 的 K/V。
   在所有相关写入 stream 完成后读，复制到独立 CPU DRAM，等待 D2H 完成。
   gather 产生小范围 GPU 暂存，不是整个 KV pool 的同步拷贝。
6. 复制完成前不让 scheduler 返回原来的 release/cache 分支；复制后重查行 generation
   和 slots，再保存 owned CPU 数据。重查只是防误用，**不能替代复制期间排他生命周期**。
7. 每 Rank 独立写文件，不增加 NCCL/barrier/all_gather。跨 Rank 聚合由离线读取完成；
   任一 Rank 缺失/出错则离线拒绝这组 Snapshot，不能用零块补齐冒充原数据。
   错误处理不能让一个 Rank 跳过正常后续 collective；IO 故障应本地记录并正常推理。
8. 完成后继续原有 cache/free 路径，不写池、不改变 allocator/refcount。
   文件 IO 最好在 owned CPU copy 后排队落盘；队列有界，超预算跳过采集。

适合该设计的安全点是“最后一个 prefill forward 已完成，映射尚未被 cache/free 重写”。
不是任意请求结束回调，也不是在另一个进程中读取一份过时 slots 列表。
`common.py:195` `release_kv_cache` 会调用 cache_finished，再在 `:237` 释放请求行。
设备同步只能等待 GPU 工作，不能阻止另一个 CPU 线程释放槽位。

### generation 的限制

请求池有行 generation；检查过 `mem_cache/allocator/` 的普通实现，未发现
可供本实验使用的每物理 KV slot generation 接口。不能虚构 `slot_generation()`。
第一版元数据必须写 `slot_generation=null`、`slot_generation_available=false`，
靠“同线程、写完成、复制结束前不释放”的短期不变量保证；后续在线保护需要
allocator 生命周期 epoch/slot generation 的明确设计，才能异步排队并防 ABA。

当前阻断原因是无法在本会话运行 GPU/真实 scheduler 来验证上述前提，且已有接口
没有通用请求固定/复制事务。**不安装未经验证的在线 hook**，不提供一个接受
`safe=True` 就冒充安全的提取接口。C/D 仍未完成；不能仅靠手动运行一个外部脚本完成 C。

## 9. Prefix caching 与共享槽位

`radix_cache.py:517` `cache_unfinished_req` 可插入、释放重复 slots（`:542`）、
匹配已有 prefix 并重写 req_to_token（`:559`），还更新 lock refs（`:570`）。
`:459` `cache_finished_req` 也会转交/释放 slots；`:622`/`:637` 是树节点锁引用。
因此同一个 physical slot 可能服务多个请求，row generation 不等于独占所有权。
快照只表示“此时这个请求逻辑 token 引用了这些 bytes”，不是这些 bytes 归它独有。

初版设计禁用 prefix caching 以缩小验证范围。未来支持时需保存 prefix 的共享身份、
token 内容/hash、cache_salt/extra_key、模型/权重 epoch、range 与槽位映射，
复制期间保持现有锁引用并阻止本地映射重写。不能为了 Snapshot 释放或夺取共享槽位。
编码条带也不能简单按 request_id 假定底层块互不重叠。

## 独立 CPU XOR 与 D2H 基准

文件：`xor_codec.py`、`benchmark.py`、`snapshot.py`、`test_kv_ec.py`。
不 import SGLang，不 monkeypatch，不触发 server，不调用 torch.distributed/NCCL。

`encode` 输入 N 个等长 contiguous uint8 一维块；输出独立 CPU parity。
`recover` 强制恰好一个 None，只用 parity 和幸存块恢复。保留原块仅作正确性 oracle，
不传给恢复函数。输入不修改；不存在浮点加法、dtype cast 或近似比较。
Snapshot 中 BF16/FP16 用 `.view(torch.uint8)`，不是 `.to(torch.uint8)`。
非连续 tensor 只在已拥有的 CPU 副本上 contiguous 打包；不包含底层 stride padding。

默认 TP=2/4；64KiB、256KiB、1MiB、4MiB、16MiB；每组合 10 warmup + 30 测量。
CUDA 模式每数据块对应一个 visible GPU，先在 GPU 建合成 uint8 数据，
再顺序 blocking D2H 到预分配 pinned CPU buffer。初始化/H2D/校验不计时，
D2H 是全部 N 块的墙钟合计（总字节 N×block_bytes），不是每块均值或并行 PCIe 带宽。
XOR/recovery 包含输出数组分配；所有时间为毫秒，CPU ndarray bitwise_xor。
CPU 模式 d2h_ms 留空，不能读成 0ms GPU 传输；total_d2h_bytes=0。
无自动改变进程 CPU affinity、系统 NUMA/CUDA 设置；CUDA 大块可能显著占 PCIe/DRAM，
手动执行前请确认空闲 GPU。NumPy/torch 版本、测量设置见 CSV 同名 JSON。

已执行：5 个 unittest 方法全部通过，覆盖全部尺寸/TP/单块丢失位置、输入不变、
不合法输入拒绝、全部 65536 FP16/BF16 bit patterns、非连续 CPU Snapshot 打包、
不同 Rank 的合成内容、重复 Rank/不同 capture 拒绝、checksum 损坏检测。
`results/synthetic_cpu.csv` 为真实执行的 **300 行 CPU-only** 基准，全部 correct=True。
各组 30 行，d2h_ms 全空。16MiB 的 XOR 中位数本次约 TP2=5.083ms、TP4=11.642ms，
恢复约 5.729ms/11.567ms；只是本次 CPU 墙钟结果，不代表在线 Serving 开销。
当前工具会话的 CUDA 探测返回明确错误“CUDA is not accessible”；另有
`results/synthetic_cuda.csv`（300 行）及同名 JSON，记录一次已完成的
**synthetic CUDA→CPU** 基准；它不是 SGLang KV，也不是本轮真实导出证据。

## 离线 Snapshot 格式（不是在线提取）

`write_snapshot(path,k,v,metadata)` 只接受**已由调用方安全获得的 CPU 副本**，
equal-shape 3-D FP16/BF16 K/V，打包为 K bytes || V bytes 的非 pickle npz。
文件用 exclusive-create，不覆盖原文件。保存 source shape/stride/device、packed stride、
dtype、layer、TP、token 半开范围、physical slots、request/row generation、
slot generation 可用性、capture_id、source_commit、prefix_sharing、token hash 和 SHA256。
source_k_shape/source_v_shape 必须由调用方记录源池层 Tensor 的 shape，
`k_shape/v_shape` 为选取后打包的形状。stride 单位为元素。

`verify_snapshots` 检查 SHA256、大小、完整 TP ranks、同 capture/request/layer/range/
shape/dtype/commit/token hash/provenance；XOR 整个等长 payload，逐个模拟每 Rank
丢失并逐字节检查恢复。它不验证真实来源：`provenance=sglang_request` 是调用方声明，
不能仅凭这个字段通过 C。必须同时提供运行命令、request input/rid、所有 Rank 采集日志、
实际 slots/generation/layout 与源 Tensor 对比证据。测试中的 provenance 明确为
`synthetic_fixture`。此工具没有产出真实 KV 数据。

## 手动运行命令

无需新增依赖。代码使用现有 torch/numpy。不要改全局安装。

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
# 也支持显式指定仓库根目录作为 discovery 的 top-level：
# python -m unittest discover -s experiments/kv_ec -t . -p 'test_*.py' -v

# 已有 synthetic_cpu.csv，重跑务必换名字；输出存在会拒绝覆盖。
python experiments/kv_ec/benchmark.py --source cpu \
  --output experiments/kv_ec/results/synthetic_cpu_rerun.csv

# 确认空闲显存与 GPU 型号，脚本不会清理其他进程。
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.free --format=csv
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
python experiments/kv_ec/benchmark.py --source cuda --devices 0,1,2,3 \
  --output experiments/kv_ec/results/synthetic_cuda.csv

# 仅当未来 hook 已实现且获得真实文件后，才运行；当前没有这些文件。
python experiments/kv_ec/snapshot.py /path/capture/rank0.npz /path/capture/rank1.npz
# TP4 则传四个同 capture 的 rank 文件。
```

推理回归沿用用户已验证的启动参数，先确认端口未被使用、GPU 可用；不要启动第二个
抢占共享 GPU 的服务。若已有服务可直接测试：

```bash
curl --noproxy '*' --fail-with-body http://127.0.0.1:30182/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/media/sda1/iic_data/luotao/models/Qwen3-8B","messages":[{"role":"user","content":"What is KV cache?"}],"max_tokens":64,"temperature":0}'
```

未来 hook 首次验证建议普通非 overlap、无 prefix、无 chunking 的隔离 server，
具体 CLI 以本地 `sglang serve --help` 为准；不是给现有服务强行切模式。
在 hook 尚未实现前，任何启动命令都不会自动生成真实 Snapshot。

## GPU Scratch Restore（离线实验）

`gpu_scratch_restore.py` 读取已经完成的 `kv-ec-rank-bin-v1` 真实 TP=4
目录，重新验证四个 Rank，生成 CPU XOR parity，分别模拟丢失 Rank 0--3，
并把恢复的 BF16/FP16 原始字节复制到本进程新分配的临时 CUDA Tensor。
它不连接 SGLang、不调用 NCCL、不调用 `set_kv_buffer`，也不会写入在线 KV Pool。
GPU Tensor 会重新读取为字节并与丢失 Rank 的原始 payload 做 byte-exact 校验。

GPU 服务器命令（每个场景建议先 `--warmup 2`，再重复 10 次）：

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
export OUT=/media/sda1/iic_data/luotao/kv_ec_runs

CUDA_VISIBLE_DEVICES=0 \
python experiments/kv_ec/gpu_scratch_restore.py \
  kv_ec_runs/tp4-l0-128-20261016T141855 \
  --device cuda:0 --warmup 2 --repeat 10 \
  --output "$OUT/tp4-l0-128-20261016T141855-gpu-scratch.json"

CUDA_VISIBLE_DEVICES=0 \
python experiments/kv_ec/gpu_scratch_restore.py \
  kv_ec_runs/tp4-all-256-20261016T142603 \
  --device cuda:0 --warmup 2 --repeat 10 \
  --output "$OUT/tp4-all-256-20261016T142603-gpu-scratch.json"
```

`xor` 和 `cpu_recovery` 是 CPU 保护路径指标；`h2d_restore` 和
`gpu_verification` 是本实验的 scratch 验证额外开销。输出为每个丢失 Rank
分别计算的 median/P95，不代表在线故障恢复或真实 KV 写回性能。

## 下一阶段：接入 CPU DRAM 在线保护

先在可访问 GPU 的隔离实验进程验证上述只读 hook，与无 hook 的固定输入/seed 输出对比，
收集全部 Rank 真实 K/V metadata + byte checks；再量化 gather/D2H/序列化的独立开销。
可先按 `TP stripe = 同 capture/layer/token range 的各 Rank shard` 做 CPU XOR，
parity 留 DRAM，异步落盘只作证据。生产保护需 versioned manifest、每槽 generation、
持有/复制生命周期协议、完成事件和有界 pinned buffer ring；不是把 synchronous
debug snapshot 放进每个 token 的调度路径。

下一步增量更新只能在全部 Rank 的同一逻辑 epoch 已提交后发布 parity，避免混合旧/新块；
定义 prefix 去重、部分页、取消/retract、迁移与模型更新语义，并记录保护滞后窗口。
异步复制必须正确关联 forward events，在 copy 完成前不得复用源 slots，避免 ABA。
先测 NUMA placement、PCIe/DRAM 带宽、backpressure 和 p99 serving 延迟，再做优化。
XOR 只能恢复一个 shard 丢失，单节点 CPU parity 不保护整 Node 故障；
跨 Node RS(k,m)、参数优化和线上故障请求重启均留给之后，不能以本实验替代。

## 本轮真实 KV 只读导出原型

本轮新增的唯一 SGLang 核心变更是
`python/sglang/srt/managers/scheduler_components/batch_result_processor.py`：
在 `process_batch_result_prefill`（当前约 `:270`）进入原有
`req.time_stats.set_prefill_finished_time()` 之前，检查
`SGLANG_KV_EC_DEBUG=1` 后调用 `sglang.srt.debug_utils.kv_ec.maybe_capture_prefill`。
默认环境变量不是 `1` 时完全不 import 实验包。bridge 捕获异常只写固定类型/原因日志，
不会跳过任何 distributed collective、改变采样或改变释放逻辑。

实验实现：`debug_capture.py`、`rank_snapshot.py`、`launch_debug.py`、
`request_debug.py`。不支持的模式在访问 pool 前 fail closed：单请求、普通 EXTEND、
完整 prompt、TP=1/4、单 node、无 overlap/prefix/chunked/mixed prefill、无 speculation、
无 PP/DP/DCP、无 offload/HiCache/Unified/page-major/量化/LoRA、page_size=1、
CUDA graph decode/prefill disabled、Qwen3-8B、普通 `MHATokenToKVPool`、NHD、BF16。

### 实际读取与完成依赖

每个 TP scheduler worker 使用自己的 `scheduler.tp_worker.model_runner.token_to_kv_pool`，
并核对 `scheduler.req_to_token_pool` 与 worker 引用、allocator 反向引用一致；
`scheduler.ps.tp_rank`、`ps.tp_size`、`ps.gpu_id` 写入 rank metadata。可见设备由
`CUDA_VISIBLE_DEVICES` 保存，不能把 visible ordinal 当 PCI 物理编号。

hook 的调用点已经处于 prefill 结果处理阶段，且原有 `result.copy_done.synchronize()`
（若存在）先完成；随后采集器执行 `torch.cuda.synchronize(pool.k_buffer[0].device)`，
再读取 `req_to_token[row, token_start:end]`。每层只对这些 physical slots 做
`get_kv_buffer(layer_id)` + `index_select(0, slots)`，然后同步 D2H 到独立 CPU bytes；
不复制整个 pool，不调用 `set_kv_buffer`。复制后复读选中 slots，检查 request row、
`req_generation` 与映射未变，最后才返回原处理器，让它执行 output append、prefix cache
或 `release_kv_cache`。因此调试模式下同一 scheduler 线程在复制期间不进入释放/复用路径。

`torch.cuda.synchronize` 是隔离调试模式的粗粒度完成依赖，涵盖 `set_kv_buffer` 可能
使用的 forward/辅助 stream；它不是生产 hook 的性能方案，也没有放入 forward 或 CUDA
Graph capture。`get_key_buffer/get_value_buffer` 的 layer-transfer wait 语义通过
明确要求 `pool.layer_transfer_counter is None` 而不是忽略；任何层级迁移直接拒绝。
这仍不是通用 allocator 锁：如果将来引入异步 scheduler/外部释放线程，必须增加真实
slot epoch/引用协议。当前 hook 只在明确的 non-overlap 单 scheduler 条件下声称安全。

`req_generation` 仅是请求行复用计数，捕获前后复查用于检测 ABA；当前源码没有物理
KV slot generation。因此 metadata 明确保存 `slot_generation: null`，不能把 row generation
当作物理槽位生命周期保护。

### 运行命令（必须由 GPU 服务器手动执行）

以下 launcher 只检查端口、不杀进程，使用隔离端口和显式 artificial request ID。
先启动 baseline，再启动 capture；两次使用相同 GPU/模型/参数和固定人工 token prompt。
baseline 只生成回归输出，不导出；capture 设置 hook。不要在已有服务相同端口上启动。

```bash
conda activate sglang-ec
cd /media/sda1/iic_data/luotao/sglangec/sglang
export KV_EC_OUT=/media/sda1/iic_data/luotao/kv_ec_runs

# TP=1, layer 0, first 128 tokens
python experiments/kv_ec/launch_debug.py --mode baseline --tp 1 --gpus 0 \
  --port 30281 --output "$KV_EC_OUT" --snapshot-id tp1-baseline --request-id kv-ec-artificial-request
python experiments/kv_ec/request_debug.py request --url http://127.0.0.1:30281 \
  --output "$KV_EC_OUT/tp1-baseline-output.json"
# stop this foreground server normally (Ctrl-C only for this isolated process)
python experiments/kv_ec/launch_debug.py --mode capture --tp 1 --gpus 0 \
  --port 30282 --output "$KV_EC_OUT" --snapshot-id tp1-layer0-128 --request-id kv-ec-artificial-request \
  --token-count 128 --layers 0
python experiments/kv_ec/request_debug.py request --url http://127.0.0.1:30282 \
  --output "$KV_EC_OUT/tp1-capture-output.json"
python experiments/kv_ec/request_debug.py compare \
  "$KV_EC_OUT/tp1-baseline-output.json" "$KV_EC_OUT/tp1-capture-output.json"
python experiments/kv_ec/rank_snapshot.py "$KV_EC_OUT/tp1-layer0-128"

# TP=4, same sequence, each Rank writes rank_0..rank_3; then offline XOR.
python experiments/kv_ec/launch_debug.py --mode capture --tp 4 --gpus 0,1,2,3 \
  --port 30284 --output "$KV_EC_OUT" --snapshot-id tp4-layer0-128 --request-id kv-ec-artificial-request \
  --token-count 128 --layers 0
python experiments/kv_ec/request_debug.py request --url http://127.0.0.1:30284 \
  --output "$KV_EC_OUT/tp4-capture-output.json"
python experiments/kv_ec/rank_snapshot.py "$KV_EC_OUT/tp4-layer0-128"

# After layer 0/128 succeeds, repeat with --token-count 256, then --layers all.
```

The launcher assumes this checkout's installed editable package is active. If `which python`
does not point to the intended conda environment, stop and activate it; do not change global
installation. `rank_snapshot.py` refuses missing ranks, unequal lengths, incompatible model/
request/capture metadata, incomplete sidecars or corrupted bytes. It creates `manifest.json`
and `recovery.json` only once; parity remains CPU memory during verification and is not silently
written as a fake rank. It reports all four lost-rank cases and K/V/checksum/shape checks.

TP=4 manifest 结构示例（值为示意，不能当作真实证据）：

```json
{
  "format": "kv-ec-rank-bin-v1",
  "snapshot_id": "tp4-layer0-128",
  "request_id": "sha256-of-artificial-rid",
  "model_identifier": "/media/sda1/iic_data/luotao/models/Qwen3-8B",
  "tp_size": 4,
  "layers": [0],
  "token_start": 0,
  "token_count": 128,
  "dtype": "torch.bfloat16",
  "packed_byte_count": 65536,
  "ranks": [{"tp_rank": 0, "req_pool_idx": 17, "req_generation": 3,
             "k_sha256": "...", "v_sha256": "...",
             "snapshot_completion_status": "complete"}],
  "snapshot_completion_status": "complete"
}
```

实际 manifest 会包含四个完整 rank sidecar；每个 sidecar 还记录 GPU/设备、物理 slots、
源 shape/stride、模型和源码 fingerprint，以及 K/V aggregate SHA256。

The capture hook itself creates `rank_R.bin` plus `rank_R.json`; `manifest.json` is generated
only after all ranks have arrived. No prompt text is logged or saved. `request_debug.py` stores
only artificial input token count/hash and output token IDs.

### Current evidence

The current tool session still reports `torch.cuda.is_available() == False` and `nvidia-smi`
cannot connect to the driver. Therefore no TP=1/TP=4 real worker log, rank binary, manifest,
or real-KV SHA256 exists yet. The new tests pass only synthetic/CPU fixtures and are labeled
`synthetic_fixture`; they are not evidence of C/D completion. Run the commands above on the
real GPU server and preserve stdout, per-rank JSON, `manifest.json`, `recovery.json`, and the
baseline/capture exact output comparison as the acceptance bundle.
