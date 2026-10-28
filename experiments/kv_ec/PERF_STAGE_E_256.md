# Stage E 256-token TP2 性能基线

本方案只接受两台物理机上的真实 Stage E PASS。默认观测关闭；启用时只
设置 `KV_EC_PERF_DIR`、`KV_EC_PERF_RUN_ID` 和 `KV_EC_PERF_ROLE`，不会改变
Scheduler、GPU KV 写回、RS 或 RDMA 数据通路。每个进程使用自己的
`monotonic_ns`；GPU H2D/读回和 native verbs 使用已有 CUDA Event/CQ 统计。
不同主机的时间戳不相减，`inclusive` 区域不相加。

## 计时字段

`perf-*.jsonl` 中的 `stage/category/source` 对应：

| 指标 | 进程/主机 | 计时来源 |
|---|---|---|
| `gpu_kv_snapshot`, `gpu_snapshot_gather` | origin Scheduler，232/233 | 本进程 monotonic；已有 CUDA stream fence |
| `gpu_to_host` | origin TP rank，232/233 | CUDA Event |
| `local_xor_encode`, `local_xor_validation` | OnlineChunkManager，源节点 | monotonic；validation 单列 |
| `rs_upload_inclusive`, native `rdma_native` | exporter client，232/233 | monotonic 外围；verbs CQ 原生字段 |
| `rs_parity_encode`, `global_committed` | RS server，233 | server monotonic；GLOBAL_COMMITTED 是状态标记 |
| `rs_recovery_rpc_inclusive`, native recovery legs | 232 client/233 server | monotonic 外围；四条 CQ 记录 |
| `delivery_send_inclusive`, `delivery_receive_inclusive` | 232/233 | monotonic 外围；verbs CQ |
| `gpu_kv_writeback` | target TP rank，233 | monotonic 外围；已有 CUDA Event |
| `gpu_layer_readback_validation` | target TP rank，233 | monotonic；实验校验单列 |
| `scheduler_request_rebuild` | target Scheduler，233 | monotonic |
| `first_restored_decode_cuda` | target TP rank，233 | CUDA Event |
| `model_weight_load` | target ModelRunner，233 | 已有 `weight_load_time` |

观测异常只写日志并被忽略；不会让正常服务失败。现有
`stats.h2d_ms/verify_ms`、`rank_performance_ms`、`server_rs_encode_ms` 和
`native_stats` 是原始数据，报告不得把校验耗时伪装成必要开销。

## 对照组

`perf_benchmark inference --scheme no-ec` 使用同一 Qwen3-8B、256 token、TP2、
采样参数和 17 个输出；服务必须关闭 radix cache、overlap、chunked prefill，
CUDA graph disabled。`reprefill` 是无 EC 的故障后重新 Prefill 控制，启动另一个
同配置服务并重复提交同一输入；其 CLI 明确标注不包含故障发现时间。

`xor-only` 接受真实 GPU Capture，测量受控单 TP rank 丢失的局部 XOR；它不声称
GPU/Scheduler 恢复。`rs-only` 应使用现有 Stage C/`physical_two_node_rs.py` 的
Node Block 故障入口；若没有真实 Stage C PASS，结果必须保持 NOT IMPLEMENTED。
`two-level` 仅由 Stage E `check` 通过的真实运行产生。

## 232/233 命令

在仓库 `sglang` 目录中，先准备一个不覆盖历史运行的 JSON 配置（路径和 GPU
按现场修改）：

```json
{
  "repo": "/media/sda1/iic_data/luotao/sglangec/sglang",
  "python": "/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python",
  "model": "/media/sda1/iic_data/models/Qwen3-8B",
  "prefix_tokens": 256, "tp_size": 2, "rs": [2, 1], "resource_interval": 1.0,
  "remote_root": "/tmp/kv-ec-perf",
  "return_host": "192.168.2.243", "return_port": 30890, "delivery_port": 30892,
  "target_port": 30986,
  "rs_server": {"host": "192.168.2.245", "port": 30700},
  "nodes": [
    {"ssh": "iic@10.102.0.232", "gpus": "0,1", "origin_port": 30984,
     "gid_index": 5, "rdma_device": "mlx5_0", "template": "/tmp/real-node0-template/global_node.bin",
     "input_ids": "/tmp/input_ids.json", "repo": "/media/sda1/iic_data/luotao/sglangec/sglang",
     "python": "/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python",
     "model": "/media/sda1/iic_data/models/Qwen3-8B"},
    {"ssh": "iic@10.102.0.233", "gpus": "0,1", "origin_port": 30985,
     "gid_index": 3, "rdma_device": "mlx5_0", "template": "/tmp/real-node1-template/global_node.bin",
     "input_ids": "/tmp/input_ids.json", "repo": "/media/sda1/iic_data/luotao/sglangec/sglang",
     "python": "/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python",
     "model": "/media/sda1/iic_data/models/Qwen3-8B"}
  ]
}
```

RS server must already be running in a dedicated 233 terminal using the Stage E
document's native-RDMA command. The coordinator never stops it. On 232:

```bash
cd /media/sda1/iic_data/luotao/sglangec/sglang
export PYTHONPATH="$PWD/python:$PWD"
RUN=/tmp/kv-ec-perf-$(date +%Y%m%d-%H%M%S); mkdir "$RUN"
python -m experiments.kv_ec.perf_benchmark repeat \
  --config "$RUN/config.json" --output "$RUN/two-level-r5" --repeat 5 --resource-interval 1
python -m experiments.kv_ec.perf_benchmark collect \
  --roots "$RUN/two-level-r5" --output "$RUN/summary.json"
```

执行 10 次时只将 `--repeat 5` 改为 `--repeat 10` 并使用新目录。配置文件
内容必须与上面一致；`repeat` 每次创建唯一 trial/run ID。服务、日志、JSONL
和 verdict 都位于该 RUN 下；失败运行保留，不会覆盖历史记录。每次 trial 的
232 目录需有 `acceptance.json`、`stage-e-check.json`、`native-metrics.json`；
233 的 JSON 证据通过 SCP 传回，KV payload 不通过 SCP。

汇总仅在样本数达到阈值时输出 P50/P95/P99，否则为 null；同时输出均值、样本
标准差、资源快照和被拒样本。保护额外开销比例只有在同一模型/输入/资源条件
下有匹配 EC-on/off 在线样本时才能计算，当前摘要明确标记为 NOT COMPUTED。

本机无 CUDA、RDMA 或 CPU mock 的结果不进入上述 two-level 统计。Stage E 真实
PASS 之前，`repeat` 和 `collect` 会拒绝该实验，不会虚构结果。
