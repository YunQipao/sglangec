"""Read-only per-GPU memory admission for explicitly shared debug runs."""

import json
from pathlib import Path
import subprocess

MIB = 1024 * 1024
GIB = 1024 ** 3


def parse_gpu_rows(text):
    rows = {}
    for line in text.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise ValueError("invalid_nvidia_smi_memory_row")
        index, uuid, total, free = fields
        rows[index] = dict(index=index, uuid=uuid, total_bytes=int(total) * MIB,
                           free_bytes=int(free) * MIB)
    return rows


def read_gpu_memory():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
    return parse_gpu_rows(result)


def read_gpu_tasks():
    result = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT)
    tasks = {}
    for line in result.splitlines():
        fields = [part.strip() for part in line.split(",", 2)]
        if len(fields) == 3:
            tasks.setdefault(fields[0], []).append(dict(pid=fields[1], name=fields[2]))
    return tasks


def model_weight_bytes(model):
    """Require an exact local safetensors index; fail closed if unavailable."""
    path = Path(model)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        metadata = json.loads(index.read_text()).get("metadata", {})
        size = metadata.get("total_size")
        if type(size) is int and size > 0:
            return size
    # A single-file safetensors model has an exact file size, including its
    # small header; the estimate intentionally errs slightly high.
    files = list(path.glob("*.safetensors")) if path.is_dir() else []
    if files:
        return sum(item.stat().st_size for item in files)
    raise ValueError("shared_gpu_weight_size_unavailable")


def estimate_required_bytes(model, tp, context_length, total_bytes, mem_fraction_static):
    if not (1 <= tp <= 4 and context_length >= 1
            and 0 < mem_fraction_static <= 1):
        raise ValueError("invalid_gpu_memory_budget_inputs")
    config_path = Path(model) / "config.json"
    config = json.loads(config_path.read_text())
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]))
    # BF16 K+V, conservatively assume KV heads are not sharded when TP is
    # larger than their count; the supported Qwen3-8B TP4 path shards them.
    kv_bytes = layers * 2 * context_length * kv_heads * head_dim * 2
    weight_bytes = model_weight_bytes(model)
    weight_per_rank = (weight_bytes + tp - 1) // tp
    # Explicit headroom: weight loading/fragmentation (+20%), CUDA context,
    # batch-1 prefill/decode temporaries, plus one protected chunk and recovery
    # staging. Do not count a margin as usable KV capacity.
    weight_with_margin = (weight_per_rank * 120 + 99) // 100
    fixed_runtime = 2 * GIB + 2 * GIB + 512 * 1024 * 1024
    estimated = weight_with_margin + fixed_runtime + kv_bytes * 2
    # SGLang's static fraction governs weights+KV pool against device capacity.
    requested_static = int(total_bytes * mem_fraction_static)
    minimum_static = weight_per_rank + kv_bytes
    estimated = max(estimated, requested_static + fixed_runtime)
    return estimated, dict(
        model_weights_per_rank_bytes=weight_per_rank,
        cuda_context_reserve_bytes=2 * GIB,
        compute_temporary_reserve_bytes=2 * GIB,
        ec_snapshot_restore_reserve_bytes=512 * 1024 * 1024,
        kv_context_estimate_bytes=kv_bytes,
        mem_fraction_static_reserve_bytes=requested_static,
        minimum_static_for_weights_and_context_bytes=minimum_static,
        safety_margin_percent=20)


def check_shared_gpu_budget(gpus, model, tp, context_length, mem_fraction_static,
                            *, memory_rows=None, tasks_by_uuid=None, report=None):
    """Inspect physical nvidia-smi indices; optionally report before admission.

    ``report`` receives all selected GPUs, including their estimates, before
    any memory rejection. No CUDA initialization or process control is used.
    """
    memory_rows = read_gpu_memory() if memory_rows is None else memory_rows
    tasks_by_uuid = read_gpu_tasks() if tasks_by_uuid is None else tasks_by_uuid
    accepted = []
    for index in gpus:
        row = memory_rows.get(str(index))
        if row is None:
            raise ValueError(f"INSUFFICIENT_GPU_MEMORY GPU index={index} reason=GPU_NOT_FOUND")
        required, details = estimate_required_bytes(
            model, tp, context_length, row["total_bytes"], mem_fraction_static)
        accepted.append(dict(
            gpu_index=str(index), gpu_uuid=row["uuid"],
            total_memory_bytes=row["total_bytes"],
            used_memory_bytes=row["total_bytes"] - row["free_bytes"],
            free_memory_bytes=row["free_bytes"], required_estimated_memory_bytes=required,
            other_tasks=tasks_by_uuid.get(row["uuid"], []), estimate=details))
    if report is not None:
        report(accepted)
    for item in accepted:
        index = item["gpu_index"]
        details = item["estimate"]
        required = item["required_estimated_memory_bytes"]
        free = item["free_memory_bytes"]
        if details["mem_fraction_static_reserve_bytes"] < details[
                "minimum_static_for_weights_and_context_bytes"]:
            raise ValueError(
                "INSUFFICIENT_GPU_MEMORY "
                f"GPU index={index} free_memory_bytes={free} "
                f"required_estimated_memory_bytes={details['minimum_static_for_weights_and_context_bytes']} "
                "rejection_reason=mem_fraction_static_below_weights_plus_context_kv")
        if free < required:
            raise ValueError(
                "INSUFFICIENT_GPU_MEMORY "
                f"GPU index={index} free_memory_bytes={free} "
                f"required_estimated_memory_bytes={required} "
                "rejection_reason=per_gpu_free_memory_below_estimate")
    return accepted
