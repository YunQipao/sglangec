"""Stage E resource policy only; never changes recovery correctness gates."""

import os
import re


def mode(value):
    if type(value) is not str or value not in ('strict', 'shared'):
        raise ValueError('GPU_ADMISSION_MODE_INVALID: expected strict or shared')
    return value


def config_mode(cfg):
    value = cfg.get('gpu_admission', {'mode': 'strict'})
    if type(value) is not dict or set(value) != {'mode'}:
        raise ValueError('GPU_ADMISSION_CONFIG_INVALID: expected {"mode": "strict|shared"}')
    selected = mode(value['mode'])
    for node in cfg.get('nodes', []):
        if 'gpu_admission' in node or 'gpu_admission_mode' in node:
            raise ValueError('GPU_ADMISSION_MODE_MISMATCH: policy must be top-level only')
    return selected


def args_mode(args):
    selected = getattr(args, 'gpu_admission_mode', None)
    if selected is None:
        selected = 'shared' if getattr(args, 'allow_shared_gpu_debug', False) else 'strict'
    selected = mode(selected)
    expected = os.environ.get('KV_EC_GPU_ADMISSION_MODE')
    if expected is not None and mode(expected) != selected:
        raise ValueError('GPU_ADMISSION_MODE_MISMATCH: coordinator/launcher')
    return selected


def environment(selected):
    selected = mode(selected)
    # Explicitly override accidental parent settings, including the old alias.
    return dict(KV_EC_GPU_ADMISSION_MODE=selected,
                SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK='1' if selected == 'strict' else '0',
                SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK='0' if selected == 'strict' else '1')


def metadata(selected, fraction, resources=None):
    selected = mode(selected)
    return dict(gpu_admission_mode=selected, mem_fraction_static=fraction,
                tp_memory_imbalance_check=selected == 'strict',
                shared_resource_environment=selected == 'shared',
                isolated_performance_eligible=False,  # only set after successful live evidence
                gpu_resources_before_start=resources or [])


def validate_evidence(record, selected, tp_size=None):
    if (record.get('gpu_admission_mode') != selected or
            record.get('tp_memory_imbalance_check') is not (selected == 'strict') or
            record.get('shared_resource_environment') is not (selected == 'shared')):
        raise ValueError('GPU_ADMISSION_MODE_MISMATCH: startup evidence')
    rows = record.get('gpu_resources_before_start')
    tp_size = record.get('tp_size', 2) if tp_size is None else tp_size
    if tp_size not in (2, 4) or record.get('tp_size', 2) != tp_size:
        raise ValueError('GPU_ADMISSION_TP_SIZE_MISMATCH')
    if not isinstance(rows, list) or len(rows) != tp_size:
        raise ValueError('GPU_ADMISSION_EVIDENCE_MISSING: expected complete TP resource snapshot')
    if len({row['gpu_index'] for row in rows}) != tp_size or len({row['gpu_uuid'] for row in rows}) != tp_size:
        raise ValueError('GPU_ADMISSION_EVIDENCE_INVALID: duplicate GPU')
    for row in rows:
        if (not 0 < row['required_estimated_memory_bytes'] <= row['free_memory_bytes'] <= row['total_memory_bytes'] or
                not isinstance(row['other_tasks'], list) or
                (selected == 'strict' and row['other_tasks'])):
            raise ValueError('GPU_ADMISSION_EVIDENCE_INVALID: insufficient/busy GPU')


def normalized_uuid(value):
    return str(value).lower().removeprefix('gpu-').replace('-', '')


def _valid_uuid(value):
    # Missing/placeholder identities must never compare equal to one another.
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{32}', normalized_uuid(value)) is not None


def runtime_rank_binding(scheduler, resources):
    """Bind the actual CUDA KV pool to the admitted physical device per rank."""
    import torch
    ps = scheduler.model_worker.ps
    device = scheduler.model_worker.model_runner.token_to_kv_pool.k_buffer[0].device
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    rank, tp = int(ps.tp_rank), int(ps.tp_size)
    if (tp not in (2, 4) or len(resources) != tp or len(visible) != tp or
            device.type != 'cuda' or device.index != rank or int(ps.gpu_id) != rank):
        raise ValueError('GPU_RANK_DEVICE_BINDING_MISMATCH')
    row = resources[rank]
    actual_uuid = str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable'))
    if (visible[rank] != row['gpu_index'] or
            not _valid_uuid(actual_uuid) or not _valid_uuid(row.get('gpu_uuid')) or
            normalized_uuid(actual_uuid) != normalized_uuid(row['gpu_uuid'])):
        raise ValueError('GPU_RANK_UUID_BINDING_MISMATCH')
    return dict(tp_rank=rank, gpu_index=row['gpu_index'], gpu_uuid=row['gpu_uuid'],
                cuda_device=str(device), runtime_gpu_uuid=actual_uuid)


def validate_rank_binding(binding, record, rank):
    """Independent checker for new reports; legacy archives remain readable."""
    rows = record['gpu_resources_before_start']
    selected = record['gpus'].split(',')
    expected = rows[rank]
    if (binding.get('tp_rank') != rank or binding.get('gpu_index') != selected[rank] or
            binding.get('gpu_index') != expected['gpu_index'] or
            binding.get('gpu_uuid') != expected['gpu_uuid'] or
            not _valid_uuid(expected.get('gpu_uuid')) or
            not _valid_uuid(binding.get('runtime_gpu_uuid')) or
            binding.get('cuda_device') != f'cuda:{rank}' or
            normalized_uuid(binding.get('runtime_gpu_uuid')) != normalized_uuid(expected['gpu_uuid'])):
        raise ValueError('GPU_RANK_DEVICE_BINDING_MISMATCH')


def capture_rank_binding(scheduler, record):
    """Observe the actual KV pool device at capture, not just at startup."""
    binding = runtime_rank_binding(scheduler, record['gpu_resources_before_start'])
    rank = binding['tp_rank']
    validate_rank_binding(binding, record, rank)
    return dict(tp_rank=rank, tp_size=int(scheduler.model_worker.ps.tp_size),
                gpu_id=int(scheduler.model_worker.ps.gpu_id),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cuda_device=binding['cuda_device'], gpu_uuid=binding['runtime_gpu_uuid'],
                physical_gpu_index=binding['gpu_index'])


def validate_capture_rank_binding(captured, binding, record, rank):
    """Return every failed comparison; all identities are local to this node."""
    validate_rank_binding(binding, record, rank)
    expected = dict(tp_rank=rank, tp_size=record.get('tp_size', 2), gpu_id=rank,
                    cuda_visible_devices=record['gpus'], cuda_device=f'cuda:{rank}',
                    physical_gpu_index=binding['gpu_index'], gpu_uuid=binding['gpu_uuid'])
    failures = {}
    for field, value in expected.items():
        actual = captured.get(field)
        matches = (actual == value if field != 'gpu_uuid' else
                   _valid_uuid(actual) and normalized_uuid(actual) == normalized_uuid(value))
        if not matches:
            failures[field] = dict(actual=actual, expected=value)
    return failures
