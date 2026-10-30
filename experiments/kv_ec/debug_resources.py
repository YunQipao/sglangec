"""Read-only debug admission; no CUDA initialization or process management."""

import math

from .gpu_memory_budget import check_shared_gpu_budget, read_gpu_memory, read_gpu_tasks


def admit(gpus, model, tp, context, fraction, *, shared, minimum=0):
    if (not math.isfinite(minimum) or minimum < 0 or len(gpus) != tp or
            len(set(gpus)) != tp or not all(g.isdigit() for g in gpus)):
        raise ValueError('invalid_debug_gpu_selection_or_minimum')
    rows, tasks = read_gpu_memory(), read_gpu_tasks()
    for gpu in gpus:
        if gpu not in rows:
            raise ValueError(f'GPU {gpu}: physical device not found')
        row = rows[gpu]
        if row['free_bytes'] < minimum * 1024**3:
            raise ValueError(f'GPU {gpu} ({row["uuid"]}): free_bytes={row["free_bytes"]} '
                             f'below min_free_gpu_gb={minimum}')
        if not shared and tasks.get(row['uuid']):
            raise ValueError(f'GPU {gpu} ({row["uuid"]}): STRICT rejects existing compute tasks')
    # SGLang sizes a common TP KV pool from MIN free memory, never the mean.
    return check_shared_gpu_budget(gpus, model, tp, context, fraction,
        memory_rows=rows, tasks_by_uuid=tasks, actual_free_static=shared)
