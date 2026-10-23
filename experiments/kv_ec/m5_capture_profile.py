"""Default-off capture tracing; no tensors or request contents are serialized."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import itertools
import json
import os
from pathlib import Path

import torch

from .capture_timing import CaptureTiming

ENV = "SGLANG_KV_EC_CAPTURE_PROFILE_DIR"
_ACTIVE = ContextVar("kv_ec_capture_profile", default=None)
_SEQUENCE = itertools.count()


@contextmanager
def _timed_stage(timing, name):
    with timing.region(name), torch.profiler.record_function("kv_ec.capture." + name):
        yield


def stage(name):
    # Outside an admitted profiling run, do not enable the PyTorch profiler.
    timing = _ACTIVE.get()
    return _timed_stage(timing, name) if timing is not None else nullcontext()


def profile_capture(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        directory = os.environ.get(ENV)
        if not directory:
            return function(*args, **kwargs)
        root = Path(directory)
        if not root.is_absolute():
            raise ValueError("capture_profile_requires_absolute_directory")
        root.mkdir(parents=True, exist_ok=True)
        prefix = root / f"capture-{os.getpid()}-{next(_SEQUENCE)}"
        path = prefix.with_suffix(".trace.json")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # No record_shapes, stacks, tensor values, memory dumps or raw KV.
        # Kineto device memcpy activities, not host .cpu() spans, measure D2H.
        with torch.profiler.profile(activities=activities) as profiler:
            timing = CaptureTiming(detailed=True)
            token = _ACTIVE.set(timing)
            try:
                with torch.profiler.record_function("kv_ec.capture.total"):
                    result = function(*args, **kwargs)
                metrics = timing.finish()
                metrics.update(
                    tp_rank=getattr(args[1], "tp_rank", None) if len(args) > 1 else None,
                    snapshot_id=kwargs.get("snapshot_id"),
                    token_start=kwargs.get("token_start", 0),
                    mode=os.environ.get("SGLANG_KV_EC_EXPERIMENT_MODE"))
            finally:
                _ACTIVE.reset(token)
        # Collection/export overhead is outside capture.total but still blocks
        # the foreground gate. Profiled latency is not optimized inference.
        profiler.export_chrome_trace(str(path))
        with prefix.with_suffix(".timing.json").open("x") as stream:
            json.dump(metrics, stream, indent=2)
        return result
    return wrapped
