"""Default-off, process-local Stage E observations. Never synchronize CUDA here.

Set KV_EC_PERF_DIR and KV_EC_PERF_RUN_ID in *every* participating process.
JSONL files are per host/PID, outside correctness artifacts. Durations are
always measured in one process; inclusive spans must never be summed.
"""

from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid


def enabled():
    return bool(os.environ.get("KV_EC_PERF_DIR") and os.environ.get("KV_EC_PERF_RUN_ID"))


def emit(stage, category, *, start_ns=None, end_ns=None, duration_ms=None,
         source="monotonic_ns", status="ok", **fields):
    if not enabled():
        return
    end_ns = time.monotonic_ns() if end_ns is None else end_ns
    if duration_ms is None:
        duration_ms = (end_ns - start_ns) / 1e6
    record = dict(schema="kv-ec-perf-v1", run_id=os.environ["KV_EC_PERF_RUN_ID"],
                  host=socket.gethostname(), pid=os.getpid(), thread=threading.get_ident(),
                  stage=stage, category=category, source=source, status=status,
                  role=os.environ.get('KV_EC_PERF_ROLE', 'unspecified'),
                  start_ns=start_ns, end_ns=end_ns, duration_ms=duration_ms,
                  inclusive=True, **fields)
    # Observation failures must not alter A/B/C/D/E correctness or serving.
    try:
        root = Path(os.environ["KV_EC_PERF_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{record['host']}-{record['pid']}.jsonl"
        data = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except OSError:
        pass


@contextmanager
def span(stage, category, **fields):
    if not enabled():
        yield
        return
    started, status = time.monotonic_ns(), "ok"
    span_id = uuid.uuid4().hex
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        ended = time.monotonic_ns()
        emit(stage, category, start_ns=started, end_ns=ended, status=status,
             span_id=span_id, **fields)


def observed(stage, category):
    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            with span(stage, category):
                return fn(*args, **kwargs)
        return call
    return decorate
