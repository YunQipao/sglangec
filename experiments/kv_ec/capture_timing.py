"""Exclusive CPU wall regions for the opt-in snapshot hook; no CUDA imports."""

from contextlib import contextmanager
import time


class CaptureTiming:
    def __init__(self, detailed=False):
        self.detailed = detailed
        self.wall = {}
        self.thread_cpu = {}
        self.layers = {}
        self._active = False
        self.started = time.perf_counter_ns()

    @contextmanager
    def region(self, name, layer=None):
        if self._active:
            raise RuntimeError("nested capture timing would double count")
        self._active = True
        cpu = time.thread_time_ns() if self.detailed else 0
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = (time.perf_counter_ns() - started) / 1e6
            self.wall[name] = self.wall.get(name, 0.0) + elapsed
            if self.detailed:
                self.thread_cpu[name] = self.thread_cpu.get(name, 0.0) + (time.thread_time_ns() - cpu) / 1e6
                if layer is not None:
                    record = self.layers.setdefault(str(layer), {})
                    record[name] = record.get(name, 0.0) + elapsed
            self._active = False

    def finish(self):
        total = (time.perf_counter_ns() - self.started) / 1e6
        accounted = sum(self.wall.values())
        residual = total - accounted
        if residual < -1e-6:
            raise RuntimeError("capture timings overlap")
        return {"wall_regions_ms": dict(self.wall),
                "thread_cpu_regions_ms": dict(self.thread_cpu),
                "layer_wall_ms": dict(self.layers),
                "scheduler_capture_ms": total, "accounted_ms": accounted,
                "unaccounted_ms": max(0.0, residual)}
