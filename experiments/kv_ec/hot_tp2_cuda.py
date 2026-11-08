"""Default-off bounded CUDA scratch adapters, not live SGLang KV pool writers.

Prepare allocations outside the fault clock. Keep single-owner access until
finish/abort; no graph, NCCL, allocator lease, model or SSE integration claimed.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np

from .hot_tp2 import Options, Rejected, sha


def _torch(options, device):
    options.require()
    import torch
    d = torch.device(device)
    if d.type != "cuda" or not torch.cuda.is_available():
        raise Rejected("real_cuda_required")
    if d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return torch, d


@dataclass
class _Slot:
    host: object
    readback: object
    expected_gpu: object
    unequal_gpu: object
    bad_gpu: object
    bad_host: object
    written: object
    validated: object
    index: int = -1
    size: int = 0


class VerifiedCudaSink:
    """Persistent scratch target + bounded H2D/GPU-compare/D2H validation ring.

    Full byte compare is enqueued on validation_stream independently of H2D.
    Full D2H SHA-256 additionally authenticates the target against the immutable
    checkpoint; CPU hashing occurs only AFTER the slot's CUDA validation event.
    These costs are included in recover()'s verified-ready boundary.
    """

    def __init__(self, cp, lost_rank, *, device="cuda:0", options=Options()):
        cp.validate(options)
        if type(lost_rank) is not int or lost_rank not in (0, 1):
            raise Rejected("invalid_lost_rank")
        torch, self.device = _torch(options, device)
        self.torch, self.cp, self.lost_rank = torch, cp, lost_rank
        size = min(cp.chunk_bytes, cp.total_bytes)
        self.slots = []
        with torch.cuda.device(self.device):
            self.target = torch.empty(cp.total_bytes, dtype=torch.uint8, device=self.device)
            self.write_stream = torch.cuda.Stream(device=self.device)
            self.validation_stream = torch.cuda.Stream(device=self.device)
            self.ready = torch.cuda.Event()
            for _ in range(options.depth):
                self.slots.append(_Slot(
                    torch.empty(size, dtype=torch.uint8, pin_memory=True),
                    torch.empty(size, dtype=torch.uint8, pin_memory=True),
                    torch.empty(size, dtype=torch.uint8, device=self.device),
                    torch.empty(size, dtype=torch.bool, device=self.device),
                    torch.empty((), dtype=torch.bool, device=self.device),
                    torch.empty((), dtype=torch.bool, pin_memory=True),
                    torch.cuda.Event(), torch.cuda.Event()))
        self.next_index, self.closed, self.verified = 0, False, False

    def _drain(self, slot):
        if slot.index < 0:
            return
        slot.validated.synchronize()
        expected = self.cp.chunks[slot.index].data_sha[self.lost_rank]
        if slot.bad_host.item() or sha(slot.readback.numpy()[:slot.size]) != expected:
            raise Rejected("full_cuda_destination_validation_failed")
        slot.index = -1

    def stage(self, index, payload):
        if self.closed or index != self.next_index or type(payload) is not bytes:
            raise Rejected("cuda_sink_closed_or_out_of_order")
        chunk = self.cp.chunks[index]
        if len(payload) != len(chunk.parity):
            raise Rejected("cuda_chunk_size_mismatch")
        slot = self.slots[index % len(self.slots)]
        self._drain(slot)  # backpressure; no pinned reuse until validation ends
        n, start, torch = len(payload), index * self.cp.chunk_bytes, self.torch
        np.copyto(slot.host.numpy()[:n], np.frombuffer(payload, dtype=np.uint8))
        slot.index, slot.size = index, n
        with torch.cuda.device(self.device), torch.cuda.stream(self.write_stream):
            self.target[start:start + n].copy_(slot.host[:n], non_blocking=True)
            slot.written.record(self.write_stream)
        with torch.cuda.device(self.device), torch.cuda.stream(self.validation_stream):
            self.validation_stream.wait_event(slot.written)
            slot.expected_gpu[:n].copy_(slot.host[:n], non_blocking=True)
            torch.ne(self.target[start:start + n], slot.expected_gpu[:n], out=slot.unequal_gpu[:n])
            torch.any(slot.unequal_gpu[:n], out=slot.bad_gpu)
            slot.bad_host.copy_(slot.bad_gpu, non_blocking=True)
            slot.readback[:n].copy_(self.target[start:start + n], non_blocking=True)
            slot.validated.record(self.validation_stream)
        self.next_index += 1

    def finish(self):
        if self.closed or self.next_index != len(self.cp.chunks):
            raise Rejected("cuda_sink_incomplete")
        for slot in self.slots:
            self._drain(slot)
        self.ready.record(self.validation_stream)
        self.ready.synchronize()
        self.closed, self.verified = True, True
        return True

    def wait_before_submit(self, scheduler_stream):
        if not self.verified or scheduler_stream.device != self.device:
            raise Rejected("scheduler_requires_full_validation")
        scheduler_stream.wait_event(self.ready)
        # This orders streams only, and does not authorize a Scheduler forward.

    def abort(self):
        self.closed, self.verified = True, False
        self.write_stream.synchronize()
        self.validation_stream.synchronize()
        # Do not release partially written target to Scheduler or an allocator.


class LowPriorityCapture:
    """Bounded contiguous byte capture with caller-owned source lifetime lease.

    validate_lease() must raise on stale row/slot/version before/after D2H.
    It is an adapter contract, not a substitute for real Scheduler pinning.
    This helper deliberately accepts no arbitrary pool layout or slot gather.
    """

    def __init__(self, *, device="cuda:0", options=Options()):
        self.torch, self.device = _torch(options, device)
        self.options = options
        self.width = options.chunk_mib * 1024**2
        with self.torch.cuda.device(self.device):
            # CUDA priority 0 is normal/lowest priority, negative is higher.
            self.stream = self.torch.cuda.Stream(device=self.device, priority=0)
        self.active = False

    def capture(self, source, consume, *, validate_lease):
        torch = self.torch
        if (self.active or not source.is_cuda or source.device != self.device
                or source.dtype != torch.uint8 or source.ndim != 1 or not source.is_contiguous()
                or not 0 < source.numel() <= self.options.max_bytes):
            raise Rejected("capture_requires_exclusive_contiguous_cuda_bytes")
        validate_lease()
        pending = deque()
        width = min(self.width, source.numel())
        buffers = [torch.empty(width, dtype=torch.uint8, pin_memory=True)
                   for _ in range(self.options.depth)]
        events = [torch.cuda.Event() for _ in buffers]
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(producer)
        self.active = True

        def drain():
            index, slot, n = pending.popleft()
            events[slot].synchronize()
            validate_lease()
            consume(index, buffers[slot].numpy()[:n].tobytes())

        try:
            for index, start in enumerate(range(0, source.numel(), width)):
                if len(pending) == len(buffers):
                    drain()
                validate_lease()
                slot, n = index % len(buffers), min(width, source.numel() - start)
                with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                    buffers[slot][:n].copy_(source[start:start + n], non_blocking=True)
                    events[slot].record(self.stream)
                pending.append((index, slot, n))
            while pending:
                drain()
            validate_lease()
        finally:
            self.stream.synchronize()  # buffers/source live until all D2H finishes
            self.active = False
        return dict(chunks=(source.numel() + width - 1) // width,
                    pinned_bytes=width * len(buffers), stream_priority=self.stream.priority,
                    gpudirect=False, scheduler_lease_integrated=False)
