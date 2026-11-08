"""Explicit prebuilt native-library adapter; no import-time JIT or CUDA work."""

import ctypes
from pathlib import Path

from .hot_tp2 import Rejected, _bytes, rs_chunk

OPS = {"encode": 0, "recover0": 1, "recover1": 2, "xor": 3}


def require_p2p(target, source, *, enabled=False, capability):
    if (enabled is not True or type(target) is not int or type(source) is not int
            or target < 0 or source < 0 or target == source
            or capability(target, source) is not True):
        raise Rejected("p2p_unavailable_use_explicit_cpu_pinned_fallback")


class NativeCodec:
    def __init__(self, library, *, enabled=False):
        if enabled is not True:
            raise Rejected("native_rs_disabled")
        path = Path(library)
        if not path.is_absolute() or not path.is_file():
            raise Rejected("explicit_prebuilt_native_library_required")
        self.lib = ctypes.CDLL(str(path))
        self.cpu = self.lib.hot_rs21_cpu
        self.cpu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t, ctypes.c_int]
        self.cpu.restype = ctypes.c_int
        # Validate the library before the fault boundary, all byte values/tails.
        a, b = bytes(range(256)) + b"abc", bytes(reversed(range(256))) + b"def"
        for op in ("encode", "recover0", "recover1"):
            if self(a, b, op) != rs_chunk(a, b, op):
                raise Rejected("native_cpu_preflight_mismatch")

    def __call__(self, a, b, operation):
        _bytes(a)
        _bytes(b, len(a))
        if operation not in OPS:
            raise Rejected("unknown_rs_operation")
        out = ctypes.create_string_buffer(len(a))
        if self.cpu(ctypes.c_char_p(a), ctypes.c_char_p(b), out, len(a), OPS[operation]):
            raise Rejected("native_cpu_codec_failed")
        return out.raw

    def cuda_into(self, a, b, target, *, operation, stream, allow_p2p=False):
        """Launch fused a/op(b) directly into target; returns a lifetime ticket.

        Call ticket['done'].synchronize() before dropping it. Independent full
        destination validation is still REQUIRED before any scheduler submit.
        """
        import torch
        if operation not in OPS or not torch.cuda.is_available():
            raise Rejected("cuda_codec_unavailable")
        tensors = (a, b, target)
        if any(not t.is_cuda or t.dtype != torch.uint8 or t.ndim != 1
               or not t.is_contiguous() or t.numel() != target.numel()
               for t in tensors) or target.numel() <= 0:
            raise Rejected("contiguous_equal_cuda_uint8_required")
        if stream.device != target.device:
            raise Rejected("target_stream_device_mismatch")
        # No aliasing, including overlapping views into the same allocation.
        for t in (a, b):
            if (t.device == target.device and t.data_ptr() < target.data_ptr() + target.numel()
                    and target.data_ptr() < t.data_ptr() + t.numel()):
                raise Rejected("target_must_not_alias_survivor_or_parity")
        try:
            kernel = self.lib.hot_rs21_cuda
            peer = self.lib.hot_rs21_enable_peer
        except AttributeError as exc:
            raise Rejected("native_library_not_cuda_built") from exc
        kernel.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        kernel.restype = ctypes.c_int
        peer.argtypes, peer.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
        events = []
        with torch.cuda.device(target.device):
            for t in (a, b):
                if t.device != target.device:
                    require_p2p(target.device.index, t.device.index, enabled=allow_p2p,
                                capability=torch.cuda.can_device_access_peer)
                    if peer(target.device.index, t.device.index):
                        raise Rejected("p2p_enable_failed_use_explicit_cpu_pinned_fallback")
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(t.device))
                stream.wait_event(event)
                events.append(event)
            # Event ordering protects any earlier target use on its owner stream.
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(target.device))
            stream.wait_event(event)
            events.append(event)
            status = kernel(a.data_ptr(), b.data_ptr(), target.data_ptr(), target.numel(),
                            OPS[operation], stream.cuda_stream)
            if status:
                stream.synchronize()
                raise Rejected(f"native_cuda_launch_failed_{status}")
            done = torch.cuda.Event()
            done.record(stream)
        return dict(done=done, keepalive=(a, b, target, *events), gpudirect=False,
                    validation_complete=False)
