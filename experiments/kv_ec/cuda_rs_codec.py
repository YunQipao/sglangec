"""CUDA GF(2^8) RS(2,1) codec with a single fused transform kernel.

Triton is used because the deployment PyTorch image has a CUDA runtime but no
standalone ``nvcc``.  The checked-in native/hot_rs21.cu remains the prebuilt
C++/CUDA alternative.  ``auto`` falls back to the vectorized CPU oracle only
when a CUDA device or Triton is unavailable and reports the selected backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _rs21_kernel(a, b, out, n: tl.constexpr, OP: tl.constexpr,
                     BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        x = tl.load(a + i, mask=mask, other=0).to(tl.uint8)
        y = tl.load(b + i, mask=mask, other=0).to(tl.uint8)
        if OP == 2:  # inverse of multiply-by-two
            z = x ^ y
            value = (z >> 1) ^ ((z & 1) * 0x8E)
        elif OP == 3:  # local TP XOR recovery
            value = x ^ y
        else:  # encode and recover node 0: x XOR 2*y
            value = x ^ ((y << 1) & 0xFF) ^ ((y >> 7) * 0x1D)
        tl.store(out + i, value.to(tl.uint8), mask=mask)

    @triton.jit
    def _xor_scatter_kernel(a, b, pool, slots, n: tl.constexpr,
                            BYTES_PER_TOKEN: tl.constexpr,
                            BLOCK: tl.constexpr):
        """P xor survivor directly into non-contiguous SGLang KV slots."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        token = i // BYTES_PER_TOKEN
        within = i - token * BYTES_PER_TOKEN
        slot = tl.load(slots + token, mask=mask, other=0).to(tl.int64)
        value = tl.load(a + i, mask=mask, other=0).to(tl.uint8) ^ tl.load(
            b + i, mask=mask, other=0
        ).to(tl.uint8)
        tl.store(pool + slot * BYTES_PER_TOKEN + within, value, mask=mask)

except (ImportError, OSError):  # CPU-only service hosts remain supported.
    torch = triton = tl = None


OPS = {"encode": 0, "recover0": 1, "recover1": 2, "xor": 3}


def cpu_transform(a: bytes, b: bytes, operation: str) -> bytes:
    if type(a) is not bytes or type(b) is not bytes or not a or len(a) != len(b):
        raise ValueError("equal_nonempty_immutable_bytes_required")
    if operation not in OPS:
        raise ValueError("invalid_rs_operation")
    x = np.frombuffer(a, dtype=np.uint8)
    y = np.frombuffer(b, dtype=np.uint8)
    if operation == "recover1":
        z = np.bitwise_xor(x, y)
        out = (z >> np.uint8(1)) ^ ((z & np.uint8(1)) * np.uint8(0x8E))
    elif operation == "xor":
        out = np.bitwise_xor(x, y)
    else:
        doubled = (y << np.uint8(1)) ^ ((y >> np.uint8(7)) * np.uint8(0x1D))
        out = np.bitwise_xor(x, doubled)
    return out.tobytes()


@dataclass
class CodecStats:
    backend: str
    operation: str
    bytes: int
    elapsed_ms: float
    h2d_included: bool
    d2h_included: bool
    fused_transform_write: bool


class CudaRSCodec:
    def __init__(self, backend="auto", device=None):
        if backend not in ("auto", "cuda", "cpu"):
            raise ValueError("invalid_rs_backend")
        available = bool(
            torch is not None and triton is not None and torch.cuda.is_available()
        )
        if backend == "cuda" and not available:
            raise RuntimeError("cuda_rs_requested_but_unavailable")
        self.backend = "cuda" if available and backend != "cpu" else "cpu"
        self.device = None if self.backend == "cpu" else torch.device(
            device or os.environ.get("SGLANG_KV_EC_RS_DEVICE", "cuda:0")
        )
        self.last_stats = None
        if self.backend == "cuda":
            # Compile and verify all operations before serving a real stripe.
            a = bytes(range(1, 65)); b = bytes(range(64, 0, -1))
            for op in OPS:
                got = self.transform(a, b, op, preflight=False)
                if got != cpu_transform(a, b, op):
                    raise RuntimeError("cuda_rs_preflight_mismatch:" + op)

    def transform_into(self, a, b, out, operation, *, stream=None):
        if self.backend != "cuda" or operation not in OPS:
            raise RuntimeError("cuda_transform_into_unavailable")
        if any(
            not x.is_cuda or x.dtype != torch.uint8 or x.ndim != 1
            or not x.is_contiguous() or x.numel() != out.numel()
            for x in (a, b, out)
        ) or a.device != b.device or a.device != out.device:
            raise ValueError("equal_contiguous_cuda_uint8_required")
        stream = stream or torch.cuda.current_stream(out.device)
        with torch.cuda.device(out.device), torch.cuda.stream(stream):
            _rs21_kernel[(triton.cdiv(out.numel(), 256),)](
                a, b, out, n=out.numel(), OP=OPS[operation], BLOCK=256
            )
            done = torch.cuda.Event()
            done.record(stream)
        return done

    def xor_scatter_into(self, parity, survivor, pool, slots, *, stream=None):
        """One CUDA kernel performs XOR and writes the final KV pool slots."""
        if self.backend != "cuda":
            raise RuntimeError("cuda_xor_scatter_unavailable")
        if (
            any(not x.is_cuda or x.dtype != torch.uint8 or x.ndim != 1
                or not x.is_contiguous() for x in (parity, survivor))
            or parity.numel() != survivor.numel() or parity.numel() <= 0
            or not pool.is_cuda or pool.dtype != torch.bfloat16 or not pool.is_contiguous()
            or not slots.is_cuda or slots.dtype != torch.int64 or slots.ndim != 1
            or slots.numel() <= 0 or parity.numel() % slots.numel()
            or len({parity.device, survivor.device, pool.device, slots.device}) != 1
        ):
            raise ValueError("invalid_fused_xor_scatter_geometry")
        bytes_per_token = parity.numel() // slots.numel()
        expected = pool[0].numel() * pool.element_size()
        if bytes_per_token != expected:
            raise ValueError("fused_xor_scatter_pool_shape_mismatch")
        stream = stream or torch.cuda.current_stream(pool.device)
        with torch.cuda.device(pool.device), torch.cuda.stream(stream):
            _xor_scatter_kernel[(triton.cdiv(parity.numel(), 256),)](
                parity, survivor, pool.view(torch.uint8).reshape(-1), slots,
                n=parity.numel(), BYTES_PER_TOKEN=bytes_per_token, BLOCK=256,
            )
            done = torch.cuda.Event(); done.record(stream)
        return done

    def transform(self, a: bytes, b: bytes, operation: str, *, preflight=True):
        started = time.perf_counter()
        if self.backend == "cpu":
            result = cpu_transform(a, b, operation)
            self.last_stats = CodecStats(
                "cpu-vectorized", operation, len(a),
                (time.perf_counter() - started) * 1000, False, False, False,
            )
            return result
        if type(a) is not bytes or type(b) is not bytes or not a or len(a) != len(b):
            raise ValueError("equal_nonempty_immutable_bytes_required")
        # Pinned staging + nonblocking H2D feeds the one fused GF/write kernel.
        ha = torch.frombuffer(bytearray(a), dtype=torch.uint8).pin_memory()
        hb = torch.frombuffer(bytearray(b), dtype=torch.uint8).pin_memory()
        with torch.cuda.device(self.device):
            stream = torch.cuda.Stream(device=self.device, priority=0)
            with torch.cuda.stream(stream):
                da = ha.to(self.device, non_blocking=True)
                db = hb.to(self.device, non_blocking=True)
                out = torch.empty_like(da)
            done = self.transform_into(da, db, out, operation, stream=stream)
            host = torch.empty_like(ha, pin_memory=True)
            with torch.cuda.stream(stream):
                stream.wait_event(done)
                host.copy_(out, non_blocking=True)
                complete = torch.cuda.Event(); complete.record(stream)
            complete.synchronize()
        result = host.numpy().tobytes()
        self.last_stats = CodecStats(
            "triton-cuda", operation, len(a),
            (time.perf_counter() - started) * 1000, True, True, True,
        )
        return result

    def encode(self, node0: bytes, node1: bytes):
        return self.transform(node0, node1, "encode")

    def encode_xor(self, node0: bytes, node1: bytes):
        """RS(2,1) generator row [1,1] used by the node-level protocol."""
        width = max(len(node0), len(node1))
        return self.transform(
            node0 + bytes(width - len(node0)),
            node1 + bytes(width - len(node1)),
            "xor",
        )

    def recover_xor(self, blocks, parity: bytes, lengths):
        if len(blocks) != 2 or len(lengths) != 2 or sum(x is None for x in blocks) != 1:
            raise ValueError("rs21_requires_one_missing_node")
        lost = 0 if blocks[0] is None else 1
        survivor = blocks[1 - lost]
        width = max(lengths)
        if len(parity) != width or len(survivor) != lengths[1 - lost]:
            raise ValueError("rs21_length_mismatch")
        raw = self.transform(
            parity, survivor + bytes(width - len(survivor)), "xor"
        )
        if any(raw[lengths[lost]:]):
            raise ValueError("rs21_nonzero_padding")
        return raw[: lengths[lost]]
    def recover(self, blocks, parity: bytes, lengths):
        if len(blocks) != 2 or len(lengths) != 2 or sum(x is None for x in blocks) != 1:
            raise ValueError("rs21_requires_one_missing_node")
        lost = 0 if blocks[0] is None else 1
        survivor = blocks[1 - lost]
        width = max(lengths)
        if len(parity) != width or len(survivor) != lengths[1 - lost]:
            raise ValueError("rs21_length_mismatch")
        survivor = survivor + bytes(width - len(survivor))
        raw = self.transform(parity, survivor, "recover0" if lost == 0 else "recover1")
        return raw[: lengths[lost]]
