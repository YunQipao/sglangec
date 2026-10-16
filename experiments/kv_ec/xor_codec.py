"""Single-erasure XOR over CPU bytes. No floating point arithmetic or SGLang import."""

from collections.abc import Sequence

import numpy as np


def byte_block(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise TypeError("blocks must be uint8 numpy arrays (reinterpret, do not cast)")
    if value.ndim != 1 or not value.flags.c_contiguous or value.size == 0:
        raise ValueError("blocks must be nonempty, flat, contiguous bytes")
    return value


def encode(blocks: Sequence[np.ndarray]) -> np.ndarray:
    if len(blocks) < 2:
        raise ValueError("XOR protection requires at least two data blocks")
    first = byte_block(blocks[0])
    # Parity owns storage; none of the input bytes are modified.
    parity = first.copy()
    for block in blocks[1:]:
        block = byte_block(block)
        if block.size != first.size:
            raise ValueError("data blocks must have equal byte lengths")
        np.bitwise_xor(parity, block, out=parity)
    return parity


def recover(blocks: Sequence[np.ndarray | None], parity: np.ndarray) -> np.ndarray:
    if len(blocks) < 2 or sum(b is None for b in blocks) != 1:
        raise ValueError("exactly one missing data block is supported")
    restored = byte_block(parity).copy()
    for block in blocks:
        if block is None:
            continue
        block = byte_block(block)
        if block.size != restored.size:
            raise ValueError("survivors and parity must have equal byte lengths")
        np.bitwise_xor(restored, block, out=restored)
    return restored
