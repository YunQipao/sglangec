"""Systematic RS(2,1), GF(256), polynomial 0x11d, primitive element 2.

G = [[1,0],[0,1],[1,1]]. This single-parity specialization IS ordinary XOR;
it has no coding advantage over XOR and cannot recover two erased data blocks.
"""

import numpy as np

CODE = {
    "name": "RS(2,1)-XOR-special-case",
    "field": "GF(2^8)",
    "polynomial": "0x11d",
    "primitive_element": 2,
    "generator_matrix": [[1, 0], [0, 1], [1, 1]],
}


def gf_mul(a, b):
    if not 0 <= a < 256 or not 0 <= b < 256:
        raise ValueError("GF operands must be bytes")
    value = 0
    while b:
        if b & 1:
            value ^= a
        a <<= 1
        if a & 256:
            a ^= 0x11D
        b >>= 1
    return value


def xor(a, b):
    if len(a) != len(b):
        raise ValueError("equal padded block lengths required")
    return np.bitwise_xor(
        np.frombuffer(a, dtype=np.uint8), np.frombuffer(b, dtype=np.uint8)
    ).tobytes()


def encode(d0, d1):
    size = max(len(d0), len(d1))
    if size == 0:
        raise ValueError("empty blocks")
    return xor(d0.ljust(size, b"\0"), d1.ljust(size, b"\0"))


def recover(blocks, parity, lengths):
    if len(blocks) != 2 or sum(b is None for b in blocks) != 1 or parity is None:
        raise ValueError("one missing node and committed parity required")
    if len(lengths) != 2 or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("invalid original lengths")
    size = max(lengths)
    lost = blocks.index(None)
    survivor = blocks[1 - lost]
    if len(parity) != size or len(survivor) != lengths[1 - lost]:
        raise ValueError("length mismatch")
    recovered = xor(parity, survivor.ljust(size, b"\0"))
    if any(recovered[lengths[lost] :]):
        raise ValueError("nonzero padding")
    return recovered[: lengths[lost]]
