"""Systematic Reed-Solomon RS(2,1) over GF(2^8), CPU byte arrays only."""

import numpy as np

CODEC = "rs-2-1-gf256"
CODEC_VERSION = 1
PRIMITIVE_POLYNOMIAL = 0x11D


def _tables():
    exp = np.zeros(512, dtype=np.uint8)
    log = np.zeros(256, dtype=np.uint8)
    value = 1
    for index in range(255):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x100:
            value ^= PRIMITIVE_POLYNOMIAL
    exp[255:] = exp[:257]
    return exp, log


_EXP, _LOG = _tables()


def _block(value, label):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise TypeError(f"{label} must be a uint8 numpy array")
    if value.ndim != 1 or value.size == 0 or not value.flags.c_contiguous:
        raise ValueError(f"{label} must be nonempty, flat, and contiguous")
    return value


def _mul(value, coefficient):
    if coefficient == 0:
        return np.zeros_like(value)
    if coefficient == 1:
        return value.copy()
    result = np.zeros_like(value)
    nonzero = value != 0
    result[nonzero] = _EXP[
        _LOG[value[nonzero]].astype(np.uint16) + int(_LOG[coefficient])
    ]
    return result


def encode(data0, data1):
    """Return (D0, D1, P) using P = D0 XOR 2*D1 in GF(256)."""
    d0, d1 = _block(data0, "D0"), _block(data1, "D1")
    if d0.size != d1.size:
        raise ValueError("encoded D0 and D1 must be explicitly padded to equal length")
    parity = np.bitwise_xor(d0, _mul(d1, 2))
    return d0.copy(), d1.copy(), parity


def recover(*, data0, data1, parity, lengths):
    """Recover one erased role and return original, unpadded D0 and D1 bytes."""
    if not isinstance(lengths, (tuple, list)) or len(lengths) != 2:
        raise ValueError("exactly two original data lengths are required")
    if any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("original data lengths must be positive integers")
    present = [data0 is not None, data1 is not None, parity is not None]
    if sum(present) != 2:
        raise ValueError("exactly one RS block must be missing")
    blocks = [_block(item, f"block{index}") for index, item in enumerate((data0, data1, parity)) if item is not None]
    width = max(lengths)
    if any(length > width for length in lengths):
        raise ValueError("original data length exceeds padded block length")
    if any(block.size != width for block in blocks):
        raise ValueError("available padded blocks must have the same length")
    if data0 is None:
        d0 = np.bitwise_xor(parity, _mul(data1, 2))
        d1 = data1.copy()
    elif data1 is None:
        inverse_two = int(_EXP[255 - int(_LOG[2])])
        d1 = _mul(np.bitwise_xor(parity, data0), inverse_two)
        d0 = data0.copy()
    elif parity is None:
        d0, d1 = data0.copy(), data1.copy()
    else:
        raise ValueError("invalid erasure set")
    if np.any(d0[lengths[0]:]) or np.any(d1[lengths[1]:]):
        raise ValueError("recovered padding is not zero")
    return d0[:lengths[0]].copy(), d1[:lengths[1]].copy()
