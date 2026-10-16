"""Offline format for already-owned CPU K/V copies; NOT a live KV extractor.

Provenance is supplied by a future safe hook, not proved by this file format.
Never load pickled model/request objects. No SGLang imports or monkeypatches.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

if __package__:
    from .xor_codec import encode, recover
else:
    from xor_codec import encode, recover

FORMAT = "kv-ec-owned-cpu-v1"
REQUIRED = {"request_id", "req_pool_idx", "request_row_generation", "slot_generation",
            "capture_id", "layer_id", "tp_rank", "tp_size", "token_range", "physical_slots",
            "source_k_stride", "source_v_stride", "source_device", "pool_class",
            "source_k_shape", "source_v_shape", "process_epoch",
            "provenance", "source_commit", "prefix_sharing", "token_ids_sha256"}


def digest(block):
    return hashlib.sha256(block).hexdigest()


def raw_tensor(tensor):
    if tensor.device.type != "cpu" or tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("only already-owned CPU FP16/BF16 snapshots are supported")
    # Reinterpret storage bytes, including NaN payloads and signed zeros. Packing
    # noncontiguous selected tensors copies elements but never converts dtype.
    return tensor.detach().contiguous().view(torch.uint8).reshape(-1).numpy().copy()


def write_snapshot(path, k, v, metadata):
    """Pack [selected tokens, local KV heads, head dim] K then V; no pool access."""
    missing = REQUIRED - metadata.keys()
    if missing:
        raise ValueError(f"missing metadata: {sorted(missing)}")
    if k.ndim != 3 or v.ndim != 3 or k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("first version requires equal 3-D K/V shapes and dtype")
    start, end = metadata["token_range"]
    if not (0 <= start < end and end - start == k.shape[0] == len(metadata["physical_slots"])):
        raise ValueError("token range/slots do not match selected K/V")
    if metadata["provenance"] not in ("synthetic_fixture", "sglang_request"):
        raise ValueError("explicit provenance required")
    if metadata["tp_size"] not in (1, 2, 4) or not 0 <= metadata["tp_rank"] < metadata["tp_size"]:
        raise ValueError("invalid TP rank/size")
    if metadata["req_pool_idx"] < 1 or metadata["request_row_generation"] < 1:
        raise ValueError("request row 0 is padding; live row generation must be positive")
    if any(not isinstance(s, int) or s <= 0 for s in metadata["physical_slots"]):
        raise ValueError("positive physical token slots required")
    kb, vb = raw_tensor(k), raw_tensor(v)
    payload = np.concatenate([kb, vb])
    meta = dict(metadata, format=FORMAT, k_shape=list(k.shape), v_shape=list(v.shape),
                dtype=str(k.dtype), packed_k_stride=list(k.contiguous().stride()),
                packed_v_stride=list(v.contiguous().stride()), k_nbytes=kb.size,
                v_nbytes=vb.size, sha256=digest(payload), layout="packed_K_then_V",
                source_stride_unit="elements", slot_generation_available=metadata["slot_generation"] is not None)
    # A single uncompressed archive keeps metadata and byte payload together.
    with Path(path).open("xb") as f:
        np.savez(f, payload=payload, metadata=np.array(json.dumps(meta)))


def read_snapshot(path):
    with np.load(path, allow_pickle=False) as archive:
        payload = archive["payload"].copy()
        meta = json.loads(str(archive["metadata"].item()))
    if meta.get("format") != FORMAT or REQUIRED - meta.keys():
        raise ValueError("unsupported or incomplete snapshot format")
    if payload.dtype != np.uint8 or payload.ndim != 1:
        raise ValueError("snapshot payload is not a flat byte block")
    if meta["dtype"] not in ("torch.float16", "torch.bfloat16"):
        raise ValueError("only raw FP16/BF16 format supported")
    for prefix in ("k", "v"):
        shape = meta[prefix + "_shape"]
        if len(shape) != 3 or any(not isinstance(d, int) or d <= 0 for d in shape):
            raise ValueError("invalid selected K/V shape")
        if 2 * int(np.prod(shape)) != meta[prefix + "_nbytes"]:
            raise ValueError("selected K/V shape does not match byte count")
    if payload.size != meta["k_nbytes"] + meta["v_nbytes"] or digest(payload) != meta["sha256"]:
        raise ValueError("snapshot byte count or checksum mismatch")
    return payload, meta


def verify_snapshots(paths):
    entries = [read_snapshot(p) for p in paths]
    if len(entries) not in (2, 4):
        raise ValueError("expected TP=2 or TP=4")
    blocks, metas = zip(*entries)
    first = metas[0]
    common = ("capture_id", "request_id", "layer_id", "token_range", "dtype", "k_shape",
              "v_shape", "tp_size", "source_commit", "provenance", "token_ids_sha256")
    if first["tp_size"] != len(entries):
        raise ValueError("tp_size does not match number of shards")
    if {m["tp_rank"] for m in metas} != set(range(len(entries))):
        raise ValueError("missing or duplicate TP rank")
    if any(any(m[key] != first[key] for key in common) for m in metas[1:]):
        raise ValueError("snapshots are not the same logical request/layer/capture")
    start = time.perf_counter_ns()
    parity = encode(blocks)
    xor_ms = (time.perf_counter_ns() - start) / 1e6
    checks = []
    for lost, expected in enumerate(blocks):
        survivors = [None if i == lost else b for i, b in enumerate(blocks)]
        start = time.perf_counter_ns()
        restored = recover(survivors, parity)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        correct = np.array_equal(restored, expected)
        if not correct:
            raise AssertionError("recovery was not byte-exact")
        checks.append({"lost_rank": metas[lost]["tp_rank"], "correct": bool(correct),
                       "recovery_ms": elapsed, "sha256": digest(restored)})
    return {"tp": len(blocks), "block_bytes": blocks[0].size, "xor_ms": xor_ms,
            "provenance": first["provenance"],
            "provenance_is_not_independently_attested": True,
            "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_snapshots(args.snapshots), indent=2))
