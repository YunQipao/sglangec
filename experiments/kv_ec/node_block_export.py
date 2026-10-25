"""Offline real capture -> CPU-owned TP2 node block. No GPU/allocator access.

Pack: magic KVNODE01, uint32 BE JSON length, canonical UTF-8 JSON, rank0,
rank1; ranks use ascending layer / K / V / logical token order from capture.
No online export hook is installed. One offline node is processed at a time,
bounded by MAX_NODE_BYTES; no service-critical source ownership is extended.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from .rs_codec import xor

MAGIC = b"KVNODE01"
MAX_NODE_BYTES = 160 * 1024**2
MAX_META_BYTES = 1024**2
IDENTITY = (
    "format",
    "provenance",
    "snapshot_id",
    "request_id",
    "request_id_encoding",
    "request_version",
    "checkpoint_version",
    "manager_epoch",
    "model_identifier",
    "model_sha256",
    "source_commit",
    "core_diff_sha256",
    "token_start",
    "token_count",
    "token_ids_sha256",
    "layers",
    "dtype",
    "tp_size",
    "packed_byte_count",
    "lifetime",
    "completion_dependency",
    "pool_class",
)


def canonical(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def layout(meta):
    return [
        {
            k: r[k]
            for k in (
                "layer_id",
                "dtype",
                "packed_shape",
                "k_offset",
                "k_nbytes",
                "v_offset",
                "v_nbytes",
            )
        }
        for r in meta["layer_records"]
    ]


def validate_rank(raw, meta, rank, *, allow_fixture=False):
    if meta.get("provenance") != "sglang_request" and not (
        allow_fixture and meta.get("provenance") == "synthetic_fixture"
    ):
        raise ValueError("real SGLang capture required; fixtures are test-only")
    for key in IDENTITY:
        if key not in meta or meta[key] is None or meta[key] == "":
            raise ValueError("missing identity: " + key)
    if (
        meta["tp_size"] != 2
        or meta["tp_rank"] != rank
        or meta["dtype"] != "torch.bfloat16"
        or meta["format"] != "kv-ec-rank-bin-v1"
        or meta.get("snapshot_completion_status") != "complete"
        or meta.get("source_unchanged") is not True
        or meta.get("checksum_deferred")
        or len(raw) != meta["packed_byte_count"]
        or sha(raw) != meta["sha256"]
    ):
        raise ValueError("invalid rank completion/type/checksum")
    count = meta["token_count"]
    slots = meta["physical_slots"]
    if (
        type(count) is not int
        or count <= 0
        or type(meta["token_start"]) is not int
        or meta["token_start"] < 0
        or len(slots) != count
        or len(set(slots)) != count
        or any(type(s) is not int or s <= 0 for s in slots)
        or type(meta["req_generation"]) is not int
        or meta["req_generation"] <= 0
        or type(meta["req_pool_idx"]) is not int
        or meta["req_pool_idx"] <= 0
    ):
        raise ValueError("invalid range/generation/slots")
    generations = meta.get("slot_generation")
    if generations is not None and (
        len(generations) != count
        or any(type(g) is not int or g <= 0 for g in generations)
    ):
        raise ValueError("invalid slot generations")
    records = meta["layer_records"]
    if (
        not records
        or [r["layer_id"] for r in records] != meta["layers"]
        or meta["layers"] != sorted(set(meta["layers"]))
    ):
        raise ValueError("invalid layer order")
    offset = 0
    hashes = {k: hashlib.sha256() for k in ("k", "v")}
    for r in records:
        if r["dtype"] != meta["dtype"]:
            raise ValueError("layer dtype mismatch")
        for kind in ("k", "v"):
            shape, n = r["packed_shape"][kind], r[kind + "_nbytes"]
            if (
                len(shape) != 3
                or any(type(d) is not int or d <= 0 for d in shape)
                or shape[0] != count
                or math.prod(shape) * 2 != n
                or r[kind + "_offset"] != offset
                or offset + n > len(raw)
            ):
                raise ValueError("invalid segment layout")
            if meta["provenance"] == "sglang_request" and (
                count not in (128, 256, 512)
                or shape[1:] != [4, 128]
                or not 0 <= r["layer_id"] < 36
            ):
                raise ValueError("Qwen3-8B TP2 capture geometry required")
            segment = raw[offset : offset + n]
            if sha(segment) != r[kind + "_sha256"]:
                raise ValueError("segment checksum mismatch")
            hashes[kind].update(segment)
            offset += n
    if offset != len(raw) or any(
        h.hexdigest() != meta[k + "_sha256"] for k, h in hashes.items()
    ):
        raise ValueError("aggregate checksum mismatch")


def validate_meta_pair(ranks):
    if (
        len(ranks) != 2
        or any(ranks[0].get(k) != ranks[1].get(k) for k in IDENTITY)
        or layout(ranks[0]) != layout(ranks[1])
    ):
        raise ValueError("mixed rank request/chunk/version/layout")


def pack(node, blocks, ranks, *, allow_fixture=False):
    if type(node) is not int or node not in (0, 1) or len(blocks) != 2:
        raise ValueError("exactly two ranks per node")
    validate_meta_pair(ranks)
    for i in range(2):
        validate_rank(blocks[i], ranks[i], i, allow_fixture=allow_fixture)
    header = canonical(
        {
            "format": "kv-node-v1",
            "node_id": node,
            "ranks": ranks,
            "rank_lengths": [len(b) for b in blocks],
        }
    )
    if (
        len(header) > MAX_META_BYTES
        or 12 + len(header) + sum(map(len, blocks)) > MAX_NODE_BYTES
    ):
        raise ValueError("node byte budget exceeded")
    return MAGIC + struct.pack("!I", len(header)) + header + b"".join(blocks)


def unpack(block, *, allow_fixture=False):
    if len(block) < 12 or len(block) > MAX_NODE_BYTES or block[:8] != MAGIC:
        raise ValueError("invalid node block")
    n = struct.unpack("!I", block[8:12])[0]
    if not 0 < n <= MAX_META_BYTES or 12 + n > len(block):
        raise ValueError("invalid node header length")
    meta = json.loads(block[12 : 12 + n])
    if (
        canonical(meta) != block[12 : 12 + n]
        or meta["format"] != "kv-node-v1"
        or type(meta["node_id"]) is not int
        or meta["node_id"] not in (0, 1)
    ):
        raise ValueError("noncanonical node metadata")
    lengths = meta["rank_lengths"]
    if (
        len(lengths) != 2
        or any(type(n) is not int or n <= 0 for n in lengths)
        or 12 + n + sum(lengths) != len(block)
    ):
        raise ValueError("invalid node lengths")
    start = 12 + n
    blocks = (block[start : start + lengths[0]], block[start + lengths[0] :])
    validate_meta_pair(meta["ranks"])
    for rank, raw in enumerate(blocks):
        validate_rank(raw, meta["ranks"][rank], rank, allow_fixture=allow_fixture)
    return meta, blocks


def descriptor(block, *, allow_fixture=False):
    meta, _ = unpack(block, allow_fixture=allow_fixture)
    return {"length": len(block), "sha256": sha(block), "metadata": meta}


def export(directory, node):
    # Existing validator additionally checks exact sidecar sets and Torch views.
    from .rank_snapshot import assemble

    directory = Path(directory)
    if sum(p.stat().st_size for p in directory.glob("rank_*.bin")) > MAX_NODE_BYTES:
        raise ValueError("offline rank byte budget exceeded")
    manifest, arrays = assemble(directory)
    ranks = manifest["ranks"]
    if manifest["tp_size"] != 2:
        raise ValueError("TP2 capture required")
    blocks = [a.tobytes() for a in arrays]
    start = time.perf_counter()
    local = xor(*blocks)
    xor_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    block = pack(node, blocks, ranks)
    pack_ms = (time.perf_counter() - start) * 1000
    return block, {
        "node_id": node,
        "mode": "offline_real_capture",
        "rank_bytes": list(map(len, blocks)),
        "xor_ms": xor_ms,
        "pack_ms": pack_ms,
        "local_parity_sha256": sha(local),
        "descriptor": descriptor(block),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--node-id", type=int, choices=(0, 1), required=True)
    p.add_argument("--output", type=Path, required=True, help="new export directory")
    args = p.parse_args()
    block, report = export(args.snapshot, args.node_id)
    from .rs_experiment import save_report

    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "node.bin").open("xb") as f:
        f.write(block)
    save_report(args.output / "export", report)
    print(json.dumps({"output": str(args.output), "sha256": sha(block)}))


if __name__ == "__main__":
    main()
