"""Assemble committed online chunks into one exact-boundary TP checkpoint."""

from copy import deepcopy
import hashlib

import numpy as np

from .node_block_export import canonical, pack, sha


def _assemble_rank(chunks, boundary, token_ids_sha256):
    if not chunks:
        raise ValueError("missing incremental chunks")
    chunks = sorted(chunks, key=lambda item: item[0]["token_start"])
    metas = [item[0] for item in chunks]
    raws = [bytes(item[1]) for item in chunks]
    first = metas[0]
    grain = int(first["token_count"])
    rank = int(first["tp_rank"])
    layers = list(first["layers"])
    if (
        boundary != grain * len(chunks)
        or [m["token_start"] for m in metas] != list(range(0, boundary, grain))
        or any(
            m["token_count"] != grain or m["tp_rank"] != rank
            or m["tp_size"] != first["tp_size"] or m["layers"] != layers
            or m["model_sha256"] != first["model_sha256"]
            or m["request_id"] != first["request_id"]
            for m in metas
        )
    ):
        raise ValueError("incremental chunk identity or coverage mismatch")

    output = bytearray()
    records = []
    k_aggregate, v_aggregate = hashlib.sha256(), hashlib.sha256()
    offset = 0
    for layer_index, layer in enumerate(layers):
        template = deepcopy(metas[0]["layer_records"][layer_index])
        record = dict(template, layer_id=layer)
        for kind, aggregate in (("k", k_aggregate), ("v", v_aggregate)):
            pieces = []
            for meta, raw in zip(metas, raws):
                source = meta["layer_records"][layer_index]
                start, size = int(source[kind + "_offset"]), int(source[kind + "_nbytes"])
                pieces.append(raw[start:start + size])
            segment = b"".join(pieces)
            shape = list(template["packed_shape"][kind])
            shape[0] = boundary
            record["packed_shape"][kind] = shape
            record[kind + "_offset"] = offset
            record[kind + "_nbytes"] = len(segment)
            record[kind + "_sha256"] = sha(segment)
            output.extend(segment)
            aggregate.update(segment)
            offset += len(segment)
        records.append(record)

    raw = bytes(output)
    meta = deepcopy(first)
    prefix_identity = {
        "format": "kv-ec-incremental-prefix-v1",
        "request_id": first["request_id"],
        "model_sha256": first["model_sha256"],
        "token_count": boundary,
    }
    prefix_id = sha(canonical(prefix_identity))
    physical_slots = [slot for m in metas for slot in m["physical_slots"]]
    meta.update(
        snapshot_id=prefix_id,
        checkpoint_version=prefix_id,
        chunk_id=0,
        token_start=0,
        token_count=boundary,
        token_range=[0, boundary],
        capture_grain_tokens=grain,
        token_ids_sha256=token_ids_sha256,
        packed_byte_count=len(raw),
        physical_slots=physical_slots,
        layer_records=records,
        sha256=sha(raw),
        k_sha256=k_aggregate.hexdigest(),
        v_sha256=v_aggregate.hexdigest(),
        snapshot_completion_status="complete",
        source_unchanged=True,
        lifetime="committed_incremental_prefix_host_dram",
        completion_dependency="all_chunks_local_and_global_committed",
    )
    return raw, meta


def build_node_checkpoint(node_id, rank_chunks, boundary, token_ids, *, allow_fixture=False):
    """Build a validated TP2 node block from both ranks' incremental chunks."""
    if node_id not in (0, 1) or set(rank_chunks) != {0, 1}:
        raise ValueError("TP2 rank chunks and node identity required")
    token_hash = hashlib.sha256(np.asarray(token_ids, dtype="<i8").tobytes()).hexdigest()
    blocks, metas = [], []
    for rank in (0, 1):
        raw, meta = _assemble_rank(rank_chunks[rank], boundary, token_hash)
        blocks.append(raw); metas.append(meta)
    return pack(node_id, blocks, metas, allow_fixture=allow_fixture), tuple(blocks)
