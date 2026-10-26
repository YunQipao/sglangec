"""Bind two independent request snapshots into an immutable stripe manifest."""

import re

from .node_block_export import canonical, descriptor, layout, sha, MAX_NODE_BYTES
from .rs_codec import CODE


def validate_manifest(manifest, *, allow_fixture=False):
    if (
        manifest.get("format") != "kv-rs-stripe-v1"
        or manifest.get("code") != CODE
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", manifest.get("stripe_id", ""))
    ):
        raise ValueError("invalid stripe identity/code")
    nodes = manifest["nodes"]
    if len(nodes) != 2:
        raise ValueError("two nodes required")
    tp_sizes = set()
    for i, d in enumerate(nodes):
        metadata = d.get("metadata", {})
        ranks = metadata.get("ranks", [])
        if (
            type(d["length"]) is not int
            or not 0 < d["length"] <= MAX_NODE_BYTES
            or not re.fullmatch("[0-9a-f]{64}", d["sha256"])
            or metadata.get("node_id") != i
            or len(ranks) not in (2, 4)
            or any(m.get("tp_size") != len(ranks) for m in ranks)
        ):
            raise ValueError(
                f"invalid node descriptor: node={i}; check length, sha256, "
                "node_id, rank count and tp_size"
            )
        tp_sizes.add(ranks[0].get("tp_size") if ranks else None)
        for rank, m in enumerate(ranks):
            if (
                m["tp_rank"] != rank
                or m["tp_size"] != len(ranks)
                or m["dtype"] != "torch.bfloat16"
                or m["snapshot_completion_status"] != "complete"
                or (
                    m["provenance"] != "sglang_request"
                    and not (allow_fixture and m["provenance"] == "synthetic_fixture")
                )
            ):
                raise ValueError(
                    f"invalid descriptor provenance/layout: node={i} rank={rank}; "
                    f"tp_rank={m['tp_rank']!r}, tp_size={m['tp_size']!r}, "
                    f"dtype={m['dtype']!r}, "
                    f"snapshot_completion_status={m['snapshot_completion_status']!r}, "
                    f"provenance={m['provenance']!r}"
                )
    if len(tp_sizes) != 1 or next(iter(tp_sizes)) not in (2, 4):
        raise ValueError("both nodes must use the same TP2 or TP4 layout")
    a_ranks, b_ranks = [d["metadata"]["ranks"] for d in nodes]
    a, b = a_ranks[0], b_ranks[0]
    # Model installation paths, request IDs, generations and checkpoint versions
    # belong to EACH node. Model compatibility is pinned by content fingerprints,
    # not by the local model_identifier path, which stays in each descriptor.
    # They need not be equal across independent requests; their full documents
    # are pinned by the manifest hash and verified on every submission.
    shared = (
        "model_sha256",
        "source_commit",
        "core_diff_sha256",
        "dtype",
        "tp_size",
        "token_start",
        "token_count",
        "layers",
    )
    mismatches = [
        f"{key}: node0={a[key]!r}, node1={b[key]!r}"
        for key in shared if a[key] != b[key]
    ]
    for rank, (left, right) in enumerate(zip(a_ranks, b_ranks)):
        left_layout, right_layout = layout(left), layout(right)
        if len(left_layout) != len(right_layout):
            mismatches.append(
                f"rank={rank} layer_records.length: "
                f"node0={len(left_layout)}, node1={len(right_layout)}"
            )
        for index, (left_record, right_record) in enumerate(
            zip(left_layout, right_layout)
        ):
            mismatches.extend(
                f"rank={rank} layer_records[{index}].{key}: "
                f"node0={left_record[key]!r}, node1={right_record[key]!r}"
                for key in left_record if left_record[key] != right_record[key]
            )
    if mismatches:
        raise ValueError(
            "cross-node chunk/model/layout mismatch: " + "; ".join(mismatches)
        )
    if a["snapshot_id"] == b["snapshot_id"] or a["request_id"] == b["request_id"]:
        raise ValueError("independent node snapshot/request identities required")
    from .online_rs_identity import validate_online_manifest

    validate_online_manifest(manifest)


def build(stripe_id, blocks, *, allow_fixture=False):
    manifest = {
        "format": "kv-rs-stripe-v1",
        "stripe_id": stripe_id,
        "code": CODE,
        "nodes": [descriptor(b, allow_fixture=allow_fixture) for b in blocks],
    }
    validate_manifest(manifest, allow_fixture=allow_fixture)
    return manifest


def manifest_sha(manifest):
    return sha(canonical(manifest))
