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
    for i, d in enumerate(nodes):
        if (
            type(d["length"]) is not int
            or not 0 < d["length"] <= MAX_NODE_BYTES
            or not re.fullmatch("[0-9a-f]{64}", d["sha256"])
            or d["metadata"]["node_id"] != i
            or len(d["metadata"]["ranks"]) != 2
        ):
            raise ValueError("invalid node descriptor")
        for rank, m in enumerate(d["metadata"]["ranks"]):
            if (
                m["tp_rank"] != rank
                or m["tp_size"] != 2
                or m["dtype"] != "torch.bfloat16"
                or m["snapshot_completion_status"] != "complete"
                or (
                    m["provenance"] != "sglang_request"
                    and not (allow_fixture and m["provenance"] == "synthetic_fixture")
                )
            ):
                raise ValueError("invalid descriptor provenance/layout")
    a, b = [d["metadata"]["ranks"][0] for d in nodes]
    # Request IDs, generations and checkpoint versions belong to EACH node.
    # They need not be equal across independent requests; their full documents
    # are pinned by the manifest hash and verified on every submission.
    shared = (
        "model_identifier",
        "model_sha256",
        "source_commit",
        "core_diff_sha256",
        "dtype",
        "tp_size",
        "token_start",
        "token_count",
        "layers",
    )
    if any(a[k] != b[k] for k in shared) or layout(a) != layout(b):
        raise ValueError("cross-node chunk/model/layout mismatch")
    if a["snapshot_id"] == b["snapshot_id"] or a["request_id"] == b["request_id"]:
        raise ValueError("independent node snapshot/request identities required")


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
