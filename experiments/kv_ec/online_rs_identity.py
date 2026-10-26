"""Explicit cross-request pairing, separate from node-local checkpoint identity."""

from .node_block_export import canonical, sha


def stripe_id(identity):
    return "online-" + sha(canonical(identity))


def identity_for(run_id, protection_id, checkpoint_version, rank):
    identity = dict(
        format="kv-online-rs-identity-v1",
        run_id=run_id,
        protection_id=protection_id,
        checkpoint_version=checkpoint_version,
        chunk_id=rank["chunk_id"],
        token_range=rank["token_range"],
        layer_range=rank["layer_range"],
    )
    validate_identity(identity, rank)
    return identity


def validate_identity(identity, rank):
    if (
        not isinstance(identity, dict)
        or set(identity)
        != {
            "format",
            "run_id",
            "protection_id",
            "checkpoint_version",
            "chunk_id",
            "token_range",
            "layer_range",
        }
        or identity["format"] != "kv-online-rs-identity-v1"
        or any(
            not isinstance(identity[k], str) or not 0 < len(identity[k]) <= 128
            for k in ("run_id", "protection_id", "checkpoint_version")
        )
        or type(identity["chunk_id"]) is not int
        or identity["chunk_id"] < 0
        or identity["token_range"]
        != [identity["chunk_id"] * 256, (identity["chunk_id"] + 1) * 256]
        or identity["layer_range"] != [0, 36]
        or rank.get("chunk_id") != identity["chunk_id"]
        or rank.get("token_range") != identity["token_range"]
        or rank.get("layer_range") != identity["layer_range"]
        or rank.get("token_start") != identity["token_range"][0]
        or rank.get("token_count") != 256
        or rank.get("layers") != list(range(36))
    ):
        raise ValueError("invalid global protection identity/range")


def validate_online_manifest(manifest):
    """Optional extension: existing offline manifests remain unchanged."""
    ranks = [r for d in manifest["nodes"] for r in d["metadata"]["ranks"]]
    if not any("global_protection" in r for r in ranks):
        return
    identity = ranks[0].get("global_protection")
    for rank in ranks:
        validate_identity(rank.get("global_protection"), rank)
        if rank["global_protection"] != identity:
            raise ValueError("mixed global protection identities")
    if manifest["stripe_id"] != stripe_id(identity):
        raise ValueError("stripe ID not bound to global protection identity")
