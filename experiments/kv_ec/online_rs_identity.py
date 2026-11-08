"""Explicit cross-request pairing, separate from node-local checkpoint identity."""

from .node_block_export import canonical, sha


def stripe_id(identity):
    return "online-" + sha(canonical(identity))


def identity_for(
    run_id, protection_id, checkpoint_version, rank, *, full_prefix=False
):
    if type(full_prefix) is not bool:
        raise ValueError("full_prefix must be boolean")
    identity = dict(
        format=(
            "kv-cold-rs-prefix-identity-v1"
            if full_prefix
            else "kv-online-rs-identity-v1"
        ),
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
    grain = rank.get("capture_grain_tokens", 256)
    chunk = rank.get("chunk_id")
    identity_format = identity.get("format") if isinstance(identity, dict) else None
    full_prefix = identity_format == "kv-cold-rs-prefix-identity-v1"
    if full_prefix:
        prefix_tokens = rank.get("token_count")
        valid_extent = (
            type(prefix_tokens) is int
            and prefix_tokens in (256, 512, 1024, 2048, 4096)
            and chunk == 0
        )
        expected_range = [0, prefix_tokens] if valid_extent else None
    else:
        valid_extent = grain in (128, 256, 512, 1024)
        expected_range = (
            [chunk * grain, (chunk + 1) * grain]
            if type(chunk) is int and type(grain) is int
            else None
        )
    layer_range = rank.get("layer_range")
    layer_count = (
        layer_range[1] if isinstance(layer_range, list)
        and len(layer_range) == 2 and layer_range[0] == 0
        and type(layer_range[1]) is int else -1
    )
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
        or identity["format"] not in (
            "kv-online-rs-identity-v1",
            "kv-cold-rs-prefix-identity-v1",
        )
        or any(
            not isinstance(identity[k], str) or not 0 < len(identity[k]) <= 128
            for k in ("run_id", "protection_id", "checkpoint_version")
        )
        or type(identity["chunk_id"]) is not int
        or identity["chunk_id"] < 0
        or not valid_extent
        or identity["token_range"] != expected_range
        or not 0 < layer_count <= 128
        or identity["layer_range"] != [0, layer_count]
        or rank.get("chunk_id") != identity["chunk_id"]
        or rank.get("token_range") != identity["token_range"]
        or rank.get("layer_range") != identity["layer_range"]
        or rank.get("token_start") != identity["token_range"][0]
        or rank.get("token_count") != (
            identity["token_range"][1] if full_prefix else grain
        )
        or rank.get("layers") != list(range(layer_count))
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
