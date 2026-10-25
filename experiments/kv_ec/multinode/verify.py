"""Strict RS stripe metadata, checksums, padding, and erasure verification."""

import hashlib
import argparse
import json

import numpy as np

from .rs_codec import CODEC, CODEC_VERSION, encode, recover

REQUIRED = {"experiment_id", "stripe_id", "chunk_id", "source_node_id", "model_fingerprint",
            "payload_length", "padded_length", "padding_length", "codec", "codec_version",
            "rs_parameters", "checksum", "checkpoint_version", "protection_state", "block_role"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_stripe(metadata, blocks):
    if set(blocks) != {"D0", "D1", "P"}:
        raise ValueError("complete D0/D1/P blocks required for stripe verification")
    for role, meta in metadata.items():
        missing = REQUIRED - meta.keys()
        if missing:
            raise ValueError(f"{role} metadata missing {sorted(missing)}")
        if meta["block_role"] != role or meta["protection_state"] != "COMMITTED":
            raise ValueError(f"{role} role or protection state mismatch")
        if meta["codec"] != CODEC or meta["codec_version"] != CODEC_VERSION:
            raise ValueError("unsupported RS codec metadata")
    identity = ("experiment_id", "stripe_id", "chunk_id", "checkpoint_version", "codec", "codec_version")
    if any(any(metadata[role][key] != metadata["D0"][key] for key in identity) for role in ("D1", "P")):
        raise ValueError("stripe identity metadata mismatch")
    for key in ("model_fingerprint", "rs_parameters", "padded_length"):
        if any(metadata[role][key] != metadata["D0"][key] for role in ("D1", "P")):
            raise ValueError(f"stripe {key} metadata mismatch")
    if (metadata["D0"]["source_node_id"] != "node-a"
            or metadata["D1"]["source_node_id"] != "node-b"
            or metadata["P"]["source_node_id"] != "node-b"):
        raise ValueError("initial two-node block placement mismatch")
    d0, d1, parity = (blocks[role] for role in ("D0", "D1", "P"))
    if d0.size != d1.size or d0.size != parity.size:
        raise ValueError("RS padded lengths do not agree")
    lengths = (metadata["D0"]["payload_length"], metadata["D1"]["payload_length"])
    if any(type(length) is not int or length <= 0 or length > d0.size for length in lengths):
        raise ValueError("original payload length metadata invalid")
    if any(metadata[role]["payload_length"] + metadata[role]["padding_length"] != d0.size
           for role in ("D0", "D1")):
        raise ValueError("data block padding metadata mismatch")
    if metadata["P"]["payload_length"] != d0.size or metadata["P"]["padding_length"] != 0:
        raise ValueError("parity block length metadata mismatch")
    if any(metadata[role]["padded_length"] != d0.size for role in ("D0", "D1", "P")):
        raise ValueError("padded length metadata mismatch")
    for role, block in blocks.items():
        if sha256(block.tobytes()) != metadata[role]["checksum"]:
            raise ValueError(f"{role} checksum mismatch")
    expected_p = encode(d0, d1)[2]
    if not np.array_equal(expected_p, parity):
        raise ValueError("parity does not encode D0 and D1")
    results = {}
    for lost in ("D0", "D1", "P"):
        available = {role: (None if role == lost else blocks[role]) for role in blocks}
        restored = recover(data0=available["D0"], data1=available["D1"], parity=available["P"], lengths=lengths)
        if lost == "D0" and restored[0].tobytes() != d0[:lengths[0]].tobytes():
            raise ValueError("D0 byte-exact recovery failed")
        if lost == "D1" and restored[1].tobytes() != d1[:lengths[1]].tobytes():
            raise ValueError("D1 byte-exact recovery failed")
        if lost == "P" and not np.array_equal(encode(d0, d1)[2], parity):
            raise ValueError("P recomputation was not byte exact")
        results[lost] = "byte_exact"
    return {"state": "PASS", "erasures": results, "shared_node_failure_recovery": False}


def verify_synthetic(d0_length, d1_length, *, seed=20261025):
    if any(type(length) is not int or not 0 < length <= 256 * 1024**2
           for length in (d0_length, d1_length)):
        raise ValueError("synthetic block lengths must be in 1..256 MiB")
    raw = [np.random.default_rng(seed + index).integers(0, 256, length, dtype=np.uint8)
           for index, length in enumerate((d0_length, d1_length))]
    width = max(d0_length, d1_length)
    d0, d1 = [np.pad(block, (0, width - len(block))) for block in raw]
    _, _, parity = encode(d0, d1)
    blocks = {"D0": d0, "D1": d1, "P": parity}
    params = {"k": 2, "m": 1, "field": "GF(2^8)",
              "primitive_polynomial": "0x11d", "coefficients": [1, 2]}
    metadata = {}
    for role, block in blocks.items():
        payload_length = len(raw[0]) if role == "D0" else len(raw[1]) if role == "D1" else width
        metadata[role] = {
            "experiment_id": "verify-synthetic", "stripe_id": "stripe-0", "chunk_id": "chunk-0",
            "source_node_id": "node-a" if role == "D0" else "node-b",
            "model_fingerprint": f"synthetic-seed-{seed}",
            "payload_length": payload_length, "padded_length": width,
            "padding_length": width - payload_length if role != "P" else 0,
            "codec": CODEC, "codec_version": CODEC_VERSION, "rs_parameters": params,
            "checksum": sha256(block.tobytes()), "checkpoint_version": "checkpoint-v1",
            "protection_state": "COMMITTED", "block_role": role,
        }
    return verify_stripe(metadata, blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--d1-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    print(json.dumps(verify_synthetic(args.d0_bytes, args.d1_bytes, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
