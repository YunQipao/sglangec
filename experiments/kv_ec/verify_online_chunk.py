"""Compare retained online bytes with an independent offline rank-bin capture."""

import argparse
import json
from pathlib import Path

import numpy as np

from .rank_snapshot import KV_LAYOUT_FIELDS, rank_manifest, validate_payload
from .snapshot import digest
from .xor_codec import encode, recover


def verify_reference(online, reference):
    online, reference = Path(online), Path(reference)
    manifest = json.loads((online / "manifest.json").read_text())
    if manifest.get("state") != "COMMITTED" or not manifest.get("retained_rank_files"):
        raise ValueError("complete online evidence with --retain-ranks required")
    tp = manifest["tp_size"]
    blocks, online_entries, reference_entries = [], [], []
    for rank in range(tp):
        meta = json.loads((online / f"rank_{rank}.json").read_text())
        ref = json.loads((reference / f"rank_{rank}.json").read_text())
        block = np.fromfile(online / f"rank_{rank}.bin", dtype=np.uint8)
        expected = np.fromfile(reference / f"rank_{rank}.bin", dtype=np.uint8)
        validate_payload(block, meta)
        validate_payload(expected, ref)
        for key in (
            "request_id",
            "model_sha256",
            "token_ids_sha256",
            "token_start",
            "token_count",
            "layers",
            "tp_size",
            "tp_rank",
            "dtype",
        ):
            if meta[key] != ref[key]:
                raise ValueError("online/reference identity mismatch: " + key)
        layout = lambda m: [
            {k: r[k] for k in KV_LAYOUT_FIELDS} for r in m["layer_records"]
        ]
        if layout(meta) != layout(ref) or not np.array_equal(block, expected):
            raise ValueError("online/reference bytes differ")
        if meta != manifest["ranks"][rank]:
            # write_rank_files may update performance after saving its sidecar;
            # online manager writes its manifest after those diagnostics.
            left, right = dict(meta), dict(manifest["ranks"][rank])
            left.pop("performance_ms", None)
            right.pop("performance_ms", None)
            if left != right:
                raise ValueError("online manifest/sidecar mismatch")
        blocks.append(block)
        online_entries.append(meta)
        reference_entries.append(ref)
    rank_manifest(online_entries)
    rank_manifest(reference_entries)
    parity = np.fromfile(online / "parity.bin", dtype=np.uint8)
    if digest(parity) != manifest["parity_sha256"] or not np.array_equal(
        encode(blocks), parity
    ):
        raise ValueError("online XOR parity mismatch")
    for lost in range(tp):
        restored = recover(
            [None if i == lost else b for i, b in enumerate(blocks)], parity
        )
        if not np.array_equal(restored, blocks[lost]):
            raise ValueError("online XOR recovery mismatch")
    return {
        "byte_exact": True,
        "tp_size": tp,
        "chunk_id": manifest["online_identity"]["chunk_id"],
        "recovered_ranks": list(range(tp)),
        "reference": str(reference),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("online", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_reference(args.online, args.reference), indent=2))


if __name__ == "__main__":
    main()
