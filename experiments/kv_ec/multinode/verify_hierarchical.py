"""Revalidate committed Node B hierarchical KV evidence without SGLang/GPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..rank_snapshot import validate_payload
from .kv_hierarchical import _decode_node, _sha
from .rs_codec import encode, recover
from ..xor_codec import encode as xor_encode


def verify(result_directory, local_snapshot):
    root = Path(result_directory)
    report = json.loads((root / "hierarchical-result.json").read_text())
    if report.get("state") != "COMMITTED" or report.get("transport") not in ("tcp", "rdma"):
        raise ValueError("result is not a committed TCP/RDMA stripe")
    if report.get("shared_node_failure_recovery") is not False:
        raise ValueError("two-node placement must not claim whole-node failure recovery")
    d0padded, d1padded = (root / "D0.bin").read_bytes(), (root / "D1.bin").read_bytes()
    parity_raw = (root / "P.bin").read_bytes()
    stripe = report["stripe"]
    width = stripe["padded_length"]
    if any(len(raw) != width for raw in (d0padded, d1padded, parity_raw)):
        raise ValueError("stored RS block lengths do not match committed metadata")
    d0, d1, parity = [np.frombuffer(raw, dtype=np.uint8).copy() for raw in (d0padded, d1padded, parity_raw)]
    d0raw = d0padded[:stripe["D0"]["length"]]
    d1raw = d1padded[:stripe["D1"]["length"]]
    if np.any(d0[stripe["D0"]["length"]:]) or np.any(d1[stripe["D1"]["length"]:]):
        raise ValueError("RS data-block padding is not zero")
    checksums = {"D0": _sha(d0raw), "D1": _sha(d1raw), "P": _sha(parity_raw)}
    for role in ("D0", "D1", "P"):
        if checksums[role] != stripe[role]["checksum"]:
            raise ValueError(f"{role} checksum mismatch")
    d0restored, _ = recover(data0=None, data1=d1, parity=parity,
                            lengths=(stripe["D0"]["length"], stripe["D1"]["length"]))
    _, d1restored = recover(data0=d0, data1=None, parity=parity,
                            lengths=(stripe["D0"]["length"], stripe["D1"]["length"]))
    _, _ = recover(data0=d0, data1=d1, parity=None,
                   lengths=(stripe["D0"]["length"], stripe["D1"]["length"]))
    _, _, rebuilt_parity = encode(d0, d1)
    if not np.array_equal(rebuilt_parity, parity):
        raise ValueError("RS parity recomputation mismatch")
    if d0restored.tobytes() != d0raw or d1restored.tobytes() != d1raw:
        raise ValueError("RS data-block recovery mismatch")
    headers = []
    for raw in (d0raw, d1raw):
        header, ranks = _decode_node(raw)
        headers.append(header)
        if len(ranks) != header['tp_size']:
            raise ValueError("node snapshot rank count mismatch")
    from .kv_hierarchical import _load_node, _compatible
    from ..rank_snapshot import rank_manifest
    _compatible(*(rank_manifest(header['rank_metadata']) for header in headers))
    if report.get('tp_size', headers[0]['tp_size']) != headers[0]['tp_size']:
        raise ValueError('report TP mismatch')
    _, local_blocks, _, _, _, local_header, _, _ = _load_node(local_snapshot, headers[1]['node_id'])
    # v1 historically omitted capture_compatibility; decoding already derives
    # and verifies it. Compare every other field and exact payload bytes.
    local_identity = {k: v for k, v in local_header.items() if k != 'capture_compatibility'}
    stored_identity = {k: v for k, v in headers[1].items() if k != 'capture_compatibility'}
    _, stored_blocks = _decode_node(d1raw)
    if local_identity != stored_identity or any(
            not np.array_equal(a, b) for a, b in zip(local_blocks, stored_blocks)):
        raise ValueError('local snapshot does not match committed D1')
    local_parity = xor_encode(local_blocks)
    if _sha(local_parity.tobytes()) != report["local_xor"]["parity_sha256"]:
        raise ValueError("local XOR parity checksum mismatch")
    for rank, block in enumerate(local_blocks):
        report_metadata = json.loads((Path(local_snapshot) / f"rank_{rank}.json").read_text())
        validate_payload(block, report_metadata)
        if _sha(block.tobytes()) != report_metadata["sha256"]:
            raise ValueError("local rank checksum mismatch")
    return {"state": "PASS", "transport": report["transport"],
            "tp_size": headers[0]['tp_size'],
            "shared_node_failure_recovery": False, "rs_blocks_sha256": checksums,
            "D0_and_D1_recovery_byte_exact": True, "P_recomputation_verified": True,
            "local_xor_parity_verified": True, "per_rank_snapshot_checksums_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--local-snapshot", required=True, type=Path,
                        help="Node B snapshot directory matching D1")
    args = parser.parse_args()
    print(json.dumps(verify(args.result_directory, args.local_snapshot), indent=2))


if __name__ == "__main__":
    main()
