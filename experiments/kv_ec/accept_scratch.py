"""Validate a saved scratch report and its real local input evidence, without CUDA."""

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .gpu_scratch_restore import load_real_snapshot
from .prepare_scratch import validate_capture_evidence

METRICS = ("xor_ms", "cpu_recovery_ms", "h2d_restore_ms", "gpu_verification_ms")


def accept(report, request, initialization, *, snapshot_root=None):
    if (report.get("tool") != "gpu_scratch_restore_sequence"
            or report.get("status") != "byte_exact"
            or report.get("online_kv_pool_modified") is not False
            or report.get("repeat") != 10
            or report.get("erasure_cases") != 12
            or not str(report.get("device", "")).startswith("cuda")):
        raise ValueError("not a successful isolated 12-case, 10-repeat CUDA report")
    chunks = report["chunks"]
    if len(chunks) != 3:
        raise ValueError("three chunk results required")
    manifests = []
    for index, chunk in enumerate(chunks):
        directory = (Path(snapshot_root) / chunk["snapshot_id"]
                     if snapshot_root is not None else Path(chunk["snapshot_dir"]))
        if snapshot_root is not None and Path(chunk["snapshot_id"]).name != chunk["snapshot_id"]:
            raise ValueError("snapshot ID must be a directory basename")
        manifest, blocks = load_real_snapshot(directory)
        del blocks
        manifests.append(manifest)
        if chunk.get("index") != index:
            raise ValueError("chunk index mismatch")
        for key in ("snapshot_id", "token_start", "token_count"):
            if chunk[key] != manifest[key]:
                raise ValueError(f"report snapshot mismatch: {key}")
        if chunk["rank_sha256"] != [rank["sha256"] for rank in manifest["ranks"]]:
            raise ValueError("report payload digest mismatch")
        ranks = chunk["ranks"]
        if [rank["lost_rank"] for rank in ranks] != list(range(4)):
            raise ValueError("missing or duplicate rank result")
        for rank in ranks:
            if (rank["byte_exact"] is not True
                    or rank["restored_bytes"] != manifest["packed_byte_count"]):
                raise ValueError("restore is not byte exact or has incorrect size")
            for metric in METRICS:
                summary = rank[metric]
                samples = summary["samples_ms"]
                if (len(samples) != 10 or any(
                        isinstance(value, bool) or not isinstance(value, (float, int))
                        or not math.isfinite(value) or value < 0 for value in samples)):
                    raise ValueError(f"invalid measurement samples: {metric}")
                for name, value in (("median_ms", statistics.median(samples)),
                                    ("p95_ms", float(np.percentile(samples, 95)))):
                    if not math.isclose(summary[name], value, rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f"incorrect summary: {metric}.{name}")
    coverage = validate_capture_evidence(manifests, request, initialization)
    if report["coverage"] != coverage:
        raise ValueError("report coverage mismatch")
    return {"status": "accepted", "erasure_cases": 12, "measurements": 120,
            "coverage": coverage, "online_kv_pool_modified": False,
            "note": "Local evidence consistency check, not independent GPU attestation or online recovery"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--request-evidence", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path,
                        help="explicit relocated evidence root; original report is never changed")
    args = parser.parse_args()
    print(json.dumps(accept(json.loads(args.report.read_text()),
                            json.loads(args.request_evidence.read_text()),
                            json.loads(args.initialization.read_text()),
                            snapshot_root=args.snapshot_root), indent=2))


if __name__ == "__main__":
    main()
