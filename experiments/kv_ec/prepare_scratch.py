"""Explicit offline preparation of three DEBUG_SAFE rank snapshots.

Never called by capture or FAST_PARITY. No online pools or CUDA operations.
Provenance is cross-checked against local request/startup evidence, not attested.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from .gpu_scratch_restore import validate_snapshot_sequence
from .rank_snapshot import assemble, verify


def validate_capture_evidence(manifests, request, initialization):
    coverage = validate_snapshot_sequence(manifests)
    if (request.get("kind") != "real_server_artificial_prompt"
            or request.get("input_token_count") != 768
            or not request.get("output_ids")
            or request.get("finish_reason", {}).get("type") not in ("length", "stop")):
        raise ValueError("requires completed real artificial 768-token request evidence")
    request_hash = hashlib.sha256(request["request_id"].encode()).hexdigest()
    chunks = request.get("input_chunks", [])
    if len(chunks) != 3:
        raise ValueError("request evidence needs three token chunk hashes; rerun request_debug")
    identity = initialization["identity"]
    for manifest, chunk in zip(manifests, chunks):
        if (manifest["request_id_encoding"] != "sha256"
                or manifest["request_id"] != request_hash):
            raise ValueError("request ID mismatch")
        if str(Path(request["model_identifier"]).resolve()) != manifest["model_identifier"]:
            raise ValueError("request model mismatch")
        for key in ("model_identifier", "model_sha256", "source_commit", "core_diff_sha256"):
            if manifest[key] != identity[key]:
                raise ValueError(f"startup identity mismatch: {key}")
        for key in ("token_start", "token_count", "token_ids_sha256"):
            if manifest[key] != chunk[key]:
                raise ValueError(f"request token chunk mismatch: {key}")
        for rank in manifest["ranks"]:
            if rank.get("prompt_token_count") != 768:
                raise ValueError("rank did not capture a 768-token request")
    return coverage


def prepare(directories, request_path, initialization_path, *, wait_s=30):
    if not math.isfinite(wait_s) or wait_s < 0:
        raise ValueError("wait must be finite and nonnegative")
    directories = [Path(path) for path in directories]
    if len(directories) != 3 or len(set(path.resolve() for path in directories)) != 3:
        raise ValueError("three distinct chunk directories required")
    # Preflight all outputs before writing any; an interrupted preparation is
    # deliberately not retried over existing evidence.
    for directory in directories:
        for name in ("manifest.json", "recovery.json"):
            if (directory / name).exists():
                raise FileExistsError(f"refusing to overwrite {directory / name}")
    deadline = time.monotonic() + wait_s
    while True:
        missing = []
        for directory in directories:
            if list(directory.glob("rank_*.error.json")):
                raise ValueError(f"worker capture failed: {directory}")
            for rank in range(4):
                for suffix in ("bin", "json", "timing.json", "diagnostic.json"):
                    path = directory / f"rank_{rank}.{suffix}"
                    if not path.is_file():
                        missing.append(str(path))
        if not missing:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"incomplete capture; missing files: {missing}")
        time.sleep(min(0.1, remaining))
    manifests = []
    for directory in directories:
        manifest, blocks = assemble(directory)
        manifests.append(manifest)
        del blocks
    request = json.loads(Path(request_path).read_text())
    initialization = json.loads(Path(initialization_path).read_text())
    coverage = validate_capture_evidence(manifests, request, initialization)
    reports = [verify(directory) for directory in directories]
    return {"status": "verified", "coverage": coverage, "chunks": reports,
            "online_kv_pool_modified": False,
            "provenance_is_not_independently_attested": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs=3)
    parser.add_argument("--request-evidence", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--wait-s", type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(prepare(args.directories, args.request_evidence,
                             args.initialization, wait_s=args.wait_s), indent=2))


if __name__ == "__main__":
    main()
