"""Offline rank-bin validation, manifest assembly and exact XOR recovery.

No CUDA, model collectives or SGLang imports. Provenance is a declaration, not
independent attestation: inspect worker logs and request/regression evidence too.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

if __package__:
    from .snapshot import digest
    from .xor_codec import encode, recover
else:
    from snapshot import digest
    from xor_codec import encode, recover

FORMAT = "kv-ec-rank-bin-v1"
COMMON = ("format", "provenance", "snapshot_id", "request_id", "request_id_encoding",
          "model_identifier", "model_sha256", "source_commit", "core_diff_sha256",
          "tp_size", "layers", "token_start", "token_count", "token_ids_sha256",
          "dtype", "packed_byte_count", "pool_class", "lifetime", "completion_dependency")


def write_rank_files(directory, payload, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rank = metadata["tp_rank"]
    if payload.dtype != np.uint8 or payload.ndim != 1 or not payload.flags.c_contiguous:
        raise ValueError("flat contiguous raw bytes required")
    # No overwrite. A partial binary cannot pass validation without its sidecar.
    disk_started = time.perf_counter_ns()
    with (directory / f"rank_{rank}.bin").open("xb") as f:
        f.write(payload.tobytes())
    with (directory / f"rank_{rank}.json").open("x") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    performance = metadata.get("performance_ms")
    if performance is not None:
        performance["disk_write_ms"] = (time.perf_counter_ns() - disk_started) / 1e6


def _validate_payload(payload, meta, *, verify_checksums, compute_checksums):
    if meta.get("checksum_deferred") is True:
        verify_checksums = False
    if meta["format"] != FORMAT or meta["snapshot_completion_status"] != "complete":
        raise ValueError("incomplete or unsupported rank snapshot")
    if meta["provenance"] not in ("sglang_request", "synthetic_fixture"):
        raise ValueError("unknown provenance")
    if meta["dtype"] not in ("torch.float16", "torch.bfloat16") or meta["source_unchanged"] is not True:
        raise ValueError("unsupported dtype or missing read-only verification")
    if payload.dtype != np.uint8 or payload.ndim != 1 or payload.size != meta["packed_byte_count"]:
        raise ValueError("rank payload length/type mismatch")
    payload_sha256 = digest(payload) if compute_checksums else None
    if verify_checksums and payload_sha256 != meta["sha256"]:
        raise ValueError("rank payload checksum mismatch")
    if len(meta["physical_slots"]) != meta["token_count"] or meta["req_pool_idx"] <= 0 or meta["req_generation"] <= 0:
        raise ValueError("invalid request/slot metadata")
    if meta["token_start"] < 0 or meta["token_count"] < 1:
        raise ValueError("invalid logical range")
    if len(set(meta["physical_slots"])) != meta["token_count"] or any(s <= 0 for s in meta["physical_slots"]):
        raise ValueError("invalid physical slots")
    records = meta["layer_records"]
    if not records or [r["layer_id"] for r in records] != meta["layers"] or meta["layers"] != sorted(set(meta["layers"])):
        raise ValueError("layer order mismatch")
    offset = 0
    hashes = {"k": hashlib.sha256(), "v": hashlib.sha256()} if compute_checksums else None
    layer_hashes = {}
    dtype = getattr(torch, meta["dtype"].removeprefix("torch."))
    for record in records:
        if record["dtype"] != meta["dtype"]:
            raise ValueError("layer dtype mismatch")
        for kind in ("k", "v"):
            shape = record["packed_shape"][kind]
            nbytes = record[kind + "_nbytes"]
            if (len(shape) != 3 or any(not isinstance(d, int) or d <= 0 for d in shape)
                    or shape[0] != meta["token_count"] or math.prod(shape) * 2 != nbytes):
                raise ValueError("packed shape/byte count mismatch")
            if record[kind + "_offset"] != offset or offset + nbytes > payload.size:
                raise ValueError("noncanonical offsets or truncated data")
            segment = payload[offset:offset + nbytes]
            segment_sha256 = digest(segment) if compute_checksums else None
            if verify_checksums and segment_sha256 != record[kind + "_sha256"]:
                raise ValueError("layer K/V checksum mismatch")
            if compute_checksums:
                layer_hashes[(record["layer_id"], kind)] = segment_sha256
            # Metadata can reconstruct the exact tensor view without float
            # casts. ``np.frombuffer(bytes)`` may be read-only; make an
            # independent writable view before handing it to PyTorch so its
            # documented storage contract is satisfied even if a downstream
            # validator mutates the temporary tensor.
            restored_view = torch.from_numpy(segment.copy()).view(dtype).reshape(shape)
            if list(restored_view.shape) != shape or str(restored_view.dtype) != meta["dtype"]:
                raise ValueError("tensor metadata roundtrip failed")
            if compute_checksums:
                hashes[kind].update(segment)
            offset += nbytes
    if offset != payload.size:
        raise ValueError("unaccounted bytes")
    if compute_checksums:
        for kind, hasher in hashes.items():
            aggregate_sha256 = hasher.hexdigest()
            if verify_checksums and aggregate_sha256 != meta[kind + "_sha256"]:
                raise ValueError("aggregate K/V checksum mismatch")
    # Deferred FAST_PARITY submissions use this derived record to move all
    # checksum work to the manager after it owns an immutable CPU copy.
    if compute_checksums and not verify_checksums:
        return {
            "sha256": payload_sha256,
            "k_sha256": hashes["k"].hexdigest(),
            "v_sha256": hashes["v"].hexdigest(),
            "layer_sha256": layer_hashes,
        }


def validate_payload(payload, meta, *, verify_checksums=True):
    return _validate_payload(
        payload, meta, verify_checksums=verify_checksums, compute_checksums=True
    )


def validate_payload_structure(payload, meta):
    """Validate layout after the capture scan already hashed every byte.

    The parity manager independently calls validate_payload() on every
    received rank, so this helper does not replace the integrity boundary.
    """
    _validate_payload(payload, meta, verify_checksums=False, compute_checksums=False)


def assemble(directory):
    directory = Path(directory)
    if list(directory.glob("rank_*.error.json")):
        raise ValueError("worker reported capture failure; see rank error sidecars")
    first = json.loads((directory / "rank_0.json").read_text())
    tp = first["tp_size"]
    if tp not in (1, 2, 4):
        raise ValueError("TP must be 1, 2 or 4")
    expected = {f"rank_{r}.json" for r in range(tp)}
    diagnostic = {f"rank_{r}.{suffix}.json" for r in range(tp)
                  for suffix in ("timing", "diagnostic")}
    if {p.name for p in directory.glob("rank_*.json")} - diagnostic != expected:
        raise ValueError("missing, duplicate or unexpected rank sidecars")
    if {p.name for p in directory.glob("rank_*.bin")} != {f"rank_{r}.bin" for r in range(tp)}:
        raise ValueError("missing or unexpected rank binaries")
    entries, blocks = [], []
    for rank in range(tp):
        meta = json.loads((directory / f"rank_{rank}.json").read_text())
        if meta["tp_rank"] != rank or any(meta[k] != first[k] for k in COMMON):
            raise ValueError("incompatible rank/model/request/range/capture")
        if [r["packed_shape"] for r in meta["layer_records"]] != [r["packed_shape"] for r in first["layer_records"]]:
            raise ValueError("unequal shard shapes; no padding supported")
        payload = np.fromfile(directory / f"rank_{rank}.bin", dtype=np.uint8)
        if meta.get("checksum_deferred"):
            raise ValueError("offline snapshots require recorded, non-deferred checksums")
        validate_payload(payload, meta)
        entries.append(meta)
        blocks.append(payload)
    manifest = {k: first[k] for k in COMMON}
    manifest.update(snapshot_completion_status="complete", ranks=entries,
                    provenance_is_not_independently_attested=True,
                    layout="ascending_layer_then_K_then_V_logical_token_order",
                    note="Physical slots, PID, device and request row generations are rank-local.")
    return manifest, blocks


def verify(directory, *, write=True):
    manifest, blocks = assemble(directory)
    tp = manifest["tp_size"]
    report = {"snapshot_id": manifest["snapshot_id"], "tp": tp,
              "provenance": manifest["provenance"], "packed_byte_count": blocks[0].size,
              "xor_performed": tp > 1, "source_unchanged_recorded": True,
              "node_failure_test": False, "checks": []}
    if tp > 1:
        start = time.perf_counter_ns()
        parity = encode(blocks)  # CPU DRAM only, do not persist parity by default.
        report["xor_ms"] = (time.perf_counter_ns() - start) / 1e6
        report["parity_sha256"] = digest(parity)
        originals = [digest(b) for b in blocks]
        for lost in range(tp):
            survivors = [None if rank == lost else b for rank, b in enumerate(blocks)]
            start = time.perf_counter_ns()
            restored = recover(survivors, parity)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if not np.array_equal(restored, blocks[lost]) or digest(restored) != originals[lost]:
                raise AssertionError("byte-exact recovery failed")
            # Includes all K/V segment hashes and exact dtype/shape reconstruction.
            validate_payload(restored, manifest["ranks"][lost])
            report["checks"].append({"lost_rank": lost, "byte_exact": True, "sha256_equal": True,
                                     "all_K_V_checksums_equal": True, "dtype_shape_valid": True,
                                     "recovery_ms": elapsed})
        if originals != [digest(b) for b in blocks]:
            raise AssertionError("input blocks were changed")
    report["status"] = "verified"
    if write:
        # Reserve both outputs first, refusing to clobber prior evidence.
        directory = Path(directory)
        if (directory / "manifest.json").exists() or (directory / "recovery.json").exists():
            raise FileExistsError("manifest or recovery already exists; use --check-only to revalidate")
        for filename, obj in (("manifest.json", manifest), ("recovery.json", report)):
            with (directory / filename).open("x") as f:
                json.dump(obj, f, indent=2)
                f.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, write=not args.check_only), indent=2))
