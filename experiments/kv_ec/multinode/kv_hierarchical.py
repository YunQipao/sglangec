"""Offline two-layer protection for captured Qwen3-8B TP=4 KV chunks.

This deliberately consumes the existing rank snapshot format after capture. It
does not hook the SGLang scheduler or claim online checkpoint semantics.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time

import numpy as np

from ..rank_snapshot import (COMMON, KV_LAYOUT_FIELDS, assemble, capture_compatibility,
                             rank_manifest, validate_model_identity, validate_payload)
from ..xor_codec import encode as xor_encode, recover as xor_recover
from .rs_codec import CODEC, CODEC_VERSION, encode as rs_encode, recover as rs_recover
from .rdma_transport import RDMATransport, receive_file as rdma_receive_file
from .tcp_transport import TCPTransport, receive_file as tcp_receive_file

MAGIC = b"KVECNODE1"
ALIGNMENT_KEYS = ("format", "model_sha256", "dtype", "tp_size", "layers",
                  "token_start", "token_count", "pool_class", "packed_byte_count", "core_diff_sha256")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _checkpoint_version(manifest):
    return manifest["ranks"][0].get("checkpoint_version", manifest["snapshot_id"])


def _validate_node_manifest(manifest):
    derived = rank_manifest(manifest["ranks"])
    for key in COMMON:
        if manifest.get(key) != derived[key]:
            raise ValueError(f"node snapshot metadata mismatch: {key}")
    validate_model_identity(manifest)
    if manifest["tp_size"] != 4 or manifest["dtype"] != "torch.bfloat16":
        raise ValueError("only verified Qwen3-8B BF16 TP=4 snapshots are supported")
    if (manifest["token_count"] != 256 or len(manifest["layers"]) != 36
            or manifest["layers"] != list(range(36))):
        raise ValueError("only complete 36-layer Qwen3-8B 256-token chunks are supported")
    if manifest["pool_class"] != "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool":
        raise ValueError("only the ordinary MHATokenToKVPool is supported")
    for meta in manifest["ranks"]:
        if any(record["packed_shape"] != {"k": [256, 2, 128], "v": [256, 2, 128]}
               for record in meta["layer_records"]):
            raise ValueError("unsupported Qwen3-8B TP=4 rank KV layout")
    return capture_compatibility(manifest)


def _load_node(directory, node_id):
    manifest, blocks = assemble(directory)
    compatibility = _validate_node_manifest(manifest)
    for payload, metadata in zip(blocks, manifest["ranks"]):
        validate_payload(payload, metadata)
    xor_started = time.perf_counter()
    parity = xor_encode(blocks)
    xor_encode_ms = (time.perf_counter() - xor_started) * 1000
    checks = []
    xor_recovery_started = time.perf_counter()
    for rank, expected in enumerate(blocks):
        shards = [None if i == rank else block for i, block in enumerate(blocks)]
        restored = xor_recover(shards, parity)
        if not np.array_equal(restored, expected):
            raise ValueError(f"local XOR rank {rank} recovery mismatch")
        validate_payload(restored, manifest["ranks"][rank])
        checks.append({"rank": rank, "byte_exact": True, "sha256": _sha(restored.tobytes())})
    xor_recovery_ms = (time.perf_counter() - xor_recovery_started) * 1000
    header = {
        "format": "kv-ec-node-block-v1", "node_id": node_id,
        "request_id": manifest["request_id"], "snapshot_id": manifest["snapshot_id"],
        "chunk_id": manifest["snapshot_id"],
        "model_fingerprint": manifest["model_sha256"], "checkpoint_version": _checkpoint_version(manifest),
        "capture_compatibility": compatibility,
        "tp_size": 4, "dtype": manifest["dtype"], "layers": manifest["layers"],
        "layer_range": [min(manifest["layers"]), max(manifest["layers"]) + 1],
        "token_range": [manifest["token_start"], manifest["token_start"] + manifest["token_count"]],
        "rank_blocks": [{"tp_rank": rank, "length": int(block.size),
                         "kv_head_range": [rank * (8 // 4), (rank + 1) * (8 // 4)],
                         "source_checksum": _sha(block.tobytes()),
                         "layer_records": manifest["ranks"][rank]["layer_records"]}
                        for rank, block in enumerate(blocks)],
        "rank_metadata": manifest["ranks"],
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    body = bytearray(MAGIC + struct.pack("!Q", len(encoded_header)) + encoded_header)
    for block in blocks:
        body += struct.pack("!Q", int(block.size)) + block.tobytes()
    return manifest, blocks, parity, checks, bytes(body), header, xor_encode_ms, xor_recovery_ms


def _decode_node(data):
    view = memoryview(data)
    if len(view) < len(MAGIC) + 8 or bytes(view[:len(MAGIC)]) != MAGIC:
        raise ValueError("invalid node-block magic")
    offset = len(MAGIC)
    hlen = struct.unpack_from("!Q", view, offset)[0]
    offset += 8
    if hlen <= 0 or offset + hlen > len(view):
        raise ValueError("truncated node-block header")
    header = json.loads(bytes(view[offset:offset + hlen]))
    offset += hlen
    if (header.get("format") != "kv-ec-node-block-v1" or
            type(header.get("tp_size")) is not int or header["tp_size"] != 4):
        raise ValueError("unsupported or incomplete node-block metadata")
    if (not header.get("node_id") or not header.get("request_id")
            or not header.get("model_fingerprint") or not header.get("checkpoint_version")
            or len(header.get("rank_blocks", [])) != 4
            or len(header.get("rank_metadata", [])) != 4):
        raise ValueError("incomplete node-block identity or TP metadata")
    manifest = rank_manifest(header["rank_metadata"])
    compatibility = _validate_node_manifest(manifest)
    expected_header = {
        "model_fingerprint": manifest["model_sha256"], "tp_size": manifest["tp_size"],
        "dtype": manifest["dtype"], "layers": manifest["layers"],
        "request_id": manifest["request_id"], "snapshot_id": manifest["snapshot_id"],
        "chunk_id": manifest["snapshot_id"], "checkpoint_version": _checkpoint_version(manifest),
        "layer_range": [0, 36],
        "token_range": [manifest["token_start"], manifest["token_start"] + manifest["token_count"]],
    }
    for key, expected in expected_header.items():
        if header.get(key) != expected:
            raise ValueError(f"node-block header/rank metadata mismatch: {key}")
    for key in ("layers", "layer_range", "token_range"):
        if any(type(value) is not int for value in header[key]):
            raise ValueError(f"node-block header integer range required: {key}")
    # Old v1 envelopes have no profile field. Derive it from their unmodified
    # provenance too; a received compatibility claim cannot widen this policy.
    if header.get("capture_compatibility", compatibility) != compatibility:
        raise ValueError("node-block capture compatibility profile mismatch")
    blocks = []
    for rank in range(4):
        if offset + 8 > len(view):
            raise ValueError("truncated node-block length")
        length = struct.unpack_from("!Q", view, offset)[0]
        offset += 8
        if length <= 0 or offset + length > len(view):
            raise ValueError("truncated node-block payload")
        block = np.frombuffer(view[offset:offset + length], dtype=np.uint8).copy()
        offset += length
        record = header["rank_blocks"][rank]
        if (type(record.get("tp_rank")) is not int or record["tp_rank"] != rank or
                type(record.get("length")) is not int or record["length"] != length or
                _sha(block.tobytes()) != record.get("source_checksum")):
            raise ValueError("rank payload checksum/metadata mismatch")
        if (record.get("kv_head_range") != [rank * 2, (rank + 1) * 2] or
                record.get("layer_records") != header["rank_metadata"][rank]["layer_records"]):
            raise ValueError("rank KV layout/header mismatch")
        validate_payload(block, header["rank_metadata"][rank])
        blocks.append(block)
    if offset != len(view):
        raise ValueError("trailing bytes in node block")
    return header, blocks


def _compatible(a, b):
    left_profile = _validate_node_manifest(a)
    right_profile = _validate_node_manifest(b)
    for key in ALIGNMENT_KEYS:
        if a.get(key) != b.get(key):
            raise ValueError(f"incompatible node snapshots: {key}")
    if left_profile != right_profile:
        raise ValueError("source_commit compatibility not proven; recapture both nodes at a common reviewed source")
    for rank in range(4):
        left, right = a["ranks"][rank]["layer_records"], b["ranks"][rank]["layer_records"]
        fields = KV_LAYOUT_FIELDS
        if [{key: item.get(key) for key in fields} for item in left] != [
                {key: item.get(key) for key in fields} for item in right]:
            raise ValueError("incompatible node snapshots: rank KV layout")
    return left_profile


def _persist(directory, name, data):
    path = directory / name
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != data:
        raise IOError(f"persisted data verification failed: {name}")
    return path


def _write_report(directory, report):
    path = directory / "hierarchical-result.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    fields = ["transport", "local_xor_encode_ms", "rs_encode_ms", "transfer_ms",
              "protection_completion_ms", "xor_recovery_ms", "rs_recovery_ms",
              "cross_node_bytes", "cpu_seconds", "host_dram_bytes", "ttft_ms", "tpot_ms",
              "throughput_tokens_s", "cpu_utilization_percent", "host_dram_peak_rss_bytes", "state"]
    with (directory / "hierarchical-result.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: report.get(key) for key in fields})


def run_receive(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    local_manifest, local_blocks, local_parity, xor_checks, local_envelope, local_header, xor_encode_ms, xor_recovery_ms = _load_node(args.snapshot, args.node_id)
    _persist(out, f"{args.node_id}-local-xor.bin", local_parity.tobytes())
    _persist(out, f"{args.node_id}-node-block.bin", local_envelope)
    remote_path = out / "remote-node-block.bin"
    if args.transport == "tcp":
        transfer_seconds = tcp_receive_file(args.bind_host, args.data_port, remote_path, timeout=args.timeout)
        rdma_receive_stats = None
    else:
        rdma_receive_stats = rdma_receive_file(
            args.data_port, remote_path, device=args.rdma_device,
            bind_host=args.bind_host, gid_index=args.rdma_gid_index,
            port_num=args.rdma_port_num, timeout=args.timeout, return_stats=True,
        )
        transfer_seconds = rdma_receive_stats["elapsed_s"]
    remote_envelope = remote_path.read_bytes()
    remote_header, remote_blocks = _decode_node(remote_envelope)
    remote_manifest = rank_manifest(remote_header["rank_metadata"])
    compatibility = _compatible(local_manifest, remote_manifest)
    if remote_header["node_id"] == args.node_id:
        raise ValueError("node IDs must be distinct")
    d0raw, d1raw = remote_envelope, local_envelope
    width = max(len(d0raw), len(d1raw))
    d0 = np.frombuffer(d0raw + bytes(width - len(d0raw)), dtype=np.uint8).copy()
    d1 = np.frombuffer(d1raw + bytes(width - len(d1raw)), dtype=np.uint8).copy()
    rs_encode_started = time.perf_counter()
    _, _, parity = rs_encode(d0, d1)
    rs_encode_ms = (time.perf_counter() - rs_encode_started) * 1000
    rs_started = time.perf_counter()
    recovered0, _ = rs_recover(data0=None, data1=d1, parity=parity, lengths=(len(d0raw), len(d1raw)))
    _, recovered1 = rs_recover(data0=d0, data1=None, parity=parity, lengths=(len(d0raw), len(d1raw)))
    _, _ = rs_recover(data0=d0, data1=d1, parity=None, lengths=(len(d0raw), len(d1raw)))
    _, _, rebuilt_parity = rs_encode(d0, d1)
    rs_recovery_ms = (time.perf_counter() - rs_started) * 1000
    if (recovered0.tobytes() != d0raw or recovered1.tobytes() != d1raw
            or not np.array_equal(rebuilt_parity, parity)):
        raise ValueError("RS byte-exact recovery failed")
    _decode_node(recovered0.tobytes())
    _decode_node(recovered1.tobytes())
    _persist(out, "D0.bin", d0.tobytes())
    _persist(out, "D1.bin", d1.tobytes())
    _persist(out, "P.bin", parity.tobytes())
    after = resource.getrusage(resource.RUSAGE_SELF)
    request_metrics = _request_metrics(args.request_metrics)
    global_chunk_id = args.chunk_id or args.stripe_id
    stripe_common = {"global_stripe_id": args.stripe_id, "chunk_id": global_chunk_id,
                     "checkpoint_version": _checkpoint_version(local_manifest),
                     "model_fingerprint": local_manifest["model_sha256"]}
    report = {
        "experiment_id": args.experiment_id, "global_stripe_id": args.stripe_id,
        "chunk_id": global_chunk_id, "transport": args.transport,
        "shared_node_failure_recovery": False,
        "encoding_object": "complete_serialized_four_rank_node_snapshot",
        "snapshot_compatibility": {
            "format": local_manifest["format"], "profile": compatibility,
            "model_sha256": local_manifest["model_sha256"],
            "sources": [{"node_id": node_id, **{key: manifest[key] for key in
                         ("model_identifier", "source_commit", "core_diff_sha256")}}
                        for node_id, manifest in ((args.node_id, local_manifest),
                                                  (remote_header["node_id"], remote_manifest))],
        },
        "local_xor": {"node_id": args.node_id, "parity_sha256": _sha(local_parity.tobytes()),
                      "rank_recovery_checks": xor_checks},
        "stripe": {**stripe_common, "codec": CODEC, "codec_version": CODEC_VERSION,
                   "rs_parameters": {"k": 2, "m": 1, "field": "GF(2^8)", "coefficients": [1, 2]},
                   "D0": {"node_id": remote_header["node_id"], "request_id": remote_header["request_id"],
                          "local_snapshot_id": remote_header["snapshot_id"],
                          "tp_ranks": remote_header["rank_blocks"], "layer_range": remote_header["layer_range"],
                          "token_range": remote_header["token_range"], "length": len(d0raw), "checksum": _sha(d0raw)},
                   "D1": {"node_id": args.node_id, "request_id": local_manifest["request_id"],
                          "local_snapshot_id": local_manifest["snapshot_id"],
                          "tp_ranks": local_header["rank_blocks"], "layer_range": local_header["layer_range"],
                          "token_range": local_header["token_range"], "length": len(d1raw), "checksum": _sha(d1raw)},
                   "P": {"node_id": args.node_id, "buffer": "independent", "length": width,
                         "checksum": _sha(parity.tobytes())},
                   "padded_length": width, "padding_length": [width - len(d0raw), width - len(d1raw)],
                   "protection_state": "COMMITTED"},
        "verification": {"D0_lost_byte_exact": True, "D1_lost_byte_exact": True,
                         "P_lost_recomputed": True, "all_rank_sha256_valid": True},
        "local_xor_encode_ms": xor_encode_ms, "rs_encode_ms": rs_encode_ms,
        "transfer_ms": transfer_seconds * 1000,
        "protection_completion_ms": (time.perf_counter() - started) * 1000,
        "xor_recovery_ms": xor_recovery_ms, "rs_recovery_ms": rs_recovery_ms,
        "cross_node_bytes": len(remote_envelope), "cpu_seconds": (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
        "host_dram_bytes": sum(b.nbytes for b in local_blocks) + local_parity.nbytes + len(remote_envelope) + len(parity),
        "host_dram_peak_rss_bytes": after.ru_maxrss * 1024,
        "gpu_memory_bytes": None,
        "request_metrics": request_metrics,
        "ttft_ms": request_metrics.get("ttft_ms"), "tpot_ms": request_metrics.get("tpot_ms"),
        "throughput_tokens_s": request_metrics.get("throughput_tokens_s"),
        "checkpoint": {"data_transfer_completed": True, "rank_checksums_valid": True,
                       "local_xor_verified": True, "rs_recoveries_verified": True},
        "rdma_receive": rdma_receive_stats,
    }
    report["cpu_utilization_percent"] = (
        100 * report["cpu_seconds"] / (report["protection_completion_ms"] / 1000)
        if report["protection_completion_ms"] else 0.0)
    report["state"] = "COMMITTED"
    report["checkpoint"]["state"] = "COMMITTED"
    _write_report(out, report)
    return report


def run_send(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest, blocks, parity, checks, envelope, header, xor_encode_ms, xor_recovery_ms = _load_node(args.snapshot, args.node_id)
    _persist(out, f"{args.node_id}-local-xor.bin", parity.tobytes())
    _persist(out, f"{args.node_id}-node-block.bin", envelope)
    transport = TCPTransport() if args.transport == "tcp" else RDMATransport(
        device=args.rdma_device, gid_index=args.rdma_gid_index,
        port_num=args.rdma_port_num, timeout=args.timeout,
    )
    if args.transport == "rdma":
        send_stats = transport.send_with_stats(args.peer, args.data_port, envelope)
        elapsed = send_stats["elapsed_s"]
        transport.close()
    else:
        send_stats = None
        elapsed = transport.send(args.peer, args.data_port, envelope)
    if args.transport == "rdma" and transport.name != "rdma":
        raise ValueError("RDMA transport identity mismatch")
    request_metrics = _request_metrics(args.request_metrics)
    report = {"state": "SENT", "transport": args.transport, "node_id": args.node_id,
              "request_id": manifest["request_id"], "snapshot_id": manifest["snapshot_id"],
              "global_stripe_id": args.stripe_id, "local_xor_parity_sha256": _sha(parity.tobytes()),
              "local_xor_rank_recovery_checks": checks, "cross_node_bytes": len(envelope),
              "local_xor_encode_ms": xor_encode_ms, "xor_recovery_ms": xor_recovery_ms,
              "request_metrics": request_metrics, "ttft_ms": request_metrics.get("ttft_ms"),
              "tpot_ms": request_metrics.get("tpot_ms"),
              "throughput_tokens_s": request_metrics.get("throughput_tokens_s"),
              "transfer_ms": elapsed * 1000, "shared_node_failure_recovery": False,
              "rdma_send": send_stats,
              "host_dram_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "gpu_memory_bytes": None,
              "note": "Node B must validate and commit the global stripe."}
    _write_report(out, report)
    return report


def _request_metrics(path):
    if path is None:
        return {"ttft_ms": None, "tpot_ms": None, "throughput_tokens_s": None,
                "gpu_memory_bytes": None,
                "note": "inference metrics unavailable; pass request_debug response JSON to attach client metrics"}
    record = json.loads(Path(path).read_text())
    metrics = record.get("client_metrics_ms", {})
    total_ms = metrics.get("request_total_ms")
    count = len(record.get("output_ids", []))
    return {"ttft_ms": metrics.get("ttft_ms"), "tpot_ms": metrics.get("tpot_ms"),
            "throughput_tokens_s": count * 1000 / total_ms if total_ms else None,
            "gpu_memory_bytes": None,
            "request_id": record.get("request_id"),
            "note": metrics.get("note", "client-side streaming metrics include HTTP/SSE overhead")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    for role in ("send", "receive"):
        p = sub.add_parser(role)
        p.add_argument("--snapshot", type=Path, required=True)
        p.add_argument("--node-id", required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--transport", choices=("tcp", "rdma"), required=True)
        p.add_argument("--data-port", type=int, default=19481)
        p.add_argument("--stripe-id", required=True)
        p.add_argument("--chunk-id", help="shared logical chunk ID; defaults to --stripe-id")
        p.add_argument("--experiment-id", default="qwen3-kv-hierarchical")
        p.add_argument("--timeout", type=float, default=120)
        p.add_argument("--rdma-device", default="-")
        p.add_argument("--rdma-gid-index", type=int)
        p.add_argument("--rdma-port-num", type=int, default=1)
        p.add_argument("--request-metrics", type=Path,
                       help="request_debug response JSON; attaches TTFT/TPOT/client throughput")
        if role == "send":
            p.add_argument("--peer", required=True)
        else:
            p.add_argument("--bind-host", default="0.0.0.0")
    args = parser.parse_args()
    result = run_send(args) if args.role == "send" else run_receive(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
