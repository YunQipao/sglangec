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

from ..rank_snapshot import assemble, validate_payload
from ..xor_codec import encode as xor_encode, recover as xor_recover
from .rs_codec import CODEC, CODEC_VERSION, encode as rs_encode, recover as rs_recover
from .rdma_transport import RDMATransport, receive_file as rdma_receive_file
from .tcp_transport import TCPTransport, receive_file as tcp_receive_file

MAGIC = b"KVECNODE1"
ALIGNMENT_KEYS = ("model_identifier", "model_sha256", "dtype", "tp_size", "layers",
                  "token_start", "token_count", "pool_class", "source_commit", "core_diff_sha256")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _checkpoint_version(manifest):
    return manifest["ranks"][0].get("checkpoint_version", manifest["snapshot_id"])


def _load_node(directory, node_id):
    manifest, blocks = assemble(directory)
    if manifest["tp_size"] != 4 or manifest["dtype"] != "torch.bfloat16":
        raise ValueError("only verified Qwen3-8B BF16 TP=4 snapshots are supported")
    if (manifest["token_count"] != 256 or len(manifest["layers"]) != 36
            or manifest["layers"] != list(range(36))):
        raise ValueError("only complete 36-layer Qwen3-8B 256-token chunks are supported")
    if "qwen3-8b" not in manifest["model_identifier"].lower():
        raise ValueError("model identifier must name Qwen3-8B")
    if manifest["pool_class"].split(".")[-1] != "MHATokenToKVPool":
        raise ValueError("only the ordinary MHATokenToKVPool is supported")
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
    if header.get("format") != "kv-ec-node-block-v1" or header.get("tp_size") != 4:
        raise ValueError("unsupported or incomplete node-block metadata")
    if (not header.get("node_id") or not header.get("request_id")
            or not header.get("model_fingerprint") or not header.get("checkpoint_version")
            or len(header.get("rank_blocks", [])) != 4
            or len(header.get("rank_metadata", [])) != 4):
        raise ValueError("incomplete node-block identity or TP metadata")
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
        if record.get("tp_rank") != rank or record.get("length") != length or _sha(block.tobytes()) != record.get("source_checksum"):
            raise ValueError("rank payload checksum/metadata mismatch")
        validate_payload(block, header["rank_metadata"][rank])
        blocks.append(block)
    if offset != len(view):
        raise ValueError("trailing bytes in node block")
    return header, blocks


def _compatible(a, b):
    for key in ALIGNMENT_KEYS:
        if a.get(key) != b.get(key):
            raise ValueError(f"incompatible node snapshots: {key}")
    for rank in range(4):
        left, right = a["ranks"][rank]["layer_records"], b["ranks"][rank]["layer_records"]
        fields = ("layer_id", "dtype", "packed_shape", "k_offset", "k_nbytes", "v_offset", "v_nbytes")
        if [{key: item.get(key) for key in fields} for item in left] != [
                {key: item.get(key) for key in fields} for item in right]:
            raise ValueError("incompatible node snapshots: rank KV layout")


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
    else:
        transfer_seconds = rdma_receive_file(args.data_port, remote_path, device=args.rdma_device,
                                             bind_host=args.bind_host)
    remote_envelope = remote_path.read_bytes()
    remote_header, remote_blocks = _decode_node(remote_envelope)
    _compatible({**local_manifest, "ranks": local_manifest["ranks"]},
                {**local_manifest, "ranks": remote_header["rank_metadata"],
                 "layers": remote_header["layers"], "model_identifier": remote_header["rank_metadata"][0]["model_identifier"],
                 "model_sha256": remote_header["model_fingerprint"], "dtype": remote_header["dtype"],
                 "tp_size": remote_header["tp_size"], "token_start": remote_header["token_range"][0],
                 "token_count": remote_header["token_range"][1] - remote_header["token_range"][0],
                 "pool_class": remote_header["rank_metadata"][0]["pool_class"],
                 "source_commit": remote_header["rank_metadata"][0]["source_commit"],
                 "core_diff_sha256": remote_header["rank_metadata"][0]["core_diff_sha256"]})
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
    transport = TCPTransport() if args.transport == "tcp" else RDMATransport(device=args.rdma_device)
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
