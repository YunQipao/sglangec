"""Bounded, pickle-free framing for the two-logical-node EC prototype.

Payload bytes are always opaque. This module does not persist them.
"""

import hashlib
import json
import struct

PROTOCOL = "sglang-kv-ec-relay-v1"
MAX_HEADER_BYTES = 16 * 1024
MAX_CHUNK_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
REQUIRED = (
    "transaction_id", "stripe_id", "checkpoint_id", "rank_id", "chunk_index",
    "chunk_offset", "chunk_length", "total_length", "source_node",
    "destination_node", "chunk_sha256", "manager_epoch", "protocol_version",
)


def validate_header(header, payload_length=None):
    if not isinstance(header, dict) or any(k not in header for k in REQUIRED):
        raise ValueError("relay_header_missing_fields")
    if header["protocol_version"] != PROTOCOL:
        raise ValueError("relay_protocol_version_mismatch")
    for key in ("transaction_id", "stripe_id", "checkpoint_id", "source_node",
                "destination_node", "manager_epoch"):
        if not isinstance(header[key], str) or not header[key] or len(header[key]) > 256:
            raise ValueError("relay_invalid_" + key)
    for key in ("rank_id", "chunk_index", "chunk_offset", "chunk_length", "total_length"):
        if type(header[key]) is not int or header[key] < 0:
            raise ValueError("relay_invalid_" + key)
    length, offset, total = header["chunk_length"], header["chunk_offset"], header["total_length"]
    if not 0 < length <= MAX_CHUNK_BYTES or total <= 0 or total > MAX_TOTAL_BYTES:
        raise ValueError("relay_chunk_or_total_limit")
    if offset + length > total or (payload_length is not None and payload_length != length):
        raise ValueError("relay_length_mismatch")
    if not isinstance(header["chunk_sha256"], str) or len(header["chunk_sha256"]) != 64:
        raise ValueError("relay_invalid_sha256")
    return header


def encode_frame(header, payload):
    payload = bytes(payload)
    header = dict(header, chunk_length=len(payload))
    header.setdefault("protocol_version", PROTOCOL)
    header.setdefault("chunk_sha256", hashlib.sha256(payload).hexdigest())
    validate_header(header, len(payload))
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError("relay_header_too_large")
    return struct.pack("!I", len(raw)) + raw + payload


def recv_exact(sock, size):
    parts = []
    while size:
        part = sock.recv(size)
        if not part:
            raise ConnectionError("relay_peer_closed_mid_frame")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def recv_frame(sock):
    header_size, = struct.unpack("!I", recv_exact(sock, 4))
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise ValueError("relay_header_size_limit")
    header = json.loads(recv_exact(sock, header_size).decode("utf-8"))
    validate_header(header)
    payload = recv_exact(sock, header["chunk_length"])
    validate_header(header, len(payload))
    if hashlib.sha256(payload).hexdigest() != header["chunk_sha256"]:
        raise ValueError("relay_chunk_checksum_mismatch")
    return header, payload


def encode_ack(ack):
    raw = json.dumps(ack, separators=(",", ":"), sort_keys=True).encode()
    return struct.pack("!I", len(raw)) + raw


def recv_ack(sock):
    size, = struct.unpack("!I", recv_exact(sock, 4))
    if not 0 < size <= 4096:
        raise ValueError("relay_ack_size_limit")
    ack = json.loads(recv_exact(sock, size).decode("utf-8"))
    if ack.get("ok") is not True:
        raise RuntimeError("gateway_rejected:" + str(ack.get("reason", "unknown")))
    return ack
