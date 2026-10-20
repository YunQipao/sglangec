"""Length-prefixed Unix-socket protocol for CPU KV parity checkpoints.

The protocol is deliberately small and pickle-free: a JSON header describes a
single rank payload, followed by exactly ``payload_bytes`` raw uint8 bytes.
"""

import json
import struct
import time


PROTOCOL = "kv-ec-parity-v1"
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def encode_header(metadata):
    header = json.dumps(dict(metadata, protocol=PROTOCOL), separators=(",", ":")).encode()
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("checkpoint metadata header is too large")
    return struct.pack("!I", len(header)) + header


def decode_header(raw):
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError("checkpoint metadata header is too large")
    metadata = json.loads(raw.decode())
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError("unsupported checkpoint protocol")
    payload_bytes = metadata.get("payload_bytes")
    if not isinstance(payload_bytes, int) or not 0 <= payload_bytes <= MAX_PAYLOAD_BYTES:
        raise ValueError("invalid checkpoint payload length")
    return metadata


def recv_exact(sock, size, deadline=None):
    chunks = []
    remaining = size
    while remaining:
        if deadline is not None:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise TimeoutError("checkpoint receive deadline exceeded")
            sock.settimeout(remaining_time)
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("checkpoint socket closed before message completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock):
    timeout = sock.gettimeout()
    deadline = time.monotonic() + timeout if timeout is not None else None
    header_size = struct.unpack("!I", recv_exact(sock, 4, deadline))[0]
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise ValueError("invalid checkpoint header length")
    metadata = decode_header(recv_exact(sock, header_size, deadline))
    return metadata, recv_exact(sock, metadata["payload_bytes"], deadline)


def send_message(sock, metadata, payload):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("checkpoint payload must be bytes-like")
    # Reuses existing immutable bytes; copies mutable bytes-like inputs.
    started = time.perf_counter_ns()
    payload = bytes(payload)
    metadata = dict(metadata, payload_bytes=len(payload))
    header = encode_header(metadata)
    encode_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    sock.sendall(header)
    header_send_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    if payload:
        sock.sendall(payload)
    return {"header_json_encode_ms": encode_ms, "header_send_ms": header_send_ms,
            "payload_send_ms": (time.perf_counter_ns() - started) / 1e6}
