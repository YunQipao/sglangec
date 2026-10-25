"""Bounded TCP framing: uint32 BE JSON bytes, JSON, raw payload (no pickle).

One request/response per connection. SHA256 covers the binary in both directions.
Byte counters are exact application TCP stream bytes, NOT Ethernet/IP overhead.
"""

import json
import struct
import time

from .node_block_export import canonical, sha, MAX_NODE_BYTES

MAX_HEADER = 2 * 1024**2


def send(sock, metadata, payload=b"", observer=None):
    payload = bytes(payload)
    if len(payload) > MAX_NODE_BYTES:
        raise ValueError("payload limit")
    head = canonical(
        dict(
            metadata,
            protocol="kv-remote-rs-v1",
            payload_bytes=len(payload),
            payload_sha256=sha(payload),
        )
    )
    if len(head) > MAX_HEADER:
        raise ValueError("header limit")
    timeout = sock.gettimeout()
    deadline = time.monotonic() + timeout if timeout is not None else None

    def write(raw, is_payload):
        view = memoryview(raw)
        while view:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("send deadline exceeded")
                sock.settimeout(remaining)
            sent = sock.send(view)
            if sent <= 0:
                raise ConnectionError("socket closed while sending")
            if observer:
                observer(sent, is_payload)
            view = view[sent:]

    start = time.perf_counter()
    write(struct.pack("!I", len(head)) + head, False)
    header_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    write(payload, True)
    return {
        "stream_bytes": 4 + len(head) + len(payload),
        "payload_bytes": len(payload),
        "header_send_ms": header_ms,
        "payload_send_ms": (time.perf_counter() - start) * 1000,
    }


def receive(sock, timeout, observer=None):
    start = time.perf_counter()
    deadline = time.monotonic() + timeout

    def exact(n, is_payload=False):
        parts = []
        while n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("frame deadline exceeded")
            sock.settimeout(left)
            part = sock.recv(min(n, 1024**2))
            if not part:
                raise ConnectionError("truncated frame")
            if observer:
                observer(len(part), is_payload)
            n -= len(part)
            parts.append(part)
        return b"".join(parts)

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("header limit")
    meta = json.loads(exact(size))
    n = meta.get("payload_bytes")
    if (
        meta.get("protocol") != "kv-remote-rs-v1"
        or type(n) is not int
        or not 0 <= n <= MAX_NODE_BYTES
    ):
        raise ValueError("invalid protocol/payload length")
    raw = exact(n, True)
    if sha(raw) != meta.get("payload_sha256"):
        raise ValueError("TCP payload SHA256 mismatch")
    return (
        meta,
        raw,
        {
            "stream_bytes": 4 + size + n,
            "payload_bytes": n,
            "receive_ms": (time.perf_counter() - start) * 1000,
        },
    )
