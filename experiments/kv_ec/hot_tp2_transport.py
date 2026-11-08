"""Opt-in bounded in-memory chunk protocol over an ALREADY connected socket.

No bind/connect/service lifecycle, port choice, disk, RDMA, or GPUDirect. The
operator retains socket ownership and supplies authenticated/isolated channels.
ACK means chunk staged, not remote checkpoint transaction committed/durable.
"""

import json
import math
import struct
import time

from .hot_tp2 import Chunk, MemoryCheckpoint, Options, Rejected, sha

MAX_HEADER = 4096


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _unique(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise Rejected("duplicate_frame_key")
        out[k] = v
    return out


class ChunkSocket:
    transport = "host-tcp-connected-socket"
    gpudirect = False

    def __init__(self, sock, *, options=Options(), timeout_s=5):
        options.require()
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
            raise Rejected("finite_chunk_transport_timeout_required")
        self.sock, self.options, self.timeout = sock, options, timeout_s
        self.broken = False

    def _budget(self, deadline):
        remaining = deadline - time.monotonic()
        if self.broken or remaining <= 0:
            raise Rejected("chunk_channel_broken_or_deadline_exceeded")
        self.sock.settimeout(remaining)

    def _exact(self, size, deadline):
        out = bytearray(size)
        view = memoryview(out)
        offset = 0
        while offset < size:
            self._budget(deadline)
            count = self.sock.recv_into(view[offset:], size - offset)
            if not count:
                raise Rejected("truncated_chunk_frame")
            offset += count
        return bytes(out)

    def _header(self, value, deadline):
        raw = _json(value)
        if not 0 < len(raw) <= MAX_HEADER:
            raise Rejected("chunk_header_limit")
        self._budget(deadline)
        self.sock.sendall(struct.pack("!I", len(raw)) + raw)

    def _read_header(self, deadline):
        size = struct.unpack("!I", self._exact(4, deadline))[0]
        if not 0 < size <= MAX_HEADER:
            raise Rejected("chunk_header_limit")
        return json.loads(self._exact(size, deadline), object_pairs_hook=_unique)

    def _validate(self, c):
        # Validate an individual chunk without assuming index==0. Full ordered
        # checkpoint publication is deliberately left to a transaction adapter.
        if (type(c) is not Chunk or type(c.index) is not int or c.index < 0
                or c.index >= (self.options.max_bytes + self.options.chunk_mib * 1024**2 - 1)
                // (self.options.chunk_mib * 1024**2)):
            raise Rejected("chunk_index_limit")
        single = MemoryCheckpoint(c.checkpoint, self.options.chunk_mib * 1024**2,
                                  len(c.parity), (Chunk(c.checkpoint, 0, c.generation,
                                                       c.data_sha, c.parity_sha, c.parity, c.codec),))
        single.validate(self.options)
        if sha(c.parity) != c.parity_sha:
            raise Rejected("transport_parity_sha256_mismatch")

    def send(self, chunk):
        self._validate(chunk)
        deadline = time.monotonic() + self.timeout
        try:
            self._header(dict(schema="hot-tp2-chunk-v1", checkpoint=chunk.checkpoint,
                              index=chunk.index, generation=chunk.generation,
                              data_sha=chunk.data_sha, parity_sha=chunk.parity_sha,
                              codec=chunk.codec, size=len(chunk.parity)), deadline)
            self._budget(deadline)
            self.sock.sendall(chunk.parity)
            ack = self._read_header(deadline)
            if ack != dict(receipt_sha256=sha(_json(chunk.receipt))):
                raise Rejected("chunk_transport_ack_mismatch")
            return chunk.receipt
        except BaseException:
            self.broken = True  # partial protocol frames are never retried in place
            raise

    def receive(self, stage):
        deadline = time.monotonic() + self.timeout
        try:
            h = self._read_header(deadline)
            if (type(h) is not dict or h.get("schema") != "hot-tp2-chunk-v1"
                    or type(h.get("size")) is not int
                    or not 0 < h["size"] <= min(self.options.chunk_mib * 1024**2, self.options.max_bytes)):
                raise Rejected("invalid_chunk_frame_geometry")
            payload = self._exact(h["size"], deadline)
            c = Chunk(h["checkpoint"], h["index"], h["generation"], tuple(h["data_sha"]),
                      h["parity_sha"], payload, h["codec"])
            self._validate(c)
            if stage(c) != c.receipt:
                raise Rejected("receiver_did_not_stage_bound_chunk")
            self._header(dict(receipt_sha256=sha(_json(c.receipt))), deadline)
            return c
        except BaseException:
            self.broken = True
            raise
