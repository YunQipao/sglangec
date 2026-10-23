"""Scheme A (whole-block CPU XOR) and Scheme C (chunk-streaming CPU XOR).

The implementation accepts CPU-owned bytes. CUDA capture/D2H is intentionally
kept at the existing rank_snapshot boundary and is not claimed by this module.
"""

import hashlib
import argparse
import json
import time

import numpy as np

try:
    from .xor_codec import encode
except ImportError:
    from xor_codec import encode


def scheme_a(blocks):
    """Reference centralized XOR after all complete blocks are available."""
    if not blocks or any(len(b) != len(blocks[0]) for b in blocks):
        raise ValueError("scheme_a_equal_nonempty_blocks_required")
    started = time.perf_counter()
    arrays = [np.frombuffer(b, dtype=np.uint8) for b in blocks]
    parity = encode(arrays).tobytes()
    return parity, {"xor_ms": (time.perf_counter() - started) * 1000,
                    "first_parity_chunk_ms": None, "peak_host_bytes": sum(map(len, blocks)) + len(parity)}


def scheme_c(chunk_sources, *, chunk_bytes=256 * 1024, on_parity_chunk=None):
    """XOR aligned source chunks as soon as each rank's current chunk exists.

    Each source is a bytes-like complete CPU snapshot in this CPU prototype;
    iteration emits parity incrementally and never builds a full set of NumPy
    views before starting. Network senders can replace sources with iterators.
    """
    if len(chunk_sources) < 2 or chunk_bytes <= 0:
        raise ValueError("scheme_c_requires_multiple_sources_and_positive_chunk")
    total = len(chunk_sources[0])
    if total <= 0 or any(len(source) != total for source in chunk_sources):
        raise ValueError("scheme_c_equal_nonempty_blocks_required")
    first_at = time.perf_counter()
    first_chunk_ms = None
    parity_parts = []
    for index, offset in enumerate(range(0, total, chunk_bytes)):
        length = min(chunk_bytes, total - offset)
        views = [np.frombuffer(source, dtype=np.uint8, count=length, offset=offset)
                 for source in chunk_sources]
        parity_chunk = encode(views).tobytes()
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - first_at) * 1000
        if on_parity_chunk is not None:
            on_parity_chunk(index, offset, parity_chunk)
        parity_parts.append(parity_chunk)
    parity = b"".join(parity_parts)
    return parity, {"xor_ms": (time.perf_counter() - first_at) * 1000,
                    "first_parity_chunk_ms": first_chunk_ms,
                    "chunk_count": len(parity_parts),
                    "peak_host_bytes": min(total, chunk_bytes) * (len(chunk_sources) + 1)}


def verify_a_c(blocks, *, chunk_bytes=256 * 1024):
    a, stats_a = scheme_a(blocks)
    c, stats_c = scheme_c(blocks, chunk_bytes=chunk_bytes)
    if a != c:
        raise ValueError("scheme_a_c_parity_mismatch")
    return {"status": "byte_exact", "scheme_a_sha256": hashlib.sha256(a).hexdigest(),
            "scheme_c_sha256": hashlib.sha256(c).hexdigest(), "parity_bytes": len(a),
            "scheme_a": stats_a, "scheme_c": stats_c}


class ChunkXorAggregator:
    """Online XOR collector: emits a parity stripe as soon as all rank chunks arrive."""

    def __init__(self, *, rank_ids, total_length, chunk_bytes, on_parity_chunk,
                 max_inflight_chunks=2):
        if (len(set(rank_ids)) < 2 or total_length <= 0 or chunk_bytes <= 0
                or max_inflight_chunks < 1):
            raise ValueError("invalid_chunk_aggregator_configuration")
        self.rank_ids = frozenset(rank_ids)
        self.total_length = total_length
        self.chunk_bytes = chunk_bytes
        self.on_parity_chunk = on_parity_chunk
        self.max_inflight_chunks = max_inflight_chunks
        self.pending = {}
        self.next_index = 0
        self.emitted = 0

    def push(self, *, rank_id, chunk_index, chunk_offset, total_length, payload):
        payload = bytes(payload)
        if rank_id not in self.rank_ids or total_length != self.total_length:
            raise ValueError("chunk_rank_or_total_mismatch")
        expected_offset = chunk_index * self.chunk_bytes
        expected_length = min(self.chunk_bytes, self.total_length - expected_offset)
        if (chunk_index < self.next_index or expected_offset < 0
                or chunk_offset != expected_offset or expected_length <= 0
                or len(payload) != expected_length):
            raise ValueError("chunk_offset_length_or_replay_mismatch")
        if chunk_index not in self.pending:
            if len(self.pending) >= self.max_inflight_chunks:
                raise BufferError("chunk_aggregator_backpressure")
            self.pending[chunk_index] = {}
        rank_chunks = self.pending[chunk_index]
        if rank_id in rank_chunks:
            if rank_chunks[rank_id] != payload:
                raise ValueError("conflicting_rank_chunk")
            return False
        rank_chunks[rank_id] = payload
        while self.next_index in self.pending and set(self.pending[self.next_index]) == self.rank_ids:
            index = self.next_index
            offset = index * self.chunk_bytes
            parts = [np.frombuffer(self.pending[index][rank], dtype=np.uint8)
                     for rank in sorted(self.rank_ids)]
            parity = encode(parts).tobytes()
            self.on_parity_chunk(index, offset, parity)
            del self.pending[index]
            self.next_index += 1
            self.emitted += 1
        return True

    def finish(self):
        expected_chunks = (self.total_length + self.chunk_bytes - 1) // self.chunk_bytes
        if self.next_index != expected_chunks or self.pending:
            raise ValueError("checkpoint_incomplete_chunks")
        return {"chunks_committed": self.emitted, "total_length": self.total_length}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes-per-rank", type=int, default=9 * 1024 * 1024)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bytes_per_rank <= 0 or args.ranks < 2:
        parser.error("need positive payload and at least two ranks")
    offsets = np.arange(args.bytes_per_rank, dtype=np.uint64)
    blocks = [((offsets * (rank * 2 + 3) + rank * 71) & 0xff).astype(np.uint8).tobytes()
              for rank in range(args.ranks)]
    report = verify_a_c(blocks, chunk_bytes=args.chunk_bytes)
    report.update(ranks=args.ranks, bytes_per_rank=args.bytes_per_rank,
                  chunk_bytes=args.chunk_bytes, payload_persisted=False,
                  scope="deterministic CPU synthetic data; not GPU KV")
    with open(args.output, "x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
