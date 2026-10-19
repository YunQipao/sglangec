"""Committed multi-chunk protection protocol for one complete prefill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .recovery_lease import LeaseRejected


@dataclass(frozen=True)
class CommittedChunk:
    snapshot_id: str
    request_id: str
    request_version: str
    checkpoint_version: str
    manager_epoch: str
    token_start: int
    token_count: int
    tp_size: int
    committed_ranks: frozenset[int]
    state: str = "COMMITTED"

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count


class CompletePrefillProtection:
    """Accept exactly contiguous, independently committed chunks."""

    def __init__(
        self,
        *,
        request_id: str,
        request_version: str,
        prompt_tokens: int,
        chunk_tokens: int,
        manager_epoch: str,
        checkpoint_version: str,
    ):
        if prompt_tokens <= 0 or chunk_tokens <= 0 or prompt_tokens % chunk_tokens:
            raise ValueError("prompt must be divisible into complete chunks")
        self.request_id = request_id
        self.request_version = request_version
        self.prompt_tokens = prompt_tokens
        self.chunk_tokens = chunk_tokens
        self.manager_epoch = manager_epoch
        self.checkpoint_version = checkpoint_version
        self._chunks: dict[str, CommittedChunk] = {}

    def add(self, chunk: CommittedChunk) -> str:
        if chunk.snapshot_id in self._chunks:
            raise LeaseRejected("duplicate_chunk_snapshot")
        if chunk.state != "COMMITTED":
            raise LeaseRejected("chunk_not_committed")
        if (
            chunk.request_id != self.request_id
            or chunk.request_version != self.request_version
            or chunk.manager_epoch != self.manager_epoch
            or chunk.checkpoint_version != self.checkpoint_version
            or chunk.tp_size != 4
            or chunk.committed_ranks != frozenset(range(4))
            or chunk.token_count != self.chunk_tokens
            or chunk.token_start % self.chunk_tokens != 0
        ):
            raise LeaseRejected("chunk_metadata_mismatch")
        if chunk.token_start < 0 or chunk.token_end > self.prompt_tokens:
            raise LeaseRejected("chunk_range_invalid")
        expected_start = len(self._chunks) * self.chunk_tokens
        if chunk.token_start != expected_start:
            raise LeaseRejected("chunk_gap_or_overlap")
        self._chunks[chunk.snapshot_id] = chunk
        if len(self._chunks) == self.prompt_tokens // self.chunk_tokens:
            return "COMPLETE"
        return "PENDING"

    def status(self) -> dict:
        ordered = sorted(self._chunks.values(), key=lambda chunk: chunk.token_start)
        complete = len(ordered) == self.prompt_tokens // self.chunk_tokens
        return {
            "state": "COMPLETE" if complete else "PENDING",
            "request_id": self.request_id,
            "request_version": self.request_version,
            "manager_epoch": self.manager_epoch,
            "checkpoint_version": self.checkpoint_version,
            "prompt_tokens": self.prompt_tokens,
            "chunk_tokens": self.chunk_tokens,
            "chunk_count": len(ordered),
            "snapshot_ids": [chunk.snapshot_id for chunk in ordered],
            "ranges": [
                [chunk.token_start, chunk.token_end] for chunk in ordered
            ],
            "all_ranks_committed": all(
                chunk.committed_ranks == frozenset(range(4)) for chunk in ordered
            ),
        }
