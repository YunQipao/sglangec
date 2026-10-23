"""Fail-closed, read-only KV request mapping and protection-plan helpers.

This module deliberately does not import SGLang and never writes a KV tensor.
It is suitable for unit tests and for a future scheduler-thread audit hook.
It must not be used as proof that a slot remains owned after the caller
returns: this checkout has request-row generations, but no physical-slot
generation or lease protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable, Mapping

import torch


_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,96}")


class KVPoolAuditRejected(ValueError):
    """A read-only audit could not establish the required invariants."""


@dataclass(frozen=True)
class ChunkSpec:
    """One contiguous logical portion of a single request version."""

    snapshot_id: str
    request_id: str
    request_version: str
    token_start: int
    token_count: int
    token_end: int


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise KVPoolAuditRejected(reason)


def _validate_id(value: str, field: str) -> None:
    _require(isinstance(value, str) and bool(_ID_RE.fullmatch(value)), f"invalid_{field}")


def make_chunk_plan(
    snapshot_id: str,
    request_id: str,
    *,
    prompt_tokens: int,
    chunk_tokens: int = 256,
    request_version: str = "prefill-v1",
) -> list[ChunkSpec]:
    """Create a bounded contiguous plan without touching a live KV pool.

    Each chunk gets a unique Snapshot ID, while ``request_version`` ties all
    chunks to the same logical request and model/prompt version.  The plan is
    intentionally metadata-only; a caller must independently COMMIT every
    chunk before considering the full prompt protected.
    """

    _validate_id(snapshot_id, "snapshot_id")
    _validate_id(request_id, "request_id")
    _validate_id(request_version, "request_version")
    _require(type(prompt_tokens) is int and prompt_tokens > 0, "invalid_prompt_tokens")
    _require(type(chunk_tokens) is int and chunk_tokens > 0, "invalid_chunk_tokens")
    _require(prompt_tokens <= 4096 and chunk_tokens <= 4096, "chunk_plan_too_large")

    result: list[ChunkSpec] = []
    start = 0
    index = 1
    while start < prompt_tokens:
        count = min(chunk_tokens, prompt_tokens - start)
        child_id = f"{snapshot_id}-chunk-{index}"
        _validate_id(child_id, "chunk_snapshot_id")
        result.append(
            ChunkSpec(
                snapshot_id=child_id,
                request_id=request_id,
                request_version=request_version,
                token_start=start,
                token_count=count,
                token_end=start + count,
            )
        )
        start += count
        index += 1
    return result


def validate_chunk_group(
    chunks: Iterable[ChunkSpec | Mapping[str, Any]],
    *,
    expected_prompt_tokens: int | None = None,
    expected_request_id: str | None = None,
    expected_request_version: str | None = None,
) -> dict[str, Any]:
    """Validate that independently committed chunks form one prompt version."""

    raw = list(chunks)
    _require(bool(raw), "empty_chunk_group")

    def get(item: ChunkSpec | Mapping[str, Any], key: str) -> Any:
        return getattr(item, key) if isinstance(item, ChunkSpec) else item.get(key)

    normalized = []
    for item in raw:
        values = {key: get(item, key) for key in (
            "snapshot_id", "request_id", "request_version",
            "token_start", "token_count", "token_end",
        )}
        _validate_id(values["snapshot_id"], "snapshot_id")
        _validate_id(values["request_id"], "request_id")
        _validate_id(values["request_version"], "request_version")
        _require(
            all(type(values[key]) is int for key in ("token_start", "token_count", "token_end")),
            "invalid_chunk_range",
        )
        _require(
            values["token_start"] >= 0
            and values["token_count"] > 0
            and values["token_end"] == values["token_start"] + values["token_count"],
            "invalid_chunk_range",
        )
        normalized.append(values)

    normalized.sort(key=lambda item: item["token_start"])
    first = normalized[0]
    request_id = expected_request_id or first["request_id"]
    request_version = expected_request_version or first["request_version"]
    _require(all(item["request_id"] == request_id for item in normalized), "chunk_request_mismatch")
    _require(
        all(item["request_version"] == request_version for item in normalized),
        "chunk_request_version_mismatch",
    )
    _require(
        len({item["snapshot_id"] for item in normalized}) == len(normalized),
        "duplicate_chunk_snapshot",
    )
    cursor = 0
    for item in normalized:
        _require(item["token_start"] == cursor, "chunk_range_gap_or_overlap")
        cursor = item["token_end"]
    if expected_prompt_tokens is not None:
        _require(type(expected_prompt_tokens) is int and cursor == expected_prompt_tokens,
                 "chunk_prompt_length_mismatch")
    return {
        "request_id": request_id,
        "request_version": request_version,
        "token_start": 0,
        "token_count": cursor,
        "token_end": cursor,
        "chunk_count": len(normalized),
        "snapshot_ids": [item["snapshot_id"] for item in normalized],
    }


def _cpu_int_list(value: Any) -> list[int]:
    if not isinstance(value, torch.Tensor):
        raise KVPoolAuditRejected("mapping_must_be_tensor")
    _require(value.ndim == 1, "mapping_slice_must_be_1d")
    return [int(x) for x in value.detach().to(device="cpu", dtype=torch.int64).tolist()]


def inspect_request_mapping(
    req_to_token_pool: Any,
    req: Any,
    *,
    token_start: int,
    token_count: int,
    expected_generation: int | None = None,
) -> dict[str, Any]:
    """Read and validate one request's logical-to-physical slot mapping.

    The returned record is an observation, not a lease.  In particular,
    ``slot_generation_available`` is always false for the current allocator.
    """

    _require(type(token_start) is int and token_start >= 0, "invalid_token_start")
    _require(type(token_count) is int and token_count > 0, "invalid_token_count")
    row = getattr(req, "req_pool_idx", None)
    _require(type(row) is int and row > 0, "request_row_not_allocated")
    free_slots = set(int(x) for x in getattr(req_to_token_pool, "free_slots", []))
    _require(row not in free_slots, "request_row_is_free")

    generations = getattr(req_to_token_pool, "req_generation", None)
    _require(isinstance(generations, torch.Tensor) and row < generations.numel(),
             "request_generation_unavailable")
    generation = int(generations[row].item())
    _require(generation > 0, "invalid_request_generation")
    if expected_generation is not None:
        _require(type(expected_generation) is int and generation == expected_generation,
                 "request_generation_mismatch")

    mapping = getattr(req_to_token_pool, "req_to_token", None)
    _require(isinstance(mapping, torch.Tensor) and mapping.ndim == 2, "request_mapping_unavailable")
    _require(row < mapping.shape[0] and token_start + token_count <= mapping.shape[1],
             "request_range_out_of_bounds")
    slots = _cpu_int_list(mapping[row, token_start:token_start + token_count])
    _require(len(set(slots)) == token_count, "duplicate_physical_slots")
    _require(all(0 < slot < int(getattr(req_to_token_pool, "size", mapping.shape[0]))
                 for slot in slots), "physical_slot_out_of_bounds")

    packed = ",".join(str(slot) for slot in slots).encode()
    return {
        "req_pool_idx": row,
        "req_generation": generation,
        "token_start": token_start,
        "token_count": token_count,
        "physical_slots": slots,
        "physical_slots_sha256": hashlib.sha256(packed).hexdigest(),
        "slot_generation": None,
        "slot_generation_available": False,
        "lease_required_for_writeback": True,
        "writeback_supported": False,
    }


def reject_writeback(*_args: Any, **_kwargs: Any) -> None:
    """Explicit guard used by future callers until a slot lease exists."""

    raise RuntimeError(
        "KV writeback is disabled: physical slot generation/lease protocol is unavailable"
    )
