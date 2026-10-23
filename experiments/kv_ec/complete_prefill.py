"""Validate three committed 256-token checkpoints as one 768-token prefill."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .chunk_protection import CommittedChunk, CompletePrefillProtection
from .parity_manager import manager_hello, submit_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--request-version", required=True)
    parser.add_argument("--checkpoint-version", required=True)
    parser.add_argument("--manager-epoch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if manager_hello(args.socket) != args.manager_epoch:
        raise SystemExit("live Manager epoch differs from requested protection epoch")

    request_id = hashlib.sha256(args.request_id.encode()).hexdigest()
    group = CompletePrefillProtection(
        request_id=request_id,
        request_version=args.request_version,
        prompt_tokens=768,
        chunk_tokens=256,
        manager_epoch=args.manager_epoch,
        checkpoint_version=args.checkpoint_version,
    )
    statuses = []
    for index, start in enumerate((0, 256, 512), 1):
        snapshot_id = f"{args.base_snapshot}-chunk-{index}"
        status = submit_once(
            args.socket, {"kind": "status", "snapshot_id": snapshot_id}, b"", timeout=2
        )
        statuses.append(status)
        if status.get("state") != "COMMITTED":
            raise SystemExit(f"chunk {index} is not COMMITTED: {status}")
        metadata = status
        if (
            metadata.get("tp_size") != 4
            or metadata.get("token_start") != start
            or metadata.get("token_count") != 256
            or len(metadata.get("oracle_checks", [])) != 4
            or not all(item.get("byte_exact") for item in metadata["oracle_checks"])
        ):
            raise SystemExit(f"chunk {index} failed structural/oracle validation: {metadata}")
        if any(metadata.get(key) != expected for key, expected in (
            ("request_id", request_id),
            ("snapshot_id", snapshot_id),
            ("prompt_token_count", 768),
            ("parity_bytes", 9437184),
            ("request_version", args.request_version),
            ("checkpoint_version", args.checkpoint_version),
            ("manager_epoch", args.manager_epoch),
        )):
            raise SystemExit(f"chunk {index} version mismatch: {metadata}")
        group.add(CommittedChunk(
            snapshot_id=snapshot_id,
            request_id=request_id,
            request_version=args.request_version,
            checkpoint_version=args.checkpoint_version,
            manager_epoch=args.manager_epoch,
            token_start=start,
            token_count=256,
            tp_size=4,
            committed_ranks=frozenset(range(4)),
        ))
    report = {"state": group.status()["state"], "chunks": statuses, "protection": group.status()}
    if report["state"] != "COMPLETE":
        raise SystemExit(f"complete prefill remains incomplete: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if manager_hello(args.socket) != args.manager_epoch:
        raise SystemExit("Manager restarted during complete-prefill validation")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"state": "COMPLETE", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
