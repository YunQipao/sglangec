"""External acceptance: both LOCAL/GLOBAL receipts, native RDMA P/D0/D1 recovery.

Run after copying both nodes' evidence into distinct directories. Nothing here
runs in an inference/manager worker. A PASS is point-in-time experimental
single-block recovery, not whole-node fault tolerance or GPU pool writeback.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .node_block_export import descriptor, sha, unpack, xor
from .online_rs_identity import stripe_id
from .remote_rs_client import RemoteRSClient
from .rs_codec import encode, recover
from .rs_stripe_builder import manifest_sha, validate_manifest


def verify_blocks(blocks, parity, manifest, parity_sha256, *, allow_fixture=False):
    validate_manifest(manifest, allow_fixture=allow_fixture)
    for i, block in enumerate(blocks):
        if descriptor(block, allow_fixture=allow_fixture) != manifest["nodes"][i]:
            raise ValueError(f"D{i} descriptor/checksum mismatch")
    if sha(parity) != parity_sha256:
        raise ValueError("P independent checksum mismatch")
    lengths = [d["length"] for d in manifest["nodes"]]
    checks = {}
    for missing in (0, 1):
        restored = recover(
            [None if i == missing else b for i, b in enumerate(blocks)], parity, lengths
        )
        if restored != blocks[missing]:
            raise ValueError(f"D{missing} recovery not byte exact")
        checks[f"D{missing}"] = dict(sha256=sha(restored), byte_exact=True)
    restored_p = encode(*blocks)
    if restored_p != parity:
        raise ValueError("P reconstruction not byte exact")
    checks["P"] = dict(sha256=sha(restored_p), byte_exact=True)
    return checks


def verify_local(directory, receipt, block, *, allow_fixture=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    meta, ranks = unpack(block, allow_fixture=allow_fixture)
    originals = [
        {k: v for k, v in r.items() if k != "global_protection"} for r in meta["ranks"]
    ]
    if (
        manifest["state"] != "COMMITTED"
        or receipt["local_state"] != "LOCAL_COMMITTED"
        or receipt["global_state"] != "GLOBAL_COMMITTED"
        or manifest["snapshot_id"] != receipt["local_snapshot_id"]
        or originals != manifest["ranks"]
        or any(
            r["global_protection"] != receipt["global_protection"]
            for r in meta["ranks"]
        )
        or receipt["stripe_id"] != stripe_id(receipt["global_protection"])
        or descriptor(block, allow_fixture=allow_fixture) != receipt["node_descriptor"]
    ):
        raise ValueError("local/global checkpoint evidence mismatch")
    parity = (directory / "parity.bin").read_bytes()
    if sha(parity) != manifest["parity_sha256"] or xor(*ranks) != parity:
        raise ValueError("local XOR checksum/bytes mismatch")
    for missing in range(len(ranks)):
        # XOR of P with every surviving rank must equal the missing rank.
        restored = bytearray(parity)
        view = np.frombuffer(restored, dtype=np.uint8)
        for i, rank in enumerate(ranks):
            if i != missing:
                np.bitwise_xor(view, np.frombuffer(rank, dtype=np.uint8), out=view)
        if bytes(restored) != ranks[missing]:
            raise ValueError("local XOR recovery mismatch")


def committed(root, expected, timeout):
    deadline = time.monotonic() + timeout
    while True:
        found = {}
        for path in Path(root).glob("*/global_handoff_error.json"):
            raise ValueError(f"global handoff failed: {path}: {path.read_text()}")
        for path in Path(root).glob("*/global_status.json"):
            receipt = json.loads(path.read_text())
            state = receipt["global_state"]
            if state == "SKIPPED_UNMAPPED":
                continue
            if state in {
                "FAILED",
                "EXPIRED",
                "CANCELLED",
                "DROPPED_CAPACITY",
                "FAILED_IDENTITY_CONFLICT",
            }:
                raise ValueError(f"global protection failed: {path}: {receipt}")
            if state == "GLOBAL_COMMITTED":
                sid = receipt["stripe_id"]
                if sid in found:
                    raise ValueError("duplicate global identity evidence")
                found[sid] = (path.parent, receipt)
        if len(found) == expected:
            identities = [r[1]["global_protection"] for r in found.values()]
            if len(
                {
                    (i["run_id"], i["protection_id"], i["checkpoint_version"])
                    for i in identities
                }
            ) != 1 or sorted(
                r[1]["global_protection"]["chunk_id"] for r in found.values()
            ) != list(
                range(expected)
            ):
                raise ValueError("expected exactly one protection group, chunks 0..N-1")
            return found
        if len(found) > expected or time.monotonic() >= deadline:
            raise TimeoutError(f"expected {expected} global chunks, found {len(found)}")
        time.sleep(0.1)


def run_verify(args):
    groups = [
        committed(root, args.expected_chunks, args.timeout)
        for root in (args.node0_root, args.node1_root)
    ]
    if set(groups[0]) != set(groups[1]):
        raise ValueError("two nodes have different global identities")
    args.output.mkdir(parents=True, exist_ok=False)
    client = RemoteRSClient(args.remote_host, args.remote_port, args.timeout)
    reports = []
    for sid in sorted(groups[0]):
        directory = args.output / sid
        directory.mkdir()
        receipts = [group[sid][1] for group in groups]
        blocks = [(group[sid][0] / "global_node.bin").read_bytes() for group in groups]
        for i, (group, receipt, block) in enumerate(zip(groups, receipts, blocks)):
            if receipt["node_id"] != i:
                raise ValueError("node roots reversed")
            verify_local(group[sid][0], receipt, block)
        key = {k: receipts[0][k] for k in ("handle", "manifest_sha256")}
        if any(any(r.get(k) != v for k, v in key.items()) for r in receipts):
            raise ValueError("global receipts not bound to the same remote incarnation")
        status, _ = client.call("get_stripe_status", stripe_id=sid, **key)
        if (
            status["state"] != "COMMITTED"
            or status["manifest_sha256"] != manifest_sha(status["manifest"])
            or status["manifest"]["stripe_id"] != sid
            or any(r["parity_sha256"] != status["parity_sha256"] for r in receipts)
        ):
            raise ValueError("remote commit evidence mismatch/expired")
        rdma = dict(
            device=args.rdma_device,
            gid_index=args.rdma_gid_index,
            port_num=args.rdma_port_num,
            timeout=args.timeout,
            return_host=args.rdma_return_host,
            return_port=args.rdma_return_port,
        )
        status, parity, p_transfer = client.rdma_fetch_parity(
            sid, **key, **rdma, destination=directory / "P.bin"
        )
        checks = verify_blocks(
            blocks, parity, status["manifest"], status["parity_sha256"]
        )
        remote_recovery = {}
        for missing in (0, 1):
            _, restored, transfer = client.rdma_reconstruct_missing_node(
                sid,
                missing,
                blocks[1 - missing],
                **key,
                **rdma,
                destination=directory / f"recovered-D{missing}.bin",
            )
            if restored != blocks[missing]:
                raise ValueError(f"remote D{missing} recovery not byte exact")
            remote_recovery[f"D{missing}"] = dict(byte_exact=True, transfer=transfer)
        reports.append(
            dict(
                stripe_id=sid,
                checks=checks,
                local_xor_all_ranks_byte_exact=True,
                remote_recovery=remote_recovery,
                parity_transfer=p_transfer,
                node_metrics=[r["metrics"] for r in receipts],
            )
        )
    report = dict(
        state="PASS",
        transport="rdma",
        chunks=reports,
        failure_domain="two physical nodes; single block erasure only",
        whole_node_recovery="NOT GUARANTEED",
        gpu_pool_writeback=False,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    wait = sub.add_parser("wait", help="wait for local global-commit receipts")
    wait.add_argument("--root", type=Path, required=True)
    wait.add_argument("--expected-chunks", type=int, default=3)
    wait.add_argument("--timeout", type=float, default=240)
    verify = sub.add_parser(
        "verify", help="native RDMA parity + both recovery directions"
    )
    verify.add_argument("--node0-root", type=Path, required=True)
    verify.add_argument("--node1-root", type=Path, required=True)
    verify.add_argument("--expected-chunks", type=int, default=3)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--remote-host", required=True)
    verify.add_argument("--remote-port", type=int, default=30300)
    verify.add_argument("--rdma-device", default="mlx5_0")
    verify.add_argument("--rdma-gid-index", type=int, required=True)
    verify.add_argument("--rdma-port-num", type=int, default=1)
    verify.add_argument("--rdma-return-host", required=True)
    verify.add_argument("--rdma-return-port", type=int, default=30490)
    verify.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.expected_chunks < 1 or not 0 < args.timeout < float("inf"):
        parser.error("positive chunk count and finite timeout required")
    if args.op == "wait":
        result = committed(args.root, args.expected_chunks, args.timeout)
        print(json.dumps({"state": "GLOBAL_COMMITTED", "stripes": sorted(result)}))
    else:
        run_verify(args)


if __name__ == "__main__":
    main()
