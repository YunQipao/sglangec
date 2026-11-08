"""Bounded parent-run TCP test against the configured physical P host.

The command reads two previously captured authentic NodeBlocks, registers only
their descriptors, uploads them over TCP, checks parity-only retention, then
performs five byte-checked parity reads and both single-erasure recovery paths.
It never starts a service, accesses GPUs, injects a fault, or reads either input
again remotely during reconstruction. The source files are only read locally
for upload and post-recovery byte comparison.
"""

import argparse
import json
import re
import time
from pathlib import Path

from ..node_block_export import MAX_NODE_BYTES, canonical, descriptor, sha
from ..remote_rs_client import binding
from ..rs_codec import encode
from ..rs_stripe_builder import build
from .admission import placement, local_identity
from .remote import client as tcp_client

ARTIFACTS = Path(__file__).parent / "artifacts"


def _read_node(path, node_id):
    path = Path(path)
    size = path.stat().st_size
    if not 0 < size <= MAX_NODE_BYTES:
        raise ValueError(f"D{node_id} NodeBlock size outside bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"D{node_id} NodeBlock changed while reading")
    meta = descriptor(raw)
    if meta["metadata"].get("node_id") != node_id:
        raise ValueError(f"D{node_id} NodeBlock role mismatch")
    return raw, meta


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run(config_path, node_paths, output, *, stripe_id, repetitions=5, connect=tcp_client):
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions must be in 1..5")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", stripe_id or ""):
        raise ValueError("fresh stripe ID must match [A-Za-z0-9_.-]{1,128}")
    config = json.loads(Path(config_path).read_text())
    config_hash = placement(config)
    if config["transport"] != "tcp":
        raise ValueError("three-host orchestration is explicitly TCP-only; RDMA fails closed")
    if len(node_paths) != 2:
        raise ValueError("ordered D0 and D1 NodeBlock paths required")
    blocks, descriptors = [], []
    for i, path in enumerate(node_paths):
        raw, meta = _read_node(path, i)
        blocks.append(raw)
        descriptors.append(meta)
    manifest = build(stripe_id, blocks)
    reserve = 3 * max(map(len, blocks))
    if reserve > config["max_bytes"]:
        raise ValueError("conservative three-copy server reservation exceeds max_bytes")

    output = Path(output).resolve()
    artifacts = ARTIFACTS.resolve()
    if output.parent != artifacts or output.name in ("", ".", ".."):
        raise ValueError("output must be a new direct child of design_study/artifacts")
    output.mkdir(parents=False, exist_ok=False)
    report = dict(
        schema="kv-ec-three-host-tcp-v1", state="RUNNING", transport="TCP",
        config_sha256=config_hash, stripe_id=stripe_id, repetitions=repetitions,
        physical_node_attestation="NOT INDEPENDENTLY VERIFIED",
        rdma="NOT USED; configured physical243 has no HostRDMA device",
        inputs=[dict(path=str(Path(p).resolve()), descriptor=d) for p, d in zip(node_paths, descriptors)],
        reservation_bytes=reserve, operations=[],
        recovery_contract="Only the surviving NodeBlock is sent as recovery input; missing original is compared locally after response.",
    )
    _write_json(output / "manifest.json", dict(config_sha256=config_hash, manifest=manifest,
                                                    source_paths=[str(Path(p).resolve()) for p in node_paths]))
    try:
        peer = connect(config)
        registered = []
        for node in (0, 1):
            status, _ = peer.call("register_node", stripe_id=stripe_id, node_id=node,
                                  descriptor=descriptors[node], transport="tcp")
            registered.append(status)
        status = registered[-1]
        if status.get("state") != "PENDING" or status.get("manifest_sha256") != sha(canonical(manifest)):
            raise ValueError("P did not bind both authentic node descriptors into PENDING stripe")
        key = binding(status)
        for node in (0, 1):
            started = time.perf_counter_ns()
            status, _ = peer.call("submit_node_block", node_id=node, payload=blocks[node],
                                  transport="tcp", **key)
            report["operations"].append(dict(phase="upload", node=node,
                                             elapsed_ns=time.perf_counter_ns() - started,
                                             state=status.get("state")))
        status, _ = peer.call("get_stripe_status", transport="tcp", **key)
        expected_parity = encode(blocks[0], blocks[1])
        if (status.get("state") != "COMMITTED" or status.get("retained_node_block_bytes") != 0
                or status.get("parity_bytes") != len(expected_parity)):
            raise ValueError("committed parity-only store retained source data or has bad parity size")
        report["committed_store"] = dict(state=status["state"], retained_node_block_bytes=0,
                                         parity_bytes=status["parity_bytes"],
                                         submitted_nodes=status.get("submitted_nodes"),
                                         server_epoch=status.get("server_epoch"))
        for repeat in range(repetitions):
            started = time.perf_counter_ns()
            parity_status, parity = peer.call("fetch_parity", transport="tcp", **key)
            elapsed = time.perf_counter_ns() - started
            if (parity_status.get("state") != "COMMITTED" or parity != expected_parity
                    or sha(parity) != status.get("parity_sha256")):
                raise ValueError("remote parity read bytecheck/binding failed")
            report["operations"].append(dict(phase="fetch_parity", repeat=repeat,
                                             elapsed_ns=elapsed, payload_bytes=len(parity),
                                             sha256=sha(parity), transfer=peer.transfers[-1]))
            for lost in (0, 1):
                survivor_node = 1 - lost
                started = time.perf_counter_ns()
                recovered_status, recovered = peer.call(
                    "reconstruct_missing_node", missing_nodes=[lost], payload=blocks[survivor_node],
                    transport="tcp", **key,
                )
                elapsed = time.perf_counter_ns() - started
                if (recovered_status.get("recovered_node") != lost or recovered != blocks[lost]
                        or descriptor(recovered) != descriptors[lost]):
                    raise ValueError(f"D{lost} reconstructed block bytecheck failed")
                report["operations"].append(dict(
                    phase="recover", repeat=repeat, missing_node=lost,
                    recovery_input_node=survivor_node, recovery_input_sha256=sha(blocks[survivor_node]),
                    recovery_input_bytes=len(blocks[survivor_node]), result_bytes=len(recovered),
                    elapsed_ns=elapsed, recovered_sha256=sha(recovered),
                    transfer=peer.transfers[-1],
                ))
        report.update(state="PASS", transfer_count=len(peer.transfers), transfers=peer.transfers,
                      interpretation="TCP application frames on remote CPU DRAM; not disk, RDMA, GPU, serving, node-failure, or RTO evidence")
        return report
    except Exception as exc:
        report.update(state="FAIL", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        _write_json(output / "raw.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, nargs=2, required=True, metavar=("D0_NODEBLOCK", "D1_NODEBLOCK"))
    parser.add_argument("--stripe-id", required=True, help="fresh ID, never reuse an existing or expired ID")
    parser.add_argument("--repetitions", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--output", type=Path, required=True, help="new direct child of design_study/artifacts")
    args = parser.parse_args()
    result = run(args.config, args.nodes, args.output, stripe_id=args.stripe_id,
                 repetitions=args.repetitions)
    print(json.dumps(dict(state=result["state"], stripe_id=result["stripe_id"],
                          transport=result["transport"], artifact=str(args.output))))


if __name__ == "__main__":
    main()
