"""Independent physical-node TCP/RDMA RS(2,1) workflow.

The ``node`` role runs on Node A or Node B and uploads only that machine's
complete serialized node block.  The ``coordinator`` role runs against the
Remote RS server (normally on Node B) and waits for both registrations and
submissions.  This is an offline post-capture workflow: it never reads a GPU
KV slot and never claims online serving or GPU-pool writeback.
"""

import argparse
import csv
import json
from pathlib import Path
import time

from .node_block_export import descriptor, sha, unpack
from .remote_rs_client import RemoteRSClient, binding
from .rs_experiment import environment, network_observation, process_usage, save_report, usage_delta


def _write_csv(path, report):
    fields = [
        "state", "transport", "node_id", "stripe_id", "tp_size", "node_block_bytes",
        "register_ms", "submit_ms", "commit_ms", "network_payload_bytes",
        "network_stream_bytes", "network_control_bytes", "device_setup_ms", "connection_setup_ms", "memory_registration_ms",
        "qp_setup_ms", "exchange_ms", "completion_ms", "effective_throughput_gbps",
        "recovery_ms", "byte_exact",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: report.get(key) for key in fields})


def _save(output, report):
    output.mkdir(parents=True, exist_ok=True)
    save_report(output / "result", report)
    _write_csv(output / "metrics.csv", report)


def _status(client, stripe_id):
    response, _ = client.call("get_stripe_status", stripe_id=stripe_id)
    return response


def run_node(args):
    chunk_tokens = getattr(args, "chunk_tokens", 256)
    network_profile = getattr(args, "network_profile", "physical")
    transport = getattr(args, "transport", "tcp")
    output = args.output.resolve()
    started = time.perf_counter()
    before = process_usage()
    block = args.node_block.read_bytes()
    node_meta = descriptor(block, allow_fixture=args.allow_fixture)
    unpack(block, allow_fixture=args.allow_fixture)
    tp_size = node_meta["metadata"]["ranks"][0]["tp_size"]
    if node_meta["metadata"]["ranks"][0]["token_count"] != chunk_tokens:
        raise ValueError("--chunk-tokens differs from the captured node block")
    client = RemoteRSClient(args.remote_host, args.remote_port, args.timeout)
    report = {
        "state": "RUNNING", "mode": "independent_physical_node_upload",
        "online_remote_serving_interference": "NOT VERIFIED",
        "GPU_pool_writeback": "NOT IMPLEMENTED", "physical_network": "NOT VERIFIED",
        "network_profile": network_profile, "chunk_tokens": chunk_tokens,
        "transport": transport,
        "node_id": args.node_id, "stripe_id": args.stripe_id, "tp_size": tp_size,
        "node_block_bytes": len(block), "node_block_sha256": sha(block),
        "request_id": node_meta["metadata"]["ranks"][0]["request_id"],
        "snapshot_id": node_meta["metadata"]["ranks"][0]["snapshot_id"],
        "model_identifier": node_meta["metadata"]["ranks"][0]["model_identifier"],
        "model_fingerprint": node_meta["metadata"]["ranks"][0]["model_sha256"],
        "dtype": node_meta["metadata"]["ranks"][0]["dtype"],
        "chunk": {
            "token_start": node_meta["metadata"]["ranks"][0]["token_start"],
            "token_count": node_meta["metadata"]["ranks"][0]["token_count"],
            "layers": node_meta["metadata"]["ranks"][0]["layers"],
        },
        "environment": environment(), "process_before": before,
    }
    export_report = args.node_block.parent / "export.json"
    if export_report.exists():
        record = json.loads(export_report.read_text())
        if record.get("descriptor") != node_meta:
            raise ValueError("export.json descriptor does not match node.bin")
        report.update(
            export_measurements=record,
            local_xor_ms=record.get("xor_ms"),
            node_pack_ms=record.get("pack_ms"),
            local_parity_sha256=record.get("local_parity_sha256"),
        )
    try:
        client.call("health")
        report["network_before"] = network_observation(args.remote_host)
        register_started = time.perf_counter()
        registration, _ = client.register_node(
            args.stripe_id, args.node_id, block, allow_fixture=args.allow_fixture
        )
        report["register_ms"] = (time.perf_counter() - register_started) * 1000
        deadline = time.monotonic() + args.timeout
        status = registration
        while status.get("state") == "REGISTERING":
            if time.monotonic() >= deadline:
                raise TimeoutError("second node did not register before timeout")
            time.sleep(args.poll_interval)
            status = _status(client, args.stripe_id)
        if status.get("state") not in ("PENDING", "COMMITTED"):
            raise ValueError(f"stripe registration failed: {status}")
        key = binding(status)
        submit_started = time.perf_counter()
        if transport == "rdma":
            status, transfer_stats, _ = client.rdma_submit_node_block(
                args.stripe_id, args.node_id, block, **key,
                device=args.rdma_device, gid_index=args.rdma_gid_index,
                port_num=args.rdma_port_num, timeout=args.timeout,
            )
            report["rdma_submit"] = transfer_stats
        else:
            status, _ = client.call("submit_node_block", node_id=args.node_id,
                                     payload=block, **key)
        report["submit_ms"] = (time.perf_counter() - submit_started) * 1000
        report["state"] = status["state"]
        report["server_status"] = status
        report["binding"] = key
        transfers = [t for t in client.transfers if t["op"] == "submit_node_block"]
        report["network_payload_bytes"] = sum(t.get("tx", {}).get("payload_bytes", 0) for t in transfers)
        report["network_stream_bytes"] = sum(t.get("tx", {}).get("stream_bytes", 0) for t in transfers)
        if transport == "rdma":
            report["network_payload_bytes"] = transfer_stats.get("bytes", len(block))
            report["network_stream_bytes"] = transfer_stats.get("bytes", len(block))
            report["network_control_bytes"] = sum(
                t.get("tx", {}).get("stream_bytes", 0)
                + t.get("rx", {}).get("stream_bytes", 0)
                for t in client.transfers if t.get("transport") == "tcp"
            )
            report["device_setup_ms"] = transfer_stats.get("device_setup_ms")
            report["connection_setup_ms"] = transfer_stats.get("exchange_ms")
            report["memory_registration_ms"] = transfer_stats.get("memory_registration_ms")
            report["qp_setup_ms"] = transfer_stats.get("qp_setup_ms")
            report["exchange_ms"] = transfer_stats.get("exchange_ms")
            report["completion_ms"] = transfer_stats.get("completion_ms")
        report["commit_ms"] = status.get("commit_ms")
        report["network"] = {
            "application_payload_bytes": report["network_payload_bytes"],
            "application_stream_bytes": report["network_stream_bytes"],
            "framing_overhead_bytes": report["network_stream_bytes"] - report["network_payload_bytes"],
            "note": ("RDMA payload bytes; TCP QP metadata/control and Ethernet/IP headers are excluded"
                     if transport == "rdma" else
                     "TCP application bytes; Ethernet/IP/TCP headers and retransmits are excluded"),
        }
        transfer_seconds = (transfer_stats.get("elapsed_s") if transport == "rdma" else
                            (report["submit_ms"] / 1000))
        report["effective_throughput_gbps"] = (
            report["network_payload_bytes"] * 8 / transfer_seconds / 1e9
            if transfer_seconds and transfer_seconds > 0 else None
        )
        report["network_after"] = network_observation(args.remote_host)
        if (network_profile != "loopback-test"
                and report["network_before"].get("physical_device")
                and report["network_after"].get("physical_device")):
            report["physical_network"] = "PHYSICAL_INTERFACE_ROUTES_OBSERVED"
        report["state"] = "PASS" if status["state"] in ("PENDING", "COMMITTED") else "FAIL"
    except Exception as exc:
        report.update(state="FAIL", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        report["transfers"] = client.transfers
        report["process_after"] = process_usage()
        report["process_delta"] = usage_delta(before, report["process_after"])
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000
        _save(output, report)
    return report


def run_coordinator(args):
    chunk_tokens = getattr(args, "chunk_tokens", 256)
    network_profile = getattr(args, "network_profile", "physical")
    transport = getattr(args, "transport", "tcp")
    output = args.output.resolve()
    started = time.perf_counter()
    client = RemoteRSClient(args.remote_host, args.remote_port, args.timeout)
    report = {
        "state": "RUNNING", "mode": "independent_stripe_coordinator",
        "online_remote_serving_interference": "NOT VERIFIED",
        "GPU_pool_writeback": "NOT IMPLEMENTED", "physical_network": "NOT VERIFIED",
        "network_profile": network_profile, "transport": transport,
        "stripe_id": args.stripe_id, "environment_B": environment(),
        "process_B_before": process_usage(),
    }
    try:
        client.call("health")
        report["network_before"] = network_observation(args.remote_host)
        deadline = time.monotonic() + args.timeout
        status = _status(client, args.stripe_id)
        while status.get("state") != "COMMITTED":
            if status.get("state") == "EXPIRED":
                raise ValueError("stripe expired before COMMITTED")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"stripe did not commit: {status}")
            time.sleep(args.poll_interval)
            status = _status(client, args.stripe_id)
        if (status["manifest"]["nodes"][0]["metadata"]["ranks"][0]["token_count"]
                != chunk_tokens):
            raise ValueError("--chunk-tokens differs from the committed stripe")
        key = binding(status)
        parity_started = time.perf_counter()
        if transport == "rdma":
            parity_status, parity, parity_transfer = client.rdma_fetch_parity(
                args.stripe_id, **key, destination=output / "parity.bin",
                return_host=args.rdma_return_host,
                return_port=args.rdma_return_port,
                device=args.rdma_device, gid_index=args.rdma_gid_index,
                port_num=args.rdma_port_num, timeout=args.timeout,
            )
            report["rdma_parity"] = parity_transfer
        else:
            parity_status, parity = client.call("fetch_parity", **key)
        report["fetch_parity_ms"] = (time.perf_counter() - parity_started) * 1000
        if sha(parity) != parity_status["parity_sha256"]:
            raise ValueError("remote parity checksum mismatch")
        output.mkdir(parents=True, exist_ok=True)
        parity_path = output / "parity.bin"
        if not parity_path.exists():
            parity_path.write_bytes(parity)
        report.update(state="PASS", server_status=status, binding=key,
                      parity_sha256=sha(parity), parity_bytes=len(parity),
                      rs_encode_ms=status.get("rs_encode_ms"),
                      remote_committed_ms=status.get("commit_ms"))
        if transport == "rdma":
            report["network_payload_bytes"] = parity_transfer.get("payload_bytes", len(parity))
            report["network_control_bytes"] = sum(
                t.get("tx", {}).get("stream_bytes", 0)
                + t.get("rx", {}).get("stream_bytes", 0)
                for t in client.transfers if t.get("transport") == "tcp"
            )
            report["effective_throughput_gbps"] = (
                report["network_payload_bytes"] * 8 /
                (parity_transfer.get("stats", {}).get("elapsed_s") or 1e-9) / 1e9
            )
        else:
            report["network_payload_bytes"] = client.transfers[-1]["rx"]["payload_bytes"]
        report["remote_metrics"], _ = client.call("get_metrics")
        report["network_after"] = network_observation(args.remote_host)
        if (network_profile != "loopback-test"
                and report["network_before"].get("physical_device")
                and report["network_after"].get("physical_device")):
            report["physical_network"] = "PHYSICAL_INTERFACE_ROUTES_OBSERVED"
        if args.missing_node is not None:
            if not args.survivor:
                raise ValueError("--survivor is required for recovery")
            survivor = args.survivor.read_bytes()
            recovery_started = time.perf_counter()
            if transport == "rdma":
                recovered_path = output / "recovered-node.bin"
                result, recovered, recovery_transfer = client.rdma_reconstruct_missing_node(
                    args.stripe_id, args.missing_node, survivor, **key,
                    destination=recovered_path,
                    return_host=args.rdma_return_host,
                    return_port=args.rdma_return_port,
                    device=args.rdma_device, gid_index=args.rdma_gid_index,
                    port_num=args.rdma_port_num, timeout=args.timeout,
                )
                report["rdma_recovery"] = recovery_transfer
            else:
                result, recovered = client.call(
                    "reconstruct_missing_node", payload=survivor,
                    missing_nodes=[args.missing_node], **key
                )
            recovery_ms = (time.perf_counter() - recovery_started) * 1000
            recovered_path = output / "recovered-node.bin"
            # output is reserved only after all remote operations complete.
            recovered_path.parent.mkdir(parents=True, exist_ok=True)
            recovered_path.write_bytes(recovered)
            unpack(recovered, allow_fixture=args.allow_fixture)
            exact = (descriptor(recovered, allow_fixture=args.allow_fixture)
                     == status["manifest"]["nodes"][args.missing_node])
            if args.expected:
                exact = recovered == args.expected.read_bytes()
                if not exact:
                    raise ValueError("recovered node is not byte exact to expected block")
            report.update(recovered_node=args.missing_node,
                          recovery_ms=recovery_ms,
                          server_recovery_ms=(recovery_transfer.get("rs_reconstruct_ms")
                                              if transport == "rdma" else result.get("recovery_ms")),
                          recovered_sha256=sha(recovered), byte_exact=exact)
    except Exception as exc:
        report.update(state="FAIL", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        report["transfers"] = client.transfers
        report["process_B_after"] = process_usage()
        report["process_B_delta"] = usage_delta(report["process_B_before"], report["process_B_after"])
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000
        _save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    node = sub.add_parser("node", help="register and upload one physical node block")
    node.add_argument("--remote-host", required=True)
    node.add_argument("--remote-port", type=int, default=30300)
    node.add_argument("--stripe-id", required=True)
    node.add_argument("--node-id", type=int, choices=(0, 1), required=True)
    node.add_argument("--node-block", type=Path, required=True)
    node.add_argument("--output", type=Path, required=True)
    node.add_argument("--timeout", type=float, default=300)
    node.add_argument("--poll-interval", type=float, default=0.25)
    node.add_argument("--chunk-tokens", type=int, choices=(128, 256, 512), default=256)
    node.add_argument("--network-profile", choices=("physical", "tc", "loopback-test"), default="physical")
    node.add_argument("--transport", choices=("tcp", "rdma"), default="tcp")
    node.add_argument("--rdma-device", default="mlx5_0")
    node.add_argument("--rdma-gid-index", type=int)
    node.add_argument("--rdma-port-num", type=int, default=1)
    node.add_argument("--rdma-return-host", default="127.0.0.1")
    node.add_argument("--rdma-return-port", type=int, default=30490)
    node.add_argument("--allow-fixture", action="store_true")
    coord = sub.add_parser("coordinator", help="wait for both nodes and fetch committed parity")
    coord.add_argument("--remote-host", required=True)
    coord.add_argument("--remote-port", type=int, default=30300)
    coord.add_argument("--stripe-id", required=True)
    coord.add_argument("--output", type=Path, required=True)
    coord.add_argument("--timeout", type=float, default=300)
    coord.add_argument("--poll-interval", type=float, default=0.25)
    coord.add_argument("--chunk-tokens", type=int, choices=(128, 256, 512), default=256)
    coord.add_argument("--network-profile", choices=("physical", "tc", "loopback-test"), default="physical")
    coord.add_argument("--transport", choices=("tcp", "rdma"), default="tcp")
    coord.add_argument("--rdma-device", default="mlx5_0")
    coord.add_argument("--rdma-gid-index", type=int)
    coord.add_argument("--rdma-port-num", type=int, default=1)
    coord.add_argument("--rdma-return-host", default="127.0.0.1")
    coord.add_argument("--rdma-return-port", type=int, default=30490)
    coord.add_argument("--missing-node", type=int, choices=(0, 1))
    coord.add_argument("--survivor", type=Path)
    coord.add_argument("--expected", type=Path)
    coord.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    result = run_node(args) if args.role == "node" else run_coordinator(args)
    print(json.dumps({"state": result["state"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
