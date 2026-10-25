"""M2/M3 offline real KV -> physical TCP RS -> logical erasure validation.

No serving interference claims: remote work starts AFTER inference/capture.
"""

import argparse
import ipaddress
import json
from pathlib import Path
import time

from .node_block_export import descriptor, sha, MAX_NODE_BYTES
from .remote_rs_client import RemoteRSClient, binding
from .rs_experiment import (
    environment,
    process_usage,
    save_report,
    usage_delta,
    network_observation,
    network_delta,
)
from .rs_fault_recovery import verify_faults
from .rs_stripe_builder import build


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "state": "RUNNING",
        "mode": "offline_real_KV_remote_RS",
        "online_remote_serving_interference": "NOT VERIFIED",
        "GPU_pool_writeback": "NOT IMPLEMENTED",
        "physical_network": "NOT VERIFIED",
        "network_profile": args.network_profile,
        "chunk_tokens": args.chunk_tokens,
        "environment_A": environment(),
        "process_A_before": process_usage(),
    }
    client = RemoteRSClient(args.remote_host, args.remote_port, args.timeout)
    start = time.perf_counter()
    try:
        client.call("health")
        peer = client.transfers[0]["peer"][0]
        loopback = ipaddress.ip_address(peer).is_loopback
        if loopback and args.network_profile != "loopback-test":
            raise ValueError("loopback cannot represent physical or tc LAN measurement")
        report["network_A_before"] = network_observation(peer)
        for path in (args.node0_block, args.node1_block):
            if not 0 < path.stat().st_size <= MAX_NODE_BYTES:
                raise ValueError("node file exceeds byte budget")
        blocks = [p.read_bytes() for p in (args.node0_block, args.node1_block)]
        manifest = build(args.stripe_id, blocks)
        if any(
            d["metadata"]["ranks"][0]["token_count"] != args.chunk_tokens
            for d in manifest["nodes"]
        ):
            raise ValueError("--chunk-tokens differs from actual capture")
        report["manifest"] = manifest
        report["model"] = manifest["nodes"][0]["metadata"]["ranks"][0][
            "model_identifier"
        ]
        report.update(
            tp=manifest["nodes"][0]["metadata"]["ranks"][0]["tp_size"],
            kv_dtype=manifest["nodes"][0]["metadata"]["ranks"][0]["dtype"],
            node_lengths=list(map(len, blocks)),
        )
        report["export_measurements"] = []
        for path, block in zip((args.node0_block, args.node1_block), blocks):
            export = path.parent / "export.json"
            if export.exists():
                record = json.loads(export.read_text())
                if record["descriptor"] != descriptor(block):
                    raise ValueError("export report does not match node bytes")
                report["export_measurements"].append(record)
            else:
                report["export_measurements"].append(
                    {"xor_ms": None, "pack_ms": None, "status": "NOT VERIFIED"}
                )
        committed, _ = client.call(
            "create_stripe", stripe_id=args.stripe_id, manifest=manifest
        )
        report["binding"] = binding(committed)
        # Must reject recovery before either node has committed.
        if committed["state"] != "PENDING":
            raise ValueError("experiment requires a fresh stripe ID")
        try:
            client.call(
                "reconstruct_missing_node",
                missing_nodes=[0],
                payload=blocks[1],
                **binding(committed),
            )
        except ValueError:
            report["uncommitted_recovery_rejected"] = True
        else:
            raise AssertionError("uncommitted recovery accepted")
        report["remote_before"], _ = client.call("get_metrics")
        submit_start = time.perf_counter()
        for node, block in enumerate(blocks):
            committed, _ = client.call(
                "submit_node_block", node_id=node, payload=block, **binding(committed)
            )
            if committed["state"] != ("PENDING" if node == 0 else "COMMITTED"):
                raise AssertionError("incorrect commit state")
        report["remote_committed_observed_ms"] = (
            time.perf_counter() - submit_start
        ) * 1000
        report["commit"] = committed
        status, parity = client.call("fetch_parity", **binding(committed))
        if sha(parity) != status["parity_sha256"]:
            raise AssertionError("parity integrity failure")
        report["fault_checks"] = verify_faults(client, committed, blocks)
        report["remote_after"], _ = client.call("get_metrics")
        report["network_A_after"] = network_observation(peer)
        report["nic_A_delta"] = network_delta(
            report["network_A_before"], report["network_A_after"]
        )
        report["nic_B_delta"] = network_delta(
            report["remote_before"]["network_B"], report["remote_after"]["network_B"]
        )
        report["physical_network"] = (
            "PHYSICAL_INTERFACE_ROUTES_OBSERVED"
            if not loopback
            and report["network_A_before"]["physical_device"]
            and report["remote_after"]["network_B"]["physical_device"]
            else "NOT VERIFIED"
        )
        submits = [t for t in client.transfers if t["op"] == "submit_node_block"]
        wire = sum(t["tx"]["stream_bytes"] for t in submits)
        payload = sum(t["tx"]["payload_bytes"] for t in submits)
        elapsed = sum(t["roundtrip_ms"] for t in submits)
        report["network"] = dict(
            submit_payload_bytes=payload,
            submit_application_stream_bytes=wire,
            framing_overhead_bytes=wire - payload,
            submit_roundtrip_ms=elapsed,
            effective_payload_mbps=(
                payload * 8 / (elapsed * 1000) if elapsed > 0 else None
            ),
            simulated_bandwidth_mbps=None,
            link_bandwidth_mbps=report["network_A_before"].get("link_speed_mbps"),
            note="application TCP bytes; excludes Ethernet/IP/TCP headers, ACKs and retransmits; goodput includes ACK/encode/validation wait; profile is a label, not a bandwidth emulator",
        )
        report["state"] = "PASS"
    except Exception as exc:
        report.update(state="FAIL", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        report.update(
            transfers=client.transfers,
            process_A_after=process_usage(),
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        report["process_A_delta"] = usage_delta(
            report["process_A_before"], report["process_A_after"]
        )
        if "remote_before" in report and "remote_after" in report:
            report["process_B_delta"] = usage_delta(
                report["remote_before"]["process"], report["remote_after"]["process"]
            )
        save_report(args.output / "result", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote-host", required=True)
    p.add_argument("--remote-port", type=int, default=30300)
    p.add_argument("--chunk-tokens", type=int, choices=(128, 256, 512), default=256)
    p.add_argument(
        "--network-profile",
        choices=("physical", "tc", "loopback-test"),
        default="physical",
    )
    p.add_argument("--stripe-id", required=True)
    p.add_argument("--node0-block", type=Path, required=True)
    p.add_argument("--node1-block", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--output", type=Path, required=True, help="new evidence directory")
    args = p.parse_args()
    print(json.dumps({"state": run(args)["state"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
