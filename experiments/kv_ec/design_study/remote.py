"""Admitted parity-only / replica operations. No SSH or GPU launch."""

import argparse
import json
from pathlib import Path

from .admission import local_identity, placement
from .storage import ReplicaStore, get_replica, put_replica
from ..node_block_export import descriptor
from ..remote_rs_client import RemoteRSClient, binding
from ..remote_rs_server import RemoteRSServer, StripeStore


class TCPOnlyStripeStore(StripeStore):
    """Explicit TCP service; reject verbs requests before spawning helpers."""

    def dispatch(self, request, payload):
        if request.get("transport", "tcp") != "tcp" or request.get("op", "").startswith("prepare_rdma"):
            raise ValueError("physical243 TCP only; HostRDMA unavailable; no fallback")
        if request.get("op") == "register_node":
            import re
            if not isinstance(request.get("stripe_id"), str) or not re.fullmatch(
                    r"[A-Za-z0-9_.-]{1,128}", request["stripe_id"]):
                raise ValueError("bounded stripe ID required")
        return super().dispatch(request, payload)


def client(config, *, replica=False):
    placement(config)
    if config["transport"] != "tcp":
        raise ValueError("243 operation supports TCP only; RDMA fails closed")
    return RemoteRSClient(config["nodes"][2]["host"],
                          config["replica_port" if replica else "parity_port"], config["timeout_s"])


def register(config, role, physical_id, stripe_id, raw):
    local_identity(config, role, physical_id)
    if role not in ("D0", "D1"):
        raise ValueError("data role required")
    return client(config).register_node(stripe_id, int(role[-1]), raw)[0]


def submit(config, role, physical_id, status, raw):
    local_identity(config, role, physical_id)
    if role not in ("D0", "D1"):
        raise ValueError("data role required")
    node = int(role[-1])
    if descriptor(raw) != status["manifest"]["nodes"][node]:
        raise ValueError("local node differs from immutable registration")
    return client(config).call("submit_node_block", node_id=node, payload=raw, **binding(status))[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    sub = p.add_subparsers(dest="op", required=True)
    sub.add_parser("admit")
    s = sub.add_parser("serve")
    s.add_argument("--physical-id", required=True)
    s.add_argument("--kind", choices=("parity", "replica"), required=True)
    for op in ("register", "submit"):
        q = sub.add_parser(op)
        q.add_argument("--role", choices=("D0", "D1"), required=True)
        q.add_argument("--physical-id", required=True)
        q.add_argument("--node-block", type=Path, required=True)
        q.add_argument("--stripe-id", required=True)
        if op == "submit":
            q.add_argument("--status", type=Path, required=True)
        q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("status")
    q.add_argument("--stripe-id", required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("recover")
    q.add_argument("--status", type=Path, required=True)
    q.add_argument("--missing-node", type=int, choices=(0, 1), required=True)
    q.add_argument("--survivor", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    for op in ("put-replica", "get-replica"):
        q = sub.add_parser(op)
        q.add_argument("--output", type=Path, required=True)
        if op == "put-replica":
            q.add_argument("--node-block", type=Path, required=True)
            q.add_argument("--object-id", required=True)
            q.add_argument("--role", choices=("D0", "D1"), required=True)
            q.add_argument("--physical-id", required=True)
        else:
            q.add_argument("--binding", type=Path, required=True)
            q.add_argument("--descriptor", type=Path, required=True, help="pre-fault descriptor JSON; no lost KV bytes needed")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    fingerprint = placement(config)
    if args.op == "admit":
        print(json.dumps(dict(config_sha256=fingerprint, transport=config["transport"],
                              physical_connectivity="NOT TESTED", physical_failure="NOT TESTED")))
        return
    if args.op == "serve":
        node = local_identity(config, "P", args.physical_id)
        client(config)  # fail closed for any unimplemented transport before bind
        if args.kind == "parity":
            store = TCPOnlyStripeStore(ttl=config["ttl_s"], max_stripes=config["max_objects"],
                                max_bytes=config["max_bytes"], rdma_directory=Path.cwd() / "design-rdma-unused")
        else:
            store = ReplicaStore(ttl=config["ttl_s"], max_objects=config["max_objects"], max_bytes=config["max_bytes"])
        port = config["parity_port" if args.kind == "parity" else "replica_port"]
        with RemoteRSServer((node["host"], port), store, config["timeout_s"]) as server:
            print(json.dumps(dict(state="READY", kind=args.kind, transport="TCP", physical_id="243",
                                  config_sha256=fingerprint)), flush=True)
            server.serve_forever()
        return
    # Reserve output before any external mutation; never overwrite prior receipt.
    with args.output.open("xb") as stream:
        if args.op == "status":
            status, _ = client(config).call("get_stripe_status", stripe_id=args.stripe_id)
            stream.write(json.dumps(status, indent=2).encode())
            return
        if args.op == "recover":
            status = json.loads(args.status.read_text())
            missing = args.missing_node
            survivor = args.survivor.read_bytes()
            if status["state"] != "COMMITTED" or descriptor(survivor) != status["manifest"]["nodes"][1-missing]:
                raise ValueError("committed stripe and bound surviving node required")
            _, restored = client(config).call("reconstruct_missing_node", missing_nodes=[missing],
                                              payload=survivor, **binding(status))
            if descriptor(restored) != status["manifest"]["nodes"][missing]:
                raise ValueError("recovery integrity/request mismatch")
            stream.write(restored)
            return
        if args.op == "get-replica":
            key = json.loads(args.binding.read_text())
            stream.write(get_replica(client(config, replica=True), key, json.loads(args.descriptor.read_text())))
            return
        raw = args.node_block.read_bytes()
        if args.op == "put-replica":
            local_identity(config, args.role, args.physical_id)
            if descriptor(raw)["metadata"]["node_id"] != int(args.role[-1]):
                raise ValueError("replica source role mismatch")
            status = put_replica(client(config, replica=True), args.object_id, raw)
            stream.write(json.dumps(status, indent=2).encode())
            return
        if args.op == "register":
            status = register(config, args.role, args.physical_id, args.stripe_id, raw)
        else:
            prior = json.loads(args.status.read_text())
            if prior["stripe_id"] != args.stripe_id:
                raise ValueError("stripe mismatch")
            status = submit(config, args.role, args.physical_id, prior, raw)
        stream.write(json.dumps(status, indent=2).encode())


if __name__ == "__main__":
    main()
