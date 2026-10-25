"""One TCP connection per remote operation, verified binary responses."""

import argparse
import json
import math
from pathlib import Path
import socket
import time

from .remote_rs_protocol import receive, send


class RemoteRSClient:
    def __init__(self, host, port=30300, timeout=30):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite timeout required")
        self.host, self.port, self.timeout = host, port, timeout
        self.transfers = []

    def call(self, op, *, payload=b"", **fields):
        start = time.perf_counter()
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        ) as sock:
            connected = time.perf_counter()
            peer, local = sock.getpeername(), sock.getsockname()
            tx = send(sock, dict(op=op, **fields), payload)
            response, raw, rx = receive(sock, self.timeout)
        record = dict(
            op=op,
            peer=peer,
            local=local,
            connect_ms=(connected - start) * 1000,
            roundtrip_ms=(time.perf_counter() - start) * 1000,
            tx=tx,
            rx=rx,
            server_receive_ms=response.get("server_receive_ms"),
        )
        self.transfers.append(record)
        if response.get("ok") is not True:
            raise ValueError(response.get("error", "remote operation failed"))
        return response, raw


def binding(status):
    return {k: status[k] for k in ("stripe_id", "handle", "manifest_sha256")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote-host", required=True)
    p.add_argument("--remote-port", type=int, default=30300)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument(
        "--op",
        choices=(
            "health",
            "create_stripe",
            "submit_node_block",
            "get_stripe_status",
            "fetch_parity",
            "reconstruct_missing_node",
            "get_metrics",
        ),
        required=True,
    )
    p.add_argument("--stripe-id")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--binding", type=Path, help="prior create/status JSON or report")
    p.add_argument("--node-id", type=int, choices=(0, 1))
    p.add_argument("--missing-nodes", type=int, nargs="+", choices=(0, 1))
    p.add_argument("--payload", type=Path)
    p.add_argument("--output", type=Path, required=True, help="new result directory")
    args = p.parse_args()
    fields = {}
    if args.op not in ("health", "get_metrics"):
        if not args.stripe_id:
            p.error("--stripe-id required")
        fields["stripe_id"] = args.stripe_id
    if args.op == "create_stripe":
        if not args.manifest:
            p.error("--manifest required")
        fields["manifest"] = json.loads(args.manifest.read_text())
    elif args.op not in ("health", "get_metrics"):
        if not args.binding:
            p.error("--binding required")
        saved = json.loads(args.binding.read_text())
        saved = saved.get("response", saved)
        if saved["stripe_id"] != args.stripe_id:
            p.error("binding stripe mismatch")
        fields.update(binding(saved))
    if args.op == "submit_node_block":
        if args.node_id is None or args.payload is None:
            p.error("--node-id and --payload required")
        fields["node_id"] = args.node_id
    if args.op == "reconstruct_missing_node":
        if args.missing_nodes is None or args.payload is None:
            p.error("--missing-nodes and survivor --payload required")
        fields["missing_nodes"] = args.missing_nodes
    # Reserve the output before creating remote state; never overwrite evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    client = RemoteRSClient(args.remote_host, args.remote_port, args.timeout)
    from .rs_experiment import save_report

    try:
        response, raw = client.call(
            args.op,
            payload=args.payload.read_bytes() if args.payload else b"",
            **fields
        )
        report = dict(response=response, transfers=client.transfers, state="PASS")
        if raw:
            with (args.output / "payload.bin").open("xb") as f:
                f.write(raw)
    except Exception as exc:
        save_report(
            args.output / "result",
            dict(state="FAIL", error=str(exc), transfers=client.transfers),
        )
        raise
    save_report(args.output / "result", report)
    print(json.dumps(response))


if __name__ == "__main__":
    main()
