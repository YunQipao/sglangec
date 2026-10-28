"""Verified TCP control operations with optional native RDMA payloads."""

import argparse
import json
import math
from pathlib import Path
import socket
import threading
import time

from .remote_rs_protocol import rdma_diagnostic, receive, send
from .node_block_export import descriptor as describe_node
from .node_block_export import sha
from .perf_observer import observed


class RemoteRSClient:
    def __init__(self, host, port=30300, timeout=30):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite timeout required")
        self.host, self.port, self.timeout = host, port, timeout
        self.transfers = []

    @staticmethod
    def _rdma_item(status, key):
        values = status.get("rdma", {})
        return values.get(key, values.get(str(key), {}))

    def _receive_async(self, port, destination, *, device, gid_index,
                       port_num, timeout):
        from .multinode.rdma_transport import receive_file

        result = {}
        done = threading.Event()

        def run():
            try:
                result["stats"] = receive_file(
                    port, destination, device=device, gid_index=gid_index,
                    port_num=port_num, timeout=timeout, return_stats=True,
                )
            except Exception as exc:  # re-raised by wait below
                result["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        # The helper owns the listener and has no Python-level ready callback.
        # Give it a scheduling slice before issuing the control request.  The
        # sender itself retries the TCP metadata connection for ~30 seconds.
        time.sleep(0.05)
        return thread, done, result

    def _wait_rdma_status(self, stripe_id, key, binding_fields, timeout,
                          poll_interval=0.1):
        deadline = time.monotonic() + timeout
        status = None
        while time.monotonic() < deadline:
            status, _ = self.call("get_stripe_status", stripe_id=stripe_id,
                                  **binding_fields)
            item = self._rdma_item(status, key)
            if item.get("state") == "COMPLETED":
                return status, item
            if item.get("state") == "FAILED":
                raise RuntimeError(item.get("error", "RDMA transfer failed"))
            time.sleep(poll_interval)
        raise TimeoutError(f"RDMA transfer did not complete: {status}")

    @observed("rs_upload_inclusive", "communication")
    def rdma_submit_node_block(self, stripe_id, node_id, block, *, handle,
                               manifest_sha256, device, gid_index,
                               port_num=1, timeout=120):
        """Register an RDMA receive on the server, then perform a real Write."""
        block = bytes(block)
        block_sha256 = sha(block)
        rdma_diagnostic(
            "client", stripe_id=stripe_id, node_id=node_id,
            block_length=len(block), block_sha256=block_sha256,
            manifest_sha256=manifest_sha256,
        )
        status, _ = self.call(
            "prepare_rdma", stripe_id=stripe_id, node_id=node_id,
            payload_length=len(block), node_block_sha256=block_sha256,
            handle=handle, manifest_sha256=manifest_sha256,
        )
        item = self._rdma_item(status, node_id)
        if item.get("state") == "COMPLETED":
            return status, item.get("stats", {}), "idempotent"
        port = item.get("port")
        if type(port) is not int:
            raise RuntimeError("server did not return an RDMA receive port")
        from .multinode.rdma_transport import RDMATransport

        tx = RDMATransport(device=device, gid_index=gid_index,
                           port_num=port_num, timeout=timeout)
        try:
            stats = tx.send_with_stats(self.host, port, block)
        finally:
            tx.close()
        status, item = self._wait_rdma_status(
            stripe_id, node_id,
            {"handle": handle, "manifest_sha256": manifest_sha256}, timeout,
        )
        if status.get("state") not in ("PENDING", "COMMITTED"):
            raise RuntimeError(f"RDMA upload did not commit: {status}")
        record = dict(op="submit_node_block", transport="rdma",
                      rdma=stats, payload_bytes=len(block))
        self.transfers.append(record)
        return status, stats, "completed"

    def rdma_fetch_parity(self, stripe_id, *, handle, manifest_sha256,
                          destination, return_host, return_port, device,
                          gid_index, port_num=1, timeout=120):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        thread, done, result = self._receive_async(
            return_port, destination, device=device, gid_index=gid_index,
            port_num=port_num, timeout=timeout,
        )
        status, _ = self.call(
            "fetch_parity", stripe_id=stripe_id, transport="rdma",
            rdma_host=return_host, rdma_port=return_port,
            handle=handle, manifest_sha256=manifest_sha256,
        )
        status, item = self._wait_rdma_status(
            stripe_id, "parity_download",
            {"handle": handle, "manifest_sha256": manifest_sha256}, timeout,
        )
        thread.join(timeout=max(0.0, timeout))
        if not done.is_set():
            raise TimeoutError("RDMA parity receiver did not finish")
        if "error" in result:
            raise result["error"]
        raw = destination.read_bytes()
        if sha(raw) != status.get("parity_sha256") or sha(raw) != item.get("payload_sha256"):
            raise ValueError("RDMA parity checksum mismatch")
        return status, raw, item

    @observed("rs_recovery_rpc_inclusive", "recovery")
    def rdma_reconstruct_missing_node(self, stripe_id, missing_node,
                                      survivor, *, handle, manifest_sha256,
                                      destination, return_host, return_port,
                                      device, gid_index, port_num=1,
                                      timeout=120):
        survivor = bytes(survivor)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        receiver, done, result = self._receive_async(
            return_port, destination, device=device, gid_index=gid_index,
            port_num=port_num, timeout=timeout,
        )
        status, _ = self.call(
            "prepare_rdma_recovery", stripe_id=stripe_id,
            missing_nodes=[missing_node], transport="rdma",
            rdma_host=return_host, rdma_return_port=return_port,
            survivor_length=len(survivor), survivor_sha256=sha(survivor),
            handle=handle, manifest_sha256=manifest_sha256,
        )
        key = f"recovery_{missing_node}"
        item = self._rdma_item(status, key)
        if item.get("state") == "COMPLETED":
            raise RuntimeError("RDMA recovery already completed; submit a fresh recovery request")
        port = item.get("port")
        if type(port) is not int:
            raise RuntimeError("server did not return an RDMA recovery port")
        from .multinode.rdma_transport import RDMATransport

        tx = RDMATransport(device=device, gid_index=gid_index,
                           port_num=port_num, timeout=timeout)
        try:
            send_stats = tx.send_with_stats(self.host, port, survivor)
        finally:
            tx.close()
        if send_stats.get("elapsed_s") is not None:
            from .perf_observer import emit
            emit("survivor_upload", "communication", source="rdma_cq",
                 duration_ms=send_stats["elapsed_s"] * 1000, bytes=len(survivor),
                 native_stats=send_stats)
        status, item = self._wait_rdma_status(
            stripe_id, key,
            {"handle": handle, "manifest_sha256": manifest_sha256}, timeout,
        )
        receiver.join(timeout=max(0.0, timeout))
        if not done.is_set():
            raise TimeoutError("RDMA recovery receiver did not finish")
        if "error" in result:
            raise result["error"]
        raw = destination.read_bytes()
        if result.get("stats", {}).get("elapsed_s") is not None:
            from .perf_observer import emit
            emit("recovered_block_return", "communication", source="rdma_cq",
                 duration_ms=result["stats"]["elapsed_s"] * 1000, bytes=len(raw),
                 native_stats=result["stats"])
        expected = status["manifest"]["nodes"][missing_node]["sha256"]
        if sha(raw) != expected or sha(raw) != item.get("payload_sha256"):
            raise ValueError("RDMA recovered node checksum mismatch")
        # Keep all four native legs. ``item.send_stats`` belongs to the server
        # and previously hid the client's survivor upload under the same key.
        return status, raw, {**item, "client_send_stats": send_stats,
                             "client_receive_stats": result["stats"]}

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
            op=op, transport="tcp",
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

    def register_node(self, stripe_id, node_id, block, *, allow_fixture=False):
        """Register a node descriptor before uploading its bytes."""
        from .node_block_export import descriptor
        meta = descriptor(block, allow_fixture=allow_fixture)
        return self.call("register_node", stripe_id=stripe_id, node_id=node_id,
                         descriptor=meta)


def binding(status):
    return {k: status[k] for k in ("stripe_id", "handle", "manifest_sha256")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote-host", required=True)
    p.add_argument("--remote-port", type=int, default=30300)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--transport", choices=("tcp", "rdma"), default="tcp")
    p.add_argument("--rdma-device", default="mlx5_0")
    p.add_argument("--rdma-gid-index", type=int)
    p.add_argument("--rdma-port-num", type=int, default=1)
    p.add_argument("--rdma-return-host", default="127.0.0.1")
    p.add_argument("--rdma-return-port", type=int, default=30490)
    p.add_argument(
        "--op",
        choices=(
            "health",
            "register_node",
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
    p.add_argument("--descriptor", type=Path,
                   help="descriptor JSON for register_node (or export.json containing descriptor)")
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
    elif args.op == "register_node":
        if args.node_id is None:
            p.error("--node-id required")
        if args.descriptor:
            value = json.loads(args.descriptor.read_text())
            fields["descriptor"] = value.get("descriptor", value)
        elif args.payload:
            fields["descriptor"] = describe_node(args.payload.read_bytes())
        else:
            p.error("--descriptor or --payload required")
        fields["node_id"] = args.node_id
    elif args.op not in ("health", "get_metrics"):
        if args.op == "get_stripe_status" and not args.binding:
            # Safe read-only poll while the second physical node is registering.
            pass
        elif not args.binding:
            p.error("--binding required")
        else:
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
        wire_payload = (
            args.payload.read_bytes()
            if args.payload and args.op in ("submit_node_block", "reconstruct_missing_node")
            else b""
        )
        rdma_ops = {"submit_node_block", "fetch_parity", "reconstruct_missing_node"}
        if args.transport == "tcp" or args.op not in rdma_ops:
            response, raw = client.call(args.op, payload=wire_payload, **fields)
        else:
            response, raw = None, b""
        rdma_stats = None
        if args.transport == "rdma" and args.op in rdma_ops:
            saved = json.loads(args.binding.read_text())
            saved = saved.get("response", saved)
            key = binding(saved)
            if args.op == "submit_node_block":
                response, rdma_stats, _ = client.rdma_submit_node_block(
                    args.stripe_id, args.node_id, wire_payload, **key,
                    device=args.rdma_device, gid_index=args.rdma_gid_index,
                    port_num=args.rdma_port_num, timeout=args.timeout,
                )
            elif args.op == "fetch_parity":
                response, raw, rdma_stats = client.rdma_fetch_parity(
                    args.stripe_id, **key, destination=args.output / "payload.bin",
                    return_host=args.rdma_return_host,
                    return_port=args.rdma_return_port,
                    device=args.rdma_device, gid_index=args.rdma_gid_index,
                    port_num=args.rdma_port_num, timeout=args.timeout,
                )
            elif args.op == "reconstruct_missing_node":
                response, raw, rdma_stats = client.rdma_reconstruct_missing_node(
                    args.stripe_id, args.missing_nodes[0], wire_payload, **key,
                    destination=args.output / "payload.bin",
                    return_host=args.rdma_return_host,
                    return_port=args.rdma_return_port,
                    device=args.rdma_device, gid_index=args.rdma_gid_index,
                    port_num=args.rdma_port_num, timeout=args.timeout,
                )
            elif args.op not in ("health", "get_metrics", "register_node", "create_stripe", "get_stripe_status", "prepare_rdma"):
                p.error(f"RDMA transport is not implemented for {args.op}")
        report = dict(response=response, transfers=client.transfers, state="PASS")
        if rdma_stats is not None:
            report.update(transport="rdma", rdma=rdma_stats)
        if raw:
            payload_path = args.output / "payload.bin"
            if not payload_path.exists():
                with payload_path.open("xb") as f:
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
