"""TCP receiver and opaque Gateway relay for the logical two-node prototype."""

import argparse
from collections import OrderedDict
import json
import logging
import os
import queue
import socket
import socketserver
import threading
import time

try:
    from .gateway_protocol import (PROTOCOL, encode_ack, encode_frame, recv_frame,
                                   recv_ack, recv_exact)
except ImportError:
    from gateway_protocol import PROTOCOL, encode_ack, encode_frame, recv_frame, recv_ack, recv_exact

LOG = logging.getLogger("kv_ec_gateway")


def _read_ack(sock):
    import struct
    n, = struct.unpack("!I", recv_exact(sock, 4))
    if not 0 < n <= 4096:
        raise ValueError("invalid_receiver_ack")
    return json.loads(recv_exact(sock, n).decode())


class _BoundedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, *, max_connections=8, timeout_s=10):
        self.slots = threading.BoundedSemaphore(max_connections)
        self.timeout_s = timeout_s
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(encode_ack({"ok": False, "reason": "connection_capacity"}))
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            request.settimeout(self.timeout_s)
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class RelayState:
    """Bounded idempotency receipts and counters; no payloads retained."""

    def __init__(self, receiver, *, manager_epoch, source_nodes=("node0", "node1"),
                 destination_node="node1", timeout_s=10, max_receipts=65536):
        self.receiver = receiver
        self.manager_epoch = manager_epoch
        self.source_nodes = set(source_nodes)
        self.destination_node = destination_node
        self.timeout_s = timeout_s
        self.max_receipts = max_receipts
        self.receipts = OrderedDict()
        self.lock = threading.Lock()
        self.metrics = {"rx_bytes": 0, "tx_bytes": 0, "chunks": 0,
                        "retries": 0, "failures": 0, "queue_delay_ms_sum": 0.0,
                        "relay_ms_sum": 0.0}

    @staticmethod
    def key(h):
        return tuple(h[k] for k in ("transaction_id", "stripe_id", "checkpoint_id",
                                    "rank_id", "chunk_index"))

    def relay(self, header, payload):
        if header["manager_epoch"] != self.manager_epoch:
            raise ValueError("manager_epoch_mismatch")
        if header["source_node"] not in self.source_nodes or header["destination_node"] != self.destination_node:
            raise ValueError("route_not_allowed")
        key, digest = self.key(header), header["chunk_sha256"]
        with self.lock:
            previous = self.receipts.get(key)
            if previous is not None:
                if previous != digest:
                    raise ValueError("duplicate_chunk_conflict")
                self.metrics["retries"] += 1
                self.receipts.move_to_end(key)
                return {"ok": True, "duplicate": True, "chunk_sha256": digest}
        started = time.monotonic()
        wire = encode_frame(header, payload)
        with socket.create_connection(self.receiver, timeout=self.timeout_s) as downstream:
            downstream.settimeout(self.timeout_s)
            downstream.sendall(wire)
            ack = _read_ack(downstream)
        if ack.get("ok") is not True or ack.get("chunk_sha256") != digest:
            raise RuntimeError("receiver_ack_mismatch")
        elapsed_ms = (time.monotonic() - started) * 1000
        with self.lock:
            existing = self.receipts.get(key)
            if existing is not None and existing != digest:
                raise ValueError("duplicate_chunk_conflict")
            if existing is None:
                if len(self.receipts) >= self.max_receipts:
                    self.receipts.popitem(last=False)
                self.receipts[key] = digest
                self.metrics["chunks"] += 1
                self.metrics["rx_bytes"] += len(payload)
                self.metrics["tx_bytes"] += len(payload)
                self.metrics["relay_ms_sum"] += elapsed_ms
        return {"ok": True, "duplicate": False, "chunk_sha256": digest,
                "relay_ms": elapsed_ms}


def make_gateway_handler(state):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                header, payload = recv_frame(self.request)
                ack = state.relay(header, payload)
                self.request.sendall(encode_ack(ack))
            except Exception as exc:
                with state.lock:
                    state.metrics["failures"] += 1
                LOG.warning("relay rejected (%s)", type(exc).__name__)
                try:
                    self.request.sendall(encode_ack({"ok": False,
                        "reason": str(exc)[:160] or type(exc).__name__}))
                except OSError:
                    pass
    return Handler


class ReceiverState:
    def __init__(self, *, manager_epoch, node_id="node1", max_queue=32):
        self.manager_epoch, self.node_id = manager_epoch, node_id
        self.items = queue.Queue(maxsize=max_queue)
        self.seen = {}
        self.lock = threading.Lock()
        self.received_bytes = 0
        self.received_chunks = 0

    def accept(self, header, payload):
        if header["manager_epoch"] != self.manager_epoch:
            raise ValueError("manager_epoch_mismatch")
        if header["destination_node"] != self.node_id:
            raise ValueError("wrong_destination_node")
        key = RelayState.key(header)
        digest = header["chunk_sha256"]
        with self.lock:
            if key in self.seen:
                if self.seen[key] != digest:
                    raise ValueError("duplicate_chunk_conflict")
                return {"ok": True, "duplicate": True, "chunk_sha256": digest}
            # Queue owns bytes only in bounded volatile memory; consumers can
            # stream them directly into XOR/reassembly. No disk writes occur.
            self.items.put((header, payload), timeout=2)
            self.seen[key] = digest
            self.received_bytes += len(payload)
            self.received_chunks += 1
        return {"ok": True, "duplicate": False, "chunk_sha256": digest}


def make_receiver_handler(state):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                header, payload = recv_frame(self.request)
                ack = state.accept(header, payload)
            except Exception as exc:
                ack = {"ok": False, "reason": str(exc)[:160] or type(exc).__name__}
            try:
                self.request.sendall(encode_ack(ack))
            except OSError:
                pass
    return Handler


def serve_receiver(host, port, epoch, *, node_id="node1", max_connections=8, max_queue=256):
    state = ReceiverState(manager_epoch=epoch, node_id=node_id, max_queue=max_queue)
    server = _BoundedTCPServer((host, port), make_receiver_handler(state), max_connections=max_connections)
    server.receiver_state = state
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("gateway", "receiver"), required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--receiver", help="gateway downstream host:port")
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--node-id", default="node1")
    parser.add_argument("--max-connections", type=int, default=8)
    parser.add_argument("--max-queue", type=int, default=256,
                        help="finite volatile chunk slots (up to 256 MiB at max chunk size)")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--metrics-json", help="write metadata-only counters on graceful shutdown")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.role == "receiver":
        server = serve_receiver(args.bind, args.port, args.epoch, node_id=args.node_id,
                                max_connections=args.max_connections, max_queue=args.max_queue)
    else:
        if not args.receiver or ":" not in args.receiver:
            parser.error("gateway requires --receiver HOST:PORT")
        host, port = args.receiver.rsplit(":", 1)
        state = RelayState((host, int(port)), manager_epoch=args.epoch,
                           timeout_s=args.timeout)
        server = _BoundedTCPServer((args.bind, args.port), make_gateway_handler(state),
                                   max_connections=args.max_connections, timeout_s=args.timeout)
        server.relay_state = state
    LOG.info("listening role=%s endpoint=%s:%d epoch=%s", args.role, args.bind, args.port, args.epoch)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if args.metrics_json:
            if args.role == "gateway":
                report = dict(role="gateway", endpoint=list(server.server_address),
                              receiver=list(state.receiver), manager_epoch=args.epoch,
                              metrics=dict(state.metrics), payload_persisted=False)
            else:
                rs = server.receiver_state
                report = dict(role="receiver", endpoint=list(server.server_address),
                              node_id=args.node_id, manager_epoch=args.epoch,
                              chunks=rs.received_chunks, payload_bytes=rs.received_bytes,
                              queue_depth=rs.items.qsize(), queue_capacity=rs.items.maxsize,
                              payload_persisted=False)
            temporary = args.metrics_json + ".writing"
            with open(temporary, "x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, args.metrics_json)


if __name__ == "__main__":
    main()
