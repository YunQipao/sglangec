"""CPU-only remote RS parity in bounded process DRAM, no SGLang/Torch import.

Committed stripes retain parity + full manifest, NEVER original node blocks.
Immutable UUID handle and manifest hash bind retries to a server incarnation.
Expired IDs remain tombstoned until restart. Restart invalidates every handle.
Unauthenticated experimental protocol: bind only on an isolated trusted LAN.
"""

import argparse
from copy import deepcopy
import json
import math
import socketserver
import threading
import time
import uuid

from .node_block_export import descriptor, sha
from .remote_rs_protocol import receive, send
from .rs_codec import encode, recover
from .rs_stripe_builder import manifest_sha, validate_manifest


class StripeStore:
    def __init__(
        self, *, ttl=300, max_stripes=16, max_bytes=512 * 1024**2, allow_fixture=False
    ):
        if not math.isfinite(ttl) or ttl <= 0 or max_stripes < 1 or max_bytes < 1:
            raise ValueError("positive finite store budgets required")
        self.ttl, self.max_stripes, self.max_bytes = ttl, max_stripes, max_bytes
        self.allow_fixture = allow_fixture
        self.stripes = {}
        self.lock = threading.RLock()
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "rx_stream_bytes": 0,
            "tx_stream_bytes": 0,
            "rx_payload_bytes": 0,
            "tx_payload_bytes": 0,
            "receive_ms": 0.0,
            "send_ms": 0.0,
        }
        self.epoch = str(uuid.uuid4())
        from .rs_experiment import command
        from pathlib import Path

        self.environment = {
            "git": command(
                ["git", "rev-parse", "HEAD"], Path(__file__).resolve().parents[2]
            ),
            "uname": command(["uname", "-a"]),
        }

    def expire(self):
        for s in self.stripes.values():
            if s["state"] != "EXPIRED" and time.monotonic() >= s["deadline"]:
                s["state"] = "EXPIRED"
                s["blocks"].clear()
                s["parity"] = None

    def status(self, s):
        return {
            k: deepcopy(v)
            for k, v in s.items()
            if k not in ("blocks", "parity", "deadline", "created")
        }

    def usage(self):
        return sum(
            sum(map(len, s["blocks"].values()))
            + (len(s["parity"]) if s["parity"] is not None else 0)
            for s in self.stripes.values()
        )

    def dispatch(self, request, payload):
        with self.lock:
            self.expire()
            op = request["op"]
            if op not in ("submit_node_block", "reconstruct_missing_node") and payload:
                raise ValueError("unexpected payload")
            if op == "health":
                return {"state": "READY", "server_epoch": self.epoch}, b""
            if op == "get_metrics":
                from .rs_experiment import process_usage

                return (
                    dict(
                        self.metrics,
                        resident_data_bytes=self.usage(),
                        server_epoch=self.epoch,
                        process=process_usage(),
                        environment_B=self.environment,
                        stripes=len(self.stripes),
                    ),
                    b"",
                )
            sid = request["stripe_id"]
            if op == "create_stripe":
                manifest = request["manifest"]
                validate_manifest(manifest, allow_fixture=self.allow_fixture)
                if manifest["stripe_id"] != sid:
                    raise ValueError("stripe ID mismatch")
                h = manifest_sha(manifest)
                existing = self.stripes.get(sid)
                if existing is not None:
                    if (
                        existing["manifest_sha256"] != h
                        or existing["state"] == "EXPIRED"
                    ):
                        raise ValueError("stripe ID conflict/expired tombstone")
                    return self.status(existing), b""
                if len(self.stripes) >= self.max_stripes:
                    raise ValueError(
                        "stripe count budget exhausted; restart with fresh IDs"
                    )
                # Reserve 3*max node length for both pending inputs and parity
                # construction. Reservation is conservative and deterministic.
                reserved = 3 * max(d["length"] for d in manifest["nodes"])
                used = sum(
                    s["reserved_bytes"]
                    for s in self.stripes.values()
                    if s["state"] != "EXPIRED"
                )
                if used + reserved > self.max_bytes:
                    raise ValueError("DRAM reservation budget exceeded")
                s = dict(
                    stripe_id=sid,
                    handle=str(uuid.uuid4()),
                    manifest_sha256=h,
                    manifest=deepcopy(manifest),
                    state="PENDING",
                    blocks={},
                    parity=None,
                    submitted_nodes=[],
                    created=time.monotonic(),
                    deadline=time.monotonic() + self.ttl,
                    reserved_bytes=reserved,
                    rs_encode_ms=None,
                    commit_ms=None,
                )
                self.stripes[sid] = s
                return self.status(s), b""
            s = self.stripes.get(sid)
            if s is None:
                raise ValueError("unknown stripe")
            if (
                request.get("handle") != s["handle"]
                or request.get("manifest_sha256") != s["manifest_sha256"]
            ):
                raise ValueError("stale handle or stripe manifest mismatch")
            if s["state"] == "EXPIRED":
                raise ValueError("stripe expired")
            if op == "get_stripe_status":
                return self.status(s), b""
            if op == "submit_node_block":
                node = request["node_id"]
                if type(node) is not int or node not in (0, 1):
                    raise ValueError("invalid node")
                if (
                    descriptor(payload, allow_fixture=self.allow_fixture)
                    != s["manifest"]["nodes"][node]
                ):
                    raise ValueError("node data/identity differs from manifest")
                if time.monotonic() >= s["deadline"]:
                    self.expire()
                    raise ValueError("stripe expired during validation")
                # Exact retry is idempotent, even after original input release.
                if node in s["submitted_nodes"]:
                    return self.status(s), b""
                if s["state"] != "PENDING":
                    raise ValueError("stripe closed")
                if time.monotonic() >= s["deadline"]:
                    self.expire()
                    raise ValueError("stripe expired during validation")
                s["blocks"][node] = bytes(payload)
                s["submitted_nodes"].append(node)
                s["submitted_nodes"].sort()
                if len(s["blocks"]) == 2:
                    start = time.perf_counter()
                    parity = encode(s["blocks"][0], s["blocks"][1])
                    encode_ms = (time.perf_counter() - start) * 1000
                    if time.monotonic() >= s["deadline"]:
                        self.expire()
                        raise ValueError("stripe expired during encode")
                    s.update(
                        parity=parity,
                        parity_sha256=sha(parity),
                        rs_encode_ms=encode_ms,
                        commit_ms=(time.monotonic() - s["created"]) * 1000,
                        state="COMMITTED",
                    )
                    s["blocks"].clear()
                return self.status(s), b""
            if s["state"] != "COMMITTED" or s["parity"] is None:
                raise ValueError("committed parity required")
            if sha(s["parity"]) != s["parity_sha256"]:
                raise ValueError("stored parity corruption")
            if op == "fetch_parity":
                if time.monotonic() >= s["deadline"]:
                    self.expire()
                    raise ValueError("stripe expired during parity verification")
                return self.status(s), s["parity"]
            if op == "reconstruct_missing_node":
                missing = request["missing_nodes"]
                if (
                    not isinstance(missing, list)
                    or len(missing) != 1
                    or type(missing[0]) is not int
                    or missing[0] not in (0, 1)
                ):
                    raise ValueError("exactly one missing node required")
                lost = missing[0]
                if (
                    descriptor(payload, allow_fixture=self.allow_fixture)
                    != s["manifest"]["nodes"][1 - lost]
                ):
                    raise ValueError("survivor corruption/identity mismatch")
                start = time.perf_counter()
                blocks = [None, None]
                blocks[1 - lost] = payload
                raw = recover(
                    blocks, s["parity"], [d["length"] for d in s["manifest"]["nodes"]]
                )
                reconstruct_ms = (time.perf_counter() - start) * 1000
                verify_started = time.perf_counter()
                if (
                    descriptor(raw, allow_fixture=self.allow_fixture)
                    != s["manifest"]["nodes"][lost]
                ):
                    raise ValueError("recovered node integrity failure")
                decode_verify_ms = (time.perf_counter() - verify_started) * 1000
                if time.monotonic() >= s["deadline"]:
                    self.expire()
                    raise ValueError("stripe expired during recovery")
                return (
                    dict(
                        self.status(s),
                        recovered_node=lost,
                        rs_reconstruct_ms=reconstruct_ms,
                        node_unpack_verify_ms=decode_verify_ms,
                        recovery_ms=(time.perf_counter() - start) * 1000,
                    ),
                    raw,
                )
            raise ValueError("unsupported operation")


class RemoteRSServer(socketserver.TCPServer):
    # Serial bounded processing is deliberate: no unlimited connection threads,
    # fixed frame budget and fixed store reservation. Slow peers time out.
    allow_reuse_address = True

    def __init__(self, address, store=None, timeout=30):
        self.store = store or StripeStore()
        self.io_timeout = timeout
        super().__init__(address, Handler)

    def service_actions(self):
        with self.store.lock:
            self.store.expire()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        store = self.server.store
        with store.lock:
            store.metrics["requests"] += 1
        self.request.settimeout(self.server.io_timeout)

        def count(direction):
            def observe(size, is_payload):
                with store.lock:
                    store.metrics[direction + "_stream_bytes"] += size
                    if is_payload:
                        store.metrics[direction + "_payload_bytes"] += size

            return observe

        received_at = time.perf_counter()
        try:
            req, raw, metrics = receive(
                self.request, self.server.io_timeout, count("rx")
            )
            with store.lock:
                store.metrics["receive_ms"] += metrics["receive_ms"]
            result, payload = store.dispatch(req, raw)
            if req["op"] == "get_metrics":
                from .rs_experiment import network_observation

                result["network_B"] = network_observation(self.client_address[0])
            result = dict(result, ok=True, server_receive_ms=metrics["receive_ms"])
        except Exception as exc:
            with store.lock:
                store.metrics["errors"] += 1
                if "metrics" not in locals():
                    store.metrics["receive_ms"] += (
                        time.perf_counter() - received_at
                    ) * 1000
            result, payload = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}, b""
        try:
            self.request.settimeout(self.server.io_timeout)
            m = send(self.request, result, payload, count("tx"))
            with store.lock:
                store.metrics["send_ms"] += m["header_send_ms"] + m["payload_send_ms"]
        except OSError:
            pass  # Committed state survives lost ACK; retry with same handle.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30300)
    p.add_argument("--ttl", type=float, default=300)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--max-stripes", type=int, default=16)
    p.add_argument("--max-bytes", type=int, default=512 * 1024**2)
    args = p.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("positive finite timeout required")
    store = StripeStore(
        ttl=args.ttl, max_stripes=args.max_stripes, max_bytes=args.max_bytes
    )
    with RemoteRSServer((args.host, args.port), store, args.timeout) as server:
        print(
            json.dumps(
                {
                    "state": "READY",
                    "address": server.server_address,
                    "epoch": store.epoch,
                }
            ),
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
