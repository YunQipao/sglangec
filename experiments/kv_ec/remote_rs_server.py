"""CPU-only remote RS parity in bounded process DRAM, no SGLang/Torch import.

Committed stripes retain parity + full manifest, NEVER original node blocks.
Immutable UUID handle and manifest hash bind retries to a server incarnation.
Expired IDs remain tombstoned until restart. Restart invalidates every handle.
Unauthenticated experimental protocol: bind only on an isolated trusted LAN.

When configured with ``--rdma-device``, TCP remains the control channel while
Node Blocks, parity, and recovery payloads use the native verbs transport in
``multinode.rdma_transport``.  The server never changes an RDMA request into a
TCP payload after a verbs error; the per-stripe RDMA state is FAILED instead.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import socketserver
import tempfile
import threading
import time
import uuid

from .node_block_export import descriptor, sha, MAX_NODE_BYTES
from .remote_rs_protocol import RDMA_NODE_SHA256_FIELD, rdma_diagnostic, receive, send
from .rs_codec import CODE
from .cuda_rs_codec import CudaRSCodec
from .rs_stripe_builder import manifest_sha, validate_manifest


class StripeStore:
    def __init__(
        self, *, ttl=300, max_stripes=16, max_bytes=512 * 1024**2, allow_fixture=False,
        rdma_device="-", rdma_gid_index=None, rdma_port_num=1,
        rdma_port_base=30400, rdma_timeout=120, rdma_directory=None,
        rs_backend="auto"
    ):
        if (not math.isfinite(ttl) or ttl <= 0 or max_stripes < 1 or max_bytes < 1
                or not math.isfinite(float(rdma_timeout)) or float(rdma_timeout) <= 0
                or not 1 <= int(rdma_port_num) <= 255
                or not 1 <= int(rdma_port_base) <= 65535):
            raise ValueError("positive finite store budgets required")
        self.ttl, self.max_stripes, self.max_bytes = ttl, max_stripes, max_bytes
        self.allow_fixture = allow_fixture
        self.rdma_device = rdma_device
        self.rdma_gid_index = rdma_gid_index
        self.rdma_port_num = int(rdma_port_num)
        self.rdma_port_base = int(rdma_port_base)
        self.rdma_timeout = float(rdma_timeout)
        self.rdma_directory = Path(rdma_directory or tempfile.mkdtemp(prefix="kv-ec-rdma-server-"))
        self.rdma_directory.mkdir(parents=True, exist_ok=True)
        self.rdma_ports = {}
        self.rdma_threads = {}
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
            "rdma_rx_payload_bytes": 0,
            "rdma_tx_payload_bytes": 0,
            "rdma_completion_count": 0,
            "rdma_transfer_ms": 0.0,
        }
        self.epoch = str(uuid.uuid4())
        # auto prefers the fused CUDA kernel and falls back to the vectorized
        # CPU oracle on parity-only hosts.  Selection is recorded in status.
        self.rs_codec = CudaRSCodec(rs_backend)
        from .rs_experiment import command

        self.environment = {
            "git": command(
                ["git", "rev-parse", "HEAD"], Path(__file__).resolve().parents[2]
            ),
            "uname": command(["uname", "-a"]),
        }

    def _rdma_config(self):
        return {"device": self.rdma_device, "gid_index": self.rdma_gid_index,
                "port_num": self.rdma_port_num, "port_base": self.rdma_port_base,
                "timeout_s": self.rdma_timeout}

    def _rdma_port_locked(self, stripe_id, purpose):
        """Reserve a TCP rendezvous port for one verbs helper invocation."""
        for offset in range(max(4, self.max_stripes * 4)):
            port = self.rdma_port_base + offset
            if port > 65535:
                break
            if port not in self.rdma_ports:
                self.rdma_ports[port] = (stripe_id, purpose)
                return port
        raise ValueError("no free RDMA rendezvous port")

    def _release_rdma_port_locked(self, port):
        self.rdma_ports.pop(port, None)

    def _commit_block_locked(self, s, node, payload):
        """Validate and commit one complete node block.

        This is shared by the TCP handler and the asynchronous RDMA receiver so
        both transports have exactly the same identity/lifetime checks.
        """
        if descriptor(payload, allow_fixture=self.allow_fixture) != s["manifest"]["nodes"][node]:
            raise ValueError("node data/identity differs from manifest")
        if node in s["submitted_nodes"]:
            return
        if s["state"] != "PENDING":
            raise ValueError("stripe closed")
        if time.monotonic() >= s["deadline"]:
            self.expire()
            raise ValueError("stripe expired during validation")
        s["blocks"][node] = bytes(payload)
        s["submitted_nodes"].append(node)
        s["submitted_nodes"].sort()
        if len(s["blocks"]) == 2:
            from .perf_observer import span
            with span("rs_parity_encode", "ec_compute", stripe_id=s["manifest"]["stripe_id"]):
                start = time.perf_counter()
                parity = self.rs_codec.encode_xor(s["blocks"][0], s["blocks"][1])
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
            # A committed store deliberately does not retain either source
            # Node Block.  The manifest and parity are sufficient for the
            # single-erasure protocol, while retrying the original upload is
            # still idempotent through submitted_nodes.
            s["blocks"].clear()

    def _init_streaming_locked(self, s, chunk_bytes):
        lengths = [d["length"] for d in s["manifest"]["nodes"]]
        width = max(lengths)
        chunk = chunk_bytes
        s["streaming"] = dict(
            chunk_bytes=chunk,
            nodes={i: bytearray(lengths[i]) for i in (0, 1)},
            received={0: set(), 1: set()},
            parity=bytearray(width),
            encoded=set(),
            encode_ms=0.0,
        )

    def _submit_stream_chunk_locked(self, s, node, offset, payload, chunk_bytes):
        if s["state"] != "PENDING" or node not in (0, 1):
            raise ValueError("streaming stripe not pending")
        if type(chunk_bytes) is not int or not 1024**2 <= chunk_bytes <= 64 * 1024**2:
            raise ValueError("invalid streaming node chunk size")
        state = s.get("streaming")
        if state is None:
            self._init_streaming_locked(s, chunk_bytes)
            state = s["streaming"]
        elif state["chunk_bytes"] != chunk_bytes:
            raise ValueError("streaming node chunk size changed mid-stripe")
        chunk = state["chunk_bytes"]
        expected_len = s["manifest"]["nodes"][node]["length"]
        expected = min(chunk, expected_len - offset)
        if (
            type(offset) is not int or offset < 0 or offset % chunk
            or expected <= 0 or len(payload) != expected
        ):
            raise ValueError("invalid streaming node chunk geometry")
        index = offset // chunk
        prior = state["received"][node]
        if index in prior:
            if bytes(state["nodes"][node][offset:offset + expected]) != payload:
                raise ValueError("conflicting streaming chunk retry")
            return
        state["nodes"][node][offset:offset + expected] = payload
        prior.add(index)
        other = 1 - node
        if index in state["received"][other] and index not in state["encoded"]:
            width = min(chunk, len(state["parity"]) - offset)
            a = bytes(state["nodes"][0][offset:min(offset + width, len(state["nodes"][0]))])
            b = bytes(state["nodes"][1][offset:min(offset + width, len(state["nodes"][1]))])
            a += bytes(width - len(a)); b += bytes(width - len(b))
            started = time.perf_counter()
            encoded = self.rs_codec.encode_xor(a, b)
            state["encode_ms"] += (time.perf_counter() - started) * 1000
            state["parity"][offset:offset + width] = encoded
            state["encoded"].add(index)
        total_chunks = (expected_len + chunk - 1) // chunk
        if len(prior) == total_chunks and node not in s["submitted_nodes"]:
            raw = bytes(state["nodes"][node])
            if descriptor(raw, allow_fixture=self.allow_fixture) != s["manifest"]["nodes"][node]:
                raise ValueError("streamed node integrity differs from manifest")
            s["submitted_nodes"].append(node); s["submitted_nodes"].sort()
        max_chunks = (len(state["parity"]) + chunk - 1) // chunk
        if len(s["submitted_nodes"]) == 2 and len(state["encoded"]) == max_chunks:
            parity = bytes(state["parity"])
            s.update(
                parity=parity, parity_sha256=sha(parity),
                rs_encode_ms=state["encode_ms"],
                commit_ms=(time.monotonic() - s["created"]) * 1000,
                state="COMMITTED", streaming_pipeline=True,
                streaming_chunk_bytes=chunk,
            )
            s["streaming"] = None

    def _rdma_receive_worker(self, sid, node, port, expected_length, expected_sha):
        from .multinode.rdma_transport import receive_file

        path = self.rdma_directory / f"{sid}-{node}-{uuid.uuid4().hex}.bin"
        try:
            started = time.perf_counter()
            stats = receive_file(
                port, path, device=self.rdma_device,
                gid_index=self.rdma_gid_index, port_num=self.rdma_port_num,
                timeout=self.rdma_timeout, return_stats=True,
            )
            payload = path.read_bytes()
            if len(payload) != expected_length:
                raise ValueError("RDMA payload length mismatch")
            if sha(payload) != expected_sha:
                raise ValueError("RDMA payload SHA256 mismatch")
            with self.lock:
                s = self.stripes.get(sid)
                if s is None or s["state"] == "EXPIRED":
                    raise ValueError("stripe expired during RDMA receive")
                self._commit_block_locked(s, node, payload)
                item = s.setdefault("rdma", {}).setdefault(node, {})
                item.update(
                    state="COMPLETED", payload_bytes=len(payload),
                    payload_sha256=sha(payload), stats=stats,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                self.metrics["rdma_rx_payload_bytes"] += len(payload)
                self.metrics["rdma_completion_count"] += 1
                self.metrics["rdma_transfer_ms"] += stats.get("elapsed_s", 0.0) * 1000
        except Exception as exc:
            with self.lock:
                s = self.stripes.get(sid)
                if s is not None:
                    s.setdefault("rdma", {}).setdefault(node, {}).update(
                        state="FAILED", error=f"{type(exc).__name__}:{exc}"
                    )
        finally:
            path.unlink(missing_ok=True)
            with self.lock:
                self._release_rdma_port_locked(port)
                self.rdma_threads.pop((sid, "upload", node), None)

    def _rdma_send_worker(self, sid, purpose, host, port, payload, expected_sha):
        from .multinode.rdma_transport import RDMATransport

        try:
            tx = RDMATransport(
                device=self.rdma_device, gid_index=self.rdma_gid_index,
                port_num=self.rdma_port_num, timeout=self.rdma_timeout,
            )
            stats = tx.send_with_stats(host, port, payload)
            tx.close()
            if sha(payload) != expected_sha:
                raise ValueError("internal RDMA send checksum mismatch")
            with self.lock:
                s = self.stripes.get(sid)
                if s is not None:
                    s.setdefault("rdma", {}).setdefault(purpose, {}).update(
                        state="COMPLETED", payload_bytes=len(payload),
                        payload_sha256=expected_sha, stats=stats,
                    )
                    self.metrics["rdma_tx_payload_bytes"] += len(payload)
                    self.metrics["rdma_completion_count"] += 1
                    self.metrics["rdma_transfer_ms"] += stats.get("elapsed_s", 0.0) * 1000
        except Exception as exc:
            with self.lock:
                s = self.stripes.get(sid)
                if s is not None:
                    s.setdefault("rdma", {}).setdefault(purpose, {}).update(
                        state="FAILED", error=f"{type(exc).__name__}:{exc}"
                    )
        finally:
            with self.lock:
                self.rdma_threads.pop((sid, "download", purpose), None)

    def _rdma_recovery_worker(self, sid, missing, port, expected_length,
                              expected_sha, return_host, return_port, tx_key):
        from .multinode.rdma_transport import receive_file, RDMATransport

        survivor_path = self.rdma_directory / f"{sid}-survivor-{uuid.uuid4().hex}.bin"
        try:
            receive_stats = receive_file(
                port, survivor_path, device=self.rdma_device,
                gid_index=self.rdma_gid_index, port_num=self.rdma_port_num,
                timeout=self.rdma_timeout, return_stats=True,
            )
            survivor = survivor_path.read_bytes()
            with self.lock:
                s = self.stripes.get(sid)
                if s is None or s["state"] != "COMMITTED":
                    raise ValueError("committed stripe required")
                if len(survivor) != s["manifest"]["nodes"][1 - missing]["length"]:
                    raise ValueError("survivor length mismatch")
                if descriptor(survivor, allow_fixture=self.allow_fixture) != s["manifest"]["nodes"][1 - missing]:
                    raise ValueError("survivor corruption/identity mismatch")
                blocks = [None, None]
                blocks[1 - missing] = survivor
                from .perf_observer import span
                with span("rs_decode", "ec_compute", stripe_id=sid):
                    decode_started = time.perf_counter()
                    raw = self.rs_codec.recover_xor(blocks, s["parity"], [d["length"] for d in s["manifest"]["nodes"]])
                    decode_ms = (time.perf_counter() - decode_started) * 1000
                if len(raw) != expected_length or sha(raw) != expected_sha:
                    raise ValueError("recovered node integrity failure")
            tx = RDMATransport(
                device=self.rdma_device, gid_index=self.rdma_gid_index,
                port_num=self.rdma_port_num, timeout=self.rdma_timeout,
            )
            send_stats = tx.send_with_stats(return_host, return_port, raw)
            tx.close()
            with self.lock:
                s = self.stripes.get(sid)
                if s is not None:
                    s.setdefault("rdma", {}).setdefault(tx_key, {}).update(
                        state="COMPLETED", payload_bytes=len(raw),
                        payload_sha256=expected_sha,
                        rs_reconstruct_ms=decode_ms,
                        receive_stats=receive_stats, send_stats=send_stats,
                    )
                    self.metrics["rdma_rx_payload_bytes"] += len(survivor)
                    self.metrics["rdma_tx_payload_bytes"] += len(raw)
                    self.metrics["rdma_completion_count"] += 2
                    self.metrics["rdma_transfer_ms"] += (
                        receive_stats.get("elapsed_s", 0.0)
                        + send_stats.get("elapsed_s", 0.0)
                    ) * 1000
        except Exception as exc:
            with self.lock:
                s = self.stripes.get(sid)
                if s is not None:
                    s.setdefault("rdma", {}).setdefault(tx_key, {}).update(
                        state="FAILED", error=f"{type(exc).__name__}:{exc}"
                    )
        finally:
            survivor_path.unlink(missing_ok=True)
            with self.lock:
                self._release_rdma_port_locked(port)
                self.rdma_threads.pop((sid, "recovery", tx_key), None)

    def expire(self):
        for s in self.stripes.values():
            if s["state"] != "EXPIRED" and time.monotonic() >= s["deadline"]:
                s["state"] = "EXPIRED"
                s["blocks"].clear()
                s["parity"] = None
                s["streaming"] = None
                s.get("registrations", {}).clear()

    def status(self, s):
        result = {
            k: deepcopy(v)
            for k, v in s.items()
            if k not in ("blocks", "parity", "streaming", "deadline", "created")
        }
        result["registered_nodes"] = sorted(s.get("registrations", {}).keys())
        result["retained_node_block_bytes"] = sum(map(len, s.get("blocks", {}).values()))
        result["parity_bytes"] = len(s["parity"]) if s.get("parity") is not None else 0
        result["rdma_config"] = self._rdma_config()
        result["rs_backend"] = self.rs_codec.backend
        return result

    def usage(self):
        return sum(
            sum(map(len, s["blocks"].values()))
            + (len(s["parity"]) if s["parity"] is not None else 0)
            + (sum(map(len, s.get("streaming", {}).get("nodes", {}).values()))
               + len(s.get("streaming", {}).get("parity", ()))
               if s.get("streaming") else 0)
            for s in self.stripes.values()
        )

    def dispatch(self, request, payload):
        with self.lock:
            self.expire()
            op = request["op"]
            if op not in (
                "submit_node_block", "submit_node_chunk", "reconstruct_missing_node",
                "reconstruct_node_chunk",
            ) and payload:
                raise ValueError("unexpected payload")
            if op == "health":
                return {"state": "READY", "server_epoch": self.epoch,
                        "rdma_upload_sha256_field": RDMA_NODE_SHA256_FIELD}, b""
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
            if op == "register_node":
                node = request.get("node_id")
                candidate = request.get("descriptor")
                if type(node) is not int or node not in (0, 1):
                    raise ValueError("invalid node")
                if not isinstance(candidate, dict):
                    raise ValueError("node descriptor required")
                if candidate.get("metadata", {}).get("node_id") != node:
                    raise ValueError("descriptor/node identity mismatch")
                if (type(candidate.get("length")) is not int
                        or not 0 < candidate["length"] <= MAX_NODE_BYTES
                        or not isinstance(candidate.get("sha256"), str)):
                    raise ValueError("invalid node descriptor")
                existing = self.stripes.get(sid)
                if existing is None:
                    if len(self.stripes) >= self.max_stripes:
                        raise ValueError("stripe count budget exhausted; restart with fresh IDs")
                    existing = dict(
                        stripe_id=sid,
                        handle=None,
                        manifest_sha256=None,
                        manifest=None,
                        state="REGISTERING",
                        blocks={},
                        parity=None,
                        submitted_nodes=[],
                        registrations={},
                        rdma={},
                        streaming=None,
                        created=time.monotonic(),
                        deadline=time.monotonic() + self.ttl,
                        reserved_bytes=0,
                        rs_encode_ms=None,
                        commit_ms=None,
                    )
                    self.stripes[sid] = existing
                if existing["state"] == "EXPIRED":
                    raise ValueError("stripe ID expired/tombstoned")
                old = existing.setdefault("registrations", {}).get(node)
                if old is not None and old != candidate:
                    raise ValueError("node registration conflict")
                proposed = deepcopy(existing["registrations"])
                proposed[node] = deepcopy(candidate)
                if len(proposed) == 2 and existing["state"] == "REGISTERING":
                    nodes = [proposed[i] for i in (0, 1)]
                    manifest = {"format": "kv-rs-stripe-v1", "stripe_id": sid,
                                "code": CODE,
                                "nodes": nodes}
                    validate_manifest(manifest, allow_fixture=self.allow_fixture)
                    reserved = 3 * max(d["length"] for d in nodes)
                    used = sum(s["reserved_bytes"] for s in self.stripes.values()
                               if s["state"] != "EXPIRED")
                    if used + reserved > self.max_bytes:
                        raise ValueError("DRAM reservation budget exceeded")
                    existing.update(
                        handle=str(uuid.uuid4()), manifest_sha256=manifest_sha(manifest),
                        manifest=deepcopy(manifest), state="PENDING",
                        reserved_bytes=reserved,
                    )
                existing["registrations"] = proposed
                return self.status(existing), b""
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
                    rdma={},
                    streaming=None,
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
            # Registration is deliberately usable by either physical node before
            # a handle exists. Once both descriptors are bound, all mutating and
            # recovery operations require the immutable handle and manifest hash.
            if op == "get_stripe_status" and s["state"] == "REGISTERING":
                return self.status(s), b""
            if op == "get_stripe_status" and not request.get("handle"):
                # Read-only rendezvous polling is allowed without a handle so
                # the first physical node can observe the second registration.
                return self.status(s), b""
            if (
                request.get("handle") != s["handle"]
                or request.get("manifest_sha256") != s["manifest_sha256"]
            ):
                raise ValueError("stale handle or stripe manifest mismatch")
            if s["state"] == "EXPIRED":
                raise ValueError("stripe expired")
            if op == "get_stripe_status":
                return self.status(s), b""
            if op == "prepare_rdma":
                node = request.get("node_id")
                if type(node) is not int or node not in (0, 1):
                    raise ValueError("invalid node")
                expected = s["manifest"]["nodes"][node]
                length = request.get("payload_length")
                digest = request.get(RDMA_NODE_SHA256_FIELD)
                mismatches = []
                if type(length) is not int or length != expected["length"]:
                    mismatches.append("length")
                if not isinstance(digest, str) or digest != expected["sha256"]:
                    mismatches.append("sha256")
                rdma_diagnostic(
                    "server", stripe_id=sid, node_id=node,
                    received_length=length, received_sha256=digest,
                    expected_length=expected["length"], expected_sha256=expected["sha256"],
                    manifest_sha256=s["manifest_sha256"], mismatches=mismatches,
                )
                if mismatches:
                    raise ValueError(
                        "RDMA payload does not match node manifest: "
                        f"stripe_id={sid} node_id={node} mismatched={','.join(mismatches)}; "
                        f"RDMA SHA field={RDMA_NODE_SHA256_FIELD}"
                    )
                if s["state"] != "PENDING":
                    if node in s["submitted_nodes"]:
                        return self.status(s), b""
                    raise ValueError("stripe closed")
                old = s.setdefault("rdma", {}).get(node)
                if old is not None:
                    if (old.get("payload_length") != length
                            or old.get("payload_sha256") != digest):
                        raise ValueError("RDMA retry metadata conflict")
                    if old.get("state") in ("RECEIVING", "COMPLETED"):
                        return self.status(s), b""
                port = self._rdma_port_locked(sid, ("upload", node))
                s["rdma"][node] = {
                    "direction": "upload", "state": "RECEIVING", "port": port,
                    "payload_length": length, "payload_sha256": digest,
                    "transport": "rdma",
                }
                thread = threading.Thread(
                    target=self._rdma_receive_worker,
                    args=(sid, node, port, length, digest), daemon=True,
                )
                self.rdma_threads[(sid, "upload", node)] = thread
                thread.start()
                return self.status(s), b""
            if op == "submit_node_block":
                node = request["node_id"]
                if type(node) is not int or node not in (0, 1):
                    raise ValueError("invalid node")
                # Verify every retry's bytes against the immutable manifest,
                # including after the first commit released source blocks.
                if descriptor(payload, allow_fixture=self.allow_fixture) != s["manifest"]["nodes"][node]:
                    raise ValueError("node data/identity differs from manifest")
                # Exact retry is idempotent, even after original input release.
                if node in s["submitted_nodes"]:
                    return self.status(s), b""
                rdma_item = s.setdefault("rdma", {}).get(node)
                if rdma_item is not None and rdma_item.get("state") == "RECEIVING":
                    raise ValueError("RDMA upload is active; TCP payload fallback is forbidden")
                self._commit_block_locked(s, node, payload)
                return self.status(s), b""
            if op == "submit_node_chunk":
                node = request.get("node_id")
                offset = request.get("offset")
                self._submit_stream_chunk_locked(
                    s, node, offset, bytes(payload),
                    request.get("chunk_bytes", 4 * 1024**2),
                )
                return self.status(s), b""
            if s["state"] != "COMMITTED" or s["parity"] is None:
                raise ValueError("committed parity required")
            if sha(s["parity"]) != s["parity_sha256"]:
                raise ValueError("stored parity corruption")
            if op == "reconstruct_node_chunk":
                lost = request.get("missing_node")
                offset = request.get("offset")
                chunk_bytes = request.get("chunk_bytes")
                if (
                    lost not in (0, 1) or type(offset) is not int or offset < 0
                    or type(chunk_bytes) is not int
                    or not 1024**2 <= chunk_bytes <= 64 * 1024**2
                    or offset % chunk_bytes
                ):
                    raise ValueError("invalid streaming recovery chunk request")
                lengths = [d["length"] for d in s["manifest"]["nodes"]]
                lost_n = min(chunk_bytes, max(0, lengths[lost] - offset))
                survivor_n = min(chunk_bytes, max(0, lengths[1 - lost] - offset))
                width = min(chunk_bytes, max(lengths) - offset)
                if lost_n <= 0 or len(payload) != survivor_n or width <= 0:
                    raise ValueError("streaming recovery chunk geometry mismatch")
                survivor_chunk = bytes(payload) + bytes(width - len(payload))
                parity_chunk = s["parity"][offset:offset + width]
                started = time.perf_counter()
                raw = self.rs_codec.transform(parity_chunk, survivor_chunk, "xor")[:lost_n]
                return (
                    dict(
                        self.status(s), recovered_node=lost, offset=offset,
                        chunk_bytes=len(raw), chunk_sha256=sha(raw),
                        rs_reconstruct_ms=(time.perf_counter() - started) * 1000,
                        streaming_recovery=True,
                    ),
                    raw,
                )
            if op == "fetch_parity":
                if time.monotonic() >= s["deadline"]:
                    self.expire()
                    raise ValueError("stripe expired during parity verification")
                active_download = s.setdefault("rdma", {}).get("parity_download")
                if (request.get("transport", "tcp") == "tcp"
                        and active_download is not None
                        and active_download.get("state") == "SENDING"):
                    raise ValueError("RDMA parity transfer is active; TCP fallback is forbidden")
                if request.get("transport", "tcp") == "rdma":
                    host = request.get("rdma_host")
                    port = request.get("rdma_port")
                    if not isinstance(host, str) or not host or type(port) is not int or not 1 <= port <= 65535:
                        raise ValueError("RDMA return host/port required")
                    key = "parity_download"
                    old = s.setdefault("rdma", {}).get(key)
                    if old is None or old.get("state") == "COMPLETED":
                        s["rdma"][key] = {"direction": "download", "state": "SENDING",
                                           "payload_length": len(s["parity"]),
                                           "payload_sha256": s["parity_sha256"],
                                           "transport": "rdma", "remote_host": host,
                                           "remote_port": port}
                        thread = threading.Thread(
                            target=self._rdma_send_worker,
                            args=(sid, key, host, port, bytes(s["parity"]), s["parity_sha256"]),
                            daemon=True,
                        )
                        self.rdma_threads[(sid, "download", key)] = thread
                        thread.start()
                    elif old.get("state") == "FAILED":
                        raise ValueError(old.get("error", "RDMA parity transfer failed"))
                    return self.status(s), b""
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
                raw = self.rs_codec.recover_xor(
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
            if op == "prepare_rdma_recovery":
                missing = request.get("missing_nodes")
                if (not isinstance(missing, list) or len(missing) != 1
                        or type(missing[0]) is not int or missing[0] not in (0, 1)):
                    raise ValueError("exactly one missing node required")
                lost = missing[0]
                host, return_port = request.get("rdma_host"), request.get("rdma_return_port")
                length, digest = request.get("survivor_length"), request.get("survivor_sha256")
                survivor_expected = s["manifest"]["nodes"][1 - lost]
                recovered_expected = s["manifest"]["nodes"][lost]
                if (not isinstance(host, str) or not host or type(return_port) is not int
                        or not 1 <= return_port <= 65535 or type(length) is not int
                        or length != survivor_expected["length"]
                        or not isinstance(digest, str) or digest != survivor_expected["sha256"]):
                    raise ValueError("invalid RDMA recovery metadata")
                key = f"recovery_{lost}"
                old = s.setdefault("rdma", {}).get(key)
                if old is not None:
                    if old.get("state") == "FAILED":
                        raise ValueError(old.get("error", "RDMA recovery failed"))
                    return self.status(s), b""
                port = self._rdma_port_locked(sid, ("recovery", key))
                s["rdma"][key] = {
                    "direction": "bidirectional", "state": "RECEIVING_SURVIVOR",
                    "port": port, "return_host": host, "return_port": return_port,
                    "payload_length": length, "payload_sha256": digest,
                    "recovered_length": recovered_expected["length"],
                    "recovered_sha256": recovered_expected["sha256"],
                    "transport": "rdma", "missing_node": lost,
                }
                thread = threading.Thread(
                    target=self._rdma_recovery_worker,
                    args=(sid, lost, port, recovered_expected["length"],
                          recovered_expected["sha256"], host, return_port, key),
                    daemon=True,
                )
                self.rdma_threads[(sid, "recovery", key)] = thread
                thread.start()
                return self.status(s), b""
            raise ValueError("unsupported operation")


class RemoteRSServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    # Receive independent node chunks concurrently. Stripe mutation and codec
    # execution remain serialized by StripeStore.lock, while socket reads no
    # longer block the peer node's upload.
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = True

    def __init__(self, address, store=None, timeout=30, workers=4):
        self.store = store or StripeStore()
        self.io_timeout = timeout
        if not 1 <= int(workers) <= 32:
            raise ValueError("remote_rs_workers_out_of_range")
        self.worker_gate = threading.BoundedSemaphore(int(workers))
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.worker_gate.acquire(timeout=self.io_timeout):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.worker_gate.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.worker_gate.release()

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
                self.request, self.server.io_timeout, count("rx"),
                max_payload=store.max_bytes,
            )
            with store.lock:
                store.metrics["receive_ms"] += metrics["receive_ms"]
            result, payload = store.dispatch(req, raw)
            if req["op"] == "get_metrics":
                from .rs_experiment import network_observation

                result["network_B"] = network_observation(self.client_address[0])
            # RDMA state uses integer node IDs internally and string operation
            # names (for example, parity_download). JSON object keys must have
            # one comparable type while canonical() sorts them.
            rdma = result.get("rdma")
            if isinstance(rdma, dict):
                result["rdma"] = {str(key): value for key, value in rdma.items()}
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rdma-device", default="-",
                   help="libibverbs device for --transport rdma (e.g. mlx5_0)")
    p.add_argument("--rdma-gid-index", type=int, default=None)
    p.add_argument("--rdma-port-num", type=int, default=1)
    p.add_argument("--rdma-port-base", type=int, default=30400)
    p.add_argument("--rdma-timeout", type=float, default=120)
    p.add_argument("--rdma-directory", type=Path)
    p.add_argument(
        "--rs-backend", choices=("auto", "cuda", "cpu"), default="auto",
        help="auto (default) uses the fused Triton CUDA kernel when available",
    )
    args = p.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("positive finite timeout required")
    store = StripeStore(
        ttl=args.ttl, max_stripes=args.max_stripes, max_bytes=args.max_bytes,
        rdma_device=args.rdma_device, rdma_gid_index=args.rdma_gid_index,
        rdma_port_num=args.rdma_port_num, rdma_port_base=args.rdma_port_base,
        rdma_timeout=args.rdma_timeout, rdma_directory=args.rdma_directory,
        rs_backend=args.rs_backend,
    )
    with RemoteRSServer(
        (args.host, args.port), store, args.timeout, workers=args.workers
    ) as server:
        print(
            json.dumps(
                {
                    "state": "READY",
                    "address": server.server_address,
                    "epoch": store.epoch,
                    "rs_backend": store.rs_codec.backend,
                    "rdma_upload_sha256_field": RDMA_NODE_SHA256_FIELD,
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
