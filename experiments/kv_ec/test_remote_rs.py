"""CPU fixtures are unit evidence only; real TCP tests skip if bind is denied."""

from copy import deepcopy
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from experiments.kv_ec.node_block_export import pack, unpack, descriptor, sha, canonical
from experiments.kv_ec.remote_rs_client import RemoteRSClient, binding
from experiments.kv_ec.remote_rs_protocol import send, receive, MAX_HEADER
from experiments.kv_ec.remote_rs_server import StripeStore, RemoteRSServer, Handler
from experiments.kv_ec.rs_codec import encode, recover, gf_mul
from experiments.kv_ec.rs_experiment import save_report
from experiments.kv_ec.rs_fault_recovery import verify_faults
from experiments.kv_ec.rs_stripe_builder import build
from experiments.kv_ec.test_parity_manager import metadata_and_blocks


def node_fixture(node):
    arrays, ranks = metadata_and_blocks(tp=2, count=4)
    for m in ranks:
        m.update(
            snapshot_id=f"snapshot-{node}",
            request_id=f"request-{node}",
            request_version=f"request-version-{node}",
            checkpoint_version=f"checkpoint-{node}",
            manager_epoch=f"manager-{node}",
        )
    return pack(node, [b.tobytes() for b in arrays], ranks, allow_fixture=True)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [node_fixture(i) for i in (0, 1)]
        self.manifest = build("stripe", self.blocks, allow_fixture=True)
        self.store = StripeStore(allow_fixture=True)
        self.created, _ = self.store.dispatch(
            dict(op="create_stripe", stripe_id="stripe", manifest=self.manifest), b""
        )
        self.key = binding(self.created)

    def call(self, op, raw=b"", **fields):
        return self.store.dispatch(dict(op=op, **self.key, **fields), raw)

    def commit(self):
        for node in (0, 1):
            status, _ = self.call("submit_node_block", self.blocks[node], node_id=node)
        return status

    def test_GF_matrix_padding_and_two_erasures(self):
        self.assertEqual(gf_mul(2, 128), 29)
        for a in range(256):
            self.assertEqual(gf_mul(a, 1), a)
            self.assertEqual(gf_mul(a, 0), 0)
        a, b = b"abc", b"1234567"
        parity = encode(a, b)
        self.assertEqual(recover([None, b], parity, [3, 7]), a)
        self.assertEqual(recover([a, None], parity, [3, 7]), b)
        for blocks, p in (([None, None], parity), ([None, b], None), ([a, b], parity)):
            with self.assertRaises(ValueError):
                recover(blocks, p, [3, 7])

    def test_pack_roundtrip_cpu_ownership_and_real_gate(self):
        meta, ranks = unpack(self.blocks[0], allow_fixture=True)
        self.assertEqual(
            pack(0, ranks, meta["ranks"], allow_fixture=True), self.blocks[0]
        )
        with self.assertRaises(ValueError):
            unpack(self.blocks[0])
        bad = bytearray(self.blocks[0])
        bad[-1] ^= 1
        with self.assertRaises(ValueError):
            unpack(bytes(bad), allow_fixture=True)
        for field in (
            "request_version",
            "checkpoint_version",
            "manager_epoch",
            "token_start",
            "request_id",
        ):
            changed = deepcopy(meta["ranks"])
            changed[1][field] = "stale"
            with self.subTest(field=field), self.assertRaises(ValueError):
                pack(0, ranks, changed, allow_fixture=True)

    def test_uncommitted_and_missing_parity_refused(self):
        for op in ("fetch_parity", "reconstruct_missing_node"):
            with self.assertRaises(ValueError):
                self.call(op, missing_nodes=[0])
        self.commit()
        self.store.stripes["stripe"]["parity"] = None
        with self.assertRaises(ValueError):
            self.call("reconstruct_missing_node", self.blocks[1], missing_nodes=[0])

    def test_commit_release_exact_recovery_and_idempotence(self):
        status = self.commit()
        self.assertEqual(status["state"], "COMMITTED")
        self.assertEqual(self.store.stripes["stripe"]["blocks"], {})
        self.assertEqual(self.store.usage(), max(map(len, self.blocks)))
        for lost in (0, 1):
            result, raw = self.call(
                "reconstruct_missing_node", self.blocks[1 - lost], missing_nodes=[lost]
            )
            self.assertEqual(raw, self.blocks[lost])
            self.assertEqual(result["recovered_node"], lost)
        old_parity = self.store.stripes["stripe"]["parity"]
        self.assertEqual(
            self.call("submit_node_block", self.blocks[0], node_id=0)[0], status
        )
        self.assertIs(self.store.stripes["stripe"]["parity"], old_parity)
        with self.assertRaises(ValueError):
            self.call("reconstruct_missing_node", missing_nodes=[0, 1])

    def test_payload_and_version_conflicts_never_pollute_pending(self):
        self.call("submit_node_block", self.blocks[0], node_id=0)
        for raw in (self.blocks[0], self.blocks[1][:-1], self.blocks[1][:-1] + b"x"):
            with self.assertRaises(ValueError):
                self.call("submit_node_block", raw, node_id=1)
            self.assertEqual(self.store.stripes["stripe"]["submitted_nodes"], [0])
        meta, ranks = unpack(self.blocks[1], allow_fixture=True)
        for m in meta["ranks"]:
            m["checkpoint_version"] = "another-version"
        stale = pack(1, ranks, meta["ranks"], allow_fixture=True)
        with self.assertRaises(ValueError):
            self.call("submit_node_block", stale, node_id=1)
        with self.assertRaises(ValueError):
            self.store.dispatch(
                dict(op="get_stripe_status", **dict(self.key, handle="old-server")), b""
            )
        self.assertEqual(self.commit()["state"], "COMMITTED")

    def test_corrupt_parity_and_survivor_detected(self):
        self.commit()
        with self.assertRaises(ValueError):
            self.call(
                "reconstruct_missing_node",
                self.blocks[1][:-1] + b"!",
                missing_nodes=[0],
            )
        s = self.store.stripes["stripe"]
        s["parity"] = b"!" + s["parity"][1:]
        for op in ("fetch_parity", "reconstruct_missing_node"):
            with self.assertRaises(ValueError):
                self.call(
                    op,
                    self.blocks[1] if op == "reconstruct_missing_node" else b"",
                    missing_nodes=[0],
                )

    def test_expiry_releases_and_tombstones(self):
        self.call("submit_node_block", self.blocks[0], node_id=0)
        self.store.stripes["stripe"]["deadline"] = time.monotonic() - 1
        with self.assertRaises(ValueError):
            self.call("submit_node_block", self.blocks[1], node_id=1)
        self.assertEqual(self.store.usage(), 0)
        self.assertEqual(self.store.stripes["stripe"]["state"], "EXPIRED")
        with self.assertRaises(ValueError):
            self.store.dispatch(
                dict(op="create_stripe", stripe_id="stripe", manifest=self.manifest),
                b"",
            )

    def test_manifest_mixed_chunk_request_and_reservation(self):
        for field in ("token_start", "model_sha256", "token_count"):
            bad = deepcopy(self.manifest)
            bad["nodes"][1]["metadata"]["ranks"][0][field] = "wrong"
            with self.subTest(field=field), self.assertRaises(ValueError):
                StripeStore(allow_fixture=True).dispatch(
                    dict(op="create_stripe", stripe_id="stripe", manifest=bad), b""
                )
        with self.assertRaises(ValueError):
            StripeStore(max_bytes=1, allow_fixture=True).dispatch(
                dict(op="create_stripe", stripe_id="stripe", manifest=self.manifest),
                b"",
            )

    def test_reports_never_overwrite(self):
        with tempfile.TemporaryDirectory() as path:
            prefix = Path(path) / "report"
            save_report(prefix, {"ok": True, "n": 3})
            old = prefix.with_suffix(".json").read_bytes()
            with self.assertRaises(FileExistsError):
                save_report(prefix, {"ok": False})
            self.assertEqual(prefix.with_suffix(".json").read_bytes(), old)
            self.assertIn("metric,value_json", prefix.with_suffix(".csv").read_text())


class MemorySocket:
    def __init__(self, data=b""):
        self.data = data

    def sendall(self, data):
        self.data += data

    def send(self, data):
        size = min(len(data), 11)
        self.data += bytes(data[:size])
        return size

    def gettimeout(self):
        return 5

    def recv(self, n):
        raw, self.data = self.data[: min(n, 7)], self.data[min(n, 7) :]
        return raw

    def settimeout(self, n):
        pass


class ProtocolTests(unittest.TestCase):
    def test_fragmentation_counters_and_checksum(self):
        sock = MemorySocket()
        tx = send(sock, {"op": "health"}, b"actual-payload")
        self.assertEqual(tx["stream_bytes"], len(sock.data))
        meta, raw, rx = receive(MemorySocket(sock.data), 5)
        self.assertEqual(raw, b"actual-payload")
        self.assertEqual(rx["stream_bytes"], tx["stream_bytes"])
        for bad in (
            sock.data[:-1],
            sock.data[:-1] + b"!",
            struct.pack("!I", MAX_HEADER + 1),
        ):
            with self.assertRaises((ValueError, ConnectionError)):
                receive(MemorySocket(bad), 5)
        with patch(
            "experiments.kv_ec.remote_rs_protocol.time.monotonic", side_effect=[0, 10]
        ):
            with self.assertRaises(TimeoutError):
                receive(MemorySocket(sock.data), 1)

    def test_handler_end_to_end_without_kernel_socket_is_explicit_mock(self):
        from types import SimpleNamespace

        store = StripeStore(allow_fixture=True)
        server = SimpleNamespace(store=store, io_timeout=5)

        def rpc(op, payload=b"", **fields):
            sock = MemorySocket()
            send(sock, dict(op=op, **fields), payload)
            request_bytes = len(sock.data)
            Handler(sock, ("mock", 0), server)
            result, data, _ = receive(sock, 5)
            self.assertGreaterEqual(store.metrics["rx_stream_bytes"], request_bytes)
            if not result["ok"]:
                raise ValueError(result["error"])
            return result, data

        client = SimpleNamespace(call=rpc)
        blocks = [node_fixture(i) for i in (0, 1)]
        manifest = build("mock-wire", blocks, allow_fixture=True)
        status, _ = rpc("create_stripe", stripe_id="mock-wire", manifest=manifest)
        for node in (0, 1):
            status, _ = rpc(
                "submit_node_block", blocks[node], node_id=node, **binding(status)
            )
        self.assertEqual(status["state"], "COMMITTED")
        self.assertEqual(
            len(verify_faults(client, status, blocks, allow_fixture=True)), 8
        )
        # Malformed/incomplete frame contributes received bytes, cannot mutate
        # store state, and produces a structured negative response.
        bad = MemorySocket(struct.pack("!I", 100) + b"{")
        before = store.metrics["rx_stream_bytes"]
        Handler(bad, ("mock", 0), server)
        error, _, _ = receive(bad, 5)
        self.assertFalse(error["ok"])
        self.assertEqual(store.metrics["rx_stream_bytes"] - before, 5)
        self.assertEqual(len(store.stripes), 1)


class TCPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = StripeStore(allow_fixture=True)
        try:
            self.server = RemoteRSServer(("127.0.0.1", 0), self.store, timeout=0.3)
        except PermissionError as exc:
            self.skipTest(f"NOT VERIFIED: kernel TCP bind forbidden: {exc}")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = RemoteRSClient(*self.server.server_address, timeout=5)

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(2)

    def test_real_tcp_end_to_end_fixtures(self):
        blocks = [node_fixture(i) for i in (0, 1)]
        m = build("tcp", blocks, allow_fixture=True)
        s, _ = self.client.call("create_stripe", stripe_id="tcp", manifest=m)
        with self.assertRaises(ValueError):
            self.client.call("fetch_parity", **binding(s))
        for node in (0, 1):
            s, _ = self.client.call(
                "submit_node_block", node_id=node, payload=blocks[node], **binding(s)
            )
        self.assertEqual(s["state"], "COMMITTED")
        checks = verify_faults(self.client, s, blocks, allow_fixture=True)
        self.assertEqual(len(checks), 8)
        metrics, _ = self.client.call("get_metrics")
        self.assertGreater(metrics["rx_payload_bytes"], sum(map(len, blocks)))
        self.assertEqual(metrics["resident_data_bytes"], max(map(len, blocks)))

    def test_real_tcp_truncation_timeout_and_retry(self):
        # Partial frame must never create a stripe; server resumes after timeout.
        with socket.create_connection(self.server.server_address, timeout=1) as sock:
            sock.sendall(struct.pack("!I", 200) + b"{")
            time.sleep(0.4)
        health, _ = self.client.call("health")
        self.assertEqual(health["state"], "READY")
        self.assertEqual(self.store.stripes, {})


if __name__ == "__main__":
    unittest.main()
