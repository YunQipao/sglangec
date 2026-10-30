"""Real client/framing/Handler regression; only sockets and verbs are mocked."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr
from copy import deepcopy
import io
import json
import os
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.kv_ec.node_block_export import descriptor, pack, sha, unpack
from experiments.kv_ec.remote_rs_client import RemoteRSClient, binding
from experiments.kv_ec.remote_rs_protocol import receive, send
from experiments.kv_ec.remote_rs_server import Handler, StripeStore
from experiments.kv_ec.test_remote_rs import MemorySocket
from experiments.kv_ec.test_tp4_physical_rs import tp4_block
from experiments.kv_ec.multinode import rdma_transport


class RPCBuffer(MemorySocket):
    """Fragment large TP4 manifests without byte-at-a-time quadratic copies."""

    def send(self, data):
        size = min(len(data), 4096)
        self.data += bytes(data[:size])
        return size

    def recv(self, n):
        size = min(n, 4093)
        raw, self.data = self.data[:size], self.data[size:]
        return raw


class HandlerSocket:
    """One real framed RPC through Handler, without a kernel socket."""

    def __init__(self, server):
        self.server = server
        self.buffer = RPCBuffer()
        self.responded = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def getpeername(self):
        return ("cpu-mock", 30300)

    def getsockname(self):
        return ("cpu-mock", 40000)

    def gettimeout(self):
        return self.buffer.gettimeout()

    def settimeout(self, value):
        self.buffer.settimeout(value)

    def send(self, raw):
        return self.buffer.send(raw)

    def recv(self, count):
        if not self.responded:
            self.responded = True
            Handler(self.buffer, ("cpu-mock", 40000), self.server)
        return self.buffer.recv(count)


@contextmanager
def mock_rdma_uploads(store, before_send=None):
    """Replace native bytes movement only; retain prepare and commit validation."""
    def send(host, port, block):
        with store.lock:
            sid, (_, node) = store.rdma_ports[port]
        if before_send:
            before_send(sid, node, block)
        with store.lock:
            stripe = store.stripes[sid]
            store._commit_block_locked(stripe, node, block)
            stripe["rdma"][node].update(state="COMPLETED", stats={"transport": "CPU_MOCK"})
            store._release_rdma_port_locked(port)
        return {"transport": "CPU_MOCK"}

    tx = SimpleNamespace(send_with_stats=send, close=lambda: None)
    with patch.object(store, "_rdma_receive_worker"), patch.object(
        rdma_transport, "RDMATransport", return_value=tx
    ):
        yield


class RDMAUploadControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = StripeStore(allow_fixture=True, rdma_directory=self.temp.name)
        server = SimpleNamespace(store=self.store, io_timeout=5)
        connection = patch(
            "experiments.kv_ec.remote_rs_client.socket.create_connection",
            side_effect=lambda *args, **kwargs: HandlerSocket(server),
        )
        connection.start()
        self.addCleanup(connection.stop)
        self.client = RemoteRSClient("cpu-mock", timeout=5)

    def register(self, sid):
        blocks = []
        for node in (0, 1):
            meta, ranks = unpack(tp4_block(node), allow_fixture=True)
            for rank in meta["ranks"]:
                rank.update(snapshot_id=f"{sid}-snapshot-{node}", request_id=f"{sid}-request-{node}")
            block = pack(node, ranks, meta["ranks"], allow_fixture=True)
            blocks.append(block)
            status, _ = self.client.call(
                "register_node", stripe_id=sid, node_id=node,
                descriptor=descriptor(block, allow_fixture=True),
            )
        return blocks, binding(status)

    def test_real_framing_preserves_upload_sha256_for_both_nodes(self):
        blocks, key = self.register("framed-upload")
        # This idempotent state avoids native verbs while still executing the
        # server's strict prepare_rdma length/SHA check before it returns.
        self.store.stripes[key["stripe_id"]]["rdma"] = {
            node: dict(state="COMPLETED", payload_length=len(block),
                       payload_sha256=sha(block), stats={"transport": "CPU_MOCK"})
            for node, block in enumerate(blocks)
        }
        for node, block in enumerate(blocks):
            with self.subTest(node=node):
                status, stats, disposition = self.client.rdma_submit_node_block(
                    key["stripe_id"], node, block,
                    handle=key["handle"], manifest_sha256=key["manifest_sha256"],
                    device="cpu-mock", gid_index=0,
                )
                self.assertEqual(disposition, "idempotent")
                self.assertEqual(stats["transport"], "CPU_MOCK")
                self.assertEqual(status["manifest"]["nodes"][node],
                                 descriptor(block, allow_fixture=True))

    def test_upload_through_real_control_frames_and_mock_verbs_commits(self):
        blocks, key = self.register("framed-upload-complete")
        with mock_rdma_uploads(self.store):
            for node, block in enumerate(blocks):
                status, _, disposition = self.client.rdma_submit_node_block(
                    key["stripe_id"], node, block, handle=key["handle"],
                    manifest_sha256=key["manifest_sha256"], device="cpu-mock", gid_index=0,
                )
                self.assertEqual(disposition, "completed")
        self.assertEqual(status["state"], "COMMITTED")

    def test_rdma_control_status_keys_serialize_on_tcp_boundary(self):
        blocks, key = self.register("rdma-control-json-keys")
        with mock_rdma_uploads(self.store):
            for node, block in enumerate(blocks):
                self.client.rdma_submit_node_block(
                    key["stripe_id"], node, block, handle=key["handle"],
                    manifest_sha256=key["manifest_sha256"], device="cpu-mock", gid_index=0,
                )

        stripe = self.store.stripes[key["stripe_id"]]
        self.assertEqual(set(stripe["rdma"]), {0, 1})
        # Suppress only the asynchronous verbs worker. The dispatch and framed
        # TCP response below execute the production control path.
        class UnstartedThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        with patch("experiments.kv_ec.remote_rs_server.threading.Thread", UnstartedThread):
            status, _ = self.client.call(
                "fetch_parity", transport="rdma", rdma_host="cpu-mock",
                rdma_port=30490, **key,
            )
            self.assertEqual(status["state"], "COMMITTED")
            self.assertEqual(RemoteRSClient._rdma_item(status, 0)["state"], "COMPLETED")
            self.assertEqual(RemoteRSClient._rdma_item(status, 1)["state"], "COMPLETED")
            self.assertEqual(RemoteRSClient._rdma_item(status, "parity_download")["state"], "SENDING")

            for missing in (0, 1):
                recovery, _ = self.client.call(
                    "prepare_rdma_recovery", missing_nodes=[missing],
                    transport="rdma", rdma_host="cpu-mock", rdma_return_port=30491,
                    survivor_length=len(blocks[1 - missing]),
                    survivor_sha256=sha(blocks[1 - missing]), **key,
                )
                recovery_key = f"recovery_{missing}"
                self.assertEqual(
                    RemoteRSClient._rdma_item(recovery, recovery_key)["state"],
                    "RECEIVING_SURVIVOR",
                )

        # Exercise the exact canonical TCP frame encoder/decoder on the full
        # response returned by fetch_parity, including all three RDMA entries.
        frame = MemorySocket()
        send(frame, status)
        decoded, _, _ = receive(MemorySocket(frame.data), 5)
        for key_id in (0, 1, "parity_download"):
            self.assertIn(str(key_id), decoded["rdma"])
        self.assertEqual(RemoteRSClient._rdma_item(decoded, 0)["state"], "COMPLETED")
        self.assertEqual(RemoteRSClient._rdma_item(decoded, 1)["state"], "COMPLETED")
        self.assertEqual(RemoteRSClient._rdma_item(decoded, "parity_download")["state"], "SENDING")

    def test_mismatch_is_rejected_before_receiver_without_changing_manifest(self):
        blocks, key = self.register("negative-upload")
        manifest = deepcopy(self.store.stripes[key["stripe_id"]]["manifest"])
        cases = (
            (len(blocks[0]) + 1, sha(blocks[0]), "length"),
            (len(blocks[0]), "f" * 64, "sha256"),
            (len(blocks[0]) + 1, "f" * 64, "length,sha256"),
            (len(blocks[0]), sha(blocks[1]), "sha256"),
        )
        with patch.object(self.store, "_rdma_receive_worker") as receiver:
            for length, digest, reason in cases:
                with self.subTest(reason=reason), self.assertRaisesRegex(
                    ValueError, f"mismatched={reason}"
                ):
                    self.client.call("prepare_rdma", node_id=0, payload_length=length,
                                     node_block_sha256=digest, **key)
            receiver.assert_not_called()
        stripe = self.store.stripes[key["stripe_id"]]
        self.assertEqual(stripe["manifest"], manifest)
        self.assertEqual(stripe["state"], "PENDING")
        self.assertEqual(stripe["rdma"], {})

    def test_legacy_frame_sha_is_never_accepted_as_node_sha(self):
        blocks, key = self.register("legacy-sha-field")
        with patch.object(self.store, "dispatch", wraps=self.store.dispatch) as dispatch:
            with self.assertRaisesRegex(ValueError, "mismatched=sha256"):
                self.client.call("prepare_rdma", node_id=0, payload_length=len(blocks[0]),
                                 payload_sha256=sha(blocks[0]), **key)
        request, payload = dispatch.call_args.args
        self.assertEqual(payload, b"")
        self.assertEqual(request["payload_length"], len(blocks[0]))
        self.assertEqual(request["payload_sha256"], sha(b""))
        self.assertNotEqual(request["payload_sha256"], sha(blocks[0]))

    def test_three_stripes_with_six_concurrent_uploads_do_not_mix_fields(self):
        pairs = [self.register(f"parallel-chunk-{i}") for i in range(3)]
        gate = threading.Barrier(6)
        with mock_rdma_uploads(self.store, lambda *args: gate.wait(5)):
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [
                    pool.submit(self.client.rdma_submit_node_block, key["stripe_id"], node,
                                block, handle=key["handle"], manifest_sha256=key["manifest_sha256"],
                                device="cpu-mock", gid_index=0)
                    for blocks, key in pairs for node, block in enumerate(blocks)
                ]
                for future in futures:
                    future.result(timeout=10)
        for blocks, key in pairs:
            stripe = self.store.stripes[key["stripe_id"]]
            self.assertEqual(stripe["state"], "COMMITTED")
            for node, block in enumerate(blocks):
                self.assertEqual(stripe["rdma"][node]["payload_sha256"], sha(block))
                self.assertEqual(stripe["manifest"]["nodes"][node],
                                 descriptor(block, allow_fixture=True))

    def test_foreign_stripe_binding_rejected_before_receiver(self):
        blocks, key = self.register("binding-first")
        _, other = self.register("binding-other")
        with patch.object(self.store, "_rdma_receive_worker") as receiver:
            with self.assertRaisesRegex(ValueError, "stale handle"):
                self.client.call("prepare_rdma", node_id=0, payload_length=len(blocks[0]),
                                 node_block_sha256=sha(blocks[0]), **dict(key, stripe_id=other["stripe_id"]))
            receiver.assert_not_called()

    def test_diagnostics_are_opt_in_metadata_only_and_name_mismatch(self):
        blocks, key = self.register("diagnostics")
        for enabled in ("0", "1"):
            output = io.StringIO()
            with patch.dict(os.environ, KV_EC_RS_RDMA_DIAGNOSTICS=enabled), redirect_stderr(output):
                with self.assertRaisesRegex(ValueError, "mismatched=sha256"):
                    self.client.rdma_submit_node_block(
                        key["stripe_id"], 0, blocks[1], handle=key["handle"],
                        manifest_sha256=key["manifest_sha256"], device="cpu-mock", gid_index=0,
                    )
            if enabled == "0":
                self.assertEqual(output.getvalue(), "")
                continue
            records = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()]
            self.assertEqual([r["side"] for r in records], ["client", "server"])
            client, server = records
            self.assertEqual(client, dict(side="client", stripe_id=key["stripe_id"], node_id=0,
                                         block_length=len(blocks[1]), block_sha256=sha(blocks[1]),
                                         manifest_sha256=key["manifest_sha256"]))
            self.assertEqual(server, dict(side="server", stripe_id=key["stripe_id"], node_id=0,
                                         received_length=len(blocks[1]), received_sha256=sha(blocks[1]),
                                         expected_length=len(blocks[0]), expected_sha256=sha(blocks[0]),
                                         manifest_sha256=key["manifest_sha256"], mismatches=["sha256"]))

    def test_health_advertises_loaded_upload_contract(self):
        status, _ = self.client.call("health")
        self.assertEqual(status["rdma_upload_sha256_field"], "node_block_sha256")


if __name__ == "__main__":
    unittest.main()
