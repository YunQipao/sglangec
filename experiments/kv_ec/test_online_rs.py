"""Socket/GPU-free integration using the actual StripeStore and RS codec.

Only the transport is mocked. These tests never constitute native RDMA or GPU
evidence; production configuration cannot enable the fixture/mock boundary.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .node_block_export import descriptor, pack, sha, unpack
from .online_chunk_manager import OnlineChunkManager
from .online_rs_exporter import OnlineRSExporter
from .online_rs_identity import identity_for, stripe_id
from .remote_rs_server import StripeStore
from .remote_rs_client import RemoteRSClient
from .rs_stripe_builder import build, validate_manifest
from .test_debug_capture import rank_fixture
from .test_remote_rs import node_fixture
from .test_rdma_upload_control import HandlerSocket, mock_rdma_uploads
from .verify_online_rs import committed, verify_blocks, verify_local


def fixture(node, tp=4, chunk=0):
    blocks, entries = [], []
    request = sha(f"independent-request-{node}".encode())
    run = f"local{node}"
    checkpoint = sha(f"{run}:{request}:{chunk}".encode())
    for rank in range(tp):
        _, meta = rank_fixture(rank, tp, layers=(0,), count=1, heads=1)
        parts, records = [], []
        hashes = {k: hashlib.sha256() for k in ("k", "v")}
        offset = 0
        for layer in range(36):
            record = dict(
                layer_id=layer,
                dtype="torch.bfloat16",
                packed_shape={k: [256, 1, 1] for k in ("k", "v")},
            )
            for kind in ("k", "v"):
                raw = (
                    bytes([(node * 53 + rank * 17 + layer + (kind == "v")) % 256]) * 512
                )
                record.update(
                    {
                        kind + "_offset": offset,
                        kind + "_nbytes": len(raw),
                        kind + "_sha256": sha(raw),
                    }
                )
                offset += len(raw)
                parts.append(raw)
                hashes[kind].update(raw)
            records.append(record)
        raw = b"".join(parts)
        meta.update(
            request_id=request,
            snapshot_id=checkpoint,
            request_version=run,
            manager_epoch=run,
            checkpoint_version=checkpoint,
            chunk_id=chunk,
            token_range=[chunk * 256, (chunk + 1) * 256],
            token_start=chunk * 256,
            token_count=256,
            physical_slots=list(range(1, 257)),
            layer_range=[0, 36],
            layers=list(range(36)),
            layer_records=records,
            packed_byte_count=len(raw),
            model_version=meta["model_sha256"],
            sha256=sha(raw),
            k_sha256=hashes["k"].hexdigest(),
            v_sha256=hashes["v"].hexdigest(),
            token_ids_sha256=sha(f"different tokens {node}".encode()),
            performance_ms={"gpu_to_host_ms": 1, "queue_wait_ms": 2},
            # Distinct metadata lengths exercise serialization-length padding,
            # even though corresponding rank KV layouts are identical.
            node_debug_label="node0" if node == 0 else "independent-node1-long-label",
        )
        blocks.append(np.frombuffer(raw, dtype=np.uint8).copy())
        entries.append(meta)
    return blocks, entries


class MockRDMAClient:
    """Real store API, CPU payload injection in place of native verbs only."""

    def __init__(self, store, gate=None, entered=None, fail=False):
        self.store, self.gate, self.entered, self.fail = store, gate, entered, fail
        self.timeout = 1

    def call(self, op, *, payload=b"", **fields):
        return self.store.dispatch(dict(op=op, **fields), payload)

    def rdma_submit_node_block(
        self, stripe_id, node_id, block, *, handle, manifest_sha256, timeout, **rdma
    ):
        if self.entered:
            self.entered.set()
        if self.gate and not self.gate.wait(2):
            raise TimeoutError("mock upload gate")
        if self.fail:
            raise OSError("mock RDMA unavailable; no TCP fallback")
        status, _ = self.call(
            "submit_node_block",
            stripe_id=stripe_id,
            node_id=node_id,
            payload=block,
            handle=handle,
            manifest_sha256=manifest_sha256,
        )
        return status, {"transport": "CPU_MOCK", "completion_ms": 0.25}, "completed"


class OnlineRSTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = StripeStore(allow_fixture=True, rdma_directory=self.root / "rdma")

    def exporter(self, node, **options):
        blocks, entries = fixture(node)
        defaults = dict(
            node_id=node,
            run_id="global-run",
            checkpoint_version="global-epoch-1",
            requests={entries[0]["request_id"]: "explicit-pair-A"},
            remote_host="cpu-mock-only",
            timeout=3,
            poll_interval=0.005,
            max_pending=4,
            workers=2,
            client_factory=lambda: MockRDMAClient(self.store),
            allow_fixture=True,
        )
        defaults.update(options)
        exporter = OnlineRSExporter(**defaults)
        self.addCleanup(exporter.close)
        return exporter

    def manager(self, node, exporter, **options):
        return OnlineChunkManager(
            self.root / f"node{node}",
            f"local{node}",
            global_exporter=exporter,
            test_recovery=True,
            **options,
        )

    def submit(self, manager, node, tp=4, chunk=0, changes=None):
        blocks, entries = fixture(node, tp, chunk)
        for block, meta in zip(blocks, entries):
            if changes:
                meta.update(changes)
            result = manager.submit(meta, block)
        self.assertEqual(result["state"], "COMMITTED")
        directory = manager.output / entries[0]["snapshot_id"]
        return directory, entries, blocks

    def wait(self, directory, states=("GLOBAL_COMMITTED",), timeout=4):
        deadline = time.monotonic() + timeout
        path = directory / "global_status.json"
        while time.monotonic() < deadline:
            if path.exists():
                receipt = json.loads(path.read_text())
                if receipt["global_state"] in states:
                    return receipt
                if receipt["global_state"] in ("FAILED", "EXPIRED", "CANCELLED"):
                    self.fail(str(receipt))
            threading.Event().wait(0.005)
        self.fail(
            f"status timeout: {path.read_text() if path.exists() else 'no receipt'}"
        )

    def test_tp2_tp4_three_chunks_with_local_model_paths_and_all_recoveries(self):
        model_paths = (
            "/media/sda1/iic/luotao/models/Qwen3-8B",
            "/media/sda1/iic_data/luotao/models/Qwen3-8B",
        )
        for tp in (2, 4):
            with self.subTest(tp=tp):
                # Fresh managers/store and global epoch for each TP layout.
                self.store = StripeStore(
                    allow_fixture=True, rdma_directory=self.root / "rdma"
                )
                exporters = [self.exporter(i) for i in (0, 1)]
                managers = [
                    OnlineChunkManager(
                        self.root / f"tp{tp}-node{i}",
                        f"local{i}",
                        global_exporter=exporters[i],
                        test_recovery=True,
                    )
                    for i in (0, 1)
                ]
                pairs = []
                for chunk in range(3):
                    pairs.append(
                        [self.submit(managers[i], i, tp, chunk,
                                     changes={"model_identifier": model_paths[i]})[0]
                         for i in (0, 1)]
                    )
                for directories in pairs:
                    receipts = [self.wait(d) for d in directories]
                    self.assertEqual(receipts[0]["stripe_id"], receipts[1]["stripe_id"])
                    blocks = [(d / "global_node.bin").read_bytes() for d in directories]
                    self.assertNotEqual(len(blocks[0]), len(blocks[1]))
                    key = {
                        k: receipts[0][k]
                        for k in ("stripe_id", "handle", "manifest_sha256")
                    }
                    status, parity = self.store.dispatch(
                        dict(op="fetch_parity", **key), b""
                    )
                    for node, block in enumerate(blocks):
                        self.assertEqual(status["manifest"]["nodes"][node],
                                         descriptor(block, allow_fixture=True))
                        self.assertTrue(all(
                            r["model_identifier"] == model_paths[node]
                            for r in status["manifest"]["nodes"][node]["metadata"]["ranks"]
                        ))
                    checks = verify_blocks(
                        blocks,
                        parity,
                        status["manifest"],
                        status["parity_sha256"],
                        allow_fixture=True,
                    )
                    self.assertEqual(set(checks), {"D0", "D1", "P"})
                    for missing in (0, 1):
                        _, restored = self.store.dispatch(
                            dict(
                                op="reconstruct_missing_node",
                                **key,
                                missing_nodes=[missing],
                            ),
                            blocks[1 - missing],
                        )
                        self.assertEqual(restored, blocks[missing])
                    for d, r, b in zip(directories, receipts, blocks):
                        verify_local(d, r, b, allow_fixture=True)
                        self.assertEqual(r["local_state"], "LOCAL_COMMITTED")
                        self.assertEqual(r["remote_outcome"], "COMMITTED")
                        for metric in (
                            "queue_wait_ms",
                            "pack_validate_ms",
                            "register_ms",
                            "peer_wait_ms",
                            "rdma_native",
                            "rdma_upload_wall_ms",
                            "server_rs_encode_ms",
                            "global_commit_wait_ms",
                        ):
                            self.assertIn(metric, r["metrics"])
                for m in managers:
                    self.assertEqual(len(committed(m.output, 3, 1)), 3)
                for e in exporters:
                    e.close()

    def test_exporter_sends_exact_persisted_registered_bytes_through_real_framing(self):
        server = SimpleNamespace(store=self.store, io_timeout=5)
        sent = []

        def check_upload(sid, node, block):
            meta, _ = unpack(block, allow_fixture=True)
            directory = self.root / f"node{node}" / meta["ranks"][0]["snapshot_id"]
            self.assertEqual((directory / "global_node.bin").read_bytes(), block)
            with self.store.lock:
                self.assertEqual(self.store.stripes[sid]["manifest"]["nodes"][node],
                                 descriptor(block, allow_fixture=True))
                sent.append((sid, node))

        with patch("experiments.kv_ec.remote_rs_client.socket.create_connection",
                   side_effect=lambda *args, **kwargs: HandlerSocket(server)), \
                mock_rdma_uploads(self.store, check_upload):
            exporters = [self.exporter(i, timeout=10, client_factory=lambda: RemoteRSClient(
                "cpu-mock", timeout=5)) for i in (0, 1)]
            managers = [self.manager(i, exporters[i]) for i in (0, 1)]
            pairs = [[self.submit(managers[i], i, chunk=chunk)[0] for i in (0, 1)]
                     for chunk in range(3)]
            for directories in pairs:
                receipts = [self.wait(d, timeout=10) for d in directories]
                sid = receipts[0]["stripe_id"]
                self.assertEqual(sid, receipts[1]["stripe_id"])
                blocks = [(d / "global_node.bin").read_bytes() for d in directories]
                stripe = self.store.stripes[sid]
                checks = verify_blocks(blocks, stripe["parity"], stripe["manifest"],
                                       stripe["parity_sha256"], allow_fixture=True)
                self.assertEqual(set(checks), {"D0", "D1", "P"})
                for d, receipt, block in zip(directories, receipts, blocks):
                    verify_local(d, receipt, block, allow_fixture=True)
                    self.assertEqual(receipt["remote_outcome"], "COMMITTED")
            for exporter in exporters:
                exporter.close()
        self.assertEqual(len(sent), 6)
        self.assertEqual(len(set(sent)), 6)
        for manager in managers:
            self.assertEqual(len(committed(manager.output, 3, 1)), 3)

    def test_local_commit_never_waits_for_upload_or_implies_global_commit(self):
        gate, entered = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        a = self.exporter(
            0, client_factory=lambda: MockRDMAClient(self.store, gate, entered)
        )
        b = self.exporter(1)
        ma, mb = self.manager(0, a), self.manager(1, b)
        d0, entries, blocks = self.submit(ma, 0)
        self.wait(d0, ("WAITING_PEER",))
        d1, _, _ = self.submit(mb, 1)
        self.assertTrue(entered.wait(2))
        self.assertEqual(
            json.loads((d0 / "manifest.json").read_text())["state"], "COMMITTED"
        )
        self.assertNotEqual(
            json.loads((d0 / "global_status.json").read_text())["global_state"],
            "GLOBAL_COMMITTED",
        )
        duplicate = ma.submit(entries[-1], blocks[-1])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(a.accepted, 1)
        gate.set()
        self.wait(d0)
        self.wait(d1)
        a.close()
        self.assertEqual(a.enqueue(d0, entries, blocks), "DUPLICATE")
        self.wait(d0)

    def test_unmatched_peer_expires_and_local_xor_is_preserved(self):
        e = self.exporter(0)
        m = self.manager(0, e)
        directory, _, _ = self.submit(m, 0)
        waiting = self.wait(directory, ("WAITING_PEER",))
        with e.lock:
            e.active[waiting["stripe_id"]]["deadline"] = time.monotonic() - 1
        r = self.wait(directory, ("EXPIRED",))
        self.assertEqual(r["remote_outcome"], "NOT_SUBMITTED")
        self.assertEqual(
            json.loads((directory / "manifest.json").read_text())["state"], "COMMITTED"
        )
        self.assertFalse(self.store.stripes[r["stripe_id"]]["submitted_nodes"])

    def test_capacity_bytes_and_total_checkpoint_budget(self):
        for limit in (
            dict(max_pending=1, workers=1),
            dict(max_bytes=1),
            dict(max_checkpoints=1),
        ):
            e = self.exporter(0, **limit)
            root = self.root / str(len(list(self.root.iterdir())))
            root.mkdir()
            blocks, entries = fixture(0)
            d0, d1 = root / "a", root / "b"
            d0.mkdir()
            d1.mkdir()
            state = e.enqueue(d0, entries, blocks)
            if limit.get("max_bytes") == 1:
                self.assertEqual(state, "DROPPED_CAPACITY")
                self.assertEqual(e.bytes_pending, 0)
            else:
                self.assertEqual(state, "QUEUED")
                blocks1, entries1 = fixture(0, chunk=1)
                self.assertEqual(e.enqueue(d1, entries1, blocks1), "DROPPED_CAPACITY")
            e.close()

    def test_queued_deadline_includes_queue_wait_and_never_registers_expired_job(self):
        gate, entered = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        e0 = self.exporter(
            0,
            workers=1,
            client_factory=lambda: MockRDMAClient(self.store, gate, entered),
        )
        e1 = self.exporter(1)
        m0, m1 = self.manager(0, e0), self.manager(1, e1)
        d0, _, _ = self.submit(m0, 0)
        self.submit(m1, 1)
        self.assertTrue(entered.wait(2))
        queued, _, _ = self.submit(m0, 0, chunk=1)
        # Deterministically advance deadlines while a mock upload owns the
        # only worker. No wall-clock race against JSON packing/slow CI hosts.
        with e0.lock:
            for job in e0.active.values():
                job["deadline"] = time.monotonic() - 1
        gate.set()
        r = self.wait(queued, ("EXPIRED",))
        self.assertNotIn(r["stripe_id"], self.store.stripes)
        self.assertFalse((queued / "global_node.bin").exists())
        self.wait(d0, ("EXPIRED",))

    def test_unmapped_request_and_closed_exporter(self):
        e = self.exporter(0, requests={sha(b"other"): "other-pair"})
        directory, _, _ = self.submit(self.manager(0, e), 0)
        self.wait(directory, ("SKIPPED_UNMAPPED",))
        self.assertFalse(self.store.stripes)
        e = self.exporter(1)
        e.close()
        d1, _, _ = self.submit(self.manager(1, e), 1)
        self.wait(d1, ("CANCELLED",))

    def test_cancel_waiting_peer_releases_memory(self):
        e = self.exporter(0)
        d, _, _ = self.submit(self.manager(0, e), 0)
        self.wait(d, ("WAITING_PEER",))
        e.close()
        self.wait(d, ("CANCELLED",))
        self.assertEqual(e.bytes_pending, 0)
        self.assertFalse(e.active)

    def test_rdma_failure_is_unknown_remote_outcome_without_fallback(self):
        a = self.exporter(
            0, client_factory=lambda: MockRDMAClient(self.store, fail=True)
        )
        b = self.exporter(1, timeout=1)
        d0, _, _ = self.submit(self.manager(0, a), 0)
        d1, _, _ = self.submit(self.manager(1, b), 1)
        r = self.wait(d0, ("FAILED",))
        self.assertEqual(r["remote_outcome"], "UNKNOWN")
        self.assertIn("no TCP fallback", r["error"])
        r1 = self.wait(d1, ("EXPIRED",))
        self.assertEqual(r1["remote_outcome"], "UNKNOWN")
        self.assertEqual(self.store.stripes[r["stripe_id"]]["submitted_nodes"], [1])

    def test_lost_ack_and_stale_binding_never_become_global_commit(self):
        for failure in ("lost_ack", "stale_binding"):
            with self.subTest(failure=failure):
                store = StripeStore(
                    allow_fixture=True, rdma_directory=self.root / "rdma"
                )

                class BadAck(MockRDMAClient):
                    def rdma_submit_node_block(self, *args, **kwargs):
                        status, stats, disposition = super().rdma_submit_node_block(
                            *args, **kwargs
                        )
                        if failure == "lost_ack":
                            raise ConnectionError(
                                "ACK lost after remote data submission"
                            )
                        status["handle"] = "stale-server-incarnation"
                        return status, stats, disposition

                self.store = store
                a = self.exporter(0, client_factory=lambda: BadAck(store))
                b = self.exporter(1)
                managers = [
                    OnlineChunkManager(
                        self.root / f"{failure}-{i}",
                        f"local{i}",
                        global_exporter=e,
                        test_recovery=True,
                    )
                    for i, e in enumerate((a, b))
                ]
                directories = [self.submit(m, i)[0] for i, m in enumerate(managers)]
                r0 = self.wait(directories[0], ("FAILED",))
                r1 = self.wait(directories[1])
                self.assertEqual(r0["remote_outcome"], "UNKNOWN")
                self.assertEqual(store.stripes[r1["stripe_id"]]["state"], "COMMITTED")
                self.assertEqual(
                    json.loads((directories[0] / "manifest.json").read_text())["state"],
                    "COMMITTED",
                )
                a.close()
                b.close()

    def test_incompatible_model_fails_registration_before_upload(self):
        a, b = self.exporter(0), self.exporter(1)
        d0, _, _ = self.submit(self.manager(0, a), 0)
        waiting = self.wait(d0, ("WAITING_PEER",))
        d1, _, _ = self.submit(
            self.manager(1, b),
            1,
            changes={"model_sha256": "other-model", "model_version": "other-model"},
        )
        r1 = self.wait(d1, ("FAILED",))
        self.assertIn("model/layout mismatch", r1["error"])
        self.assertEqual(r1["remote_outcome"], "NOT_SUBMITTED")
        self.assertEqual(
            self.store.stripes[waiting["stripe_id"]]["submitted_nodes"], []
        )
        a.close()
        self.wait(d0, ("CANCELLED",))

    def test_handoff_exception_does_not_revoke_local_commit(self):
        class Broken:
            def enqueue(self, *args):
                raise OSError("controlled handoff error")

        d, _, _ = self.submit(self.manager(0, Broken()), 0)
        self.assertEqual(
            json.loads((d / "global_handoff_error.json").read_text())["global_state"],
            "FAILED",
        )
        with self.assertRaisesRegex(ValueError, "global handoff failed"):
            committed(d.parent, 1, 1)

    def test_missing_rank_never_exports_and_default_remains_off(self):
        e = self.exporter(0)
        manager = self.manager(0, e)
        blocks, ranks = fixture(0)
        for b, m in zip(blocks[:-1], ranks[:-1]):
            self.assertEqual(manager.submit(m, b)["state"], "PENDING")
        self.assertFalse(self.store.stripes)
        self.assertEqual(e.accepted, 0)
        directory, _, _ = self.submit(self.manager(1, None), 1)
        self.assertFalse((directory / "global_status.json").exists())

    def test_global_identity_binding_and_legacy_offline_compatibility(self):
        exported = []
        for node in (0, 1):
            blocks, ranks = fixture(node)
            identity = identity_for("run", "pair", "epoch", ranks[0])
            exported.append(
                pack(
                    node,
                    [b.tobytes() for b in blocks],
                    [dict(r, global_protection=identity) for r in ranks],
                    allow_fixture=True,
                )
            )
        sid = stripe_id(identity)
        good = build(sid, exported, allow_fixture=True)
        for mutate in (
            lambda m: m.update(stripe_id="wrong"),
            lambda m: m["nodes"][1]["metadata"]["ranks"][2].pop("global_protection"),
            lambda m: m["nodes"][1]["metadata"]["ranks"][0]["global_protection"].update(
                checkpoint_version="stale"
            ),
            lambda m: m["nodes"][1]["metadata"]["ranks"][0].update(
                model_sha256="other"
            ),
            lambda m: m["nodes"][1]["metadata"]["ranks"][0]["layer_records"][0][
                "packed_shape"
            ].update(k=[256, 2, 1]),
        ):
            bad = deepcopy(good)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_manifest(bad, allow_fixture=True)
        legacy = build("legacy", [node_fixture(i) for i in (0, 1)], allow_fixture=True)
        validate_manifest(legacy, allow_fixture=True)

    def test_independent_data_and_parity_corruption_are_rejected(self):
        a, b = self.exporter(0), self.exporter(1)
        dirs = [self.submit(self.manager(i, e), i)[0] for i, e in enumerate((a, b))]
        receipts = [self.wait(d) for d in dirs]
        blocks = [(d / "global_node.bin").read_bytes() for d in dirs]
        key = {k: receipts[0][k] for k in ("stripe_id", "handle", "manifest_sha256")}
        status, parity = self.store.dispatch(dict(op="fetch_parity", **key), b"")
        for lost in range(3):
            changed = list(blocks) + [parity]
            changed[lost] = changed[lost][:-1] + bytes([changed[lost][-1] ^ 1])
            with self.assertRaises(ValueError):
                verify_blocks(
                    changed[:2],
                    changed[2],
                    status["manifest"],
                    status["parity_sha256"],
                    allow_fixture=True,
                )
        bad_receipt = deepcopy(receipts[0])
        bad_receipt["global_protection"]["protection_id"] = "wrong-pair"
        with self.assertRaises(ValueError):
            verify_local(dirs[0], bad_receipt, blocks[0], allow_fixture=True)

    def test_cli_forbids_fixture_and_invalid_config(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"allow_fixture": True}))
        with self.assertRaises(ValueError):
            OnlineRSExporter.from_config(path)
        with self.assertRaises(ValueError):
            self.exporter(0, timeout=float("nan"))
        with self.assertRaises(ValueError):
            self.exporter(0, requests={"raw-request-id": "pair"})


if __name__ == "__main__":
    unittest.main()
