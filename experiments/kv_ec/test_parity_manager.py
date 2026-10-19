"""CPU-only tests for the bounded IPC parity-only prototype."""

from copy import deepcopy
import tempfile
from pathlib import Path
import threading
import time
import unittest
import socket
from unittest.mock import patch

import numpy as np

from experiments.kv_ec.parity_manager import (
    COMMITTED, EXPIRED, FAILED, PENDING, AsyncParityClient, ParityManager,
    ParityManagerServer, submit_once,
)
from experiments.kv_ec.test_debug_capture import rank_fixture
from experiments.kv_ec.checkpoint_protocol import recv_message, send_message


def metadata_and_blocks(tp=4, layers=(0,), count=4):
    blocks, metadata = [], []
    for rank in range(tp):
        payload, meta = rank_fixture(rank, tp, layers=layers, count=count, heads=8 // tp)
        meta.update(snapshot_id="checkpoint", model_identifier="fixture-model",
                    model_sha256="fixture", source_commit="fixture",
                    core_diff_sha256="fixture", request_id="request",
                    token_ids_sha256="tokens", lifetime="fixture",
                    completion_dependency="fixture")
        meta.update(format="kv-ec-rank-bin-v1", packed_byte_count=int(payload.size),
                    sha256=__import__("hashlib").sha256(payload).hexdigest(),
                    snapshot_completion_status="complete", source_unchanged=True)
        blocks.append(payload)
        metadata.append(meta)
    return blocks, metadata


class ManagerTests(unittest.TestCase):
    def test_deferred_fast_checksum_is_derived_and_committed(self):
        blocks, metadata = metadata_and_blocks(tp=2)
        for meta in metadata:
            meta.pop("k_sha256", None)
            meta.pop("v_sha256", None)
            for record in meta["layer_records"]:
                record.pop("k_sha256", None)
                record.pop("v_sha256", None)
            meta["checksum_deferred"] = True
        manager = ParityManager(test_oracle=True)
        self.assertEqual(manager.submit(metadata[0], blocks[0])["state"], PENDING)
        status = manager.submit(metadata[1], blocks[1])
        self.assertEqual(status["state"], COMMITTED)
        self.assertEqual(status["parity_bytes"], blocks[0].size)
        self.assertEqual(len(status["oracle_checks"]), 2)
        self.assertEqual(set(status["rank_hashes"]), {"0", "1"})

    def test_deferred_fast_checksum_corruption_fails_closed(self):
        blocks, metadata = metadata_and_blocks(tp=2)
        for meta in metadata:
            meta.pop("k_sha256", None)
            meta.pop("v_sha256", None)
            for record in meta["layer_records"]:
                record.pop("k_sha256", None)
                record.pop("v_sha256", None)
            meta["checksum_deferred"] = True
        blocks[1][0] ^= 1
        manager = ParityManager()
        manager.submit(metadata[0], blocks[0])
        self.assertEqual(manager.submit(metadata[1], blocks[1])["state"], FAILED)
    def test_tp2_commit_and_each_erasure_releases_sources(self):
        blocks, metadata = metadata_and_blocks(tp=2)
        originals = [block.copy() for block in blocks]
        manager = ParityManager(test_oracle=True)
        self.assertEqual(manager.submit(metadata[0], blocks[0])["ranks"], [0])
        status = manager.submit(metadata[1], blocks[1])
        self.assertEqual(status["state"], COMMITTED)
        self.assertEqual(status["oracle_checks"], [
            {"lost_rank": rank, "byte_exact": True} for rank in range(2)])
        checkpoint = manager._checkpoints["checkpoint"]
        self.assertEqual(checkpoint.ranks, {})
        self.assertIsNone(checkpoint.oracle)
        self.assertEqual(checkpoint.parity.nbytes, blocks[0].nbytes)
        self.assertEqual(set(status["rank_hashes"]), {"0", "1"})
        for actual, original in zip(blocks, originals):
            np.testing.assert_array_equal(actual, original)
        # Retained sources are allowed ONLY for the explicit test API.
        oracle = ParityManager(retain_oracle=True)
        for meta, block in zip(metadata, blocks):
            oracle.submit(meta, block)
        for lost in range(2):
            np.testing.assert_array_equal(oracle.recover_for_test("checkpoint", lost), originals[lost])
        with self.assertRaises(ValueError):
            oracle.recover_for_test("checkpoint", 2)
        self.assertEqual(ParityManager().status("checkpoint")["state"], "UNKNOWN")
        manager.release("checkpoint")
        self.assertEqual(manager.status("checkpoint")["state"], "UNKNOWN")

    def test_tp2_missing_rank_expires(self):
        blocks, metadata = metadata_and_blocks(tp=2)
        manager = ParityManager(timeout=0.01)
        self.assertEqual(manager.submit(metadata[0], blocks[0])["state"], PENDING)
        with patch("experiments.kv_ec.parity_manager.time.monotonic",
                   return_value=manager._checkpoints["checkpoint"].deadline + 1):
            self.assertEqual(manager.status("checkpoint")["state"], EXPIRED)
        self.assertFalse(manager._checkpoints["checkpoint"].ranks)

    def test_tp2_duplicates_corruption_and_identity_mismatches(self):
        blocks, metadata = metadata_and_blocks(tp=2)
        manager = ParityManager()
        manager.submit(metadata[0], blocks[0])
        self.assertEqual(manager.submit(metadata[0], blocks[0])["reason"], "DUPLICATE_RANK")
        changes = {"request_id": "other", "model_sha256": "other", "format": "v2",
                   "token_start": 1, "layers": [1], "dtype": "torch.float16"}
        for key, value in changes.items():
            with self.subTest(key=key):
                manager = ParityManager()
                manager.submit(metadata[0], blocks[0])
                bad = deepcopy(metadata[1]); bad[key] = value
                self.assertEqual(manager.submit(bad, blocks[1])["state"], FAILED)
                self.assertFalse(manager._checkpoints["checkpoint"].ranks)
        manager = ParityManager()
        manager.submit(metadata[0], blocks[0])
        bad = blocks[1].copy(); bad[0] ^= 1
        self.assertEqual(manager.submit(metadata[1], bad)["state"], FAILED)

    def test_tp2_tp4_cannot_mix_in_either_order(self):
        for first_tp, second_tp in ((2, 4), (4, 2)):
            with self.subTest(first_tp=first_tp):
                blocks, metadata = metadata_and_blocks(first_tp)
                other_blocks, other_metadata = metadata_and_blocks(second_tp)
                manager = ParityManager()
                manager.submit(metadata[0], blocks[0])
                self.assertEqual(manager.submit(other_metadata[1], other_blocks[1])["reason"],
                                 "METADATA_MISMATCH")
                self.assertEqual(manager.status("checkpoint")["state"], FAILED)

    def test_tp2_invalid_rank_or_version_never_commits(self):
        blocks, metadata = metadata_and_blocks(2)
        for key, value in (("tp_rank", 2), ("tp_rank", -1), ("tp_size", 3), ("format", "v2")):
            with self.subTest(key=key, value=value):
                bad = deepcopy(metadata[0]); bad[key] = value
                manager = ParityManager()
                self.assertEqual(manager.submit(bad, blocks[0])["state"], FAILED)

    def test_tp2_tp4_all_layer_sizes_and_all_erasures(self):
        # CPU-generated fixtures exercise the online shape validator, not GPU provenance.
        for tp in (2, 4):
            for count in (128, 256, 512):
                with self.subTest(tp=tp, count=count):
                    blocks, metadata = metadata_and_blocks(tp, tuple(range(36)), count)
                    expected = 36 * count * (8 // tp) * 128 * 2 * 2
                    manager = ParityManager(test_oracle=True)
                    for meta, block in zip(metadata, blocks):
                        self.assertEqual(block.nbytes, expected)
                        meta["provenance"] = "sglang_request"
                        status = manager.submit(meta, block)
                    self.assertEqual(status["state"], COMMITTED)
                    self.assertEqual(status["parity_bytes"], expected)
                    self.assertEqual(len(status["oracle_checks"]), tp)
                    self.assertIsNone(manager._checkpoints["checkpoint"].oracle)

    def test_tp2_wrong_runtime_shape_is_rejected(self):
        payload, meta = rank_fixture(0, 2, count=128, heads=2)
        meta["provenance"] = "sglang_request"
        self.assertEqual(ParityManager().submit(meta, payload)["state"], FAILED)

    def test_four_rank_commit_releases_inputs_and_keeps_only_parity(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(timeout=1, max_pending=1)
        statuses = [manager.submit(meta, block) for meta, block in zip(metadata, blocks)]
        self.assertEqual([s["state"] for s in statuses], [PENDING, PENDING, PENDING, COMMITTED])
        status = manager.status("checkpoint")
        self.assertEqual(status["state"], COMMITTED)
        self.assertEqual(status["parity_bytes"], blocks[0].size)
        self.assertEqual(manager._checkpoints["checkpoint"].ranks, {})
        self.assertIsNotNone(manager._checkpoints["checkpoint"].parity)
        self.assertEqual(manager.submit(metadata[0], blocks[0])["reason"], "CHECKPOINT_CLOSED")
        self.assertEqual(manager.status("checkpoint")["state"], COMMITTED)
        with self.assertRaisesRegex(RuntimeError, "oracle is disabled"):
            manager.recover_for_test("checkpoint", 0)

    def test_test_only_oracle_recovers_all_ranks(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(retain_oracle=True)
        for meta, block in zip(metadata, blocks):
            manager.submit(meta, block)
        for lost in range(4):
            np.testing.assert_array_equal(manager.recover_for_test("checkpoint", lost), blocks[lost])

    def test_missing_rank_expires(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(timeout=0.01)
        manager.submit(metadata[0], blocks[0])
        time.sleep(0.02)
        self.assertEqual(manager.status("checkpoint")["state"], EXPIRED)

    def test_duplicate_rank_and_metadata_mismatch_fail(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager()
        self.assertEqual(manager.submit(metadata[0], blocks[0])["state"], PENDING)
        self.assertEqual(manager.submit(metadata[0], blocks[0])["reason"], "DUPLICATE_RANK")
        manager = ParityManager()
        manager.submit(metadata[0], blocks[0])
        bad = deepcopy(metadata[1])
        bad["token_count"] = 99
        self.assertEqual(manager.submit(bad, blocks[1])["reason"], "METADATA_MISMATCH")

    def test_corruption_and_layout_mismatch_fail_before_commit(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager()
        bad = blocks[0].copy()
        bad[0] ^= 1
        self.assertEqual(manager.submit(metadata[0], bad)["state"], FAILED)
        manager = ParityManager()
        manager.submit(metadata[0], blocks[0])
        bad_meta = deepcopy(metadata[1])
        bad_meta["layer_records"][0]["packed_shape"]["k"][0] = 3
        self.assertEqual(manager.submit(bad_meta, blocks[1])["reason"], "METADATA_MISMATCH")

    def test_pending_capacity_is_explicit(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(max_pending=1)
        manager.submit(metadata[0], blocks[0])
        second = deepcopy(metadata[1])
        second["snapshot_id"] = "other"
        self.assertEqual(manager.submit(second, blocks[1])["reason"], "PENDING_CAPACITY")

    def test_different_snapshot_ids_never_mix(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(max_pending=2)
        manager.submit(metadata[0], blocks[0])
        for rank in range(1, 4):
            metadata[rank]["snapshot_id"] = "different"
            manager.submit(metadata[rank], blocks[rank])
        self.assertEqual(manager.status("checkpoint")["state"], PENDING)
        self.assertEqual(manager.status("different")["state"], PENDING)

    def test_transient_oracle_releases_full_data(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(test_oracle=True)
        for rank in range(4):
            manager.submit(metadata[rank], blocks[rank])
        checkpoint = manager._checkpoints["checkpoint"]
        self.assertEqual(len(manager.status("checkpoint")["oracle_checks"]), 4)
        self.assertFalse(checkpoint.ranks)
        self.assertIsNone(checkpoint.oracle)

    def test_corruption_fails_existing_pending_checkpoint(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager()
        manager.submit(metadata[0], blocks[0])
        blocks[1][0] ^= 1
        manager.submit(metadata[1], blocks[1])
        self.assertEqual(manager.status("checkpoint")["state"], FAILED)
        self.assertFalse(manager._checkpoints["checkpoint"].ranks)

    def test_input_mutation_after_submit_does_not_change_owned_copy(self):
        blocks, metadata = metadata_and_blocks()
        manager = ParityManager(retain_oracle=True)
        original = blocks[0].copy()
        manager.submit(metadata[0], blocks[0])
        blocks[0][:] = 0
        self.assertTrue(np.array_equal(manager._checkpoints["checkpoint"].oracle[0], original))


class IpcTests(unittest.TestCase):
    def test_submit_once_normalizes_numpy_and_reuses_owned_bytes(self):
        block = np.arange(16, dtype=np.uint8)
        owned = block.tobytes()
        for payload in (block, owned):
            with self.subTest(payload_type=type(payload).__name__):
                with patch("experiments.kv_ec.parity_manager.socket.socket"), \
                     patch("experiments.kv_ec.parity_manager.send_message") as send, \
                     patch("experiments.kv_ec.parity_manager.recv_message",
                           return_value=({"state": PENDING}, b"")):
                    response = submit_once("/unused", {}, payload)
                sent = send.call_args.args[2]
                self.assertIsInstance(sent, bytes)
                self.assertEqual(sent, owned)
                if payload is owned:
                    self.assertIs(sent, owned)
                self.assertEqual(response["state"], PENDING)

    def test_unix_socket_service_commits_without_files(self):
        for tp in (2, 4):
            with self.subTest(tp=tp):
                self._socket_service_commits_without_files(tp)

    def _socket_service_commits_without_files(self, tp):
        blocks, metadata = metadata_and_blocks(tp)
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "parity.sock"
            import socket
            try:
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.bind(str(socket_path))
                socket_path.unlink()
            except PermissionError:
                self.skipTest("sandbox disallows Unix domain sockets")
            manager = ParityManager(retain_oracle=True)
            server = ParityManagerServer(socket_path, manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.assertTrue(server.ready.wait(timeout=2))
            try:
                for meta, block in zip(metadata, blocks):
                    response = submit_once(socket_path, meta, block)
            except PermissionError:
                server.stop()
                thread.join(timeout=1)
                self.skipTest("sandbox disallows Unix domain sockets")
            self.assertEqual(response["state"], COMMITTED)
            self.assertEqual(manager.status("checkpoint")["state"], COMMITTED)
            self.assertEqual(list(Path(directory).iterdir()), [socket_path])
            server.stop()
            thread.join(timeout=1)

    def test_async_queue_full_is_nonblocking_and_explicit(self):
        blocks, metadata = metadata_and_blocks()
        with patch("threading.Thread.start"):
            client = AsyncParityClient("/nonexistent/socket", max_queue=1, timeout=0.01)
            first = client.submit(metadata[0], blocks[0])
            second = client.submit(metadata[1], blocks[1])
            third = client.submit(metadata[2], blocks[2])
        self.assertTrue(first["accepted"])
        self.assertIn(second["accepted"], (True, False))
        if not second["accepted"]:
            self.assertEqual(second["reason"], "QUEUE_FULL")
        self.assertFalse(third["accepted"])
        self.assertEqual(third["reason"], "QUEUE_FULL")

    def test_async_queue_owns_data_and_metadata(self):
        blocks, metadata = metadata_and_blocks()
        original = blocks[0].tobytes()
        with patch("threading.Thread.start"):
            client = AsyncParityClient("/unused", max_queue=1)
        client.submit(metadata[0], blocks[0])
        blocks[0][:] = 0
        metadata[0]["layers"].append(999)
        queued_metadata, queued_bytes = client.queue.get_nowait()
        self.assertEqual(queued_bytes, original)
        self.assertEqual(queued_metadata["layers"], [0])

    def test_protocol_roundtrip_with_partial_reads(self):
        from experiments.kv_ec.checkpoint_protocol import encode_header
        raw = encode_header({"payload_bytes": 4}) + b"abcd"
        class Reader:
            def gettimeout(self):
                return None
            def recv(self, size):
                nonlocal raw
                result, raw = raw[:min(size, 3)], raw[min(size, 3):]
                return result
        metadata, payload = recv_message(Reader())
        self.assertEqual(payload, b"abcd")
        self.assertEqual(metadata["payload_bytes"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
