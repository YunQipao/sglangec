import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.kv_ec.multinode import coordinator, worker
from experiments.kv_ec.multinode.rs_codec import CODEC, CODEC_VERSION, encode, recover
from experiments.kv_ec.multinode.verify import verify_stripe, verify_synthetic
from experiments.kv_ec.multinode.worker import _check_recovered_block, _write_error_result


def args_for(tmpdir, *, timeout=1):
    return SimpleNamespace(
        transport="tcp", node_b="worker", control_port=19470, data_port=19471,
        return_bind_host="127.0.0.1", return_port=19472,
        d0_bytes=128, d1_bytes=64, seed=20261025, rdma_device="-",
        rdma_gid_index=None, rdma_port_num=1,
        chunk_id="chunk-0", experiment_id="test", stripe_id="stripe-0",
        checkpoint_version="checkpoint-v1", timeout=timeout,
        output=Path(tmpdir) / "coordinator",
    )


def committed_result(args, elapsed=.01):
    """A complete, independently verifiable unequal-length stripe fixture."""
    width = max(args.d0_bytes, args.d1_bytes)
    raw = [np.random.default_rng(args.seed + index).integers(0, 256, length, dtype=np.uint8)
           for index, length in enumerate((args.d0_bytes, args.d1_bytes))]
    d0, d1 = [np.pad(block, (0, width - len(block))) for block in raw]
    blocks = dict(zip(("D0", "D1", "P"), encode(d0, d1)))
    binding = {key: getattr(args, key) for key in
               ("experiment_id", "stripe_id", "checkpoint_version")}
    metadata = []
    for role, block in blocks.items():
        length = args.d0_bytes if role == "D0" else args.d1_bytes if role == "D1" else width
        metadata.append({
            **binding, "chunk_id": args.chunk_id, "block_role": role,
            "source_node_id": "node-a" if role == "D0" else "node-b",
            "model_fingerprint": "synthetic-seeded-data",
            "payload_length": length, "padded_length": width,
            "padding_length": width - length,
            "codec": CODEC, "codec_version": CODEC_VERSION,
            "rs_parameters": {"k": 2, "m": 1, "field": "GF(2^8)"},
            "checksum": hashlib.sha256(block.tobytes()).hexdigest(),
            "protection_state": "COMMITTED",
        })
    return {
        **binding, "state": "COMMITTED", "transport": args.transport,
        "node_b_return_transfer_ms": elapsed * 1000, "stripe_metadata": metadata,
        "checkpoint": {"state": "COMMITTED", "data_received": True,
                       "parity_encoded": True, "checksums_valid": True},
        "verification": {"D0_lost_byte_exact": True, "D1_lost_byte_exact": True,
                         "P_recomputed_byte_exact": True},
    }


class TCPFailurePathTests(unittest.TestCase):
    def run_reverse_case(self, temporary, *, elapsed=.01, receive_error=None,
                         worker_state="COMMITTED", wait_for_cancel=False,
                         send_error=None, corrupt=False, ready=True, ack_error=None,
                         result_error=None, mutate_result=None, transport="tcp",
                         ack_response=None):
        args = args_for(temporary, timeout=.2)
        args.transport = transport
        self.ack_requests = []
        sent = threading.Event()
        exited = threading.Event()
        fake_transport = mock.Mock(name="transport")
        fake_transport.name = transport

        def send(*unused):
            sent.set()
            if send_error is not None:
                raise send_error
            return .001

        fake_transport.send.side_effect = send
        fake_transport.send_with_stats.side_effect = lambda *a: {"elapsed_s": send(*a)}

        def receive(host, port, destination, *, timeout, ready_event, status, cancel_event):
            try:
                if ready:
                    status["phase"] = "listening"
                    ready_event.set()
                if wait_for_cancel or not ready:
                    if not cancel_event.wait(2):
                        raise AssertionError("receiver was not cancelled")
                    raise TimeoutError("test receiver cancelled")
                if not sent.wait(2):
                    raise AssertionError("D0 was not sent")
                if receive_error is not None:
                    raise receive_error
                d1 = np.random.default_rng(args.seed + 1).integers(
                    0, 256, args.d1_bytes, dtype=np.uint8).tobytes()
                Path(destination).write_bytes(b"corrupt" if corrupt else d1)
                status["phase"] = "complete"
                return elapsed
            finally:
                exited.set()

        def request(host, port, operation, *, timeout=5):
            if operation == "health":
                return {"state": "READY"}
            if isinstance(operation, dict):
                self.ack_requests.append(operation)
                if ack_error is not None:
                    raise ack_error
                if ack_response is not None:
                    return ack_response
                return {"state": "COMMITTED", "acknowledgement": "ACKNOWLEDGED",
                        **{key: operation[key] for key in
                           ("experiment_id", "stripe_id", "checkpoint_version")}}
            if result_error is not None:
                raise result_error
            if worker_state != "COMMITTED":
                return {"state": worker_state, "error": "worker checksum failed"}
            result = committed_result(args, elapsed)
            if mutate_result is not None:
                mutate_result(result)
            return result

        def receive_rdma(port, destination, **options):
            return {"elapsed_s": receive(
                args.return_bind_host, port, destination, timeout=options["timeout"],
                ready_event=threading.Event(), status={}, cancel_event=threading.Event())}

        try:
            with mock.patch.object(coordinator, "TCPTransport", return_value=fake_transport), \
                    mock.patch.object(coordinator, "RDMATransport", return_value=fake_transport), \
                    mock.patch.object(coordinator, "tcp_receive_file", side_effect=receive), \
                    mock.patch.object(coordinator, "rdma_receive_file", side_effect=receive_rdma), \
                    mock.patch.object(coordinator, "request_json", side_effect=request):
                return coordinator.run(args)
        finally:
            self.assertTrue(exited.wait(1), "receive thread did not exit")
            if not ready:
                fake_transport.send.assert_not_called()

    def test_reverse_elapsed_seconds_are_success(self):
        for elapsed in (.01, 0.0):
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as temporary:
                result = self.run_reverse_case(temporary, elapsed=elapsed)
                self.assertEqual(result["state"], "COMMITTED")
                self.assertEqual(result["acknowledgement"], "ACKNOWLEDGED")
                self.assertEqual(len(self.ack_requests), 1)
                self.assertTrue((Path(temporary) / "coordinator/result.json").is_file())

    def test_reverse_connection_error_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            error = ConnectionError("reverse peer closed early")
            with self.assertRaisesRegex(ConnectionError, "reverse peer closed early") as raised:
                self.run_reverse_case(temporary, receive_error=error)
            self.assertIs(raised.exception, error)
            self.assertIn("reverse_listener=", str(error))
            self.assertFalse((Path(temporary) / "coordinator").exists())

    def test_returned_exception_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            error = ConnectionError("returned receiver error")
            with self.assertRaisesRegex(ConnectionError, "returned receiver error"):
                self.run_reverse_case(temporary, elapsed=error)

    def test_reverse_timeout_cancels_receiver(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TimeoutError, "reverse data transfer did not complete"):
                self.run_reverse_case(temporary, worker_state="PENDING", wait_for_cancel=True)

    def test_reverse_readiness_timeout_cancels_receiver(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TimeoutError, "reverse listener did not become ready"):
                self.run_reverse_case(temporary, ready=False)

    def test_d0_connection_failure_cancels_receiver(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ConnectionError, "D0 connection refused"):
                self.run_reverse_case(temporary, send_error=ConnectionError("D0 connection refused"),
                                      wait_for_cancel=True)

    def test_worker_error_cancels_receiver_and_does_not_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "worker checksum failed"):
                self.run_reverse_case(temporary, worker_state="ERROR", wait_for_cancel=True)
            self.assertFalse((Path(temporary) / "coordinator").exists())

    def test_successful_receive_does_not_hide_worker_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "worker checksum failed"):
                self.run_reverse_case(temporary, worker_state="ERROR")
            self.assertFalse((Path(temporary) / "coordinator").exists())

    def test_successful_receive_still_checks_d1_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "D1 transfer is not byte exact"):
                self.run_reverse_case(temporary, corrupt=True)
            self.assertFalse((Path(temporary) / "coordinator").exists())

    def test_rdma_uses_same_commit_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_reverse_case(temporary, transport="rdma")
            self.assertEqual(result["state"], "COMMITTED")
            self.assertEqual(result["acknowledgement"], "ACKNOWLEDGED")
            self.assertEqual(len(self.ack_requests), 1)

    def test_ack_loss_cannot_publish_coordinator_success(self):
        for error in (ConnectionRefusedError("ACK connection refused"),
                      ConnectionError("ACK response ended early"), TimeoutError("ACK timed out")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(type(error)):
                    self.run_reverse_case(temporary, ack_error=error)
                self.assertEqual(len(self.ack_requests), 1)
                self.assertFalse((Path(temporary) / "coordinator/result.json").exists())

    def test_invalid_ack_response_cannot_publish_success(self):
        for response in ({"state": "ERROR", "reason": "ack_identity_mismatch"},
                         {"state": "COMMITTED"},
                         {"state": "COMMITTED", "acknowledgement": "ACKNOWLEDGED",
                          "experiment_id": "other", "stripe_id": "stripe-0",
                          "checkpoint_version": "checkpoint-v1"}):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
                    self.run_reverse_case(temporary, ack_response=response)
                self.assertFalse((Path(temporary) / "coordinator/result.json").exists())

    def test_worker_exit_before_result_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ConnectionRefusedError):
                self.run_reverse_case(temporary, result_error=ConnectionRefusedError("worker exited"))
            self.assertEqual(self.ack_requests, [])
            self.assertFalse((Path(temporary) / "coordinator").exists())

    def test_commit_identity_mismatches_are_rejected_before_ack(self):
        for key in ("experiment_id", "stripe_id", "checkpoint_version"):
            for top_level in (True, False):
                with self.subTest(key=key, top_level=top_level), tempfile.TemporaryDirectory() as temporary:
                    def mutate(result):
                        target = result if top_level else result["stripe_metadata"][0]
                        target[key] = "other-run"

                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        self.run_reverse_case(temporary, mutate_result=mutate)
                    self.assertEqual(self.ack_requests, [])

    def test_incomplete_or_corrupt_metadata_is_rejected_before_ack(self):
        mutations = (
            lambda r: r["checkpoint"].update(checksums_valid=False),
            lambda r: r["verification"].update(D1_lost_byte_exact=False),
            lambda r: r["stripe_metadata"].pop(),
            lambda r: r["stripe_metadata"][1].update(checksum="corrupt"),
            lambda r: r["stripe_metadata"][1].update(payload_length=63),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    self.run_reverse_case(temporary, mutate_result=mutate)
                self.assertEqual(self.ack_requests, [])

    def test_recovery_checksum_uses_unpadded_payload_length(self):
        raw0 = np.arange(16, dtype=np.uint8)
        raw1 = np.arange(8, dtype=np.uint8) + 32
        d0 = raw0.copy()
        d1 = np.pad(raw1, (0, 8))
        _, _, parity = encode(d0, d1)
        _, recovered1 = recover(data0=d0, data1=None, parity=parity,
                                lengths=(len(raw0), len(raw1)))
        import hashlib
        _check_recovered_block(
            "D1", recovered1, raw1.tobytes(), hashlib.sha256(raw1).hexdigest())

    def test_unequal_data_lengths_are_padded_and_verified(self):
        self.assertEqual(verify_synthetic(17, 9)["state"], "PASS")

    def test_corrupt_payload_is_rejected_before_rs_commit(self):
        d0 = np.array([1, 2, 3, 0], dtype=np.uint8)
        d1 = np.array([4, 5, 6, 7], dtype=np.uint8)
        _, _, parity = encode(d0, d1)
        blocks = {"D0": d0, "D1": d1, "P": parity}
        import hashlib
        common = {
            "experiment_id": "e", "stripe_id": "s", "chunk_id": "c",
            "model_fingerprint": "fixture", "padded_length": 4,
            "codec": CODEC, "codec_version": CODEC_VERSION,
            "rs_parameters": {"k": 2, "m": 1, "field": "GF(2^8)"},
            "checkpoint_version": "v1", "protection_state": "COMMITTED",
        }
        metadata = {}
        for role, block in blocks.items():
            metadata[role] = {
                **common,
                "source_node_id": "node-a" if role == "D0" else "node-b",
                "payload_length": 3 if role == "D0" else 4,
                "padding_length": 1 if role == "D0" else 0,
                "checksum": hashlib.sha256(block.tobytes()).hexdigest(),
                "block_role": role,
            }
        blocks["D1"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_stripe(metadata, blocks)

    def test_worker_error_result_contains_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "worker"
            error = ValueError("D1 checksum mismatch")
            try:
                raise error
            except ValueError as exc:
                _write_error_result(output, exc)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["state"], "ERROR")
            self.assertEqual(result["error"], "D1 checksum mismatch")
            self.assertIn("ValueError", result["traceback"])

    def test_reverse_worker_failure_is_not_reported_as_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = args_for(temporary)
            fake_transport = mock.Mock(name="transport")
            fake_transport.name = "tcp"
            fake_transport.send.return_value = .001

            def receive_failure(host, port, destination, *, timeout, ready_event, status, cancel_event):
                status["phase"] = "listening"
                ready_event.set()
                # Simulate a Worker that accepted D0 and failed during the
                # RS/checksum phase before it could send D1 back.
                time.sleep(.02)
                raise ValueError("D1 checksum mismatch")

            def request(host, port, operation, *, timeout=5):
                if operation == "health":
                    return {"state": "READY"}
                return {"state": "ERROR", "error": "worker failed before D1 return"}

            with mock.patch.object(coordinator, "TCPTransport", return_value=fake_transport), \
                    mock.patch.object(coordinator, "tcp_receive_file", side_effect=receive_failure), \
                    mock.patch.object(coordinator, "request_json", side_effect=request):
                with self.assertRaisesRegex(ValueError, "D1 checksum mismatch") as raised:
                    coordinator.run(args)
            self.assertIn("reverse_listener=127.0.0.1:19472", str(raised.exception))
            fake_transport.send.assert_called_once()

    def test_reverse_bind_failure_is_reported_before_d0(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = args_for(temporary)
            fake_transport = mock.Mock(name="transport")
            fake_transport.name = "tcp"

            def bind_failure(host, port, destination, *, timeout, ready_event, status, cancel_event):
                status.update(phase="bind_failed", error="OSError: address already in use")
                ready_event.set()
                raise OSError("address already in use")

            with mock.patch.object(coordinator, "TCPTransport", return_value=fake_transport), \
                    mock.patch.object(coordinator, "tcp_receive_file", side_effect=bind_failure), \
                    mock.patch.object(coordinator, "request_json", return_value={"state": "READY"}):
                with self.assertRaisesRegex(OSError, "address already in use"):
                    coordinator.run(args)
            fake_transport.send.assert_not_called()

    def test_owned_worker_is_reaped_after_failure(self):
        from experiments.kv_ec.multinode.tests.test_tcp_integration import stop_process

        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            stop_process(process)
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class WorkerControlLifecycleTests(unittest.TestCase):
    @contextmanager
    def running_worker(self, temporary, *, transport="tcp", timeout=.5, protection_error=None,
                       protection=None):
        args = args_for(temporary, timeout=timeout)
        args.output = Path(temporary) / "worker"
        args.transport = transport
        args.control_host = "127.0.0.1"
        args.data_host, args.data_port = "127.0.0.1", 19471
        args.return_host, args.return_port = "127.0.0.1", 19472
        requests, responses = queue.Queue(), queue.Queue()
        published = threading.Event()
        servers, results, errors = [], [], []

        def serve(host, port, result_path, ready, stop, acknowledged, binding, timeout):
            servers.append(threading.current_thread())
            ready.set()
            while not stop.is_set() and not acknowledged.is_set():
                try:
                    request = requests.get(timeout=.01)
                except queue.Empty:
                    continue
                response = worker._control_response(request, result_path, binding)
                responses.put(response)
                if request not in ("health", "result") and response.get("acknowledgement") == "ACKNOWLEDGED":
                    acknowledged.set()

        def protect(args, output):
            if protection_error is not None:
                raise protection_error
            if protection is None:
                result = committed_result(args)
                worker._atomic_json(output / "result.json", result)
            else:
                result = protection(args, output)
            published.set()
            return result

        def run():
            try:
                results.append(worker.run(args))
            except BaseException as exc:
                errors.append(exc)

        def request(operation):
            requests.put(operation if isinstance(operation, str) else json.dumps(operation))
            return responses.get(timeout=1)

        with mock.patch.object(worker, "_control_server", side_effect=serve), \
                mock.patch.object(worker, "_run_protection", side_effect=protect):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                yield args, thread, request, published, results, errors
            finally:
                thread.join(timeout + 1)
                self.assertFalse(thread.is_alive(), "worker did not exit")
                self.assertTrue(servers)
                self.assertTrue(all(not server.is_alive() for server in servers))
                self.assertTrue(all(not server.daemon for server in servers))

    def test_committed_worker_serves_result_until_bound_ack_then_exits(self):
        for transport in ("tcp", "rdma"):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as temporary:
                with self.running_worker(temporary, transport=transport) as state:
                    args, thread, request, published, results, errors = state
                    self.assertTrue(published.wait(1))
                    self.assertTrue(thread.is_alive())
                    result = request("result")
                    self.assertEqual(result["state"], "COMMITTED")
                    self.assertEqual(result["acknowledgement"], "WAITING")
                    self.assertEqual(result["transport"], transport)
                    self.assertEqual(result["stripe_metadata"][0]["payload_length"], 128)
                    self.assertEqual(result["stripe_metadata"][1]["payload_length"], 64)
                    binding = {key: getattr(args, key) for key in
                               ("experiment_id", "stripe_id", "checkpoint_version")}
                    response = request({"op": "ack", **binding})
                    self.assertEqual(response["acknowledgement"], "ACKNOWLEDGED")
                    thread.join(1)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(results[0]["acknowledgement"], "ACKNOWLEDGED")
                    persisted = json.loads((args.output / "result.json").read_text())
                    self.assertEqual(persisted["state"], "COMMITTED")
                    receipt = json.loads((args.output / "acknowledgement.json").read_text())
                    self.assertEqual(receipt, {"state": "ACKNOWLEDGED", **binding})

    def test_worker_rs_persistence_and_ack_for_both_transports(self):
        # Exercise real worker RS/persistence and coordinator validation;
        # only the unavailable network/HCA I/O is replaced with CPU fixtures.
        protection = worker._run_protection
        for transport in ("tcp", "rdma"):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as temporary:
                fixture_args = args_for(temporary)
                raw0 = np.random.default_rng(fixture_args.seed).integers(
                    0, 256, fixture_args.d0_bytes, dtype=np.uint8).tobytes()
                conn = mock.MagicMock()
                conn.__enter__.return_value = conn
                conn.recv.side_effect = [struct.pack("!Q", len(raw0)), raw0]
                server = mock.MagicMock()
                server.__enter__.return_value = server
                server.accept.return_value = (conn, ("127.0.0.1", 1))
                returned = []

                def send(host, port, data):
                    returned.append(data)
                    return .01

                sender = mock.Mock()
                sender.send.side_effect = send
                sender.send_with_stats.side_effect = lambda *a: {"elapsed_s": send(*a)}

                def receive(port, path, **options):
                    Path(path).write_bytes(raw0)
                    return {"elapsed_s": .01}

                with mock.patch.object(worker.socket, "socket", return_value=server), \
                        mock.patch.object(worker, "TCPTransport", return_value=sender), \
                        mock.patch("experiments.kv_ec.multinode.rdma_transport.RDMATransport", return_value=sender), \
                        mock.patch.object(worker, "receive_file", side_effect=receive):
                    with self.running_worker(temporary, transport=transport, protection=protection) as state:
                        args, thread, request, published, results, errors = state
                        self.assertTrue(published.wait(1), errors)
                        self.assertTrue(thread.is_alive())
                        result = request("result")
                        self.assertEqual(result["state"], "COMMITTED")
                        blocks = {role: np.frombuffer((args.output / filename).read_bytes(), dtype=np.uint8)
                                  for role, filename in (("D0", "D0.padded.bin"),
                                                         ("D1", "D1.padded.bin"), ("P", "P.bin"))}
                        coordinator._validate_commit(args, result, blocks)
                        self.assertEqual(returned, [blocks["D1"][:args.d1_bytes].tobytes()])
                        binding = {key: getattr(args, key) for key in
                                   ("experiment_id", "stripe_id", "checkpoint_version")}
                        request({"op": "ack", **binding})
                    self.assertEqual(errors, [])
                    self.assertEqual(results[0]["acknowledgement"], "ACKNOWLEDGED")

    def test_wrong_ack_identities_do_not_stop_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.running_worker(temporary) as state:
                args, thread, request, published, results, errors = state
                self.assertTrue(published.wait(1))
                binding = {key: getattr(args, key) for key in
                           ("experiment_id", "stripe_id", "checkpoint_version")}
                for key in binding:
                    response = request({"op": "ack", **binding, key: "other-run"})
                    self.assertEqual(response["reason"], "ack_identity_mismatch")
                    self.assertTrue(thread.is_alive())
                    self.assertFalse((args.output / "acknowledgement.json").exists())
                self.assertEqual(request("result")["state"], "COMMITTED")
                request({"op": "ack", **binding})
            self.assertEqual(errors, [])

    def test_missing_ack_times_out_without_destroying_commit(self):
        for transport in ("tcp", "rdma"):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as temporary:
                with self.running_worker(temporary, transport=transport, timeout=.05) as state:
                    args, thread, request, published, results, errors = state
                    self.assertTrue(published.wait(1))
                    original = (args.output / "result.json").read_bytes()
                    thread.join(1)
                    self.assertEqual(results, [])
                    self.assertIsInstance(errors[0], TimeoutError)
                    self.assertIn("COMMITTED but coordinator ACK", str(errors[0]))
                    # main()'s failure reporting must not overwrite COMMITTED.
                    worker._write_error_result(args.output, errors[0])
                    self.assertEqual((args.output / "result.json").read_bytes(), original)
                    self.assertFalse((args.output / "acknowledgement.json").exists())

    def test_worker_failure_before_commit_stops_control_and_publishes_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            error = ConnectionError("D0 ended early")
            with self.running_worker(temporary, protection_error=error) as state:
                args, thread, request, published, results, errors = state
                thread.join(1)
                self.assertEqual(results, [])
                self.assertIs(errors[0], error)
                self.assertEqual(json.loads((args.output / "result.json").read_text())["state"], "ERROR")
                self.assertFalse((args.output / "acknowledgement.json").exists())

    def test_ack_before_commit_is_rejected_and_duplicate_ack_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = args_for(temporary)
            path = Path(temporary) / "result.json"
            binding = {key: getattr(args, key) for key in
                       ("experiment_id", "stripe_id", "checkpoint_version")}
            ack = json.dumps({"op": "ack", **binding})
            self.assertEqual(worker._control_response(ack, path, binding)["reason"], "checkpoint_not_committed")
            worker._atomic_json(path, committed_result(args))
            original = path.read_bytes()
            first = worker._control_response(ack, path, binding)
            second = worker._control_response(ack, path, binding)
            self.assertEqual(first, second)
            self.assertEqual(first["acknowledgement"], "ACKNOWLEDGED")
            self.assertEqual(path.read_bytes(), original)

    def test_lost_ack_response_closes_control_without_changing_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = args_for(temporary)
            path = Path(temporary) / "result.json"
            worker._atomic_json(path, committed_result(args))
            original = path.read_bytes()
            binding = {key: getattr(args, key) for key in
                       ("experiment_id", "stripe_id", "checkpoint_version")}
            conn = mock.MagicMock()
            conn.__enter__.return_value = conn
            conn.recv.return_value = (json.dumps({"op": "ack", **binding}) + "\n").encode()
            conn.sendall.side_effect = BrokenPipeError("coordinator disconnected")
            server = mock.MagicMock()
            server.__enter__.return_value = server
            server.accept.return_value = (conn, ("127.0.0.1", 1))
            ready, stop, acknowledged = (threading.Event() for _ in range(3))
            with mock.patch.object(worker.socket, "socket", return_value=server):
                worker._control_server("127.0.0.1", 1, path, ready, stop,
                                       acknowledged, binding, .2)
            self.assertTrue(acknowledged.is_set())
            self.assertEqual(path.read_bytes(), original)
            server.__exit__.assert_called_once()
            conn.__exit__.assert_called_once()

    def test_control_server_serves_commit_and_rejects_wrong_ack_before_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = args_for(temporary)
            path = Path(temporary) / "result.json"
            worker._atomic_json(path, committed_result(args))
            binding = {key: getattr(args, key) for key in
                       ("experiment_id", "stripe_id", "checkpoint_version")}
            ack = json.dumps({"op": "ack", **binding}).encode() + b"\n"
            requests = [[b""], [b"result\n"], [b"not-json\n"]]
            requests.extend([(json.dumps({"op": "ack", **binding, key: "wrong"}) + "\n").encode()]
                            for key in binding)
            requests.append([ack[:10], ack[10:]])
            connections = []
            for chunks in requests:
                conn = mock.MagicMock()
                conn.__enter__.return_value = conn
                conn.recv.side_effect = chunks
                connections.append(conn)
            server = mock.MagicMock()
            server.__enter__.return_value = server
            server.accept.side_effect = [(conn, ("127.0.0.1", 1)) for conn in connections]
            ready, stop, acknowledged = (threading.Event() for _ in range(3))
            with mock.patch.object(worker.socket, "socket", return_value=server):
                worker._control_server("127.0.0.1", 1, path, ready, stop,
                                       acknowledged, binding, .2)
            replies = [json.loads(conn.sendall.call_args.args[0]) for conn in connections[1:]]
            self.assertEqual(replies[0]["state"], "COMMITTED")
            self.assertEqual(replies[0]["acknowledgement"], "WAITING")
            self.assertEqual(replies[1]["state"], "ERROR")
            self.assertTrue(all(reply["reason"] == "ack_identity_mismatch" for reply in replies[2:-1]))
            self.assertEqual(replies[-1]["acknowledgement"], "ACKNOWLEDGED")
            self.assertTrue(acknowledged.is_set())
            self.assertTrue(ready.is_set())
            server.__exit__.assert_called_once()
            for conn in connections:
                conn.__exit__.assert_called_once()

    def test_control_startup_failure_does_not_start_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = args_for(temporary)
            args.control_host = "127.0.0.1"
            with mock.patch.object(worker, "_control_server", side_effect=OSError("address in use")), \
                    mock.patch.object(worker, "_run_protection") as protection:
                with self.assertRaisesRegex(OSError, "address in use"):
                    worker.run(args)
            protection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
