import unittest
import tempfile
import threading
import time
import socket
import traceback
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

from experiments.kv_ec.checkpoint_protocol import send_message
from experiments.kv_ec.cross_rank_control import CrossRankControlServer, request
from experiments.kv_ec.cross_rank_lease import CrossRankLeaseCoordinator
from experiments.kv_ec.recovery_lease import LeaseRejected
from experiments.kv_ec.test_cross_rank_lease import observations
from experiments.kv_ec.cross_rank_worker import coordinate_local_lease
from experiments.kv_ec import cross_rank_worker


class CrossRankControlTests(unittest.TestCase):
    def setUp(self):
        self.server = CrossRankControlServer(
            "/tmp/kv-ec-control-test.sock",
            CrossRankLeaseCoordinator(manager_epoch="epoch-1"),
        )

    def test_control_dispatches_all_rank_protocol(self):
        tx = "control-tx"
        raw = [item.__dict__ for item in observations()]
        response = self.server._dispatch({
            "op": "prepare", "transaction_id": tx,
            "observations": raw, "checkpoint_version": "cp-v1",
        })
        self.assertEqual(response["status"]["state"], "PREPARE")
        for rank in range(4):
            self.assertEqual(
                self.server._dispatch({
                    "op": "local_leased", "transaction_id": tx, "tp_rank": rank,
                })["status"]["state"],
                "ALL_RANKS_READY" if rank == 3 else "LOCAL_LEASED",
            )
        for rank in range(4):
            self.server._dispatch({
                "op": "cuda_completed", "transaction_id": tx, "tp_rank": rank,
            })
        for rank in range(4):
            result = self.server._dispatch({
                "op": "validated", "transaction_id": tx, "tp_rank": rank,
            })
        self.assertEqual(result["status"]["state"], "ALL_RANKS_VALIDATED")
        self.assertEqual(
            self.server._dispatch({"op": "authorize_release", "transaction_id": tx})
            ["status"]["state"],
            "RELEASE_AUTHORIZED",
        )
        for rank in range(3):
            self.assertEqual(
                self.server._dispatch({
                    "op": "local_released", "transaction_id": tx, "tp_rank": rank,
                })["status"]["state"],
                "RELEASE_AUTHORIZED",
            )
        self.assertEqual(
            self.server._dispatch({
                "op": "local_released", "transaction_id": tx, "tp_rank": 3,
            })["status"]["state"],
            "GLOBAL_RELEASED",
        )

    def test_worker_failure_cleans_locally_without_global_success(self):
        for failure in ("cuda_incomplete", "generation_changed", "release_failed"):
            with self.subTest(failure=failure):
                server = CrossRankControlServer(
                    "/tmp/unused.sock",
                    CrossRankLeaseCoordinator(manager_epoch="epoch-1"),
                )
                lock = threading.Lock()
                leases, errors = [], []
                class LocalLease:
                    def __init__(self, rank):
                        self.rank = rank
                        self.metadata = {
                            "tp_rank": rank, "tp_size": 4,
                            "request_version": "v1", "manager_epoch": "epoch-1",
                            "checkpoint_version": "cp-v1", "req_generation": 9,
                            "token_start": 0, "token_count": 256,
                            "slot_generations_before": [1, 1],
                        }
                        self.cleaned = False
                    def validate_completion(self, *args, **kwargs):
                        if self.rank == 2 and failure == "generation_changed":
                            raise LeaseRejected("slot_generation_changed")
                        return {"event_recorded": True, "event_complete": not (
                            self.rank == 2 and failure == "cuda_incomplete"
                        )}
                    def release_local(self):
                        if self.rank == 2 and failure == "release_failed":
                            raise LeaseRejected("allocator_release_failed")
                        self.cleaned = True
                        self.metadata["state"] = "LOCAL_RELEASED"
                        return dict(self.metadata)
                    def abort(self):
                        self.cleaned = True
                        return {"state": "ABORTED_RELEASED"}
                def dispatch(path, message, *, timeout):
                    with lock:
                        return server._dispatch(message)
                def run(rank):
                    lease = LocalLease(rank)
                    leases.append(lease)
                    try:
                        coordinate_local_lease(
                            lease, SimpleNamespace(),
                            SimpleNamespace(rid="req", req_pool_idx=rank + 1),
                            (rank * 100 + 1, rank * 100 + 2),
                            control_socket="/tmp/unused.sock", timeout_s=2,
                            checkpoint_version="cp-v1",
                        )
                    except Exception as exc:
                        errors.append(exc)
                with patch.object(cross_rank_worker, "request", side_effect=dispatch):
                    workers = [threading.Thread(target=run, args=(r,)) for r in range(4)]
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        worker.join(4)
                    self.assertFalse(any(w.is_alive() for w in workers))
                self.assertEqual(len(errors), 4)
                self.assertTrue(all(lease.cleaned for lease in leases))
                tx = leases[0].metadata["transaction_id"]
                status = server.coordinator.status(tx)
                self.assertEqual(status.state.value, "FAILED")
                self.assertNotEqual(status.released_ranks, (0, 1, 2, 3))
                self.assertTrue(all(not l.metadata["global_release_confirmed"] for l in leases))

    def test_socket_handler_dispatches_without_real_socket(self):
        from unittest.mock import MagicMock
        conn = MagicMock()
        with patch("experiments.kv_ec.cross_rank_control.recv_message",
                   return_value=({"op": "hello"}, b"")), \
             patch("experiments.kv_ec.cross_rank_control.send_message") as send:
            self.server._handle(conn)
        self.assertEqual(send.call_args.args[1]["state"], "OK")
        self.assertEqual(send.call_args.args[1]["control_epoch"], self.server.control_epoch)

    def test_duplicate_and_unknown_messages_are_errors(self):
        with self.assertRaises(LeaseRejected):
            self.server._dispatch({"op": "status", "transaction_id": "missing"})
        with self.assertRaises(LeaseRejected):
            self.server._dispatch({"op": "unknown", "transaction_id": "x"})

    def test_live_manager_epoch_change_rejects_control_messages(self):
        epoch = ["epoch-1"]
        server = CrossRankControlServer(
            "/tmp/unused.sock",
            CrossRankLeaseCoordinator(manager_epoch="epoch-1"),
            manager_epoch_provider=lambda: epoch[0],
        )
        self.assertEqual(server._dispatch({"op": "hello"})["manager_epoch"], "epoch-1")
        epoch[0] = "epoch-2"
        with self.assertRaisesRegex(LeaseRejected, "manager_epoch_changed"):
            server._dispatch({"op": "status", "transaction_id": "missing"})

    def test_real_socket_parallel_epoch_checks_do_not_hold_dispatch_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.sock"
            try:
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.bind(str(path))
            except PermissionError:
                self.skipTest("sandbox disallows Unix domain sockets")
            path.unlink()
            calls = []
            calls_lock = threading.Lock()
            epoch_checks = threading.Barrier(4)

            def delayed_epoch():
                with calls_lock:
                    calls.append(time.monotonic())
                # A dispatch lock around the epoch check would prevent all
                # four handlers from reaching this barrier.
                epoch_checks.wait(timeout=5)
                return "epoch-1"

            server = CrossRankControlServer(
                path,
                CrossRankLeaseCoordinator(manager_epoch="epoch-1"),
                manager_epoch_provider=delayed_epoch,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.assertTrue(server.ready.wait(2))
            raw = observations()

            def send_prepare(rank):
                return request(
                    path,
                    {
                        "op": "prepare_rank",
                        "transaction_id": "parallel-delay",
                        "checkpoint_version": "cp-v1",
                        "observation": raw[rank].__dict__,
                    },
                    timeout=10.0,
                )

            try:
                with ThreadPoolExecutor(max_workers=4) as workers:
                    futures = [workers.submit(send_prepare, rank) for rank in range(4)]
                    results = [future.result(timeout=15) for future in futures]
                self.assertTrue(all(result["state"] == "OK" for result in results))
                for rank, result in enumerate(results):
                    self.assertIn(rank, result["status"]["prepared_ranks"])
                # Responses are snapshots, indexed by rank rather than
                # dispatch/completion order. Query after every worker joined.
                final = request(path, {"op": "status", "transaction_id": "parallel-delay"}, timeout=2)
                self.assertEqual(final["state"], "OK")
                self.assertEqual(final["status"]["prepared_ranks"], [0, 1, 2, 3])
                self.assertEqual(len(calls), 4)
            finally:
                server.stop()
                thread.join(2)
                self.assertFalse(thread.is_alive())

    def test_parallel_epoch_checks_allow_rank_three_to_finish_before_rank_two(self):
        # Exercise the real handler without requiring permission to bind a
        # Unix socket, and force the ordering that exposed the flaky assertion.
        epoch_checks = threading.Barrier(4)
        earlier_replies = threading.Event()
        local = threading.local()
        results = {}
        results_lock = threading.Lock()

        def delayed_epoch():
            epoch_checks.wait(timeout=5)
            if local.rank == 2 and not earlier_replies.wait(5):
                raise TimeoutError("other_rank_responses_missing")
            return "epoch-1"

        server = CrossRankControlServer(
            "/tmp/unused.sock", CrossRankLeaseCoordinator(manager_epoch="epoch-1"),
            manager_epoch_provider=delayed_epoch,
        )
        raw = observations()

        def receive(conn):
            local.rank = conn.rank
            return {
                "op": "prepare_rank", "transaction_id": "reordered-prepare",
                "checkpoint_version": "cp-v1", "observation": raw[conn.rank].__dict__,
            }, b""

        def send(conn, response, payload):
            with results_lock:
                results[conn.rank] = response
                if {0, 1, 3}.issubset(results):
                    earlier_replies.set()

        conns = [MagicMock(rank=rank) for rank in range(4)]
        with patch("experiments.kv_ec.cross_rank_control.recv_message", side_effect=receive), patch(
                "experiments.kv_ec.cross_rank_control.send_message", side_effect=send):
            with ThreadPoolExecutor(max_workers=4) as workers:
                futures = [workers.submit(server._handle, conn) for conn in conns]
                for future in futures:
                    future.result(timeout=15)
        self.assertTrue(all(result["state"] == "OK" for result in results.values()), results)
        self.assertEqual(sorted(results), [0, 1, 2, 3])
        self.assertNotIn(2, results[3]["status"]["prepared_ranks"])
        self.assertEqual(results[2]["status"]["prepared_ranks"], (0, 1, 2, 3))
        final = server._dispatch(
            {"op": "status", "transaction_id": "reordered-prepare"}, check_epoch=False)
        self.assertEqual(final["status"]["prepared_ranks"], (0, 1, 2, 3))

    def test_real_socket_client_disconnect_has_no_uncaught_handler_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.sock"
            try:
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.bind(str(path))
            except PermissionError:
                self.skipTest("sandbox disallows Unix domain sockets")
            path.unlink()
            server = CrossRankControlServer(
                path, CrossRankLeaseCoordinator(manager_epoch="epoch-1")
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.assertTrue(server.ready.wait(2))
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.connect(str(path))
                send_message(client, {"op": "hello"}, b"")
                client.close()
                time.sleep(0.1)
                self.assertTrue(thread.is_alive())
            finally:
                try:
                    client.close()
                except OSError:
                    pass
                server.stop()
                thread.join(2)

    def test_real_unix_socket_four_rank_worker_handshake(self):
        import socket

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.sock"
            try:
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.bind(str(path))
            except PermissionError:
                self.skipTest("sandbox disallows Unix domain sockets")
            path.unlink()
            server = CrossRankControlServer(
                path, CrossRankLeaseCoordinator(manager_epoch="epoch-1")
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.assertTrue(server.ready.wait(2))

            class LocalLease:
                def __init__(self, rank):
                    self.metadata = {
                        "transaction_id": "live-ipc-tx",
                        "tp_rank": rank,
                        "tp_size": 4,
                        "request_version": "req-v1",
                        "manager_epoch": "epoch-1",
                        "checkpoint_version": "cp-v1",
                        "req_generation": 9,
                        "token_start": 0,
                        "token_count": 256,
                        "slot_generations_before": [2, 3],
                    }
                    self.released = False

                def validate_completion(self, scheduler, req, *, slots):
                    self.metadata["event_recorded"] = True
                    self.metadata["event_complete"] = True
                    return self.metadata

                def release_local(self):
                    self.released = True
                    self.metadata["state"] = "LOCAL_RELEASED"
                    return dict(self.metadata)

            results = [None] * 4
            errors = []
            failures = []
            failures_lock = threading.Lock()

            def worker(rank):
                lease = LocalLease(rank)
                worker_started = time.monotonic()
                try:
                    results[rank] = coordinate_local_lease(
                        lease,
                        SimpleNamespace(),
                        SimpleNamespace(rid="req", req_pool_idx=rank + 1),
                        (rank * 100 + 1, rank * 100 + 2),
                        control_socket=path,
                        timeout_s=5,
                        checkpoint_version="cp-v1",
                    )
                except Exception as exc:
                    with failures_lock:
                        failures.append({
                            "monotonic_ns": time.monotonic_ns(),
                            "rank": rank,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "transaction_id": lease.metadata.get("transaction_id"),
                            "control_failure_detail": lease.metadata.get(
                                "control_failure_detail"
                            ),
                            "control_trace": lease.metadata.get("control_trace", []),
                            "elapsed_ms": (time.monotonic() - worker_started) * 1000,
                            "traceback": traceback.format_exc(),
                        })
                    errors.append(exc)

            workers = [threading.Thread(target=worker, args=(rank,)) for rank in range(4)]
            try:
                for item in workers:
                    item.start()
                for item in workers:
                    item.join(7)
                self.assertFalse(any(item.is_alive() for item in workers))
                if failures:
                    failures.sort(key=lambda item: item["monotonic_ns"])
                    self.fail(
                        "first real-socket worker failure:\n"
                        + json.dumps(failures[0], indent=2, sort_keys=True)
                    )
                self.assertTrue(all(item["state"] == "LOCAL_RELEASED" for item in results))
                self.assertTrue(all(item["global_release_confirmed"] for item in results))
                self.assertTrue(all(item["coordinator_state"] == "GLOBAL_RELEASED" for item in results))
            finally:
                server.stop()
                thread.join(2)

    def test_four_worker_client_calls_wait_for_actual_release_acks(self):
        server = CrossRankControlServer(
            "/tmp/unused-control.sock",
            CrossRankLeaseCoordinator(manager_epoch="epoch-1"),
        )
        dispatch_lock = threading.Lock()

        class LocalLease:
            def __init__(self, rank):
                self.metadata = {
                    "transaction_id": "old-local-id",
                    "tp_rank": rank,
                    "tp_size": 4,
                    "request_version": "req-v1",
                    "manager_epoch": "epoch-1",
                    "checkpoint_version": "cp-v1",
                    "req_generation": 9,
                    "token_start": 0,
                    "token_count": 256,
                    "slot_generations_before": [2, 3],
                }
                self.released = False

            def validate_completion(self, scheduler, req, *, slots):
                self.metadata["event_recorded"] = True
                self.metadata["event_complete"] = True
                return self.metadata

            def release_local(self):
                self.released = True
                self.metadata["state"] = "LOCAL_RELEASED"
                return dict(self.metadata)

        def fake_request(path, message, *, timeout):
            with dispatch_lock:
                return server._dispatch(message)

        results = [None] * 4
        errors = []
        with patch.object(cross_rank_worker, "request", side_effect=fake_request):
            workers = []
            for rank in range(4):
                def run(r=rank):
                    try:
                        results[r] = coordinate_local_lease(
                            LocalLease(r), SimpleNamespace(),
                            SimpleNamespace(rid="req", req_pool_idx=r + 1),
                            (r * 100 + 1, r * 100 + 2),
                            control_socket="/tmp/unused-control.sock",
                            timeout_s=5,
                            checkpoint_version="cp-v1",
                        )
                    except Exception as exc:
                        errors.append(exc)
                worker = threading.Thread(target=run)
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join(7)
            self.assertFalse(any(worker.is_alive() for worker in workers))
        request_hash = __import__("hashlib").sha256(b"req").hexdigest()
        version_hash = __import__("hashlib").sha256(b"req-v1").hexdigest()[:16]
        tx = (
            f"{server.control_epoch}:cp-v1:{request_hash}:"
            f"{version_hash}:0:256"
        )
        final_state = server.coordinator.status(tx)
        self.assertEqual(errors, [], f"state={final_state}")
        self.assertTrue(all(result["global_release_confirmed"] for result in results))
        self.assertEqual(
            {result["transaction_id"] for result in results},
            {results[0]["transaction_id"]},
        )


if __name__ == "__main__":
    unittest.main()
