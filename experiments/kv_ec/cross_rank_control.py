"""Independent Unix-socket control plane for the CPU cross-rank lease protocol.

The channel carries only lease metadata and state transitions. It never carries
KV bytes, invokes NCCL, or performs GPU synchronization. A live Scheduler
integration must create/validate CUDA events locally before sending the
``cuda_completed`` and ``validated`` messages.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import socket
import threading
import os
import uuid
import logging
import time

from .checkpoint_protocol import recv_message, send_message
from .cross_rank_lease import CrossRankLeaseCoordinator, CrossRankStatus
from .recovery_lease import LeaseRejected, RankLeaseObservation

logger = logging.getLogger(__name__)


def _request_tp_rank(request: dict):
    if request.get("tp_rank") is not None:
        return request.get("tp_rank")
    observation = request.get("observation")
    if isinstance(observation, dict):
        return observation.get("tp_rank")
    return None


_EPOCH_CHECK_OPERATIONS = frozenset({
    "hello",
    "prepare",
    "prepare_rank",
    "local_leased",
    "cuda_completed",
    "validated",
    "authorize_release",
    "local_released",
    "fail",
    "abort_ack",
})


def _status(status: CrossRankStatus) -> dict:
    data = asdict(status)
    data["state"] = status.state.value
    return data


def _observation(data: dict) -> RankLeaseObservation:
    fields = (
        "tp_rank", "tp_size", "request_id", "request_version", "req_pool_idx",
        "req_generation", "physical_slots", "slot_generations",
        "checkpoint_version", "manager_epoch", "token_start", "token_count",
    )
    missing = [key for key in fields if key not in data]
    if missing:
        raise LeaseRejected("missing_observation_fields")
    return RankLeaseObservation(
        tp_rank=int(data["tp_rank"]), tp_size=int(data["tp_size"]),
        request_id=str(data["request_id"]),
        request_version=str(data["request_version"]),
        req_pool_idx=int(data["req_pool_idx"]),
        req_generation=int(data["req_generation"]),
        physical_slots=tuple(int(x) for x in data["physical_slots"]),
        slot_generations=tuple(int(x) for x in data["slot_generations"]),
        checkpoint_version=str(data["checkpoint_version"]),
        manager_epoch=str(data["manager_epoch"]),
        token_start=int(data["token_start"]),
        token_count=int(data["token_count"]),
    )


class CrossRankControlServer:
    """Threaded, bounded-metadata control server independent of inference."""

    def __init__(
        self,
        path: str | Path,
        coordinator: CrossRankLeaseCoordinator,
        *,
        manager_epoch_provider=None,
    ):
        self.path = Path(path)
        self.coordinator = coordinator
        self.manager_epoch_provider = manager_epoch_provider
        self.control_epoch = str(uuid.uuid4())
        self._stop = threading.Event()
        self.ready = threading.Event()
        self._dispatch_lock = threading.RLock()
        self._server: socket.socket | None = None

    def serve_forever(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to replace existing control socket path: {self.path}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            self._server = server
            server.bind(str(self.path))
            os.chmod(self.path, 0o600)
            server.listen(16)
            server.settimeout(0.2)
            self.ready.set()
            try:
                while not self._stop.is_set():
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        if self._stop.is_set():
                            break
                        raise
                    threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            finally:
                self.path.unlink(missing_ok=True)

    def _handle(self, conn: socket.socket):
        with conn:
            conn.settimeout(2.0)
            request = {}
            response = {"state": "ERROR", "reason": "control_request_failed"}
            started = time.monotonic()
            lock_wait_ms = 0.0
            epoch_check_ms = 0.0
            dispatch_ms = 0.0
            response_send_ms = 0.0
            failure_reason = None
            try:
                request, payload = recv_message(conn)
                if payload:
                    raise LeaseRejected("payload_for_control_message")
                op = request.get("op")
                tx = request.get("transaction_id")
                current_epoch = self.coordinator.manager_epoch
                if op in _EPOCH_CHECK_OPERATIONS:
                    epoch_started = time.monotonic()
                    current_epoch = self._check_manager_epoch()
                    epoch_check_ms += (time.monotonic() - epoch_started) * 1000
                    if op != "hello" and request.get("manager_epoch") not in (None, current_epoch):
                        raise LeaseRejected("request_manager_epoch_mismatch")
                dispatch_started = time.monotonic()
                lock_started = time.monotonic()
                with self._dispatch_lock:
                    lock_wait_ms = (time.monotonic() - lock_started) * 1000
                    response = self._dispatch(request, check_epoch=False)
                dispatch_ms = (time.monotonic() - dispatch_started) * 1000
            except Exception as exc:
                failure_reason = str(exc) or type(exc).__name__
                response = {"state": "ERROR", "reason": str(exc) or type(exc).__name__}
                if request.get("transaction_id"):
                    self._fail_transaction(request["transaction_id"], failure_reason)
            try:
                send_started = time.monotonic()
                send_message(conn, response, b"")
                response_send_ms = (time.monotonic() - send_started) * 1000
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                failure_reason = failure_reason or f"client_closed:{type(exc).__name__}"
                logger.info(
                    "KV_EC_CONTROL client closed before response op=%s tx=%s tp_rank=%s reason=%s",
                    request.get("op"), request.get("transaction_id"),
                    _request_tp_rank(request), failure_reason,
                )
            finally:
                logger.info(
                    "KV_EC_CONTROL_TIMING %s",
                    json.dumps({
                        "op": request.get("op"),
                        "transaction_id": request.get("transaction_id"),
                        "tp_rank": _request_tp_rank(request),
                        "lock_wait_ms": lock_wait_ms,
                        "epoch_check_ms": epoch_check_ms,
                        "dispatch_ms": dispatch_ms,
                        "state_transition_ms": max(0.0, dispatch_ms - lock_wait_ms),
                        "response_send_ms": response_send_ms,
                        "elapsed_ms": (time.monotonic() - started) * 1000,
                        "failure_reason": failure_reason,
                    }, sort_keys=True),
                )

    def _check_manager_epoch(self):
        if self.manager_epoch_provider is None:
            return self.coordinator.manager_epoch
        try:
            current_epoch = self.manager_epoch_provider()
        except Exception as exc:
            raise LeaseRejected(f"manager_unavailable:{type(exc).__name__}") from exc
        if current_epoch != self.coordinator.manager_epoch:
            raise LeaseRejected("manager_epoch_changed")
        return current_epoch

    def _fail_transaction(self, transaction_id, reason):
        if not transaction_id:
            return
        try:
            with self._dispatch_lock:
                self.coordinator.fail(transaction_id, reason)
        except LeaseRejected:
            pass

    def _dispatch(self, request: dict, *, check_epoch: bool = True) -> dict:
        if check_epoch:
            self._check_manager_epoch()
        op = request.get("op")
        if op == "hello":
            current_epoch = self.coordinator.manager_epoch
            if current_epoch != self.coordinator.manager_epoch:
                raise LeaseRejected("manager_epoch_changed")
            return {
                "state": "OK",
                "control_epoch": self.control_epoch,
                "manager_epoch": current_epoch,
            }
        tx = request.get("transaction_id")
        if not isinstance(tx, str) or not tx:
            raise LeaseRejected("transaction_id_required")
        if op == "prepare_rank":
            result = self.coordinator.prepare_rank(
                tx, _observation(request.get("observation", {})),
                checkpoint_version=str(request.get("checkpoint_version", "")),
            )
        elif op == "prepare":
            raw = request.get("observations")
            if not isinstance(raw, list):
                raise LeaseRejected("observations_required")
            result = self.coordinator.prepare(
                tx, [_observation(item) for item in raw],
                checkpoint_version=str(request.get("checkpoint_version", "")),
            )
        elif op == "local_leased":
            result = self.coordinator.mark_local_leased(tx, int(request["tp_rank"]))
        elif op == "cuda_completed":
            result = self.coordinator.mark_cuda_completed(tx, int(request["tp_rank"]))
        elif op == "validated":
            result = self.coordinator.mark_validated(tx, int(request["tp_rank"]))
        elif op == "authorize_release":
            result = self.coordinator.authorize_release(tx)
        elif op == "local_released":
            result = self.coordinator.mark_local_released(tx, int(request["tp_rank"]))
        elif op == "fail":
            result = self.coordinator.fail(tx, str(request.get("reason", "rank_failure")))
        elif op == "abort_ack":
            result = self.coordinator.acknowledge_abort(tx, int(request["tp_rank"]))
        elif op == "status":
            result = self.coordinator.status(tx)
        else:
            raise LeaseRejected("unknown_control_operation")
        if op != "status":
            logger.info(
                "KV_EC_CONTROL %s",
                json.dumps({
                    "transaction_id": tx, "op": op,
                    "tp_rank": request.get("tp_rank"),
                    "state": result.state.value,
                    "released_ranks": result.released_ranks,
                    "control_epoch": self.control_epoch,
                    "manager_epoch": self.coordinator.manager_epoch,
                }, sort_keys=True),
            )
        return {
            "state": "OK",
            "control_epoch": self.control_epoch,
            "manager_epoch": self.coordinator.manager_epoch,
            "status": _status(result),
        }

    def stop(self):
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass


def request(path: str | Path, message: dict, *, timeout: float = 2.0) -> dict:
    """Send one bounded control request; no retry hides a lost coordinator."""
    started = time.monotonic()
    detail = {
        "operation": message.get("op"),
        "transaction_id": message.get("transaction_id"),
        "tp_rank": _request_tp_rank(message),
        "timeout_s": timeout,
        "stage": "connect",
        "connect_ms": None,
        "send_ms": None,
        "response_wait_ms": None,
        "elapsed_ms": None,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        try:
            connect_started = time.monotonic()
            client.connect(str(path))
            detail["connect_ms"] = (time.monotonic() - connect_started) * 1000
            detail["stage"] = "send"
            send_started = time.monotonic()
            send_message(client, dict(message), b"")
            detail["send_ms"] = (time.monotonic() - send_started) * 1000
            detail["stage"] = "recv"
            response_started = time.monotonic()
            response, payload = recv_message(client)
            detail["response_wait_ms"] = (time.monotonic() - response_started) * 1000
        except Exception as exc:
            detail["elapsed_ms"] = (time.monotonic() - started) * 1000
            try:
                exc.control_ipc_detail = dict(detail)
            except Exception:
                pass
            raise
        if payload:
            raise LeaseRejected("unexpected_control_payload")
        detail["elapsed_ms"] = (time.monotonic() - started) * 1000
        response["_control_ipc_detail"] = detail
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--parity-socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .parity_manager import manager_hello

    manager_epoch = manager_hello(args.parity_socket)
    epoch_provider = lambda: manager_hello(args.parity_socket)
    CrossRankControlServer(
        args.socket,
        CrossRankLeaseCoordinator(
            timeout_s=args.timeout, manager_epoch=manager_epoch
        ),
        manager_epoch_provider=epoch_provider,
    ).serve_forever()


if __name__ == "__main__":
    main()
