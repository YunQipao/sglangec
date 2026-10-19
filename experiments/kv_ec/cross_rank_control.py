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

from .checkpoint_protocol import recv_message, send_message
from .cross_rank_lease import CrossRankLeaseCoordinator, CrossRankStatus
from .recovery_lease import LeaseRejected, RankLeaseObservation


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

    def __init__(self, path: str | Path, coordinator: CrossRankLeaseCoordinator):
        self.path = Path(path)
        self.coordinator = coordinator
        self._stop = threading.Event()
        self._server: socket.socket | None = None

    def serve_forever(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            self._server = server
            server.bind(str(self.path))
            server.listen(16)
            server.settimeout(0.2)
            try:
                while not self._stop.is_set():
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            finally:
                self.path.unlink(missing_ok=True)

    def _handle(self, conn: socket.socket):
        with conn:
            conn.settimeout(2.0)
            try:
                request, payload = recv_message(conn)
                if payload:
                    raise LeaseRejected("payload_for_control_message")
                response = self._dispatch(request)
            except Exception as exc:
                response = {"state": "ERROR", "reason": str(exc) or type(exc).__name__}
            send_message(conn, response, b"")

    def _dispatch(self, request: dict) -> dict:
        op = request.get("op")
        tx = request.get("transaction_id")
        if not isinstance(tx, str) or not tx:
            raise LeaseRejected("transaction_id_required")
        if op == "prepare":
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
        elif op == "release_global":
            result = self.coordinator.release_global(tx)
        elif op == "fail":
            result = self.coordinator.fail(tx, str(request.get("reason", "rank_failure")))
        elif op == "abort_ack":
            result = self.coordinator.acknowledge_abort(tx, int(request["tp_rank"]))
        elif op == "status":
            result = self.coordinator.status(tx)
        else:
            raise LeaseRejected("unknown_control_operation")
        return {"state": "OK", "status": _status(result)}

    def stop(self):
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass


def request(path: str | Path, message: dict, *, timeout: float = 2.0) -> dict:
    """Send one bounded control request; no retry hides a lost coordinator."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        send_message(client, dict(message), b"")
        response, payload = recv_message(client)
        if payload:
            raise LeaseRejected("unexpected_control_payload")
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--manager-epoch", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    CrossRankControlServer(
        args.socket,
        CrossRankLeaseCoordinator(
            timeout_s=args.timeout, manager_epoch=args.manager_epoch
        ),
    ).serve_forever()


if __name__ == "__main__":
    main()
