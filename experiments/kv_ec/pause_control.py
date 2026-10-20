"""Bounded, metadata-only control service for one isolated TP4 pause experiment."""

import argparse
import json
import logging
import math
from pathlib import Path
import threading
import time
import uuid

from .cross_rank_control import CrossRankControlServer, request
from .checkpoint_protocol import recv_message, send_message


class PauseCoordinator:
    """One transaction per process. Terminal failures cannot be revived."""

    def __init__(self, transaction_id, *, timeout_s=30, clock=time.monotonic):
        if not transaction_id or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("invalid pause transaction/deadline")
        self.transaction_id = transaction_id
        self.epoch = str(uuid.uuid4())
        self.timeout_s = timeout_s
        self.clock = clock
        self.deadline = None
        self.state = "PREPARE"
        self.failure_reason = None
        self.observations = {}
        self.validated = set()
        self.released = set()
        self.armed = set()
        self.lock = threading.RLock()

    def status(self):
        return {
            "transaction_id": self.transaction_id, "epoch": self.epoch,
            "state": self.state, "failure_reason": self.failure_reason,
            "paused_ranks": sorted(self.observations),
            "validated_ranks": sorted(self.validated),
            "released_ranks": sorted(self.released), "armed_ranks": sorted(self.armed),
        }

    def dispatch(self, message):
        with self.lock:
            if (self.deadline is not None and self.clock() >= self.deadline
                    and self.state not in ("RESUME_AUTHORIZED", "FAILED")):
                self.state, self.failure_reason = "FAILED", "total_deadline_exceeded"
            if (message.get("epoch") != self.epoch
                    or message.get("transaction_id") != self.transaction_id):
                raise ValueError("pause_coordinator_identity_changed")
            op = message["op"]
            if self.state == "FAILED" or op == "status":
                return self.status()
            if op == "fail":
                self.state, self.failure_reason = "FAILED", str(message["reason"])
            elif op == "authorize":
                if self.state == "PAUSED":
                    self.state = "RESUME_REQUESTED"
                elif self.state not in (
                    "RESUME_REQUESTED", "RELEASE_AUTHORIZED", "ALL_RELEASED",
                    "RESUME_AUTHORIZED",
                ):
                    raise ValueError("cannot_authorize_before_all_ranks_paused")
            elif op == "paused":
                rank = message["tp_rank"]
                observation = message["observation"]
                if type(rank) is not int or rank not in range(4):
                    raise ValueError("invalid_rank")
                if rank in self.observations:
                    if self.observations[rank] != observation:
                        self.state, self.failure_reason = "FAILED", "duplicate_observation_mismatch"
                    return self.status()
                if self.state != "PREPARE":
                    raise ValueError("late_pause_observation")
                if not observation.get("event_complete"):
                    raise ValueError("cuda_event_incomplete")
                for key in ("request_hash", "request_version", "token_count",
                            "model_identifier", "model_sha256"):
                    if not observation.get(key):
                        raise ValueError(f"missing_observation:{key}")
                    if self.observations and observation[key] != next(
                            iter(self.observations.values()))[key]:
                        self.state, self.failure_reason = "FAILED", f"rank_identity_mismatch:{key}"
                        return self.status()
                if self.deadline is None:
                    self.deadline = self.clock() + self.timeout_s
                self.observations[rank] = observation
                if set(self.observations) == set(range(4)):
                    self.state = "PAUSED"
            else:
                stages = {
                    "validated": (self.validated, "RESUME_REQUESTED", "RELEASE_AUTHORIZED"),
                    "released": (self.released, "RELEASE_AUTHORIZED", "ALL_RELEASED"),
                    "armed": (self.armed, "ALL_RELEASED", "RESUME_AUTHORIZED"),
                }
                if op not in stages:
                    raise ValueError("unknown_pause_operation")
                ranks, required, target = stages[op]
                rank = message["tp_rank"]
                if type(rank) is not int or rank not in range(4) or rank not in self.observations:
                    raise ValueError("invalid_rank")
                if rank not in ranks:
                    if self.state != required:
                        raise ValueError(f"invalid_transition:{self.state}->{op}")
                    ranks.add(rank)
                    if ranks == set(range(4)):
                        self.state = target
            return self.status()


class PauseControlServer(CrossRankControlServer):
    """Reuse the tested Unix Socket framing/disconnect handling, not lease state."""

    def __init__(self, path, coordinator):
        super().__init__(path, coordinator)

    def _handle(self, conn):
        with conn:
            conn.settimeout(1)
            message = {}
            started = time.monotonic()
            try:
                message, payload = recv_message(conn)
                if payload:
                    raise ValueError("pause_control_must_not_carry_KV")
                response = {"state": "OK", "status": self.coordinator.dispatch(message)}
            except Exception as exc:
                response = {"state": "ERROR", "reason": str(exc)}
            try:
                send_message(conn, response, b"")
            except OSError:
                # An executed operation remains recorded. A lost response
                # cannot imply release/resume on the client.
                pass
            logging.getLogger(__name__).info(
                "KV_EC_PAUSE_CONTROL %s",
                json.dumps({"op": message.get("op"), "tp_rank": message.get("tp_rank"),
                            "monotonic": started,
                            "elapsed_ms": (time.monotonic() - started) * 1000,
                            "response": response}),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument("--transaction-id", required=True)
    serve.add_argument("--timeout-s", type=float, default=30)
    serve.add_argument("--identity-output", type=Path, required=True)
    for op in ("status", "authorize", "fail"):
        client = sub.add_parser(op)
        client.add_argument("--socket", type=Path, required=True)
        client.add_argument("--identity", type=Path, required=True)
        if op == "fail":
            client.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        logging.basicConfig(level=logging.INFO)
        coordinator = PauseCoordinator(args.transaction_id, timeout_s=args.timeout_s)
        with args.identity_output.open("x") as stream:
            json.dump({"epoch": coordinator.epoch,
                       "transaction_id": coordinator.transaction_id}, stream)
        PauseControlServer(args.socket, coordinator).serve_forever()
    else:
        message = dict(json.loads(args.identity.read_text()), op=args.command)
        if args.command == "fail":
            message["reason"] = args.reason
        response = request(args.socket, message, timeout=1)
        print(json.dumps(response, indent=2))
        if response.get("state") != "OK" or response["status"]["state"] == "FAILED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
