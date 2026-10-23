"""Scheduler-side bounded client for non-destructive TP=4 lease coordination."""

from __future__ import annotations

import hashlib
import os
import time

from .cross_rank_control import request
from .recovery_lease import RankLeaseObservation, LeaseRejected


def _ok(response: dict) -> dict:
    if response.get("state") != "OK" or "status" not in response:
        raise LeaseRejected(
            f"control_request_failed:{response.get('reason', response.get('state'))}"
        )
    return response["status"]


def _poll(call, tx, expected, deadline):
    ordered = (
        "PREPARE",
        "LOCAL_LEASED",
        "ALL_RANKS_READY",
        "CUDA_COMPLETED",
        "ALL_RANKS_VALIDATED",
        "RELEASE_AUTHORIZED",
        "GLOBAL_RELEASED",
    )
    while time.monotonic() < deadline:
        status = call({"op": "status", "transaction_id": tx})
        if status["state"] in ("FAILED", "EXPIRED"):
            raise LeaseRejected(f"cross_rank_{status['state'].lower()}:{status.get('failure_reason')}")
        if (
            status["state"] in ordered
            and ordered.index(status["state"]) >= ordered.index(expected)
        ):
            return status
        time.sleep(0.05)
    raise TimeoutError(f"cross_rank_wait_timeout:{expected}")


def coordinate_local_lease(
    lease,
    scheduler,
    req,
    slots,
    *,
    control_socket,
    timeout_s: float,
    checkpoint_version: str,
) -> dict:
    """Keep a real local allocator Lease until every Rank has released it.

    The caller invokes this after local lease creation and snapshot copying,
    before the prefill result processor resumes. Every socket operation is
    bounded; no NCCL collective is entered by this helper.
    """
    if not control_socket or timeout_s <= 0:
        raise ValueError("enabled cross-rank coordination requires socket and timeout")
    started = time.monotonic()
    deadline = started + timeout_s
    ipc_timeout_s = float(os.environ.get("SGLANG_KV_EC_CONTROL_IPC_TIMEOUT_S", "2.0"))
    if ipc_timeout_s <= 0:
        raise ValueError("control IPC timeout must be positive")
    meta = lease.metadata
    rank = int(meta["tp_rank"])
    tx = None
    current_operation = None
    trace = meta.setdefault("control_trace", [])

    def remaining_ms():
        return max(0.0, (deadline - time.monotonic()) * 1000)

    def record(operation, phase_started, *, error=None):
        item = {
            "exception_type": type(error).__name__ if error else None,
            "exception_message": str(error) if error else None,
            "control_operation": operation,
            "transaction_id": tx,
            "tp_rank": rank,
            "elapsed_ms": (time.monotonic() - phase_started) * 1000,
            "remaining_deadline_ms": remaining_ms(),
        }
        trace.append(item)
        return item

    def request_with_trace(message):
        nonlocal current_operation
        current_operation = message.get("op")
        phase_started = time.monotonic()
        timeout = min(ipc_timeout_s, max(0.001, deadline - time.monotonic()))
        try:
            response = request(control_socket, message, timeout=timeout)
            item = record(current_operation, phase_started)
            if "_control_ipc_detail" in response:
                item["control_ipc_detail"] = response["_control_ipc_detail"]
            return response
        except Exception as exc:
            item = record(current_operation, phase_started, error=exc)
            if hasattr(exc, "control_ipc_detail"):
                item["control_ipc_detail"] = exc.control_ipc_detail
            raise

    if checkpoint_version != meta["checkpoint_version"]:
        raise LeaseRejected("cross_rank_checkpoint_version_mismatch")
    hello = request_with_trace({"op": "hello"})
    if hello.get("state") != "OK":
        raise LeaseRejected("cross_rank_control_unavailable")
    control_epoch = hello.get("control_epoch")
    manager_epoch = hello.get("manager_epoch")
    if not control_epoch or manager_epoch != meta["manager_epoch"]:
        raise LeaseRejected("cross_rank_epoch_mismatch")
    request_hash = hashlib.sha256(str(req.rid).encode()).hexdigest()
    # Each captured token range is an independent lease transaction. In
    # particular, three chunks of one Prefill must not reuse a terminal tx ID.
    token_start = int(meta["token_start"])
    token_count = int(meta["token_count"])
    request_version = str(meta["request_version"])
    version_hash = hashlib.sha256(request_version.encode()).hexdigest()[:16]
    tx = (
        f"{control_epoch}:{checkpoint_version}:{request_hash}:"
        f"{version_hash}:{token_start}:{token_count}"
    )
    meta["transaction_id"] = tx

    def call(message):
        message = dict(message, manager_epoch=manager_epoch)
        response = request_with_trace(message)
        status = _ok(response)
        if (
            response.get("control_epoch") != control_epoch
            or response.get("manager_epoch") != manager_epoch
        ):
            raise LeaseRejected(
                "cross_rank_epoch_mismatch:"
                f"expected_control={control_epoch},"
                f"actual_control={response.get('control_epoch')},"
                f"expected_manager={manager_epoch},"
                f"actual_manager={response.get('manager_epoch')}"
            )
        return status

    observation = RankLeaseObservation(
        tp_rank=rank,
        tp_size=int(meta["tp_size"]),
        request_id=request_hash,
        request_version=str(meta["request_version"]),
        req_pool_idx=int(req.req_pool_idx),
        req_generation=int(meta["req_generation"]),
        physical_slots=tuple(int(slot) for slot in slots),
        slot_generations=tuple(int(g) for g in meta["slot_generations_before"]),
        checkpoint_version=checkpoint_version,
        manager_epoch=str(meta["manager_epoch"]),
        token_start=int(meta["token_start"]),
        token_count=int(meta["token_count"]),
    )
    try:
        prepared = call({
            "op": "prepare_rank", "transaction_id": tx,
            "checkpoint_version": checkpoint_version,
            "observation": observation.__dict__,
        })
        while set(prepared["prepared_ranks"]) != {0, 1, 2, 3}:
            if time.monotonic() >= deadline:
                raise TimeoutError("cross_rank_prepare_timeout")
            time.sleep(0.05)
            prepared = call({"op": "status", "transaction_id": tx})
            if prepared["state"] in ("FAILED", "EXPIRED"):
                raise LeaseRejected(f"cross_rank_{prepared['state'].lower()}")

        call({
            "op": "local_leased", "transaction_id": tx, "tp_rank": rank,
        })
        _poll(call, tx, "ALL_RANKS_READY", deadline)

        validated = lease.validate_completion(scheduler, req, slots=tuple(slots))
        if not validated.get("event_recorded") or not validated.get("event_complete"):
            raise LeaseRejected("worker_cuda_event_not_complete")
        call({
            "op": "cuda_completed", "transaction_id": tx, "tp_rank": rank,
        })
        _poll(call, tx, "CUDA_COMPLETED", deadline)

        call({
            "op": "validated", "transaction_id": tx, "tp_rank": rank,
        })
        _poll(call, tx, "ALL_RANKS_VALIDATED", deadline)
        call({
            "op": "authorize_release", "transaction_id": tx,
        })
        _poll(call, tx, "RELEASE_AUTHORIZED", deadline)

        # Release only after every rank has validated and the coordinator has
        # authorized release. The ACK is sent strictly after allocator release.
        released = lease.release_local()
        call({
            "op": "local_released", "transaction_id": tx, "tp_rank": rank,
        })
        final = _poll(call, tx, "GLOBAL_RELEASED", deadline)
        released.update(
            cross_rank_coordinated=True,
            global_release_confirmed=True,
            transaction_id=tx,
            coordinator_state=final["state"],
            coordinator_released_ranks=final["released_ranks"],
            control_elapsed_ms=(time.monotonic() - started) * 1000,
        )
        return released
    except Exception as exc:
        failure_message = str(exc) if isinstance(exc, LeaseRejected) else str(exc)
        failure = {
            "exception_type": type(exc).__name__,
            "exception_message": failure_message,
            "control_operation": current_operation,
            "transaction_id": tx,
            "tp_rank": rank,
            "elapsed_ms": (time.monotonic() - started) * 1000,
            "remaining_deadline_ms": remaining_ms(),
        }
        trace.append(failure)
        meta.update(
            cross_rank_coordinated=False,
            global_release_confirmed=False,
            control_failure=type(exc).__name__,
            control_failure_message=failure_message,
            control_failure_detail=failure,
        )
        try:
            exc.control_failure_detail = failure
            exc.control_trace = list(trace)
        except Exception:
            pass
        try:
            request_with_trace({
                "op": "fail", "transaction_id": tx,
                "reason": failure_message,
            })
        except Exception:
            pass
        # Abort is strictly local cleanup of this non-destructive experiment.
        # Never acknowledge cleanup until a Worker-owned CUDA fence permits
        # allocator release. An unreachable coordinator cannot imply success.
        try:
            cleanup = lease.abort()
            if cleanup.get("state") in ("ABORTED_RELEASED", "LOCAL_RELEASED"):
                call({"op": "abort_ack", "transaction_id": tx, "tp_rank": rank})
        except Exception:
            pass
        raise
