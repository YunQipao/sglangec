"""Node A coordinator for one CPU-only synthetic RS stripe."""

import argparse
import json
from pathlib import Path
import socket
import time
import tempfile
import threading

import numpy as np

from .rs_codec import encode, recover
from .tcp_transport import TCPTransport
from .rdma_transport import RDMATransport, receive_file as rdma_receive_file
from .tcp_transport import receive_file as tcp_receive_file
from .verify import verify_stripe


def request_json(host, port, request, *, timeout=5):
    deadline = time.monotonic() + timeout
    if isinstance(request, dict):
        request = json.dumps(request)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall((request + "\n").encode())
        data = bytearray()
        while not data.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker control response timed out")
            sock.settimeout(remaining)
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("worker control response ended early")
            data.extend(chunk)
    return json.loads(data)


def _validate_commit(args, result, blocks):
    binding = {key: getattr(args, key) for key in
               ("experiment_id", "stripe_id", "checkpoint_version")}
    if any(result.get(key) != value for key, value in binding.items()):
        raise ValueError("worker commit identity mismatch")
    checkpoint = result.get("checkpoint", {})
    if (checkpoint.get("state") != "COMMITTED" or
            any(checkpoint.get(key) is not True for key in
                ("data_received", "parity_encoded", "checksums_valid"))):
        raise ValueError("worker checkpoint verification incomplete")
    if any(result.get("verification", {}).get(key) is not True for key in
           ("D0_lost_byte_exact", "D1_lost_byte_exact", "P_recomputed_byte_exact")):
        raise ValueError("worker RS verification incomplete")
    entries = result.get("stripe_metadata", [])
    metadata = {entry.get("block_role"): entry for entry in entries}
    if len(entries) != 3 or set(metadata) != {"D0", "D1", "P"}:
        raise ValueError("complete D0/D1/P metadata required")
    for entry in entries:
        if (any(entry.get(key) != value for key, value in binding.items()) or
                entry.get("chunk_id") != args.chunk_id):
            raise ValueError("worker stripe identity mismatch")
    if (metadata["D0"].get("payload_length") != args.d0_bytes or
            metadata["D1"].get("payload_length") != args.d1_bytes):
        raise ValueError("worker payload length metadata mismatch")
    verify_stripe(metadata, blocks)


def _acknowledge_commit(args):
    binding = {key: getattr(args, key) for key in
               ("experiment_id", "stripe_id", "checkpoint_version")}
    response = request_json(args.node_b, args.control_port, {"op": "ack", **binding},
                            timeout=args.timeout)
    if (response.get("state") != "COMMITTED" or
            response.get("acknowledgement") != "ACKNOWLEDGED" or
            any(response.get(key) != value for key, value in binding.items())):
        raise RuntimeError(f"worker did not acknowledge coordinator acceptance: {response}")


def _reverse_context(args, state, *, prefix):
    """Build a useful error for the reverse transfer failure path."""
    endpoint = f"{args.return_bind_host}:{args.return_port}"
    phase = state.get("phase", "unknown")
    details = [f"{prefix}; reverse_listener={endpoint}", f"stage={phase}"]
    if state.get("error"):
        details.append(f"listener_error={state['error']}")
    if state.get("control_error"):
        details.append(f"worker_control_error={state['control_error']}")
    if state.get("worker_state"):
        details.append(f"worker_state={state['worker_state']}")
    return "; ".join(details)


def _join_receiver(receiver, timeout):
    """Join a receive helper and return whether it exited."""
    receiver.join(max(0.0, timeout))
    return not receiver.is_alive()


def _raise_reverse_failure(error, args, state, *, prefix):
    """Re-raise a receiver exception with endpoint/stage context attached."""
    message = _reverse_context(args, state, prefix=f"{prefix}: {error}")
    if isinstance(error, BaseException):
        # Python 3.11's notes preserve the original exception type and
        # traceback while making the cross-thread context visible to callers.
        if hasattr(error, "add_note"):
            error.add_note(message)
        if "reverse_listener=" not in str(error):
            error.args = (f"{error}; {message}", *error.args[1:])
        raise error
    raise RuntimeError(message)


def run(args):
    if args.transport == "rdma":
        transport = RDMATransport(
            device=args.rdma_device, gid_index=args.rdma_gid_index,
            port_num=args.rdma_port_num, timeout=args.timeout,
        )
    else:
        transport = TCPTransport(retry_timeout=args.timeout)
    health_deadline = time.monotonic() + args.timeout
    while True:
        try:
            health = request_json(args.node_b, args.control_port, "health")
            if health.get("state") == "READY":
                break
            raise RuntimeError(f"Node B health check failed: {health}")
        except (OSError, TimeoutError, ConnectionError):
            if time.monotonic() >= health_deadline:
                raise TimeoutError("Node B control service did not become ready")
            time.sleep(.1)
    rng = np.random.default_rng(args.seed)
    d0 = rng.integers(0, 256, args.d0_bytes, dtype=np.uint8).tobytes()
    d1 = np.random.default_rng(args.seed + 1).integers(0, 256, args.d1_bytes, dtype=np.uint8).tobytes()
    experiment_id = args.experiment_id
    stripe_id = args.stripe_id
    checkpoint_version = args.checkpoint_version
    started = time.perf_counter()
    reverse_ready = threading.Event()
    reverse_done = threading.Event()
    reverse_cancel = threading.Event()
    reverse_state = {
        # None means pending; only an explicit True means receive succeeded.
        "success": None,
        "result": None,
        "error": None,
        "phase": "starting",
        "bind_host": args.return_bind_host,
        "port": args.return_port,
    }
    with tempfile.TemporaryDirectory(prefix="kv-ec-reverse-") as temporary:
        return_path = Path(temporary) / "D1.bin"

        def receive_reverse():
            try:
                if args.transport == "tcp":
                    result = tcp_receive_file(
                        args.return_bind_host, args.return_port, return_path,
                        timeout=args.timeout, ready_event=reverse_ready,
                        status=reverse_state, cancel_event=reverse_cancel,
                    )
                else:
                    # RDMA's verbs helper owns its listener.  It has no
                    # socket-level readiness event, so the existing helper
                    # timeout and error path remain authoritative.
                    reverse_state["phase"] = "rdma_starting"
                    result = rdma_receive_file(
                        args.return_port, return_path, device=args.rdma_device,
                        bind_host=args.return_bind_host,
                        gid_index=args.rdma_gid_index, port_num=args.rdma_port_num,
                        timeout=args.timeout, return_stats=True,
                    )
                if isinstance(result, BaseException):
                    raise result
                reverse_state.update(success=True, result=result)
            except BaseException as exc:
                reverse_state.update(success=False, error=exc)
                if not reverse_ready.is_set():
                    reverse_ready.set()
            finally:
                reverse_done.set()

        receiver = threading.Thread(target=receive_reverse, daemon=True)
        receiver.start()
        if args.transport == "tcp":
            # Bind/listen must complete before D0 is sent.  Otherwise a fast
            # worker can fail its reverse connect and the coordinator reports
            # a misleading timeout later.
            if not reverse_ready.wait(args.timeout):
                reverse_cancel.set()
                _join_receiver(receiver, args.timeout + .25)
                raise TimeoutError(_reverse_context(
                    args, reverse_state,
                    prefix="reverse listener did not become ready before D0"))
            if (reverse_state["success"] is False or
                    (reverse_state["success"] is not True and reverse_state.get("phase") != "listening")):
                reverse_cancel.set()
                _join_receiver(receiver, args.timeout + .25)
                error = reverse_state["error"] if reverse_state["error"] is not None else "unknown listener failure"
                _raise_reverse_failure(error, args, reverse_state,
                                       prefix="reverse listener failed")
        try:
            if args.transport == "rdma":
                transfer_stats = transport.send_with_stats(args.node_b, args.data_port, d0)
                transfer_seconds = transfer_stats["elapsed_s"]
            else:
                transfer_stats = None
                transfer_seconds = transport.send(args.node_b, args.data_port, d0)
        except Exception:
            # Let this run's receive helper hit its bounded timeout and exit;
            # do not leave a temporary listener/helper behind on failure.
            reverse_cancel.set()
            _join_receiver(receiver, args.timeout + .25)
            raise
        # Wait in short intervals so a Worker error is reported immediately,
        # instead of being hidden behind the reverse socket timeout.
        reverse_deadline = time.monotonic() + args.timeout
        worker_result = None
        next_status_poll = time.monotonic()
        control_failure_since = None
        while not reverse_done.is_set() and time.monotonic() < reverse_deadline:
            receiver.join(min(.1, max(0.0, reverse_deadline - time.monotonic())))
            if reverse_done.is_set():
                break
            if time.monotonic() >= next_status_poll:
                try:
                    worker_result = request_json(
                        args.node_b, args.control_port, "result", timeout=.5)
                    reverse_state["worker_state"] = worker_result.get("state", "UNKNOWN")
                    control_failure_since = None
                except (OSError, TimeoutError, ConnectionError) as exc:
                    reverse_state["control_error"] = f"{type(exc).__name__}: {exc}"
                    if control_failure_since is None:
                        control_failure_since = time.monotonic()
                    elif time.monotonic() - control_failure_since >= .5:
                        worker_result = {
                            "state": "ERROR",
                            "error": "worker control service stopped responding",
                        }
                next_status_poll = time.monotonic() + .25
                if worker_result and worker_result.get("state") == "ERROR":
                    break
        if (reverse_state["success"] is False or
                (worker_result and worker_result.get("state") == "ERROR")):
            error = (reverse_state["error"] if reverse_state["success"] is False
                     else worker_result.get("error", worker_result))
            reverse_cancel.set()
            _join_receiver(receiver, .5)
            _raise_reverse_failure(error, args, reverse_state,
                                   prefix="Worker failed before reverse transfer")
        if receiver.is_alive():
            reverse_cancel.set()
            _join_receiver(receiver, .25)
            raise TimeoutError(_reverse_context(
                args, reverse_state, prefix="reverse data transfer did not complete before timeout"))
        receiver.join()
        if reverse_state["success"] is False:
            _raise_reverse_failure(reverse_state["error"], args, reverse_state,
                                   prefix="reverse transfer failed")
        if reverse_state["success"] is not True:
            raise RuntimeError(_reverse_context(
                args, reverse_state, prefix="reverse transfer failed: no result"))
        received_d1 = return_path.read_bytes()
    if received_d1 != d1:
        raise ValueError("Node B to Node A D1 transfer is not byte exact")
    result = request_json(args.node_b, args.control_port, "result", timeout=args.timeout)
    deadline = time.monotonic() + args.timeout
    while result.get("state") == "PENDING" and time.monotonic() < deadline:
        time.sleep(.05)
        result = request_json(args.node_b, args.control_port, "result",
                              timeout=max(.001, deadline - time.monotonic()))
    if result.get("state") != "COMMITTED":
        raise RuntimeError(f"worker checkpoint did not commit: {result}")
    padded = max(args.d0_bytes, args.d1_bytes)
    d0_pad = np.frombuffer(d0 + bytes(padded - len(d0)), dtype=np.uint8).copy()
    d1_pad = np.frombuffer(d1 + bytes(padded - len(d1)), dtype=np.uint8).copy()
    _, _, parity = encode(d0_pad, d1_pad)
    # Independent coordinator-side validation of all three erasure cases.
    restored0, _ = recover(data0=None, data1=d1_pad, parity=parity,
                           lengths=(len(d0), len(d1)))
    _, restored1 = recover(data0=d0_pad, data1=None, parity=parity,
                           lengths=(len(d0), len(d1)))
    _, _ = recover(data0=d0_pad, data1=d1_pad, parity=None,
                   lengths=(len(d0), len(d1)))
    if restored0.tobytes() != d0 or restored1.tobytes() != d1:
        raise ValueError("coordinator RS recovery byte comparison failed")
    if result["transport"] != transport.name:
        raise ValueError("worker/coordinator transport identity mismatch")
    _validate_commit(args, result, {"D0": d0_pad, "D1": d1_pad, "P": parity})
    reverse_seconds = result["node_b_return_transfer_ms"] / 1000
    result.update(experiment_id=experiment_id, stripe_id=stripe_id,
                  checkpoint_version=checkpoint_version, d0_bytes=len(d0), d1_bytes=len(d1),
                  transport=transport.name, transfer_ms=transfer_seconds * 1000,
                  reverse_transfer_ms=reverse_seconds * 1000,
                  effective_network_gbps=((len(d0) + len(d1)) * 8 /
                                          (transfer_seconds + reverse_seconds) / 1e9
                                          if transfer_seconds + reverse_seconds else None),
                  total_transferred_bytes=len(d0) + len(d1),
                  total_wall_ms=(time.perf_counter() - started) * 1000,
                  protection_completion_ms=(time.perf_counter() - started) * 1000,
                  shared_node_failure_recovery=False,
                  rdma_send=transfer_stats,
                  rdma_reverse=(result.get("rdma_return") if args.transport == "rdma" else None))
    _acknowledge_commit(args)
    result["acknowledgement"] = "ACKNOWLEDGED"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    import csv
    fields = ["transport", "block_bytes", "rs_encode_ms", "rs_decode_ms", "transfer_ms", "reverse_transfer_ms",
              "effective_network_gbps", "protection_completion_ms", "recovery_ms",
              "total_transferred_bytes", "cpu_percent", "host_dram_bytes", "state"]
    with (output / "result.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: result.get(field) for field in fields})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("tcp", "rdma"), default="tcp")
    parser.add_argument("--node-b", required=True)
    parser.add_argument("--control-port", type=int, default=19470)
    parser.add_argument("--data-port", type=int, default=19471)
    parser.add_argument("--return-bind-host", default="0.0.0.0")
    parser.add_argument("--return-port", type=int, default=19472)
    parser.add_argument("--rdma-device", default="-")
    parser.add_argument("--rdma-gid-index", type=int)
    parser.add_argument("--rdma-port-num", type=int, default=1)
    parser.add_argument("--d0-bytes", type=int, required=True)
    parser.add_argument("--d1-bytes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--experiment-id", default="synthetic-rs21-experiment")
    parser.add_argument("--stripe-id", default="stripe-0")
    parser.add_argument("--chunk-id", default="chunk-0")
    parser.add_argument("--checkpoint-version", default="checkpoint-v1")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not 0 < args.d0_bytes <= 256 * 1024**2
            or not 0 < args.d1_bytes <= 256 * 1024**2):
        parser.error("D0/D1 sizes must be in 1..256 MiB")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
