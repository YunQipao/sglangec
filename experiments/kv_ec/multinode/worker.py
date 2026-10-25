"""CPU-only receiver/worker for synthetic two-node RS(2,1)."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import struct
import sys
import threading
import time
import traceback

import numpy as np

from .rdma_transport import receive_file
from .rs_codec import CODEC, CODEC_VERSION, encode, recover
from .tcp_transport import TCPTransport
from .verify import verify_stripe


def _recv_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("TCP payload ended early")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _control_response(request, result_path, binding):
    """Keep the durable protection result separate from its ACK receipt."""
    if request == "health":
        return {"state": "READY", **binding}
    result = (json.loads(result_path.read_text()) if result_path.exists()
              else {"state": "PENDING", **binding})
    ack_path = result_path.with_name("acknowledgement.json")
    if request == "result":
        receipt = json.loads(ack_path.read_text()) if ack_path.exists() else {}
        return {**result, "acknowledgement": receipt.get("state", "WAITING")}
    try:
        message = json.loads(request)
    except ValueError:
        return {"state": "ERROR", "reason": "unknown_control_request"}
    if not isinstance(message, dict) or message.get("op") != "ack":
        return {"state": "ERROR", "reason": "unknown_control_request"}
    if any(message.get(key) != value for key, value in binding.items()):
        return {"state": "ERROR", "reason": "ack_identity_mismatch"}
    if (result.get("state") != "COMMITTED" or
            any(result.get(key) != value for key, value in binding.items())):
        return {"state": "ERROR", "reason": "checkpoint_not_committed"}
    if not ack_path.exists():
        _atomic_json(ack_path, {"state": "ACKNOWLEDGED", **binding})
    return {"state": "COMMITTED", "acknowledgement": "ACKNOWLEDGED", **binding}


def _control_server(host, port, result_path, ready, stop, acknowledged, binding, timeout):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(.1)
        ready.set()
        while not stop.is_set() and not acknowledged.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(min(timeout, .2))
                accepted_ack = False
                try:
                    data = bytearray()
                    deadline = time.monotonic() + min(timeout, .2)
                    while not data.endswith(b"\n"):
                        if stop.is_set() or time.monotonic() >= deadline:
                            raise TimeoutError("control request timed out")
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data.extend(chunk)
                        if len(data) > 65536:
                            raise ValueError("control request too large")
                    if not data:
                        continue
                    request = data.decode().strip()
                    response = _control_response(request, result_path, binding)
                    accepted_ack = (request not in ("health", "result") and
                                    response.get("acknowledgement") == "ACKNOWLEDGED")
                    conn.sendall((json.dumps(response) + "\n").encode())
                except (OSError, ValueError):
                    # A disconnected client must not kill the control plane.
                    # A lost ACK response still makes the coordinator fail.
                    continue
                finally:
                    if accepted_ack:
                        acknowledged.set()


@contextmanager
def _control_lifecycle(args, result_path):
    ready, stop, acknowledged = (threading.Event() for _ in range(3))
    binding = {key: getattr(args, key) for key in
               ("experiment_id", "stripe_id", "checkpoint_version")}
    errors = []

    def serve():
        try:
            _control_server(args.control_host, args.control_port, result_path,
                            ready, stop, acknowledged, binding, args.timeout)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    thread = threading.Thread(target=serve, daemon=False)
    thread.start()
    try:
        if not ready.wait(min(5, args.timeout)):
            raise TimeoutError("control server startup timed out")
        if errors:
            raise errors[0]
        yield acknowledged
    finally:
        stop.set()
        thread.join(args.timeout + .5)
        if thread.is_alive():
            raise TimeoutError("control server did not stop")
    if errors:
        raise errors[0]


def _csv_report(path, result):
    fields = ["transport", "block_bytes", "rs_encode_ms", "rs_decode_ms", "transfer_ms",
              "reverse_transfer_ms",
              "effective_network_gbps", "protection_completion_ms", "recovery_ms",
              "total_transferred_bytes", "cpu_percent", "host_dram_bytes", "state"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: result.get(key) for key in fields})


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_error_result(output, exc):
    """Publish a failed checkpoint so a coordinator can see the root cause."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    # Never overwrite a result from an earlier run.  A fresh worker output
    # directory is part of the experiment contract.
    if result_path.exists():
        return
    _atomic_json(result_path, {
        "state": "ERROR",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    })


def _check_recovered_block(role, recovered, expected_raw, expected_checksum):
    """Validate the unpadded byte stream returned by RS recovery."""
    if recovered.tobytes() != expected_raw:
        raise ValueError(f"{role} recovery was not byte exact")
    if hashlib.sha256(recovered).hexdigest() != expected_checksum:
        raise ValueError(f"{role} recovery checksum mismatch")


def _write_block(path, block):
    with path.open("xb") as stream:
        stream.write(block.tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    written = path.read_bytes()
    if written != block.tobytes():
        raise IOError(f"persisted block failed byte verification: {path.name}")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"
    with _control_lifecycle(args, result_path) as acknowledged:
        try:
            result = _run_protection(args, output)
        except Exception as exc:
            _write_error_result(output, exc)
            raise
        if not acknowledged.wait(args.timeout):
            # COMMITTED data remains intact even if the coordinator vanished.
            raise TimeoutError("checkpoint COMMITTED but coordinator ACK was not received")
    return {**result, "acknowledgement": "ACKNOWLEDGED"}


def _run_protection(args, output):
    process_usage_before = resource.getrusage(resource.RUSAGE_SELF)
    protection_started = time.perf_counter()
    run_timeout = getattr(args, "timeout", 300)
    result_path = output / "result.json"
    received_path = output / "D0.bin"
    if args.transport == "tcp":
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.data_host, args.data_port))
            server.listen(1)
            server.settimeout(run_timeout)
            conn, _ = server.accept()
            with conn, received_path.open("wb") as stream:
                conn.settimeout(run_timeout)
                length = struct.unpack("!Q", _recv_exact(conn, 8))[0]
                if length != args.d0_bytes:
                    raise ValueError("D0 payload length does not match configured length")
                remaining = length
                while remaining:
                    chunk = conn.recv(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("incomplete TCP D0 payload")
                    stream.write(chunk)
                    remaining -= len(chunk)
                conn.sendall(b"\x01")
    else:
        rdma_receive_stats = receive_file(
            args.data_port, received_path, device=args.rdma_device,
            gid_index=args.rdma_gid_index, port_num=args.rdma_port_num,
            timeout=run_timeout, return_stats=True,
        )
        if received_path.stat().st_size != args.d0_bytes:
            raise ValueError("D0 payload length does not match configured length")
    d0_raw = received_path.read_bytes()
    expected_d0 = np.random.default_rng(args.seed).integers(0, 256, args.d0_bytes, dtype=np.uint8).tobytes()
    if hashlib.sha256(d0_raw).hexdigest() != hashlib.sha256(expected_d0).hexdigest():
        raise ValueError("D0 source checksum mismatch")
    d1_raw = np.random.default_rng(args.seed + 1).integers(0, 256, args.d1_bytes, dtype=np.uint8).tobytes()
    padded = max(len(d0_raw), len(d1_raw))
    d0 = np.frombuffer(d0_raw + bytes(padded - len(d0_raw)), dtype=np.uint8).copy()
    d1 = np.frombuffer(d1_raw + bytes(padded - len(d1_raw)), dtype=np.uint8).copy()
    digest = lambda payload: hashlib.sha256(payload).hexdigest()
    # RS blocks are padded to a common width, while recover() deliberately
    # returns the original unpadded payload.  Keep both checksum domains:
    # padded checksums are used for stripe metadata, raw checksums validate
    # the byte-exact recovery result.
    raw_hashes = {"D0": digest(d0_raw), "D1": digest(d1_raw)}
    encode_started = time.perf_counter()
    _, _, parity = encode(d0, d1)
    encode_ms = (time.perf_counter() - encode_started) * 1000
    blocks = {"D0": d0, "D1": d1, "P": parity}
    hashes = {role: digest(block.tobytes()) for role, block in blocks.items()}
    stripe = {"experiment_id": args.experiment_id, "stripe_id": args.stripe_id,
              "chunk_id": args.chunk_id, "source_node_id": "node-a", "model_fingerprint": "synthetic-seeded-data",
              "payload_length": len(d0_raw), "padded_length": padded,
              "padding_length": padded - len(d0_raw), "codec": CODEC,
              "codec_version": CODEC_VERSION, "rs_parameters": {"k": 2, "m": 1,
              "field": "GF(2^8)", "primitive_polynomial": "0x11d", "coefficients": [1, 2]},
              "checksum": hashes["D0"], "checkpoint_version": args.checkpoint_version,
              "protection_state": "PENDING", "block_role": "D0",
              "local_request_id": "node-a-synthetic-request"}
    d1_meta = dict(stripe, source_node_id="node-b", payload_length=len(d1_raw),
                   padding_length=padded - len(d1_raw), block_role="D1",
                   checksum=hashes["D1"], local_request_id="node-b-different-synthetic-request")
    parity_meta = dict(stripe, source_node_id="node-b", payload_length=padded,
                       padding_length=0, block_role="P", checksum=hashes["P"])
    decode_times = []
    for erased in ("D0", "D1", "P"):
        decode_started = time.perf_counter()
        inputs = {role: None if role == erased else block for role, block in blocks.items()}
        restored = recover(data0=inputs["D0"], data1=inputs["D1"], parity=inputs["P"],
                           lengths=(len(d0_raw), len(d1_raw)))
        recovered = restored[0] if erased == "D0" else restored[1] if erased == "D1" else None
        decode_ms = (time.perf_counter() - decode_started) * 1000
        decode_times.append(decode_ms)
        if erased == "D0":
            _check_recovered_block("D0", recovered, d0_raw, raw_hashes["D0"])
        if erased == "D1":
            _check_recovered_block("D1", recovered, d1_raw, raw_hashes["D1"])
        if erased == "P" and not np.array_equal(encode(d0, d1)[2], parity):
            raise ValueError("P recomputation mismatch")
    for block_meta in (stripe, d1_meta, parity_meta):
        block_meta["protection_state"] = "COMMITTED"
    verification = verify_stripe({"D0": stripe, "D1": d1_meta, "P": parity_meta}, blocks)
    if args.transport == "tcp":
        return_transport = TCPTransport(retry_timeout=run_timeout)
    else:
        from .rdma_transport import RDMATransport
        return_transport = RDMATransport(
            device=args.rdma_device, gid_index=args.rdma_gid_index,
            port_num=args.rdma_port_num, timeout=run_timeout,
        )
    if args.transport == "rdma":
        return_stats = return_transport.send_with_stats(args.return_host, args.return_port, d1_raw)
        return_seconds = return_stats["elapsed_s"]
        return_transport.close()
    else:
        return_stats = None
        return_seconds = return_transport.send(args.return_host, args.return_port, d1_raw)
    # Persist the complete padded stripe before publishing COMMITTED metadata.
    _write_block(output / "D0.padded.bin", d0)
    _write_block(output / "D1.padded.bin", d1)
    _write_block(output / "P.bin", parity)
    protection_ms = (time.perf_counter() - protection_started) * 1000
    process_usage_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = ((process_usage_after.ru_utime + process_usage_after.ru_stime) -
                   (process_usage_before.ru_utime + process_usage_before.ru_stime))
    result = {"state": "COMMITTED", "transport": args.transport,
              "experiment_id": args.experiment_id, "stripe_id": args.stripe_id,
              "checkpoint_version": args.checkpoint_version,
              "shared_node_failure_recovery": False,
              "placement": {"D0": "node-a", "D1": "node-b", "P": "node-b-independent-buffer"},
              "stripe_metadata": [stripe, d1_meta, parity_meta],
              "block_bytes": padded, "rs_encode_ms": encode_ms,
              "rs_decode_ms": max(decode_times), "transfer_ms": None,
              "reverse_transfer_ms": return_seconds * 1000,
              "effective_network_gbps": None,
              "protection_completion_ms": protection_ms,
              "recovery_ms": max(decode_times), "total_transferred_bytes": len(d0_raw) + len(d1_raw),
              "cpu_percent": (100 * cpu_seconds / (protection_ms / 1000)
                              if protection_ms else 0),
              "host_dram_bytes": int(d0.nbytes + d1.nbytes + parity.nbytes),
              "max_rss_bytes": process_usage_after.ru_maxrss * 1024,
              "node_b_return_transfer_ms": return_seconds * 1000,
              "rdma_receive": locals().get("rdma_receive_stats"),
              "rdma_return": return_stats,
              "checksum": hashes,
              "verification": {"D0_lost_byte_exact": True, "D1_lost_byte_exact": True,
                               "P_recomputed_byte_exact": True, **verification},
              "checkpoint": {"state": "COMMITTED", "data_received": True,
                             "parity_encoded": True, "checksums_valid": True}}
    _atomic_json(result_path, result)
    _csv_report(output / "result.csv", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("tcp", "rdma"), required=True)
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=19470)
    parser.add_argument("--data-host", default="0.0.0.0")
    parser.add_argument("--data-port", type=int, default=19471)
    parser.add_argument("--return-host", default="127.0.0.1")
    parser.add_argument("--return-port", type=int, default=19472)
    parser.add_argument("--rdma-device", default="-")
    parser.add_argument("--rdma-gid-index", type=int)
    parser.add_argument("--rdma-port-num", type=int, default=1)
    parser.add_argument("--d0-bytes", type=int, required=True)
    parser.add_argument("--d1-bytes", type=int, required=True)
    parser.add_argument("--experiment-id", default="synthetic-rs21-experiment")
    parser.add_argument("--stripe-id", default="stripe-0")
    parser.add_argument("--chunk-id", default="chunk-0")
    parser.add_argument("--checkpoint-version", default="checkpoint-v1")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.d0_bytes <= 256 * 1024**2 or not 0 < args.d1_bytes <= 256 * 1024**2:
        parser.error("D0/D1 sizes must be in 1..256 MiB")
    try:
        result = run(args)
    except Exception as exc:
        try:
            _write_error_result(args.output, exc)
        except Exception as persist_error:
            print(f"failed to publish worker error state: {persist_error}", file=sys.stderr)
        # Keep the non-zero exit status and traceback for the invoking test or
        # operator.  The result file above gives the coordinator a structured
        # copy before the control thread disappears.
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
