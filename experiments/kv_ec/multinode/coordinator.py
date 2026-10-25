"""Node A coordinator for one CPU-only synthetic RS stripe."""

import argparse
import json
from pathlib import Path
import socket
import time
import tempfile
import threading

import numpy as np

from .rs_codec import CODEC, CODEC_VERSION, encode, recover
from .tcp_transport import TCPTransport
from .rdma_transport import RDMATransport, receive_file as rdma_receive_file
from .tcp_transport import receive_file as tcp_receive_file


def request_json(host, port, request):
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall((request + "\n").encode())
        data = bytearray()
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("worker control response ended early")
            data.extend(chunk)
    return json.loads(data)


def run(args):
    if args.transport == "rdma":
        transport = RDMATransport(device=args.rdma_device)
    else:
        transport = TCPTransport()
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
    transfer_errors = []
    with tempfile.TemporaryDirectory(prefix="kv-ec-reverse-") as temporary:
        return_path = Path(temporary) / "D1.bin"

        def receive_reverse():
            try:
                if args.transport == "tcp":
                    transfer_errors.append(tcp_receive_file(args.return_bind_host,
                                                            args.return_port, return_path))
                else:
                    rdma_receive_file(args.return_port, return_path, device=args.rdma_device,
                                      bind_host=args.return_bind_host)
                    transfer_errors.append(0.0)
            except Exception as exc:
                transfer_errors.append(exc)

        receiver = threading.Thread(target=receive_reverse, daemon=True)
        receiver.start()
        try:
            transfer_seconds = transport.send(args.node_b, args.data_port, d0)
        except Exception:
            # Let this run's receive helper hit its bounded timeout and exit;
            # do not leave a temporary listener/helper behind on failure.
            receiver.join(125)
            raise
        receiver.join(min(args.timeout, 125))
        if receiver.is_alive():
            raise TimeoutError("reverse data transfer did not complete")
        if not transfer_errors or isinstance(transfer_errors[0], Exception):
            raise RuntimeError(f"reverse transfer failed: {transfer_errors[0] if transfer_errors else 'no result'}")
        received_d1 = return_path.read_bytes()
    if received_d1 != d1:
        raise ValueError("Node B to Node A D1 transfer is not byte exact")
    result = request_json(args.node_b, args.control_port, "result")
    deadline = time.monotonic() + args.timeout
    while result.get("state") == "PENDING" and time.monotonic() < deadline:
        time.sleep(.05)
        result = request_json(args.node_b, args.control_port, "result")
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
                  shared_node_failure_recovery=False)
    result["stripe_metadata"][0].update(experiment_id=experiment_id, stripe_id=stripe_id,
                                        checkpoint_version=checkpoint_version,
                                        chunk_id=args.chunk_id,
                                        codec=CODEC, codec_version=CODEC_VERSION)
    result["stripe_metadata"][1].update(experiment_id=experiment_id, stripe_id=stripe_id,
                                        checkpoint_version=checkpoint_version, chunk_id=args.chunk_id)
    result["stripe_metadata"][2].update(experiment_id=experiment_id, stripe_id=stripe_id,
                                        checkpoint_version=checkpoint_version, chunk_id=args.chunk_id)
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
