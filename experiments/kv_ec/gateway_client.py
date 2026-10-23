"""Sender client and synthetic network probe. Never writes payload data to disk."""

import argparse
import hashlib
import json
import socket
import time

try:
    from .gateway_protocol import MAX_CHUNK_BYTES, PROTOCOL, encode_frame, recv_ack
except ImportError:
    from gateway_protocol import MAX_CHUNK_BYTES, PROTOCOL, encode_frame, recv_ack


def send_chunk(endpoint, header, payload, *, timeout_s=10):
    started = time.monotonic()
    with socket.create_connection(endpoint, timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(encode_frame(header, payload))
        ack = recv_ack(sock)
    ack["round_trip_ms"] = (time.monotonic() - started) * 1000
    return ack


def synthetic_chunk(seed, offset, length):
    """Deterministic bounded test bytes, generated without a payload file."""
    out = bytearray()
    counter = offset // 32
    skip = offset % 32
    while len(out) < length + skip:
        out.extend(hashlib.sha256(f"{seed}:{counter}".encode()).digest())
        counter += 1
    return bytes(out[skip:skip + length])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", required=True, help="GATEWAY_HOST:PORT")
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--transaction", default="synthetic-network-probe")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--total-bytes", type=int, default=9 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024)
    parser.add_argument("--source-node", default="node0")
    parser.add_argument("--destination-node", default="node1")
    parser.add_argument("--output", required=True, help="metadata-only metrics JSON path")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.total_bytes <= 0 or args.chunk_bytes <= 0 or args.chunk_bytes > MAX_CHUNK_BYTES:
        parser.error("invalid byte limits")
    host, port = args.gateway.rsplit(":", 1)
    endpoint = (host, int(port))
    chunk_rtts, errors = [], 0
    total_started = time.monotonic()
    count = (args.total_bytes + args.chunk_bytes - 1) // args.chunk_bytes
    for index in range(count):
        offset = index * args.chunk_bytes
        length = min(args.chunk_bytes, args.total_bytes - offset)
        payload = synthetic_chunk(args.transaction, offset, length)
        header = dict(transaction_id=args.transaction, stripe_id="stripe-0",
                      checkpoint_id=args.transaction + "-checkpoint", rank_id=args.rank,
                      chunk_index=index, chunk_offset=offset, chunk_length=length,
                      total_length=args.total_bytes, source_node=args.source_node,
                      destination_node=args.destination_node, manager_epoch=args.epoch,
                      protocol_version=PROTOCOL, chunk_sha256=hashlib.sha256(payload).hexdigest())
        for attempt in range(args.retries + 1):
            try:
                ack = send_chunk(endpoint, header, payload, timeout_s=args.timeout)
                chunk_rtts.append(ack["round_trip_ms"])
                break
            except Exception:
                errors += 1
                if attempt == args.retries:
                    raise
    elapsed = time.monotonic() - total_started
    report = dict(status="sent", source_endpoint="sender->" + args.gateway,
                  destination_node=args.destination_node, chunks=count,
                  payload_bytes=args.total_bytes, elapsed_s=elapsed,
                  throughput_mib_s=args.total_bytes / max(elapsed, 1e-9) / 1024**2,
                  chunk_rtt_ms_min=min(chunk_rtts), chunk_rtt_ms_mean=sum(chunk_rtts) / len(chunk_rtts),
                  chunk_rtt_ms_max=max(chunk_rtts), failures=errors,
                  payload_persisted=False, protocol=PROTOCOL)
    with open(args.output, "x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
