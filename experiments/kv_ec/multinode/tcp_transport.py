"""Explicit TCP payload fallback for development; never identified as RDMA."""

import socket
import struct
import time
import struct

from .transport import Transport


class TCPTransport(Transport):
    name = "tcp"

    def send(self, host, port, payload):
        started = time.perf_counter()
        deadline = time.monotonic() + 30
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=5)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        with sock:
            sock.sendall(struct.pack("!Q", len(payload)))
            sock.sendall(payload)
            response = sock.recv(1)
            if response != b"\x01":
                raise ConnectionError("TCP worker did not acknowledge complete payload")
        return time.perf_counter() - started


def receive_file(host, port, destination, *, timeout=60):
    started = time.perf_counter()
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.settimeout(timeout)
        conn, _ = server.accept()
        with conn, open(destination, "wb") as stream:
            conn.settimeout(timeout)
            size = struct.unpack("!Q", _recv_exact(conn, 8))[0]
            if size <= 0 or size > 256 * 1024**2:
                raise ValueError("invalid TCP payload length")
            remaining = size
            while remaining:
                data = conn.recv(min(1024 * 1024, remaining))
                if not data:
                    raise ConnectionError("TCP payload ended before expected length")
                stream.write(data)
                remaining -= len(data)
            conn.sendall(b"\x01")
    return time.perf_counter() - started


def _recv_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("TCP payload ended before expected length")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)
