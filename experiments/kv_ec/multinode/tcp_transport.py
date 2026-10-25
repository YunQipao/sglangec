"""Explicit TCP payload fallback for development; never identified as RDMA."""

import socket
import struct
import time

from .transport import Transport


class TCPTransport(Transport):
    name = "tcp"

    def __init__(self, *, retry_timeout=30, connect_timeout=5, retry_interval=.05):
        if retry_timeout <= 0 or connect_timeout <= 0 or retry_interval <= 0:
            raise ValueError("TCP retry and connect timeouts must be positive")
        self.retry_timeout = float(retry_timeout)
        self.connect_timeout = float(connect_timeout)
        self.retry_interval = float(retry_interval)

    def send(self, host, port, payload):
        started = time.perf_counter()
        deadline = time.monotonic() + self.retry_timeout
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"TCP connection to {host}:{port} timed out after {self.retry_timeout}s")
                sock = socket.create_connection(
                    (host, port), timeout=min(self.connect_timeout, remaining))
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(self.retry_interval, max(0, deadline - time.monotonic())))
        with sock:
            sock.sendall(struct.pack("!Q", len(payload)))
            sock.sendall(payload)
            response = sock.recv(1)
            if response != b"\x01":
                raise ConnectionError("TCP worker did not acknowledge complete payload")
        return time.perf_counter() - started


def receive_file(host, port, destination, *, timeout=60, ready_event=None,
                 status=None, cancel_event=None):
    """Receive one framed TCP payload.

    ``ready_event``, ``status`` and ``cancel_event`` are optional observability
    hooks used by the coordinator.  The event is set after bind/listen
    succeeds (and also when bind fails), so a sender is never started against
    an unknown listener state.  The original call signature remains valid for
    existing callers.
    """

    def update(phase, **values):
        if status is not None:
            status.update(phase=phase, **values)

    started = time.perf_counter()
    deadline = time.monotonic() + float(timeout)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        update("binding", bind_host=host, port=port)
        try:
            server.bind((host, port))
            server.listen(1)
        except BaseException as exc:
            update("bind_failed", error=f"{type(exc).__name__}: {exc}")
            if ready_event is not None:
                ready_event.set()
            raise
        update("listening", address=server.getsockname())
        if ready_event is not None:
            ready_event.set()
        # A cancellation event lets the coordinator tear down a failed run
        # promptly instead of waiting for the full transfer timeout.
        poll_timeout = min(float(timeout), .2) if cancel_event is not None else timeout
        server.settimeout(poll_timeout)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                update("cancelled")
                raise TimeoutError("TCP reverse receive cancelled")
            try:
                conn, address = server.accept()
                break
            except socket.timeout:
                if time.monotonic() >= deadline:
                    update("accept_timeout", error="TCP reverse receive timed out")
                    raise TimeoutError("TCP reverse receive timed out")
                continue
            except BaseException as exc:
                update("accept_failed", error=f"{type(exc).__name__}: {exc}")
                raise
        update("connected", peer=address)
        with conn, open(destination, "wb") as stream:
            conn.settimeout(poll_timeout)
            size = struct.unpack("!Q", _recv_exact(
                conn, 8, cancel_event=cancel_event, deadline=deadline))[0]
            if size <= 0 or size > 256 * 1024**2:
                raise ValueError("invalid TCP payload length")
            update("receiving", expected_bytes=size)
            remaining = size
            while remaining:
                try:
                    data = conn.recv(min(1024 * 1024, remaining))
                except socket.timeout:
                    if ((cancel_event is not None and cancel_event.is_set())
                            or time.monotonic() >= deadline):
                        update("cancelled")
                        raise TimeoutError("TCP reverse receive timed out or was cancelled")
                    raise
                if not data:
                    raise ConnectionError("TCP payload ended before expected length")
                stream.write(data)
                remaining -= len(data)
            conn.sendall(b"\x01")
            update("complete", bytes=size)
    return time.perf_counter() - started


def _recv_exact(sock, size, *, cancel_event=None, deadline=None):
    chunks = []
    while size:
        try:
            chunk = sock.recv(size)
        except socket.timeout:
            if ((cancel_event is not None and cancel_event.is_set())
                    or (deadline is not None and time.monotonic() >= deadline)):
                raise TimeoutError("TCP reverse receive timed out or was cancelled")
            raise
        if not chunk:
            raise ConnectionError("TCP payload ended before expected length")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)
