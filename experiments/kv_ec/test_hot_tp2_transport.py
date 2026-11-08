"""Pure-memory protocol tests; real socketpair smoke is explicitly opt-in.

The default suite performs no kernel socket I/O and is not network acceptance.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import socket
import struct
import threading
import time
import unittest

from experiments.kv_ec.hot_tp2 import MIB, Options, Rejected, protect, recover
from experiments.kv_ec.hot_tp2_transport import ChunkSocket, MAX_HEADER
from experiments.kv_ec.test_hot_tp2 import MemorySink


class _Lane:
    def __init__(self):
        self.data, self.closed = bytearray(), False
        self.condition = threading.Condition()


class _MemoryEndpoint:
    """Test double with 64KiB credits, finite waits and EOF; no kernel I/O."""
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing, self.timeout = incoming, outgoing, 1

    def settimeout(self, seconds):
        self.timeout = seconds

    def sendall(self, raw):
        deadline, offset, lane = time.monotonic() + self.timeout, 0, self.outgoing
        while offset < len(raw):
            with lane.condition:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("memory channel write timeout")
                if lane.closed:
                    raise ConnectionError("memory channel closed")
                if len(lane.data) == 65536:
                    lane.condition.wait(remaining)
                    continue
                n = min(len(raw) - offset, 65536 - len(lane.data))
                lane.data.extend(raw[offset:offset + n])
                offset += n
                lane.condition.notify_all()

    def recv_into(self, view, size):
        deadline, lane = time.monotonic() + self.timeout, self.incoming
        with lane.condition:
            while not lane.data and not lane.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("memory channel read timeout")
                lane.condition.wait(remaining)
            n = min(size, len(lane.data))
            view[:n] = lane.data[:n]
            del lane.data[:n]
            lane.condition.notify_all()
            return n

    def shutdown(self, how):
        with self.outgoing.condition:
            self.outgoing.closed = True
            self.outgoing.condition.notify_all()

    def close(self):
        self.shutdown(socket.SHUT_WR)


class MemoryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.options = Options(enabled=True, chunk_mib=1, depth=2)

    def pair(self):
        incoming, outgoing = _Lane(), _Lane()
        return self.channels(_MemoryEndpoint(incoming, outgoing), _MemoryEndpoint(outgoing, incoming))

    def channels(self, a, b):
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return (ChunkSocket(a, options=self.options, timeout_s=.5),
                ChunkSocket(b, options=self.options, timeout_s=.5))

    def test_default_off(self):
        with self.assertRaisesRegex(Rejected, "disabled"):
            ChunkSocket(None)

    def test_streamed_protect_ack_then_recover_in_memory(self):
        sender, receiver = self.pair()
        a, b = b"a" * (2 * MIB + 11), b"b" * (2 * MIB + 11)
        staged = []
        def receive():
            for _ in range(3):
                receiver.receive(lambda c: staged.append(c) or c.receipt)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(receive)
            cp, stats = protect(a, b, checkpoint="socketpair-memory", options=self.options,
                                send=sender.send)
            future.result(timeout=3)
        self.assertEqual(tuple(staged), cp.chunks)
        self.assertEqual(stats["peak_pending"], 2)
        self.assertFalse(sender.gpudirect)
        sink = MemorySink(cp, 0)
        recover(cp, lost_rank=0, fetch_survivor=lambda i, n: b[i*MIB:i*MIB+n],
                sink=sink, options=self.options)
        self.assertEqual(b"".join(sink.output), a)

    def test_oversize_and_truncated_frames_fail_closed(self):
        for payload in (struct.pack("!I", MAX_HEADER + 1), struct.pack("!I", 20) + b"short"):
            sender, receiver = self.pair()
            sender.sock.sendall(payload)
            sender.sock.shutdown(socket.SHUT_WR)
            with self.assertRaises(Rejected):
                receiver.receive(lambda c: c.receipt)
            self.assertTrue(receiver.broken)

    def test_corrupt_parity_not_sent(self):
        sender, _ = self.pair()
        cp, _ = protect(b"abc", b"def", checkpoint="corrupt", options=self.options)
        with self.assertRaisesRegex(Rejected, "parity_sha256"):
            sender.send(replace(cp.chunks[0], parity=b"bad"))

    def test_no_ack_timeout_breaks_channel_no_reconnect_or_port_change(self):
        sender, _ = self.pair()
        sender.timeout = .02
        cp, _ = protect(b"abc", b"def", checkpoint="timeout", options=self.options)
        with self.assertRaises(TimeoutError):
            sender.send(cp.chunks[0])
        self.assertTrue(sender.broken)
        with self.assertRaises(Rejected):
            sender.send(cp.chunks[0])

    def test_receiver_stage_rejection_never_acks(self):
        sender, receiver = self.pair()
        sender.timeout = .05
        cp, _ = protect(b"abc", b"def", checkpoint="bad-stage", options=self.options)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(receiver.receive, lambda c: None)
            with self.assertRaises(TimeoutError):
                sender.send(cp.chunks[0])
            with self.assertRaisesRegex(Rejected, "did_not_stage"):
                future.result(timeout=1)


@unittest.skipUnless(os.environ.get("KV_EC_HOT_TP2_SOCKETPAIR_SMOKE") == "1",
                     "real kernel socketpair smoke is OFF; protocol tests use memory doubles")
class ExplicitSocketpairSmoke(MemoryProtocolTests):
    def pair(self):
        return self.channels(*socket.socketpair())


if __name__ == "__main__":
    unittest.main()
