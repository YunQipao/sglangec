import hashlib
import socket
import threading
import unittest

if __package__:
    from .gateway_client import send_chunk
    from .gateway_protocol import PROTOCOL
    from .gateway_server import (RelayState, _BoundedTCPServer, make_gateway_handler,
                                 make_receiver_handler, ReceiverState)
    from .gateway_pipeline import verify_a_c, ChunkXorAggregator
else:
    from gateway_client import send_chunk
    from gateway_protocol import PROTOCOL
    from gateway_server import (RelayState, _BoundedTCPServer, make_gateway_handler,
                                make_receiver_handler, ReceiverState)
    from gateway_pipeline import verify_a_c, ChunkXorAggregator


def sockets_available():
    try:
        sock = socket.socket()
        sock.close()
        return True
    except OSError:
        return False


TCP_TESTS_AVAILABLE = sockets_available()


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class GatewayPrototypeTest(unittest.TestCase):
    def setUp(self):
        if self._testMethodName not in ("test_real_loopback_relay_and_idempotent_retry",
                                        "test_conflicting_duplicate_is_rejected",
                                        "test_epoch_mismatch_is_rejected"):
            return
        self.epoch = "epoch-test"
        self.receiver_state = ReceiverState(manager_epoch=self.epoch, max_queue=8)
        self.receiver = _BoundedTCPServer(("127.0.0.1", free_port()),
                                          make_receiver_handler(self.receiver_state), max_connections=4)
        self.gateway_state = RelayState(self.receiver.server_address, manager_epoch=self.epoch)
        self.gateway = _BoundedTCPServer(("127.0.0.1", free_port()),
                                         make_gateway_handler(self.gateway_state), max_connections=4)
        self.threads = [threading.Thread(target=s.serve_forever, daemon=True)
                        for s in (self.receiver, self.gateway)]
        for thread in self.threads:
            thread.start()

    def tearDown(self):
        if not hasattr(self, "gateway"):
            return
        for server in (self.gateway, self.receiver):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)

    def header(self, payload, **updates):
        result = dict(transaction_id="tx", stripe_id="s0", checkpoint_id="cp-v1",
                      rank_id=0, chunk_index=0, chunk_offset=0,
                      chunk_length=len(payload), total_length=len(payload),
                      source_node="node0", destination_node="node1",
                      manager_epoch=getattr(self, "epoch", "epoch-test"), protocol_version=PROTOCOL,
                      chunk_sha256=hashlib.sha256(payload).hexdigest())
        result.update(updates)
        return result

    @unittest.skipUnless(TCP_TESTS_AVAILABLE, "sandbox denies TCP sockets; run on target Linux host")
    def test_real_loopback_relay_and_idempotent_retry(self):
        payload = bytes(range(256)) * 1024
        header = self.header(payload)
        address = self.gateway.server_address
        first = send_chunk(address, header, payload)
        second = send_chunk(address, header, payload)
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        received_header, received_payload = self.receiver_state.items.get(timeout=1)
        self.assertEqual(received_header, header)
        self.assertEqual(received_payload, payload)
        self.assertEqual(self.receiver_state.received_bytes, len(payload))
        self.assertEqual(self.gateway_state.metrics["chunks"], 1)
        self.assertEqual(self.gateway_state.metrics["rx_bytes"], len(payload))

    @unittest.skipUnless(TCP_TESTS_AVAILABLE, "sandbox denies TCP sockets; run on target Linux host")
    def test_conflicting_duplicate_is_rejected(self):
        a, b = b"abcd", b"wxyz"
        h = self.header(a)
        send_chunk(self.gateway.server_address, h, a)
        with self.assertRaises(RuntimeError):
            send_chunk(self.gateway.server_address, self.header(b), b)
        self.assertEqual(self.receiver_state.received_chunks, 1)

    @unittest.skipUnless(TCP_TESTS_AVAILABLE, "sandbox denies TCP sockets; run on target Linux host")
    def test_epoch_mismatch_is_rejected(self):
        payload = b"test"
        h = self.header(payload, manager_epoch="old-epoch")
        with self.assertRaises(RuntimeError):
            send_chunk(self.gateway.server_address, h, payload)
        self.assertEqual(self.receiver_state.received_chunks, 0)

    def test_a_c_xor_byte_exact_and_c_emits_incrementally(self):
        blocks = [bytes((i * 3 + rank) % 256 for i in range(9 * 1024 * 1024))
                  for rank in range(4)]
        emitted = []
        if __package__:
            from .gateway_pipeline import scheme_a, scheme_c
        else:
            from gateway_pipeline import scheme_a, scheme_c
        a, _ = scheme_a(blocks)
        c, metrics = scheme_c(blocks, chunk_bytes=256 * 1024,
                              on_parity_chunk=lambda index, offset, chunk: emitted.append((index, offset, len(chunk))))
        self.assertEqual(a, c)
        self.assertEqual(metrics["chunk_count"], 36)
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted[0], (0, 0, 256 * 1024))
        self.assertLess(metrics["peak_host_bytes"], len(a) * 4)
        self.assertEqual(verify_a_c(blocks)["status"], "byte_exact")

    def test_stream_aggregator_emits_before_full_checkpoint_arrives(self):
        blocks = [bytes((i + rank * 17) % 256 for i in range(1000)) for rank in range(4)]
        output = []
        aggregator = ChunkXorAggregator(rank_ids=range(4), total_length=1000,
            chunk_bytes=128, on_parity_chunk=lambda i, o, b: output.append((i, o, b)),
            max_inflight_chunks=2)
        # Offset 0 becomes parity-complete while the remaining 872 bytes per
        # rank have not yet been submitted.
        for rank in range(4):
            aggregator.push(rank_id=rank, chunk_index=0, chunk_offset=0,
                total_length=1000, payload=blocks[rank][:128])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0][2], bytes(byte ^ blocks[1][i] ^ blocks[2][i] ^ blocks[3][i]
                                              for i, byte in enumerate(blocks[0][:128])))
        # Finish all remaining chunks, intentionally interleaving rank/chunk.
        for index in range(1, 8):
            start = index * 128
            end = min(start + 128, 1000)
            for rank in range(4):
                aggregator.push(rank_id=rank, chunk_index=index, chunk_offset=start,
                    total_length=1000, payload=blocks[rank][start:end])
        self.assertEqual(aggregator.finish()["chunks_committed"], 8)

    def test_protocol_rejects_oversized_and_inconsistent_chunks(self):
        if __package__:
            from .gateway_protocol import MAX_CHUNK_BYTES, validate_header
        else:
            from gateway_protocol import MAX_CHUNK_BYTES, validate_header
        header = self.header(b"1234", chunk_length=MAX_CHUNK_BYTES + 1,
                             total_length=MAX_CHUNK_BYTES + 1)
        with self.assertRaises(ValueError):
            validate_header(header)


if __name__ == "__main__":
    unittest.main()
