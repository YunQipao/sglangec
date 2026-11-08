"""Targeted CPU acceptance; no services, real network, model, or fault signals."""

from dataclasses import replace
import builtins
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from experiments.kv_ec.hot_tp2 import (
    MIB, Options, Rejected, bounded_map, protect, recover, rs_chunk, sha,
    update_dirty, OnlinePredictor, PredictionKey, SameProcessSwitchPrototype,
)
from experiments.kv_ec.hot_tp2_domains import preflight
from experiments.kv_ec.hot_tp2_native import NativeCodec
from experiments.kv_ec.multinode import rs_codec as oracle


class MemorySink:
    def __init__(self, cp, lost):
        self.cp, self.lost, self.output = cp, lost, []
        self.aborted = self.finished = False

    def stage(self, index, payload):
        if index != len(self.output):
            raise AssertionError("unordered sink")
        self.output.append(payload)

    def finish(self):
        self.finished = (len(self.output) == len(self.cp.chunks)
                         and all(sha(v) == c.data_sha[self.lost]
                                 for c, v in zip(self.cp.chunks, self.output)))
        return self.finished

    def abort(self):
        self.aborted = True
        self.output.clear()


class MemoryPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261104)
        cls.a, cls.b = rng.bytes(2 * MIB + 19), rng.bytes(2 * MIB + 19)
        cls.opts = Options(enabled=True, chunk_mib=1, depth=2)

    def checkpoint(self, **kwargs):
        return protect(self.a, self.b, checkpoint="request-epoch-checkpoint-1",
                       options=self.opts, **kwargs)[0]

    def fetch(self, raw):
        return lambda index, size: raw[index * MIB:index * MIB + size]

    def test_defaults_off_and_chunk_candidates(self):
        with self.assertRaisesRegex(Rejected, "disabled"):
            protect(b"a", b"b", checkpoint="x")
        for mib in (1, 4, 16, 64):
            cp, _ = protect(b"a", b"b", checkpoint="x", options=Options(True, mib))
            self.assertEqual(cp.chunk_bytes, mib * MIB)
        for opts in (Options(True, 2), Options(True, True), Options(True, 1, 0),
                     Options(True, 1, 9), Options(True, 1, 2, 0)):
            with self.assertRaises(Rejected):
                opts.require()

    def test_memory_only_both_erasure_directions_and_full_tail(self):
        cp = self.checkpoint()
        self.assertEqual(len(cp.chunks), 3)
        for lost, survivor, expected in ((0, self.b, self.a), (1, self.a, self.b)):
            sink = MemorySink(cp, lost)
            # The fault path cannot open either Python files or io.open paths.
            with patch.object(builtins, "open", side_effect=AssertionError("disk I/O")), \
                    patch.object(io, "open", side_effect=AssertionError("disk I/O")):
                stats = recover(cp, lost_rank=lost, fetch_survivor=self.fetch(survivor),
                                sink=sink, options=self.opts)
            self.assertEqual(b"".join(sink.output), expected)
            self.assertFalse(stats["scheduler_ready"])
            self.assertFalse(stats["gpudirect"])
            self.assertTrue(stats["strict_sha256"])
            self.assertLessEqual(stats["peak_pending"], 2)

    def test_mutable_or_file_backed_inputs_not_admitted(self):
        for a, b in ((bytearray(b"a"), b"b"), (memoryview(b"a"), b"b"),
                     (b"a", b"bb"), (b"", b"")):
            with self.assertRaises(Rejected):
                protect(a, b, checkpoint="x", options=self.opts)
        with tempfile.TemporaryDirectory() as tmp:
            data = np.memmap(Path(tmp) / "input", mode="w+", dtype=np.uint8, shape=(8,))
            with self.assertRaises(Rejected):
                protect(data, data, checkpoint="x", options=self.opts)
            del data

    def test_ack_must_bind_checkpoint_index_generation_codec_and_hash(self):
        for response in (None, (), ("other",)):
            with self.assertRaisesRegex(Rejected, "ack_binding"):
                self.checkpoint(send=lambda c: response)
        seen = []
        cp = self.checkpoint(send=lambda c: seen.append(c.index) or c.receipt)
        self.assertEqual(seen, list(range(len(cp.chunks))))

    def test_strict_corruption_and_worker_error_abort_no_commit(self):
        cp = self.checkpoint()
        for damaged in ("survivor", "parity", "output", "codec", "fetch"):
            with self.subTest(damaged=damaged):
                selected, fetch, codec = cp, self.fetch(self.b), rs_chunk
                if damaged == "parity":
                    c = cp.chunks[0]
                    bad = bytes([c.parity[0] ^ 1]) + c.parity[1:]
                    selected = replace(cp, chunks=(replace(c, parity=bad), *cp.chunks[1:]))
                if damaged == "survivor":
                    fetch = self.fetch(bytes([self.b[0] ^ 1]) + self.b[1:])
                if damaged == "output":
                    codec = lambda a, b, op: b"\x00" * len(a)
                if damaged == "codec":
                    selected = replace(cp, chunks=(replace(cp.chunks[0], codec="xor"), *cp.chunks[1:]))
                if damaged == "fetch":
                    def fetch(index, size):
                        raise TimeoutError("finite transport timeout")
                sink = MemorySink(selected, 0)
                with self.assertRaises((Rejected, TimeoutError)):
                    recover(selected, lost_rank=0, fetch_survivor=fetch, sink=sink,
                            options=self.opts, codec=codec)
                # Manifest rejection happens before acquiring the sink.
                if damaged != "codec":
                    self.assertTrue(sink.aborted)
                self.assertFalse(sink.finished)

    def test_sink_failure_or_missing_verification_cannot_pass(self):
        cp = self.checkpoint()
        for failure in ("stage", "finish"):
            sink = MemorySink(cp, 0)
            if failure == "stage":
                sink.stage = lambda *args: (_ for _ in ()).throw(RuntimeError("sink failed"))
            else:
                sink.finish = lambda: False
            with self.assertRaises((Rejected, RuntimeError)):
                recover(cp, lost_rank=0, fetch_survivor=self.fetch(self.b), sink=sink,
                        options=self.opts)
            self.assertTrue(sink.aborted)

    def test_bounded_pipeline_backpressure_and_error_join(self):
        produced, consumed = [], []

        def source():
            for i in range(9):
                produced.append(i)
                yield i

        def consume(i):
            # Before this ACK, producer cannot advance beyond depth credits.
            self.assertLessEqual(len(produced) - len(consumed), 2)
            consumed.append(i)

        stats = bounded_map(source(), lambda i: i, consume, depth=2)
        self.assertEqual(consumed, list(range(9)))
        self.assertEqual(stats["peak_pending"], 2)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            bounded_map(range(99), lambda i: i,
                        lambda i: (_ for _ in ()).throw(RuntimeError("stop")), depth=2)
        self.assertFalse(any(t.name.startswith("hot-tp2") for t in threading.enumerate()))

    def test_dirty_only_hashes_and_replaces_changed_chunk(self):
        cp = self.checkpoint()
        index, old0, old1 = 1, self.a[MIB:2*MIB], self.b[MIB:2*MIB]
        new0, new1 = old0[::-1], old1[::-1]
        seen = []
        with patch("experiments.kv_ec.hot_tp2.sha", wraps=sha) as hashes:
            updated, stats = update_dirty(cp, {index: (0, old0, old1, new0, new1)},
                                         options=self.opts, send=lambda c: seen.append(c.index) or c.receipt)
        self.assertEqual(hashes.call_count, 6)  # old D0,D1,P; new D0,D1,P
        self.assertEqual(stats, dict(dirty_chunks=1, updated_bytes=MIB, clean_chunks_hashed=0))
        self.assertEqual(seen, [1])
        self.assertIs(cp.chunks[0], updated.chunks[0])
        self.assertIs(cp.chunks[2], updated.chunks[2])
        self.assertEqual(updated.chunks[1].generation, 1)
        expected = self.a[:MIB] + new0 + self.a[2*MIB:]
        survivor = self.b[:MIB] + new1 + self.b[2*MIB:]
        sink = MemorySink(updated, 0)
        recover(updated, lost_rank=0, fetch_survivor=self.fetch(survivor),
                sink=sink, options=self.opts)
        self.assertEqual(b"".join(sink.output), expected)

    def test_dirty_error_keeps_original_checkpoint_and_rejects_replay(self):
        cp = self.checkpoint()
        old0, old1 = self.a[:MIB], self.b[:MIB]
        for change in ((1, old0, old1, old0, old1),
                       (0, old1, old0, old0, old1)):
            with self.assertRaises(Rejected):
                update_dirty(cp, {0: change}, options=self.opts)
        with self.assertRaisesRegex(Rejected, "ack_binding"):
            update_dirty(cp, {0: (0, old0, old1, old0, old1)}, options=self.opts,
                         send=lambda c: cp.chunks[0].receipt)
        self.assertEqual(cp.chunks[0].generation, 0)
        unchanged, stats = update_dirty(cp, {}, options=self.opts)
        self.assertEqual(unchanged, cp)
        self.assertEqual(stats["updated_bytes"], 0)


class CodecOracleTests(unittest.TestCase):
    def test_all_byte_pairs_against_existing_gf256_oracle(self):
        a = np.repeat(np.arange(256, dtype=np.uint8), 256)
        b = np.tile(np.arange(256, dtype=np.uint8), 256)
        _, _, p = oracle.encode(a, b)
        self.assertEqual(rs_chunk(a.tobytes(), b.tobytes(), "encode"), p.tobytes())
        self.assertEqual(rs_chunk(p.tobytes(), b.tobytes(), "recover0"), a.tobytes())
        self.assertEqual(rs_chunk(p.tobytes(), a.tobytes(), "recover1"), b.tobytes())
        self.assertNotEqual(p.tobytes(), np.bitwise_xor(a, b).tobytes())

    def test_native_default_off(self):
        with self.assertRaisesRegex(Rejected, "disabled"):
            NativeCodec("/no/library")

    def test_cpp_vectorized_codec_and_pipeline_against_oracle(self):
        import shutil
        if shutil.which("g++") is None:
            self.skipTest("g++ unavailable; native CPU NOT VERIFIED")
        source = Path(__file__).parent / "native/hot_rs21.cpp"
        with tempfile.TemporaryDirectory(prefix="hot-rs21-test-") as tmp:
            library = Path(tmp) / "rs.so"
            result = subprocess.run(["g++", "-O3", "-Wall", "-Wextra", "-Werror", "-std=c++17", "-shared", "-fPIC",
                                     str(source), "-o", str(library)], capture_output=True,
                                    text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            native = NativeCodec(library, enabled=True)
            for size in (1, 7, 8, 9, 31, 257, 65536):
                a = np.resize(np.repeat(np.arange(256, dtype=np.uint8), 256), size)
                b = np.resize(np.tile(np.arange(256, dtype=np.uint8), 256), size)
                _, _, p = oracle.encode(a, b)
                self.assertEqual(native(a.tobytes(), b.tobytes(), "encode"), p.tobytes())
                self.assertEqual(native(p.tobytes(), b.tobytes(), "recover0"), a.tobytes())
                self.assertEqual(native(p.tobytes(), a.tobytes(), "recover1"), b.tobytes())
                self.assertEqual(native(a.tobytes(), b.tobytes(), "xor"), (a ^ b).tobytes())
            opts = Options(enabled=True, chunk_mib=1)
            cp, _ = protect(b"abc" * 113, b"def" * 113, checkpoint="native",
                            codec=native, options=opts)
            sink = MemorySink(cp, 1)
            recover(cp, lost_rank=1, fetch_survivor=lambda i, n: b"abc" * 113,
                    codec=native, sink=sink, options=opts)
            self.assertEqual(sink.output, [b"def" * 113])


class PolicyAndSwitchTests(unittest.TestCase):
    def setUp(self):
        self.key = PredictionKey("model-sha", "pid-start-boot", "epoch", MIB, 256, 1, "host-tcp")
        self.policy = OnlinePredictor(enabled=True)

    def samples(self):
        for path, ms in (("EC", 5), ("RE_PREFILL", 9)):
            for now in (10, 11):
                self.policy.observe(self.key, path, ms, model_already_resident=True,
                                    full_boundary=True, now=now)

    def test_policy_default_off_and_no_optimistic_fallback(self):
        self.assertEqual(OnlinePredictor().choose(self.key, ec_committed=True,
                                                 live_request_valid=True)["action"], "FAIL_CLOSED")
        result = self.policy.choose(self.key, ec_committed=True, live_request_valid=True, now=10)
        self.assertEqual(result["action"], "FAIL_CLOSED")

    def test_same_resident_model_and_boundary_predicts_and_records_reason(self):
        self.samples()
        result = self.policy.choose(self.key, ec_committed=True, live_request_valid=True, now=12)
        self.assertEqual(result["action"], "EC")
        self.assertEqual(result["reason"], "minimum_same_boundary_online_ewma")
        self.assertAlmostEqual(result["predictions_ms"]["EC"], 5)
        self.assertEqual(result["key"], self.key.__dict__)
        self.assertEqual(self.policy.choose(self.key, ec_committed=False,
                                            live_request_valid=True, now=12)["action"], "RE_PREFILL")
        for now, key, live in ((400, self.key, True), (12, replace(self.key, tokens=512), True),
                               (12, replace(self.key, resident_process="another-model"), True),
                               (12, self.key, False), (0, self.key, True)):
            self.assertEqual(self.policy.choose(key, ec_committed=True, live_request_valid=live,
                                                now=now)["action"], "FAIL_CLOSED")

    def test_cold_partial_or_invalid_samples_rejected(self):
        for key, ms, resident, full in ((self.key, 1, False, True), (self.key, 1, True, False),
                                       (self.key, float("nan"), True, True),
                                       (replace(self.key, verification="edge"), 1, True, True),
                                       (replace(self.key, boundary="kv_copy_only"), 1, True, True)):
            with self.assertRaises(Rejected):
                self.policy.observe(key, "EC", ms, model_already_resident=resident,
                                    full_boundary=full)

    def test_reprefill_can_win_and_expired_history_requires_new_samples(self):
        self.samples()
        for now in (12, 13):
            self.policy.observe(self.key, "RE_PREFILL", 1, model_already_resident=True,
                                full_boundary=True, now=now)
        choice = self.policy.choose(self.key, ec_committed=True, live_request_valid=True, now=14)
        self.assertEqual(choice["action"], "RE_PREFILL")
        self.assertLess(choice["predictions_ms"]["RE_PREFILL"], choice["predictions_ms"]["EC"])
        self.policy.observe(self.key, "EC", 1, model_already_resident=True,
                            full_boundary=True, now=500)
        self.assertEqual(self.policy.samples[self.key, "EC"][0], 1)
        self.assertEqual(self.policy.choose(self.key, ec_committed=True, live_request_valid=True,
                                            now=501)["action"], "FAIL_CLOSED")

    def test_same_process_compatibility_wrapper_requires_real_commit(self):
        class Event:
            def __init__(self): self.waited = False
            def synchronize(self): self.waited = True
            def query(self): return self.waited
        event = Event()
        args = dict(before=("process", "model", "request", "generation"),
                    current=("process", "model", "request", "generation"),
                    model_ready_ns=1, fault_ns=2, stage_b_prefill_count=0,
                    verification_event=event)
        for switch, reason in ((SameProcessSwitchPrototype(), "disabled"),
                               (SameProcessSwitchPrototype(enabled=True), "callback_required")):
            with self.assertRaisesRegex(Rejected, reason):
                switch.submit(**args)
        self.assertTrue(event.waited)
        committed = SameProcessSwitchPrototype(enabled=True).submit(
            **args, scheduler_commit=lambda: {"state": "COMMITTED", "prefill": 0}
        )
        self.assertEqual(committed, {"state": "COMMITTED", "prefill": 0})
        for change in (dict(stage_b_prefill_count=1), dict(current=("changed",)),
                       dict(model_ready_ns=3), dict(verification_event=None)):
            with self.assertRaises(Rejected):
                SameProcessSwitchPrototype(enabled=True).submit(**(args | change))


class ThreeDomainTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "design_study/placement.json"
        self.config = json.loads(path.read_text())

    def probe(self, node, challenge, timeout):
        return dict(challenge=challenge, physical_id=node["physical_id"], host=node["host"],
                    machine_id=hashlib.md5(node["physical_id"].encode()).hexdigest(),
                    local_interface_verified=True, authenticated=True)

    def test_default_off_and_no_probe_blocked(self):
        with self.assertRaisesRegex(Rejected, "disabled"):
            preflight(self.config)
        self.assertEqual(preflight(self.config, enabled=True)["status"], "BLOCKED")

    def test_three_fresh_identities_are_only_preflight_not_fault_proof(self):
        result = preflight(self.config, enabled=True, probe=self.probe)
        self.assertEqual(result["status"], "PREFLIGHT_ONLY")
        self.assertFalse(result["physical_node_fault_tolerance_verified"])

    def test_245_alias_and_duplicate_physical_host_rejected(self):
        self.config["nodes"][2]["host"] = "192.168.2.245"
        with self.assertRaisesRegex(ValueError, "physical address"):
            preflight(self.config, enabled=True, probe=self.probe)
        self.setUp()
        def duplicate(node, challenge, timeout):
            return self.probe(node, challenge, timeout) | dict(machine_id="abcdef01" * 4)
        with self.assertRaisesRegex(Rejected, "distinct_physical"):
            preflight(self.config, enabled=True, probe=duplicate)

    def test_243_unreachable_never_accepted(self):
        def unavailable(node, challenge, timeout):
            if node["physical_id"] == "243":
                raise TimeoutError("unreachable")
            return self.probe(node, challenge, timeout)
        result = preflight(self.config, enabled=True, probe=unavailable)
        self.assertEqual((result["status"], result["blocked_role"]), ("BLOCKED", "P"))
        self.assertFalse(result["physical_node_fault_tolerance_verified"])

    def test_unbound_or_unauthenticated_evidence_rejected(self):
        for overrides in (dict(challenge="old"), dict(authenticated=False),
                          dict(local_interface_verified=False), dict(machine_id="233")):
            with self.assertRaises(Rejected):
                preflight(self.config, enabled=True, probe=lambda n, c, t: self.probe(n, c, t) | overrides)


if __name__ == "__main__":
    unittest.main()
