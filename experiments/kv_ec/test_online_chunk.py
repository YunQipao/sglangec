"""CPU fixtures and real-CUDA staging tests (skipped without a CUDA device)."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import queue
import socket
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.kv_ec.debug_capture import CaptureRejected
from experiments.kv_ec.node_block_export import pack, unpack
from experiments.kv_ec.online_chunk import (
    OnlineProtector,
    StagedChunk,
    completed_chunks,
    finish_metadata,
    gather_layers,
    validate_runtime,
)
from experiments.kv_ec.online_chunk_manager import OnlineChunkManager, serve
from experiments.kv_ec.rank_snapshot import write_rank_files
from experiments.kv_ec.test_debug_capture import lifetime_fixture, rank_fixture
from experiments.kv_ec.verify_online_chunk import verify_reference


def online_fixture(rank, tp=2, chunk=0):
    payload, meta = rank_fixture(rank, tp, layers=range(36), count=256, heads=1)
    checkpoint = hashlib.sha256(
        f"run:{meta['request_id']}:{chunk}".encode()
    ).hexdigest()
    meta.update(
        chunk_id=chunk,
        token_range=[chunk * 256, (chunk + 1) * 256],
        layer_range=[0, 36],
        token_start=chunk * 256,
        checkpoint_version=checkpoint,
        snapshot_id=checkpoint,
        manager_epoch="run",
        request_version="run",
        model_version=meta["model_sha256"],
        performance_ms={"gpu_to_host_ms": 1, "queue_wait_ms": 2},
    )
    return payload, meta


class ChunkTests(unittest.TestCase):
    def test_actual_gather_matches_independent_offline_layout(self):
        for tp in (2, 4):
            heads = 8 // tp
            k = (
                torch.arange(300 * heads * 128, dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
                .reshape(300, heads, 128)
            )
            v = k.flip(0).contiguous()
            slots = torch.arange(280, 24, -1)
            packed = torch.empty(36 * 256 * heads * 128 * 4, dtype=torch.uint8)
            records = gather_layers(
                NS(get_kv_buffer=lambda _: (k, v)), slots, tp, packed
            )
            expected = np.concatenate(
                [
                    t.index_select(0, slots).view(torch.uint8).flatten().numpy()
                    for _ in range(36)
                    for t in (k, v)
                ]
            )
            k.zero_()
            v.zero_()
            self.assertTrue(np.array_equal(packed.numpy(), expected))
            self.assertEqual(
                records[-1]["v_offset"] + records[-1]["v_nbytes"], packed.numel()
            )

    def test_completed_ranges_and_sampled_token_not_yet_in_kv(self):
        _, batch, req, _ = lifetime_fixture()
        req.origin_input_ids = [1] * 255
        req.kv_committed_len = 255
        batch.seq_lens_cpu = [255]
        self.assertEqual(list(completed_chunks(req, batch, 0, 0)), [])
        req.output_ids = [2]
        self.assertEqual(list(completed_chunks(req, batch, 0, 0)), [])
        batch.forward_mode.name = "DECODE"
        req.kv_committed_len = batch.seq_lens_cpu[0] = 256
        self.assertEqual(list(completed_chunks(req, batch, 0, 0)), [0])
        self.assertEqual(list(completed_chunks(req, batch, 0, 1)), [])
        req.origin_input_ids = [1] * 768
        req.kv_committed_len = batch.seq_lens_cpu[0] = 768
        self.assertEqual(list(completed_chunks(req, batch, 0, 1)), [1, 2])

    def test_incomplete_abort_retract_and_length_mismatch(self):
        _, batch, req, _ = lifetime_fixture()
        for field, value in (
            ("is_retracted", True),
            ("to_finish", object()),
            ("req_pool_idx", None),
        ):
            old = getattr(req, field, None)
            setattr(req, field, value)
            self.assertEqual(list(completed_chunks(req, batch, 0, 0)), [])
            setattr(req, field, old)
        req.kv_committed_len += 1
        with self.assertRaises(CaptureRejected):
            completed_chunks(req, batch, 0, 0)

    def test_runtime_rejects_unsafe_modes_before_reading_pool(self):
        scheduler, _, _, _ = lifetime_fixture()
        validate_runtime(scheduler)
        for field in ("enable_overlap",):
            setattr(scheduler, field, True)
            with self.assertRaises(CaptureRejected):
                validate_runtime(scheduler)
            setattr(scheduler, field, False)
        scheduler.server_args.disable_radix_cache = False
        with self.assertRaises(CaptureRejected):
            validate_runtime(scheduler)

    def protector(self):
        s, batch, req, _ = lifetime_fixture()
        s.req_to_token_pool = NS(req_generation=[0] * 8, free_slots=[])
        s.req_to_token_pool.req_generation[7] = 1
        p = OnlineProtector.__new__(OnlineProtector)
        p.scheduler, p.active, p.metrics = s, {}, queue.Queue(maxsize=128)
        p.capacity, p.queue = threading.BoundedSemaphore(1), queue.Queue(1)
        p._stage = Mock(side_effect=lambda *_: NS(cancelled=None))
        return p, batch, req

    def test_backpressure_reserves_before_staging_and_deduplicates(self):
        p, batch, req = self.protector()
        p.after_forward(batch)
        p.after_forward(batch)
        self.assertEqual(p._stage.call_count, 1)
        req.origin_input_ids *= 2
        req.kv_committed_len = batch.seq_lens_cpu[0] = 768
        p.after_forward(batch)
        self.assertEqual(p._stage.call_count, 1)
        p.queue.get_nowait()
        p.capacity.release()
        p.after_forward(batch)
        self.assertEqual(p._stage.call_args.args[1], 1)

    def test_cancel_recycle_and_request_end_cleanup(self):
        for normal in (False, True):
            p, batch, req = self.protector()
            p.after_forward(batch)
            staged = p.queue.get_nowait()
            req.finished = lambda: normal
            req.is_retracted = not normal
            p.scheduler.req_to_token_pool.free_slots.append(7)
            p.after_result(batch)
            self.assertFalse(p.active)
            self.assertEqual(staged.cancelled.is_set(), not normal)
        p, batch, req = self.protector()
        p.after_forward(batch)
        old = p.queue.get_nowait()
        p.capacity.release()
        p.scheduler.req_to_token_pool.req_generation[7] += 1
        p.after_forward(batch)
        self.assertTrue(old.cancelled.is_set())

    def test_scheduler_trigger_order_and_default_off(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/srt/managers/scheduler.py"
        )
        tree = ast.parse(path.read_text())
        scheduler = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        loop = next(
            n
            for n in scheduler.body
            if isinstance(n, ast.FunctionDef) and n.name == "event_loop_normal"
        )
        calls = [
            (n.lineno, n.func.attr)
            for n in ast.walk(loop)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        positions = {name: line for line, name in calls}
        self.assertLess(positions["run_batch"], positions["after_forward"])
        self.assertLess(positions["after_forward"], positions["process_batch_result"])
        self.assertLess(positions["process_batch_result"], positions["after_result"])
        self.assertIn('os.environ.get("SGLANG_KV_EC_ONLINE") == "1"', path.read_text())

    def test_weight_or_storage_mutation_disables_old_model_epoch(self):
        for name in (
            "UpdateWeightFromDiskReqInput",
            "UpdateWeightsFromTensorReqInput",
            "ReleaseMemoryOccupationReqInput",
            "ResumeMemoryOccupationReqInput",
        ):
            p, batch, _ = self.protector()
            p.after_forward(batch)
            item = p.queue.get_nowait()
            p.before_inputs([type(name, (), {})()])
            p.after_forward(batch)
            self.assertFalse(p.enabled)
            self.assertTrue(item.cancelled.is_set())
            self.assertFalse(p.active)
            self.assertEqual(p._stage.call_count, 1)


class ManagerTests(unittest.TestCase):
    def test_tp2_tp4_reference_bytes_and_every_rank_recovery(self):
        for tp in (2, 4):
            with self.subTest(tp=tp), tempfile.TemporaryDirectory() as tmp:
                manager = OnlineChunkManager(
                    Path(tmp) / "online", "run", retain_ranks=True, test_recovery=True
                )
                reference = Path(tmp) / "reference"
                for rank in reversed(range(tp)):
                    block, meta = online_fixture(rank, tp)
                    offline = deepcopy(meta)
                    offline["snapshot_id"] = "offline"
                    write_rank_files(reference, block, offline)
                    status = manager.submit(meta, block)
                self.assertEqual(status["state"], "COMMITTED")
                online = manager.output / status["snapshot_id"]
                result = verify_reference(online, reference)
                self.assertTrue(result["byte_exact"])
                self.assertEqual(result["recovered_ranks"], list(range(tp)))
                entries = json.loads((online / "manifest.json").read_text())["ranks"]
                raw_blocks = [
                    (online / f"rank_{i}.bin").read_bytes() for i in range(tp)
                ]
                wire = pack(0, raw_blocks, entries, allow_fixture=True)
                _, unpacked = unpack(wire, allow_fixture=True)
                self.assertEqual(unpacked, tuple(raw_blocks))
                self.assertFalse(manager.pending)
                self.assertTrue(manager.submit(meta, block)["duplicate"])
                conflict = deepcopy(meta)
                conflict["token_ids_sha256"] = "different-request-reusing-rid"
                self.assertEqual(manager.submit(conflict, block)["state"], "FAILED")
                with (online / "parity.bin").open("r+b") as stream:
                    stream.write(b"bad")
                with self.assertRaises(ValueError):
                    verify_reference(online, reference)

    def test_mismatch_duplicate_timeout_capacity_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = OnlineChunkManager(Path(tmp) / "online", "run", max_pending=1)
            block, meta = online_fixture(0)
            self.assertEqual(manager.submit(meta, block)["state"], "PENDING")
            self.assertTrue(manager.submit(meta, block)["duplicate"])
            other, other_meta = online_fixture(0, chunk=1)
            self.assertEqual(manager.submit(other_meta, other)["state"], "DROPPED")
            manager.pending[meta["snapshot_id"]]["created"] -= 31
            manager.expire()
            self.assertEqual(manager.submit(meta, block)["state"], "EXPIRED")
            corrupt = other.copy()
            corrupt[0] ^= 1
            with self.assertRaises(ValueError):
                manager.submit(other_meta, corrupt)
            other_meta["checkpoint_version"] = "stale"
            with self.assertRaises(ValueError):
                manager.submit(other_meta, other)

    def test_no_commit_on_request_model_token_or_layer_mismatch(self):
        for field in ("request_id", "model_sha256", "token_ids_sha256", "layer_range"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                manager = OnlineChunkManager(Path(tmp) / "online", "run")
                block, meta = online_fixture(0)
                manager.submit(meta, block)
                block, meta = online_fixture(1)
                meta[field] = "wrong"
                try:
                    status = manager.submit(meta, block)
                except (ValueError, TypeError):
                    status = {"state": "FAILED"}
                self.assertNotEqual(status["state"], "COMMITTED")
                self.assertFalse(list(manager.output.glob("*/manifest.json")))

    def test_partial_group_cannot_reuse_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "online"
            OnlineChunkManager(path, "run")
            with self.assertRaises(FileExistsError):
                OnlineChunkManager(path, "run")


class AsyncWorkerTests(unittest.TestCase):
    def worker(self, path):
        p = OnlineProtector.__new__(OnlineProtector)
        p.path, p.timeout = str(path), 2
        p.queue, p.metrics = queue.Queue(maxsize=1), queue.Queue(maxsize=128)
        p.capacity = threading.BoundedSemaphore(1)
        p.capacity.acquire()
        p.stop = threading.Event()
        p.stop.set()  # drain this one queued task, then exit
        return p

    def test_cancel_waits_for_d2h_without_reading_host_or_opening_socket(self):
        p = self.worker("/unused")
        block, meta = online_fixture(0)
        cancelled = threading.Event()
        cancelled.set()
        done = Mock()
        host = Mock()
        p.queue.put(
            StagedChunk(meta, host, None, done, Mock(), time.perf_counter(), cancelled)
        )
        with patch("experiments.kv_ec.online_chunk.socket.socket") as sock:
            p._run()
        done.synchronize.assert_called_once()
        host.numpy.assert_not_called()
        sock.assert_not_called()
        self.assertTrue(p.capacity.acquire(blocking=False))

    def test_worker_completion_and_cpu_aggregation_with_mock_transport(self):
        # Always runnable even when the sandbox forbids socket.bind. Only the
        # transport is mocked: hashing, manager validation, XOR and persistence
        # execute their real implementations.
        with tempfile.TemporaryDirectory() as tmp:
            manager = OnlineChunkManager(
                Path(tmp) / "online", "run", test_recovery=True
            )
            for rank in range(2):
                p = self.worker("/unused")
                block, meta = online_fixture(rank)
                host = torch.zeros(block.size, dtype=torch.uint8)
                done = NS(synchronize=lambda: host.copy_(torch.from_numpy(block)))
                begin = NS(elapsed_time=lambda _: 1.25)
                p.queue.put(
                    StagedChunk(meta, host, None, done, begin, time.perf_counter())
                )
                reply = []

                def send(_sock, metadata, raw):
                    reply.append(
                        manager.submit(metadata, np.frombuffer(raw, dtype=np.uint8))
                    )

                with patch("experiments.kv_ec.online_chunk.socket.socket"), patch(
                    "experiments.kv_ec.online_chunk.send_message", side_effect=send
                ), patch(
                    "experiments.kv_ec.online_chunk.recv_message",
                    side_effect=lambda _: (reply[-1], b""),
                ):
                    p._run()
                self.assertTrue(p.capacity.acquire(blocking=False))
                self.assertEqual(
                    reply[-1]["state"], "PENDING" if rank == 0 else "COMMITTED"
                )
            self.assertEqual(len(list(manager.output.glob("*/manifest.json"))), 1)

    def test_background_worker_to_real_local_socket_and_xor(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.bind(str(Path(tmp) / "probe.sock"))
                (Path(tmp) / "probe.sock").unlink()
            except PermissionError:
                self.skipTest("Unix socket bind unavailable in sandbox")
            manager = OnlineChunkManager(
                Path(tmp) / "online", "run", test_recovery=True
            )
            path = Path(tmp) / "manager.sock"
            stop, ready = threading.Event(), threading.Event()
            server = threading.Thread(
                target=serve,
                args=(path, manager),
                kwargs={"stop": stop, "ready": ready},
                daemon=True,
            )
            server.start()
            self.assertTrue(ready.wait(3))
            try:
                for rank in range(2):
                    p = self.worker(path)
                    block, meta = online_fixture(rank)
                    host = torch.zeros(block.size, dtype=torch.uint8)
                    # The event populates Host storage. Hashing/transmission
                    # before synchronize would submit invalid zero bytes.
                    done = NS(synchronize=lambda: host.copy_(torch.from_numpy(block)))
                    begin = NS(elapsed_time=lambda _: 1.25)
                    p.queue.put(
                        StagedChunk(meta, host, None, done, begin, time.perf_counter())
                    )
                    p._run()
                    self.assertTrue(p.capacity.acquire(blocking=False))
                manifests = list(manager.output.glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                evidence = json.loads(manifests[0].read_text())
                self.assertEqual(len(evidence["recovery_checks"]), 2)
                self.assertEqual(
                    evidence["rank_performance_ms"][0]["gpu_to_host_ms"], 1.25
                )
            finally:
                stop.set()
                server.join(3)
                self.assertFalse(server.is_alive())


@unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA device")
class CudaStagingTests(unittest.TestCase):
    def test_detached_staging_equals_offline_after_pool_reuse(self):
        # Real CUDA streams/copies, synthetic tensors, no model required. The
        # serving integration still needs an actual Qwen3 TP server run.
        for tp in (2, 4):
            for rank in range(tp):
                with self.subTest(tp=tp, rank=rank):
                    heads = 8 // tp
                    k = (
                        torch.arange(
                            300 * heads * 128, device="cuda", dtype=torch.int32
                        )
                        .to(torch.int16)
                        .view(torch.bfloat16)
                        .reshape(300, heads, 128)
                    )
                    v = k.clone()
                    slots = torch.randperm(299, device="cuda")[:256] + 1
                    rp = NS(
                        req_to_token=slots.unsqueeze(0).expand(2, -1),
                        req_generation=[0, 1],
                        free_slots=[],
                    )
                    pool = NS(
                        k_buffer=[k],
                        size=299,
                        dtype=torch.bfloat16,
                        get_kv_buffer=lambda _: (k, v),
                    )
                    scheduler = NS(
                        req_to_token_pool=rp,
                        model_worker=NS(
                            ps=NS(tp_size=tp, tp_rank=rank),
                            model_runner=NS(token_to_kv_pool=pool),
                        ),
                    )
                    p = OnlineProtector.__new__(OnlineProtector)
                    p.scheduler, p.max_bytes, p.run_id = scheduler, 64 * 1024**2, "run"
                    p.identity = {"model_sha256": "fixture"}
                    p.stream = torch.cuda.Stream()
                    req = NS(
                        req_pool_idx=1,
                        rid="fixture",
                        origin_input_ids=[1] * 256,
                        output_ids=[],
                    )
                    # Independent offline reference: select BF16 token rows,
                    # then reinterpret bytes, instead of selecting byte views.
                    reference = np.concatenate(
                        [
                            t.index_select(0, slots)
                            .contiguous()
                            .view(torch.uint8)
                            .flatten()
                            .cpu()
                            .numpy()
                            for _ in range(36)
                            for t in (k, v)
                        ]
                    )
                    item = p._stage(req, 0, 1)
                    k.zero_()
                    v.zero_()
                    rp.free_slots.append(1)
                    rp.req_generation[1] += 1
                    item.done.synchronize()
                    self.assertTrue(np.array_equal(item.host.numpy(), reference))
                    finish_metadata(item.metadata, item.host.numpy())
                    with self.assertRaises(CaptureRejected):
                        p._stage(req, 0, 1)


if __name__ == "__main__":
    unittest.main()
