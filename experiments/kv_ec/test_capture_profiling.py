"""CPU-backed capture integration; mocked CUDA is never real GPU evidence."""

from contextlib import ExitStack
import json
import hashlib
import os
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.kv_ec import debug_capture
from experiments.kv_ec.capture_plan import capture_plan
from experiments.kv_ec.capture_timing import CaptureTiming
from experiments.kv_ec.parity_manager import AsyncParityClient
from experiments.kv_ec.profile_requests import validate_timing
from experiments.kv_ec.rank_snapshot import validate_payload
from experiments.kv_ec.test_debug_capture import lifetime_fixture


class CPUBackedTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")


class ProfilingTests(unittest.TestCase):
    def test_exclusive_wall_and_thread_cpu(self):
        clock = iter((0, 1_000_000, 4_000_000, 10_000_000))
        with patch("experiments.kv_ec.capture_timing.time.perf_counter_ns", side_effect=lambda: next(clock)), \
             patch("experiments.kv_ec.capture_timing.time.thread_time_ns", side_effect=(0, 2_000_000)):
            timing = CaptureTiming(detailed=True)
            with timing.region("gather", 0):
                with self.assertRaises(RuntimeError):
                    with timing.region("nested"):
                        pass
            result = timing.finish()
        self.assertEqual(result["wall_regions_ms"], {"gather": 3.0})
        self.assertEqual(result["thread_cpu_regions_ms"], {"gather": 2.0})
        self.assertEqual(result["unaccounted_ms"], 7.0)
        self.assertEqual(result["layer_wall_ms"], {"0": {"gather": 3.0}})

    def test_bounded_unique_plan(self):
        self.assertEqual(capture_plan("snap", "rid"), {"rid": "snap"})
        plan = capture_plan("snap", "rid", 2, 10)
        self.assertEqual(len(plan), 12)
        self.assertEqual(len(set(plan.values())), 12)
        self.assertEqual(plan["rid-ecwarmup-1"], "snap-ecwarmup-1")
        self.assertEqual(plan["rid-measure-10"], "snap-measure-10")
        for warmups, repeat in ((-1, 1), (0, 0), (2, 15)):
            with self.assertRaises(ValueError):
                capture_plan("snap", "rid", warmups, repeat)

    def test_request_plan_only_matches_explicit_ids_once(self):
        scheduler, batch, req, _ = lifetime_fixture()
        scheduler.model_worker = NS(ps=NS(tp_rank=0))
        env = {"SGLANG_KV_EC_DEBUG": "1", "SGLANG_KV_EC_REQUEST_ID": "rid",
               "SGLANG_KV_EC_SNAPSHOT_ID": "snap", "SGLANG_KV_EC_OUTPUT_DIR": "/tmp",
               "SGLANG_KV_EC_CAPTURE_PLAN": json.dumps(capture_plan("snap", "rid", 1, 1))}
        with patch.dict(os.environ, env, clear=True), patch.object(debug_capture, "_ATTEMPTED", set()), \
             patch.object(debug_capture, "capture") as capture:
            req.rid = "unlisted"
            debug_capture.maybe_capture(scheduler, batch, req)
            capture.assert_not_called()
            for rid in ("rid-ecwarmup-1", "rid-measure-1", "rid-measure-1"):
                req.rid = rid
                debug_capture.maybe_capture(scheduler, batch, req)
            self.assertEqual(capture.call_count, 2)
            self.assertEqual([c.args[3].snapshot_id for c in capture.call_args_list],
                             ["snap-ecwarmup-1", "snap-measure-1"])

    def _run_cpu_backed_capture(self, transfer, mode, cuda_events=False, fail_allocation=False,
                                admitted_uuid=None):
        scheduler, batch, req, config = lifetime_fixture()
        module = ModuleType("sglang.srt.mem_cache.memory_pool")
        class MHATokenToKVPool:
            pass
        class ReqToTokenPool:
            pass
        module.MHATokenToKVPool, module.ReqToTokenPool = MHATokenToKVPool, ReqToTokenPool
        pool, mapping = MHATokenToKVPool(), ReqToTokenPool()
        original = torch.arange(400 * 2 * 128, dtype=torch.int32).to(torch.bfloat16).reshape(400, 2, 128)
        k = original.clone().as_subclass(CPUBackedTensor)
        v = (original + 1).as_subclass(CPUBackedTensor)
        before = [t.as_subclass(torch.Tensor).clone() for t in (k, v)]
        pool.k_buffer = [k]
        pool.get_kv_buffer = Mock(return_value=(k, v))
        for key, value in dict(kv_cache_layout="nhd", use_hnd=False, post_capture_active=False,
                               layer_transfer_counter=None, is_quantized_kv_cache=False,
                               dtype=torch.bfloat16, store_dtype=torch.bfloat16,
                               layer_num=36, start_layer=0, page_size=1, size=399).items():
            setattr(pool, key, value)
        mapping.free_slots = []
        mapping.req_generation = torch.ones(8, dtype=torch.int64)
        mapping.req_to_token = torch.arange(1, 385).repeat(8, 1)
        runner = NS(token_to_kv_pool=pool, req_to_token_pool=mapping,
                    model_config=NS(hf_config=NS(model_type="qwen3", num_hidden_layers=36,
                                                 num_key_value_heads=8, head_dim=128)))
        scheduler.model_worker = NS(model_runner=runner, ps=NS(tp_rank=0, tp_size=4,
                                      attn_tp_size=4, attn_cp_size=1, attn_dcp_size=1, gpu_id=0))
        scheduler.req_to_token_pool = mapping
        scheduler.token_to_kv_pool_allocator = NS(get_kvcache=lambda: pool)
        scheduler.server_args.model_path = "/fixture"
        identity = dict(model_identifier="/fixture", model_sha256="fixture", source_commit="fixture", core_diff_sha256="fixture")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config = debug_capture.replace(config, output=Path(directory))
            env = {"SGLANG_KV_EC_PROTECTION_MODE": mode, "SGLANG_KV_EC_TRANSFER": transfer,
                   "SGLANG_KV_EC_PROFILE_DETAIL": "1", "SGLANG_KV_EC_CUDA_EVENTS": str(int(cuda_events)),
                   "SGLANG_KV_EC_PARITY_SOCKET": "/unused"}
            if admitted_uuid is not None:
                env.update(SGLANG_KV_EC_GPU_RESOURCES=json.dumps([
                    dict(gpu_index=str(i + 3), gpu_uuid=admitted_uuid if i == 0 else f'GPU-fixture-{i}')
                    for i in range(4)]), KV_EC_GPU_ADMISSION_MODE='shared')
            stack.enter_context(patch.dict(os.environ, env, clear=True))
            stack.enter_context(patch.dict("sys.modules", {module.__name__: module}))
            stack.enter_context(patch.object(debug_capture, "source_identity", return_value=identity))
            stack.enter_context(patch("threading.Thread.start"))
            client = AsyncParityClient("/unused")
            stack.enter_context(patch.object(debug_capture, "_PARITY_CLIENTS", {"/unused": client}))
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
            stack.enter_context(patch("torch.cuda.get_device_properties", return_value=NS(name="mock", uuid='GPU-fixture-0')))
            sync = stack.enter_context(patch("torch.cuda.synchronize"))
            stream = stack.enter_context(patch("torch.cuda.current_stream"))
            stack.enter_context(patch("torch.cuda.Event", side_effect=lambda **kw: NS(
                record=lambda: None, elapsed_time=lambda end: 0.125)))
            real_empty = torch.empty
            def cpu_empty(*args, **kwargs):
                if fail_allocation:
                    raise RuntimeError("test pinned allocation failure")
                kwargs.pop("pin_memory", None)
                return real_empty(*args, **kwargs)
            stack.enter_context(patch("torch.empty", side_effect=cpu_empty))
            if fail_allocation:
                with self.assertRaises(RuntimeError):
                    debug_capture.capture(scheduler, batch, req, config)
                self.assertTrue(client.queue.empty())
                return
            debug_capture.capture(scheduler, batch, req, config)
            path = Path(directory) / config.snapshot_id
            record = json.loads((path / "rank_0.timing.json").read_text())
            if admitted_uuid is not None:
                self.assertEqual(record['physical_gpu_index'], '3')
                self.assertEqual(record['gpu_uuid'], admitted_uuid)
                self.assertTrue(record['shared_gpu'])
            phase = validate_timing(record)
            tail = json.loads((path / "rank_0.diagnostic.json").read_text())
            self.assertGreaterEqual(tail["diagnostic_write_ms"], 0)
            self.assertIsNone(phase["diagnostic_write_ms"])
            self.assertGreaterEqual(tail["scheduler_through_diagnostics_ms"], phase["scheduler_capture_ms"])
            self.assertEqual(sync.call_count, 2)
            self.assertEqual(stream.return_value.synchronize.call_count, 2)
            meta, payload = client.queue.get_nowait()
            validate_payload(np.frombuffer(payload, dtype=np.uint8).copy(), meta)
            self.assertEqual(phase["d2h_bytes"], len(payload))
            expected_scanned = len(payload) if mode == "FAST_PARITY" else 3 * len(payload)
            self.assertEqual(phase["sha256_scanned_bytes"], expected_scanned)
            self.assertEqual(phase["checksum_location"],
                             "worker_payload_manager_segments" if mode == "FAST_PARITY" else "worker")
            for key in ("layer_getter_validation_ms", "segment_sha256_ms", "aggregate_kv_sha256_ms",
                        "payload_sha256_ms", "final_slot_mapping_check_ms", "client_import_lookup_ms",
                        "client_initialization_ms", "host_buffer_release_ms"):
                self.assertIn(key, phase["wall_regions_ms"])
            self.assertEqual(phase["pinned_allocation_count"], 2 if transfer == "pinned" else 0)
            self.assertEqual(phase["source_recheck_ms"] == 0, mode == "FAST_PARITY")
            for tensor, expected in zip((k, v), before):
                self.assertTrue(torch.equal(tensor.as_subclass(torch.Tensor), expected))
            self.assertFalse(list(path.glob("*.bin")))
            return phase

    def test_full_capture_cpu_backed_both_transfers_and_validation_modes(self):
        for transfer in ("blocking", "pinned"):
            for mode in ("FAST_PARITY", "DEBUG_SAFE"):
                with self.subTest(transfer=transfer, mode=mode):
                    self._run_cpu_backed_capture(transfer, mode)

    def test_actual_capture_binds_runtime_uuid_and_rejects_wrong_device(self):
        self._run_cpu_backed_capture('blocking', 'FAST_PARITY', admitted_uuid='GPU-fixture-0')
        with self.assertRaisesRegex(debug_capture.CaptureRejected, 'gpu_admission_runtime_uuid_mismatch'):
            self._run_cpu_backed_capture('blocking', 'FAST_PARITY', admitted_uuid='GPU-wrong-device')

    def test_pinned_failure_does_not_silently_fallback_or_enqueue(self):
        self._run_cpu_backed_capture("pinned", "FAST_PARITY", fail_allocation=True)

    def test_cuda_event_metrics_are_separate_from_wall_accounting(self):
        phase = self._run_cpu_backed_capture("pinned", "FAST_PARITY", cuda_events=True)
        self.assertEqual(phase["cuda_event_layers"], [{"layer_id": 0,
                          "gather_cuda_event_ms": 0.125, "d2h_cuda_event_ms": 0.125}])
        self.assertNotIn("gather_cuda_event_ms", phase["wall_regions_ms"])

    def test_cpu_profile_exclusive_output_is_readable(self):
        import pstats
        scheduler, batch, req, _ = lifetime_fixture()
        scheduler.model_worker = NS(ps=NS(tp_rank=0))
        with tempfile.TemporaryDirectory() as directory:
            env = {"SGLANG_KV_EC_DEBUG": "1", "SGLANG_KV_EC_REQUEST_ID": req.rid,
                   "SGLANG_KV_EC_SNAPSHOT_ID": "profile", "SGLANG_KV_EC_OUTPUT_DIR": directory,
                   "SGLANG_KV_EC_CPU_PROFILE": "1"}
            with patch.dict(os.environ, env, clear=True), patch.object(debug_capture, "_ATTEMPTED", set()), \
                 patch.object(debug_capture, "capture", return_value={}):
                debug_capture.maybe_capture(scheduler, batch, req)
            path = Path(directory) / "profile" / "rank_0.cpu.prof"
            self.assertTrue(pstats.Stats(str(path)).stats)

    def test_enqueue_copies_attempted_even_when_full_and_reuses_bytes(self):
        with patch("threading.Thread.start"):
            client = AsyncParityClient("/unused")
        first = client.submit({}, b"owned")
        second = client.submit({}, np.arange(12, dtype=np.uint8))
        self.assertEqual(first["copy_bytes"], 0)
        self.assertEqual(first["copy_count"], 0)
        self.assertFalse(second["accepted"])
        self.assertEqual(second["copy_bytes"], 12)
        self.assertEqual(second["copy_count"], 1)
        for result in (first, second):
            self.assertAlmostEqual(result["enqueue_ms"], sum(result[k] for k in
                ("payload_copy_ms", "metadata_deepcopy_ms", "queue_put_ms", "unaccounted_ms")))

    def test_request_runner_separates_warmups_and_validates_all_ranks(self):
        from experiments.kv_ec import profile_requests
        from experiments.kv_ec.test_parity_manager import metadata_and_blocks
        plan = capture_plan("trial", "rid", 1, 2)
        statuses = {}
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fake_request(args):
                seen.append(args.request_id)
                response = {"output_ids": list(range(16)), "client_metrics_ms": {
                    "ttft_ms": 10.0, "tpot_ms": 2.0, "request_total_ms": 40.0}}
                args.output.write_text(json.dumps(response))
                if args.request_id not in plan:
                    return
                snap = plan[args.request_id]
                block, meta = metadata_and_blocks(2, tuple(range(36)), 256)
                expected = block[0].nbytes
                status = dict(meta[0], state="COMMITTED", snapshot_id=snap,
                              request_id=hashlib.sha256(args.request_id.encode()).hexdigest(),
                              model_identifier=args.model, parity_bytes=expected, steady_parity_bytes=expected,
                              rank_hashes={"0": {}, "1": {}}, xor_ms=1.0, parity_commit_ms=3.0,
                              peak_process_rss_bytes=1000000,
                              oracle_checks=[{"lost_rank": i, "byte_exact": True} for i in range(2)])
                statuses[snap] = status
                out = root / snap
                out.mkdir()
                for rank in range(2):
                    phase = {"timing_schema": "kv-ec-exclusive-wall-v2", "wall_regions_ms": {"work": 1.0},
                             "accounted_ms": 1.0, "unaccounted_ms": .1, "scheduler_capture_ms": 1.1,
                             "protection_mode": "FAST_PARITY", "transfer_mode": "pinned", "d2h_bytes": expected,
                             "source_recheck_ms": 0.0, "kv_gather_ms": 0.1, "d2h_ms": 0.1,
                             "sha256_ms": 0.1, "ipc_enqueue_ms": 0.1}
                    profile_requests.exclusive_json(out / f"rank_{rank}.timing.json", {
                        "snapshot_id": snap, "tp_rank": rank, "parity_protection": "ENQUEUED", "performance_ms": phase})
                    profile_requests.exclusive_json(out / f"rank_{rank}.diagnostic.json", {})
            argv = ["profile_requests", "--url", "http://unused", "--output", directory,
                    "--snapshot-id", "trial", "--request-id", "rid", "--tp", "2", "--socket", "/unused",
                    "--ec-warmups", "1", "--ec-repeat", "2", "--model-warmups", "2"]
            with patch.object(sys, "argv", argv), patch.object(profile_requests, "request", side_effect=fake_request), \
                 patch.object(profile_requests, "wait_commit", side_effect=lambda sock, snap, timeout: statuses[snap]):
                profile_requests.main()
            summary = json.loads((root / "trial-profile-summary.json").read_text())
            self.assertEqual(len(seen), 5)
            self.assertEqual([r["kind"] for r in summary["records"]], ["ec_warmup", "measurement", "measurement"])
            self.assertEqual(summary["metrics"]["ttft_ms"]["n"], 2)
            self.assertEqual(set(summary["rank_metrics_ms"]), {"0", "1"})
            self.assertEqual(summary["protection_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
