"""CPU-only profiling contracts; not GPU acceptance."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import numpy as np

from experiments.kv_ec.m5_capture_profile import ENV, profile_capture, stage
from experiments.kv_ec.m5_analysis import trace_record, union_us


class CaptureProfileTests(unittest.TestCase):
    def test_real_packer_profile_preserves_every_byte_and_metadata(self):
        from experiments.kv_ec.scheduler_shadow import pack_chunk
        raw = torch.arange(257 * 2 * 128 * 2, dtype=torch.int64).to(torch.uint8)
        k = raw.view(torch.bfloat16).reshape(257, 2, 128)
        v = raw.bitwise_xor(71).view(torch.bfloat16).reshape(257, 2, 128)
        pool = NS(k_buffer=[k] * 36, v_buffer=[v] * 36, get_kv_buffer=lambda layer: (k, v))
        mapping = NS(physical_slots=tuple(range(256, 0, -1)), slot_generations=(1,) * 256,
                     req_pool_idx=1, req_generation=1, request_version="v1", tp_rank=2)
        req = NS(rid="synthetic_fixture", origin_input_ids=[1] * 768)
        with patch.dict(os.environ, {ENV: ""}):
            expected, metadata = pack_chunk(pool, mapping, req, {},
                                            snapshot_id="fixture", manager_epoch="epoch")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {ENV: tmp}), \
                patch("torch.cuda.is_available", return_value=False):
            actual, recorded = pack_chunk(pool, mapping, req, {},
                                          snapshot_id="fixture", manager_epoch="epoch")
            row = trace_record(next(Path(tmp).glob("capture-*.trace.json")))
            self.assertEqual(row["host_timing"]["tp_rank"], 2)
            self.assertEqual(row["host_timing"]["snapshot_id"], "fixture")
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(recorded, metadata)

    def test_disabled_has_no_profiler_or_output_and_preserves_exception(self):
        @profile_capture
        def f():
            with stage("cpu_copy"):
                raise ValueError("original_failure")
        with patch.dict(os.environ, {ENV: ""}), patch("torch.profiler.profile") as profiler:
            with self.assertRaisesRegex(ValueError, "original_failure"):
                f()
            profiler.assert_not_called()

    def test_enabled_cpu_trace_contains_no_tensor_values(self):
        @profile_capture
        def f():
            with stage("cpu_copy"):
                return torch.ones(4).sum().item()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {ENV: tmp}), \
                patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(f(), 4.)
            files = list(Path(tmp).glob("capture-*.trace.json"))
            self.assertEqual(len(files), 1)
            result = trace_record(files[0])
            self.assertIn("cpu_copy", result["host_wall_ms"])
            self.assertGreaterEqual(result["host_timing"]["thread_cpu_regions_ms"]["cpu_copy"], 0.)
            self.assertGreaterEqual(result["host_timing"]["unaccounted_ms"], 0.)
            self.assertIsNone(result["d2h_device_activity_ms"])
            self.assertIsNone(result["gather_kernel_activity_ms"])

    def test_relative_directory_rejected_before_function(self):
        called = []
        with patch.dict(os.environ, {ENV: "relative"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                profile_capture(lambda: called.append(True))()
        self.assertEqual(called, [])

    def test_profiled_smoke_cannot_admit_performance_repeats(self):
        from experiments.kv_ec.m4_abcd import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "smoke"
            smoke.mkdir()
            (smoke / "trials.json").write_text(json.dumps({"state": "PASS"}))
            with patch("sys.argv", [
                    "m4_abcd", "--gpus", "0,1,2,3", "--shadow-evidence", "/prior",
                    "--repeat", "3", "--smoke-evidence", str(smoke),
                    "--output", str(root / "study")]), patch(
                    "experiments.kv_ec.m4_abcd.summarize",
                    return_value={"repeat": 1, "capture_profile": True}), patch(
                    "experiments.kv_ec.m4_abcd.run_logged") as runner:
                self.assertEqual(main(), 1)
                runner.assert_not_called()
            manifest = json.loads((root / "study/trials.json").read_text())
            self.assertIn("profiled_smoke_not_performance_admission", manifest["reason"])

    def test_overlapping_device_intervals_not_added(self):
        self.assertEqual(union_us([dict(ts=0, dur=10), dict(ts=2, dur=4),
                                   dict(ts=8, dur=4), dict(ts=20, dur=2)]), 14)

    def test_device_times_are_not_host_blocking_times(self):
        events = [
            dict(ph="X", cat="user_annotation", name="kv_ec.capture.total", ts=0, dur=100, pid=1, tid=1),
            dict(ph="X", cat="user_annotation", name="kv_ec.capture.gpu_gather_host",
                 ts=0, dur=20, pid=1, tid=1),
            dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", ts=5, dur=2,
                 pid=1, tid=1, args={"correlation": 7}),
            dict(ph="X", cat="kernel", name="index_select", ts=30, dur=10, args={"correlation": 7}),
            dict(ph="X", cat="gpu_memcpy", name="Memcpy DtoH", ts=35, dur=20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text(json.dumps(dict(traceEvents=events)))
            row = trace_record(path)
        self.assertEqual(row["gather_kernel_activity_ms"], .01)
        self.assertEqual(row["d2h_device_activity_ms"], .02)
        self.assertEqual(row["device_activity_union_ms"], .025)
