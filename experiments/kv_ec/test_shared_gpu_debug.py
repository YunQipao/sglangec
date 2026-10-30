"""CPU-only tests for shared-GPU preflight and launcher configuration."""

import sys
import os
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec.accept_pause import check_gpu_preflight, gpu_inventory, server_launch_args


class SharedGPUPreflightTests(unittest.TestCase):
    def setUp(self):
        self.devices = {
            "0": {"physical_index": 0, "uuid": "GPU-physical-zero", "name": "A",
                  "total_mib": 81920, "used_mib": 40960, "free_mib": 40960,
                  "processes": [{"pid": 123, "process_name": "other", "used_memory": "40960 MiB"}]},
            "3": {"physical_index": 3, "uuid": "GPU-physical-three", "name": "B",
                  "total_mib": 81920, "used_mib": 1024, "free_mib": 80896, "processes": []},
        }

    def test_existing_compute_process_rejected_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "active compute tasks"):
            check_gpu_preflight(self.devices, ["0"], allow_shared=False, min_free_gpu_gb=0)

    def test_shared_debug_allows_process_without_interference(self):
        selected = check_gpu_preflight(self.devices, ["0"], allow_shared=True, min_free_gpu_gb=0)
        self.assertEqual(selected[0]["processes"][0]["pid"], 123)

    def test_free_memory_floor_rejects(self):
        with self.assertRaisesRegex(RuntimeError, "minimum is 50.00 GiB"):
            check_gpu_preflight(self.devices, ["0"], allow_shared=True, min_free_gpu_gb=50)

    def test_uuid_maps_to_physical_indices_not_cuda_logical_order(self):
        with patch("experiments.kv_ec.accept_pause.subprocess.check_output", side_effect=[
            "0, GPU-physical-zero, A, 81920, 40960, 40960\n"
            "3, GPU-physical-three, B, 81920, 1024, 80896\n",
            "GPU-physical-three, 987, other, 512 MiB\n",
        ]):
            inventory = gpu_inventory()
        self.assertEqual(list(inventory), ["0", "3"])
        self.assertEqual(inventory["3"]["uuid"], "GPU-physical-three")
        self.assertEqual(inventory["3"]["processes"][0]["pid"], 987)
        self.assertEqual(inventory["0"]["processes"], [])


class LauncherOptionsTests(unittest.TestCase):
    def test_acceptance_passes_configuration_to_both_server_modes(self):
        from types import SimpleNamespace
        from pathlib import Path

        args = SimpleNamespace(gpus="0,1,2,3", model="Qwen3-8B",
                               mem_fraction_static=0.35, context_length=1024)
        for label in ("baseline", "pause"):
            command = server_launch_args(args, port=30359, root=Path("/tmp/run"),
                                         transaction_id="tx", label=label)
            self.assertEqual(command[command.index("--mem-fraction-static") + 1], "0.35")
            self.assertEqual(command[command.index("--context-length") + 1], "1024")

    def test_configurable_server_options_reach_launch_server_and_defaults_remain(self):
        from experiments.kv_ec import debug_capture, launch_debug

        for extra, expected in (([], ("0.6", "4096")),
                                (["--mem-fraction-static", "0.35", "--context-length", "1024"],
                                 ("0.35", "1024"))):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                argv = ["launch_debug", "--mode", "baseline", "--tp", "1", "--gpus", "2",
                        "--output", directory, "--snapshot-id", "config-test", *extra]
                with patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True), \
                     patch("experiments.kv_ec.debug_resources.admit", return_value=[
                         {'gpu_index': '2', 'gpu_uuid': 'GPU-test-2'}]), \
                     patch("experiments.kv_ec.launch_debug.socket.socket"), \
                     patch.object(launch_debug.os, "execv") as execute:
                    launch_debug.main()
                server_argv = execute.call_args.args[1]
                self.assertEqual(server_argv[server_argv.index("--mem-fraction-static") + 1], expected[0])
                self.assertEqual(server_argv[server_argv.index("--context-length") + 1], expected[1])

    def test_context_shorter_than_existing_request_is_rejected(self):
        from experiments.kv_ec import launch_debug

        with patch.object(sys, "argv", ["launch_debug", "--mode", "baseline", "--tp", "1",
                                        "--gpus", "0", "--output", "/tmp",
                                        "--snapshot-id", "short-context", "--context-length", "783"]), \
             self.assertRaises(SystemExit):
            launch_debug.main()


if __name__ == "__main__":
    unittest.main()
