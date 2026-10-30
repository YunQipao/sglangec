"""CPU-only online launcher tests; never start CUDA or a real server."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.kv_ec import launch_online
from experiments.kv_ec.gpu_memory_budget import GIB, MIB


class OnlineGPUAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name)
        (self.model / "config.json").write_text(json.dumps(dict(
            num_hidden_layers=36, num_key_value_heads=8, head_dim=128,
            hidden_size=4096, num_attention_heads=32)))
        (self.model / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": 16 * GIB}}))
        # Deliberately unequal free memory; the selected order is not numeric.
        self.rows = {
            str(i): dict(index=str(i), uuid=f"GPU-physical-{i}",
                         total_bytes=80 * GIB, free_bytes=(40 + i * 4) * GIB)
            for i in range(4)
        }
        self.tasks = {"GPU-physical-3": [dict(pid="987", name="other-user-job")]}

    def invoke(self, *, tp=4, baseline=False, shared=False, env=None,
               inventory=None, fraction="0.35"):
        selected = "3,1" if tp == 2 else "3,1,0,2"
        argv = [
            "launch_online", "--model", str(self.model), "--gpus", selected,
            "--tp", str(tp), "--socket", "/tmp/test-online-manager.sock",
            "--run-id", "test-online-run", "--mem-fraction-static", fraction,
        ]
        if baseline:
            argv.append("--baseline")
        if shared:
            argv.append("--allow-shared-gpu-debug")
        self.output = io.StringIO()
        self.errors = io.StringIO()
        self.launches = []

        def execute(executable, command, child_env):
            self.launches.append((executable, command, dict(child_env)))

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.dict(os.environ, env or {}, clear=True))
            stack.enter_context(contextlib.redirect_stdout(self.output))
            stack.enter_context(contextlib.redirect_stderr(self.errors))
            if inventory is None:
                stack.enter_context(patch.object(launch_online, "read_gpu_memory",
                                                return_value=self.rows))
                stack.enter_context(patch.object(launch_online, "read_gpu_tasks",
                                                return_value=self.tasks))
            else:
                stack.enter_context(patch(
                    "experiments.kv_ec.gpu_memory_budget.subprocess.check_output",
                    side_effect=inventory * 2 if isinstance(inventory, list) else inventory))
            self.port_check = stack.enter_context(patch.object(launch_online, "check_port"))
            self.manager_socket = stack.enter_context(patch.object(launch_online.socket, "socket"))
            stack.enter_context(patch.object(launch_online, "send_message"))
            stack.enter_context(patch.object(launch_online, "recv_message", return_value=(
                {"state": "READY", "run_id": "test-online-run"}, b"")))
            stack.enter_context(patch.object(launch_online, "source_identity", return_value={}))
            stack.enter_context(patch.object(launch_online.os, "execve", side_effect=execute))
            launch_online.main()

    def test_default_rejects_busy_gpu_for_both_tp_and_modes(self):
        for tp in (2, 4):
            for baseline in (False, True):
                with self.subTest(tp=tp, baseline=baseline):
                    with self.assertRaises(SystemExit) as exc:
                        self.invoke(tp=tp, baseline=baseline)
                    self.assertEqual(exc.exception.code, 2)
                    self.assertIn("selected GPU is busy", self.errors.getvalue())
                    self.assertEqual(self.launches, [])
                    self.port_check.assert_not_called()
                    self.manager_socket.assert_not_called()

    def test_shared_allows_busy_gpu_and_reports_every_selected_gpu(self):
        original = copy.deepcopy(self.tasks)
        for tp in (2, 4):
            for baseline in (False, True):
                with self.subTest(tp=tp, baseline=baseline):
                    self.invoke(tp=tp, baseline=baseline, shared=True)
                    self.assertEqual(len(self.launches), 1)
                    command, env = self.launches[0][1:]
                    self.assertEqual(command[command.index("--tp-size") + 1], str(tp))
                    self.assertEqual(command[command.index("--mem-fraction-static") + 1], "0.35")
                    self.assertEqual(env["SGLANG_KV_EC_ONLINE"], "0" if baseline else "1")
                    report = json.loads(self.output.getvalue().splitlines()[-1].split(":", 1)[1])
                    self.assertEqual(len(report), tp)
                    self.assertEqual(report[0]["other_tasks"], original["GPU-physical-3"])
                    for item in report:
                        self.assertEqual(item["total_memory_bytes"],
                                         item["used_memory_bytes"] + item["free_memory_bytes"])
                        self.assertGreater(item["required_estimated_memory_bytes"], 0)
                        self.assertLessEqual(item["required_estimated_memory_bytes"],
                                             item["free_memory_bytes"])
        self.assertEqual(self.tasks, original)

    def test_insufficient_gpu_rejects_even_if_average_memory_is_sufficient(self):
        self.rows["1"]["free_bytes"] = GIB
        for tp in (2, 4):
            for baseline in (False, True):
                with self.subTest(tp=tp, baseline=baseline):
                    with self.assertRaises(SystemExit):
                        self.invoke(tp=tp, baseline=baseline, shared=True)
                    self.assertIn("GPU index=1", self.errors.getvalue())
                    self.assertIn("per_gpu_free_memory_below_estimate", self.errors.getvalue())
                    report = json.loads(self.output.getvalue().splitlines()[-1].split(":", 1)[1])
                    self.assertEqual(len(report), tp)  # Also report cards after the failing one.
                    self.assertEqual(self.launches, [])
                    self.manager_socket.assert_not_called()

    def test_static_fraction_below_model_and_context_rejects(self):
        with self.assertRaises(SystemExit):
            self.invoke(tp=2, shared=True, fraction="0.05")
        self.assertIn("GPU_STATIC_KV_POOL_INSUFFICIENT", self.errors.getvalue())
        self.assertEqual(self.launches, [])

    def test_physical_uuid_selection_overrides_parent_cuda_visible_devices(self):
        tasks = "GPU-physical-3, 987, other-user-job\n"
        memory = "\n".join(
            f"{i}, {row['uuid']}, {row['total_bytes'] // MIB}, {row['free_bytes'] // MIB}"
            for i, row in self.rows.items())
        parent_env = {"CUDA_VISIBLE_DEVICES": "0,2", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
        with self.assertRaises(SystemExit):
            self.invoke(tp=2, env=parent_env, inventory=[tasks, memory])
        self.assertIn("selected GPU is busy", self.errors.getvalue())
        self.invoke(tp=2, shared=True, env=parent_env, inventory=[tasks, memory])
        env = self.launches[0][2]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3,1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        report = json.loads(self.output.getvalue().splitlines()[-1].split(":", 1)[1])
        self.assertEqual([item["gpu_uuid"] for item in report],
                         ["GPU-physical-3", "GPU-physical-1"])

    def test_default_allows_tasks_on_unselected_gpu(self):
        self.tasks = {"GPU-physical-0": [dict(pid="123", name="unrelated-job")]}
        self.invoke(tp=2, baseline=True)
        self.assertEqual(len(self.launches), 1)

    def test_policy_explicitly_overrides_child_imbalance_without_parent_leak(self):
        for tp in (2, 4):
            for baseline in (False, True):
                for shared in (False, True):
                    for value in (None, "0", "1"):
                        with self.subTest(tp=tp, baseline=baseline, shared=shared, value=value):
                            self.tasks = {}
                            env = {} if value is None else {
                                "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": value}
                            self.invoke(tp=tp, baseline=baseline, shared=shared, env=env)
                            child_env = self.launches[0][2]
                            self.assertEqual(child_env['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK'],
                                             '0' if shared else '1')

    def test_missing_gpu_rejects_cleanly(self):
        del self.rows["3"]
        with self.assertRaises(SystemExit):
            self.invoke(shared=True)
        self.assertIn("GPU_NOT_FOUND", self.errors.getvalue())
        self.assertEqual(self.launches, [])

    def test_inventory_failure_rejects_without_starting_server(self):
        with self.assertRaises(SystemExit):
            self.invoke(shared=True, inventory=subprocess.CalledProcessError(9, "nvidia-smi"))
        self.assertIn("nvidia-smi", self.errors.getvalue())
        self.assertEqual(self.launches, [])

    def test_unavailable_model_budget_fails_closed(self):
        (self.model / "model.safetensors.index.json").unlink()
        with self.assertRaises(SystemExit):
            self.invoke(shared=True)
        self.assertIn("shared_gpu_weight_size_unavailable", self.errors.getvalue())
        self.assertEqual(self.launches, [])


if __name__ == "__main__":
    unittest.main()
