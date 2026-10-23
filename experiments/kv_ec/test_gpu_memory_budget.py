import json
from pathlib import Path
import tempfile
import unittest

from experiments.kv_ec.gpu_memory_budget import (
    GIB, check_shared_gpu_budget, estimate_required_bytes, parse_gpu_rows,
)


class GPUMemoryBudgetTests(unittest.TestCase):
    def model(self, root):
        root = Path(root)
        (root / "config.json").write_text(json.dumps(dict(
            num_hidden_layers=36, num_key_value_heads=8,
            head_dim=128, hidden_size=4096, num_attention_heads=32)))
        (root / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": 16 * GIB}}))
        return root

    def test_parse_per_gpu_memory_rows(self):
        rows = parse_gpu_rows(
            "0, GPU-a, 8192, 4096\n1, GPU-b, 8192, 1024\n")
        self.assertEqual(rows["0"]["free_bytes"], 4096 * 1024 * 1024)
        self.assertEqual(rows["1"]["uuid"], "GPU-b")

    def test_each_gpu_is_checked_individually_not_by_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(tmp)
            rows = {
                str(i): dict(index=str(i), uuid=f"GPU-{i}",
                             total_bytes=24 * GIB, free_bytes=(16 if i < 3 else 1) * GIB)
                for i in range(4)}
            with self.assertRaisesRegex(ValueError, "GPU index=3.*free_memory_bytes=.*required_estimated_memory_bytes"):
                check_shared_gpu_budget(
                    ["0", "1", "2", "3"], str(model), 4, 4096, .25,
                    memory_rows=rows, tasks_by_uuid={})

    def test_budget_reports_other_tasks_without_managing_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(tmp)
            rows = {str(i): dict(index=str(i), uuid=f"GPU-{i}",
                                 total_bytes=80 * GIB, free_bytes=32 * GIB)
                    for i in range(4)}
            tasks = {"GPU-1": [dict(pid="1234", name="other-user-job")]}
            accepted = check_shared_gpu_budget(
                ["0", "1", "2", "3"], str(model), 4, 1024, .15,
                memory_rows=rows, tasks_by_uuid=tasks)
            self.assertEqual(accepted[1]["other_tasks"], tasks["GPU-1"])
            self.assertTrue(all(item["required_estimated_memory_bytes"] <= 32 * GIB
                                for item in accepted))

    def test_estimate_contains_runtime_and_ec_reserves(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(tmp)
            required, details = estimate_required_bytes(str(model), 4, 4096, 80 * GIB, .1)
            self.assertGreater(required, details["model_weights_per_rank_bytes"])
            self.assertGreater(details["cuda_context_reserve_bytes"], 0)
            self.assertGreater(details["compute_temporary_reserve_bytes"], 0)
            self.assertGreater(details["ec_snapshot_restore_reserve_bytes"], 0)
            self.assertGreaterEqual(required,
                                    details["mem_fraction_static_reserve_bytes"]
                                    + details["cuda_context_reserve_bytes"]
                                    + details["compute_temporary_reserve_bytes"])

    def test_static_fraction_too_low_for_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(tmp)
            rows = {str(i): dict(index=str(i), uuid=f"GPU-{i}",
                                 total_bytes=24 * GIB, free_bytes=20 * GIB)
                    for i in range(4)}
            with self.assertRaisesRegex(ValueError, "mem_fraction_static_below_weights"):
                check_shared_gpu_budget(
                    ["0", "1", "2", "3"], str(model), 4, 4096, .1,
                    memory_rows=rows, tasks_by_uuid={})


if __name__ == "__main__":
    unittest.main()
