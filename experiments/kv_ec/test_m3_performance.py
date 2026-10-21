import json
import tempfile
import unittest
from pathlib import Path

from experiments.kv_ec.m3_performance import summarize_by_mode


class M3PerformanceTests(unittest.TestCase):
    def test_same_config_reports_median_and_p95(self):
        config = {"model": "fixture", "input_tokens": 768, "output_tokens": 16,
                  "gpu": "0,1,2,3", "dtype": "bf16"}
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, total in enumerate((10, 20, 30)):
                path = Path(tmp) / f"{index}.json"
                path.write_text(json.dumps({
                    "mode": "online_recovery", "config": config,
                    "ttft_ms": total, "request_total_ms": total,
                    "cpu_xor_ms": 2, "gather_ms": 3, "h2d_ms": 4,
                    "kv_writeback_ms": 5, "validation_ms": 6,
                }))
                paths.append(path)
            report = summarize_by_mode(paths)
        self.assertEqual(report["online_recovery"]["metrics"]["ttft_ms"]["median_ms"], 20)
        self.assertEqual(report["online_recovery"]["metrics"]["ttft_ms"]["p95_ms"], 29)

    def test_mixed_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, model in enumerate(("a", "b")):
                path = Path(tmp) / f"{index}.json"
                path.write_text(json.dumps({"mode": "baseline",
                                            "config": {"model": model},
                                            "request_total_ms": 1}))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "config_mismatch"):
                summarize_by_mode(paths)


if __name__ == "__main__":
    unittest.main()
