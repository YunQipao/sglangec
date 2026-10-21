"""CPU tests of reporting/gates, never GPU acceptance evidence."""

from copy import deepcopy
import unittest

from experiments.kv_ec.m4_abcd import COMMON_CONFIG
from experiments.kv_ec.m5_compare import METRICS, compare_summaries


def summary():
    row = dict(state="PASS", repeat=1, capture_profile=False,
               config={k: k for k in COMMON_CONFIG}, modes={})
    for mode, key in METRICS.values():
        row["modes"].setdefault(mode, {})[key] = dict(samples_ms=[1000.])
    row["modes"]["protection_only"]["request_total_ms"]["samples_ms"] = [2000.]
    return row


class ComparisonTests(unittest.TestCase):
    def test_positive_is_only_preliminary_and_sources_can_change(self):
        before = summary()
        after = deepcopy(before)
        after["config"]["source_commit"] = "new_commit"
        after["config"]["experiment_source_sha256"] = "new_experiment"
        after["modes"]["protection_only"]["request_total_ms"]["samples_ms"] = [1800.]
        after["modes"]["protection_only"]["protection_blocking_ms"]["samples_ms"] = [800.]
        result = compare_summaries(before, after)
        self.assertTrue(result["preliminary_positive"])
        self.assertEqual(result["metrics"]["B_minus_A_request_ms"]["reduction_pct"], 20.)
        self.assertIsNone(result["gpu_peak_memory_comparison"])
        self.assertIn("not proof", result["note"])

    def test_slowdown_and_C_regression_block_repeats(self):
        before, after = summary(), summary()
        after["modes"]["online_recovery"]["fault_to_first_decode_ms"]["samples_ms"] = [1100.]
        result = compare_summaries(before, after)
        self.assertFalse(result["preliminary_positive"])
        self.assertFalse(result["gates"]["C_fault_to_decode_within_limit"])
        self.assertFalse(result["gates"]["B_blocking_reduced"])

    def test_configuration_and_profile_mismatch_rejected(self):
        for key in ("gpu_uuids", "sampling", "core_diff_sha256", "model_sha256"):
            after = summary()
            after["config"][key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "incomparable_config"):
                compare_summaries(summary(), after)
        after["capture_profile"] = True
        with self.assertRaisesRegex(ValueError, "unprofiled"):
            compare_summaries(summary(), after)

    def test_nonfinite_timing_and_thresholds_rejected(self):
        after = summary()
        after["modes"]["baseline"]["request_total_ms"]["samples_ms"] = [float("nan")]
        with self.assertRaisesRegex(ValueError, "invalid_metric"):
            compare_summaries(summary(), after)
        with self.assertRaisesRegex(ValueError, "threshold"):
            compare_summaries(summary(), summary(), max_c_regression_pct=float("nan"))
