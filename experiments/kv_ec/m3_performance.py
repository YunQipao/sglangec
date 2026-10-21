"""Aggregate comparable M3 A/B/C/D performance records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ("baseline", "protected", "online_recovery", "prefill_retry")
METRICS = (
    "ttft_ms", "tpot_ms", "request_total_ms", "protection_overhead_ms",
    "parity_commit_ms", "recovery_ms", "cpu_xor_ms", "gather_ms",
    "h2d_ms", "kv_writeback_ms", "validation_ms",
)


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summarize_records(paths, *, expected_config=None):
    records = [json.loads(Path(path).read_text()) for path in paths]
    if not records:
        raise ValueError("no_performance_records")
    configs = {json.dumps(record.get("config"), sort_keys=True) for record in records}
    if len(configs) != 1:
        raise ValueError("performance_config_mismatch")
    if expected_config is not None and records[0].get("config") != expected_config:
        raise ValueError("performance_config_does_not_match_expected")
    modes = {record.get("mode") for record in records}
    if not modes <= set(MODES):
        raise ValueError("unknown_performance_mode")
    output = {"config": records[0]["config"], "n": len(records), "metrics": {}}
    for metric in METRICS:
        values = [record[metric] for record in records if _numeric(record.get(metric))]
        if values:
            output["metrics"][metric] = {
                "n": len(values),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
    return output


def summarize_by_mode(paths, *, expected_config=None):
    grouped = {}
    for path in paths:
        record = json.loads(Path(path).read_text())
        grouped.setdefault(record.get("mode"), []).append(path)
    return {
        mode: summarize_records(items, expected_config=expected_config)
        for mode, items in sorted(grouped.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.config.read_text()) if args.config else None
    print(json.dumps(summarize_by_mode(args.records, expected_config=expected), indent=2))


if __name__ == "__main__":
    main()
