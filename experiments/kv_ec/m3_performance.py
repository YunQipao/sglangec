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
    "fault_to_decode_ms",
)


def online_record(target, response, *, config):
    """Keep raw per-chunk costs separate from client/authorization latency."""
    chunks = target["online"]["chunks"]
    protection = target["protection"]["chunks"]
    def total(key):
        return sum(chunk["timing_ms"][key] for chunk in chunks)
    return dict(
        mode="online_recovery", config=config, **response["client_metrics_ms"],
        protection_overhead_ms=sum(c["timing_ms"]["protection_total_ms"] for c in protection),
        parity_commit_ms=sum(c["timing_ms"]["parity_commit_ms"] for c in protection),
        recovery_ms=total("chunk_recovery_ms"), cpu_xor_ms=total("xor_and_cpu_verify_ms"),
        gather_ms=total("survivor_gather_ms"), h2d_ms=total("h2d_ms"),
        kv_writeback_ms=total("kv_writeback_ms"),
        validation_ms=total("recovered_gpu_verify_ms") + total("writeback_gpu_and_hash_verify_ms"),
        fault_to_decode_ms=target["online"]["timing_ms"]["fault_to_decode_ms"],
        chunks=[dict(token_start=c["token_start"], snapshot_id=c["checkpoint"]["snapshot_id"],
                     recovery_timing_ms=c["timing_ms"], protection_timing_ms=p["timing_ms"])
                for c, p in zip(chunks, protection)],
        timing_note="recovery_ms excludes protection, external authorization wait and decode; "
                    "fault_to_decode_ms includes automatic coordinator/resume/decode latency")


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
