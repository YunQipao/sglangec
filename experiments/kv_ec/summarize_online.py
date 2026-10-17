"""Summarize one mode/range's independent online trials; never runs inference."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths, field):
    values = {}
    for path in paths:
        record = json.loads(Path(path).read_text())
        if field == "commit" and record.get("state") != "COMMITTED":
            raise ValueError(f"not protected: {path}")
        metrics = record.get(field, {}) if field != "commit" else record
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, []).append(value)
    return {key: {"n": len(data), "median": float(np.median(data)),
                  "p95": float(np.percentile(data, 95))} for key, data in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", nargs="*", type=Path, default=[])
    parser.add_argument("--timings", nargs="*", type=Path, default=[])
    parser.add_argument("--commits", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    timings = {}
    for path in args.timings:
        rank = json.loads(path.read_text())["tp_rank"]
        timings.setdefault(rank, []).append(path)
    print(json.dumps({"client_ms": summarize(args.responses, "client_metrics_ms"),
                      "rank_ms": {rank: summarize(paths, "performance_ms") for rank, paths in timings.items()},
                      "manager": summarize(args.commits, "commit")}, indent=2))


if __name__ == "__main__":
    main()
