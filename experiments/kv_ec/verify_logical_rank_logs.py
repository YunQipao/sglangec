"""Extract SGLang's emitted device/TP rank tuples from two node logs."""

import argparse
import json
from pathlib import Path
import re

try:
    from .logical_nodes import verify_rank_observations
except ImportError:
    from logical_nodes import verify_rank_observations

CONTEXT = re.compile(r"Context: .*?ps\.gpu_id=(\d+).*?ps\.tp_rank=(\d+).*?ps\.tp_size=(\d+)")


def collect(logs):
    found = {}
    for node_rank, path in enumerate(map(Path, logs)):
        devices = (0, 1) if node_rank == 0 else (2, 3)
        local_map = {physical: local for local, physical in enumerate(devices)}
        for line in path.read_text(errors="replace").splitlines():
            match = CONTEXT.search(line)
            if not match:
                continue
            gpu_id, tp_rank, tp_size = map(int, match.groups())
            item = dict(node_rank=node_rank, local_rank=gpu_id, physical_gpu=devices[gpu_id]
                        if gpu_id in (0, 1) else None, tp_rank=tp_rank,
                        tp_size=tp_size)
            if tp_rank in found and found[tp_rank] != item:
                raise ValueError("conflicting_runtime_rank_observations")
            found[tp_rank] = item
        if not any(item["node_rank"] == node_rank for item in found.values()):
            raise ValueError(f"no_model_runner_rank_context_in_node{node_rank}_log")
    result = verify_rank_observations(list(found.values()))
    return {**result, "rank_observations": [found[k] for k in sorted(found)],
            "source_logs": [str(Path(p).resolve()) for p in logs],
            "physical_gpu_mapping": {str(k): v for k, v in
                {0: 0, 1: 1, 2: 2, 3: 3}.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node0_log", type=Path)
    parser.add_argument("node1_log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect((args.node0_log, args.node1_log))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
