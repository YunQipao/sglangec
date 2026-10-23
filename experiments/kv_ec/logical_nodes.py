"""Pure helpers for the one-host/two-network-namespace TP4 experiment."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NodeRanks:
    node_rank: int
    global_tp_ranks: tuple[int, ...]
    visible_devices: tuple[int, ...]


def rank_layout(*, nnodes=2, tp_size=4, devices=(0, 1, 2, 3)):
    if nnodes != 2 or tp_size != 4 or len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("prototype_requires_two_nodes_global_tp4_and_four_distinct_gpus")
    if tp_size % nnodes:
        raise ValueError("tp_size_must_divide_evenly_across_nodes")
    per_node = tp_size // nnodes
    return tuple(NodeRanks(node, tuple(range(node * per_node, (node + 1) * per_node)),
                           tuple(devices[node * per_node:(node + 1) * per_node]))
                 for node in range(nnodes))


def rank_manifest(*, run_id, model_path, devices=(0, 1, 2, 3), diagnostic=False):
    nodes = rank_layout(devices=devices)
    return {
        "schema": "sglang-kv-ec-logical-node-layout-v1",
        "run_id": run_id,
        "model_path": model_path,
        "global_tp_size": 4,
        "nnodes": 2,
        "processes_per_node": 2,
        "diagnostic_transport": bool(diagnostic),
        "nodes": [asdict(node) for node in nodes],
        "layout_note": "CUDA_VISIBLE_DEVICES is local to each net namespace process; "
                       "global rank equals node_rank*2 + local_rank for pp_size=1",
    }


def verify_rank_observations(observations):
    """Check runtime evidence, not process-count claims."""
    if len(observations) != 4:
        raise ValueError("exactly_four_rank_observations_required")
    by_rank = {item.get("tp_rank"): item for item in observations}
    if set(by_rank) != {0, 1, 2, 3}:
        raise ValueError("global_tp_rank_set_mismatch")
    for rank, item in by_rank.items():
        expected_node, expected_local = divmod(rank, 2)
        if item.get("node_rank") != expected_node or item.get("local_rank") != expected_local:
            raise ValueError("rank_to_node_or_local_gpu_mismatch")
        if item.get("tp_size") != 4:
            raise ValueError("runtime_tp_size_mismatch")
        if "world_size" in item and item["world_size"] != 4:
            raise ValueError("runtime_world_size_mismatch")
    return {"status": "rank_map_consistent", "global_ranks": [0, 1, 2, 3],
            "node0": [0, 1], "node1": [2, 3]}
