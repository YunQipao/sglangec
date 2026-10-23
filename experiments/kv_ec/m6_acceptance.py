"""Read-only E evidence checks. No GPU access, servers or KV bytes in reports."""

import json
import math
from pathlib import Path

from .async_protection import RANK_BYTES, checkpoint_metadata
from .scheduler_shadow import validate_checkpoint


def check_async_evidence(root, identity, checkpoints, response, *, require_overlap=False):
    from .accept_pause import check_evidence
    root = Path(root)
    check_evidence(root, identity)
    if len(checkpoints) != 1:
        raise ValueError("async_requires_one_checkpoint")
    checkpoint = checkpoint_metadata(checkpoints[0])
    records = []
    submit_times = []
    decode_times = []
    consensus = None
    for rank in range(4):
        record = json.loads((root / f"pause-rank-{rank}.json").read_text())
        proof = record["async_protection_only"]
        background = json.loads((root / f"async-rank-{rank}.json").read_text())
        lifecycle = json.loads((root / f"async-lifecycle-rank-{rank}.json").read_text())
        meta = proof["metadata"]
        validate_checkpoint(checkpoint, meta, require_oracles=False)
        if (background["state"] != "COMMITTED" or background["reason"] is not None
                or background["metadata"] != meta
                or checkpoint_metadata(background["status"]) != checkpoint
                or any(item.get("transaction_id") != identity["transaction_id"]
                       or item.get("tp_rank") != rank for item in (background, lifecycle))
                or lifecycle["cancelled"] is not False
                or lifecycle["foreground_waited_for_commit"] is not False
                or lifecycle["pending_policy"] != "bounded_off_path_drain"
                or lifecycle["state_at_request_end"] not in (
                    "SNAPSHOT_READY", "COMMITTING", "COMMITTED")
                or meta["snapshot_id"] != identity["transaction_id"] + "-chunk0"
                or meta["checkpoint_version"] != meta["snapshot_id"]
                or meta["tp_rank"] != rank or meta["tp_size"] != 4
                or meta["input_ids_sha256"] != response["input_ids_sha256"]
                or meta["request_id"] != record["request_hash"]
                or meta["req_pool_idx"] != record["row"]
                or meta["req_generation"] != record["request_generation_before"]
                or meta["physical_slots"] != record["physical_slots_before"][:256]
                or meta["slot_generation"] != record["slot_generations_before"][:256]
                or meta["token_start"] != 0 or meta["token_count"] != 256
                or meta["packed_byte_count"] != RANK_BYTES
                or meta["layers"] != list(range(36))
                or len(meta["layer_records"]) != 36
                or not all(str(r["device"]).startswith("cuda") for r in meta["layer_records"])
                or meta["snapshot_completion_status"] != "complete"
                or proof["source_copy_event_complete"] is not True
                or proof["source_copy_event_synchronized"] is not True
                or proof["commit_wait_in_decode_path"] is not False
                or background["resident_snapshot_bytes"] != 0
                or background["peak_snapshot_bytes"] != RANK_BYTES
                or record["coordinator_final"]["experiment_mode"] != "async_protection_only"
                or record["coordinator_final"]["shadow_ranks"] != list(range(4))
                or not record["runtime_checks"]
                or any(v["actual"] != v["expected"] or type(v["actual"]) is not type(v["expected"])
                       for v in record["runtime_checks"].values())):
            raise ValueError(f"async_rank{rank}_missing_binding_commit_lifetime_or_GPU")
        current = proof["consensus"]
        if (consensus is not None and current != consensus
                or any(current.get(k) is not True for k in (
                    "snapshot_ready", "copy_complete", "independent_cpu_bytes"))
                or any(current.get(k) is not False for k in (
                    "checkpoint_committed", "used_committed_parity",
                    "online_kv_pool_modified", "fault_injected"))
                or any(current.get(k) != meta[k] for k in (
                    "snapshot_id", "checkpoint_version", "manager_epoch", "request_version",
                    "input_ids_sha256"))
                or current.get("request_hash") != record["request_hash"]):
            raise ValueError("async_snapshot_consensus_mismatch")
        consensus = current
        started = proof["snapshot_started_monotonic"]
        prefill = record["prefill_completed_monotonic"]
        ready = proof["snapshot_ready_monotonic"]
        released = proof["foreground_released_monotonic"]
        committed = background["committed_observed_monotonic"]
        submitted = background["submit_started_monotonic"]
        decoded = record["decode_observed_monotonic"]
        foreground_ready = proof["foreground_ready_monotonic"]
        blocking_started = proof["foreground_blocking_started_monotonic"]
        transitions = background["trace"]
        if ([t["state"] for t in transitions] != ["SNAPSHOT_READY", "COMMITTING", "COMMITTED"]
                or not prefill <= started <= ready <= transitions[0]["monotonic"] <= released
                or not released <= transitions[1]["monotonic"] <= submitted <= committed
                or not released <= foreground_ready <= decoded <= lifecycle["request_end_monotonic"]
                or not prefill <= blocking_started <= started
                or committed != transitions[-1]["monotonic"]
                or background["snapshot_started_monotonic"] != started
                or background["snapshot_ready_monotonic"] != ready
                or background["foreground_released_monotonic"] != released
                or background["prefill_completed_monotonic"] != prefill
                or require_overlap and not decoded < committed):
            raise ValueError("async_timeline_or_decode_overlap_invalid")
        formulas = {
            "snapshot_ms": (ready - started) * 1000,
            "background_committed_ms": (committed - ready) * 1000,
            "unprotected_window_ms": (committed - prefill) * 1000,
            "capture_to_committed_ms": (committed - started) * 1000,
        }
        for key, expected in formulas.items():
            if (not math.isfinite(expected) or expected <= 0
                    or not math.isclose(background[key], expected, abs_tol=1e-6)
                    or key == "snapshot_ms" and proof[key] != background[key]):
                raise ValueError("async_timing_not_bound_to_original_clock")
        if proof["foreground_protection_blocking_ms"] < proof["snapshot_ms"]:
            raise ValueError("async_foreground_missing_snapshot_cost")
        if not math.isclose(proof["foreground_protection_blocking_ms"],
                            (foreground_ready - blocking_started) * 1000, abs_tol=1e-6):
            raise ValueError("async_foreground_not_bound_to_scheduler_ready")
        submit_times.append(submitted)
        decode_times.append(decoded)
        record["async_metrics"] = dict(
            **formulas, foreground_protection_blocking_ms=proof["foreground_protection_blocking_ms"],
            decode_before_commit=decoded < committed,
            peak_snapshot_bytes=background["peak_snapshot_bytes"],
            peak_cpu_rss_bytes=max(proof["snapshot_cpu_peak_rss_bytes"],
                                   background["background_cpu_peak_rss_bytes"]),
            peak_gpu_allocated_bytes=proof["peak_gpu_allocated_bytes"],
            peak_gpu_reserved_bytes=proof["peak_gpu_reserved_bytes"])
        records.append(record)
    if require_overlap and not max(decode_times) < min(submit_times):
        raise ValueError("async_smoke_requires_all_ranks_decode_before_any_submission")
    return records


def async_performance(response, records, *, config):
    keys = ("snapshot_ms", "background_committed_ms", "unprotected_window_ms",
            "foreground_protection_blocking_ms", "peak_snapshot_bytes",
            "peak_cpu_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
    return dict(
        mode="async_protection_only", config=config, **response["client_metrics_ms"],
        **{key: max(r["async_metrics"][key] for r in records) for key in keys},
        all_ranks_decode_before_commit=all(r["async_metrics"]["decode_before_commit"] for r in records),
        rank_metrics=[r["async_metrics"] for r in records],
        timing_note="background commit observed (conservative upper bound), not foreground wait; "
                    "unprotected window starts at completed Prefill, not request arrival",
        memory_note="CPU lifetime ru_maxrss and CUDA allocator high-water at snapshot, not full-request "
                    "GPU peak or NVML usage; snapshot budget excludes bounded IPC/manager copies")


def check_m6b2_evidence(root, identity, response):
    """Require transaction-derived per-rank evidence for one real B2 restore."""
    root = Path(root)
    records = [json.loads((root / f"m6b2-rank-{rank}.json").read_text())
               for rank in range(4)]
    targets = []
    for rank, record in enumerate(records):
        recovery = record.get("recovery", {})
        if (record.get("state") != "RUNNING"
                or record.get("checkpoint_id") != identity["transaction_id"] + "-chunk0"
                or record.get("new_lease_created") is not True
                or record.get("new_lease_released") is not True
                or record.get("all_rank_validate_vote") is not True
                or record.get("all_rank_resume_vote") is not True
                or record.get("four_rank_vote") is not True
                or record.get("byte_exact") is not True
                or recovery.get("oracle_sha256") is None):
            raise ValueError(f"m6b2_rank{rank}_transaction_proof_incomplete")
        if record.get("fault_injected") or record.get("parity_used") or record.get("kv_written"):
            targets.append(rank)
    if targets != [0]:
        raise ValueError("m6b2_rank0_must_be_the_only_fault_and_write_target")
    recovery = records[0]["recovery"]
    if (recovery.get("fault_injected") is not True
            or recovery.get("parity_used") is not True
            or recovery.get("kv_written") is not True
            or recovery.get("byte_exact") is not True
            or response.get("output_ids") is None):
        raise ValueError("m6b2_real_restore_or_decode_proof_missing")
    return records
