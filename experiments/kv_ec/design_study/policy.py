"""Exact-context empirical route selection, separate from runtime safety policy.

Never fits/extrapolates a latency model or authorizes unsafe replay. All costs
must measure the same KV-ready boundary, not mix codec time with fault-to-SSE.
"""

import math
import statistics

from ..node_block_export import canonical, sha


def calibrate(records, context):
    """At least five successful independent trials per route, no hidden failures."""
    if (not isinstance(context, dict) or not all(context.get(k) for k in
            ("model_sha256", "trace_sha256", "tp", "input_tokens", "output_tokens", "batch",
             "fault_kind", "transport", "hardware_id", "measurement_profile"))):
        raise ValueError("complete exact calibration context required")
    chosen = [r for r in records if r.get("context") == context]
    estimates = {}
    for route in ("ec", "prefill"):
        rows = [r for r in chosen if r.get("route") == route]
        if (len(rows) < 5 or len({r.get("trial_id") for r in rows}) != len(rows)
                or any(not r.get("trial_id") or r.get("success") is not True
                       or r.get("boundary") != "fault_to_kv_ready"
                       or not r.get("artifact_sha256") or type(r.get("elapsed_ms")) not in (int, float)
                       or not math.isfinite(r["elapsed_ms"]) or r["elapsed_ms"] <= 0 for r in rows)):
            raise ValueError("five valid distinct trials per route; failures may not be censored")
        estimates[route] = dict(median_ms=statistics.median(r["elapsed_ms"] for r in rows),
                                min_ms=min(r["elapsed_ms"] for r in rows),
                                max_ms=max(r["elapsed_ms"] for r in rows), count=len(rows))
    return dict(schema="ec-empirical-calibration-v1", context=context, estimates=estimates,
                evidence_sha256=sha(canonical(chosen)), model="none; exact-context measured medians")


def choose(calibration, context, *, checkpoint_committed, binding_verified, replay_safe):
    if any(type(v) is not bool for v in (checkpoint_committed, binding_verified, replay_safe)):
        raise ValueError("explicit runtime safety votes required")
    if calibration.get("context") != context:
        raise ValueError("no matching measured calibration; no extrapolation")
    safe = []
    if checkpoint_committed and binding_verified:
        safe.append("ec")
    if replay_safe:
        safe.append("prefill")
    if not safe:
        return "fail_closed"
    return min(safe, key=lambda route: calibration["estimates"][route]["median_ms"])


def summarize(events):
    """Failed recoveries remain in denominator; MTTR is explicitly conditional."""
    entered = [e for e in events if e.get("recovery_entered") is True]
    successful = [e for e in entered if e.get("recovered") is True]
    return dict(requests=len(events), recovery_events=len(entered),
                recovery_success_rate=len(successful)/len(entered) if entered else None,
                request_completion_rate=sum(e.get("completed") is True for e in events)/len(events) if events else None,
                timeout_rate=sum(e.get("timeout") is True for e in events)/len(events) if events else None,
                successful_recovery_mttr_ms=statistics.mean(e["recovery_ms"] for e in successful) if successful else None,
                unconditional_mttr_ms=None,
                eitr=None, eitr_reason="requires independently measured useful inference and total wall time")
