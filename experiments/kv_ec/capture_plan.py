"""Bounded, explicit request-to-snapshot plan for protection warmup experiments."""

import re


def capture_plan(snapshot_id, request_id, warmups=0, repeat=1):
    if type(warmups) is not int or type(repeat) is not int or warmups < 0 or repeat < 1 or warmups + repeat > 16:
        raise ValueError("require 0<=warmups, 1<=repeat and warmups+repeat<=16")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("explicit request identity required")
    if warmups == 0 and repeat == 1:
        plan = {request_id: snapshot_id}
    else:
        plan = {f"{request_id}-ecwarmup-{i}": f"{snapshot_id}-ecwarmup-{i}" for i in range(1, warmups + 1)}
        plan.update({f"{request_id}-measure-{i}": f"{snapshot_id}-measure-{i}" for i in range(1, repeat + 1)})
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", name) for name in plan.values()):
        raise ValueError("invalid or too long snapshot identity")
    return plan
