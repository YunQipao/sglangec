"""Three-domain config/preflight gate. No SSH, service start or network changes.

Reuses design_study admission and its physical 232/233/243 allowlist. An
operator-supplied probe must obtain fresh machine IDs over authenticated control
channels, bind the challenge, and perform that node's local_identity check.
Caller dictionaries alone are NOT proof of independent hardware or fault RTO.
"""

import secrets

from .design_study.admission import placement
from .hot_tp2 import Rejected


def preflight(config, *, enabled=False, probe=None):
    if enabled is not True:
        raise Rejected("three_domain_gate_disabled")
    config_sha = placement(config)  # rejects .245 RDMA alias as physical P
    result = dict(status="BLOCKED", config_sha256=config_sha,
                  physical_node_fault_tolerance_verified=False, nodes=[])
    if probe is None:
        return result | dict(reason="authenticated_reachable_identity_probe_required")
    challenge = secrets.token_hex(32)
    machine_ids = set()
    for node in config["nodes"]:
        try:
            evidence = probe(dict(node), challenge, config["timeout_s"])
        except (OSError, TimeoutError) as exc:
            return result | dict(reason="node_unreachable", blocked_role=node["role"],
                                 error=type(exc).__name__)
        if (type(evidence) is not dict or evidence.get("challenge") != challenge
                or evidence.get("physical_id") != node["physical_id"]
                or evidence.get("host") != node["host"]
                or evidence.get("local_interface_verified") is not True
                or evidence.get("authenticated") is not True
                or type(evidence.get("machine_id")) is not str
                or len(evidence["machine_id"]) != 32
                or any(c not in "0123456789abcdef" for c in evidence["machine_id"])
                or len(set(evidence["machine_id"])) == 1):
            raise Rejected("missing_or_unbound_physical_identity")
        if evidence["machine_id"] in machine_ids:
            raise Rejected("D0_D1_P_must_be_three_distinct_physical_hosts")
        machine_ids.add(evidence["machine_id"])
        result["nodes"].append(dict(role=node["role"], physical_id=node["physical_id"],
                                   host=node["host"], machine_id=evidence["machine_id"]))
    return result | dict(status="PREFLIGHT_ONLY", reason="identity_gate_passed_no_fault_test")
