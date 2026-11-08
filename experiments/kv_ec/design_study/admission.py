"""Placement/resource admission with read-only local interface verification."""

import ipaddress
import json
from pathlib import Path
import subprocess

from ..node_block_export import canonical, sha

PROTECTED_PORTS = {30701, 31700, 31702, 31704}
BASELINES = ("base", "cpu_replication", "remote_replication", "rs_only", "two_level")


def placement(config):
    if config.get("schema") != "ec-three-domain-v1":
        raise ValueError("placement schema required")
    nodes = config.get("nodes", [])
    if len(nodes) != 3 or [n.get("role") for n in nodes] != ["D0", "D1", "P"]:
        raise ValueError("ordered D0,D1,P required")
    if [n.get("physical_id") for n in nodes] != ["232", "233", "243"]:
        raise ValueError("explicit physical 232/233/243 required")
    if len({n["host"] for n in nodes}) != 3:
        raise ValueError("distinct host addresses required")
    # The 192.168.2.243 and .245 aliases belong to physical232/233; neither is P.
    expected = {"232": "10.102.0.232", "233": "10.102.0.233", "243": "10.102.0.243"}
    if any(n["host"] != expected[n["physical_id"]] for n in nodes):
        raise ValueError("physical address mismatch; 192.168.2.243/.245 are D0/D1 aliases")
    transport = config.get("transport")
    if transport not in ("tcp", "rdma"):
        raise ValueError("explicit tcp/rdma required")
    if transport == "rdma" and any(n.get("rdma_available") is not True for n in nodes):
        raise ValueError("HostRDMA unavailable/unattested: no TCP fallback")
    ports = [config.get(k) for k in ("parity_port", "replica_port")]
    if any(type(p) is not int or not 1024 <= p <= 65535 or p in PROTECTED_PORTS for p in ports) or len(set(ports)) != 2:
        raise ValueError("dedicated unprivileged ports required; healthy RS ports protected")
    for name in ("max_bytes", "max_objects", "ttl_s", "timeout_s"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError("positive integer budget: " + name)
    return sha(canonical(config))


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate interface evidence key")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError("invalid JSON constant: " + value)


def _local_interface_addresses():
    """Read kernel address evidence; malformed or incomplete output fails closed."""
    result = subprocess.run(["ip", "-j", "address"], check=True,
                            capture_output=True, text=True, timeout=5)
    interfaces = json.loads(result.stdout, object_pairs_hook=_unique_json_object,
                            parse_constant=_invalid_json_constant)
    if type(interfaces) is not list or not interfaces:
        raise ValueError("missing interface evidence")
    indices, names, addresses = set(), set(), []
    for interface in interfaces:
        if type(interface) is not dict:
            raise ValueError("invalid interface evidence")
        index, name = interface.get("ifindex"), interface.get("ifname")
        flags, state, infos = (interface.get(k) for k in ("flags", "operstate", "addr_info"))
        if (type(index) is not int or index <= 0 or type(name) is not str or not name
                or index in indices or name in names
                or type(flags) is not list or any(type(f) is not str for f in flags)
                or type(state) is not str or not state
                or type(infos) is not list):
            raise ValueError("missing/ambiguous interface evidence")
        indices.add(index)
        names.add(name)
        for info in infos:
            if type(info) is not dict:
                raise ValueError("invalid address evidence")
            family, local, prefix = (info.get(k) for k in ("family", "local", "prefixlen"))
            if family not in ("inet", "inet6") or type(local) is not str:
                raise ValueError("missing address evidence")
            address = ipaddress.ip_address(local)
            if (address.version != (4 if family == "inet" else 6)
                    or type(prefix) is not int or not 0 <= prefix <= address.max_prefixlen
                    or any(type(info.get(k, False)) is not bool for k in ("tentative", "dadfailed"))):
                raise ValueError("invalid address evidence")
            usable = ("UP" in flags and state in ("UP", "UNKNOWN")
                      and not info.get("tentative", False) and not info.get("dadfailed", False))
            addresses.append((str(address), usable))
    return addresses


def local_identity(config, role, physical_id):
    """Require an explicit operator ID and one unambiguous local placement IP.

    DNS is not proof of locality. Kernel interface evidence also covers hosts
    whose hostname does not resolve to their configured placement address.
    """
    placement(config)
    matches = [n for n in config["nodes"] if n["role"] == role]
    if len(matches) != 1 or physical_id != matches[0]["physical_id"]:
        raise ValueError("local physical identity/address not verified")
    node = matches[0]
    try:
        addresses = _local_interface_addresses()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("local physical identity/address not verified: interface evidence unavailable/invalid") from exc
    placement_hosts = {n["host"] for n in config["nodes"]}
    local_hosts = [(address, usable) for address, usable in addresses if address in placement_hosts]
    if local_hosts != [(node["host"], True)]:
        raise ValueError("local physical identity/address not verified: missing/ambiguous local address")
    return node


def require_host_rdma(sysfs=Path("/sys/class/infiniband"), devices=Path("/dev/infiniband")):
    if not sysfs.is_dir() or not any(sysfs.iterdir()) or not devices.is_dir() or not any(devices.iterdir()):
        raise ValueError("HostRDMA requires HCA and device nodes; fail closed")


def uniform_resources(candidates, evidence):
    """Choose one measured configuration for ALL baselines, never extrapolate.

    candidates are ordered desired -> fallback; each baseline must have a
    passing admission record for that exact configuration and fixed trace.
    A record is an operator's preflight evidence, not a memory model.
    """
    for candidate in candidates:
        if (set(candidate) != {"tp", "input_tokens", "output_tokens", "batch", "dtype", "trace_sha256"}
                or candidate["tp"] not in (2, 4)
                or any(type(candidate[k]) is not int or candidate[k] <= 0 for k in ("input_tokens", "output_tokens", "batch"))):
            raise ValueError("invalid uniform candidate")
        accepted = []
        for baseline in BASELINES:
            matches = [r for r in evidence if r.get("baseline") == baseline and r.get("config") == candidate
                       and r.get("passed") is True and r.get("preflight_artifact_sha256")
                       and type(r.get("free_bytes")) is int and type(r.get("required_bytes")) is int
                       and 0 <= r["required_bytes"] <= r["free_bytes"]]
            if not matches:
                break
            accepted.append(matches[0])
        if len(accepted) == len(BASELINES):
            return dict(config=candidate, baselines=list(BASELINES), evidence=accepted,
                        scope="resource admission only; not serving or recovery validation")
    raise ValueError("no uniformly admitted configuration; TP4/64K/4K output not presumed")


def memory_ledger(*, original, stable_parity=0, stable_replica=0, snapshots=0, temporary=0):
    values = (original, stable_parity, stable_replica, snapshots, temporary)
    if any(type(v) is not int or v < 0 for v in values) or not original:
        raise ValueError("nonnegative exact byte counts and positive original required")
    stable = stable_parity + stable_replica
    return dict(original_bytes=original, stable_parity_bytes=stable_parity,
                stable_replica_bytes=stable_replica, snapshot_bytes=snapshots,
                temporary_bytes=temporary, stable_redundancy_bytes=stable,
                stable_overhead_pct=100 * stable / original,
                accounted_additional_peak_bytes=stable + snapshots + temporary,
                note="allocation ledger, not measured RSS; no overlapping allocations counted twice")
