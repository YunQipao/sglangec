"""Shared exclusive evidence output and Linux process/environment observations."""

import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import time


def process_usage():
    u = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "pid": os.getpid(),
        "user_cpu_s": u.ru_utime,
        "system_cpu_s": u.ru_stime,
        "peak_rss_bytes": u.ru_maxrss * 1024,
        "monotonic": time.monotonic(),
    }
    try:
        pages = Path("/proc/self/statm").read_text().split()
        result["rss_bytes"] = int(pages[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        result["rss_bytes"] = None
    return result


def command(argv, cwd=None):
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=15)
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc), "status": "NOT VERIFIED"}


def usage_delta(before, after):
    wall = after["monotonic"] - before["monotonic"]
    cpu = sum(after[k] - before[k] for k in ("user_cpu_s", "system_cpu_s"))
    return {
        "wall_s": wall,
        "cpu_s": cpu,
        "cpu_percent_one_core": cpu / wall * 100 if wall > 0 else None,
        "rss_bytes_before": before.get("rss_bytes"),
        "rss_bytes_after": after.get("rss_bytes"),
        "peak_rss_bytes_after": after.get("peak_rss_bytes"),
    }


def network_observation(peer):
    """Read routing and real interface counters; counts include other traffic."""
    route = command(["ip", "-j", "route", "get", peer])
    result = {"peer": peer, "route": route, "physical_device": False}
    try:
        interface = json.loads(route["stdout"])[0]["dev"]
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
            raise ValueError("invalid interface name")
        root = Path("/sys/class/net") / interface
        result.update(
            interface=interface,
            physical_device=(root / "device").exists(),
            counters={
                k: int((root / "statistics" / k).read_text())
                for k in ("tx_bytes", "rx_bytes", "tx_packets", "rx_packets")
            },
        )
        try:
            result["link_speed_mbps"] = int((root / "speed").read_text())
        except (OSError, ValueError):
            result["link_speed_mbps"] = None
    except (OSError, ValueError, KeyError, IndexError) as exc:
        result.update(status="NOT VERIFIED", reason=str(exc))
    return result


def network_delta(before, after):
    if (
        before.get("interface") != after.get("interface")
        or not before.get("counters")
        or not after.get("counters")
    ):
        return {"status": "NOT VERIFIED"}
    return {
        "interface": before["interface"],
        "counter_delta": {
            k: after["counters"][k] - v for k, v in before["counters"].items()
        },
        "note": "actual interface counters; includes unrelated traffic and kernel/offload accounting; not per-flow Ethernet wire bytes",
    }


def environment():
    root = Path(__file__).resolve().parents[2]
    diff = command(["git", "diff", "HEAD", "--", "experiments/kv_ec", "python"], root)
    return {
        "git": command(["git", "rev-parse", "HEAD"], root),
        "git_status": command(["git", "status", "--porcelain"], root),
        "tracked_diff_sha256": hashlib.sha256(
            diff.get("stdout", "").encode()
        ).hexdigest(),
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
                "--format=csv",
            ]
        ),
        "pcie_topology": command(["nvidia-smi", "topo", "-m"]),
        "uname": command(["uname", "-a"]),
    }


def save_report(prefix, report):
    prefix = Path(prefix)
    paths = [prefix.with_suffix(".json"), prefix.with_suffix(".csv")]
    # O_EXCL on BOTH before writing: no existing file is ever truncated.
    opened = []
    try:
        for path in paths:
            opened.append(path.open("x", newline=""))
        json.dump(report, opened[0], indent=2, allow_nan=False)
        opened[0].write("\n")
        writer = csv.writer(opened[1])
        writer.writerow(["metric", "value_json"])

        def flatten(obj, key=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(v, f"{key}.{k}" if key else k)
            else:
                writer.writerow([key, json.dumps(obj, sort_keys=True, allow_nan=False)])

        flatten(report)
    finally:
        for f in opened:
            f.close()
