"""Read-only dual TP2/TP4 health/commit check and isolated log collection."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from .parity_manager import manager_hello, submit_once


def check(nodes, timeout, tp_size=2):
    if type(tp_size) is not int or tp_size not in (2, 4):
        raise ValueError('TP must be 2 or 4')
    if len({n["socket"] for n in nodes}) != 2 or len({n["url"] for n in nodes}) != 2:
        raise ValueError("nodes require independent sockets and URLs")
    if (
        len({n["snapshot"] for n in nodes}) != 2
        or len({n["request"] for n in nodes}) != 2
    ):
        raise ValueError("nodes require independent snapshot and request identities")
    deadline = time.monotonic() + timeout
    results = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for node in nodes:
        with opener.open(node["url"].rstrip("/") + "/health", timeout=5) as response:
            if response.status != 200:
                raise ValueError("unhealthy inference instance")
        epoch = manager_hello(node["socket"])
        while True:
            status = submit_once(
                node["socket"],
                {"kind": "status", "snapshot_id": node["snapshot"]},
                b"",
                timeout=5,
            )
            if status["state"] == "COMMITTED":
                break
            if status["state"] in ("FAILED", "EXPIRED") or time.monotonic() >= deadline:
                raise RuntimeError(status)
            time.sleep(0.1)
        if (
            status.get("manager_epoch") != epoch
            or status.get("tp_size") != tp_size
            or status.get("request_id")
            != hashlib.sha256(node["request"].encode()).hexdigest()
            or set(status.get("rank_hashes", {})) != {str(r) for r in range(tp_size)}
        ):
            raise ValueError("committed node binding mismatch")
        results.append(dict(node=node, status=status))
    if results[0]["status"]["manager_epoch"] == results[1]["status"]["manager_epoch"]:
        raise ValueError("manager epochs must be independent")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tp', type=int, choices=(2, 4), default=2)
    for i, port in enumerate((30292, 30294)):
        p.add_argument(f"--node{i}-url", default=f"http://127.0.0.1:{port}")
        for field in ("socket", "snapshot", "request"):
            p.add_argument(f"--node{i}-{field}", required=True)
        p.add_argument(f"--node{i}-logs", nargs="*", type=Path, default=[])
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--output", required=True, type=Path, help="new evidence directory")
    args = p.parse_args()
    nodes = [
        {
            k: getattr(args, f"node{i}_{k}")
            for k in ("url", "socket", "snapshot", "request")
        }
        for i in range(2)
    ]
    results = check(nodes, args.timeout, args.tp)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "checks.json").open("x") as f:
        json.dump(results, f, indent=2)
    import csv

    with (args.output / "checks.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "snapshot", "state", "parity_bytes", "xor_ms"])
        for i, result in enumerate(results):
            s = result["status"]
            writer.writerow(
                [i, s["snapshot_id"], s["state"], s["parity_bytes"], s["xor_ms"]]
            )
    for i in range(2):
        for j, path in enumerate(getattr(args, f"node{i}_logs")):
            with (args.output / f"node{i}-{j}-{path.name}").open("xb") as f:
                f.write(path.read_bytes())
    print(json.dumps({"state": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
