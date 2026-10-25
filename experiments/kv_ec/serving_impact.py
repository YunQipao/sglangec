"""M4 measured dual-instance requests with explicit offline third-arm labeling.

Start matching launch_debug instances yourself: baseline / parity / capture.
No-EC and local-XOR measure live inference. Third arm measures capture overhead
while serving, THEN offline local XOR + remote RS; it is NOT online remote EC.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from .rs_experiment import command, environment, save_report


def sample_processes(roots):
    """Observe supplied experiment PIDs plus current descendants; never signal."""
    stats = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            stats[int(path.parent.name)] = dict(
                ppid=int(fields[1]),
                start_ticks=int(fields[19]),
                cpu_s=(int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
                rss_bytes=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            )
        except (OSError, ValueError, IndexError):
            continue
    selected = set(roots)
    while True:
        children = {pid for pid, s in stats.items() if s["ppid"] in selected}
        if children <= selected:
            break
        selected |= children
    return {str(pid): stats[pid] for pid in sorted(selected) if pid in stats}


class Sampler:
    def __init__(self, pids, gpus, interval):
        self.pids, self.gpus, self.interval = pids, gpus, interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop_event.is_set():
            self.samples.append(
                dict(
                    monotonic=time.monotonic(),
                    processes=sample_processes(self.pids),
                    gpu=command(
                        [
                            "nvidia-smi",
                            "-i",
                            self.gpus,
                            "--query-gpu=index,uuid,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ]
                    ),
                )
            )
            self.stop_event.wait(self.interval)

    def summary(self):
        intervals = []
        for a, b in zip(self.samples, self.samples[1:]):
            elapsed = b["monotonic"] - a["monotonic"]
            common = set(a["processes"]) & set(b["processes"])
            cpu = sum(
                b["processes"][p]["cpu_s"] - a["processes"][p]["cpu_s"]
                for p in common
                if a["processes"][p]["start_ticks"] == b["processes"][p]["start_ticks"]
            )
            intervals.append(
                dict(
                    wall_s=elapsed,
                    cpu_percent_one_core=cpu / elapsed * 100 if elapsed else None,
                    rss_sum_bytes=sum(s["rss_bytes"] for s in b["processes"].values()),
                )
            )
        return dict(
            samples=self.samples,
            intervals=intervals,
            note="sampled experiment PIDs and descendants; RSS sum can double-count shared pages; short requests may have no complete sampling interval",
        )


def validate_response_capture(response, block):
    from .node_block_export import unpack

    meta, _ = unpack(block)
    rank = meta["ranks"][0]
    chunks = {c["token_start"]: c for c in response["input_chunks"]}
    chunk = chunks.get(rank["token_start"])
    if (
        rank["request_id"]
        != hashlib.sha256(response["request_id"].encode()).hexdigest()
        or rank["model_identifier"] != response["model_identifier"]
        or not chunk
        or rank["token_count"] != chunk["token_count"]
        or rank["token_ids_sha256"] != chunk["token_ids_sha256"]
    ):
        raise ValueError("capture does not belong to this measured request/chunk")


def run(args):
    from . import request_debug

    args.output.mkdir(parents=True, exist_ok=False)
    sampler = Sampler(args.pids, args.gpu_indices, args.sample_interval)
    report = dict(
        state="RUNNING",
        mode=args.mode,
        environment=environment(),
        online_remote_serving_interference="NOT VERIFIED",
        measurements=[],
        third_arm_note="capture during inference; local XOR and remote RS AFTER both requests finish",
    )
    sampler.thread.start()
    started = time.monotonic()
    try:

        def request(node):
            p = args.output / f"node{node}-response.json"
            request_debug.request(
                SimpleNamespace(
                    url=getattr(args, f"node{node}_url"),
                    model=args.model,
                    request_id=getattr(args, f"node{node}_request"),
                    prompt_tokens=768,
                    stream_metrics=True,
                    output=p,
                )
            )
            return json.loads(p.read_text())

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(request, (0, 1)))
        finished = time.monotonic()
        report["responses"] = responses
        report["concurrent_wall_ms"] = (finished - started) * 1000
        report["aggregate_output_tokens_per_s"] = sum(
            len(r["output_ids"]) for r in responses
        ) / (finished - started)
        for node, r in enumerate(responses):
            report["measurements"].append(
                dict(
                    node=node,
                    **r["client_metrics_ms"],
                    output_tokens_per_s=len(r["output_ids"])
                    / (r["client_metrics_ms"]["request_total_ms"] / 1000),
                    protection_commit_ms=None,
                    unprotected_window_ms=None,
                )
            )
        if args.mode == "local-xor":
            from .dual_node_check import check

            nodes = [
                dict(
                    url=getattr(args, f"node{i}_url"),
                    socket=str(getattr(args, f"node{i}_socket")),
                    snapshot=getattr(args, f"node{i}_snapshot"),
                    request=getattr(args, f"node{i}_request"),
                )
                for i in (0, 1)
            ]
            commits = check(nodes, args.timeout)
            for i, c in enumerate(commits):
                s = c["status"]
                report["measurements"][i].update(
                    protection_commit_ms=s["parity_commit_ms"],
                    unprotected_window_ms=None,
                    window_note="manager first-rank receipt to commit only; capture start-to-commit window NOT VERIFIED",
                )
            report["commits"] = commits
        elif args.mode == "offline-remote-rs":
            from .node_block_export import export
            from .dual_node_rs import run as remote_run

            files = []
            for node in (0, 1):
                directory = args.output / f"node{node}-export"
                directory.mkdir()
                block, stats = export(getattr(args, f"node{node}_capture"), node)
                validate_response_capture(responses[node], block)
                with (directory / "node.bin").open("xb") as f:
                    f.write(block)
                save_report(directory / "export", stats)
                files.append(directory / "node.bin")
            remote = remote_run(
                SimpleNamespace(
                    remote_host=args.remote_host,
                    remote_port=args.remote_port,
                    timeout=args.timeout,
                    node0_block=files[0],
                    node1_block=files[1],
                    stripe_id=args.stripe_id,
                    chunk_tokens=args.chunk_tokens,
                    network_profile=args.network_profile,
                    output=args.output / "remote",
                )
            )
            report["offline_remote"] = dict(
                state=remote["state"],
                committed_ms=remote["remote_committed_observed_ms"],
                serving_overlap=False,
            )
        report["state"] = "PASS"
    except Exception as exc:
        report.update(state="FAIL", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        sampler.stop_event.set()
        sampler.thread.join(20)
        report["resources"] = sampler.summary()
        save_report(args.output / "result", report)
    return report


def compare(paths, output):
    from .request_debug import compare as compare_requests

    if len(paths) != 3:
        raise ValueError("three experiment directories required")
    reports = [json.loads((p / "result.json").read_text()) for p in paths]
    if {r["mode"] for r in reports} != {
        "no-ec",
        "local-xor",
        "offline-remote-rs",
    } or any(r["state"] != "PASS" for r in reports):
        raise ValueError("three successful distinct arms required")
    for other in paths[1:]:
        for i in (0, 1):
            compare_requests(
                [paths[0] / f"node{i}-response.json", other / f"node{i}-response.json"]
            )
    output.mkdir(parents=True, exist_ok=False)
    save_report(
        output / "comparison",
        dict(
            state="PASS",
            reports=reports,
            online_remote_serving_interference="NOT VERIFIED",
            note="third arm is offline, not online Local XOR + Remote RS",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument(
        "--mode", choices=("no-ec", "local-xor", "offline-remote-rs"), required=True
    )
    r.add_argument("--model", required=True)
    r.add_argument(
        "--pids",
        type=int,
        nargs="+",
        required=True,
        help="your experiment server and manager PIDs only",
    )
    r.add_argument("--gpu-indices", default="0,1,2,3")
    r.add_argument("--sample-interval", type=float, default=0.5)
    r.add_argument("--remote-host")
    r.add_argument("--remote-port", type=int, default=30300)
    r.add_argument("--chunk-tokens", type=int, choices=(128, 256, 512), default=256)
    r.add_argument(
        "--network-profile",
        choices=("physical", "tc", "loopback-test"),
        default="physical",
    )
    r.add_argument("--stripe-id")
    r.add_argument("--timeout", type=float, default=60)
    r.add_argument("--output", type=Path, required=True)
    for i, port in enumerate((30292, 30294)):
        r.add_argument(f"--node{i}-url", default=f"http://127.0.0.1:{port}")
        r.add_argument(f"--node{i}-request", required=True)
        r.add_argument(f"--node{i}-socket", type=Path)
        r.add_argument(f"--node{i}-snapshot")
        r.add_argument(f"--node{i}-capture", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("directories", nargs=3, type=Path)
    c.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.command == "compare":
        compare(args.directories, args.output)
    else:
        if args.sample_interval <= 0 or any(pid <= 0 for pid in args.pids):
            p.error("positive sample interval and PIDs required")
        if args.mode == "local-xor" and any(
            getattr(args, f"node{i}_{field}") is None
            for i in (0, 1)
            for field in ("socket", "snapshot")
        ):
            p.error("local-xor requires both node sockets and snapshots")
        if args.mode == "offline-remote-rs":
            if (
                not args.remote_host
                or not args.stripe_id
                or any(getattr(args, f"node{i}_capture") is None for i in (0, 1))
            ):
                p.error(
                    "offline-remote-rs requires remote host, stripe ID and both capture directories"
                )
            if args.chunk_tokens != 256:
                p.error(
                    "M4 request/chunk association currently requires --chunk-tokens 256"
                )
        run(args)


if __name__ == "__main__":
    main()
