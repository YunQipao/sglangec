"""Read-only M4 inventory and M5 Kineto traces; never infer missing GPU times."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

from .m4_abcd import summarize


def union_us(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    total, end = 0., float("-inf")
    for start, stop in intervals:
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def trace_record(path):
    trace = json.loads(Path(path).read_text())
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) >= 0]
    scopes = [e for e in events if e.get("name", "").startswith("kv_ec.capture.")]
    if not any(e["name"] == "kv_ec.capture.total" for e in scopes):
        raise ValueError("not_a_capture_trace")
    host = {}
    for event in scopes:
        key = event["name"].removeprefix("kv_ec.capture.")
        host[key] = host.get(key, 0.) + event["dur"] / 1000
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    d2h = [e for e in device if e.get("cat") == "gpu_memcpy" and
           "dtoh" in e.get("name", "").lower()]
    # Link GPU kernels to their CPU launch, rather than calling all capture
    # kernels "gather": mapping checks also launch GPU reductions/unique ops.
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    gather_scopes = [e for e in scopes if e["name"] == "kv_ec.capture.gpu_gather_host"]
    gather = []
    for event in device:
        launch = launches.get(event.get("args", {}).get("correlation"))
        if event.get("cat") == "kernel" and launch and any(
                s["pid"] == launch["pid"] and s["tid"] == launch["tid"]
                and s["ts"] <= launch["ts"] < s["ts"] + s["dur"] for s in gather_scopes):
            gather.append(event)
    return dict(
        path=str(path), host_wall_ms=host,
        host_timing=(json.loads(Path(str(path).replace(".trace.json", ".timing.json")).read_text())
                     if str(path).endswith(".trace.json") else None),
        d2h_device_activity_ms=sum(e["dur"] for e in d2h) / 1000 if d2h else None,
        gather_kernel_activity_ms=sum(e["dur"] for e in gather) / 1000 if gather else None,
        device_activity_union_ms=union_us(device) / 1000 if device else None,
        d2h_activity_count=len(d2h),
        note="Host scopes and device activities overlap. total encloses host scopes. "
             "CPU durations are host wall, not thread CPU time. Device sums may overlap across streams. "
             "Missing CUDA/CUPTI activities are null, never CPU surrogates or zero.")


def inventory(root):
    root = Path(root).resolve()
    # Use the actual saved GPU acceptance, not CPU fixtures or edited verdicts.
    with redirect_stdout(io.StringIO()):
        summary = summarize(root)
    manifest_path = root / "trials.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for trial in manifest["trials"]:
        base = Path(trial["run"])
        for mode in ("protection_only", "online_recovery"):
            ranks = [json.loads((base / mode / "ranks" / f"pause-rank-{r}.json").read_text())
                     for r in range(4)]
            checkpoints = json.loads((base / mode / "checkpoints.json").read_text())
            rows.append(dict(
                repeat=trial["repeat"], mode=mode,
                rank_metrics=[dict(
                    rank=r["tp_rank"], pause_blocking_ms=r["pause_ms"],
                    capture_d2h_hash_host_ms=sum(
                        c["timing_ms"]["capture_d2h_and_hash_ms"] for c in r["protection"]["chunks"]),
                    submit_roundtrip_host_ms=sum(
                        c["timing_ms"]["checkpoint_submit_ms"] for c in r["protection"]["chunks"]),
                    control_roundtrip_host_ms=sum(e["elapsed_ms"] for e in r["trace"]
                                                if "elapsed_ms" in e and
                                                e.get("operation") != "m5_tp_cpu_vote"),
                    tp_cpu_vote_host_ms=(sum(e["elapsed_ms"] for e in r["trace"]
                                             if e.get("operation") == "m5_tp_cpu_vote")
                                         if summary["capture_profile"] else None),
                    protection_timing=[c["timing_ms"] for c in r["protection"]["chunks"]],
                    capture_profile="capture-profiles" in [p.name for p in (base / mode).iterdir()])
                    for r in ranks],
                manager_xor_cpu_wall_ms=[c["xor_ms"] for c in checkpoints],
                manager_commit_enclosing_ms=[c["parity_commit_ms"] for c in checkpoints]))
    return dict(
        state="AUDITED_SAVED_GPU_EVIDENCE", root=str(root),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        measured_sources=manifest["source_sha256"],
        capture_profile=summary["capture_profile"], modes=summary["modes"], protection=rows,
        traces=[trace_record(p) for p in sorted(root.rglob("capture-profiles/capture-*.trace.json"))],
        missing_in_legacy_M4=["pure CUDA gather", "pure CUDA D2H", "CPU copy/concatenate",
                              "hash/metadata breakdown", "coordination sleep and vote breakdown"],
        timing_contract="Do not sum capture, submit, commit, pause, client or recovery enclosing timers. "
                        "Rank timers are concurrent; retain individual ranks. Profiler export and CUPTI "
                        "overhead perturb foreground latency. This is not a new GPU experiment.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="exclusive new JSON; never modifies input")
    args = parser.parse_args()
    result = inventory(args.run)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"PASS saved-evidence inventory: {args.output}")


if __name__ == "__main__":
    main()
