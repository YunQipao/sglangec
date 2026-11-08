"""Small immutable artifacts, exact process evidence and sampled Host RSS."""
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def code_hashes():
    root = Path(__file__).parents[2]
    paths = [root / n for n in ("online_chunk_manager.py", "online_chunk.py", "online_rs_exporter.py", "rank_snapshot.py", "node_block_export.py", "launch_online.py", "rs_codec.py", "xor_codec.py",
        "verify_gpu_cold_resume.py", "verify_gpu_restore.py", "gpu_cold_resume_hook.py", "gpu_restore_hook.py",
        "gpu_resume_hook.py", "gpu_kv_restore.py", "cold_resume_state.py", "cold_sse_bridge.py", "cold_timing.py",
        "reprefill_server.py", "perf_stage_e.py", "perf_ports.py", "multinode/tcp_transport.py",
        "multinode/rdma_transport.py", "multinode/verbs_helper.c")]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += [root / "design_study/online_compact.py", root / "design_study/workload.py"]
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in paths if p.exists()}


def process(pid):
    from ...perf_stage_e import process_identity
    identity = process_identity(pid)
    root = Path("/proc") / str(pid)
    env = dict(item.split("=", 1) for item in (root/"environ").read_text().split("\0") if "=" in item)
    argv = (root/"cmdline").read_text().rstrip("\0").split("\0")
    # Never export the full environment (may contain credentials).
    selected = {k: v for k, v in env.items() if k.startswith("SGLANG_KV_EC_") or k in ("CUDA_VISIBLE_DEVICES", "KV_EC_GPU_ADMISSION_MODE")}
    after = process_identity(pid)
    if any(identity[k] != after[k] for k in identity if k != "state"):
        raise ValueError("process changed during evidence read")
    return dict(identity={k:v for k,v in identity.items() if k != "state"},
                observed_state=after["state"], argv=argv, environment=selected)


class RSSSampler:
    """Linux process RSS, not device memory and not theoretical buffer sizes."""
    def __init__(self, pids=None, interval=.02, include_descendants=False):
        self.pids = pids or [os.getpid()]
        self.interval = interval
        self.include_descendants = include_descendants
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        values = {}
        pids = set(self.pids)
        if self.include_descendants:
            queue = list(pids)
            while queue:
                parent = queue.pop()
                for children in (Path("/proc")/str(parent)/"task").glob("*/children"):
                    try:
                        found = {int(p) for p in children.read_text().split()}-pids
                    except (OSError, ValueError):
                        continue
                    pids.update(found)
                    queue.extend(found)
        for pid in sorted(pids):
            try:
                fields = {}
                for line in (Path("/proc")/str(pid)/"status").read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        k, v = line.split(":", 1)
                        fields[k+"_bytes"] = int(v.split()[0])*1024
                stat = (Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
                fields.update(cpu_user_ticks=int(stat[11]),cpu_system_ticks=int(stat[12]),start_ticks=int(stat[19]))
                values[str(pid)] = fields
            except (OSError, ValueError, IndexError) as exc:
                values[str(pid)] = dict(unavailable=type(exc).__name__)
        self.samples.append(dict(monotonic_ns=time.monotonic_ns(), processes=values))

    def _run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(1)
        self.sample()

    def report(self):
        observed = {pid for row in self.samples for pid in row["processes"]}
        return dict(samples=self.samples, sample_interval_s=self.interval,
                    includes_descendants=self.include_descendants,
                    sampled_peak_rss_bytes={p: max((r["processes"].get(p,{}).get("VmRSS_bytes", 0) for r in self.samples), default=0) or None for p in sorted(observed)},
                    self_lifetime_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    note="sampled RSS may miss short peaks; VmHWM and ru_maxrss are lifetime highwater, not per-trial delta")
