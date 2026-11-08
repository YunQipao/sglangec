"""Read-only CPU admission. Shared hosts need no permission to foreign envs."""
import os
from pathlib import Path
import subprocess
import time


def project_activity(proc=Path("/proc")):
    """Reject verifiable project controllers/bench/owned serving, never signal.

    A denied foreign environ is evidence of unknown activity, not proof that
    this project's GPU experiment is active. A denied *identified project*
    process or an incomplete process-table read fails closed.
    """
    shared, active = [], []
    entries = list(proc.iterdir())  # failure to enumerate is not idle
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmd_raw = (entry / "cmdline").read_bytes()
            cmd = cmd_raw.replace(b"\0", b" ").decode(errors="replace").strip()
        except FileNotFoundError:
            continue
        except PermissionError:
            shared.append(dict(pid=int(entry.name), activity="unknown", reason="foreign_cmdline_unreadable"))
            continue
        if not cmd:
            continue
        # Inspect argv, not an inherited shell command containing example text.
        argv = cmd_raw.decode(errors="replace").rstrip("\0").split("\0")
        module = argv[argv.index("-m") + 1] if "-m" in argv and argv.index("-m") + 1 < len(argv) else ""
        known_project = (module.startswith("experiments.kv_ec.") or
                         any("/kv_ec/" in a and a.endswith(".py") for a in argv) or
                         any(a.endswith("/online_benchmark.py") and "kv-ec" in a for a in argv))
        env_unknown = False
        try:
            raw = (entry / "environ").read_bytes()
            env = dict(v.split(b"=", 1) for v in raw.split(b"\0") if b"=" in v)
        except FileNotFoundError:
            continue
        except PermissionError:
            env, env_unknown = {}, True
        owned = bool(env.get(b"KV_EC_PERF_RUN_ID")) or any(
            env.get(k) == b"1" for k in (b"SGLANG_KV_EC_ONLINE", b"SGLANG_KV_EC_DEBUG",
                b"SGLANG_KV_EC_GPU_RESTORE", b"SGLANG_KV_EC_GPU_RESUME", b"SGLANG_KV_EC_GPU_COLD_RESUME"))
        serving = any("sglang" in a for a in argv)
        if known_project or owned:
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] in ("Z", "X"):
                    continue
                start = int(fields[19])
            except FileNotFoundError:
                continue
            except (OSError, ValueError, IndexError) as exc:
                raise RuntimeError("PROJECT_ACTIVITY_UNKNOWN: " + entry.name) from exc
            active.append(dict(pid=int(entry.name), start_ticks=start, module=module,
                               environment="unknown" if env_unknown else "verified"))
        else:
            shared.append(dict(pid=int(entry.name), activity="unknown" if env_unknown else "foreign_process",
                               reason="foreign_environment_unreadable" if env_unknown else "no_project_binding"))
    if active:
        raise RuntimeError("PROJECT_ACTIVITY_PRESENT: " + str(active))
    return shared


def cpu_gate(mode="strict_idle"):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU_ONLY requires explicitly empty CUDA_VISIBLE_DEVICES")
    if mode not in ("shared", "strict_idle"):
        raise ValueError("unknown CPU host mode")
    shared = project_activity()
    result = dict(cpu_only=True, cpu_host_mode=mode, shared_host_activity=shared,
                  isolated_performance_eligible=False, checked_unix=time.time(),
                  gpu_activity="not_queried_in_shared_mode")
    if mode == "strict_idle":
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10, check=True)
        utilization = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10, check=True)
        samples = [int(v.strip()) for v in utilization.stdout.splitlines() if v.strip()]
        if apps.stdout.strip() or not samples or any(samples):
            raise RuntimeError("GPU_ACTIVITY_OR_UNKNOWN: strict_idle requires idle GPUs")
        result.update(compute_pids=[], utilization_gpu_percent=samples, gpu_activity="idle")
    return result
