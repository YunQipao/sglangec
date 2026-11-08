"""Parent-invoked E2-A: fresh real CUDA capture -> controlled loss -> new GPU.

No work on import. Local TP2/prefix256/17 outputs only; no SSH or RS service.
Each baseline/repeat has a fresh capture/restore_id and exact-owned lifetime.
Fault scope is KV invalidation plus instance restart, NOT physical GPU failure.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

import numpy as np

from ... import cold_sse_bridge, verify_gpu_cold_resume as cold, xor_codec
from ...cold_resume_state import load_checkpoint, require_origins_stopped, sealed_origin_processes
from ...cold_timing import PROFILE, build_timing, stream_request
from ...node_block_export import pack, unpack
from ...perf_ports import PortLease
from ...perf_stage_e import Host, SELF, wait_health
from .evidence import RSSSampler, code_hashes, save, sha

BASELINES = ("reprefill", "replication", "local-xor")
PORT = 32962
MODULE = "experiments.kv_ec.verify_gpu_cold_resume"


def read(path):
    return json.loads(Path(path).read_text())


def recovery_code_hashes():
    values=code_hashes()
    root=Path(__file__).parents[2]
    for name in ('prefill_observer.py','gpu_cold_resume_hook.py','gpu_resume_hook.py',
                 'gpu_restore_hook.py','gpu_kv_restore.py','cold_resume_state.py',
                 'cold_sse_bridge.py','cold_timing.py','verify_gpu_cold_resume.py',
                 'verify_gpu_resume.py','launch_debug.py','reprefill_server.py',
                 'perf_stage_e.py','pidfd_compat.py'):
        values[name]=sha((root/name).read_bytes())
    for name in ('python/sglang/srt/managers/scheduler.py',
                 'python/sglang/srt/managers/scheduler_components/batch_result_processor.py'):
        values[name]=sha((root.parents[1]/name).read_bytes())
    return values


def signature(config):
    return sha(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())


def validate_config(config):
    """Single local host; parent supplies the reviewed Qwen/env/template paths."""
    if config.get("ssh") or config.get("tp_size", 2) != 2 or config.get("prefix_tokens", 256) != 256:
        raise ValueError("local TP2 prefix256 only; no SSH")
    for key in ("repo", "python", "model", "input_ids", "template", "ledger", "conda_prefix", "cuda_home"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError("absolute reviewed config path required: " + key)
    gpus = config.get("gpus", "").split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus):
        raise ValueError("two explicit physical GPU indices required")
    if config.get("port", PORT) != PORT:
        raise ValueError("fixed port 32962; never rotate occupied ports")
    if config.get("gpu_admission_mode", "shared") not in ("strict", "shared"):
        raise ValueError("invalid GPU admission mode")
    if not .05 <= config.get("mem_fraction_static", .6) <= .95:
        raise ValueError("invalid static memory fraction")
    if not Path(config["ledger"]).is_dir():
        raise ValueError("parent must provide one existing shared replay ledger")
    ids = read(config["input_ids"])
    ids = ids.get("input_ids") if isinstance(ids, dict) else ids
    if not isinstance(ids, list) or len(ids) != 256 or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("exact prefix256 input IDs required")
    for key, default, low, high in (("startup_timeout", 900, 1, 1800),
                                   ("request_timeout", 300, 1, 900),
                                   ("resource_interval", 1, .5, 60)):
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("bounded config value required: " + key)
    return ids


def prepare_owned_capture(bundle, baseline):
    """Real source validation precedes any erasure; retain only owned arrays."""
    if baseline not in BASELINES:
        raise ValueError("unknown baseline")
    source, state, receipt, _ = load_checkpoint(bundle)
    raw = source.load()
    meta, payloads = unpack(raw)  # production provenance/layout/hash checks
    if state["token_count"] != 256 or state["tp_size"] != 2:
        raise ValueError("E2-A requires fresh TP2 prefix256 capture")
    active = [np.frombuffer(p, dtype=np.uint8).copy() for p in payloads]
    begin = time.monotonic_ns()
    replicas = [p.copy() for p in active] if baseline == "replication" else None
    parity = xor_codec.encode(active) if baseline == "local-xor" else None
    prepared = time.monotonic_ns()
    return dict(active=active, replicas=replicas, parity=parity, meta=meta,
        state=state, receipt=receipt, bundle_sha256=sha(raw),
        rank_sha256=[sha(p) for p in payloads],
        protection_prepare_start_ns=begin, protection_prepare_complete_ns=prepared,
        protection_prepare_ms=(prepared-begin)/1e6,
        stable_replica_bytes=sum(p.nbytes for p in replicas) if replicas is not None else 0,
        stable_local_parity_bytes=parity.nbytes if parity is not None else 0,
        source_rank_bytes=sum(p.nbytes for p in active))


def remove_owned_payload(path, *, zero=False):
    """Only a fresh trial's validated payload, no symlinks or shared hardlinks."""
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise ValueError("unsafe owned capture payload: " + str(path))
    if zero:
        with path.open("r+b") as stream:
            remaining = info.st_size
            zeros = bytes(min(1024**2, remaining))
            while remaining:
                count = min(len(zeros), remaining)
                stream.write(zeros[:count])
                remaining -= count
            stream.flush()
            os.fsync(stream.fileno())
        with path.open("rb") as stream:
            while block := stream.read(1024**2):
                if any(block):
                    raise ValueError("owned disk erasure failed")
    path.unlink()
    return dict(file=path.name, bytes=info.st_size, zeroed_before_unlink=zero, removed=True)


def erase_and_rebuild(prepared, bundle, restored, baseline, lost_rank=0):
    """No source reload after fault; rebuild from CPU replica or XOR+survivor."""
    if baseline not in BASELINES or lost_rank not in (0, 1):
        raise ValueError("invalid fault")
    bundle, restored = Path(bundle).resolve(), Path(restored).resolve()
    if bundle == restored or bundle in restored.parents or restored.exists():
        raise ValueError("independent fresh reconstructed directory required")
    payload_paths = [bundle/"global_node.bin", *(bundle/f"rank_{r}.bin" for r in range(2))]
    # Resolve and verify every target before the first destructive operation.
    for path in payload_paths:
        info = path.lstat()
        if path.is_symlink() or not path.is_file() or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError("unsafe original payload")
    hashes = [sha(path.read_bytes()) for path in payload_paths]
    if hashes != [prepared["bundle_sha256"], *prepared["rank_sha256"]]:
        raise ValueError("capture changed before fault")
    active = prepared["active"]
    lost = active[lost_rank]
    before = sha(lost)
    lost.fill(0)
    proof = dict(kind="owned_KV_loss_and_instance_restart", lost_rank=lost_rank,
                 before_sha256=before, erased_sha256=sha(lost), erased_bytes=lost.nbytes,
                 zeroed=not np.any(lost), physical_gpu_failure=False, physical_node_failure=False,
                 gpu_memory_zeroization_claimed=False)
    active[lost_rank] = None
    del lost
    save(bundle/"fault-started.json", proof)
    proof["original_payloads"] = [remove_owned_payload(p, zero=p.name == f"rank_{lost_rank}.bin") for p in payload_paths]
    save(bundle/"fault-erasure.json", proof)
    if baseline == "reprefill":
        return proof, None
    begin = time.monotonic_ns()
    if baseline == "replication":
        active[lost_rank] = prepared["replicas"][lost_rank].copy()
    else:
        active[lost_rank] = xor_codec.recover(active, prepared["parity"])
    if [sha(p) for p in active] != prepared["rank_sha256"]:
        raise ValueError("reconstructed ranks differ from fresh capture")
    block = pack(prepared["meta"]["node_id"], [p.tobytes() for p in active], prepared["meta"]["ranks"])
    if sha(block) != prepared["bundle_sha256"]:
        raise ValueError("reconstructed complete global_node hash mismatch")
    restored.mkdir()
    seals = ["request_state.json", "COMMITTED.json", *(f"rank_{r}.{suffix}.json" for r in range(2) for suffix in ("meta", "capture"))]
    seal_hashes = {}
    for name in seals:
        raw = (bundle/name).read_bytes()
        with (restored/name).open("xb") as stream:
            stream.write(raw)
        seal_hashes[name] = sha(raw)
        if sha((restored/name).read_bytes()) != sha(raw):
            raise ValueError("seal/proof copy changed")
    with (restored/"global_node.bin").open("xb") as stream:
        stream.write(block)
    for rank, payload in enumerate(active):
        with (restored/f"rank_{rank}.bin").open("xb") as stream:
            stream.write(payload)
    _, state, _, _ = load_checkpoint(restored)  # ALL original seal/layout/proof checks
    if state != prepared["state"] or any(p.exists() for p in payload_paths):
        raise ValueError("restore isolation or state check failed")
    return proof, dict(restored_bundle=str(restored), byte_exact=True, seal_sha256=seal_hashes,
        global_node_sha256=sha(block), rank_sha256=[sha(p) for p in active],
        reconstruction_validate_persist_ms=(time.monotonic_ns()-begin)/1e6,
        route="complete_cpu_replica" if baseline == "replication" else "local_cpu_xor_survivor")


def reference_view(reference, restored, output):
    """Explicit path relocation only; immutable evidence/seals are never edited."""
    reference, output = Path(reference), Path(output)
    output.mkdir(exist_ok=False)
    cfg = read(reference/"run.json")
    old_hash = sha((reference/"run.json").read_bytes())
    old_bundle = cfg["capture_bundle"]
    cfg["capture_bundle"] = str(Path(restored).resolve())
    for path in reference.iterdir():
        if path.is_file() and path.name != "run.json":
            shutil.copyfile(path, output/path.name)
    save(output/"run.json", cfg)
    return dict(original_run_sha256=old_hash, relocated_run_sha256=sha((output/"run.json").read_bytes()),
        only_changed_field="capture_bundle", original_bundle=old_bundle, reconstructed_bundle=cfg["capture_bundle"])


class LocalTrial:
    """Existing Host/SELF supervisor owns signals, sessions, pidfds and cleanup."""
    def __init__(self, config, root, run_id, deadline):
        self.config, self.root, self.run_id, self.deadline = config, Path(root), run_id, deadline
        self.host = Host(config, root, run_id, 0, config.get("gpu_admission_mode", "shared"))
        self.host.env.update({key:"0" for key in ("SGLANG_KV_EC_ONLINE", "SGLANG_KV_EC_DEBUG",
            "SGLANG_KV_EC_PAUSE_RESUME", "SGLANG_KV_EC_GPU_RESTORE", "SGLANG_KV_EC_GPU_RESUME",
            "SGLANG_KV_EC_GPU_COLD_RESUME", "SGLANG_KV_EC_RS_RESUME")})
        self.host.env.update(SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0", SGLANG_KV_EC_PREFILL_OBSERVE="1")
        self.services, self.handles = {}, []
        self.stop_pulse = threading.Event()

    def budget(self, limit):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("E2-A lifecycle budget exhausted; fresh run required")
        return min(limit, remaining)

    def run(self, module, *args, timeout=600):
        return self.host.run(module, *args, timeout=self.budget(timeout))

    def start(self, name, module, *args):
        if name in self.services:
            raise ValueError("service lifetime name reused")
        self.budget(1)
        log = (self.root/f"supervisor-{name}.log").open("x")
        self.handles.append(log)
        command = self.host.argv(SELF, "service", "--root", self.root/name,
            "--resource-interval", self.config.get("resource_interval", 1), "--",
            self.config["python"], "-m", module, *args)
        self.services[name] = subprocess.Popen(command, cwd=self.config["repo"], stdout=log, stderr=subprocess.STDOUT)

    def stop(self, name):
        if name not in self.services:
            return
        # Cleanup has its own bounded grace even when the experiment expires.
        self.host.run(SELF, "stop", "--root", self.root/name, timeout=30)
        child = self.services[name]
        if child.wait(timeout=90) != 0:
            raise RuntimeError("owned supervisor failed: " + name)
        exit_record = read(self.root/name/"exit.json")
        if exit_record.get("reason") != "owned_stop_requested":
            raise RuntimeError("owned service ended unexpectedly: " + str(exit_record))
        del self.services[name]

    def pulse(self):
        while not self.stop_pulse.wait(2):
            (self.root/"lease").touch()

    @contextmanager
    def owned(self):
        self.root.mkdir(exist_ok=False)
        save(self.root/"trial-owner.json", dict(run_id=self.run_id))
        (self.root/"lease").touch(exist_ok=False)
        previous = {k:os.environ.get(k) for k in ("KV_EC_PERF_RUN_ID", "KV_EC_PERF_DIR")}
        os.environ.update(KV_EC_PERF_RUN_ID=self.run_id, KV_EC_PERF_DIR=str(self.root/"perf"))
        thread = threading.Thread(target=self.pulse, daemon=True)
        thread.start()
        try:
            with PortLease([PORT], self.root, self.run_id):
                try:
                    yield self
                finally:
                    errors = []
                    for name in reversed(list(self.services)):
                        try:
                            self.stop(name)
                        except Exception as exc:
                            errors.append(dict(service=name, error=f"{type(exc).__name__}: {exc}"))
                    save(self.root/"cleanup.json", dict(status="FAIL" if errors else "PASS", errors=errors))
                    if errors:
                        raise RuntimeError("exact-owned cleanup failed; inspect cleanup.json")
        finally:
            self.stop_pulse.set()
            thread.join(3)
            for handle in self.handles:
                handle.close()
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def one_trial(config, root, run_id, baseline, deadline):
    record = dict(baseline=baseline, mode=baseline, run_id=run_id, status="FAIL", success=False,
        prefix_tokens=256, tp_size=2, output_tokens=17, profile=PROFILE,
        fault_scope="controlled KV loss plus original instance stop/restart",
        physical_gpu_failure=False, physical_node_failure=False, node_rto_ms=None,
        algorithm_comparison_eligible=False, asymmetric_diagnostics=True,
        isolated_performance_eligible=False, gpu_restore_verified=False)
    rss = None
    try:
        with LocalTrial(config, root, run_id, deadline).owned() as trial, RSSSampler(interval=.2, include_descendants=True) as rss:
            root = trial.root
            reference = root/(run_id+"-capture")
            target_run = root/(run_id+"-restore")
            common = ["--model", config["model"], "--gpus", config["gpus"], "--port", str(PORT),
                "--mem-fraction-static", str(config.get("mem_fraction_static", .6)),
                "--gpu-admission-mode", config.get("gpu_admission_mode", "shared")]
            trial.start("origin", MODULE, "capture", *common, "--output", str(reference),
                "--bundle", str(root/"capture"), "--template-checkpoint", config["template"],
                "--input-ids", config["input_ids"], "--bootstrap-prefix", "256", "--timeout", "120")
            url = f"http://127.0.0.1:{PORT}"
            wait_health(url, root/"origin", trial.budget(config.get("startup_timeout", 900)))
            trial.run(MODULE, "submit", "--output", reference, "--url", url,
                "--wait-ready", "1", "--timeout", config.get("request_timeout", 300))
            trial.run(MODULE, "reference", "--output", reference)
            cfg, state, refs, _ = cold.read_evidence(reference, "capture")
            expected = refs[0]["output_token_ids"]
            if state["origin_input_ids"] != validate_config(config):
                raise ValueError("fresh reference input changed from exact configured trace")
            if (Path(config["ledger"])/(state["restore_id"]+".claimed.json")).exists():
                raise ValueError("fresh capture restore ID already claimed; no replay")
            record.update(restore_id=state["restore_id"], input_ids_sha256=state["input_ids_sha256"],
                          reference_output_ids=expected, runtime_identity=state["runtime_identity"],
                          debug_profile=dict(server_argv=cfg["argv"], capture="mandatory cold CUDA/logits diagnostics"))
            # Protection owns independent CPU bytes from the validated sealed
            # snapshot while the source is still alive, before the fault clock.
            prepared = prepare_owned_capture(root/"capture", baseline)
            record["memory_ledger"] = {k:prepared[k] for k in ("source_rank_bytes", "stable_replica_bytes", "stable_local_parity_bytes", "protection_prepare_ms")}
            record["protection_prepare"] = {k:prepared[k] for k in (
                "protection_prepare_start_ns", "protection_prepare_complete_ns", "protection_prepare_ms")}
            fault = time.monotonic_ns()  # identical boundary: before exact-owned stop
            if prepared['protection_prepare_complete_ns'] > fault:
                raise ValueError('protection must be committed before the fault boundary')
            trial.stop("origin")
            require_origins_stopped(sealed_origin_processes(load_checkpoint(root/"capture")[2]))
            record["erasure"], record["reconstruction"] = erase_and_rebuild(prepared, root/"capture", root/"restored", baseline)
            del prepared
            save(root/"fault-proof.json", dict(erasure=record["erasure"], reconstruction=record["reconstruction"]))
            if baseline == "reprefill":
                target_module = "experiments.kv_ec.reprefill_server"
                target_args = ["launch", *common, "--tp", "2", "--prefill-chunk-size", "-1", "--output", str(target_run)]
            else:
                record["reference_relocation"] = reference_view(reference, root/"restored", root/"reference-view")
                target_module = MODULE
                target_args = ["launch", *common, "--bundle", str(root/"restored"),
                    "--reference", str(root/"reference-view"), "--ledger", config["ledger"],
                    "--output", str(target_run), "--timeout", "120"]
            launched = time.monotonic_ns()
            trial.start("target", target_module, *target_args)
            wait_health(url, root/"target", trial.budget(config.get("startup_timeout", 900)))
            ready = time.monotonic_ns()
            if baseline != "reprefill":
                cold_sse_bridge.run("prepare", target_run)
            client = stream_request(url, state["origin_input_ids"], state["request_id"], run_id,
                timeout=trial.budget(config.get("request_timeout", 300)))
            save(root/"client-sse.json", client)
            if client["output_token_ids"] != expected:
                raise ValueError("all 17 outputs must match fresh complete reference")
            if baseline == "reprefill":
                from ...reprefill_server import validate_generation
                validate_generation(client["response"])
                target = read(target_run/"run.json")
                if target["argv"] != cfg["argv"] or target["model_identity"] != cfg["runtime_identity"]:
                    raise ValueError("reprefill debug serving flags/model differ from fresh capture")
            else:
                cold_sse_bridge.run("finish", target_run, root/"client-sse.json")
                # Unmodified strict checker includes 16 Decode forwards, actual
                # installed GPU KV/readback, scheduler mapping, logits and ledger.
                trial.run(MODULE, "check", "--output", target_run)
                verdict = read(target_run/"check.json")
                if not verdict.get("pass_stage_d") or not verdict.get("restored_slots_used_by_decode"):
                    raise ValueError("independent GPU restore/resume acceptance failed")
                record["gpu_restore_verified"] = True
                record["gpu_verdict_sha256"] = sha((target_run/"check.json").read_bytes())
                record["h2d_per_rank"] = [dict(tp_rank=r["tp_rank"], written_bytes=r["stats"]["written_bytes"],
                    h2d_ms=r["stats"]["h2d_ms"], verify_ms=r["stats"]["verify_ms"],
                    scope="existing synchronized restore stats; not GPUDirect") for r in verdict["ranks"]]
            from .ready_timing import observe as observe_ready
            record["ready_observations"] = observe_ready(root, target_run, baseline, client, state["request_id"], expected[0], fault, ready, None if baseline == "reprefill" else verdict)
            validated = time.monotonic_ns()
            timing = build_timing(run_id, "reprefill" if baseline == "reprefill" else "two-level",
                                  fault, launched, ready, client, validated)
            timing["baseline"] = baseline
            timing.update(record["ready_observations"])
            save(root/"timing.json", timing)
            record.update(timing=timing, status="PASS", success=True, functional_correctness_pass=True,
                reference_state_sha256=sha((root/"capture/request_state.json").read_bytes()))
    except BaseException as exc:
        record.update(status="FAIL", success=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if rss is not None:
            record["rss"] = rss.report()
        fault_path = Path(root)/"capture/fault-erasure.json"
        if fault_path.is_file():
            record["erasure"] = read(fault_path)
    Path(root).mkdir(exist_ok=True)
    save(Path(root)/"result.json", record)
    return record


def execute(config, output, run_id, *, rounds=1, max_seconds=7200, smoke=None):
    validate_config(config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}", run_id):
        raise ValueError("independent new run ID (8..64 safe characters) required")
    if rounds not in (1, 5) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 86400:
        raise ValueError("bounded smoke or five-round run required")
    fingerprint = signature(config)
    input_sha256 = sha(Path(config["input_ids"]).read_bytes())
    codes = recovery_code_hashes()
    if rounds == 5:
        proof = read(smoke) if smoke else {}
        if (proof.get("schema") != "ec-gpu-fault-recovery-v1" or proof.get("status") != "PASS"
                or proof.get("rounds") != 1 or proof.get("config_sha256") != fingerprint
                or proof.get("code_sha256") != codes
                or proof.get("input_file_sha256") != input_sha256
                or proof.get("run_id") == run_id or len(proof.get("records", [])) != 3):
            raise ValueError("passing same-config three-baseline smoke required first")
    output = Path(output).resolve()
    if output.name != run_id:
        raise ValueError("output basename must equal independent run ID")
    output.mkdir(parents=True, exist_ok=False)
    # A run ID cannot be reused in another output directory with this ledger.
    save(Path(config["ledger"])/(run_id+".experiment.json"), dict(run_id=run_id, output=str(output)))
    save(output/"config.json", config)
    result = dict(schema="ec-gpu-fault-recovery-v1", status="PARTIAL", run_id=run_id,
        rounds=rounds, config_sha256=fingerprint, input_file_sha256=input_sha256, client_date="2026-11-03", records=[],
        evidence_class="current_smoke" if rounds == 1 else "current_five_round",
        code_sha256=codes, physical_node_rto="BLOCKED: no physical node failure",
        algorithm_comparison_eligible=False, isolated_performance_eligible=False)
    deadline = time.monotonic()+max_seconds
    for repeat in range(rounds):
        for baseline in BASELINES:
            trial_id = f"{run_id}-{baseline}-r{repeat}"
            rec = one_trial(config, output/trial_id, trial_id, baseline, deadline)
            rec["repeat"] = repeat
            result["records"].append(rec)
            if rec["status"] != "PASS":
                save(output/"raw.json", result)
                return result  # no retry/replay/port rotation after failed smoke
    restores = [r["restore_id"] for r in result["records"]]
    if (len(restores) == len(set(restores)) and len(restores) == 3*rounds
            and len({r["input_ids_sha256"] for r in result["records"]}) == 1
            and len({signature(r["runtime_identity"]) for r in result["records"]}) == 1
            and all(r["reference_output_ids"] == result["records"][0]["reference_output_ids"] for r in result["records"])):
        result["status"] = "PASS"
    save(output/"raw.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--rounds", type=int, choices=(1, 5), default=1)
    p.add_argument("--smoke", type=Path)
    p.add_argument("--max-seconds", type=float, default=7200)
    a = p.parse_args()
    def interrupted(signum, _):
        raise InterruptedError(f"E2-A coordinator signal {signum}; exact-owned cleanup follows")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    result = execute(read(a.config), a.output, a.run_id, rounds=a.rounds, max_seconds=a.max_seconds, smoke=a.smoke)
    print(json.dumps(dict(status=result["status"], completed=len(result["records"]), output=str(a.output))))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
