"""One isolated TP4 baseline/pause acceptance run. Never takes over busy GPUs."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from .cross_rank_control import request as control_request
from .request_debug import compare
from .launch_debug import check_port


def resolve_ports(port=None, baseline_port=None, pause_port=None):
    if port is not None and baseline_port is not None and port != baseline_port:
        raise ValueError("--port is a legacy alias for --baseline-port; conflicting values")
    baseline = baseline_port if baseline_port is not None else (port if port is not None else 30359)
    paused = pause_port if pause_port is not None else baseline + 1
    if any(type(value) is not int or not 1 <= value <= 65535 for value in (baseline, paused)):
        raise ValueError("baseline/pause ports must be in 1..65535")
    if baseline == paused:
        raise ValueError("--baseline-port and --pause-port must be different")
    return baseline, paused


def check_evidence(root, identity, *, online=False):
    expected_ranks = [0, 1, 2, 3]
    for rank in expected_ranks:
        record = json.loads((root / f"pause-rank-{rank}.json").read_text())
        for key, expected in (
            ("tp_rank", rank), ("transaction_id", identity["transaction_id"]),
            ("coordinator_epoch", identity["epoch"]), ("state", "RESUMED"),
            ("decode_observed", True), ("decode_forward_mode", "decode"),
            ("resume_authorized", True), ("tp_cpu_resume_vote", True),
            ("allocator_lease_released", True), ("row_protection_released", True),
            ("active_slot_leases_after_release", 0), ("cuda_event_complete", True),
            ("cuda_event_recorded", True),
            ("online_kv_pool_modified", online and rank == 0), ("failure", None),
        ):
            if (key not in record or type(record[key]) is not type(expected)
                    or record[key] != expected):
                raise ValueError(f"rank{rank}:{key}: expected {expected!r}")
        if record["decode_output_count"] < 2 or record["scheduler_skipped_iterations"] < 1:
            raise ValueError(f"rank{rank}: no actual pause/decode observed")
        for key in ("request_generation", "slot_generations"):
            if record[key + "_before"] != record[key + "_after"]:
                raise ValueError(f"rank{rank}:{key} changed")
        status = record["coordinator_final"]
        if (status["epoch"] != identity["epoch"]
                or status["transaction_id"] != identity["transaction_id"]
                or status["state"] != "RESUME_AUTHORIZED"
                or any(status[key] != expected_ranks for key in (
                    "paused_ranks", "validated_ranks", "released_ranks", "armed_ranks"))):
            raise ValueError(f"rank{rank}: incomplete coordinator handshake")
    if list(root.glob("*.error.json")) or list(root.glob("*.pending.json")):
        raise ValueError("rank failure or incomplete decode evidence exists")


def check_shadow_evidence(root, identity):
    """Accept only completed GPU shadow evidence, never claim online recovery."""
    check_evidence(root, identity)
    records = [json.loads((root / f"pause-rank-{rank}.json").read_text()) for rank in range(4)]
    consensus = records[0].get("shadow", {}).get("consensus")
    if not isinstance(consensus, dict):
        raise ValueError("missing_shadow_consensus")
    import numpy as np
    def int_hash(values):
        return hashlib.sha256(np.asarray(values, dtype="<i8")).hexdigest()
    for rank, record in enumerate(records):
        shadow = record.get("shadow", {})
        for key, expected in (
            ("mode", "shadow"), ("tp_rank", rank), ("recovered_rank", 0),
            ("token_start", 0), ("token_count", 256), ("layers", list(range(36))),
            ("source_unchanged", True), ("byte_exact", True),
            ("cuda_event_complete", True), ("used_committed_parity", True),
            ("online_kv_pool_modified", False),
            ("rank0_contribution", "zeros_excluded_from_recovery"),
            ("consensus", consensus),
        ):
            if (type(shadow.get(key)) is not type(expected) or shadow[key] != expected):
                raise ValueError(f"rank{rank}:shadow:{key}")
        if not str(shadow.get("device", "")).startswith("cuda"):
            raise ValueError(f"rank{rank}:shadow_not_gpu")
        slots = record.get("physical_slots_before", [])
        if (len(slots) != 768 or len(set(slots)) != 768 or any(slot <= 0 for slot in slots)
                or slots != record.get("physical_slots_after")
                or shadow.get("physical_slots_sha256") != int_hash(slots[:256])
                or shadow.get("slot_generations_sha256") != int_hash(record["slot_generations_before"][:256])
                or shadow.get("request_row") != record["row"]
                or shadow.get("request_generation") != record["request_generation_before"]):
            raise ValueError(f"rank{rank}:shadow_mapping_or_generation_mismatch")
        if (record["coordinator_final"].get("shadow_ranks") != list(range(4))
                or record["coordinator_final"].get("require_shadow") is not True
                or shadow.get("oracle_checks") != [
                    {"lost_rank": r, "byte_exact": True} for r in range(4)]):
            raise ValueError(f"rank{rank}:shadow_missing_four_rank_interlock_or_oracle")
        if (consensus.get("snapshot_id") != identity["transaction_id"] + "-chunk0"
                or consensus.get("checkpoint_version") != consensus["snapshot_id"]
                or consensus.get("request_hash") != record["request_hash"]
                or consensus.get("request_version") != shadow.get("request_version")
                or consensus.get("input_ids_sha256") != shadow.get("input_ids_sha256")
                or not consensus.get("manager_epoch") or not consensus.get("parity_sha256")
                or consensus.get("byte_exact") is not True
                or consensus.get("online_kv_pool_modified") is not False):
            raise ValueError("shadow_consensus_not_bound_to_transaction")
    for key in ("shadow_h2d_ms", "shadow_verify_ms", "shadow_vs_live_ms",
                "xor_and_cpu_verify_ms", "parity_fetch_ms", "shadow_h2d_cuda_event_ms"):
        value = records[0]["shadow"].get("timing_ms", {}).get(key)
        if type(value) not in (float, int) or not 0 <= value < float("inf"):
            raise ValueError(f"rank0:missing_gpu_recovery_timing:{key}")
    return {str(rank): record["shadow"]["timing_ms"] for rank, record in enumerate(records)}


def check_online_evidence(root, identity, checkpoint, response):
    """M2 requires rank-local GPU execution evidence, not just consensus flags."""
    check_evidence(root, identity, online=True)
    import numpy as np
    records = [json.loads((root / f"pause-rank-{rank}.json").read_text()) for rank in range(4)]
    consensus = records[0]["online"]["consensus"]
    if (consensus.get("snapshot_id") != identity["transaction_id"] + "-chunk0"
            or consensus.get("checkpoint_version") != consensus.get("snapshot_id")
            or any(consensus.get(key) is not True for key in (
                "byte_exact", "checkpoint_committed_before_fault", "used_committed_parity", "rank0_source_excluded"))):
        raise ValueError("M2_unbound_checkpoint_or_missing_consensus")
    if (checkpoint.get("state") != "COMMITTED"
            or checkpoint.get("oracle_checks") != [{"lost_rank": r, "byte_exact": True} for r in range(4)]):
        raise ValueError("M2_checkpoint_not_committed")
    checkpoint_keys = ("snapshot_id", "checkpoint_version", "manager_epoch", "parity_sha256",
                       "request_hash", "request_version", "input_ids_sha256")
    for rank, record in enumerate(records):
        proof, protection = record["online"], record["protection"]
        if (proof.get("mode") != "online" or proof.get("tp_rank") != rank
                or not str(proof.get("device", "")).startswith("cuda")
                or proof.get("recovered_rank") != 0 or proof.get("token_start") != 0
                or proof.get("token_count") != 256 or proof.get("layers") != list(range(36))
                or proof.get("input_ids_sha256") != response["input_ids_sha256"]
                or record["physical_slots_before"] != record["physical_slots_after"]
                or len(set(record["physical_slots_before"])) != 768
                or proof.get("request_generation") != record["request_generation_before"]
                or proof.get("consensus") != consensus
                or proof.get("rollback_used_for_recovery") is not False
                or record["coordinator_final"].get("require_online") is not True
                or record["coordinator_final"].get("shadow_ranks") != list(range(4))):
            raise ValueError(f"M2_rank{rank}_mapping_mode_or_consensus_mismatch")
        for key in ("cuda_event_complete", "checkpoint_committed_before_fault", "used_committed_parity",
                    "rank0_source_excluded", "slot_generations_unchanged", "mapping_unchanged"):
            if proof.get(key) is not True:
                raise ValueError(f"M2_rank{rank}_missing:{key}")
        for key in ("snapshot_id", "checkpoint_version", "manager_epoch", "parity_sha256", "request_version"):
            if (protection.get(key) != consensus.get(key) or checkpoint.get(key) != consensus.get(key)):
                raise ValueError(f"M2_rank{rank}_checkpoint_mismatch:{key}")
        if (protection.get("state") != "COMMITTED"
                or protection.get("stage") != "normal_prefill_before_paused_observation"
                or not protection["committed_monotonic"] <= record["paused_observation_monotonic"]
                or proof["checkpoint"] != {key: consensus[key] for key in checkpoint_keys}
                or protection.get("source_sha256") != proof["original_sha256"]
                or proof["original_sha256"] != checkpoint["rank_hashes"][str(rank)]["sha256"]):
            raise ValueError(f"M2_rank{rank}_protection_order_or_hash_mismatch")
        if rank != 0 and (proof.get("survivor_unchanged") is not True
                          or proof.get("online_kv_pool_modified") is not False
                          or proof.get("fault_injected") is not False):
            raise ValueError(f"M2_rank{rank}_non_target_modified")
        if proof.get("oracle_checks") != [{"lost_rank": r, "byte_exact": True} for r in range(4)]:
            raise ValueError("M2_missing_parity_oracles")
        slots = np.asarray(record["physical_slots_before"][:256], dtype="<i8")
        if proof.get("physical_slots_sha256") != hashlib.sha256(slots).hexdigest():
            raise ValueError("M2_writeback_mapping_hash_mismatch")
    rank0, record0 = records[0]["online"], records[0]
    for key in ("fault_injected", "fault_bytes_changed", "online_kv_pool_modified", "writeback_byte_exact"):
        if rank0.get(key) is not True or consensus.get(key) is not True:
            raise ValueError(f"M2_missing_rank0_execution:{key}")
    if (rank0.get("changed_byte_count", 0) <= 0
            or not record0["paused_observation_monotonic"] < rank0["fault_monotonic"] < rank0["decode_observed_monotonic"]
            or rank0.get("restored_sha256") != rank0["original_sha256"]
            or rank0.get("rollback_sha256") != rank0["original_sha256"]
            or rank0.get("rollback_attempted") or rank0.get("rollback_byte_exact")
            or consensus.get("request_hash") != record0["request_hash"]
            or consensus.get("input_ids_sha256") != response["input_ids_sha256"]):
        raise ValueError("M2_fault_order_restore_hash_or_rollback_mismatch")
    for key in ("protection_total_ms", "fault_injection_and_verify_ms", "survivor_gather_ms", "parity_fetch_ms",
                "xor_and_cpu_verify_ms", "h2d_ms", "kv_writeback_ms", "writeback_gpu_and_hash_verify_ms",
                "fault_to_decode_ms"):
        value = rank0.get("timing_ms", {}).get(key)
        if type(value) not in (float, int) or not 0 <= value < float("inf"):
            raise ValueError(f"M2_missing_timing:{key}")
    return {str(rank): dict(record["online"]["timing_ms"], scheduler_pause_resume_ms=record["pause_ms"])
            for rank, record in enumerate(records)}


def verify_run(root, baseline_response, pause_response, *, online=False):
    """Read-only M1 audit of a completed run; never authorizes a live worker."""
    verdict = {"state": "FAIL", "M1": False, "run": str(root.resolve()), "checks": {}}
    checks = verdict["checks"]
    try:
        identity = json.loads((root / "identity.json").read_text())
        verdict["transaction_id"] = identity["transaction_id"]
        verdict["coordinator_epoch"] = identity["epoch"]
        check_evidence(root / "ranks", identity, **({"online": True} if online else {}))
        checks["rank_resume_decode_generations_and_release"] = "PASS"
        records = [json.loads((root / "ranks" / f"pause-rank-{rank}.json").read_text())
                   for rank in range(4)]
        verdict["pause_ms_by_rank"] = [record["pause_ms"] for record in records]

        paused = authorized = False
        with (root / "coordinator.log").open() as stream:
            for line in stream:
                if "KV_EC_PAUSE_CONTROL " not in line:
                    continue
                event = json.loads(line.split("KV_EC_PAUSE_CONTROL ", 1)[1])
                response = event["response"]
                if response.get("state") != "OK":
                    raise ValueError("coordinator_error_response")
                status = response["status"]
                if any(status.get(key) != identity[key] for key in ("epoch", "transaction_id")):
                    raise ValueError("coordinator_identity_mismatch")
                if status["state"] == "FAILED" or event["op"] == "fail":
                    raise ValueError("coordinator_failed")
                if status["state"] == "PAUSED" and status["paused_ranks"] == list(range(4)):
                    paused = True
                if event["op"] == "authorize":
                    if not paused or status["state"] not in (
                            "RESUME_REQUESTED", "RELEASE_AUTHORIZED", "ALL_RELEASED", "RESUME_AUTHORIZED"):
                        raise ValueError("authorization_without_four_rank_pause")
                    authorized = True
        if not paused or not authorized:
            raise ValueError("missing_four_rank_pause_or_external_authorization")
        checks["four_rank_pause_and_external_authorization"] = "PASS"

        log = root / "pause-server.log"
        if not log.exists():
            log = root / "server.log"
        markers = ("Scheduler hit an exception", "Traceback (most recent call last)",
                   "SlotLeaseRejected", "protected_slot_free", "protected_slot_reuse",
                   "row_protected", "registry_closed")
        with log.open() as stream:
            for line in stream:
                for marker in markers:
                    if marker in line:
                        raise ValueError(f"server_or_allocator_failure:{marker}")
        checks["no_logged_scheduler_or_allocator_failure"] = "PASS"

        if baseline_response.resolve() == pause_response.resolve() or baseline_response.samefile(pause_response):
            raise ValueError("baseline_and_pause_response_must_be_distinct_files")
        baseline, response = [json.loads(path.read_text())
                              for path in (baseline_response, pause_response)]
        request_hash = hashlib.sha256(response["request_id"].encode()).hexdigest()
        if any(record["request_hash"] != request_hash or record["token_count"] != 768
               or len(record["slot_generations_before"]) != 768 for record in records):
            raise ValueError("response_not_bound_to_paused_768_token_request")
        if baseline["input_token_count"] != 768 or response["input_token_count"] != 768:
            raise ValueError("M1_requires_768_token_input")
        checks["response_bound_to_paused_request"] = "PASS"
        checks["output_tokens_identical"] = baseline.get("output_ids") == response.get("output_ids")
        # Retain the existing strict comparison, including sampling semantics.
        with redirect_stdout(io.StringIO()):
            compare([baseline_response, pause_response])
        checks["strict_baseline_match"] = "PASS"
        verdict.update(state="PASS", M1=True, reason="completed M1 evidence verified; no new GPU experiment performed")
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        verdict["reason"] = f"{type(exc).__name__}: {exc}"
    return verdict


def admit_online(evidence, model):
    """Audit existing shadow success without running another shadow job."""
    verdict = verify_run(evidence, evidence / "baseline-response.json", evidence / "pause-response.json")
    if not verdict["M1"]:
        raise ValueError(f"online_requires_verified_shadow_M1:{verdict['reason']}")
    identity = json.loads((evidence / "identity.json").read_text())
    check_shadow_evidence(evidence / "ranks", identity)
    rank0 = json.loads((evidence / "ranks" / "pause-rank-0.json").read_text())
    response = json.loads((evidence / "pause-response.json").read_text())
    if (rank0["shadow"]["input_ids_sha256"] != response["input_ids_sha256"]
            or rank0["model_identity"]["model_identifier"] != str(Path(model).resolve())):
        raise ValueError("online_shadow_admission_model_or_input_mismatch")
    admission = dict(rank0["model_identity"])
    # Experiments-only commits do not change SGLang's actual inference code.
    repo = Path(__file__).resolve().parents[2]
    difference = subprocess.check_output([
        "git", "diff", admission["source_commit"], "HEAD", "--", "python"], cwd=repo)
    if difference:
        raise ValueError("online_shadow_admission_sglang_source_changed")
    admission["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return admission


def stop_owned(process):
    # Every process passed here was created below with start_new_session=True.
    # Signal only that private process group, never a discovered server/PID.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGTERM:
            # The launcher can exit before its worker children.
            continue
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gpus", help="run a new experiment on four distinct physical GPU indices")
    mode.add_argument("--verify-run", type=Path, help="read-only audit of an existing experiment")
    parser.add_argument("--baseline-response", type=Path)
    parser.add_argument("--pause-response", type=Path)
    parser.add_argument("--report", type=Path, help="exclusive new offline audit report; never overwrites evidence")
    parser.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    parser.add_argument("--port", type=int, help="legacy alias for --baseline-port; pause defaults to baseline+1")
    parser.add_argument("--baseline-port", type=int, help="baseline TCP port (default: 30359)")
    parser.add_argument("--pause-port", type=int, help="pause TCP port (default: baseline+1)")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--pause-timeout", type=float, default=60)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--shadow", action="store_true",
                        help="run committed-parity GPU shadow recovery while paused; never live writes")
    recovery.add_argument("--online-restore", action="store_true",
                          help="isolated rank0 chunk0 logical fault and real KV restore; default OFF")
    parser.add_argument("--shadow-evidence", type=Path,
                        help="previous completed shadow run required for --online-restore")
    args = parser.parse_args()
    if args.online_restore and args.shadow_evidence is None:
        parser.error("--online-restore requires --shadow-evidence (no new shadow experiment)")
    if args.shadow_evidence is not None and not args.online_restore:
        parser.error("--shadow-evidence requires --online-restore")
    if args.verify_run is not None:
        verdict = verify_run(
            args.verify_run,
            args.baseline_response or args.verify_run / "baseline-response.json",
            args.pause_response or args.verify_run / "pause-response.json", online=args.online_restore)
        if args.online_restore:
            verdict["M2_PASS"] = False
            if verdict["M1"]:
                try:
                    verdict["recovery_timing_ms_by_rank"] = check_online_evidence(
                        args.verify_run / "ranks", json.loads((args.verify_run / "identity.json").read_text()),
                        json.loads((args.verify_run / "checkpoint.json").read_text()),
                        json.loads((args.pause_response or args.verify_run / "pause-response.json").read_text()))
                    verdict.update(M2_PASS=True, reason="online fault recovery verified from saved GPU evidence")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    verdict.update(state="FAIL", reason=str(exc))
        if args.shadow:
            verdict.update(M2_PASS=False, SHADOW_PASS=False)
            if verdict["M1"]:
                try:
                    identity = json.loads((args.verify_run / "identity.json").read_text())
                    verdict["recovery_timing_ms_by_rank"] = check_shadow_evidence(
                        args.verify_run / "ranks", identity)
                    response = json.loads((args.pause_response or args.verify_run / "pause-response.json").read_text())
                    for rank in range(4):
                        shadow = json.loads((args.verify_run / "ranks" / f"pause-rank-{rank}.json").read_text())["shadow"]
                        if shadow.get("input_ids_sha256") != response["input_ids_sha256"]:
                            raise ValueError("shadow_input_token_hash_mismatch")
                    verdict.update(SHADOW_PASS=True, reason="shadow verified; online recovery NOT executed")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    verdict.update(state="FAIL", reason=str(exc))
        if args.report:
            with args.report.open("x") as stream:
                json.dump(verdict, stream, indent=2)
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["state"] == "PASS" else 1
    if args.baseline_response or args.pause_response or args.report:
        parser.error("--baseline-response, --pause-response and --report require --verify-run")
    root = Path(tempfile.mkdtemp(prefix="kv-ec-pause-accept-"))
    tx = "pause-" + uuid.uuid4().hex
    print(f"RUN={root}\nTX={tx}", flush=True)
    processes, logs = [], []
    process_logs = {}

    def spawn(module, argv, log, env):
        handle = (root / log).open("x")
        logs.append(handle)
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", module, *argv], env=env,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        process_logs[process.pid] = root / log
        return process

    def alive(process, name):
        if process.poll() is not None:
            path = process_logs[process.pid]
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 3000))
                detail = handle.read().decode(errors="replace").strip()
            raise RuntimeError(f"{name} exited ({process.returncode}); log={path}; "
                               f"startup_error={detail or 'empty log'}")

    def wait_for(predicate, timeout, name, watched):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for process, label in watched:
                alive(process, label)
            if predicate():
                return
            time.sleep(0.05)
        raise TimeoutError(name)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def healthy(url):
        try:
            with opener.open(url + "/health", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def client(env, request_id, output, log, url):
        return spawn("experiments.kv_ec.request_debug", [
            "request", "--url", url, "--model", args.model,
            "--request-id", request_id, "--prompt-tokens", "768",
            "--stream-metrics", "--output", str(root / output)], log, env)

    verdict = {"state": "FAIL", "transaction_id": tx, "run": str(root), "M1": False}
    if args.shadow or args.online_restore:
        verdict.update(M2_PASS=False, SHADOW_PASS=False)
    try:
        baseline_port, pause_port = resolve_ports(args.port, args.baseline_port, args.pause_port)
        verdict.update(baseline_port=baseline_port, pause_port=pause_port)
        baseline_url = f"http://127.0.0.1:{baseline_port}"
        pause_url = f"http://127.0.0.1:{pause_port}"
        admission = admit_online(args.shadow_evidence, args.model) if args.online_restore else None
        if args.shadow and os.environ.get("SGLANG_KV_EC_ONLINE_RESTORE", "0") != "0":
            raise ValueError("shadow_mode_forbids_online_restore_environment_flag")
        gpus = args.gpus.split(",")
        if len(gpus) != 4 or len(set(gpus)) != 4 or not all(g.isdigit() for g in gpus):
            raise ValueError("--gpus must select four distinct physical GPU indices")
        listing = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT)
        gpu_ids = dict(line.strip().split(", ") for line in listing.splitlines())
        selected = {gpu_ids[g] for g in gpus}
        tasks = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT)
        if any(line.split(",")[0].strip() in selected for line in tasks.splitlines()):
            raise RuntimeError("selected GPUs have active compute tasks; no processes were stopped")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; GPU validation NOT performed")
        verdict["port_checks"] = {
            "baseline": check_port(baseline_port), "pause": check_port(pause_port)}
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("SGLANG_KV_EC_")}

        def server(label, server_env):
            port = baseline_port if label == "baseline" else pause_port
            url = baseline_url if label == "baseline" else pause_url
            process = spawn("experiments.kv_ec.launch_debug", [
                "--mode", "baseline", "--tp", "4", "--gpus", args.gpus,
                "--port", str(port), "--model", args.model,
                "--output", str(root / "evidence"), "--snapshot-id", tx + "-" + label,
                "--request-id", "kv-ec-pause-request", "--token-count", "256",
                "--layers", "all"], label + "-server.log", server_env)
            wait_for(lambda: healthy(url), args.startup_timeout, label + " health timeout",
                     [(process, label + " server")])
            return process

        baseline = server("baseline", env)
        baseline_client = client(env, "kv-ec-baseline-request", "baseline-response.json",
                                 "baseline-client.log", baseline_url)
        if baseline_client.wait(timeout=600) != 0:
            raise RuntimeError("baseline request failed; inspect baseline-client.log")
        alive(baseline, "baseline server")
        stop_owned(baseline)
        processes.remove(baseline)

        coordinator_args = [
            "serve", "--socket", str(root / "pause.sock"), "--transaction-id", tx,
            "--timeout-s", str(args.pause_timeout),
            "--identity-output", str(root / "identity.json")]
        if args.shadow:
            coordinator_args.append("--require-shadow")
        if args.online_restore:
            coordinator_args.append("--require-online")
        coordinator = spawn("experiments.kv_ec.pause_control", coordinator_args, "coordinator.log", env)
        wait_for(lambda: (root / "pause.sock").exists(), 10, "coordinator startup timeout",
                 [(coordinator, "coordinator")])
        identity = json.loads((root / "identity.json").read_text())
        pause_env = dict(env, SGLANG_KV_EC_PAUSE_RESUME="1",
                         SGLANG_KV_EC_PAUSE_REQUEST_ID="kv-ec-pause-request",
                         SGLANG_KV_EC_PAUSE_IDENTITY=json.dumps(identity),
                         SGLANG_KV_EC_PAUSE_SOCKET=str(root / "pause.sock"),
                         SGLANG_KV_EC_PAUSE_OUTPUT=str(root / "ranks"),
                         SGLANG_KV_EC_PAUSE_TIMEOUT_S=str(args.pause_timeout),
                         SGLANG_KV_EC_PAUSE_IPC_TIMEOUT_S="0.5")
        if args.shadow or args.online_restore:
            from .parity_manager import manager_hello
            manager = spawn("experiments.kv_ec.parity_manager", [
                "--socket", str(root / "parity.sock"), "--timeout", str(args.pause_timeout),
                "--test-oracle"], "parity-manager.log", env)
            wait_for(lambda: (root / "parity.sock").exists(), 10, "parity startup timeout",
                     [(manager, "parity manager")])
            pause_env.update(
                SGLANG_KV_EC_SHADOW_RESTORE="0" if args.online_restore else "1",
                SGLANG_KV_EC_ONLINE_RESTORE="1" if args.online_restore else "0",
                SGLANG_KV_EC_SHADOW_PARITY_SOCKET=str(root / "parity.sock"),
                SGLANG_KV_EC_SHADOW_MANAGER_EPOCH=manager_hello(root / "parity.sock"))
            if args.online_restore:
                pause_env.update(SGLANG_KV_EC_ISOLATED_TX=tx,
                                 SGLANG_KV_EC_ONLINE_ADMISSION=json.dumps(admission))
        paused_server = server("pause", pause_env)
        paused_client = client(env, "kv-ec-pause-request", "pause-response.json", "pause-client.log", pause_url)

        def control(op):
            response = control_request(root / "pause.sock", dict(identity, op=op), timeout=1)
            if response.get("state") != "OK" or response["status"]["state"] == "FAILED":
                raise RuntimeError(f"coordinator {op}: {response}")
            return response["status"]

        def all_paused():
            status = control("status")
            return (status["state"] == "PAUSED" and status["paused_ranks"] == [0, 1, 2, 3]
                    and (not (args.shadow or args.online_restore) or status.get("shadow_ranks") == [0, 1, 2, 3]))

        wait_for(all_paused, args.pause_timeout, "four-rank PAUSED timeout",
                 [(coordinator, "coordinator"), (paused_server, "pause server"),
                  (paused_client, "pause client")])
        # Save the observation before the external authorization.
        with (root / "paused.json").open("x") as handle:
            json.dump(control("status"), handle, indent=2)
        control("authorize")
        if paused_client.wait(timeout=600) != 0:
            raise RuntimeError("pause request failed; inspect pause-client.log")
        wait_for(lambda: all((root / "ranks" / f"pause-rank-{r}.json").exists()
                            for r in range(4)), 10, "decode evidence timeout",
                 [(paused_server, "pause server"), (coordinator, "coordinator")])
        check_evidence(root / "ranks", identity, **({"online": True} if args.online_restore else {}))
        compare([root / "baseline-response.json", root / "pause-response.json"])
        wait_for(lambda: healthy(pause_url), 10, "post-decode health failed", [(paused_server, "pause server")])
        server_log = (root / "pause-server.log").read_text()
        if "Scheduler hit an exception" in server_log or "Traceback (most recent call last)" in server_log:
            raise RuntimeError("server log contains an exception")
        verdict.update(state="PASS", M1=True, reason="TP4 paused, authorized, decoded; exact 16-token baseline match")
        if args.shadow:
            verdict["recovery_timing_ms_by_rank"] = check_shadow_evidence(root / "ranks", identity)
            from .parity_manager import submit_once
            committed = submit_once(root / "parity.sock", {
                "kind": "status", "snapshot_id": tx + "-chunk0"}, b"", timeout=2)
            rank0 = json.loads((root / "ranks" / "pause-rank-0.json").read_text())["shadow"]
            response = json.loads((root / "pause-response.json").read_text())
            if any(json.loads((root / "ranks" / f"pause-rank-{r}.json").read_text())["shadow"].get(
                    "input_ids_sha256") != response["input_ids_sha256"] for r in range(4)):
                raise ValueError("shadow_input_token_hash_mismatch")
            if (committed.get("state") != "COMMITTED"
                    or committed.get("parity_sha256") != rank0["consensus"]["parity_sha256"]
                    or committed.get("manager_epoch") != rank0["consensus"]["manager_epoch"]
                    or committed.get("oracle_checks") != rank0["oracle_checks"]):
                raise ValueError("final_committed_checkpoint_mismatch")
            with (root / "checkpoint.json").open("x") as handle:
                json.dump(committed, handle, indent=2)
            verdict.update(SHADOW_PASS=True, reason="GPU shadow byte-exact; exact baseline match; online recovery NOT executed")
        if args.online_restore:
            audit = verify_run(root, root / "baseline-response.json", root / "pause-response.json", online=True)
            if not audit["M1"]:
                raise ValueError(f"online_M1_audit_failed:{audit['reason']}")
            verdict["checks"] = audit["checks"]
            from .parity_manager import submit_once
            checkpoint = submit_once(root / "parity.sock", {
                "kind": "status", "snapshot_id": tx + "-chunk0"}, b"", timeout=2)
            with (root / "checkpoint.json").open("x") as handle:
                json.dump(checkpoint, handle, indent=2)
            response = json.loads((root / "pause-response.json").read_text())
            verdict["recovery_timing_ms_by_rank"] = check_online_evidence(root / "ranks", identity, checkpoint, response)
            verdict.update(M2_PASS=True, reason="GPU online fault/writeback byte-exact; original decode matches baseline",
                           checkpoint_committed_before_fault=True, fault_injected=True, fault_bytes_changed=True,
                           used_committed_parity=True, rank0_source_excluded=True, online_kv_pool_modified=True,
                           writeback_byte_exact=True, decode_observed=True, output_ids_identical=True)
    except (Exception, KeyboardInterrupt) as exc:
        reason = exc.output.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        verdict.update(state="FAIL", reason=f"{type(exc).__name__}: {reason}")
        if args.shadow or args.online_restore:
            verdict["SHADOW_PASS"] = False
            verdict["M2_PASS"] = False
    finally:
        for process in reversed(processes):
            stop_owned(process)
        for handle in logs:
            handle.close()
        with (root / "acceptance.json").open("x") as handle:
            json.dump(verdict, handle, indent=2)
    print(f"{verdict['state']}: {verdict['reason']}\nEvidence: {root}", flush=True)
    if args.shadow or args.online_restore:
        print(f"SHADOW_PASS={str(verdict['SHADOW_PASS']).lower()}\nM2_PASS={str(verdict['M2_PASS']).lower()}\nreason={verdict['reason']}")
    return 0 if verdict["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
