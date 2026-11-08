"""Attach-only compact online benchmark. Never launches/stops any service.

Parent supervises one service lifetime per mode, then uses this runner for all
prefix/concurrency/repeat points. max-running=1 is explicitly queue concurrency,
not true batching. RS peers must execute the identical plan concurrently.
"""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import socket
import struct
import time
from urllib.parse import urlparse

from .workload import measure, validate_trace
from .extension_v2.evidence import RSSSampler, code_hashes, process, save, sha
from ..node_block_export import canonical

MODES = ("noec", "replication", "rs-only", "xor", "xor-rs")


def plan(trace, *, prefixes=(256, 1024, 4096), concurrencies=(1, 4, 8), rounds=5):
    validate_trace(trace)
    if (rounds not in (1, 5) or not prefixes or not concurrencies
            or len(set(prefixes)) != len(prefixes) or len(set(concurrencies)) != len(concurrencies)
            or any(p not in (256, 1024, 4096) for p in prefixes)
            or any(c not in (1, 4, 8) for c in concurrencies)
            or len(trace["requests"]) < max(concurrencies)
            or any(len(r["input_ids"]) < max(prefixes) or r["output_tokens"] != 17 for r in trace["requests"])):
        raise ValueError("same trace, 17 output, sufficient requests/input and supported bounded points required")
    rows = []
    for p in prefixes:
        for c in concurrencies:
            for repeat in range(rounds):
                for row in trace["requests"]:
                    rows.append(dict(request_id=f"{row['request_id']}-p{p}-c{c}-r{repeat}",
                        input_ids=row["input_ids"][:p], output_tokens=17,
                        prefix_tokens=p, concurrency=c, repeat=repeat))
    value = dict(schema="ec-compact-plan-v2", trace_sha256=trace["trace_sha256"],
        model_sha256=trace["model_sha256"], tp_size=2, dtype="bfloat16", rounds=rounds,
        prefixes=list(prefixes), concurrencies=list(concurrencies), rows=rows)
    value["plan_sha256"] = sha(canonical(value))
    return value


def validate_plan(value):
    if (value.get("schema") != "ec-compact-plan-v2"
            or sha(canonical({k:v for k,v in value.items() if k != "plan_sha256"})) != value.get("plan_sha256")):
        raise ValueError("plan hash mismatch")


def option(argv, key):
    return argv[argv.index(key)+1]


def require_smoke(value, mode, path):
    if value["rounds"] == 1:
        return None
    smoke = json.loads(Path(path).read_text()) if path else {}
    if (smoke.get("schema") != "ec-compact-run-v2" or smoke.get("status") != "PASS"
            or smoke.get("rounds") != 1 or smoke.get("mode") != mode
            or smoke.get("trace_sha256") != value["trace_sha256"]
            or smoke.get("code_sha256") != code_hashes() or not smoke.get("records")
            or not all(r.get("success") for r in smoke["records"])):
        raise ValueError("passing same-code/mode/trace smoke required before formal lifecycle")
    return smoke


@contextmanager
def manager_connection(spec, timeout):
    """Bind every handshake/load/release to the inspected Unix peer PID."""
    from ..checkpoint_protocol import recv_message, send_message
    before = process(spec["manager_pid"])["identity"]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(min(timeout, 30))
        conn.connect(spec["manager_socket"])
        pid, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if pid != spec["manager_pid"] or uid != before["uid"]:
            raise ValueError("Unix socket peer is not the inspected manager")
        yield conn
    if process(spec["manager_pid"])["identity"] != before:
        raise ValueError("manager lifecycle changed during socket operation")


def verify_runtime(spec, value):
    """Verify real process flags/env and local listener; labels alone cannot pass."""
    from ..perf_ports import listeners
    mode = spec["mode"]
    if mode not in MODES:
        raise ValueError("unknown mode")
    actual = process(spec["pid"])
    argv, env = actual["argv"], actual["environment"]
    if ("sglang.launch_server" not in argv or option(argv, "--tp-size") != "2"
            or option(argv, "--dtype") != "bfloat16"
            or option(argv, "--kv-cache-dtype") != "auto" or option(argv,"--page-size") != "1"
            or option(argv, "--attention-backend") != "triton"
            or option(argv, "--chunked-prefill-size") != "256"
            or option(argv, "--context-length") != "8192"
            or option(argv, "--random-seed") != "42"
            or option(argv, "--max-running-requests") != "1"
            or "--skip-server-warmup" not in argv
            or json.loads(option(argv, "--cuda-graph-config")) != {"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}
            or "--disable-radix-cache" not in argv or "--disable-overlap-schedule" not in argv
            or env.get("KV_EC_GPU_ADMISSION_MODE") != "shared"
            or env.get("SGLANG_KV_EC_ONLINE") != ("0" if mode == "noec" else "1")
            or any(env.get(k)!="0" for k in ("SGLANG_KV_EC_DEBUG","SGLANG_KV_EC_PAUSE_RESUME",
                "SGLANG_KV_EC_GPU_RESTORE","SGLANG_KV_EC_GPU_RESUME","SGLANG_KV_EC_GPU_COLD_RESUME","SGLANG_KV_EC_RS_RESUME"))
            or (mode != "noec" and env.get("SGLANG_KV_EC_ONLINE_PROTECTION_MODE") != mode)
            or json.loads(env.get("SGLANG_KV_EC_PREPARED_IDENTITY", "{}" )).get("model_sha256") != value["model_sha256"]):
        raise ValueError("runtime activation/TP/precision/trace identity not verified")
    url = urlparse(spec["url"])
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or url.port != int(option(argv, "--port")):
        raise ValueError("attach endpoint must be the inspected local service")
    owners = listeners([url.port])
    if spec.get("owned_process"):
        from ..perf_ports import assert_listener_owned
        owned = json.loads(Path(spec["owned_process"]).read_text())
        if (owned.get("pid") != spec["pid"] or any(owned.get(k) != actual["identity"][k]
                for k in ("pid","uid","start_ticks","boot_id","pgid","session"))):
            raise ValueError("supervisor record differs from inspected serving lifecycle")
        assert_listener_owned(url.port, owned)
    elif not owners or any(not r["owners"] or any(any(p[k] != actual["identity"][k]
            for k in ("pid","start_ticks","uid","boot_id","pgid","session")) for p in r["owners"]) for r in owners):
        raise ValueError("listener not owned by inspected serving process")
    max_running = int(option(argv, "--max-running-requests"))
    actual.update(max_running_requests=max_running,
        concurrency_semantics="queue_concurrency_not_true_batch" if max_running == 1 else "scheduler_limit_not_measured_batch")
    normalized=argv[1:]
    at=normalized.index("--port")
    normalized=normalized[:at]+normalized[at+2:]
    actual["comparison"]=dict(serving_argv=normalized,
        physical_gpus=env.get("CUDA_VISIBLE_DEVICES"),gpu_admission=env.get("KV_EC_GPU_ADMISSION_MODE"))
    if mode != "noec":
        manager = process(spec["manager_pid"])
        service = json.loads((Path(spec["manager_output"])/"service_manifest.json").read_text())
        if (service["run_id"] != env.get("SGLANG_KV_EC_ONLINE_RUN_ID")
                or service["protection_mode"] != mode
                or "experiments.kv_ec.online_chunk_manager" not in manager["argv"]
                or Path(option(manager["argv"], "--output")).resolve() != Path(spec["manager_output"]).resolve()
                or option(manager["argv"], "--run-id") != service["run_id"]
                or spec["manager_socket"] != env.get("SGLANG_KV_EC_ONLINE_SOCKET")
                or option(manager["argv"], "--socket") != env.get("SGLANG_KV_EC_ONLINE_SOCKET")):
            raise ValueError("actual manager/service binding mismatch")
        actual.update(manager=manager, service_manifest=service)
        from ..checkpoint_protocol import recv_message, send_message
        with manager_connection(spec, 5) as conn:
            send_message(conn, dict(kind="hello"), b"")
            ready, raw = recv_message(conn)
        if raw or ready != dict(state="READY", run_id=service["run_id"], protection_mode=mode):
            raise ValueError("manager Unix handshake identity/mode mismatch")
    return actual


def protection(spec, row, *, timeout, release=False):
    if spec["mode"] == "noec":
        return []
    root = Path(spec["manager_output"])
    service = json.loads((root/"service_manifest.json").read_text())
    rid = sha(row["request_id"].encode())
    records = []
    deadline = time.monotonic()+timeout
    for chunk in range(len(row["input_ids"])//256):
        key = sha(f"{service['run_id']}:{rid}:{chunk}".encode())
        path = root/key/"manifest.json"
        while True:
            if path.exists():
                manifest_raw = path.read_bytes()
                manifest = json.loads(manifest_raw)
                receipt_path = root/key/"global_status.json"
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                global_ok = (receipt and receipt.get("global_state") == "GLOBAL_COMMITTED"
                    and receipt.get("remote_outcome") == "COMMITTED" and receipt.get("local_snapshot_id") == key)
                if manifest["state"] == "COMMITTED" and (spec["mode"] not in ("rs-only", "xor-rs") or global_ok):
                    break
                if receipt and receipt.get("global_state") in ("FAILED", "EXPIRED", "CANCELLED", "DROPPED_CAPACITY", "SKIPPED_UNMAPPED"):
                    raise ValueError("global protection failed: " + json.dumps(receipt))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"missing committed protection for {row['request_id']} chunk={chunk}")
            time.sleep(.05)
        if (manifest["protection_mode"] != spec["mode"] or manifest["request_id"] != rid
                or manifest["snapshot_id"] != key or manifest["tp_size"] != 2
                or manifest["dtype"] != "torch.bfloat16" or manifest["token_start"] != chunk*256
                or manifest["token_count"] != 256 or manifest["online_identity"]["manager_epoch"] != service["run_id"]
                or manifest["token_ids_sha256"] != sha(struct.pack("<256q", *row["input_ids"][chunk*256:(chunk+1)*256]))
                or len(manifest["ranks"]) != 2
                or any(r.get("protection_mode") != spec["mode"] or r.get("provenance") != "sglang_request" for r in manifest["ranks"])):
            raise ValueError("chunk request/range/runtime binding mismatch")
        if spec["mode"] in ("rs-only", "xor-rs"):
            from ..online_rs_identity import identity_for, stripe_id
            manager = process(spec["manager_pid"])
            config = json.loads(Path(option(manager["argv"], "--global-rs-config")).read_text())
            identity = identity_for(config["run_id"], config["requests"][rid], config["checkpoint_version"], manifest["ranks"][0])
            if (receipt.get("global_protection") != identity or receipt.get("stripe_id") != stripe_id(identity)
                    or receipt.get("node_id") != config["node_id"] or receipt.get("protection_mode") != spec["mode"]):
                raise ValueError("global receipt not bound to actual exporter/request/chunk identity")
        record = dict(snapshot_id=key, manifest_sha256=sha(manifest_raw), manifest=manifest, global_status=receipt)
        if spec["mode"] == "replication":
            from ..checkpoint_protocol import recv_message, send_message
            checks = []
            for rank in range(2):
                with manager_connection(spec, timeout) as conn:
                    send_message(conn, dict(kind="load_replica", snapshot_id=key, tp_rank=rank, request_id=rid), b"")
                    reply, raw = recv_message(conn)
                if (reply.get("state") != "COMMITTED" or reply.get("snapshot_id") != key
                        or reply.get("tp_rank") != rank or reply.get("request_id") != rid
                        or reply.get("sha256") != sha(raw)
                        or sha(raw) != manifest["ranks"][rank]["sha256"]
                        or len(raw) != manifest["ranks"][rank]["packed_byte_count"]):
                    raise ValueError("live complete DRAM replica load failed")
                checks.append(dict(rank=rank, bytes=len(raw), byte_exact=True))
            record["replica_load_checks"] = checks
            record["xor_recovery_verified"] = False
        if release:
            from ..checkpoint_protocol import recv_message, send_message
            with manager_connection(spec, timeout) as conn:
                send_message(conn,dict(kind="release_protection",snapshot_id=key,request_id=rid),b"")
                reply,_=recv_message(conn)
            if reply.get("state")!="RELEASED" or reply.get("snapshot_id")!=key:
                raise ValueError("explicit post-verification DRAM release failed")
            record["dram_released_after_validation"]=True
        records.append(record)
    return records


def run(spec, value, output, *, reference=None, timeout=120, protection_timeout=180, total_timeout=7200, smoke=None):
    validate_plan(value)
    smoke_proof = require_smoke(value, spec["mode"], smoke)
    if not 0 < timeout <= 900 or not 0 < protection_timeout <= 900 or not 0 < total_timeout <= 14400:
        raise ValueError("finite bounded timeouts required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output/"plan.json", value)
    results, groups = [], []
    report = dict(schema="ec-compact-run-v2", mode=spec["mode"], status="PARTIAL",
        plan_sha256=value["plan_sha256"], trace_sha256=value["trace_sha256"],
        rounds=value["rounds"], evidence_class="current_smoke" if value["rounds"] == 1 else "current_five_round",
        records=results, groups=groups, code_sha256=code_hashes(),
        server_prefill_ms=None, server_decode_ms=None, eit_ratio=None,
        gpu_failover_verified=False, physical_node_failure=False, node_rto_ms=None,
        isolated_performance_eligible=False, shared_resource_environment=True)
    try:
        runtime = verify_runtime(spec, value)
        report["runtime"] = runtime
        if smoke_proof is not None:
            if smoke_proof.get("runtime",{}).get("comparison") != runtime["comparison"]:
                raise ValueError("formal runtime differs from accepted smoke")
            report["smoke_artifact_sha256"] = sha(Path(smoke).read_bytes())
        refs = {}
        if spec["mode"] != "noec":
            if not reference:
                raise ValueError("NoEC reference artifact required")
            raw = Path(reference).read_bytes()
            ref = json.loads(raw)
            if ref.get("mode") != "noec" or ref.get("status") != "PASS" or ref.get("plan_sha256") != value["plan_sha256"]:
                raise ValueError("complete same-plan NoEC reference required")
            if ref.get("runtime",{}).get("comparison") != runtime["comparison"] or ref.get("code_sha256") != report["code_sha256"]:
                raise ValueError("NoEC reference must use identical serving flags, physical GPUs and benchmark source hashes")
            refs = {r["request_id"]: r["output_ids"] for r in ref["records"] if r["success"]}
            if len(refs) != len(value["rows"]):
                raise ValueError("NoEC reference missing/duplicate requests")
            report["reference"] = dict(path=str(Path(reference).resolve()), sha256=sha(raw))
        from ..perf_benchmark import resource_snapshot, command, Sampler
        report["resources_before"] = resource_snapshot()
        report["gpu_processes_before"] = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],timeout=5)
        pids = [spec["pid"]] + ([spec["manager_pid"]] if spec["mode"] != "noec" else [])
        end = time.monotonic()+total_timeout

        def request(row):
            rec = dict(request_id=row["request_id"], repeat=row["repeat"], prefix_tokens=row["prefix_tokens"],
                concurrency=row["concurrency"], success=False, timeout=False)
            try:
                remaining = end-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("lifecycle deadline exceeded")
                rec.update(measure(spec["url"], row, timeout=min(timeout, remaining), expected_output=refs.get(row["request_id"])))
                rec["normal_response_success"] = True
            except Exception as exc:
                rec.update(error=f"{type(exc).__name__}: {exc}", timeout=isinstance(exc, TimeoutError))
            return rec

        with RSSSampler(pids, interval=.2, include_descendants=True) as sampler, Sampler(output/"resources.jsonl", 1.), (output/"requests.jsonl").open("x") as stream:
            for p in value["prefixes"]:
                for c in value["concurrencies"]:
                    for repeat in range(value["rounds"]):
                        rows = [r for r in value["rows"] if (r["prefix_tokens"], r["concurrency"], r["repeat"]) == (p,c,repeat)]
                        begin = time.monotonic_ns()
                        with ThreadPoolExecutor(max_workers=c) as pool:
                            futures = [pool.submit(request, row) for row in rows]
                            current = []
                            for future in as_completed(futures):
                                rec = future.result()
                                current.append(rec)
                                results.append(rec)
                        normal_end = time.monotonic_ns()
                        # Finish the response window before waiting on protection.
                        # Otherwise manager drain occupies client concurrency slots.
                        by_id = {r["request_id"]:r for r in rows}
                        for rec in current:
                            if rec.get("normal_response_success"):
                                try:
                                    rec["protection"] = protection(spec, by_id[rec["request_id"]],
                                        timeout=min(protection_timeout, max(.001,end-time.monotonic())),
                                        release=spec.get("release_after_validation",True))
                                    rec["success"] = True
                                except Exception as exc:
                                    rec.update(error=f"{type(exc).__name__}: {exc}", timeout=isinstance(exc,TimeoutError))
                            stream.write(json.dumps(rec, allow_nan=False)+"\n")
                            stream.flush()
                        finished = time.monotonic_ns()
                        elapsed = (finished-begin)/1e9
                        normal_s = (normal_end-begin)/1e9
                        groups.append(dict(prefix_tokens=p, concurrency=c, repeat=repeat, requests=len(current),
                            completed=sum(r["success"] for r in current), wall_s=elapsed,
                            normal_response_window_s=normal_s, protection_drain_window_s=(finished-normal_end)/1e9,
                            normal_response_output_tokens_per_s=sum(r.get("actual_output_tokens",0) for r in current if r.get("normal_response_success"))/normal_s,
                            output_tokens_per_s=sum(r.get("actual_output_tokens",0) for r in current if r["success"])/elapsed,
                            includes_protection_wait=True, concurrency_semantics=runtime["concurrency_semantics"]))
        report["rss"] = sampler.report()
        report["resources_after"] = resource_snapshot()
        report["gpu_processes_after"] = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],timeout=5)
        after = verify_runtime(spec, value)
        if after["identity"] != runtime["identity"]:
            raise ValueError("service lifecycle changed during run")
        report["status"] = "PASS" if len(results) == len(value["rows"]) and all(r["success"] for r in results) else "PARTIAL"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        seen = {r["request_id"] for r in results}
        results.extend(dict(request_id=r["request_id"], prefix_tokens=r["prefix_tokens"], concurrency=r["concurrency"], repeat=r["repeat"],
            success=False, not_attempted=True, error="run blocked before request") for r in value["rows"] if r["request_id"] not in seen)
        report["requests"] = len(results)
        report["completed"] = sum(r["success"] for r in results)
        save(output/"raw.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--trace", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--prefixes", type=int, nargs="+", default=[256,1024,4096])
    q.add_argument("--concurrencies", type=int, nargs="+", default=[1,4,8])
    q.add_argument("--rounds", type=int, choices=(1,5), default=5)
    q = sub.add_parser("run")
    q.add_argument("--service", type=Path, required=True)
    q.add_argument("--plan", type=Path, required=True)
    q.add_argument("--reference", type=Path)
    q.add_argument("--smoke", type=Path)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--total-timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.op == "plan":
        value = plan(json.loads(args.trace.read_text()), prefixes=args.prefixes, concurrencies=args.concurrencies, rounds=args.rounds)
        save(args.output, value)
        mapping = args.output.with_suffix(".rs-map.json")
        save(mapping, {sha(r["request_id"].encode()):sha(r["request_id"].encode()) for r in value["rows"]})
        print(json.dumps(dict(plan_sha256=value["plan_sha256"], requests=len(value["rows"]),
            rs_request_map=str(mapping))))
    else:
        result = run(json.loads(args.service.read_text()), json.loads(args.plan.read_text()), args.output,
            reference=args.reference, total_timeout=args.total_timeout, smoke=args.smoke)
        print(json.dumps(dict(status=result["status"], completed=result["completed"], requests=result["requests"])))
        raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
