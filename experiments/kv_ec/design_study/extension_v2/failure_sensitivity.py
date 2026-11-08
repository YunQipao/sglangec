"""Actually execute request-unit Host KV faults at 5/10/15%, not schedules only.

CPU mode tests checkpoint requests, not inference concurrency. Cold serving
adapter reuses the sealed cold checkpoint validator and issues real full-prefill
requests to an already supervised NoEC server. It zeros its owned CPU KV copy;
it does NOT erase live GPU KV or claim that Host-ready means GPU-ready.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np

from .cpu_ablation import MODES, ShardAdapter, Unrecoverable, gpu_idle, load_nodes
from .evidence import RSSSampler, code_hashes, save, sha


def summarize(records):
    selected = [r for r in records if r["selected"]]
    actual = [r for r in selected if r.get("erasure")]
    recovered = [r for r in actual if r.get("recovery_success")]
    times = [r["fault_to_host_kv_ready_ms"] for r in recovered]
    return dict(requests=len(records), scheduled_faults=len(selected), actual_faults=len(actual),
        recovered=len(recovered), completed=sum(r.get("success",False) for r in records),
        request_completion_rate=sum(r.get("success",False) for r in records)/len(records) if records else None,
        recovery_success_rate=len(recovered)/len(actual) if actual else None,
        mttr_success_only_ms=statistics.mean(times) if times else None,
        mttr_all_faults_ms=statistics.mean(times) if actual and len(times)==len(actual) else None,
        timeout_rate=sum(r.get("timeout",False) for r in records)/len(records) if records else None,
        eit_ratio=None, gpu_failover_verified=False, node_rto_ms=None)


def cold_reprefill(url, checkpoint, row, *, selected, expected_output, timeout=120):
    """Minimal real Host-loss -> ordinary cold-prefill adapter, with byte erasure.

    Does not use workload's fault callback or emit a fake GPU-erasure receipt.
    The checkpoint must match the exact input and model; output must match NoEC.
    Caller is responsible for verify_runtime on the supervised NoEC endpoint.
    """
    from ...cold_resume_state import load_checkpoint
    from ...node_block_export import unpack
    from ..workload import measure
    source, state, _, _ = load_checkpoint(checkpoint)
    if state["origin_input_ids"] != row["input_ids"] or row["output_tokens"] != 17:
        raise ValueError("sealed cold checkpoint must match request input exactly")
    meta, ranks = unpack(source.load())
    if row.get("model_sha256", meta["ranks"][0]["model_sha256"]) != meta["ranks"][0]["model_sha256"]:
        raise ValueError("cold model identity mismatch")
    owned = [np.frombuffer(b,dtype=np.uint8).copy() for b in ranks]
    fault_start = time.monotonic_ns() if selected else None
    erasure = []
    if selected:
        for rank,b in enumerate(owned):
            before = sha(b)
            b.fill(0)
            erasure.append(dict(rank=rank, bytes=b.nbytes, before_sha256=before,
                after_sha256=sha(b), zeroed=not np.any(b)))
        owned.clear()
    record = measure(url,row,timeout=timeout,expected_output=expected_output)
    return dict(record, selected=selected, actual_erasure=erasure,
        recovery_route="ordinary_cold_full_prefill" if selected else None,
        fault_scope="owned Host checkpoint bytes only; archived source remains read-only",
        fault_to_first_sse_ms=(record["first_output_ns"]-fault_start)/1e6 if selected else None,
        fault_to_host_kv_ready_ms=None, recovery_success=None,
        gpu_erased=False, gpu_writeback=False, gpu_failover_verified=False,
        checkpoint_source_sha256=sha(source.load()))


def run(paths, output, *, requests=100, rounds=5, seed=11, max_seconds=1800,
        allow_fixture=False, idle_check=None, cpu_host_mode="strict_idle"):
    if type(requests) is not int or not 1 <= requests <= 1000 or rounds not in (1,5) or not 0 < max_seconds <= 7200:
        raise ValueError("bounded request/round/wall-time budget required")
    if idle_check is not None and not allow_fixture:
        raise ValueError("idle bypass test-only")
    check = idle_check or (lambda: gpu_idle(cpu_host_mode))
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    result=dict(schema="ec-request-fault-v2",status="PARTIAL",records=records,rounds=rounds,
        evidence_class="synthetic_fixture" if allow_fixture else ("current_smoke" if rounds==1 else "current_five_round"),
        code_sha256=code_hashes(), cpu_host_mode=cpu_host_mode, isolated_performance_eligible=False,
        probability_semantics="independent Bernoulli per checkpoint-request unit",
        scope="CPU adapter checkpoint requests, not live inference requests; same real captured pair reused per unit",
        gpu_adapter_status="BLOCKED: no authorized live-request GPU erasure + restore/resume binding",
        minimum_serving_adapter="cold_reprefill: verified sealed cold source + real HTTP full-prefill after owned Host bytes erased",
        gpu_failover_verified=False, physical_node_failure=False, node_rto_ms=None, eit_ratio=None)
    try:
        result["gpu_idle_before"]=check()
        nodes,result["sources"]=load_nodes(paths,allow_fixture=allow_fixture)
        deadline=time.monotonic()+max_seconds
        with RSSSampler() as rss,(output/"records.jsonl").open("x") as stream:
            for probability in (.05,.10,.15):
                for repeat in range(rounds):
                    result.setdefault("gpu_idle_checks",[]).append(check())
                    rng=random.Random(seed+repeat)
                    selected=[rng.random()<probability for _ in range(requests)]
                    for mode in MODES:
                        for index,fault in enumerate(selected):
                            rec=dict(mode=mode,probability=probability,repeat=repeat,request_index=index,isolated_performance_eligible=False,
                                request_id=f"cpu-p{probability}-r{repeat}-q{index}",selected=fault,
                                success=False,recovery_success=False,fault_to_host_kv_ready_ms=None)
                            try:
                                if time.monotonic()>=deadline:
                                    raise TimeoutError("fault sensitivity deadline exceeded")
                                adapter=ShardAdapter(nodes,mode,allow_fixture=allow_fixture)
                                if fault:
                                    begin=time.perf_counter_ns()
                                    # Alternate actual rank and node-unavailable faults.
                                    node=index%2
                                    rank=index%adapter.tp if index%2==0 else None
                                    rec["erasure"]=adapter.erase(node,rank)
                                    rec["recovery"]=adapter.recover(node,rank)
                                    rec.update(recovery_success=True,fault_to_host_kv_ready_ms=(time.perf_counter_ns()-begin)/1e6)
                                else:
                                    if any(sha(b)!=adapter.hashes[n][r] for n in (0,1) for r,b in enumerate(adapter.active[n])):
                                        raise ValueError("normal checkpoint request corrupted")
                                rec["success"]=True
                            except Exception as exc:
                                rec.update(error=f"{type(exc).__name__}: {exc}", timeout=isinstance(exc,TimeoutError),
                                    unrecoverable=isinstance(exc,Unrecoverable))
                            finally:
                                if "adapter" in locals():
                                    del adapter
                            records.append(rec)
                            stream.write(json.dumps(rec)+"\n")
                            stream.flush()
        result["rss"]=rss.report()
        result["gpu_idle_after"]=check()
        result["summary"]=[dict(mode=m,probability=p,**summarize([r for r in records if r["mode"]==m and r["probability"]==p])) for m in MODES for p in (.05,.10,.15)]
        # Unrecoverable selected faults remain in every denominator. This is a
        # complete CPU protocol run, not a claim of complete E6 GPU execution.
        result["status"]="CPU_ONLY" if len(records)==3*rounds*len(MODES)*requests and all(r.get("success") or r.get("unrecoverable") for r in records) else "PARTIAL"
    except Exception as exc:
        result["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        save(output/"raw.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes",type=Path,nargs=2,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--requests",type=int,default=100)
    p.add_argument("--rounds",type=int,choices=(1,5),default=5)
    p.add_argument("--max-seconds",type=float,default=1800)
    p.add_argument("--cpu-host-mode",choices=("shared","strict_idle"),default="strict_idle")
    args=p.parse_args()
    value=run(args.nodes,args.output,requests=args.requests,rounds=args.rounds,max_seconds=args.max_seconds,cpu_host_mode=args.cpu_host_mode)
    print(json.dumps(dict(status=value["status"],requests=len(value["records"]),error=value.get("error"))))
    raise SystemExit(0 if value["status"]=="CPU_ONLY" else 1)


if __name__=="__main__":
    main()
