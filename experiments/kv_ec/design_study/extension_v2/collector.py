"""Hash-bound E1–E6 evidence collector. Missing cells stay null/PARTIAL.

An explicit input index binds each raw artifact to its SHA256, experiment,
evidence class and configuration. Historical smoke never becomes a five-round
run by pooling files. No network, package installation, or service operations.
"""
import argparse
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import statistics

from .evidence import code_hashes, save, sha

EXPERIMENTS=("E1","E2A","E2B","E3","E4","E5","E6")
CLASSES=("historical_smoke","historical_five_round","current_smoke","current_five_round","synthetic_fixture")
METRICS=("end_to_end_ms","ttft_ms","fault_to_host_kv_ready_ms","checkpoint_ms","elapsed_ns",
    "useful_bytes_per_s","stable_local_parity_bytes","stable_replica_bytes","stable_remote_parity_bytes",
    "server_prefill_ms","server_decode_ms","h2d_ms","node_rto_ms","eit_ratio")


def numeric(value):
    return value if type(value) in (int,float) and math.isfinite(value) else None


def extract(value, path):
    for part in path.split(".") if path else []:
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value


def read_input(entry, base):
    path=Path(entry["path"])
    if not path.is_absolute():
        path=base/path
    raw=path.read_bytes()
    if sha(raw)!=entry["sha256"]:
        raise ValueError("artifact source hash mismatch: "+str(path))
    if entry["experiment"] not in EXPERIMENTS or entry["evidence_class"] not in CLASSES:
        raise ValueError("explicit known experiment and evidence class required")
    value=json.loads(raw)
    if (value.get("evidence_class")=="synthetic_fixture" or value.get("evidence")=="synthetic_fixture") and entry["evidence_class"]!="synthetic_fixture":
        raise ValueError("fixture cannot be promoted to real evidence")
    declared_rounds=value.get("rounds",value.get("repeats"))
    if entry["evidence_class"].endswith("five_round") and declared_rounds is not None and declared_rounds!=5:
        raise ValueError("smoke artifact cannot be relabeled five-round")
    manifest_path=entry.get("manifest")
    manifest=None
    if manifest_path:
        mp=Path(manifest_path)
        if not mp.is_absolute():
            mp=base/mp
        manifest_raw=mp.read_bytes()
        if sha(manifest_raw)!=entry.get("manifest_sha256"):
            raise ValueError("manifest hash mismatch")
        manifest=json.loads(manifest_raw)
        if manifest.get("raw_sha256")!=entry["sha256"]:
            raise ValueError("manifest does not bind raw artifact")
    records=extract(value,entry.get("records_key","records"))
    if not isinstance(records,list):
        raise ValueError("explicit records array required")
    source=dict(path=str(path.resolve()),sha256=entry["sha256"],schema=value.get("schema"),
        manifest_sha256=entry.get("manifest_sha256"),manifest=manifest,
        code_sha256=value.get("code_sha256") or (manifest or {}).get("code_sha256"),
        evidence_class=entry["evidence_class"],experiment=entry["experiment"],
        config=entry.get("config",{}),source_status=value.get("status"),
        provenance_independently_attested=value.get("provenance_independently_attested",False))
    rows=[]
    for i,record in enumerate(records):
        if not isinstance(record,dict):
            raise ValueError("object record required")
        row=dict(experiment=entry["experiment"],evidence_class=entry["evidence_class"],source_sha256=entry["sha256"],
            source_path=source["path"],record_index=i,config=entry.get("config",{}),
            source_schema=value.get("schema"),repeat=record.get("repeat"),
            mode=record.get("mode",value.get("mode",record.get("operation"))),
            prefix_tokens=record.get("prefix_tokens"),concurrency=record.get("concurrency"),
            fault=record.get("fault"),node=record.get("node"),rank=record.get("rank",record.get("missing_rank")),
            missing_node=record.get("missing_node"),mib=record.get("mib"),probability=record.get("probability"),
            request_id=record.get("request_id"),success=record.get("success"),
            recovery_success=record.get("recovery_success"),expected_unrecoverable=record.get("expected_unrecoverable",False),
            byte_exact=record.get("byte_exact",record.get("recovery",{}).get("byte_exact")),
            timeout=record.get("timeout",False),not_attempted=record.get("not_attempted",False),
            error=record.get("error"),physical_node_failure_verified=False,gpu_failover_verified=False)
        if value.get("schema")=="ec-real-kv-microbench-v1":
            row["success"]=record.get("byte_exact") is True and numeric(record.get("elapsed_ns")) is not None
        if "success_key" in entry:
            verdict=extract(record,entry["success_key"])
            row["success"] = verdict is True or verdict == "PASS"
        for metric in METRICS:
            row[metric]=numeric(record.get(metric))
        ledger=record.get("memory_ledger",{})
        for metric in METRICS:
            if metric.startswith("stable_"):
                row[metric]=numeric(ledger.get(metric))
        for metric,path_expr in entry.get("metrics",{}).items():
            if metric not in METRICS:
                raise ValueError("unknown metric mapping: "+metric)
            try:
                row[metric]=numeric(extract(record,path_expr))
            except (KeyError,IndexError,TypeError):
                row[metric]=None
        rows.append(row)
        # E3 chunk records retain local and remote allocations separately. No
        # naive sum of remote parity repeated on both source-node receipts.
        if entry["experiment"]=="E1":
            for j,chunk in enumerate(record.get("protection",[])):
                m=chunk["manifest"]
                cost=dict(row,experiment="E3",record_index=f"{i}:chunk:{j}",
                    checkpoint_ms=m.get("timing_ms",{}).get("commit"),
                    remote_ledger_key=(chunk.get("global_status") or {}).get("stripe_id"),
                    allocation_ledger_not_rss=True)
                for name in ("stable_local_parity_bytes","stable_replica_bytes","stable_remote_parity_bytes"):
                    cost[name]=numeric(m.get("memory_ledger",{}).get(name))
                cost["global_commit_ms"]=numeric(m.get("timing_ms",{}).get("global_commit"))
                rows.append(cost)
    return source,rows


def group_rows(rows):
    groups=defaultdict(list)
    dimensions=("experiment","evidence_class","config","source_schema","mode","prefix_tokens","concurrency",
        "fault","node","rank","missing_node","mib","probability")
    for row in rows:
        key=json.dumps({k:row.get(k) for k in dimensions},sort_keys=True)
        groups[key].append(row)
    reports=[]
    for key,records in sorted(groups.items()):
        repeats=sorted({r["repeat"] for r in records if type(r["repeat"]) is int})
        result=dict(json.loads(key),records=len(records),repeats_observed=repeats,
            successful_records=sum(r["success"] is True for r in records),
            failed_or_unverified_records=sum(r["success"] is not True for r in records),
            expected_unrecoverable=sum(r["expected_unrecoverable"] is True for r in records),
            recovery_successes=sum(r["recovery_success"] is True for r in records),
            timeout_records=sum(r["timeout"] is True for r in records),
            source_hashes=sorted({r["source_sha256"] for r in records}))
        result["five_round_eligible"]=(result["evidence_class"] in ("current_five_round","historical_five_round")
            and repeats==list(range(5)) and all(r["success"] is True for r in records))
        result["metrics"]={}
        for metric in METRICS:
            measured=[r[metric] for r in records if numeric(r.get(metric)) is not None]
            result["metrics"][metric]=dict(n=len(measured),median=statistics.median(measured) if measured else None,
                minimum=min(measured) if measured else None,maximum=max(measured) if measured else None)
        reports.append(result)
    return reports


def collect(index_path,output):
    index_path=Path(index_path)
    entries=json.loads(index_path.read_text())["inputs"]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows,sources,rejected=[],[],[]
    seen=set()
    for entry in entries:
        try:
            source,added=read_input(entry,index_path.resolve().parent)
            if (source["sha256"],source["experiment"]) in seen:
                raise ValueError("duplicate raw source for same experiment")
            seen.add((source["sha256"],source["experiment"]))
            sources.append(source)
            rows.extend(added)
        except Exception as exc:
            rejected.append(dict(input=entry,error=f"{type(exc).__name__}: {exc}"))
    groups=group_rows(rows)
    coverage={e:dict(status="PARTIAL" if any(r["experiment"]==e for r in rows) else "NOT_RUN",
        records=sum(r["experiment"]==e for r in rows),
        eligible_five_round_groups=sum(g["experiment"]==e and g["five_round_eligible"] for g in groups)) for e in EXPERIMENTS}
    report=dict(schema="ec-e1-e6-report-v2",status="PARTIAL" if rows else "NO_VALID_EVIDENCE",
        coverage=coverage,groups=groups,sources=sources,rejected=rejected,code_sha256=code_hashes(),
        raw_index_sha256=sha(index_path.read_bytes()),gpu_failover_verified=False,
        physical_node_failure_verified=False,node_rto_ms=None,
        limits=["No complete hardcoded prefix/matrix required; this report never declares full E1–E6 completion.",
            "Historical smoke and five-round results are separate evidence classes.",
            "Unknown metrics are null, not zero. Codec/Host-ready/client-SSE/GPU-RTO boundaries are not interchangeable.",
            "Expected-unrecoverable checks are not recovery successes; failed records remain in denominators.",
            "Remote parity cost must be deduplicated by stripe/remote_ledger_key; no automatic cross-node sum."])
    save(output/"report.json",report)
    save(output/"records.json",rows)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ["experiment","success"]
    with (output/"records.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    text=["# E1–E6 Evidence Report", "", f"Status: {report['status']}", "",
        "| Experiment | Status | Records | Verified five-round groups |", "|---|---|---:|---:|"]
    text.extend(f"| {e} | {c['status']} | {c['records']} | {c['eligible_five_round_groups']} |" for e,c in coverage.items())
    text.extend(["", "## Evidence limits", ""]+["- "+s for s in report["limits"]]+["", f"Rejected input artifacts: {len(rejected)}; see report.json."])
    with (output/"report.md").open("x") as stream:
        stream.write("\n".join(text)+"\n")
    maximum=max((c["records"] for c in coverage.values()),default=1) or 1
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="290"><rect width="700" height="290" fill="white"/>',
        '<text x="20" y="24" font-family="sans-serif">Evidence coverage (record counts, NOT performance/completeness)</text>']
    for i,(e,c) in enumerate(coverage.items()):
        y=42+i*32
        svg.append(f'<text x="20" y="{y+17}" font-family="sans-serif">{html.escape(e)}</text><rect x="80" y="{y}" width="{500*c["records"]/maximum}" height="22" fill="#386c9c"/><text x="590" y="{y+17}" font-family="sans-serif">{c["records"]}</text>')
    svg.append('</svg>')
    with (output/"coverage.svg").open("x") as stream:
        stream.write("\n".join(svg))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    value=collect(args.index,args.output)
    print(json.dumps(dict(status=value["status"],coverage=value["coverage"],rejected=len(value["rejected"]))))
    raise SystemExit(1 if value["rejected"] else 0)


if __name__=="__main__":
    main()
