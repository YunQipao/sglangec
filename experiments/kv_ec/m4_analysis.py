"""Read-only M3 evidence audit and non-overlapping M4 performance inventory."""

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from .accept_pause import check_online_evidence, verify_run
from .m3_performance import online_record


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def same_number(actual, expected):
    return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-5))


def audit_case(root):
    root = Path(root).resolve()
    acceptance = load(root / "acceptance.json")
    identity = load(root / "identity.json")
    checkpoints = load(root / "checkpoints.json")
    performance = load(root / "performance-online.json")
    baseline, response = [load(root / name) for name in ("baseline-response.json", "pause-response.json")]
    require(acceptance.get("state") == "PASS" and acceptance.get("M3_PASS") is True,
            "M3_acceptance_not_pass")
    audit = verify_run(root, root / "baseline-response.json", root / "pause-response.json", online=True)
    require(audit["M1"], f"M1_audit_failed:{audit.get('reason')}")
    check_online_evidence(root / "ranks", identity, checkpoints, response)
    rank, tokens = identity["lost_rank"], identity["recover_tokens"]
    require((acceptance.get("lost_rank"), acceptance.get("recover_tokens")) == (rank, tokens),
            "acceptance_identity_mismatch")
    records = [load(root / "ranks" / f"pause-rank-{r}.json") for r in range(4)]
    target = records[rank]
    require(acceptance.get("chunks") == target["online"]["chunks"], "acceptance_chunk_mismatch")
    config = performance["config"]
    require((config.get("lost_rank"), config.get("recover_tokens")) == (rank, tokens),
            "performance_plan_mismatch")
    require(config.get("input_ids_sha256") == baseline["input_ids_sha256"] == response["input_ids_sha256"],
            "performance_input_mismatch")
    require(config.get("sampling") == baseline["sampling"] == response["sampling"], "sampling_mismatch")
    require(config.get("model_identifier") == baseline["model_identifier"] == response["model_identifier"],
            "model_mismatch")
    require(config.get("input_tokens") == 768 and config.get("output_tokens") == 16
            and config.get("tp") == 4 and config.get("dtype") == "bf16"
            and config.get("layout") == "nhd" and config.get("page_size") == 1
            and config.get("overlap") is False and config.get("prefix_cache") is False,
            "unsupported_performance_config")
    uuids = config.get("gpu_uuids", [])
    require(len(uuids) == len(set(uuids)) == 4 and all(str(u).startswith("GPU-") for u in uuids),
            "missing_real_gpu_uuids")
    expected = online_record(target, response, config=config)
    for key, value in expected.items():
        if type(value) in (float, int):
            require(same_number(performance.get(key), value), f"performance_value_mismatch:{key}")
        else:
            require(performance.get(key) == value, f"performance_record_mismatch:{key}")
    # Positive execution corroboration, not just a device string in a JSON.
    for name in ("baseline-server.log", "pause-server.log"):
        log = (root / name).read_text(errors="replace")
        require(all(f"TP{r}]" in log for r in range(4)) and "Load weight end." in log
                and "Qwen3ForCausalLM" in log and '"POST /generate HTTP/1.1" 200' in log,
                f"missing_real_TP4_execution:{name}")
        require("mock" not in log.lower(), f"mock_marker_in_server_log:{name}")
    for record in records:
        require(all(record["online"]["chunks"][i]["checkpoint"]["snapshot_id"] ==
                    checkpoints[i]["snapshot_id"] for i in range(tokens // 256)),
                "rank_checkpoint_mismatch")
        require(record.get("runtime_checks"), "missing_runtime_safety_checks")
        require(all(check.get("actual") == check.get("expected")
                    and type(check.get("actual")) is type(check.get("expected"))
                    for check in record["runtime_checks"].values()), "failed_runtime_safety_check")
        require(record["model_identity"] == {k: config[k] for k in record["model_identity"]},
                "runtime_software_identity_mismatch")
    # Inspect only filenames/content types, never export or print raw KV.
    files = sorted(p for p in root.rglob("*") if p.is_file())
    require(all(p.suffix in (".json", ".log") for p in files), "unexpected_binary_artifact")
    require(not any("payload" in p.name or p.suffix in (".npy", ".npz", ".bin", ".pt") for p in files),
            "raw_KV_artifact_present")
    intervals = []
    for record in records:
        for chunk in record["protection"]["chunks"]:
            end = chunk["committed_monotonic"]
            intervals.append((end - chunk["timing_ms"]["protection_total_ms"] / 1000, end))
    chunks = target["online"]["chunks"]
    traces = target["trace"]
    verified = [t for t in traces if t.get("operation") == "shadow_verified"]
    armed = [t for t in traces if t.get("operation") == "armed"]
    row = dict(
        lost_rank=rank, recover_tokens=tokens, n=1, run=str(root), transaction_id=identity["transaction_id"],
        target_protection_sum_ms=performance["protection_overhead_ms"],
        protection_all_rank_span_ms=(max(e for _, e in intervals) - min(s for s, _ in intervals)) * 1000,
        capture_d2h_hash_verify_sum_ms=sum(c["timing_ms"]["capture_d2h_and_hash_ms"]
                                         for c in target["protection"]["chunks"]),
        checkpoint_submit_sum_ms=sum(c["timing_ms"]["checkpoint_submit_ms"]
                                     for c in target["protection"]["chunks"]),
        parity_commit_sum_ms=performance["parity_commit_ms"],
        parity_encode_sum_ms=sum(c["xor_ms"] for c in checkpoints),
        parity_oracle_and_metadata_sum_ms=sum(c["oracle_validation_ms"] for c in checkpoints),
        fault_to_decode_ms=performance["fault_to_decode_ms"], recovery_chunk_sum_ms=performance["recovery_ms"],
        xor_and_cpu_verify_sum_ms=performance["cpu_xor_ms"], gather_sum_ms=performance["gather_ms"],
        parity_fetch_sum_ms=sum(c["timing_ms"]["parity_fetch_ms"] for c in chunks),
        h2d_sum_ms=performance["h2d_ms"],
        h2d_cuda_event_sum_ms=sum(c["timing_ms"]["h2d_cuda_event_ms"] for c in chunks),
        kv_writeback_sum_ms=performance["kv_writeback_ms"], byte_validation_sum_ms=performance["validation_ms"],
        fault_injection_verify_sum_ms=sum(c["timing_ms"]["fault_injection_and_verify_ms"] for c in chunks),
        baseline_ttft_ms=baseline["client_metrics_ms"]["ttft_ms"],
        baseline_tpot_ms=baseline["client_metrics_ms"]["tpot_ms"],
        baseline_request_total_ms=baseline["client_metrics_ms"]["request_total_ms"],
        online_ttft_ms=performance["ttft_ms"], online_tpot_ms=performance["tpot_ms"],
        online_request_total_ms=performance["request_total_ms"],
        scheduler_pause_max_ms=max(r["pause_ms"] for r in records),
        # This span includes external authorization/polling, row/slot release,
        # readiness votes and decode. It is NOT pure coordinator RPC time.
        verified_to_decode_span_ms=(target["decode_observed_monotonic"] -
                                    verified[0]["monotonic"]) * 1000 if verified else None,
        armed_to_decode_span_ms=(target["decode_observed_monotonic"] -
                                 armed[0]["monotonic"]) * 1000 if armed else None,
        target_control_ipc_sum_ms=sum(t.get("elapsed_ms", 0) for t in traces if "operation" in t),
    )
    evidence = dict(state="PASS", lost_rank=rank, recover_tokens=tokens, run=str(root),
                    transaction_id=identity["transaction_id"], baseline_path=str(root / "baseline-response.json"),
                    checks=dict(M3=True, independent_baseline=True, rank_local_GPU_execution=True,
                                all_four_ranks_authorized=True, byte_exact=True, output_tokens_identical=True,
                                timing_values_recomputed=True, no_raw_KV_artifacts=True),
                    privacy_scope="local artifacts and loopback/Unix IPC source audit; "
                                  "not a historical network-egress attestation",
                    config=config,
                    file_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files if p.suffix == ".json"},
                    chunks=[dict(token_start=c["token_start"], checkpoint=c["checkpoint"],
                                 recovery_timing_ms=c["timing_ms"],
                                 protection_timing_ms=target["protection"]["chunks"][i]["timing_ms"])
                            for i, c in enumerate(chunks)])
    return row, evidence


def audit_matrix(root):
    root = Path(root).resolve()
    matrix = load(root / "matrix.json")
    require(matrix.get("M3_PASS") is True and matrix.get("state") == "PASS"
            and matrix.get("cpu_returncode") == 0, "matrix_not_pass")
    require(len(matrix.get("cases", [])) == 12, "matrix_requires_12_cases")
    rows, audits, keys, txs, baselines, configs = [], [], set(), set(), set(), set()
    for case in matrix["cases"]:
        key = (case["lost_rank"], case["recover_tokens"])
        require(key not in keys and case["state"] == "PASS" and case["returncode"] == 0,
                "duplicate_or_failed_matrix_case")
        path = root / f"rank{key[0]}-tokens{key[1]}"
        require(Path(case["run"]).resolve() == path, "matrix_run_path_mismatch")
        row, audit = audit_case(path)
        require((row["lost_rank"], row["recover_tokens"]) == key, "matrix_case_plan_mismatch")
        require(case.get("chunks") == load(path / "acceptance.json")["chunks"], "matrix_chunk_proof_mismatch")
        baseline = (path / "baseline-response.json").stat()
        inode = (baseline.st_dev, baseline.st_ino)
        require(inode not in baselines and row["transaction_id"] not in txs,
                "shared_baseline_inode_or_transaction")
        config = {k: v for k, v in audit["config"].items() if k not in ("lost_rank", "recover_tokens")}
        configs.add(json.dumps(config, sort_keys=True))
        keys.add(key)
        baselines.add(inode)
        txs.add(row["transaction_id"])
        rows.append(row)
        audits.append(audit)
    require(keys == {(r, t) for r in range(4) for t in (256, 512, 768)}, "incomplete_matrix")
    require(len(configs) == 1, "mixed_matrix_software_hardware_or_input_config")
    return dict(state="PASS", matrix=str(root), case_count=12, independent_baselines=12,
                sampling_note="one independent run per configuration; no Median/P95 inference",
                cases=audits, rows=sorted(rows, key=lambda r: (r["lost_rank"], r["recover_tokens"])))


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = report["rows"]
    with (output / "performance.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = [
        ("Rank", "lost_rank"), ("Tokens", "recover_tokens"),
        ("Protect target", "target_protection_sum_ms"), ("Protect span", "protection_all_rank_span_ms"),
        ("Parity commit", "parity_commit_sum_ms"), ("Fault->decode", "fault_to_decode_ms"),
        ("XOR+CPU verify", "xor_and_cpu_verify_sum_ms"), ("Gather", "gather_sum_ms"),
        ("H2D", "h2d_sum_ms"), ("KV write", "kv_writeback_sum_ms"),
        ("Byte verify", "byte_validation_sum_ms"),
    ]
    lines = ["# M3 GPU Performance Inventory", "", f"Evidence: `{report['matrix']}`", "",
             "**All values are single observations in milliseconds, n=1 per configuration.**",
             "No stable Median/P95 or causal rank-performance comparison is claimed.", "",
             "| " + " | ".join(name for name, _ in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) if k in ("lost_rank", "recover_tokens")
                                       else f"{row[k]:.2f}" for _, k in columns) + " |")
    lines += ["", "## Interpretation", "",
              "- Protect target is the sum of this target's per-chunk protection wall durations.",
              "- Protect span is earliest protection start to latest commit observation across four ranks.",
              "- Parity commit is manager first accepted submission to commit; it overlaps protection.",
              "- XOR+CPU verify includes hashes, version checks and recovered payload validation, not pure XOR.",
              "- Byte verify includes temporary GPU byte readback and final live/rollback/hash checks.",
              "- H2D CUDA-event measurements are nested in H2D wall time; do not add both.",
              "- Chunk recovery sums exclude inter-chunk preparation/votes; fault->decode includes them.",
              "- Do not add all table columns: enclosing intervals and parallel rank activity overlap.",
              "- Checkpoint oracle validation is enabled; this is a correctness-heavy baseline, not production EC.",
              "- A separate B (protection-only) trial and D (real prefill retry) are absent from these records.",
              "- No EC-versus-reprefill speedup can be calculated from ordinary baseline TTFT.",
              "", "See `performance.csv` for paired client metrics, parity encoding, IPC and per-stage spans.",
              "See `audit.json` for per-chunk timings, configuration and evidence file hashes.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with redirect_stdout(io.StringIO()):
        report = audit_matrix(args.matrix)
    write_report(report, args.output)
    print(f"M3_AUDIT_PASS=true cases=12 independent_baselines=12\nReport: {args.output}")


if __name__ == "__main__":
    main()
