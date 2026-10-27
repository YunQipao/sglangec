"""Stage C: native-RDMA RS bytes -> existing GPU transaction -> native Decode.

``protect`` binds authentic exporter receipts, not reconstruction. ``recover``
has no target-file argument. Original target bytes are read by check-source /
reference acceptance only, never by the runtime RS BytesCheckpointSource.
"""

import argparse
import json
from pathlib import Path

from .gpu_kv_restore import _require
from .node_block_export import sha
from .rs_recovery_source import check_source, load_bundle, recover_node, write_json
from . import verify_gpu_resume as stage_b


def protect(args):
    """Bind authentic exporter receipts for both nodes; never mint commitment."""
    receipts = [json.loads(p.read_text()) for p in (args.node0_receipt, args.node1_receipt)]
    for node, receipt in enumerate(receipts):
        _require(receipt["global_state"] == "GLOBAL_COMMITTED" and
            receipt["node_id"] == node and receipt.get("transport") == "rdma" and
            receipt.get("remote_outcome") == "COMMITTED" and
            receipt.get("recovery_byte_exact") == "NOT_YET_VERIFIED" and
            bool(receipt.get("handle")) and bool(receipt.get("manifest_sha256")) and
            bool(receipt.get("stripe_id")) and bool(receipt.get("global_protection")) and
            bool(receipt.get("node_descriptor")), "stage_c_authentic_global_receipt_required")
    first = receipts[0]
    for key in ("stripe_id", "handle", "manifest_sha256", "global_protection", "parity_sha256"):
        _require(first.get(key) == receipts[1].get(key), "stage_c_receipts_do_not_match")
    _require(receipts[args.target_node]["node_descriptor"]["metadata"]["node_id"] == args.target_node,
             "stage_c_target_receipt_node_mismatch")
    receipt = dict(first, node_id=args.target_node,
                   node_descriptor=receipts[args.target_node]["node_descriptor"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "receipt.json", receipt)
    return receipt


def accepted_source(bundle):
    source, report, meta, identity = load_bundle(bundle)
    check_path = Path(bundle) / "source-check.json"
    raw = check_path.read_bytes()
    accepted = json.loads(raw)
    _require(accepted["pass_source"] is True and accepted["bundle"] == identity and
             accepted["original_sha256"] == accepted["recovered_sha256"] ==
                 identity["block_sha256"] and accepted["real_rdma"] is True and
             accepted["rs_recovery_byte_exact"] is True and
             accepted["runtime_original_local_file_used"] is False,
             "stage_c_independent_source_acceptance_required")
    return source, report, meta, identity, sha(raw)


def launch(args):
    _, _, _, identity, accepted_hash = accepted_source(args.bundle)
    args.checkpoint = Path(identity["path"]) / "recovered_node.bin"
    args.mode = "resume"  # Preserve the proven B state machine and checker.
    _require(args.decode_steps == 16, "stage_c_requires_exactly_16_decode_steps")
    stage_b.launch(args, extra_config=dict(experiment_mode="rs-resume",
        rs_source=identity, source_acceptance_sha256=accepted_hash),
        extra_env=dict(SGLANG_KV_EC_RS_RESUME="1"))


def check_results(output, reference=None):
    output = Path(output)
    config = json.loads((output / "run.json").read_text())
    _require(config.get("experiment_mode") == "rs-resume" and
             config["mode"] == "resume" and config["decode_steps"] == 16,
             "stage_c_rs_resume_run_required")
    source, report, _, identity, accepted_hash = accepted_source(config["rs_source"]["path"])
    _require(identity == config["rs_source"] and
             accepted_hash == config["source_acceptance_sha256"] and
             Path(config["checkpoint"]).resolve() == Path(identity["path"]) / "recovered_node.bin" and
             config["checkpoint_sha256"] == identity["block_sha256"],
             "stage_c_runtime_source_binding_changed")
    # Reread original only here in the independent checker, not in the hook.
    accepted = json.loads((Path(identity["path"]) / "source-check.json").read_text())
    _require(check_source(identity["path"], accepted["original_reference"]) == accepted,
             "stage_c_independent_reference_changed")
    result = stage_b.check_results(output, reference)
    for rank in result["ranks"]:
        _require(rank.get("experiment_mode") == "rs-resume" and
                 rank.get("rs_source") == identity and
                 rank.get("source_kind") == "native-rdma-rs-immutable-bytes" and
                 rank.get("runtime_original_local_file_used") is False,
                 "stage_c_rank_source_binding_invalid")
    committed = all(r["transaction_state"] == "COMMITTED" and r["committed"] is True
                    for r in result["ranks"])
    passed = (result["pass_stage_b"] and committed and result["cuda_executed"] and
        result["restored_slots_used_by_decode"] and result["resume_decode_forward_count"] == 16 and
        result["recomputed_prefix_tokens"] == 0 and result["output_token_ids_match"] and
        result["no_slot_leak"])
    receipt = report["receipt"]
    result.update(schema="kv-rs-resume-stage-c-v1", pass_stage_c=bool(passed),
        experiment_mode="rs-resume", real_rdma=True, rs_recovery_byte_exact=True,
        gpu_restore_committed=committed, rs_source=identity,
        stripe_id=receipt["stripe_id"], handle=receipt["handle"],
        manifest_sha256=receipt["manifest_sha256"], target_node_id=receipt["node_id"],
        recovered_bytes=len(source.load()), recovered_sha256=report["recovered_sha256"],
        runtime_original_local_file_used=False, rdma_transfer=report["transfer"],
        failure_domain="controlled single Node Block loss; same-instance request continuation",
        whole_node_failover=False)
    return result


def rdma_options(parser, *, returning=False):
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=30700)
    parser.add_argument("--rdma-device", default="mlx5_0")
    parser.add_argument("--rdma-gid-index", type=int, required=True)
    parser.add_argument("--rdma-port-num", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    if returning:
        parser.add_argument("--rdma-return-host", required=True)
        parser.add_argument("--rdma-return-port", type=int, default=30890)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("protect", help="bind two authentic GLOBAL_COMMITTED receipts")
    setup.add_argument("--node0-receipt", type=Path, required=True)
    setup.add_argument("--node1-receipt", type=Path, required=True)
    setup.add_argument("--target-node", type=int, choices=(0, 1), required=True)
    setup.add_argument("--output", type=Path, required=True)
    restore = sub.add_parser("recover", help="only survivor + committed receipt, no local target")
    rdma_options(restore, returning=True)
    restore.add_argument("--receipt", type=Path, required=True)
    restore.add_argument("--survivor", type=Path, required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--bundle", type=Path, required=True)
    source_check = sub.add_parser("check-source")
    source_check.add_argument("--bundle", type=Path, required=True)
    source_check.add_argument("--original", type=Path, required=True)
    start = sub.add_parser("launch")
    start.add_argument("--bundle", type=Path, required=True)
    start.add_argument("--reference", type=Path, required=True)
    start.add_argument("--model", required=True)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--decode-steps", type=int, default=16)
    start.add_argument("--gpus", default="auto")
    start.add_argument("--port", type=int, default=30982)
    start.add_argument("--mem-fraction-static", type=float, default=0.6)
    start.add_argument("--timeout", type=float, default=120)
    send = sub.add_parser("submit")
    send.add_argument("--output", type=Path, required=True)
    send.add_argument("--input-ids", type=Path, required=True)
    send.add_argument("--url", default="http://127.0.0.1:30982")
    send.add_argument("--wait-ready", type=float, default=600)
    send.add_argument("--timeout", type=float, default=300)
    check = sub.add_parser("check")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.command == "protect":
        result = protect(args)
    elif args.command == "recover":
        result = recover_node(args)
    elif args.command == "inspect":
        _, report, meta, identity = load_bundle(args.bundle)
        result = dict(bundle=identity, target_node_id=meta["node_id"],
                      receipt=report["receipt"], recovered_bytes=report["recovered_bytes"])
    elif args.command == "check-source":
        result = check_source(args.bundle, args.original)
        write_json(args.bundle / "source-check.json", result)
    elif args.command == "launch":
        launch(args)
        return
    elif args.command == "submit":
        stage_b.submit(args)
        return
    else:
        try:
            result = check_results(args.output, args.reference)
        except Exception as exc:
            result = dict(pass_stage_c=False, failure_reason=f"{type(exc).__name__}:{exc}")
        stage_b.save_verdict(args.output, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result.get("pass_stage_c") is True else 1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
