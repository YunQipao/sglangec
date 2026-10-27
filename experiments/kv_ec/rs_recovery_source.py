"""Stage C sealed native-RDMA reconstruction, never a local-target fallback.

Recovery runs outside the scheduler. Both TP processes later load the same
hash-bound artifact into immutable bytes. TCP is allowed for control ONLY.
Reports are experimental provenance records, not signed remote attestations.
"""

import json
from pathlib import Path

from .gpu_kv_restore import BytesCheckpointSource, _require
from .gpu_restore_hook import validate_checkpoint
from .node_block_export import descriptor, sha
from .online_rs_identity import stripe_id
from .remote_rs_client import RemoteRSClient
from .rs_stripe_builder import manifest_sha, validate_manifest


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def native_stats(stats, size):
    """Require actual helper CQ evidence; permissive parser defaults don't count."""
    from .multinode.rdma_transport import RDMATransport
    parsed = RDMATransport._parse_stats(stats.get("helper_stdout", ""), 0)
    _require(stats.get("transport") == "rdma" and
             stats.get("completion_status") == "SUCCESS" and
             stats.get("bytes") == size and parsed.get("bytes") == size and
             parsed.get("completion_status") == "SUCCESS" and
             "KV_EC_RDMA_STATS " in stats.get("helper_stdout", "") and
             all(type(parsed.get(k)) is int and parsed[k] > 0 and
                 stats.get(k) == parsed[k] for k in ("qp_num", "remote_qpn")),
             "stage_c_native_verbs_evidence_missing")


def committed_status(status, receipt):
    manifest = status["manifest"]
    validate_manifest(manifest)  # no fixture option exists in this data source
    _require(receipt["global_state"] == "GLOBAL_COMMITTED" and
             status["state"] == "COMMITTED" and
             all(status[k] == receipt[k] for k in
                 ("stripe_id", "handle", "manifest_sha256", "parity_sha256")) and
             bool(status["handle"]) and
             manifest["stripe_id"] == status["stripe_id"] ==
                 stripe_id(receipt["global_protection"]) and
             manifest_sha(manifest) == status["manifest_sha256"],
             "stage_c_global_commit_binding_invalid")
    node = receipt["node_id"]
    _require(type(node) is int and node in (0, 1) and
             receipt["node_descriptor"] == manifest["nodes"][node],
             "stage_c_target_descriptor_mismatch")
    for d in manifest["nodes"]:
        for rank in d["metadata"]["ranks"]:
            _require(rank.get("global_protection") == receipt["global_protection"],
                     "stage_c_online_identity_mismatch")
    return manifest


def validate_recovery(report, block):
    """Recheck identities, all native legs and exact rank bytes on every load."""
    _require(report["schema"] == "kv-rs-gpu-source-v1" and
             report["runtime_original_local_file_used"] is False and
             report["recovery_input"] == "survivor_plus_remote_committed_parity" and
             report["real_rdma"] is True and report["rs_recovery_byte_exact"] is True,
             "stage_c_source_provenance_invalid")
    receipt = report["receipt"]
    before = committed_status(report["status_before"], receipt)
    after = committed_status(report["status_after"], receipt)
    _require(before == after, "stage_c_manifest_changed_during_recovery")
    target, survivor = before["nodes"][receipt["node_id"]], before["nodes"][1 - receipt["node_id"]]
    meta = validate_checkpoint(block)
    _require(descriptor(block) == target and meta["node_id"] == receipt["node_id"] and
             report["recovered_sha256"] == sha(block) == target["sha256"] and
             report["recovered_bytes"] == len(block) == target["length"] and
             report["survivor_sha256"] == survivor["sha256"],
             "stage_c_recovered_node_identity_mismatch")
    transfer = report["transfer"]
    key = "recovery_" + str(receipt["node_id"])
    remote = report["status_after"]["rdma"][key]
    _require(transfer["state"] == remote["state"] == "COMPLETED" and
             transfer["payload_sha256"] == remote["payload_sha256"] == sha(block) and
             transfer["payload_bytes"] == remote["payload_bytes"] == len(block) and
             transfer["receive_stats"] == remote["receive_stats"] and
             transfer["send_stats"] == remote["send_stats"],
             "stage_c_remote_recovery_incomplete")
    for name, size in (("client_send_stats", survivor["length"]),
                       ("receive_stats", survivor["length"]),
                       ("send_stats", len(block)), ("client_receive_stats", len(block))):
        native_stats(transfer[name], size)
    controls = report["control_transfers"]
    _require(controls and all(t["transport"] == "tcp" and
             t["tx"]["payload_bytes"] == t["rx"]["payload_bytes"] == 0 for t in controls),
             "stage_c_tcp_payload_forbidden")
    _require(any(t["op"] == "prepare_rdma_recovery" for t in controls),
             "stage_c_native_recovery_control_missing")
    return meta


def bundle_identity(directory):
    directory = Path(directory).resolve()
    return dict(path=str(directory), recovery_sha256=sha((directory / "recovery.json").read_bytes()),
                block_sha256=sha((directory / "recovered_node.bin").read_bytes()))


def load_bundle(directory, expected=None):
    """Single reads pin the report and bytes; no TOCTOU reload of the payload."""
    directory = Path(directory).resolve()
    report_raw = (directory / "recovery.json").read_bytes()
    block = (directory / "recovered_node.bin").read_bytes()
    identity = dict(path=str(directory), recovery_sha256=sha(report_raw), block_sha256=sha(block))
    _require(expected is None or identity == expected, "stage_c_recovery_bundle_changed")
    report = json.loads(report_raw)
    meta = validate_recovery(report, block)
    return BytesCheckpointSource(block), report, meta, identity


def recover_node(args):
    """Only the survivor is a local input; the target path is not an argument."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    client = RemoteRSClient(args.remote_host, args.remote_port, args.timeout)
    try:
        receipt = json.loads(args.receipt.read_text())
        _require(receipt["global_state"] == "GLOBAL_COMMITTED", "stage_c_not_global_committed")
        key = {k: receipt[k] for k in ("stripe_id", "handle", "manifest_sha256")}
        before, raw = client.call("get_stripe_status", **key)
        _require(not raw, "stage_c_tcp_payload_forbidden")
        manifest = committed_status(before, receipt)
        survivor = args.survivor.read_bytes()
        _require(descriptor(survivor) == manifest["nodes"][1 - receipt["node_id"]],
                 "stage_c_wrong_survivor")
        after, block, transfer = client.rdma_reconstruct_missing_node(
            receipt["stripe_id"], receipt["node_id"], survivor,
            handle=receipt["handle"], manifest_sha256=receipt["manifest_sha256"],
            destination=output / "recovered_node.bin",
            return_host=args.rdma_return_host, return_port=args.rdma_return_port,
            device=args.rdma_device, gid_index=args.rdma_gid_index,
            port_num=args.rdma_port_num, timeout=args.timeout)
        report = dict(schema="kv-rs-gpu-source-v1", receipt=receipt,
            status_before=before, status_after=after, transfer=transfer,
            remote_host=args.remote_host, remote_port=args.remote_port,
            recovered_bytes=len(block), recovered_sha256=sha(block),
            survivor_sha256=sha(survivor), real_rdma=True, rs_recovery_byte_exact=True,
            runtime_original_local_file_used=False,
            recovery_input="survivor_plus_remote_committed_parity",
            control_transfers=client.transfers)
        validate_recovery(report, block)
        _require((output / "recovered_node.bin").read_bytes() == block,
                 "stage_c_receiver_artifact_changed")
        write_json(output / "recovery.json", report)
        return report
    except Exception as exc:
        write_json(output / "failure.json", dict(pass_source=False,
            failure_reason=f"{type(exc).__name__}:{exc}", transfers=client.transfers))
        raise


def check_source(directory, original):
    """Independent acceptance only. Original bytes never feed GPU restoration."""
    source, report, _, identity = load_bundle(directory)
    reference = Path(original).resolve()
    _require(reference != Path(directory).resolve() / "recovered_node.bin",
             "stage_c_original_must_be_independent")
    raw = reference.read_bytes()
    _require(raw == source.load(), "stage_c_independent_original_bytes_differ")
    return dict(pass_source=True, bundle=identity, original_reference=str(reference),
                original_sha256=sha(raw), recovered_sha256=report["recovered_sha256"],
                rs_recovery_byte_exact=True, real_rdma=True,
                runtime_original_local_file_used=False)
