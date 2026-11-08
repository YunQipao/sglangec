"""Parent-supervised real rank first-N bytes: TCP vs HostRDMA, no echo/GPU.

prepare/collect are offline. send never launches a receiver. receive must run
under the existing parent-owned perf_stage_e supervisor. Fixed ports only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from ...multinode.tcp_transport import TCPTransport, receive_file as tcp_receive
from ...multinode.rdma_transport import RDMATransport, receive_file as rdma_receive
from ...node_block_export import unpack
from ...perf_ports import PortLease, assert_free, listeners
from ...perf_stage_e import process_identity, require_trial
from ..admission import require_host_rdma
from .evidence import code_hashes, save, sha

PORTS = {"tcp":32963, "rdma":32964}
SIZES = (1, 4, 16, 64)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_rank(path, metadata=None, rank=0):
    """No fixture flag, no expansion/padding, full recorded source validation."""
    path = Path(path)
    raw = path.read_bytes()
    if metadata is None:
        meta, payloads = unpack(raw)
        if type(rank) is not int or not 0 <= rank < len(payloads):
            raise ValueError("rank not in real NodeBlock")
        payload, rank_meta = payloads[rank], meta["ranks"][rank]
    else:
        from ...rank_snapshot import validate_payload
        rank_meta = read(metadata)
        if rank_meta.get("provenance") != "sglang_request" or rank_meta.get("checksum_deferred"):
            raise ValueError("real recorded rank checksums required")
        validate_payload(np.frombuffer(raw, dtype=np.uint8), rank_meta)
        payload = raw
    if rank_meta["token_count"] != 4096:
        raise ValueError("parent-provided real4096 KV required")
    return payload, dict(path=str(path.resolve()), sha256=sha(raw), rank=rank_meta["tp_rank"],
        metadata_path=str(Path(metadata).resolve()) if metadata else None,
        metadata_sha256=sha(Path(metadata).read_bytes()) if metadata else None,
        rank_sha256=sha(payload), rank_payload_bytes=len(payload), token_count=4096)


def prepare(source, output, run_id, *, metadata=None, rank=0, rounds=1, smoke=None):
    if rounds not in (1, 5) or not run_id or len(run_id) > 96:
        raise ValueError("new bounded run ID and smoke/five rounds required")
    payload, identity = load_rank(source, metadata, rank)
    if len(payload) < 64*1024**2:
        raise ValueError("real rank shorter than 64MiB; no synthetic repetition permitted")
    if rounds == 5:
        proof = read(smoke) if smoke else {}
        if (proof.get("schema") != "ec-transport-comparison-v1" or proof.get("status") != "PASS"
                or proof.get("rounds") != 1 or proof.get("source") != identity or proof.get("run_id") == run_id
                or proof.get("code_sha256") != code_hashes()):
            raise ValueError("passing same-source TCP/HostRDMA smoke required")
    rows = [dict(transport=transport, port=PORTS[transport], mib=mib, repeat=repeat,
                 payload_bytes=mib*1024**2, sha256=sha(payload[:mib*1024**2]))
            for transport in PORTS for mib in SIZES for repeat in range(rounds)]
    plan = dict(schema="ec-transport-plan-v1", run_id=run_id, rounds=rounds, source=identity,
                rows=rows, code_sha256=code_hashes(), gpudirect=False, h2d_ms=None)
    plan["plan_sha256"] = digest(plan)
    save(output, plan)
    return plan


def validate(plan):
    if (plan.get("schema") != "ec-transport-plan-v1" or plan.get("rounds") not in (1, 5)
            or digest({k:v for k,v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256")):
        raise ValueError("invalid plan/hash")
    expected = [(t, m, r) for t in PORTS for m in SIZES for r in range(plan["rounds"])]
    if [(r["transport"], r["mib"], r["repeat"]) for r in plan["rows"]] != expected:
        raise ValueError("complete ordered 1/4/16/64MiB TCP/RDMA matrix required")
    if any(r["port"] != PORTS[r["transport"]] or r["payload_bytes"] != r["mib"]*1024**2 for r in plan["rows"]):
        raise ValueError("fixed port and actual first-N bytes required")


def native_check(stats, row, sending):
    if (not stats.get("helper_stdout", "").startswith("KV_EC_RDMA_STATS ")
            or stats.get("completion_status") != "SUCCESS" or stats.get("bytes") != row["payload_bytes"]
            or (sending and (stats.get("write_segments", 0) < 1 or "rdma_write_cq_ms" not in stats))
            or (not sending and stats.get("receive_notifications") != 1)):
        raise ValueError("native CQ/bytes/completion evidence missing")


def require_receiver_supervisor(root, run_id):
    owner = require_trial(root)
    if owner["run_id"] != run_id:
        raise ValueError("receiver supervisor run ID mismatch")
    path = root/"receiver/owned-process.json"
    deadline = time.monotonic()+5
    while True:
        try:
            owned = read(path)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise RuntimeError("receiver must run under parent's existing exact-owned supervisor")
            time.sleep(.05)
    current = process_identity(os.getpid())
    if owned.get("run_id") != run_id or any(owned.get(k) != current[k] for k in ("pid","uid","start_ticks","boot_id","pgid","session")):
        raise RuntimeError("receiver is not the exact parent-supervised process")


def transfer(plan, output, *, sending, host, timeout=120, max_seconds=7200,
             device="-", gid_index=None, port_num=1):
    validate(plan)
    if (not math.isfinite(timeout) or not 0 < timeout <= 300 or
            not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400):
        raise ValueError("bounded operation/lifecycle timeouts required")
    output = Path(output).resolve()
    if not sending:
        # output must be a direct child of the parent's fresh trial root.
        require_receiver_supervisor(output.parent, plan["run_id"])
    payload = None
    if sending:
        source = plan["source"]
        payload, identity = load_rank(source["path"], source["metadata_path"], source["rank"])
        if identity != source:
            raise ValueError("real source changed since plan")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="ec-transport-side-v1", run_id=plan["run_id"], plan_sha256=plan["plan_sha256"],
        status="PARTIAL", side="sender" if sending else "receiver", records=[],
        gpudirect=False, h2d_ms=None, isolated_performance_eligible=False,
        direction="one-way payload plus small ACK; no payload echo", source=plan["source"])
    deadline = time.monotonic()+max_seconds
    rdma = None
    try:
        for index, row in enumerate(plan["rows"]):
            rec = dict(row, mode=row["transport"], prefix_tokens=4096, index=index, success=False)
            begin = time.monotonic_ns()
            try:
                remaining = min(timeout, deadline-time.monotonic())
                if remaining <= 0:
                    raise TimeoutError("transfer lifecycle deadline exceeded")
                if row["transport"] == "rdma":
                    require_host_rdma()  # 243 must fail; no fallback to TCP
                if sending:
                    raw = payload[:row["payload_bytes"]]
                    if len(raw) != row["payload_bytes"] or sha(raw) != row["sha256"]:
                        raise ValueError("first-N source slice changed")
                    if row["transport"] == "tcp":
                        transport = TCPTransport(retry_timeout=remaining, connect_timeout=min(5, remaining))
                        elapsed = transport.send(host, row["port"], raw)
                        stats = transport.last_stats
                        rec["acknowledged_unidirectional_ms"] = stats["acknowledged_unidirectional_ms"]
                    else:
                        if rdma is None:
                            rdma = RDMATransport(device=device, gid_index=gid_index, port_num=port_num, timeout=remaining)
                        rdma.timeout = remaining
                        stats = rdma.send_with_stats(host, row["port"], raw)
                        native_check(stats, row, True)
                        elapsed = stats["elapsed_s"]
                        rec["acknowledged_unidirectional_ms"] = stats["rdma_write_cq_ms"]+stats["tcp_ack_wait_ms"]
                    rec.update(sender_sha256=sha(raw), transport_end_to_end_ms=elapsed*1000, native_stats=stats)
                else:
                    destination = output/f"payload-{index:03d}.bin"
                    port_root = output/f"port-{index:03d}"
                    port_root.mkdir()
                    # PortLease is advisory; its kernel table read and the final
                    # pre-bind scan both fail closed, record foreign/unknown owners.
                    with PortLease([row["port"]], port_root, plan["run_id"]):
                        rec["prebind_listeners"] = listeners([row["port"]])
                        assert_free([row["port"]], "MICROBENCH_PORT_OCCUPIED")
                        if row["transport"] == "tcp":
                            status = {}
                            elapsed = tcp_receive(host, row["port"], destination, timeout=remaining, status=status)
                            stats = dict(status, elapsed_s=elapsed, cq_not_applicable=True)
                        else:
                            stats = rdma_receive(row["port"], destination, device=device, bind_host=host,
                                gid_index=gid_index, port_num=port_num, timeout=remaining, return_stats=True)
                            native_check(stats, row, False)
                            elapsed = stats["elapsed_s"]
                    received = destination.read_bytes()
                    if len(received) != row["payload_bytes"] or sha(received) != row["sha256"]:
                        raise ValueError("received SHA/actual byte count mismatch")
                    rec.update(receiver_sha256=sha(received), receiver_sha_verified=True,
                        transport_end_to_end_ms=elapsed*1000, native_stats=stats,
                        end_to_end_scope="includes accept/wait, native setup, payload and persistence; unsynchronized host clock")
                    destination.unlink()  # own fresh payload only, SHA receipt retained
                rec["success"] = True
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["wall_ms"] = (time.monotonic_ns()-begin)/1e6
            result["records"].append(rec)
            save(output/f"record-{index:03d}.json", rec)
            if not rec["success"]:
                break  # no retries that could desynchronize sender/receiver rows
        if len(result["records"]) == len(plan["rows"]) and all(r["success"] for r in result["records"]):
            result["status"] = "PASS"
    finally:
        if rdma is not None:
            rdma.close()
        save(output/"raw.json", result)
    return result


def collect(plan, sender, receiver, output):
    validate(plan)
    a, b = read(sender), read(receiver)
    for side, artifact in (("sender", a), ("receiver", b)):
        if (artifact.get("schema") != "ec-transport-side-v1" or artifact.get("side") != side
                or artifact.get("run_id") != plan["run_id"] or artifact.get("plan_sha256") != plan["plan_sha256"]):
            raise ValueError("transfer side identity mismatch")
    records = []
    for index, row in enumerate(plan["rows"]):
        tx = next((r for r in a["records"] if r["index"] == index), {})
        rx = next((r for r in b["records"] if r["index"] == index), {})
        passed = (tx.get("success") is True and rx.get("success") is True
                  and tx.get("sender_sha256") == rx.get("receiver_sha256") == row["sha256"]
                  and tx.get("payload_bytes") == rx.get("payload_bytes") == row["payload_bytes"]
                  and rx.get("receiver_sha_verified") is True)
        ack_ms=tx.get("acknowledged_unidirectional_ms")
        records.append(dict(row, mode=row["transport"], prefix_tokens=4096, byte_exact=passed,
            success=passed, sender=tx, receiver=rx,
            acknowledged_unidirectional_ms=ack_ms,
            useful_bytes_per_s=row["payload_bytes"]/(ack_ms/1000) if passed and isinstance(ack_ms,(int,float)) and ack_ms>0 else None,
            # Sender and receiver clocks are never subtracted from each other.
            gpudirect=False, h2d_ms=None))
    result = dict(schema="ec-transport-comparison-v1", run_id=plan["run_id"], rounds=plan["rounds"],
        evidence_class="current_smoke" if plan["rounds"] == 1 else "current_five_round",
        code_sha256=plan.get("code_sha256"),
        plan_sha256=plan["plan_sha256"], source=plan["source"], records=records,
        status="PASS" if all(r["success"] for r in records) else "PARTIAL",
        sender_artifact_sha256=sha(Path(sender).read_bytes()), receiver_artifact_sha256=sha(Path(receiver).read_bytes()),
        gpudirect=False, note="TCP sendall alone is not one-way arrival; compare acknowledged one-direction payload endpoints; no echo RTT vs CQ comparison")
    save(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="op", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--source", type=Path, required=True)
    q.add_argument("--metadata", type=Path)
    q.add_argument("--rank", type=int, default=0)
    q.add_argument("--run-id", required=True)
    q.add_argument("--rounds", type=int, choices=(1,5), default=1)
    q.add_argument("--smoke", type=Path)
    q.add_argument("--output", type=Path, required=True)
    for name in ("send", "receive"):
        q = sub.add_parser(name)
        q.add_argument("--plan", type=Path, required=True)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--host", required=True)
        q.add_argument("--timeout", type=float, default=120)
        q.add_argument("--max-seconds", type=float, default=7200)
        q.add_argument("--device", default="-")
        q.add_argument("--gid-index", type=int)
        q.add_argument("--port-num", type=int, default=1)
    q = sub.add_parser("collect")
    for name in ("plan", "sender", "receiver", "output"):
        q.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    if a.op == "prepare":
        result = prepare(a.source, a.output, a.run_id, metadata=a.metadata, rank=a.rank, rounds=a.rounds, smoke=a.smoke)
    elif a.op == "collect":
        result = collect(read(a.plan), a.sender, a.receiver, a.output)
    else:
        result = transfer(read(a.plan), a.output, sending=a.op == "send", host=a.host, timeout=a.timeout,
            max_seconds=a.max_seconds, device=a.device, gid_index=a.gid_index, port_num=a.port_num)
    print(json.dumps(dict(status=result.get("status", "PREPARED"), output=str(a.output))))
    if result.get("status") == "PARTIAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
