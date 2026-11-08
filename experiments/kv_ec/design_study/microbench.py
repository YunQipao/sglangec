"""Five-repeat real-KV CPU codec + optional TCP echo microbenchmark.

Requires two independently exported node.bin files; uses actual rank payload
slices, never tiles data. Small snapshots cannot support large size points.
The optional echo endpoint is a separately supervised bounded service; it is
not an RS server and is never auto-started by the benchmark.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import socket
import socketserver
import subprocess
import time

import numpy as np

from .. import xor_codec
from ..multinode import rs_codec as gf_rs
from .. import rs_codec as remote_rs
from ..node_block_export import canonical, descriptor, sha, unpack
from ..remote_rs_protocol import receive, send


def timed(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, time.perf_counter_ns() - start


def echo(host, port, raw, timeout):
    with socket.create_connection((host, port), timeout=timeout) as sock:
        tx = send(sock, {"op": "echo"}, raw)
        meta, restored, rx = receive(sock, timeout)
    if meta.get("op") != "echo" or restored != raw:
        raise ValueError("TCP echo bytecheck failed")
    return restored, dict(tx=tx, rx=rx)


class EchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(self.server.io_timeout)
        meta, raw, _ = receive(self.request, self.server.io_timeout)
        if meta.get("op") != "echo" or not 0 < len(raw) <= self.server.max_payload:
            raise ValueError("bounded echo payload required")
        send(self.request, {"op": "echo"}, raw)


def run(paths, output, *, sizes=(1, 4, 16, 64), peer=None, timeout=30, allow_fixture=False, rank_capture=None):
    if ((rank_capture is None and len(paths) != 2) or len(sizes) > 8 or
            any(type(n) not in (int, float) or not math.isfinite(n) or not 0 < n <= 64
                or n * 1024**2 != int(n * 1024**2) for n in sizes) or not sizes):
        raise ValueError("node blocks or rank capture and bounded MiB size points required")
    if not 0 < timeout <= 120:
        raise ValueError("bounded timeout required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = []
    node_payloads = []
    for i, path in enumerate(paths if rank_capture is None else []):
        path = Path(path)
        raw = path.read_bytes()
        meta, ranks = unpack(raw, allow_fixture=allow_fixture)
        if meta["node_id"] != i:
            raise ValueError("ordered real D0,D1 node exports required")
        source.append(dict(path=str(path.resolve()), descriptor=descriptor(raw, allow_fixture=allow_fixture)))
        node_payloads.append(ranks)
    if rank_capture is None:
        # Check cross-node dtype, model and layout compatibility using production path.
        from ..rs_stripe_builder import build
        build("microbench-input-validation", [Path(p).read_bytes() for p in paths], allow_fixture=allow_fixture)
        input_scope = "complete node exports; rank payload slices"
    else:
        from ..rank_snapshot import validate_payload
        path = Path(rank_capture)
        raw = path.read_bytes()
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta.get("checksum_deferred") or meta.get("provenance") != "sglang_request":
            raise ValueError("recorded real capture checksums required")
        validate_payload(np.frombuffer(raw, dtype=np.uint8), meta)
        # Four NONOVERLAPPING slices of one captured rank, never invented TP ranks.
        width = len(raw) // 4
        pieces = [raw[i*width:(i+1)*width] for i in range(4)]
        node_payloads = [pieces[:2], pieces[2:]]
        source = [dict(path=str(path.resolve()), sha256=sha(raw), metadata=meta,
                       metadata_sha256=sha(path.with_suffix(".json").read_bytes()))]
        input_scope = "codec-only: four disjoint slices of ONE historical real TP rank; no TP/node placement claim"
    records, unavailable = [], []
    for mib in sizes:
        size = int(mib * 1024**2)
        # Preserve rank separation: local XOR uses genuine same-node TP shards.
        if any(len(b) < size for ranks in node_payloads for b in ranks):
            unavailable.append(dict(mib=mib, reason="real rank payload too small; no repetition/padding fabricated"))
            continue
        shards = [np.frombuffer(b[:size], dtype=np.uint8) for b in node_payloads[0]]
        data = [np.frombuffer(ranks[0][:size], dtype=np.uint8) for ranks in node_payloads]
        for repeat in range(5):
            def record(op, ns, useful_bytes, **extra):
                records.append(dict(mib=mib, repeat=repeat, operation=op, elapsed_ns=ns,
                                    useful_bytes=useful_bytes, useful_bytes_per_s=useful_bytes/(ns/1e9),
                                    byte_exact=True, **extra))
            parity, ns = timed(lambda: xor_codec.encode(shards))
            record("local_xor_encode", ns, size*len(shards), execution_location="CPU Host DRAM")
            for lost in range(len(shards)):
                restored, ns = timed(lambda: xor_codec.recover([None if j == lost else b for j, b in enumerate(shards)], parity))
                if not np.array_equal(restored, shards[lost]):
                    raise AssertionError("XOR bytecheck failed")
                record("local_xor_decode", ns, size, missing_rank=lost, sha256=sha(restored), execution_location="CPU Host DRAM")
            (_, _, parity), ns = timed(lambda: gf_rs.encode(*data))
            record("gf256_1_2_rs_encode", ns, size*2, codec="multinode/rs_codec: P=D0 xor 2*D1")
            for lost in (0, 1):
                restored, ns = timed(lambda: gf_rs.recover(data0=None if lost == 0 else data[0],
                                                         data1=None if lost == 1 else data[1],
                                                         parity=parity, lengths=[size, size]))
                if any(not np.array_equal(restored[j], data[j]) for j in (0, 1)):
                    raise AssertionError("GF256 RS bytecheck failed")
                record("gf256_1_2_rs_decode", ns, size, missing_node=lost, sha256=sha(restored[lost]))
            raw = [b.tobytes() for b in data]
            parity, ns = timed(lambda: remote_rs.encode(*raw))
            record("remote_protocol_rs_encode", ns, size*2, codec="rs_codec: P=D0 xor D1; distinct from GF256(1,2)")
            for lost in (0, 1):
                restored, ns = timed(lambda: remote_rs.recover([None if j == lost else b for j, b in enumerate(raw)], parity, [size, size]))
                if restored != raw[lost]:
                    raise AssertionError("remote protocol RS bytecheck failed")
                record("remote_protocol_rs_decode", ns, size, missing_node=lost, sha256=sha(restored))
            if peer:
                (restored, traffic), ns = timed(lambda: echo(*peer, raw[0], timeout))
                record("tcp_echo_roundtrip_inclusive", ns, size*2, transport="TCP", traffic=traffic,
                       sha256=sha(restored), scope="connect + checked upload + checked download; not one-way latency")
                start = time.perf_counter_ns()
                (survivor, traffic), transfer_ns = timed(lambda: echo(*peer, raw[1], timeout))
                rebuilt, decode_ns = timed(lambda: remote_rs.recover([None, survivor], parity, [size, size]))
                if rebuilt != raw[0]:
                    raise AssertionError("pipeline bytecheck failed")
                total_ns = time.perf_counter_ns() - start
                record("host_pipeline_echo_then_decode", total_ns, size, transfer_ns=transfer_ns,
                       decode_ns=decode_ns, sha256=sha(rebuilt), transport="TCP",
                       scope="sequential Host pipeline; gather/D2H/H2D/writeback not measured")
    result = dict(schema="ec-real-kv-microbench-v1", repeats=5, records=records, unavailable=unavailable,
                  evidence="synthetic_fixture" if allow_fixture else "sglang_request_metadata_and_bytechecks",
                  input_scope=input_scope,
                  provenance_independently_attested=False, pipeline_gpu_stages="NOT MEASURED",
                  host_rdma="NOT MEASURED", gpudirect_rdma="NOT IMPLEMENTED")
    with (output / "raw.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    modules = [Path(__file__), Path(xor_codec.__file__), Path(gf_rs.__file__), Path(remote_rs.__file__),
               Path(__file__).parents[1] / "node_block_export.py", Path(__file__).parents[1] / "rank_snapshot.py",
               Path(__file__).parents[1] / "remote_rs_protocol.py"]
    manifest = dict(schema="ec-microbench-manifest-v1", git_head=commit, python=platform.python_version(),
                    numpy=np.__version__, host=socket.gethostname(), machine=platform.machine(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)), clock="perf_counter_ns", warmup=0,
                    sources=source, sizes_mib=list(sizes), repeats=5, peer=peer,
                    raw_sha256=sha((output / "raw.json").read_bytes()),
                    code_sha256={str(p): sha(p.read_bytes()) for p in modules})
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="op", required=True)
    q = sub.add_parser("run")
    inputs = q.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--nodes", type=Path, nargs=2)
    inputs.add_argument("--rank-capture", type=Path)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--sizes-mib", type=float, nargs="+", default=[1, 4, 16, 64])
    q.add_argument("--echo-host")
    q.add_argument("--echo-port", type=int, default=32953)
    q = sub.add_parser("echo-server")
    q.add_argument("--host", default="127.0.0.1")
    q.add_argument("--port", type=int, default=32953)
    args = p.parse_args()
    if args.op == "run":
        run(args.nodes or [], args.output, sizes=args.sizes_mib,
            peer=(args.echo_host, args.echo_port) if args.echo_host else None, rank_capture=args.rank_capture)
    else:
        from .admission import PROTECTED_PORTS
        if args.port in PROTECTED_PORTS or not 1024 <= args.port <= 65535:
            p.error("dedicated port required")
        with socketserver.TCPServer((args.host, args.port), EchoHandler) as server:
            server.io_timeout, server.max_payload = 30, 64 * 1024**2
            server.serve_forever()


if __name__ == "__main__":
    main()
