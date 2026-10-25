"""Logical erasure tests against actual packed KV bytes, no pool writeback.

Originals are an external comparison oracle ONLY. Reconstruction calls receive
only the surviving block and parity; unavailable nodes are passed as None.
"""

import time

from .node_block_export import descriptor, sha, unpack
from .remote_rs_client import binding
from .rs_codec import recover, xor


def verify_faults(client, committed, blocks, *, allow_fixture=False):
    checks = []
    for node in (0, 1):
        meta, ranks = unpack(blocks[node], allow_fixture=allow_fixture)
        parity = xor(*ranks)
        for lost in (0, 1):
            start = time.perf_counter()
            restored = xor(ranks[1 - lost], parity)
            if (
                restored != ranks[lost]
                or sha(restored) != meta["ranks"][lost]["sha256"]
            ):
                raise AssertionError("local rank recovery mismatch")
            checks.append(
                dict(
                    fault="single_rank",
                    node=node,
                    lost_rank=lost,
                    byte_exact=True,
                    recovery_ms=(time.perf_counter() - start) * 1000,
                )
            )
    for lost in (0, 1):
        start = time.perf_counter()
        result, raw = client.call(
            "reconstruct_missing_node",
            payload=blocks[1 - lost],
            missing_nodes=[lost],
            **binding(committed)
        )
        if (
            raw != blocks[lost]
            or descriptor(raw, allow_fixture=allow_fixture)
            != committed["manifest"]["nodes"][lost]
        ):
            raise AssertionError("remote node recovery mismatch")
        checks.append(
            dict(
                fault="whole_node",
                node=lost,
                byte_exact=True,
                recovery_ms=(time.perf_counter() - start) * 1000,
                server_recovery_ms=result["recovery_ms"],
                rs_reconstruct_ms=result["rs_reconstruct_ms"],
                node_unpack_verify_ms=result["node_unpack_verify_ms"],
            )
        )
    try:
        client.call(
            "reconstruct_missing_node", missing_nodes=[0, 1], **binding(committed)
        )
    except ValueError:
        checks.append(dict(fault="both_nodes", rejected=True))
    else:
        raise AssertionError("two erasures accepted")
    try:
        recover([None, blocks[1]], None, list(map(len, blocks)))
    except ValueError:
        checks.append(dict(fault="missing_parity", rejected=True))
    else:
        raise AssertionError("missing parity accepted")
    return checks
