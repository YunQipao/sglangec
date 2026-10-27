"""CPU protocol tests only; never establish remote RDMA/GPU acceptance."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .node_block_export import sha
from .rs_recovery_source import (
    committed_status, load_bundle, native_stats, validate_recovery,
)
from .verify_rs_resume import protect


class RSRecoverySourceTests(unittest.TestCase):
    def receipt(self, node):
        return dict(global_state="GLOBAL_COMMITTED", node_id=node, transport="rdma",
            remote_outcome="COMMITTED", recovery_byte_exact="NOT_YET_VERIFIED",
            stripe_id="online-test", handle="fresh-handle", manifest_sha256="a" * 64,
            parity_sha256="b" * 64, global_protection={"run_id": "r"},
            node_descriptor={"metadata": {"node_id": node}})

    def test_receipt_pair_requires_global_commit_and_same_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "d0.json", root / "d1.json"]
            for path, value in zip(paths, (self.receipt(0), self.receipt(1))):
                path.write_text(json.dumps(value))
            args = type("Args", (), dict(node0_receipt=paths[0], node1_receipt=paths[1],
                                         target_node=1, output=root / "new"))()
            result = protect(args)
            self.assertEqual(result["node_id"], 1)
            changed = self.receipt(1)
            changed["global_state"] = "COMMITTED"
            paths[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "authentic_global_receipt"):
                protect(type("Args", (), dict(node0_receipt=paths[0], node1_receipt=paths[1],
                    target_node=1, output=root / "new2"))())

    def test_committed_status_pins_manifest_handle_and_node_descriptor(self):
        identity = {"identity": "x"}
        node0 = {"metadata": {"node_id": 0, "ranks": [{"global_protection": identity}]}}
        node1 = {"metadata": {"node_id": 1, "ranks": [{"global_protection": identity}]}}
        manifest = dict(stripe_id="online-test", nodes=[node0, node1])
        status = dict(state="COMMITTED", stripe_id="online-test", handle="h",
            manifest_sha256="m", parity_sha256="p", manifest=manifest)
        receipt = dict(global_state="GLOBAL_COMMITTED", stripe_id="online-test", handle="h",
            manifest_sha256="m", parity_sha256="p", node_id=0,
            global_protection=identity, node_descriptor=node0)
        with patch("experiments.kv_ec.rs_recovery_source.validate_manifest"), \
             patch("experiments.kv_ec.rs_recovery_source.stripe_id", return_value="online-test"), \
             patch("experiments.kv_ec.rs_recovery_source.manifest_sha", return_value="m"):
            self.assertIs(committed_status(status, receipt), manifest)
            changed = dict(status, handle="stale")
            with self.assertRaisesRegex(ValueError, "global_commit_binding"):
                committed_status(changed, receipt)

    def test_verbs_stats_require_successful_native_completion_counters(self):
        line = ("KV_EC_RDMA_STATS bytes=1024 device_setup_ms=1 memory_registration_ms=1 "
                "qp_setup_ms=1 exchange_ms=1 completion_ms=1 total_ms=5 "
                "qp_num=9 remote_qpn=10 gid_index=3 port_num=1 completion_status=SUCCESS")
        native_stats(dict(transport="rdma", completion_status="SUCCESS", bytes=1024,
            qp_num=9, remote_qpn=10, helper_stdout=line), 1024)
        for bad in (dict(transport="CPU_MOCK"), dict(transport="rdma", bytes=1024,
                    completion_status="SUCCESS")):
            with self.assertRaisesRegex(ValueError, "native_verbs"):
                native_stats(bad, 1024)

    def test_bundle_uses_one_immutable_payload_and_pins_expected_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            block = b"reconstructed bytes"
            (path / "recovered_node.bin").write_bytes(block)
            (path / "recovery.json").write_text("{}")
            with patch("experiments.kv_ec.rs_recovery_source.validate_recovery", return_value={}):
                source, _, _, identity = load_bundle(path)
                self.assertEqual(source.load(), block)
                bad = dict(identity, block_sha256="0" * 64)
                with self.assertRaisesRegex(ValueError, "bundle_changed"):
                    load_bundle(path, bad)

    def test_recovery_data_source_rejects_local_file_or_tcp_substitution(self):
        with self.assertRaisesRegex(ValueError, "source_provenance"):
            validate_recovery({"schema": "kv-rs-gpu-source-v1",
                "runtime_original_local_file_used": True,
                "recovery_input": "survivor_plus_remote_committed_parity",
                "real_rdma": True, "rs_recovery_byte_exact": True}, b"x")

    def test_recovery_requires_target_hash_and_zero_tcp_payload(self):
        block = b"x"
        target, survivor = {"sha256": sha(block), "length": 1}, {"sha256": "survivor", "length": 1}
        receipt = {"node_id": 1}
        transfer = dict(state="COMPLETED", payload_sha256=sha(block), payload_bytes=1,
            receive_stats={"leg": "receive"}, send_stats={"leg": "send"},
            client_send_stats={"leg": "survivor-send"},
            client_receive_stats={"leg": "target-receive"})
        remote = dict(state="COMPLETED", payload_sha256=sha(block), payload_bytes=1,
            receive_stats=transfer["receive_stats"], send_stats=transfer["send_stats"])
        control = dict(op="prepare_rdma_recovery", transport="tcp",
            tx={"payload_bytes": 0}, rx={"payload_bytes": 0})
        report = dict(schema="kv-rs-gpu-source-v1", runtime_original_local_file_used=False,
            recovery_input="survivor_plus_remote_committed_parity", real_rdma=True,
            rs_recovery_byte_exact=True, receipt=receipt, status_before={}, status_after={},
            recovered_sha256=sha(block), recovered_bytes=1, survivor_sha256="survivor",
            transfer=transfer, control_transfers=[control])
        manifest = {"nodes": [survivor, target]}
        with patch("experiments.kv_ec.rs_recovery_source.committed_status", return_value=manifest), \
             patch("experiments.kv_ec.rs_recovery_source.validate_checkpoint",
                   return_value={"node_id": 1}), \
             patch("experiments.kv_ec.rs_recovery_source.descriptor", return_value=target), \
             patch("experiments.kv_ec.rs_recovery_source.native_stats"):
            report["status_after"] = {"rdma": {"recovery_1": remote}}
            self.assertEqual(validate_recovery(report, block), {"node_id": 1})
            report["recovered_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "recovered_node_identity"):
                validate_recovery(report, block)
            report["recovered_sha256"] = sha(block)
            control["rx"]["payload_bytes"] = 1
            with self.assertRaisesRegex(ValueError, "tcp_payload_forbidden"):
                validate_recovery(report, block)


if __name__ == "__main__":
    unittest.main()
