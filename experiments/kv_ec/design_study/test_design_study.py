import subprocess
"""Focused CPU evidence; fixtures are not physical-failure/GPU experiments."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from . import admission, microbench, policy, workload
from .storage import LocalReplicaClient, ReplicaStore, get_replica, put_replica
from .remote import TCPOnlyStripeStore
from . import three_host
from . import source_trial
from ..node_block_export import descriptor, pack, sha
from ..remote_rs_client import RemoteRSClient, binding
from ..remote_rs_server import RemoteRSServer, StripeStore
from ..rs_stripe_builder import build
from ..test_parity_manager import metadata_and_blocks
from ..test_remote_rs import node_fixture

ROOT = Path(__file__).parent


class DirectClient:
    def __init__(self, store):
        self.store = store
        self.transfers = []

    def call(self, op, payload=b"", **fields):
        started = time.perf_counter_ns()
        response, raw = self.store.dispatch(dict(op=op, **fields), payload)
        self.transfers.append(dict(op=op, transport=fields.get("transport", "tcp"),
                                   roundtrip_ms=(time.perf_counter_ns()-started)/1e6,
                                   tx={"stream_bytes": len(payload)},
                                   rx={"stream_bytes": len(raw)}))
        return response, raw


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "placement.json").read_text())

    def test_three_domains_and_alias_guard(self):
        self.assertEqual(len(admission.placement(self.config)), 64)
        for key, value in (("host", "192.168.2.243"), ("physical_id", "233"),
                           ("host", "192.168.2.245")):
            changed = deepcopy(self.config)
            changed["nodes"][2][key] = value
            with self.assertRaises(ValueError):
                admission.placement(changed)

    def test_rdma_and_protected_ports_fail_closed(self):
        changed = deepcopy(self.config)
        changed["transport"] = "rdma"
        with self.assertRaisesRegex(ValueError, "no TCP fallback"):
            admission.placement(changed)
        for port in admission.PROTECTED_PORTS:
            changed = deepcopy(self.config)
            changed["parity_port"] = port
            with self.assertRaises(ValueError):
                admission.placement(changed)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                admission.require_host_rdma(Path(tmp), Path(tmp))

    def test_wrong_local_machine_cannot_serve(self):
        with patch.object(admission.subprocess, "run", return_value=subprocess.CompletedProcess(
                ["ip", "-j", "address"], 0, stdout="[]")):
            with self.assertRaises(ValueError):
                admission.local_identity(self.config, "P", "243")

    def test_uniform_tp2_fallback_requires_every_baseline(self):
        wanted = dict(tp=4, input_tokens=65536, output_tokens=4096, batch=16, dtype="bf16", trace_sha256="a"*64)
        fallback = dict(wanted, tp=2, input_tokens=4096, output_tokens=17, batch=1)
        rows = [dict(baseline=b, config=fallback, passed=True, free_bytes=100, required_bytes=90,
                     preflight_artifact_sha256="b"*64) for b in admission.BASELINES]
        selected = admission.uniform_resources([wanted, fallback], rows)
        self.assertEqual(selected["config"], fallback)
        with self.assertRaises(ValueError):
            admission.uniform_resources([wanted, fallback], rows[:-1])
        rows[-1]["required_bytes"] = 101
        with self.assertRaises(ValueError):
            admission.uniform_resources([fallback], rows)

    def test_memory_categories_do_not_add_xor_rs_ratios(self):
        ledger = admission.memory_ledger(original=100, stable_parity=50, stable_replica=0, snapshots=100, temporary=80)
        self.assertEqual(ledger["stable_overhead_pct"], 50)
        self.assertEqual(ledger["accounted_additional_peak_bytes"], 230)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.raw = node_fixture(0)
        self.store = ReplicaStore(max_bytes=len(self.raw)*2, max_objects=2, ttl=60, allow_fixture=True)
        self.client = LocalReplicaClient(self.store)
        self.key = put_replica(self.client, "req/checkpoint", self.raw, allow_fixture=True)

    def test_cpu_replica_owns_bytes_and_binds_all_identity(self):
        self.assertEqual(get_replica(self.client, self.key, descriptor(self.raw, allow_fixture=True), allow_fixture=True), self.raw)
        self.assertEqual(put_replica(self.client, "req/checkpoint", self.raw, allow_fixture=True), self.key)
        for field in self.key:
            bad = dict(self.key, **{field: "foreign"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.client.call("get_replica", **bad)
        # Response bindings are verified too; a malicious/buggy service cannot
        # return different identity metadata beside otherwise correct bytes.
        with patch.object(self.client, "call", return_value=(dict(self.key, handle="foreign"), self.raw)):
            with self.assertRaisesRegex(ValueError, "response/request binding"):
                get_replica(self.client, self.key, descriptor(self.raw, allow_fixture=True), allow_fixture=True)
        # A different immutable checkpoint cannot replace this object.
        with self.assertRaises(ValueError):
            put_replica(self.client, "req/checkpoint", node_fixture(1), allow_fixture=True)

    def test_corruption_expiry_budget_and_restart(self):
        item = self.store.objects["req/checkpoint"]
        item["raw"] = item["raw"][:-1] + bytes([item["raw"][-1] ^ 1])
        with self.assertRaises(ValueError):
            self.client.call("get_replica", **self.key)
        item["deadline"] = 0
        with self.assertRaises(ValueError):
            put_replica(self.client, "req/checkpoint", self.raw, allow_fixture=True)
        too_small = DirectClient(ReplicaStore(max_bytes=1, max_objects=1, ttl=1, allow_fixture=True))
        with self.assertRaises(ValueError):
            put_replica(too_small, "new", self.raw, allow_fixture=True)
        restarted = DirectClient(ReplicaStore(max_bytes=len(self.raw), max_objects=1, ttl=60, allow_fixture=True))
        put_replica(restarted, "req/checkpoint", self.raw, allow_fixture=True)
        with self.assertRaises(ValueError):
            restarted.call("get_replica", **self.key)

    def test_remote_tcp_replication_uses_existing_checked_frames(self):
        try:
            server = RemoteRSServer(("127.0.0.1", 0), self.store, timeout=2)
        except PermissionError as exc:
            self.skipTest(f"loopback bind denied: {exc}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RemoteRSClient(*server.server_address, timeout=2)
            key = put_replica(client, "remote", self.raw, allow_fixture=True)
            restored = get_replica(client, key, descriptor(self.raw, allow_fixture=True), allow_fixture=True)
            self.assertEqual(restored, self.raw)
            self.assertGreater(sum(t["tx"]["stream_bytes"] for t in client.transfers), len(self.raw))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def test_existing_parity_only_protocol_reconstructs_each_block_not_node_failure(self):
        blocks = [node_fixture(i) for i in (0, 1)]
        store = StripeStore(allow_fixture=True, rdma_directory=ROOT / "artifacts/unused-rdma")
        client = DirectClient(store)
        created, _ = client.call("create_stripe", stripe_id="three-domain-test", manifest=build("three-domain-test", blocks, allow_fixture=True))
        key = binding(created)
        for i in (0, 1):
            state, _ = client.call("submit_node_block", node_id=i, payload=blocks[i], **key)
        self.assertEqual(state["state"], "COMMITTED")
        self.assertEqual(store.stripes[key["stripe_id"]]["blocks"], {})
        self.assertEqual(state["retained_node_block_bytes"], 0)
        self.assertGreater(state["parity_bytes"], 0)
        for lost in (0, 1):
            with patch.object(store.rs_codec, "recover_xor",
                              wraps=store.rs_codec.recover_xor) as recovery:
                _, raw = client.call("reconstruct_missing_node", missing_nodes=[lost], payload=blocks[1-lost], **key)
                self.assertIsNone(recovery.call_args.args[0][lost])
                self.assertEqual(store.stripes[key["stripe_id"]]["blocks"], {})
            self.assertEqual(raw, blocks[lost])
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            client.call("submit_node_block", node_id=0, payload=node_fixture(1), **key)
        with self.assertRaises(ValueError):
            client.call("reconstruct_missing_node", missing_nodes=[0, 1], payload=b"", **key)

    def test_tcp_service_rejects_rdma_without_helper_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TCPOnlyStripeStore(allow_fixture=True, rdma_directory=tmp)
            for request in (dict(op="prepare_rdma"), dict(op="fetch_parity", transport="rdma")):
                with patch("experiments.kv_ec.multinode.rdma_transport.RDMATransport") as verbs:
                    with self.assertRaisesRegex(ValueError, "TCP only"):
                        store.dispatch(request, b"")
                    verbs.assert_not_called()


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.trace = workload.trace(seed=11, requests=1, input_tokens=256, output_tokens=2,
                                    vocab_size=1000, model_sha256="a"*64)
        self.row = self.trace["requests"][0]

    @staticmethod
    def sse(outputs):
        return io.BytesIO(b"".join(b"data: " + json.dumps(dict(output_ids=ids, meta_info=dict(completion_tokens=len(ids), prompt_tokens=256))).encode() + b"\n\n" for ids in outputs) + b"data: [DONE]\n")

    def test_trace_determinism_and_tamper(self):
        self.assertEqual(workload.validate_trace(self.trace), self.trace)
        self.assertEqual(self.trace, workload.trace(seed=11, requests=1, input_tokens=256, output_tokens=2, vocab_size=1000, model_sha256="a"*64))
        self.trace["requests"][0]["input_ids"][0] += 1
        with self.assertRaises(ValueError):
            workload.validate_trace(self.trace)

    def test_fault_to_sse_and_reference_boundaries(self):
        receipt = dict(kind="kv_block_erasure", request_id=self.row["request_id"], verified=True,
                       observed_completion_tokens=0, checkpoint_binding=dict(handle="test-only", manifest_sha256="a"*64))
        with patch("urllib.request.urlopen", return_value=self.sse([[8], [8, 9]])):
            result = workload.measure("http://test", self.row, fault=lambda row: receipt, expected_output=[8, 9])
        self.assertGreaterEqual(result["fault_to_sse_ms"], result["ttft_ms"])
        self.assertFalse(result["physical_failure"])
        self.assertTrue(result["output_reference_checked"])
        self.assertEqual(result["first_observed_completion_tokens"], 1)

    def test_decode_position_fault_uses_next_sse_event_not_prefault_token(self):
        def inject(row):
            return dict(kind="kv_block_erasure", request_id=row["request_id"], verified=True,
                        observed_completion_tokens=row["observed_completion_tokens"],
                        checkpoint_binding=dict(handle="test", manifest_sha256="a"*64))
        with patch("urllib.request.urlopen", return_value=self.sse([[8], [8, 9]])):
            result = workload.measure("http://test", self.row, fault=inject, fault_after_tokens=1)
        self.assertLess(result["first_output_ns"], result["fault_boundary_ns"])
        self.assertGreater(result["recovered_first_output_ns"], result["fault_boundary_ns"])
        self.assertEqual(result["fault_receipt"]["observed_completion_tokens"], 1)
        with patch("urllib.request.urlopen", return_value=self.sse([[8, 9]])):
            with self.assertRaisesRegex(ValueError, "batching skipped"):
                workload.measure("http://test", self.row, fault=inject, fault_after_tokens=1)

    def test_physical_fault_and_stale_binding_rejected_before_http(self):
        for receipt in (dict(kind="process_failure", verified=True), dict(kind="kv_block_erasure", verified=True, request_id="wrong")):
            with patch("urllib.request.urlopen") as http:
                with self.assertRaises(ValueError):
                    workload.measure("http://test", self.row, fault=lambda row: receipt)
                http.assert_not_called()

    def test_truncated_rewritten_and_wrong_output_fail(self):
        for outputs in ([[8]], [[8], [7, 9]], [[8, 10]]):
            with patch("urllib.request.urlopen", return_value=self.sse(outputs)):
                with self.assertRaises(ValueError):
                    workload.measure("http://test", self.row, expected_output=[8, 9])

    def test_failed_requests_are_preserved_and_output_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new"
            with patch("urllib.request.urlopen", side_effect=TimeoutError("test")):
                result = workload.run("http://test", self.trace, output, "base")
            self.assertEqual(result["completed"], 0)
            self.assertTrue(result["results"][0]["timeout"])
            with self.assertRaises(FileExistsError):
                workload.run("http://test", self.trace, output, "base")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.context = dict(model_sha256="a"*64, trace_sha256="b"*64, tp=2, input_tokens=256,
                            output_tokens=17, batch=1, fault_kind="kv_block_erasure", transport="TCP",
                            hardware_id="fixture", measurement_profile="fault_to_kv_ready")
        self.rows = [dict(context=self.context, route=route, trial_id=f"{route}-{i}", success=True,
                          boundary="fault_to_kv_ready", artifact_sha256="c"*64, elapsed_ms=value+i)
                     for route, value in (("ec", 10), ("prefill", 20)) for i in range(5)]

    def test_exact_measured_context_and_safety_gate(self):
        calibration = policy.calibrate(self.rows, self.context)
        self.assertEqual(policy.choose(calibration, self.context, checkpoint_committed=True, binding_verified=True, replay_safe=True), "ec")
        self.assertEqual(policy.choose(calibration, self.context, checkpoint_committed=False, binding_verified=True, replay_safe=True), "prefill")
        self.assertEqual(policy.choose(calibration, self.context, checkpoint_committed=True, binding_verified=False, replay_safe=False), "fail_closed")
        with self.assertRaises(ValueError):
            policy.choose(calibration, dict(self.context, input_tokens=65536), checkpoint_committed=True, binding_verified=True, replay_safe=True)

    def test_no_censored_failures_or_mixed_boundaries(self):
        for key, value in (("success", False), ("boundary", "codec_only"), ("elapsed_ms", float("nan"))):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                policy.calibrate(rows, self.context)
        summary = policy.summarize([dict(recovery_entered=True, recovered=True, completed=True, recovery_ms=10),
                                    dict(recovery_entered=True, recovered=False, completed=False, timeout=True)])
        self.assertEqual(summary["recovery_success_rate"], .5)
        self.assertEqual(summary["timeout_rate"], .5)
        self.assertIsNone(summary["unconditional_mttr_ms"])


class MicrobenchTests(unittest.TestCase):
    def test_actual_codec_paths_five_repeats_bytechecks_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for node in (0, 1):
                arrays, metas = metadata_and_blocks(tp=2, count=512)
                for m in metas:
                    m.update(snapshot_id=f"capture-{node}", request_id=f"request-{node}",
                             request_version=f"version-{node}", checkpoint_version=f"checkpoint-{node}",
                             manager_epoch=f"manager-{node}")
                path = tmp / f"node{node}.bin"
                with path.open("xb") as stream:
                    stream.write(pack(node, [a.tobytes() for a in arrays], metas, allow_fixture=True))
                paths.append(path)
            result = microbench.run(paths, tmp / "bench", sizes=[1, 4], allow_fixture=True)
            self.assertEqual(result["evidence"], "synthetic_fixture")
            self.assertEqual({r["repeat"] for r in result["records"]}, set(range(5)))
            self.assertTrue(all(r["byte_exact"] for r in result["records"]))
            self.assertEqual(len(result["unavailable"]), 1)
            manifest = json.loads((tmp / "bench/manifest.json").read_text())
            self.assertEqual(manifest["raw_sha256"], sha((tmp / "bench/raw.json").read_bytes()))
            with self.assertRaises(FileExistsError):
                microbench.run(paths, tmp / "bench", allow_fixture=True)
            with self.assertRaises(ValueError):
                microbench.run(paths, tmp / "real-reject", sizes=[1])


class ThreeHostOrchestrationTests(unittest.TestCase):
    @staticmethod
    def _real_format_test_block(node):
        arrays, ranks = metadata_and_blocks(tp=2, count=128)
        for rank in ranks:
            rank.update(provenance="sglang_request", snapshot_id=f"captured-{node}",
                        request_id=f"captured-request-{node}", request_version="r1",
                        checkpoint_version="c1", manager_epoch=f"epoch-{node}",
                        model_identifier="Qwen3-8B", model_sha256="a"*64,
                        source_commit="b"*40, core_diff_sha256="c"*64)
        return pack(node, [a.tobytes() for a in arrays], ranks)

    def test_five_tcp_protocol_cycles_bind_outputs_and_drop_source_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifacts = tmp / "artifacts"
            artifacts.mkdir()
            blocks = [self._real_format_test_block(i) for i in (0, 1)]
            paths = []
            for i, raw in enumerate(blocks):
                path = tmp / f"node{i}.bin"
                path.write_bytes(raw)
                paths.append(path)
            config = json.loads((ROOT / "placement.json").read_text())
            config["max_bytes"] = 4 * max(map(len, blocks))
            config_path = tmp / "placement.json"
            config_path.write_text(json.dumps(config))
            store = StripeStore(ttl=60, max_stripes=8, max_bytes=config["max_bytes"])
            peer = DirectClient(store)
            with patch.object(three_host, "ARTIFACTS", artifacts):
                report = three_host.run(config_path, paths, artifacts / "five-round-test",
                                        stripe_id="five-round-test", connect=lambda _: peer)
            self.assertEqual(report["transport"], "TCP")
            self.assertEqual(report["state"], "PASS")
            self.assertEqual(sum(op["phase"] == "fetch_parity" for op in report["operations"]), 5)
            self.assertEqual(sum(op["phase"] == "recover" for op in report["operations"]), 10)
            self.assertEqual(report["committed_store"]["retained_node_block_bytes"], 0)
            self.assertEqual(store.stripes["five-round-test"]["blocks"], {})
            recovery_ops = [op for op in report["operations"] if op["phase"] == "recover"]
            self.assertEqual(recovery_ops[0]["recovery_input_node"], 1)
            self.assertEqual(recovery_ops[1]["recovery_input_node"], 0)
            saved = json.loads((artifacts / "five-round-test/raw.json").read_text())
            self.assertEqual(saved["state"], "PASS")

    def test_fixture_inputs_and_unsafe_output_location_rejected_before_connect(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for i in (0, 1):
                p = tmp / f"fixture{i}.bin"
                p.write_bytes(node_fixture(i))
                paths.append(p)
            called = []
            with self.assertRaises(ValueError):
                three_host.run(ROOT / "placement.json", paths, tmp / "outside",
                               stripe_id="fixture-reject", connect=lambda _: called.append(True))
            self.assertEqual(called, [])

    def test_distributed_source_worker_uses_local_survivor_only_five_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifacts = tmp / "artifacts"
            artifacts.mkdir()
            blocks = [self._real_format_test_block(i) for i in (0, 1)]
            config = json.loads((ROOT / "placement.json").read_text())
            config["max_bytes"] = 4 * max(map(len, blocks))
            store = StripeStore(ttl=60, max_stripes=8, max_bytes=config["max_bytes"])
            meta1 = descriptor(blocks[1])
            first, _ = store.dispatch(dict(op="register_node", stripe_id="distributed-test",
                                           node_id=1, descriptor=meta1, transport="tcp"), b"")
            path = tmp / "d0-node.bin"
            path.write_bytes(blocks[0])
            class AutoCommitClient(DirectClient):
                def call(self, op, payload=b"", **fields):
                    response, raw = super().call(op, payload, **fields)
                    if op == "submit_node_block" and fields.get("node_id") == 0:
                        key = binding(response)
                        super().call("submit_node_block", node_id=1, payload=blocks[1],
                                     transport="tcp", **key)
                    return response, raw
            peer = AutoCommitClient(store)
            with patch.object(source_trial, "ARTIFACTS", artifacts), \
                 patch.object(source_trial, "local_identity", return_value={}) as identity:
                report = source_trial.run(config, "D0", "232", path, "distributed-test",
                    artifacts / "D0-trial", connect=lambda _: peer)
            identity.assert_called_once_with(config, "D0", "232")
            self.assertEqual(first["state"], "REGISTERING")
            self.assertEqual(report["state"], "PASS")
            self.assertEqual(len(report["operations"]), 5)
            self.assertTrue(all(row["recovery_input_node"] == 0 and row["missing_node"] == 1
                                for row in report["operations"]))
            self.assertFalse(report["missing_original_read"])


if __name__ == "__main__":
    unittest.main()
