"""M3/M4 evidence/measurement tests; mock requests are never serving evidence."""

import argparse
import json
import os
import re
import runpy
import shlex
import warnings
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.kv_ec.rs_experiment import (
    process_usage,
    usage_delta,
    save_report,
    network_delta,
)
from experiments.kv_ec.serving_impact import sample_processes, compare, run


def response(rid):
    return dict(
        kind="real_server_artificial_prompt",
        request_id=rid,
        model_identifier="fixture",
        input_token_count=768,
        input_ids_sha256="same-input",
        sampling={"temperature": 0},
        output_ids=list(range(16)),
        finish_reason={"type": "length"},
        client_metrics_ms=dict(ttft_ms=1, tpot_ms=2, request_total_ms=31),
    )


class ExperimentTests(unittest.TestCase):
    def test_documented_cli_options_are_actual_argparse_options(self):
        doc = (
            Path(__file__)
            .with_name("DUAL_NODE_REMOTE_RS.md")
            .read_text()
            .replace("\\\n", " ")
        )
        parsers = {}

        class Parsed(Exception):
            pass

        def capture(parser, *args, **kwargs):
            parsers["current"] = parser
            raise Parsed

        def options(parser):
            result = set()
            for action in parser._actions:
                result.update(action.option_strings)
                if isinstance(action, argparse._SubParsersAction):
                    for sub in action.choices.values():
                        result.update(options(sub))
            return result

        checked = 0
        for line in doc.splitlines():
            match = re.search(
                r'"\$PY" -[mu] (?:-m )?(experiments\.kv_ec\.\w+)\s*(.*)', line
            )
            if not match:
                continue
            module, args = match.groups()
            with warnings.catch_warnings(), patch.object(argparse.ArgumentParser, "parse_args", capture):
                warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)
                with self.assertRaises(Parsed):
                    runpy.run_module(module, run_name="__main__")
            actual = options(parsers["current"])
            requested = {
                t.split("=", 1)[0] for t in shlex.split(args) if t.startswith("--")
            }
            self.assertFalse(requested - actual, (module, requested - actual))
            checked += 1
        self.assertGreater(checked, 20)

    def test_cpu_usage_and_scoped_process_sampling(self):
        before = dict(monotonic=1, user_cpu_s=1, system_cpu_s=2, rss_bytes=1024)
        after = dict(
            monotonic=3,
            user_cpu_s=2,
            system_cpu_s=2,
            rss_bytes=2048,
            peak_rss_bytes=4096,
        )
        self.assertEqual(usage_delta(before, after)["cpu_percent_one_core"], 50)
        self.assertIn(str(os.getpid()), sample_processes([os.getpid()]))
        self.assertGreater(process_usage()["rss_bytes"], 0)
        self.assertEqual(network_delta({}, {})["status"], "NOT VERIFIED")
        before = dict(interface="eth0", counters=dict(tx_bytes=1, rx_bytes=3))
        after = dict(interface="eth0", counters=dict(tx_bytes=5, rx_bytes=10))
        self.assertEqual(
            network_delta(before, after)["counter_delta"], dict(tx_bytes=4, rx_bytes=7)
        )

    def test_no_ec_mock_driver_writes_json_csv_and_null_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            args = SimpleNamespace(
                output=output,
                pids=[os.getpid()],
                gpu_indices="0,1",
                sample_interval=0.01,
                mode="no-ec",
                node0_url="mock0",
                node1_url="mock1",
                model="fixture",
                node0_request="r0",
                node1_request="r1",
            )

            def fake_request(args):
                with args.output.open("x") as stream:
                    json.dump(response(args.request_id), stream)

            with patch(
                "experiments.kv_ec.request_debug.request", side_effect=fake_request
            ), patch(
                "experiments.kv_ec.serving_impact.environment",
                return_value={"fixture": True},
            ), patch(
                "experiments.kv_ec.serving_impact.command",
                return_value={"status": "mock"},
            ):
                report = run(args)
            self.assertEqual(report["state"], "PASS")
            self.assertIsNone(report["measurements"][0]["unprotected_window_ms"])
            self.assertEqual(
                report["online_remote_serving_interference"], "NOT VERIFIED"
            )
            self.assertTrue((output / "result.csv").exists())

    def test_offline_driver_mock_end_to_end_never_claims_physical_network(self):
        from experiments.kv_ec import dual_node_rs
        from experiments.kv_ec.node_block_export import canonical
        from experiments.kv_ec.rs_stripe_builder import build
        from experiments.kv_ec.rs_fault_recovery import verify_faults
        from experiments.kv_ec.remote_rs_server import StripeStore
        from experiments.kv_ec.test_remote_rs import node_fixture

        network = dict(physical_device=False, interface="lo", counters=dict(tx_bytes=0, rx_bytes=0))
        class MockClient:
            def __init__(self, *args):
                self.store = StripeStore(allow_fixture=True)
                self.transfers = []
            def call(self, op, payload=b"", **fields):
                self.transfers.append(dict(op=op, peer=["127.0.0.1", 30300], local=["127.0.0.1", 40000],
                    roundtrip_ms=1., tx=dict(stream_bytes=len(payload)+100, payload_bytes=len(payload))))
                result, raw = self.store.dispatch(dict(op=op, **fields), payload)
                if op == "get_metrics":
                    result["network_B"] = network
                return result, raw
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in (0, 1):
                path = Path(tmp)/f"node{i}.bin"
                path.write_bytes(node_fixture(i))
                paths.append(path)
            args = SimpleNamespace(output=Path(tmp)/"result", remote_host="mock", remote_port=30300,
                timeout=5, chunk_tokens=4, stripe_id="mock-offline", network_profile="loopback-test",
                node0_block=paths[0], node1_block=paths[1])
            with patch.object(dual_node_rs, "RemoteRSClient", MockClient), \
                 patch.object(dual_node_rs, "environment", return_value={"mock": True}), \
                 patch.object(dual_node_rs, "network_observation", return_value=network), \
                 patch.object(dual_node_rs, "build", side_effect=lambda sid, b: build(sid, b, allow_fixture=True)), \
                 patch.object(dual_node_rs, "verify_faults", side_effect=lambda c, s, b: verify_faults(c, s, b, allow_fixture=True)):
                result = dual_node_rs.run(args)
            self.assertEqual(result["state"], "PASS")
            self.assertEqual(result["physical_network"], "NOT VERIFIED")
            self.assertEqual(result["GPU_pool_writeback"], "NOT IMPLEMENTED")
            self.assertEqual(len(result["fault_checks"]), 8)
            self.assertTrue((args.output/"result.json").exists())
            self.assertTrue((args.output/"result.csv").exists())

    def test_comparison_requires_exact_request_outputs_and_offline_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for mode in ("no-ec", "local-xor", "offline-remote-rs"):
                path = Path(tmp) / mode
                path.mkdir()
                save_report(path / "result", dict(state="PASS", mode=mode))
                for node in (0, 1):
                    with (path / f"node{node}-response.json").open("x") as f:
                        json.dump(response(f"node{node}"), f)
                paths.append(path)
            compare(paths, Path(tmp) / "good")
            record = json.loads((Path(tmp) / "good/comparison.json").read_text())
            self.assertEqual(
                record["online_remote_serving_interference"], "NOT VERIFIED"
            )
            bad = response("node1")
            bad["output_ids"][0] = 999
            with (paths[2] / "node1-response.json").open("w") as f:
                json.dump(bad, f)
            with self.assertRaises(AssertionError):
                compare(paths, Path(tmp) / "bad")


if __name__ == "__main__":
    unittest.main()
