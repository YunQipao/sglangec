import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from experiments.kv_ec.multinode import coordinator
from experiments.kv_ec.multinode.coordinator import request_json, run


def free_port(exclude=()):
    excluded = set(exclude)
    for _ in range(20):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in excluded:
            return port
    raise RuntimeError("could not allocate a unique test port")


def tail(path, limit=8192):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode(errors="replace")
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"


def stop_process(process):
    """Stop only the worker created by this test and reap it."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class TCPIntegrationTests(unittest.TestCase):
    def test_cpu_tcp_transport_and_committed_rs_verification(self):
        self.run_tcp_case(16384, 8192)

    def test_cpu_tcp_transport_with_larger_d1(self):
        self.run_tcp_case(8192, 16384)

    def run_tcp_case(self, d0_bytes, d1_bytes):
        control_port = free_port()
        data_port = free_port((control_port,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            worker_output = output / "worker"
            return_port = free_port((control_port, data_port))
            stdout_path = output / "worker.stdout.log"
            stderr_path = output / "worker.stderr.log"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen([
                    sys.executable, "-u", "-m", "experiments.kv_ec.multinode.worker",
                    "--transport", "tcp", "--control-host", "127.0.0.1",
                    "--control-port", str(control_port), "--data-host", "127.0.0.1",
                    "--data-port", str(data_port), "--d0-bytes", str(d0_bytes),
                    "--d1-bytes", str(d1_bytes), "--return-host", "127.0.0.1",
                    "--return-port", str(return_port), "--timeout", "10",
                    "--output", str(worker_output)],
                    stdout=stdout, stderr=stderr)
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            self.fail(
                                f"worker exited with code {process.returncode}; "
                                f"stdout tail:\n{tail(stdout_path)}\n"
                                f"stderr tail:\n{tail(stderr_path)}")
                        try:
                            with socket.create_connection(("127.0.0.1", control_port), timeout=.1):
                                break
                        except OSError:
                            time.sleep(.05)
                    else:
                        self.fail(
                            f"worker did not become ready (exit={process.poll()}); "
                            f"stdout tail:\n{tail(stdout_path)}\n"
                            f"stderr tail:\n{tail(stderr_path)}")
                    args = type("Args", (), {
                        "transport": "tcp", "node_b": "127.0.0.1",
                        "control_port": control_port, "data_port": data_port,
                        "return_port": return_port,
                        "return_bind_host": "127.0.0.1",
                        "d0_bytes": d0_bytes, "d1_bytes": d1_bytes, "seed": 20261025,
                        "rdma_device": "-", "chunk_id": "chunk-0",
                        "experiment_id": "synthetic-rs21-experiment", "stripe_id": "stripe-0",
                        "checkpoint_version": "checkpoint-v1", "timeout": 10,
                        "output": output / "coordinator"})()
                    acknowledge = coordinator._acknowledge_commit

                    def check_then_ack(args):
                        # Delay the final query deliberately: the worker must
                        # remain available after durable COMMITTED publication.
                        time.sleep(.1)
                        self.assertIsNone(process.poll())
                        committed = request_json("127.0.0.1", control_port, "result")
                        self.assertEqual(committed["state"], "COMMITTED")
                        self.assertEqual(committed["acknowledgement"], "WAITING")
                        binding = {key: getattr(args, key) for key in
                                   ("experiment_id", "stripe_id", "checkpoint_version")}
                        for key in binding:
                            rejected = request_json("127.0.0.1", control_port,
                                                    {"op": "ack", **binding, key: "wrong-run"})
                            self.assertEqual(rejected["reason"], "ack_identity_mismatch")
                        acknowledge(args)

                    try:
                        with mock.patch.object(coordinator, "_acknowledge_commit", side_effect=check_then_ack):
                            result = run(args)
                    except Exception as exc:
                        self.fail(
                            f"coordinator failed: {exc}\n"
                            f"worker exit={process.poll()}\n"
                            f"stdout tail:\n{tail(stdout_path)}\n"
                            f"stderr tail:\n{tail(stderr_path)}")
                    self.assertEqual(result["state"], "COMMITTED")
                    self.assertEqual(result["acknowledgement"], "ACKNOWLEDGED")
                    self.assertEqual(result["transport"], "tcp")
                    self.assertFalse(result["shared_node_failure_recovery"])
                    self.assertTrue(all(result["verification"][key] for key in (
                        "D0_lost_byte_exact", "D1_lost_byte_exact", "P_recomputed_byte_exact")))
                    self.assertTrue((output / "coordinator/result.csv").is_file())
                    process.wait(timeout=5)
                    self.assertEqual(process.returncode, 0, tail(stderr_path))
                    worker_result = json.loads((worker_output / "result.json").read_text())
                    receipt = json.loads((worker_output / "acknowledgement.json").read_text())
                    self.assertEqual(worker_result["state"], "COMMITTED")
                    self.assertEqual(receipt["state"], "ACKNOWLEDGED")
                    for key in ("experiment_id", "stripe_id", "checkpoint_version"):
                        self.assertEqual(receipt[key], getattr(args, key))
                    for port in (control_port, data_port, return_port):
                        with self.assertRaises(ConnectionRefusedError):
                            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                                pass
                finally:
                    stop_process(process)


if __name__ == "__main__":
    unittest.main()
