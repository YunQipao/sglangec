import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from experiments.kv_ec.multinode.coordinator import run


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TCPIntegrationTests(unittest.TestCase):
    def test_cpu_tcp_transport_and_committed_rs_verification(self):
        try:
            probe = socket.socket()
            probe.close()
        except PermissionError as exc:
            self.skipTest(f"sandbox denies local TCP sockets: {exc}")
        control_port, data_port = free_port(), free_port()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            worker_output = output / "worker"
            process = subprocess.Popen([
                sys.executable, "-m", "experiments.kv_ec.multinode.worker",
                "--transport", "tcp", "--control-host", "127.0.0.1",
                "--control-port", str(control_port), "--data-host", "127.0.0.1",
                "--data-port", str(data_port), "--d0-bytes", "16384",
                "--d1-bytes", "8192", "--return-host", "127.0.0.1",
                "--return-port", str(free_port()), "--output", str(worker_output)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(process.stderr.read().decode())
                    try:
                        with socket.create_connection(("127.0.0.1", control_port), timeout=.1):
                            break
                    except OSError:
                        time.sleep(.05)
                else:
                    self.fail("worker did not become ready")
                args = type("Args", (), {
                    "transport": "tcp", "node_b": "127.0.0.1",
                    "control_port": control_port, "data_port": data_port,
                    "return_port": int(process.args[process.args.index("--return-port") + 1]),
                    "return_bind_host": "127.0.0.1",
                    "d0_bytes": 16384, "d1_bytes": 8192, "seed": 20261025,
                    "rdma_device": "-", "chunk_id": "chunk-0",
                    "experiment_id": "synthetic-rs21-experiment", "stripe_id": "stripe-0",
                    "checkpoint_version": "checkpoint-v1", "timeout": 10,
                    "output": output / "coordinator"})()
                result = run(args)
                self.assertEqual(result["state"], "COMMITTED")
                self.assertEqual(result["transport"], "tcp")
                self.assertFalse(result["shared_node_failure_recovery"])
                self.assertTrue(all(result["verification"][key] for key in (
                    "D0_lost_byte_exact", "D1_lost_byte_exact", "P_recomputed_byte_exact")))
                self.assertTrue((output / "coordinator/result.csv").is_file())
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
