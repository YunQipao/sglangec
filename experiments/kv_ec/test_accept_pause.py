"""CPU checks for the acceptance verifier, not GPU experiment evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import hashlib
import errno
import json
from pathlib import Path
import subprocess
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from experiments.kv_ec.accept_pause import check_evidence, main, resolve_ports, verify_run
from experiments.kv_ec.launch_debug import check_port


IDENTITY = {"transaction_id": "test", "epoch": "test-epoch"}


def rank_record(rank):
    return dict(
        tp_rank=rank, state="RESUMED", transaction_id="test", coordinator_epoch="test-epoch",
        decode_observed=True, decode_forward_mode="decode", decode_output_count=2,
        scheduler_skipped_iterations=1, resume_authorized=True, tp_cpu_resume_vote=True,
        allocator_lease_released=True, row_protection_released=True,
        active_slot_leases_after_release=0, cuda_event_complete=True,
        cuda_event_recorded=True, online_kv_pool_modified=False, failure=None,
        request_generation_before=1, request_generation_after=1,
        slot_generations_before=[1] * 768, slot_generations_after=[1] * 768,
        pause_ms=1, token_count=768,
        request_hash=hashlib.sha256(b"artificial").hexdigest(),
        coordinator_final=dict(IDENTITY, state="RESUME_AUTHORIZED", paused_ranks=list(range(4)),
                               validated_ranks=list(range(4)), released_ranks=list(range(4)),
                               armed_ranks=list(range(4))))


def completed_run(root):
    (root / "ranks").mkdir()
    (root / "identity.json").write_text(json.dumps(IDENTITY))
    for rank in range(4):
        (root / "ranks" / f"pause-rank-{rank}.json").write_text(json.dumps(rank_record(rank)))
    (root / "server.log").write_text("server started\n")
    events = [dict(op=op, response={"state": "OK", "status": dict(
        IDENTITY, state=state, paused_ranks=list(range(4)))})
        for op, state in (("paused", "PAUSED"), ("authorize", "RESUME_REQUESTED"),
                          ("authorize", "RESUME_AUTHORIZED"))]
    (root / "coordinator.log").write_text("".join(
        "INFO:module:KV_EC_PAUSE_CONTROL " + json.dumps(event) + "\n" for event in events))
    response = dict(kind="real_server_artificial_prompt", request_id="artificial",
                    model_identifier="fixture", input_ids_sha256="fixture", input_token_count=768,
                    sampling={"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
                    output_ids=list(range(16)), finish_reason={"type": "length", "length": 16})
    for name in ("baseline-response.json", "pause-response.json"):
        (root / name).write_text(json.dumps(response))
    return root / "baseline-response.json", root / "pause-response.json"


class AcceptanceTests(unittest.TestCase):
    def test_distinct_ports_defaults_alias_and_explicit_values(self):
        self.assertEqual(resolve_ports(), (30359, 30360))
        self.assertEqual(resolve_ports(30400), (30400, 30401))
        self.assertEqual(resolve_ports(None, 31000, 32000), (31000, 32000))
        for args in ((None, 30359, 30359), (30359, 30400, None), (65535, None, None),
                     (None, 0, 30360)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_ports(*args)

    def test_reuseaddr_probe_preserves_errors_and_does_not_reuseport(self):
        first, second, connection = [MagicMock() for _ in range(3)]
        for probe in (first, second, connection):
            probe.__enter__.return_value = probe
        first.bind.side_effect = OSError(errno.EADDRINUSE, "plain bind busy")
        with patch("experiments.kv_ec.launch_debug.socket.socket", side_effect=[first, second]):
            self.assertIn("TIME_WAIT", check_port(30359))
        second.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        second.listen.assert_called_once()
        second.bind.side_effect = OSError(errno.EADDRINUSE, "reuse bind busy")
        connection.connect_ex.return_value = 0
        with patch("experiments.kv_ec.launch_debug.socket.socket", side_effect=[first, second, connection]):
            with self.assertRaisesRegex(OSError, "active_listener.*plain bind busy.*reuse bind busy") as raised:
                check_port(30359)
        self.assertEqual(raised.exception.errno, errno.EADDRINUSE)
        self.assertIs(raised.exception.__cause__, second.bind.side_effect)
        connection.connect_ex.return_value = errno.ECONNREFUSED
        with patch("experiments.kv_ec.launch_debug.socket.socket", side_effect=[first, second, connection]):
            with self.assertRaisesRegex(OSError, "non_listener_or_nonreusable_TIME_WAIT"):
                check_port(30359)
        first.bind.side_effect = OSError(errno.EACCES, "permission denied")
        with patch("experiments.kv_ec.launch_debug.socket.socket", return_value=first):
            with self.assertRaisesRegex(OSError, "socket_error.*permission denied") as raised:
                check_port(30359)
        self.assertEqual(raised.exception.errno, errno.EACCES)

    def test_real_listener_is_rejected_even_with_reuseaddr(self):
        try:
            listener = socket.socket()
        except PermissionError:
            self.skipTest("sandbox disallows TCP sockets")
        with listener:
            try:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen(4)
            except PermissionError:
                self.skipTest("sandbox disallows TCP listeners")
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(OSError, "active_listener") as raised:
                check_port(port)
            self.assertEqual(raised.exception.errno, errno.EADDRINUSE)
            self.assertGreaterEqual(listener.fileno(), 0)

    def test_both_ports_checked_before_spawn_and_conflict_kills_nobody(self):
        for conflict in ("baseline", "pause", "same"):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as tmp, patch(
                    "sys.argv", ["accept_pause", "--gpus", "0,1,2,3", "--baseline-port", "30359",
                                 "--pause-port", "30359" if conflict == "same" else "30360"]), patch(
                    "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output", side_effect=[
                        "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n", ""]), patch(
                    "torch.cuda.is_available", return_value=True), patch(
                    "experiments.kv_ec.accept_pause.check_port", side_effect=(
                        OSError(errno.EADDRINUSE, "active_listener 30359") if conflict == "baseline" else
                        ["available", OSError(errno.EADDRINUSE, "active_listener 30360")])) as check, patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as spawn, patch(
                    "experiments.kv_ec.accept_pause.os.killpg") as kill:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 1)
                spawn.assert_not_called()
                kill.assert_not_called()
                reason = json.loads((Path(tmp) / "acceptance.json").read_text())["reason"]
                if conflict == "same":
                    check.assert_not_called()
                    self.assertIn("must be different", reason)
                else:
                    self.assertIn("Errno 98", reason)
                    self.assertIn("active_listener", reason)
                    self.assertEqual([c.args[0] for c in check.call_args_list],
                                     [30359] if conflict == "baseline" else [30359, 30360])

    def test_server_and_client_urls_use_their_own_ports(self):
        # Mock subprocesses test orchestration only, not GPU acceptance.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processes, commands = [], []
            def spawn(argv, **kwargs):
                commands.append(argv)
                process = MagicMock()
                process.poll.return_value = None
                process.wait.return_value = 0
                process.pid = 90000 + len(processes)
                processes.append(process)
                if "experiments.kv_ec.pause_control" in argv:
                    (root / "identity.json").write_text(json.dumps(IDENTITY))
                    (root / "pause.sock").touch()
                if "experiments.kv_ec.request_debug" in argv and "pause-response.json" in argv[-1]:
                    (root / "ranks").mkdir()
                    for rank in range(4):
                        (root / "ranks" / f"pause-rank-{rank}.json").write_text("{}")
                return process
            response = MagicMock()
            response.__enter__.return_value.status = 200
            opener = MagicMock()
            opener.open.return_value = response
            with patch("sys.argv", ["accept_pause", "--gpus", "0,1,2,3", "--port", "31000",
                                     "--pause-port", "32000"]), patch(
                    "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output", side_effect=[
                        "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n", ""]), patch(
                    "torch.cuda.is_available", return_value=True), patch(
                    "experiments.kv_ec.accept_pause.check_port", return_value="available"), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen", side_effect=spawn), patch(
                    "experiments.kv_ec.accept_pause.stop_owned") as stop, patch(
                    "experiments.kv_ec.accept_pause.urllib.request.build_opener", return_value=opener), patch(
                    "experiments.kv_ec.accept_pause.control_request", return_value={"state": "OK", "status": {
                        "state": "PAUSED", "paused_ranks": [0, 1, 2, 3]}}), patch(
                    "experiments.kv_ec.accept_pause.check_evidence"), patch(
                    "experiments.kv_ec.accept_pause.compare"), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            servers = [argv for argv in commands if "experiments.kv_ec.launch_debug" in argv]
            clients = [argv for argv in commands if "experiments.kv_ec.request_debug" in argv]
            self.assertEqual([argv[argv.index("--port") + 1] for argv in servers], ["31000", "32000"])
            self.assertEqual([argv[argv.index("--url") + 1] for argv in clients],
                             ["http://127.0.0.1:31000", "http://127.0.0.1:32000"])
            self.assertEqual([c.args[0] for c in opener.open.call_args_list], [
                "http://127.0.0.1:31000/health", "http://127.0.0.1:32000/health",
                "http://127.0.0.1:32000/health"])
            self.assertEqual(stop.call_args_list[0].args, (processes[0],))
            self.assertTrue(all(c.args[0] in processes for c in stop.call_args_list))

    def test_shadow_pause_startup_failure_retains_errno_and_only_cleans_owned_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands, processes = [], []
            def spawn(argv, **kwargs):
                commands.append(argv)
                process = MagicMock(pid=91000 + len(processes))
                process.wait.return_value = 0
                process.poll.return_value = None
                processes.append(process)
                if "experiments.kv_ec.pause_control" in argv:
                    (root / "identity.json").write_text(json.dumps(IDENTITY))
                    (root / "pause.sock").touch()
                if "experiments.kv_ec.parity_manager" in argv:
                    (root / "parity.sock").touch()
                if "experiments.kv_ec.launch_debug" in argv and "pause-server.log" in kwargs["stdout"].name:
                    kwargs["stdout"].write("launch_debug.py: error: [Errno 98] active_listener 30360\n")
                    kwargs["stdout"].flush()
                    process.poll.return_value = 2
                    process.returncode = 2
                return process
            opener = MagicMock()
            opener.open.return_value.__enter__.return_value.status = 200
            with patch("sys.argv", ["accept_pause", "--shadow", "--gpus", "0,1,2,3"]), patch(
                    "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output", side_effect=[
                        "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n", ""]), patch(
                    "torch.cuda.is_available", return_value=True), patch(
                    "experiments.kv_ec.accept_pause.check_port", return_value="available"), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen", side_effect=spawn), patch(
                    "experiments.kv_ec.accept_pause.stop_owned") as stop, patch(
                    "experiments.kv_ec.accept_pause.os.killpg") as kill, patch(
                    "experiments.kv_ec.accept_pause.urllib.request.build_opener", return_value=opener), patch(
                    "experiments.kv_ec.parity_manager.manager_hello", return_value="manager-epoch"), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            verdict = json.loads((root / "acceptance.json").read_text())
            self.assertIn("pause server exited (2)", verdict["reason"])
            self.assertIn("Errno 98", verdict["reason"])
            self.assertFalse(verdict["SHADOW_PASS"])
            self.assertFalse(verdict["M2_PASS"])
            self.assertEqual((verdict["baseline_port"], verdict["pause_port"]), (30359, 30360))
            self.assertTrue(any("experiments.kv_ec.parity_manager" in argv for argv in commands))
            self.assertTrue(any("--require-shadow" in argv for argv in commands))
            self.assertEqual(len(stop.call_args_list), len(processes))
            self.assertTrue(all(c.args[0] in processes for c in stop.call_args_list))
            kill.assert_not_called()

    def test_incomplete_or_failed_rank_evidence_is_rejected(self):
        identity = IDENTITY
        base = rank_record(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank in range(4):
                (root / f"pause-rank-{rank}.json").write_text(json.dumps(dict(base, tp_rank=rank)))
            check_evidence(root, identity)
            for key, value in (
                ("decode_observed", False), ("state", "PAUSED"), ("row_protection_released", False),
                ("active_slot_leases_after_release", 1), ("online_kv_pool_modified", True),
                ("slot_generations_after", [2]), ("transaction_id", "stale"),
                ("decode_output_count", 1), ("tp_cpu_resume_vote", 1),
            ):
                with self.subTest(key=key):
                    record = dict(deepcopy(base), tp_rank=0)
                    record[key] = value
                    (root / "pause-rank-0.json").write_text(json.dumps(record))
                    with self.assertRaises(ValueError):
                        check_evidence(root, identity)
            (root / "pause-rank-0.json").write_text(json.dumps(dict(base, tp_rank=0)))
            (root / "pause-rank-0.error.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "failure"):
                check_evidence(root, identity)

    def test_unavailable_gpu_reports_fail_without_starting_processes(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "sys.argv", ["accept_pause", "--gpus", "0,1,2,3"]), patch(
                    "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output",
                    side_effect=subprocess.CalledProcessError(9, "nvidia-smi", output="driver unavailable")), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as popen:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(), 1)
            popen.assert_not_called()
            self.assertIn("FAIL:", output.getvalue())
            verdict = json.loads((Path(tmp) / "acceptance.json").read_text())
            self.assertFalse(verdict["M1"])
            self.assertIn("driver unavailable", verdict["reason"])

    def test_busy_gpu_never_starts_or_stops_existing_processes(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "sys.argv", ["accept_pause", "--gpus", "0,1,2,3"]), patch(
                    "experiments.kv_ec.accept_pause.tempfile.mkdtemp", return_value=tmp), patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output", side_effect=[
                        "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n", "GPU-2, 1234\n"]), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as popen, patch(
                    "experiments.kv_ec.accept_pause.os.killpg") as killpg:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            popen.assert_not_called()
            killpg.assert_not_called()
            verdict = json.loads((Path(tmp) / "acceptance.json").read_text())
            self.assertFalse(verdict["M1"])
            self.assertIn("active compute tasks", verdict["reason"])

    def test_read_only_audit_accepts_complete_matching_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, paused = completed_run(root)
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            verdict = verify_run(root, baseline, paused)
            self.assertTrue(verdict["M1"], verdict)
            self.assertTrue(verdict["checks"]["output_tokens_identical"])
            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_identical_tokens_do_not_allow_sampling_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, paused = completed_run(root)
            record = json.loads(baseline.read_text())
            del record["sampling"]["ignore_eos"]
            baseline.write_text(json.dumps(record))
            verdict = verify_run(root, baseline, paused)
            self.assertFalse(verdict["M1"])
            self.assertTrue(verdict["checks"]["output_tokens_identical"])
            self.assertIn("mismatch: sampling", verdict["reason"])

    def test_missing_authorization_allocator_error_and_unbound_response_reject(self):
        for failure in ("authorization", "allocator", "request", "identity", "same_file"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline, paused = completed_run(root)
                if failure == "authorization":
                    (root / "coordinator.log").write_text("")
                elif failure == "allocator":
                    (root / "server.log").write_text("RuntimeError: protected_slot_free\n")
                elif failure == "request":
                    record = json.loads(paused.read_text())
                    record["request_id"] = "different"
                    paused.write_text(json.dumps(record))
                elif failure == "identity":
                    (root / "identity.json").write_text(json.dumps(dict(IDENTITY, epoch="restarted")))
                else:
                    baseline = paused
                verdict = verify_run(root, baseline, paused)
                self.assertFalse(verdict["M1"], verdict)
                self.assertEqual(verdict["state"], "FAIL")

    def test_offline_cli_never_checks_or_starts_gpu_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed_run(root)
            with patch("sys.argv", ["accept_pause", "--verify-run", tmp,
                                    "--report", str(root / "audit.json")]), patch(
                    "experiments.kv_ec.accept_pause.subprocess.Popen") as popen, patch(
                    "experiments.kv_ec.accept_pause.subprocess.check_output") as check_output:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
                popen.assert_not_called()
                check_output.assert_not_called()
                self.assertTrue(json.loads((root / "audit.json").read_text())["M1"])


if __name__ == "__main__":
    unittest.main()
