"""No GPU subprocesses: test ordered fail-stop orchestration only."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.kv_ec import accept_m3


class M3MatrixTests(unittest.TestCase):
    def run_matrix(self, tmp, *, cpu_failed=False, failed_case=None, smoke=False):
        calls = []
        root = Path(tmp) / "matrix"
        def execute(command, **kwargs):
            calls.append(command)
            if "unittest" in command:
                return NS(returncode=int(cpu_failed))
            rank = int(command[command.index("--lost-rank") + 1])
            tokens = int(command[command.index("--recover-tokens") + 1])
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            failed = (rank, tokens) == failed_case
            (output / "acceptance.json").write_text(json.dumps(
                dict(M3_PASS=not failed, reason="injected failure" if failed else "test", chunks=[])))
            return NS(returncode=int(failed))
        argv = ["accept_m3", "--gpus", "0,1,2,3", "--shadow-evidence", "/prior-shadow",
                "--output", str(root)]
        if smoke:
            argv.append("--smoke")
        with patch("sys.argv", argv), patch.object(accept_m3.subprocess, "run", side_effect=execute), \
                redirect_stdout(io.StringIO()):
            code = accept_m3.main()
        return code, json.loads((root / "matrix.json").read_text()), calls

    def test_cpu_failure_starts_no_gpu_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, verdict, calls = self.run_matrix(tmp, cpu_failed=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(c["state"] == "NOT_RUN" for c in verdict["cases"]))

    def test_failure_in_b_never_runs_c_or_remaining_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, verdict, calls = self.run_matrix(tmp, failed_case=(1, 256))
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual([c["state"] for c in verdict["cases"][:3]], ["PASS", "FAIL", "NOT_RUN"])
        self.assertTrue(all(c["state"] == "NOT_RUN" for c in verdict["cases"][2:]))

    def test_full_order_and_smoke(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as tmp:
                code, verdict, calls = self.run_matrix(tmp, smoke=smoke)
                self.assertEqual(code, 0)
                self.assertTrue(verdict["M3_PASS"])
                expected = accept_m3.ORDERED_CASES[:1] if smoke else accept_m3.ORDERED_CASES
                self.assertEqual([(c["lost_rank"], c["recover_tokens"]) for c in verdict["cases"]], list(expected))
                self.assertEqual(len(calls), len(expected) + 1)


if __name__ == "__main__":
    unittest.main()
