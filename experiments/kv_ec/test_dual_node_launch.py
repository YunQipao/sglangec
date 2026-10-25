"""M1 compatibility/port isolation tests; no model startup."""

import ast
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from experiments.kv_ec.launch_debug import check_port
from experiments.kv_ec.dual_node_check import check


class LaunchTests(unittest.TestCase):
    def test_host_default_and_memory_fraction_preserved(self):
        tree = ast.parse(Path(__file__).with_name("launch_debug.py").read_text())
        defaults = {
            n.args[0].value: next(
                (k.value.value for k in n.keywords if k.arg == "default"), None
            )
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument"
            and n.args
            and isinstance(n.args[0], ast.Constant)
        }
        self.assertEqual(defaults["--host"], "127.0.0.1")
        self.assertEqual(defaults["--mem-fraction-static"], 0.60)
        self.assertEqual(defaults["--port"], 30281)

    def test_port_probe_uses_requested_host(self):
        with patch("experiments.kv_ec.launch_debug.socket.socket") as mock:
            self.assertEqual(check_port(30294, "0.0.0.0"), "available")
            mock.return_value.__enter__.return_value.bind.assert_called_with(
                ("0.0.0.0", 30294)
            )

    def test_reject_shared_manager_before_io(self):
        with self.assertRaises(ValueError):
            check(
                [
                    dict(socket="same", url=str(i), snapshot=str(i), request=str(i))
                    for i in range(2)
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
