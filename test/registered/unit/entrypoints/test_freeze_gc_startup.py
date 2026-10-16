"""CPU-only tests of the actual startup helper, without model/CUDA imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests


class TestFreezeGCStartup(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        source = root / "python/sglang/srt/entrypoints/http_server.py"
        function = next(
            node for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_freeze_gc_after_server_warmup"
        )
        self.clock = 0.0
        self.time = SimpleNamespace(monotonic=lambda: self.clock, sleep=self.advance)
        self.logger = Mock()
        namespace = {"requests": requests, "time": self.time, "logger": self.logger,
                     "ServerArgs": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        self.freeze = namespace[function.name]
        self.args = SimpleNamespace(
            admin_api_key="admin", api_key="user",
            url=lambda: "http://127.0.0.1:30286", ssl_verify=lambda: True,
        )
        self.session = Mock()
        self.session.__enter__ = Mock(return_value=self.session)
        self.session.__exit__ = Mock(return_value=False)
        self.patcher = patch.object(requests, "Session", return_value=self.session)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def advance(self, seconds):
        self.clock += seconds

    def test_connection_refused_then_ready(self):
        response = Mock()
        self.session.post.side_effect = [requests.ConnectionError("refused"), response]
        self.freeze(self.args)
        self.assertEqual(self.session.post.call_count, 2)
        self.assertFalse(self.session.trust_env)
        response.raise_for_status.assert_called_once()
        self.logger.warning.assert_not_called()
        self.session.post.assert_called_with(
            "http://127.0.0.1:30286/freeze_gc",
            headers={"Authorization": "Bearer admin"}, timeout=9.9, verify=True,
        )

    def test_deadline_failure_only_warns_and_returns(self):
        self.session.post.side_effect = requests.ConnectionError("refused")
        self.freeze(self.args)
        self.assertGreaterEqual(self.clock, 10)
        self.assertLessEqual(self.session.post.call_count, 102)
        self.logger.warning.assert_called_once_with(
            "post-warmup freeze_gc failed", exc_info=True
        )

    def test_http_error_timeout_and_ssl_are_not_retried(self):
        for error in (requests.HTTPError("403"), requests.ReadTimeout(),
                      requests.ConnectTimeout(), requests.exceptions.SSLError()):
            with self.subTest(error=type(error).__name__):
                self.session.post.reset_mock()
                self.logger.reset_mock()
                self.session.post.side_effect = error
                self.freeze(self.args)
                self.session.post.assert_called_once()
                self.logger.warning.assert_called_once()

    def test_api_key_fallback_and_anonymous(self):
        self.session.post.side_effect = None
        self.args.admin_api_key = None
        for key in ("user", None):
            self.args.api_key = key
            self.freeze(self.args)
            expected = {"Authorization": "Bearer user"} if key else {}
            self.assertEqual(self.session.post.call_args.kwargs["headers"], expected)

    def test_unsuccessful_response_only_warns(self):
        self.session.post.return_value.raise_for_status.side_effect = requests.HTTPError("403")
        self.freeze(self.args)
        self.session.post.assert_called_once()
        self.logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
