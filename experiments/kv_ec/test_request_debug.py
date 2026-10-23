"""Response validation must fail before creating any success artifact."""

from copy import deepcopy
from contextlib import redirect_stdout
import http.client
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import urllib.error

from experiments.kv_ec.request_debug import compare, read_response, request, validate_result


def result():
    return {"output_ids": list(range(16)),
            "meta_info": {"finish_reason": {"type": "length"}, "completion_tokens": 16}}


def stream(*events, done=True):
    lines = [b"data: " + json.dumps(event).encode() + b"\n\n" for event in events]
    if done:
        lines.append(b"data: [DONE]\n\n")
    return io.BytesIO(b"".join(lines))


class RequestDebugTests(unittest.TestCase):
    def test_finish_reason_and_meta_info_are_checked(self):
        cases = [
            ({"output_ids": [1], "meta_info": None}, "invalid_meta_info"),
            ({"output_ids": [1], "meta_info": []}, "invalid_meta_info"),
            ({"output_ids": [1], "meta_info": {"finish_reason": None}}, "incomplete_response"),
            ({"output_ids": [1], "meta_info": {"finish_reason": []}}, "incomplete_response"),
            ({"output_ids": [1], "meta_info": {"finish_reason": {"type": "abort"}}},
             "request_aborted_or_failed"),
            ({"error": "server crashed"}, "server_error"),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                validate_result(response)

    def test_output_must_have_16_integer_tokens(self):
        self.assertEqual(validate_result(result()), result())
        for ids in ([], [1], [1] * 17, ["1"] * 16, [True] * 16):
            response = result()
            response["output_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_result(response)
        response = result()
        response["meta_info"]["completion_tokens"] = 15
        with self.assertRaisesRegex(ValueError, "completion_tokens"):
            validate_result(response)

    def test_stream_requires_done_and_checks_terminal_event(self):
        partial = deepcopy(result())
        partial["meta_info"]["finish_reason"] = None
        with self.assertRaisesRegex(ValueError, "incomplete_stream"):
            read_response(stream(partial, done=False), True, Mock())
        response = read_response(stream(partial), True, Mock())
        with self.assertRaisesRegex(ValueError, "incomplete_response"):
            validate_result(response)
        terminal = {"meta_info": {"finish_reason": {"type": "abort"}}}
        response = read_response(stream(result(), terminal), True, Mock())
        with self.assertRaisesRegex(ValueError, "request_aborted"):
            validate_result(response)
        with self.assertRaisesRegex(ValueError, "server_stream_error"):
            read_response(stream(result(), {"error": "crashed"}), True, Mock())
        self.assertEqual(read_response(stream(partial, result()), True, Mock()), result())

    def test_failed_request_never_writes_success_record(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "response.json"
                response = result()
                response["meta_info"]["finish_reason"] = None
                body = stream(response) if streaming else io.BytesIO(json.dumps(response).encode())
                opener = Mock()
                opener.open.return_value = body
                tokenizer = NS(encode=lambda *args, **kwargs: [1] * 768)
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
                        "experiments.kv_ec.request_debug.urllib.request.build_opener",
                        return_value=opener):
                    args = NS(model="unused", request_id="artificial", prompt_tokens=768,
                              stream_metrics=streaming, url="http://127.0.0.1:1", output=output)
                    with self.assertRaisesRegex(ValueError, "incomplete_response"):
                        request(args)
                self.assertFalse(output.exists())

    def test_transport_failures_are_explicit_and_never_saved(self):
        errors = [
            (urllib.error.HTTPError("http://localhost", 500, "failed", {}, None), "server_http_error"),
            (urllib.error.URLError("connection refused"), "server_connection_failed"),
            (http.client.IncompleteRead(b"partial"), "server_connection_failed"),
        ]
        for error, reason in errors:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "response.json"
                opener = Mock()
                opener.open.side_effect = error
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=NS(
                        encode=lambda *args, **kwargs: [1] * 768)), patch(
                        "experiments.kv_ec.request_debug.urllib.request.build_opener",
                        return_value=opener):
                    args = NS(model="unused", request_id="artificial", prompt_tokens=768,
                              stream_metrics=True, url="http://127.0.0.1:1", output=output)
                    with self.assertRaisesRegex(ValueError, reason):
                        request(args)
                self.assertFalse(output.exists())

    def test_complete_responses_are_saved_and_comparable(self):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for streaming in (False, True):
                output = Path(root) / f"response-{streaming}.json"
                paths.append(output)
                body = stream(result()) if streaming else io.BytesIO(json.dumps(result()).encode())
                opener = Mock()
                opener.open.return_value = body
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=NS(
                        encode=lambda *args, **kwargs: [1] * 768)), patch(
                        "experiments.kv_ec.request_debug.urllib.request.build_opener",
                        return_value=opener), redirect_stdout(io.StringIO()):
                    request(NS(model="unused", request_id="artificial", prompt_tokens=768,
                               stream_metrics=streaming, url="http://127.0.0.1:1", output=output))
                record = json.loads(output.read_text())
                self.assertEqual(len(record["output_ids"]), 16)
                self.assertTrue(record["sampling"]["ignore_eos"])
            with redirect_stdout(io.StringIO()):
                compare(paths)


if __name__ == "__main__":
    unittest.main()
