"""Artificial non-sensitive request and exact output-token regression comparison."""

import argparse
import hashlib
import http.client
import json
import time
from pathlib import Path
import urllib.error
import urllib.request


def validate_result(result, expected_tokens=16):
    if not isinstance(result, dict):
        raise ValueError("invalid_server_response: expected a JSON object")
    if "error" in result:
        raise ValueError("server_error: request failed; inspect the isolated server log")
    meta = result.get("meta_info")
    if not isinstance(meta, dict):
        raise ValueError("invalid_meta_info: expected a dict; request completion is unproven")
    reason = meta.get("finish_reason")
    if not isinstance(reason, dict):
        raise ValueError(
            "incomplete_response: finish_reason is missing/null or not a dict; "
            "server may have crashed or the stream was interrupted")
    if reason.get("type") not in ("length", "stop"):
        raise ValueError(f"request_aborted_or_failed: finish_type={reason.get('type')!r}")
    ids = result.get("output_ids")
    if (not isinstance(ids, list) or not ids
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("invalid_output_ids: expected nonempty integer token IDs")
    if len(ids) != expected_tokens:
        raise ValueError(
            f"incomplete_output: expected {expected_tokens} tokens, got {len(ids)}")
    if meta.get("completion_tokens", expected_tokens) != len(ids):
        raise ValueError("incomplete_output: completion_tokens disagrees with output_ids")
    return result


def read_response(response, streaming, on_output):
    if not streaming:
        return json.load(response)
    result = None
    done = False
    for line in response:
        if not line.startswith(b"data:"):
            continue
        raw = line[5:].strip()
        if raw == b"[DONE]":
            done = True
            break
        event = json.loads(raw)
        if not isinstance(event, dict) or "error" in event:
            raise ValueError("server_stream_error: invalid/error event; inspect server log")
        # Keep the final event even when it reports an abort without token IDs.
        result = event
        if event.get("output_ids"):
            on_output()
    if not done:
        raise ValueError("incomplete_stream: missing [DONE]; server crashed or connection closed")
    if result is None:
        raise ValueError("incomplete_stream: no generation events")
    return result


def request(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # Use the same fixed token sequence for 128-token and 256-token captures.
    text = "KV cache research with artificial public test data. " * 160
    prompt_tokens = getattr(args, "prompt_tokens", 384)
    ids = tokenizer.encode(text, add_special_tokens=False)[:prompt_tokens]
    if len(ids) < prompt_tokens:
        raise ValueError(f"artificial prompt has fewer than {prompt_tokens} tokens")
    streaming = getattr(args, "stream_metrics", False)
    data = {"rid": args.request_id, "input_ids": ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16,
                                "ignore_eos": True}, "stream": streaming}
    # No proxy for local traffic. Only generated response tokens, never prompt text,
    # are saved. Request IDs here are deliberately artificial.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(args.url.rstrip("/") + "/generate",
                                 data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first_token = None
    def on_output():
        nonlocal first_token
        if first_token is None:
            first_token = time.perf_counter()
    try:
        with opener.open(req, timeout=600) as response:
            result = read_response(response, streaming, on_output)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"server_http_error: status={exc.code}; inspect server log") from exc
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        raise ValueError(
            f"server_connection_failed: {type(exc).__name__}; server may have crashed") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_server_json: response was malformed or truncated") from exc
    finished = time.perf_counter()
    validate_result(result)
    import numpy as np

    record = {"kind": "real_server_artificial_prompt", "request_id": args.request_id,
              "model_identifier": args.model, "input_token_count": len(ids),
              "input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
              "input_chunks": [
                  {"token_start": start, "token_count": len(ids[start:start + 256]),
                   "token_ids_sha256": hashlib.sha256(
                       np.asarray(ids[start:start + 256], dtype="<i8").tobytes()
                   ).hexdigest()}
                  for start in range(0, len(ids), 256)
              ],
              "output_ids": result["output_ids"], "sampling": data["sampling_params"],
              "finish_reason": result["meta_info"]["finish_reason"]}
    record["client_metrics_ms"] = {
        "request_total_ms": (finished-started)*1000,
        "ttft_ms": (first_token-started)*1000 if first_token else None,
        "tpot_ms": (finished-first_token)*1000/(len(result["output_ids"])-1)
            if first_token and len(result["output_ids"]) > 1 else None,
        "note": "client streaming TTFT and amortized TPOT, includes HTTP/SSE overhead",
    }
    with args.output.open("x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({"saved": str(args.output), "input_token_count": len(ids),
                      "output_token_count": len(record["output_ids"])}))


def compare(paths):
    first, second = [json.loads(p.read_text()) for p in paths]
    for record in (first, second):
        if record.get("kind") != "real_server_artificial_prompt":
            raise ValueError("comparison requires successful real-server response records")
        validate_result({"output_ids": record.get("output_ids"),
                         "meta_info": {"finish_reason": record.get("finish_reason")}})
    keys = ("model_identifier", "input_ids_sha256", "input_token_count", "sampling", "output_ids", "finish_reason")
    if any(first[k] != second[k] for k in keys):
        raise AssertionError("baseline/capture inference regression mismatch")
    print("PASS: identical artificial input and exact output token sequence/finish reason")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("request")
    r.add_argument("--url", default="http://127.0.0.1:30281")
    r.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    r.add_argument("--request-id", default="kv-ec-artificial-request")
    r.add_argument("--prompt-tokens", type=int, choices=(384, 768), default=384,
                   help="use 768 for a 512-token protection range; use identical input in all modes")
    r.add_argument("--stream-metrics", action="store_true", help="measure client-side TTFT and amortized TPOT")
    r.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("paths", type=Path, nargs=2)
    args = p.parse_args()
    request(args) if args.command == "request" else compare(args.paths)
