"""Artificial non-sensitive request and exact output-token regression comparison."""

import argparse
import hashlib
import json
import time
from pathlib import Path
import urllib.request


def request(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # Use the same fixed token sequence for 128-token and 256-token captures.
    text = "KV cache research with artificial public test data. " * 160
    prompt_tokens = getattr(args, "prompt_tokens", 384)
    ids = tokenizer.encode(text, add_special_tokens=False)[:prompt_tokens]
    if len(ids) < prompt_tokens:
        raise ValueError("artificial prompt has fewer than 256 tokens")
    streaming = getattr(args, "stream_metrics", False)
    data = {"rid": args.request_id, "input_ids": ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16}, "stream": streaming}
    # No proxy for local traffic. Only generated response tokens, never prompt text,
    # are saved. Request IDs here are deliberately artificial.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(args.url.rstrip("/") + "/generate",
                                 data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first_token = None
    with opener.open(req, timeout=600) as response:
        if streaming:
            result = None
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    break
                event = json.loads(raw)
                if event.get("output_ids"):
                    if first_token is None:
                        first_token = time.perf_counter()
                    result = event
            if result is None:
                raise ValueError("stream returned no output token IDs")
        else:
            result = json.load(response)
    finished = time.perf_counter()
    if not isinstance(result, dict) or not isinstance(result.get("output_ids"), list) or not result["output_ids"]:
        raise ValueError("server did not return nonempty output_ids; cannot prove token regression")
    if result.get("meta_info", {}).get("finish_reason", {}).get("type") not in ("length", "stop"):
        raise ValueError("request did not finish normally")
    import numpy as np

    record = {"kind": "real_server_artificial_prompt", "request_id": args.request_id,
              "model_identifier": args.model, "input_token_count": len(ids),
              "input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
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
