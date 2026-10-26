"""Artificial serving request + complete online-chunk evidence acceptance.

Use 768/16 for three prefill chunks, or 255/258 to cross two decode boundaries.
This script never starts/stops a server or touches GPU processes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def wait_for_chunks(root, rid, expected, timeout):
    deadline = time.monotonic() + timeout
    request_hash = hashlib.sha256(rid.encode()).hexdigest()
    while True:
        found = {}
        for path in Path(root).glob("*/manifest.json"):
            meta = json.loads(path.read_text())
            if (
                meta.get("request_id") == request_hash
                and meta.get("state") == "COMMITTED"
            ):
                chunk = meta["online_identity"]["chunk_id"]
                if chunk in found:
                    raise ValueError(
                        "duplicate logical chunk evidence; use a unique request/run"
                    )
                if (
                    meta["token_start"] != chunk * 256
                    or meta["token_count"] != 256
                    or len(meta["ranks"]) != meta["tp_size"]
                    or len(meta.get("recovery_checks", [])) != meta["tp_size"]
                ):
                    raise ValueError(
                        "incomplete chunk metadata or --test-recovery was not enabled"
                    )
                found[chunk] = str(path.parent)
        if set(found) == set(range(expected)):
            return found
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"expected {expected} chunks, committed {sorted(found)}; inspect backpressure/errors"
            )
        time.sleep(0.1)


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30381")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="latency only, without protection evidence",
    )
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=768)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--manager-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if (
        args.prompt_tokens < 1
        or args.new_tokens < 1
        or args.prompt_tokens + args.new_tokens > 4096
    ):
        parser.error("positive prompt/output lengths within context 4096 required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = "KV cache research with artificial public test data. " * (
        args.prompt_tokens + 1
    )
    ids = tokenizer.encode(text, add_special_tokens=False)[: args.prompt_tokens]
    data = {
        "rid": args.request_id,
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.new_tokens,
            "ignore_eos": True,
        },
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        args.url.rstrip("/") + "/generate",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with opener.open(request, timeout=args.timeout) as response:
        result = json.load(response)
    request_ms = (time.perf_counter() - started) * 1000
    reason = result.get("meta_info", {}).get("finish_reason", {})
    if (
        reason.get("type") not in ("length", "stop")
        or len(result.get("output_ids", [])) != args.new_tokens
    ):
        raise ValueError("request aborted or output length mismatch")
    # The final sampled token has never been run through the model.
    expected = (len(ids) + args.new_tokens - 1) // 256
    chunks = (
        {}
        if args.baseline
        else wait_for_chunks(
            args.manager_output, args.request_id, expected, args.timeout
        )
    )
    report = {
        "baseline": args.baseline,
        "output_ids_sha256": hashlib.sha256(
            json.dumps(result["output_ids"]).encode()
        ).hexdigest(),
        "request_ms": request_ms,
        "prompt_tokens": len(ids),
        "output_tokens": args.new_tokens,
        "committed_chunks": chunks,
        "meta_info": result["meta_info"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
