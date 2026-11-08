"""Fixed token-ID trace and bounded client-clock SSE measurements.

Fault callbacks may integrate an already supervised logical erasure hook. This
module has no kill/process/device/network injection implementation.
"""

import argparse
import json
import math
from pathlib import Path
import random
import time
import urllib.request
import uuid

from .admission import BASELINES
from ..cold_timing import clock_identity
from ..node_block_export import canonical, sha


def trace(*, seed, requests, input_tokens, output_tokens, vocab_size, model_sha256):
    if any(type(v) is not int or v <= 0 for v in (requests, input_tokens, output_tokens, vocab_size)):
        raise ValueError("positive workload dimensions required")
    if requests > 1000 or requests * input_tokens > 2_000_000 or output_tokens > 4096:
        raise ValueError("bounded trace budget exceeded")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ValueError("model/tokenizer fingerprint required")
    rng = random.Random(seed)
    rows = [dict(request_id=f"trace-{seed}-{i}", input_ids=[rng.randrange(vocab_size) for _ in range(input_tokens)],
                 output_tokens=output_tokens) for i in range(requests)]
    value = dict(schema="ec-fixed-token-trace-v1", seed=seed, model_sha256=model_sha256,
                 token_source="fixed random IDs; special IDs not excluded", requests=rows)
    return dict(value, trace_sha256=sha(canonical(value)))


def validate_trace(value):
    plain = {k: v for k, v in value.items() if k != "trace_sha256"}
    if value.get("schema") != "ec-fixed-token-trace-v1" or sha(canonical(plain)) != value.get("trace_sha256"):
        raise ValueError("trace integrity failure")
    rows = value["requests"]
    if not rows or len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("unique trace request IDs required")
    for row in rows:
        if (not row["input_ids"] or any(type(t) is not int or t < 0 for t in row["input_ids"])
                or type(row["output_tokens"]) is not int or not 1 <= row["output_tokens"] <= 4096):
            raise ValueError("invalid token trace")
    return value


def request_fault_schedule(value, *, probability, seed):
    """Bernoulli per-request probability, not per-device/time-window failure rate."""
    validate_trace(value)
    if probability not in (.05, .10, .15):
        raise ValueError("explicit request fault probability .05/.10/.15 required")
    rng = random.Random(seed)
    return dict(probability=probability, semantics="independent Bernoulli per request", seed=seed,
                selected_request_ids=[r["request_id"] for r in value["requests"] if rng.random() < probability])


def measure(url, row, *, timeout=60, fault=None, expected_output=None, fault_after_tokens=0):
    """One bounded request, cumulative output_ids required (same as cold_timing).

    Callback returns {kind: kv_block_erasure, request_id, checkpoint_binding,
    verified: True} only after supervised erasure completes. Boundary is the
    client timestamp BEFORE callback, so includes callback/detection overhead.
    It is not a GPU/process/physical failure or new-decode measurement.
    """
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 900
            or type(fault_after_tokens) is not int or not 0 <= fault_after_tokens < row["output_tokens"]):
        raise ValueError("bounded timeout and observable pre-completion fault position required")
    identity = clock_identity()
    boundary = receipt = recovered_first = None

    def inject(observed):
        nonlocal boundary, receipt
        boundary = time.monotonic_ns()
        receipt = fault(dict(row, observed_completion_tokens=observed))
        if (not isinstance(receipt, dict) or receipt.get("kind") != "kv_block_erasure" or receipt.get("verified") is not True
                or receipt.get("request_id") != row["request_id"] or not receipt.get("checkpoint_binding")
                or receipt.get("observed_completion_tokens") != observed):
            raise ValueError("verified request/checkpoint/observed-position-bound logical erasure receipt required")

    if fault and fault_after_tokens == 0:
        inject(0)
    payload = dict(input_ids=row["input_ids"], rid=row["request_id"], stream=True,
                   sampling_params=dict(temperature=0, max_new_tokens=row["output_tokens"], ignore_eos=True),
                   return_logprob=False)
    req = urllib.request.Request(url.rstrip("/") + "/generate", data=canonical(payload),
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic_ns()
    deadline = started + int(timeout * 1e9)
    first = None
    output = []
    first_count = None
    final = None
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        # A read cannot exceed the socket timeout; total deadline also checked.
        for _ in range(100_000):
            line = response.readline(2 * 1024**2 + 1)
            if time.monotonic_ns() > deadline or len(line) > 2 * 1024**2:
                raise TimeoutError("SSE deadline/frame budget exceeded")
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            ids = item.get("output_ids", [])
            count = item.get("meta_info", {}).get("completion_tokens", 0)
            if (not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids)
                    or type(count) is not int or count != len(ids) or len(output) > count
                    or count > row["output_tokens"] or ids[:len(output)] != output):
                raise ValueError("monotonic cumulative output token IDs required")
            now = time.monotonic_ns()
            if count > len(output):
                events.append(dict(received_ns=now, cumulative_tokens=count))
                if first is None:
                    first, first_count = now, count
                if boundary is not None and recovered_first is None:
                    recovered_first = now
            output, final = ids[:], item
            if fault and boundary is None and count >= fault_after_tokens:
                if count == row["output_tokens"]:
                    raise ValueError("SSE batching skipped fault position until completion")
                inject(count)
        else:
            raise ValueError("SSE item budget exceeded")
    completed = time.monotonic_ns()
    if first is None or len(output) != row["output_tokens"] or final["meta_info"].get("prompt_tokens") != len(row["input_ids"]):
        raise ValueError("actual prompt/output lengths differ from fixed trace")
    if expected_output is not None and output != expected_output:
        raise ValueError("output IDs differ from fault-free reference")
    if fault and (boundary is None or recovered_first is None):
        raise ValueError("no post-fault output event observed")
    return dict(schema="ec-sse-measurement-v1", **identity, request_id=row["request_id"],
                request_sha256=sha(canonical(row)), actual_input_tokens=len(row["input_ids"]),
                actual_output_tokens=len(output), output_ids=output, token_events=events,
                request_start_ns=started, first_output_ns=first, response_complete_ns=completed,
                first_observed_completion_tokens=first_count, fault_boundary_ns=boundary,
                fault_receipt=receipt, recovered_first_output_ns=recovered_first,
                fault_target_completion_tokens=fault_after_tokens if fault else None,
                fault_to_sse_ms=(recovered_first-boundary)/1e6 if boundary else None,
                ttft_ms=(first-started)/1e6, end_to_end_ms=(completed-started)/1e6,
                output_tokens_per_s=len(output)/((completed-started)/1e9),
                output_reference_checked=expected_output is not None,
                physical_failure=False, gpu_failover_verified=False,
                limitation="client SSE arrival; callback receipt not independently attested; buffered/pending token possible; not new decode or server prefill latency")


def run(url, value, output, baseline, *, timeout=60, fault=None, references=None):
    validate_trace(value)
    if baseline not in BASELINES or not 0 < timeout <= 900:
        raise ValueError("known baseline and bounded timeout required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for row in value["requests"]:
        try:
            record = measure(url, row, timeout=timeout, fault=fault,
                             expected_output=references[row["request_id"]] if references else None)
            record["success"] = True
        except Exception as exc:
            record = dict(request_id=row["request_id"], success=False,
                          error_type=type(exc).__name__, timeout=isinstance(exc, TimeoutError))
        results.append(record)
    report = dict(schema="ec-trace-run-v1", run_id=uuid.uuid4().hex, baseline_label=baseline,
                  baseline_activation_verified=False, trace_sha256=value["trace_sha256"],
                  concurrency=1, results=results, requests=len(results),
                  completed=sum(r["success"] for r in results),
                  note="label does not enable protection; caller must verify service configuration")
    with (output / "raw.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="op", required=True)
    q = sub.add_parser("trace")
    for name, default in (("seed", 1), ("requests", 1), ("input-tokens", 256), ("output-tokens", 17), ("vocab-size", 1000)):
        q.add_argument("--" + name, type=int, default=default)
    q.add_argument("--model-sha256", required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("run")
    q.add_argument("--trace", type=Path, required=True)
    q.add_argument("--url", required=True)
    q.add_argument("--baseline", choices=BASELINES, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--timeout", type=float, default=60)
    args = p.parse_args()
    if args.op == "trace":
        value = trace(seed=args.seed, requests=args.requests, input_tokens=args.input_tokens,
                      output_tokens=args.output_tokens, vocab_size=args.vocab_size, model_sha256=args.model_sha256)
        with args.output.open("x") as stream:
            json.dump(value, stream)
    else:
        run(args.url, json.loads(args.trace.read_text()), args.output, args.baseline, timeout=args.timeout)


if __name__ == "__main__":
    main()
