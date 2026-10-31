"""Opt-in Cold comparison. One coordinator/client clock; no recovery algorithm.

First output means first received SSE item carrying >=1 completion token. For
restore it can contain the saved pending token; it is not first *new* Decode.
No CUDA synchronization is introduced by this module.
"""

import json
import math
import os
from pathlib import Path
import socket
import time
import urllib.request
import uuid

from .reprefill_server import read_ids, sampling


PROFILE = 'cold-client-sse-v1'
METRICS = ('target_launch_to_ready_ms', 'fault_boundary_to_first_output_token_ms',
           'fault_boundary_to_full_output_ms', 'fault_boundary_to_response_complete_ms',
           'fault_boundary_to_validation_complete_ms', 'request_to_first_output_token_ms',
           'request_to_full_output_ms', 'response_stream_ms')


def measurement_config(cfg):
    value = cfg.get('measurement', {})
    if (not isinstance(value, dict) or set(value) - {'cold_comparison', 'decode_forward_timing'} or
            any(type(v) is not bool for v in value.values())):
        raise ValueError('measurement_requires_boolean_cold_comparison_decode_forward_timing')
    return dict(cold_comparison=value.get('cold_comparison', False),
                decode_forward_timing=value.get('decode_forward_timing', False))


def clock_identity():
    return dict(host=socket.gethostname(), boot_id=Path(
        '/proc/sys/kernel/random/boot_id').read_text().strip(), pid=os.getpid(),
        rank=None, clock_domain='host_monotonic_ns')


def stream_request(url, ids, request_id, operation_id, timeout=600):
    """Same payload/HTTP/SSE completion semantics for both recovery methods.

    Timestamp immediately before urlopen; first/full timestamps immediately
    after parsing a qualifying SSE item; response-complete after stream close.
    All token correctness checks happen after response-complete is captured.
    """
    payload = dict(input_ids=ids, rid=request_id, sampling_params=sampling(),
                   stream=True, return_logprob=False)
    req = urllib.request.Request(url.rstrip('/') + '/generate', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    identity = clock_identity()
    span_id = uuid.uuid4().hex
    first = full = final = None
    first_count = None
    started = time.monotonic_ns()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            item = json.loads(data)
            count = item.get('meta_info', {}).get('completion_tokens', 0)
            if count > 0 and first is None:
                first = time.monotonic_ns()
                first_count = count
            if count == 17 and full is None:
                full = time.monotonic_ns()
            final = item
    completed = time.monotonic_ns()
    if first is None or full is None or final is None:
        raise ValueError('cold_client_missing_first_or_full_output')
    meta = final.get('meta_info', {})
    output = final.get('output_ids')
    if (meta.get('completion_tokens') != 17 or meta.get('prompt_tokens') != 256 or
            not isinstance(output, list) or len(output) != 17 or
            any(type(t) is not int for t in output)):
        raise ValueError('cold_client_output_boundary_invalid')
    return dict(schema='kv-ec-cold-client-v1', profile=PROFILE, operation_id=operation_id,
                span_id=span_id, **identity, request_id=request_id, request_start_ns=started,
                first_output_token_ns=first, first_observed_completion_tokens=first_count,
                full_output_ns=full, response_complete_ns=completed,
                output_token_ids=output, response=final)


def build_timing(operation_id, scheme, fault, launch, ready, client, validated):
    identity = clock_identity()
    if any(client.get(k) != identity[k] for k in ('host', 'boot_id', 'clock_domain')):
        raise ValueError('cold_comparison_cross_clock_domain')
    events = dict(fault_boundary=fault, target_launch=launch, target_ready=ready,
        request_start=client['request_start_ns'], first_output_token=client['first_output_token_ns'],
        full_output=client['full_output_ns'], response_complete=client['response_complete_ns'],
        validation_complete=validated)
    ordered = list(events.values())
    if (any(type(v) is not int for v in ordered) or ordered != sorted(ordered) or
            scheme not in ('two-level', 'reprefill') or client['operation_id'] != operation_id):
        raise ValueError('cold_comparison_event_order_or_identity_invalid')
    intervals = dict(target_launch_to_ready=('target_launch', 'target_ready'),
        fault_boundary_to_first_output_token=('fault_boundary', 'first_output_token'),
        fault_boundary_to_full_output=('fault_boundary', 'full_output'),
        fault_boundary_to_response_complete=('fault_boundary', 'response_complete'),
        fault_boundary_to_validation_complete=('fault_boundary', 'validation_complete'),
        request_to_first_output_token=('request_start', 'first_output_token'),
        request_to_full_output=('request_start', 'full_output'),
        response_stream=('request_start', 'response_complete'))
    return dict(schema='kv-ec-cold-comparison-timing-v1', profile=PROFILE,
        operation_id=operation_id, span_id=uuid.uuid4().hex, scheme=scheme, **identity,
        events_ns=events, intervals=intervals,
        **{name + '_ms': (events[end] - events[start]) / 1e6
           for name, (start, end) in intervals.items()},
        functional_correctness_pass=True, performance_measurement_eligible=True,
        algorithm_comparison_eligible=False, production_recovery=False,
        diagnostics=('two-rank byte readback, TP state votes, per-forward sync/logits files, '
                     'live Stage E checker' if scheme == 'two-level' else
                     'uncached prefill + 16 decodes; client checks all 17 output IDs'),
        limitation='common client endpoints; asymmetric mandatory diagnostics remain included; '
                   'first output may include restored pending token; no failure detection')


def validate_timing(record, operation_id, scheme):
    if (record.get('schema') != 'kv-ec-cold-comparison-timing-v1' or
            record.get('profile') != PROFILE or record.get('operation_id') != operation_id or
            record.get('scheme') != scheme or record.get('clock_domain') != 'host_monotonic_ns' or
            record.get('functional_correctness_pass') is not True):
        raise ValueError('cold_comparison_timing_identity_invalid')
    events = record['events_ns']
    names = ('fault_boundary', 'target_launch', 'target_ready', 'request_start',
             'first_output_token', 'full_output', 'response_complete', 'validation_complete')
    values = [events[k] for k in names]
    if any(type(v) is not int for v in values) or values != sorted(values):
        raise ValueError('cold_comparison_timing_event_order_invalid')
    for metric in METRICS:
        value = record.get(metric)
        start, end = record['intervals'][metric[:-3]]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0 or
                value != (events[end] - events[start]) / 1e6):
            raise ValueError('cold_comparison_timing_interval_invalid:' + metric)
    return record
