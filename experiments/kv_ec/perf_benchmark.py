"""Real Stage E performance preparation; synthetic/fixture data is never eligible.

No SSH, process termination, cache flushing or service launch is implicit in
inference/collect. `repeat` executes only the explicitly supplied trial command.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

from .perf_observer import emit

SCHEMES = ('no-ec', 'reprefill', 'xor-only', 'rs-only', 'two-level')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            hasher.update(block)
    return hasher.hexdigest()


def command(argv, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(argv=argv, returncode=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(argv=argv, error=str(exc))


def identity():
    sha = command(['git', 'rev-parse', 'HEAD'])
    if sha.get('returncode') != 0:
        raise ValueError('benchmark must run from the repository checkout')
    diff = command(['git', 'diff', '--binary', 'HEAD'], timeout=30)
    # Include new observation/benchmark files even before they are committed.
    root = Path(__file__).resolve().parent
    # Include untracked adapters/codecs as well as the coordinator. A commit
    # and tracked diff alone cannot detect different new files on two hosts.
    benchmark_paths = [p for p in root.rglob('*') if p.is_file() and
                       p.suffix in ('.py', '.c', '.h', '.cpp', '.cu', '.cuh')]
    files = {str(p.relative_to(root)): digest(p) for p in sorted(benchmark_paths)}
    runtime = root.parents[1] / 'python/sglang/srt'
    files.update({str(p.relative_to(root.parents[1])): digest(p)
                  for p in sorted(runtime.rglob('*.py'))})
    return dict(commit_sha=sha['stdout'],
        git_status=command(['git', 'status', '--short'])['stdout'],
        tracked_diff_sha256=hashlib.sha256(diff['stdout'].encode()).hexdigest(),
        benchmark_files=files, host=socket.gethostname(), pid=os.getpid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        python=platform.python_version(), utc=datetime.now(timezone.utc).isoformat())


def resource_snapshot():
    """Host-wide gauges, never attributed as per-request CUDA utilization."""
    result = dict(monotonic_ns=time.monotonic_ns(), host=socket.gethostname(),
                  cpu_scope='host-wide', gpu_scope='device-wide')
    for name in ('stat', 'meminfo', 'net/dev'):
        try:
            result[name] = Path('/proc', name).read_text()
        except OSError as exc:
            result[name] = dict(unavailable=str(exc))
    result['rdma_counters'] = {}
    for port in Path('/sys/class/infiniband').glob('*/ports/*'):
        for name in ('port_xmit_data', 'port_rcv_data', 'port_xmit_packets', 'port_rcv_packets'):
            p = port / 'counters' / name
            try:
                result['rdma_counters'][str(p)] = int(p.read_text())
            except (OSError, ValueError):
                pass
    result['rdma_data_counter_unit'] = '4-byte words; counters may include unrelated traffic'
    result['gpu'] = command(['nvidia-smi',
        '--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,power.draw',
        '--format=csv,noheader,nounits'], timeout=5)
    return result


class Sampler:
    def __init__(self, path, interval):
        if not math.isfinite(interval) or interval < 0.5:
            raise ValueError('resource sampling interval must be >= 0.5 seconds')
        self.path, self.interval, self.stop_event = Path(path), interval, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open('x') as stream:
            while True:
                started = time.monotonic_ns()
                record = resource_snapshot()
                record['sampler_ms'] = (time.monotonic_ns() - started) / 1e6
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                if self.stop_event.wait(self.interval):
                    break

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=10)


def stats(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean_ms=None, stddev_ms=None, p50_ms=None, p95_ms=None, p99_ms=None)
    def q(p):
        index = (len(values) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return dict(n=len(values), mean_ms=statistics.mean(values),
                stddev_ms=statistics.stdev(values) if len(values) > 1 else None,
                p50_ms=q(.5) if len(values) >= 5 else None,
                p95_ms=q(.95) if len(values) >= 20 else None,
                p99_ms=q(.99) if len(values) >= 100 else None)


def http(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def inference(args):
    """Both controls execute a full uncached prefill, followed by 16 decodes."""
    ids = read(args.input_ids)
    if isinstance(ids, dict):
        ids = ids['input_ids']
    if len(ids) != 256 or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('first baseline requires exactly 256 token IDs')
    input_ids_sha256 = hashlib.sha256(struct.pack('<256q', *ids)).hexdigest()
    expected_input = getattr(args, 'expected_input_ids_sha256', None)
    if expected_input is not None and input_ids_sha256 != expected_input:
        raise ValueError('baseline_input_ids_sha256_mismatch')
    info = http(args.url.rstrip('/') + '/server_info')
    tp_size = getattr(args, 'tp', 2)
    required = dict(tp_size=tp_size, disable_radix_cache=True, disable_overlap_schedule=True,
                    chunked_prefill_size=-1, max_running_requests=1, attention_backend='triton')
    if any(info.get(k) != v for k, v in required.items()):
        raise ValueError('baseline server flags do not match Stage E: ' + str(required))
    graph = info.get('cuda_graph_config', {})
    if any(graph.get(k, {}).get('backend') != 'disabled' for k in ('decode', 'prefill')):
        raise ValueError('baseline requires disabled CUDA graphs, matching Stage E')
    server_model = getattr(args, 'server_model', None) or args.model
    if info.get('model_path') != server_model:
        raise ValueError('baseline model_path mismatch')
    if args.expected_model_sha256:
        if getattr(args, 'model_identity_evidence', None):
            evidence = read(args.model_identity_evidence)
            actual = evidence.get('model_identity', evidence.get('runtime_identity', {})).get('model_sha256')
            if evidence.get('tp_size', 2) != tp_size:
                raise ValueError('baseline_tp_identity_mismatch')
        else:
            from .debug_capture import source_identity
            actual = source_identity(args.model).get('model_sha256')
        if actual != args.expected_model_sha256:
            raise ValueError('baseline_model_sha256_mismatch')
    # This is exactly Stage D submit's policy (temperature=0 normalizes to
    # greedy top_k=1), 17 output IDs = one pending y0 plus 16 native decodes.
    sampling = dict(max_new_tokens=17, temperature=0, ignore_eos=True, top_p=1.,
                    top_k=1, min_p=0., min_new_tokens=0, n=1,
                    frequency_penalty=0., presence_penalty=0., repetition_penalty=1.)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = identity()
    if getattr(args, 'model_identity_evidence', None):
        evidence = read(args.model_identity_evidence)
        if 'gpu_admission_mode' in evidence:
            from .gpu_admission import validate_evidence
            validate_evidence(evidence, evidence['gpu_admission_mode'], tp_size)
        for field in ('gpu_admission_mode', 'mem_fraction_static', 'tp_memory_imbalance_check',
                      'shared_resource_environment', 'gpu_resources_before_start'):
            if field in evidence:
                meta[field] = evidence[field]
        meta['isolated_performance_eligible'] = False  # standalone request is not admission proof
    model_files = [Path(args.model) / 'config.json', *sorted(Path(args.model).glob('*.safetensors.index.json'))]
    meta.update(scheme=args.scheme, model=args.model,
        model_sha256=getattr(args, 'expected_model_sha256', None), model_files={str(p): digest(p)
        for p in model_files if p.exists()}, input_sha256=digest(args.input_ids), prefix_tokens=256,
        tp_size=tp_size, input_ids_sha256=input_ids_sha256,
        generation_tokens=17, sampling=sampling, server_info=info,
        repeat=args.repeat, warmup=args.warmup, cache_policy='server radix disabled; cached_tokens must be zero',
        limitation='reprefill starts at explicit client resubmission, excludes detection and startup')
    save(args.output / 'metadata.json', meta)
    with Sampler(args.output / 'resources.jsonl', args.resource_interval):
        for i in range(args.warmup + args.repeat):
            request = dict(input_ids=ids, sampling_params=sampling, stream=True,
                           rid=f'perf-{uuid.uuid4().hex}', return_logprob=False)
            started, first, full, final = time.monotonic_ns(), None, None, None
            # SSE: timestamp receipt of first generated token on this client.
            req = urllib.request.Request(args.url.rstrip('/') + '/generate',
                data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        break
                    item = json.loads(data)
                    completion_tokens = item.get('meta_info', {}).get('completion_tokens', 0)
                    if completion_tokens > 0 and first is None:
                        first = time.monotonic_ns()
                    if completion_tokens == 17 and full is None:
                        full = time.monotonic_ns()
                    final = item
            response_complete = time.monotonic_ns()
            if final is None or first is None or full is None:
                raise ValueError('missing streaming generation evidence')
            usage = final['meta_info']
            if usage.get('cached_tokens') != 0 or usage.get('prompt_tokens') != 256 or usage.get('completion_tokens') != 17:
                raise ValueError('cached/short/wrong-prefix inference cannot enter baseline statistics')
            output_ids = final.get('output_ids')
            if not isinstance(output_ids, list) or len(output_ids) != 17:
                raise ValueError('server must return all 17 output_ids for correctness comparison')
            save(args.output / f'request-{i:03d}.json', dict(warmup=i < args.warmup,
                scheme=args.scheme, start_ns=started, first_output_token_ns=first,
                full_output_ns=full, response_complete_ns=response_complete,
                end_ns=response_complete,
                request_ms=(response_complete-started)/1e6, prefill_ttft_ms=(first-started)/1e6,
                ready_to_first_token_ms=((first-args.ready_monotonic_ns)/1e6
                    if getattr(args, 'ready_monotonic_ns', None) is not None else None),
                ready_to_decode_complete_ms=((response_complete-args.ready_monotonic_ns)/1e6
                    if getattr(args, 'ready_monotonic_ns', None) is not None else None),
                source='client_monotonic_ns', response=final, output_token_ids=output_ids,
                correctness='raw token IDs recorded; compare against Stage E reference separately'))


def collect_inference(args):
    """Collect independent ready-server inference roots without pooling identities."""
    groups = defaultdict(list)
    accepted, rejected, contexts = [], [], {}
    for root in args.roots:
        try:
            metadata = read(root / 'metadata.json')
            if metadata.get('scheme') != args.scheme:
                raise ValueError('scheme_mismatch')
            if metadata.get('prefix_tokens') != 256 or metadata.get('tp_size') not in (2, 4):
                raise ValueError('baseline_identity_mismatch')
            records = []
            for path in sorted(root.glob('request-*.json')):
                record = read(path)
                if record.get('warmup'):
                    continue
                response = record.get('response', {})
                meta = response.get('meta_info', {})
                if (meta.get('cached_tokens') != 0 or meta.get('prompt_tokens') != 256 or
                        meta.get('completion_tokens') != 17 or
                        len(record.get('output_token_ids', [])) != 17):
                    raise ValueError('inference_correctness_boundary_invalid')
                records.append(record)
            if len(records) != args.expected_trials:
                raise ValueError(f'expected_{args.expected_trials}_requests_got_{len(records)}')
            selected = metadata.get('gpu_admission_mode', 'unknown')
            if selected not in ('strict', 'shared', 'unknown'):
                raise ValueError('GPU_ADMISSION_MODE_INVALID: inference metadata')
            # A ready-server request is a different timing boundary from a
            # cold recovery. Keep model/input/source/device identities apart,
            # even when TP and admission policy happen to be identical.
            context = {k: metadata.get(k) for k in (
                'scheme', 'tp_size', 'gpu_admission_mode', 'mem_fraction_static',
                'model', 'model_sha256', 'model_files', 'input_sha256', 'input_ids_sha256',
                'prefix_tokens', 'generation_tokens', 'sampling', 'server_info',
                'commit_sha', 'tracked_diff_sha256', 'benchmark_files',
                'gpu_resources_before_start')}
            key = hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode()).hexdigest()
            contexts[key] = context
            accepted.append(str(root))
            for metric in ('request_ms', 'prefill_ttft_ms'):
                # Unknown historical mode is not evidence of isolation.
                groups[(selected, metadata['tp_size'], key, metric)].extend(r[metric] for r in records)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(root=str(root), reason=f'{type(exc).__name__}: {exc}'))
    rows = [dict(scheme=args.scheme, metric=metric, identity_sha256=key,
                 gpu_admission_mode=selected, tp_size=tp_size,
                 isolated_performance_eligible=False, **stats(values))
            for (selected, tp_size, key, metric), values in sorted(groups.items())]
    save(args.output, dict(schema='kv-ec-inference-summary-v1', scheme=args.scheme,
        accepted_roots=accepted, rejected=rejected, rows=rows,
        expected_trials=args.expected_trials, identity_contexts=contexts,
        warning='request latency is ready-server client timing; it excludes cold startup and fault detection'))
    print(json.dumps(dict(output=str(args.output), series=len(rows), rejected=len(rejected))))


def local_xor(args):
    """Minimal real-capture local-loss control, deliberately no GPU resume claim."""
    import numpy as np
    from .cold_resume_state import load_checkpoint
    from .node_block_export import unpack
    from .xor_codec import encode, recover
    args.output.mkdir(parents=True, exist_ok=False)
    source, state, _, _ = load_checkpoint(args.capture)
    meta, payloads = unpack(source.load())  # production validator; fixtures rejected
    blocks = [np.frombuffer(p, dtype=np.uint8) for p in payloads]
    if len(meta['ranks']) not in (2, 4) or state['token_count'] != 256 or not 0 <= args.lost_rank < len(blocks):
        raise ValueError('real TP2_or_TP4/256 capture and valid lost rank required')
    save(args.output / 'metadata.json', dict(**identity(), scheme='xor-only',
        capture=str(args.capture), capture_sha256=hashlib.sha256(source.load()).hexdigest(),
        runtime_identity=state['runtime_identity'], prefix_tokens=256, tp_size=len(meta['ranks']),
        limitation='CPU codec on authentic GPU capture; controlled single TP-rank byte loss; '
                   'no Scheduler/GPU resume, not native-RDMA end-to-end performance'))
    with Sampler(args.output / 'resources.jsonl', args.resource_interval):
        for i in range(args.repeat):
            start = time.monotonic_ns()
            parity = encode(blocks)
            encoded = time.monotonic_ns()
            survivors = [None if rank == args.lost_rank else b for rank, b in enumerate(blocks)]
            restored = recover(survivors, parity)
            recovered = time.monotonic_ns()
            exact = np.array_equal(restored, blocks[args.lost_rank])
            checked = time.monotonic_ns()
            save(args.output / f'xor-{i:03d}.json', dict(scheme='xor-only', iteration=i,
                xor_ms=(encoded-start)/1e6, recovery_ms=(recovered-encoded)/1e6,
                validation_ms=(checked-recovered)/1e6, byte_exact=bool(exact),
                lost_rank=args.lost_rank, decode_correctness=None, real_rdma=False))
            if not exact:
                raise ValueError('XOR controlled recovery mismatch')


def rs_only(args):
    """Consume an existing native Stage C/RS report; never invent a run."""
    report = read(args.report)
    if report.get('state') not in ('PASS', 'COMMITTED') or report.get('transport') != 'rdma':
        raise ValueError('rs-only requires an existing native RDMA PASS report')
    if report.get('network_profile') not in (None, 'physical'):
        raise ValueError('mock/loopback network reports are not eligible')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'metadata.json', dict(**identity(), scheme='rs-only',
        source_report=str(args.report.resolve()), source_report_sha256=digest(args.report),
        prefix_tokens=report.get('chunk_tokens'), tp_size=report.get('tp_size'), rs=[2, 1],
        limitation='RS/Node Block recovery only; no target Scheduler/GPU writeback or Decode'))
    metrics = {}
    for name in ('rs_encode_ms', 'remote_committed_ms', 'recovery_ms', 'server_recovery_ms'):
        if isinstance(report.get(name), (int, float)):
            metrics[name] = report[name]
    for i in range(args.repeat):
        save(args.output / f'rs-{i:03d}.json', dict(scheme='rs-only', iteration=i,
            source_report=str(args.report), real_rdma=True, decode_correctness=None,
            **metrics))


def gate_stage_e(output):
    # Re-run the actual checker including live parity admission; a JSON flag
    # or historical PASS alone never grants eligibility.
    from .verify_rs_cold_resume import check_results
    verdict = check_results(Path(output))
    required = ('pass_stage_e', 'pass_stage_d', 'real_rdma', 'cuda_executed',
                'output_token_ids_match', 'gpu_restore_committed', 'new_instance')
    if any(verdict.get(k) is not True for k in required):
        raise ValueError('strict real Stage E gate failed')
    return verdict


def repeat(args):
    from .gpu_admission import config_mode, metadata
    from .perf_preflight import preflight, validate_config
    from .perf_stage_e import process_identity, signal_owned
    import signal
    config = read(args.config)
    validate_config(config)
    selected_mode = config_mode(config)
    policy = metadata(selected_mode, config.get('mem_fraction_static', .6))
    policy.update(tp_size=config['tp_size'], prefix_tokens=config['prefix_tokens'])
    if Path.cwd().resolve() != Path(config['nodes'][0]['repo']).resolve():
        raise ValueError('repeat must run from configured node0 repo on 232')
    config['resource_interval'] = args.resource_interval or config['resource_interval']
    argv = [sys.executable, '-m', 'experiments.kv_ec.perf_stage_e', 'trial']
    args.output.mkdir(parents=True, exist_ok=False)
    recovery_mode = getattr(args, 'recovery_mode', 'two-level')
    if recovery_mode not in ('two-level', 'reprefill'):
        raise ValueError('invalid_recovery_mode')
    paired_ready_reprefill = bool(getattr(args, 'paired_ready_reprefill', False))
    if paired_ready_reprefill and recovery_mode != 'two-level':
        raise ValueError('paired_ready_reprefill_requires_two_level')
    if recovery_mode == 'reprefill':
        argv = argv + ['--recovery-mode', 'reprefill']
    batch_metadata = dict(**identity(), config=config, **policy,
         repeat=args.repeat, argv=argv, scheme=recovery_mode, resource_interval=config['resource_interval'])
    save(args.output / 'config.json', config)
    admission = preflight(config, args.output.resolve(), args.repeat)
    batch_metadata['gpu_resources_before_start'] = [
        r.get('gpu_resources_before_start', []) if r else [] for r in admission['nodes']]
    save(args.output / 'metadata.json', batch_metadata)
    save(args.output / 'preflight.json', admission)
    if admission['status'] != 'PASS':
        print(json.dumps(admission), flush=True)
        raise SystemExit(1)
    for i in range(args.repeat):
        root = args.output.resolve() / f'trial-{i:03d}-{uuid.uuid4().hex}'
        root.mkdir()
        run_id = root.name
        env = dict(os.environ, KV_EC_BENCH_TRIAL=str(root), KV_EC_BENCH_CONFIG=str((args.output / 'config.json').resolve()),
                   KV_EC_PERF_RUN_ID=run_id, KV_EC_PERF_DIR=str(root / 'perf'),
                   KV_EC_PERF_ROLE='coordinator-232', KV_EC_PERF_RECOVERY_MODE=recovery_mode,
                   KV_EC_PAIRED_READY_REPREFILL='1' if paired_ready_reprefill else '0')
        started = time.monotonic_ns()
        raw = dict(run_id=run_id, returncode=None, status='failed', pass_fail='FAIL',
                   **policy, functional_correctness_pass=False,
                   commit_sha=admission['nodes'][0]['checks']['inventory']['commit_sha'],
                   scheme=recovery_mode, real_rdma_eligible=False,
                   remote_root=str(Path(config['remote_root']) / run_id))
        try:
            # Recheck idle GPUs/ports/RS epoch and all *remaining* slots before
            # each trial, including fresh batches on a long-lived server.
            current = preflight(config, root, args.repeat - i, admission['rs']['server_epoch'])
            save(root / 'preflight.json', current)
            if current['status'] != 'PASS':
                raise ValueError('trial preflight failed: ' + current['error'])
            raw['gpu_resources_before_start'] = [r.get('gpu_resources_before_start', []) for r in current['nodes']]
            raw['capture_prefix_identity'] = [r.get('checks', {}).get('model_tp_prefix_identity')
                                               for r in current['nodes']]
            with (root / 'driver.log').open('x') as log, Sampler(root / 'resources-232.jsonl', config['resource_interval']):
                proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                owned = process_identity(proc.pid)
                save(root / 'driver-process.json', owned)
                try:
                    raw['returncode'] = proc.wait(timeout=config['trial_timeout'])
                except BaseException:
                    # Only this newly created/verified session receives signals.
                    signal_owned(owned, signal.SIGTERM)
                    try:
                        proc.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        signal_owned(owned, signal.SIGKILL)
                        proc.wait(timeout=10)
                    raise
            if raw['returncode']:
                raise ValueError('trial command failed; see driver.log')
            validate_gpu_startups(root, config, recovery_mode)
            raw['gpu_startup_evidence'] = {
                name: read(root / name) for name in ('node0-reference/run.json', 'origin-run-233.json',
                    'target-run.json' if recovery_mode == 'two-level' else 'reprefill-target-run.json')}
            raw['model_tp_prefix_identity'] = [dict(tp_size=config['tp_size'], prefix_tokens=256,
                runtime_identity=record.get('runtime_identity', record.get('model_identity')))
                for record in raw['gpu_startup_evidence'].values()]
            if recovery_mode == 'two-level':
                verdict = validate_trial(root, run_id)
                raw['rank_correctness'] = (verdict or {}).get('rank_correctness', [])
                raw['rank_gpu_kv_mapping'] = (verdict or {}).get('rank_gpu_kv_mapping', [])
            else:
                reprefill = read(root / 'reprefill.json')
                if reprefill.get('pass_reprefill') is not True or reprefill.get('output_token_ids_match') is not True:
                    raise ValueError('reprefill_control_gate_failed')
            inventories = [read(root / f'inventory-{node}.json') for node in (232, 233)]
            if inventories[0]['host'] == inventories[1]['host']:
                raise ValueError('two physical hosts required')
            if any(x['commit_sha'] != inventories[0]['commit_sha'] or
                   x['tracked_diff_sha256'] != inventories[0]['tracked_diff_sha256'] or
                   x['benchmark_files'] != inventories[0]['benchmark_files'] for x in inventories):
                raise ValueError('two host source identities differ')
            raw.update(status='accepted', pass_fail='PASS',
                functional_correctness_pass=True,
                isolated_performance_eligible=selected_mode == 'strict',
                real_rdma_eligible=(recovery_mode == 'two-level'),
                decode_correctness=True,
                verdict_sha256=(digest(root / 'stage-e-check.json')
                    if recovery_mode == 'two-level' else digest(root / 'reprefill.json')))
        except (Exception, KeyboardInterrupt) as exc:
            raw['failure_reason'] = f'{type(exc).__name__}: {exc}'
        raw['trial_command_ms'] = (time.monotonic_ns()-started)/1e6
        save(root / 'trial.json', raw)
        print(json.dumps(dict(trial=str(root), **raw)), flush=True)
        if raw['status'] != 'accepted':
            raise SystemExit(1)


def validate_gpu_startups(root, config, scheme):
    """Bind all model startup policies to the coordinator configuration."""
    from .gpu_admission import config_mode, validate_evidence
    selected = config_mode(config)
    files = ['node0-reference/run.json', 'origin-run-233.json',
             'target-run.json' if scheme == 'two-level' else 'reprefill-target-run.json']
    for name in files:
        record = read(Path(root) / name)
        validate_evidence(record, selected, config.get('tp_size', 2))
        if 'nodes' in config:
            node = config['nodes'][0 if name == 'node0-reference/run.json' else 1]
            if record.get('gpus') != node['gpus'] or [r['gpu_index'] for r in record['gpu_resources_before_start']] != node['gpus'].split(','):
                raise ValueError('GPU_ADMISSION_PHYSICAL_MAPPING_MISMATCH')
        if record.get('mem_fraction_static') != config.get('mem_fraction_static', .6):
            raise ValueError('GPU_ADMISSION_MODE_MISMATCH: startup memory fraction')


def validate_trial(root, run_id):
    """Offline audit of a current live-check receipt; never add verdict fields."""
    from .perf_stage_e import target_binding
    root = Path(root)
    acceptance = read(root / 'acceptance.json')
    if acceptance.get('live_check_exit') != 0 or acceptance.get('run_id') != run_id:
        raise ValueError('missing successful current trial live-check receipt')
    for name, key in (('stage-e-check.json', 'verdict_sha256'),
                      ('target-run.json', 'target_run_sha256'), ('protection.json', 'protection_sha256')):
        if acceptance.get(key) != digest(root / name):
            raise ValueError('trial_evidence_digest_mismatch: ' + name)
    verdict = read(root / 'stage-e-check.json')
    if verdict.get('schema') != 'kv-rs-cold-resume-stage-e-v1' or verdict.get('runtime_original_local_file_used') is not False:
        raise ValueError('real Stage E schema/recovered-source provenance required')
    for key in ('pass_stage_e', 'pass_stage_d', 'real_rdma', 'cuda_executed',
                'output_token_ids_match', 'gpu_restore_committed', 'new_instance',
                'rs_recovery_byte_exact', 'no_slot_leak', 'no_request_row_leak',
                'restored_slots_used_by_decode'):
        if verdict.get(key) is not True:
            raise ValueError('Stage E gate failed: ' + key)
    for key, value in (('resume_decode_forward_count', 16), ('original_prefix_prefill_tokens', 0),
                       ('recomputed_prefix_tokens', 0), ('tp_size', read(root / 'target-run.json').get('tp_size', 2))):
        if type(verdict.get(key)) is not int or verdict[key] != value:
            raise ValueError('Stage E gate failed: ' + key)
    binding = target_binding(verdict, read(root / 'target-run.json'), read(root / 'protection.json'), run_id)
    if any(acceptance.get(k) != v for k, v in binding.items()):
        raise ValueError('trial_acceptance_binding_mismatch')
    cleanup = read(root / 'cleanup.json')
    if cleanup['status'] != 'PASS':
        raise ValueError('trial_cleanup_incomplete')
    cq = read(root / 'cq-evidence.json')
    from .rs_recovery_source import native_stats
    for upload in cq['uploads']:
        native_stats(upload, upload['bytes'])
    for key in ('client_send_stats', 'receive_stats', 'send_stats', 'client_receive_stats'):
        native_stats(cq['recovery'][key], cq['recovery'][key]['bytes'])
    for key, field in (('delivery_send', 'send_stats'), ('delivery_receive', 'receive_stats')):
        native_stats(cq[key][field], cq[key]['archive_bytes'])
    exports = [read(root / 'export0/export.json'), read(root / 'export1.json')]
    if cq['uploads'] != [e['receipt']['metrics']['rdma_native'] for e in exports]:
        raise ValueError('trial_upload_CQ_binding_mismatch')
    recovery = read(root / 'recovered-source/protected/recovery.json')
    delivery = read(root / 'delivery-send.json')
    if cq['recovery'] != recovery['transfer'] or cq['delivery_send'] != delivery:
        raise ValueError('trial_recovery_CQ_binding_mismatch')
    from .perf_stage_e import native_metrics
    if read(root / 'native-metrics.json') != native_metrics(None, *exports, recovery, delivery):
        raise ValueError('trial_native_metric_binding_mismatch')
    inventories = [read(root / f'inventory-{node}.json') for node in (232, 233)]
    if inventories[0]['host'] == inventories[1]['host'] or inventories[0]['boot_id'] == inventories[1]['boot_id']:
        raise ValueError('two physical hosts required')
    for key in ('commit_sha', 'tracked_diff_sha256', 'benchmark_files'):
        if inventories[0][key] != inventories[1][key]:
            raise ValueError('two host source identities differ: ' + key)
    metadata = read(root.parent / 'metadata.json')
    if 'gpu_admission_mode' in metadata:
        validate_gpu_startups(root, metadata['config'], 'two-level')
    for key in ('commit_sha', 'tracked_diff_sha256', 'benchmark_files'):
        if inventories[0][key] != metadata[key]:
            raise ValueError('trial_source_changed_since_batch_admission: ' + key)
    return verdict


def collect(args):
    from .gpu_admission import config_mode
    groups, rejected, accepted, missing = defaultdict(list), [], [], []
    policies, contexts = {}, {}
    seen = set()
    for root in args.roots:
        metadata = read(root / 'metadata.json')
        # No pooling configurations or schemes. Each group has its own source.
        key = hashlib.sha256(json.dumps(dict(gpu_admission_mode=metadata.get('gpu_admission_mode', 'legacy'),
            config=metadata.get('config', metadata.get('server_info',
            metadata.get('runtime_identity', {}))), source={k: metadata.get(k) for k in
            ('commit_sha', 'tracked_diff_sha256', 'benchmark_files')}), sort_keys=True).encode()).hexdigest()
        scheme = metadata['scheme']
        selected_mode = (config_mode(metadata['config']) if 'config' in metadata else
                         metadata.get('gpu_admission_mode', 'unknown'))
        if metadata.get('gpu_admission_mode', selected_mode) != selected_mode:
            raise ValueError('GPU_ADMISSION_MODE_MISMATCH: batch metadata')
        policies[key] = (selected_mode, 'gpu_admission_mode' in metadata and selected_mode == 'strict'
                         and scheme in ('two-level', 'reprefill') and 'config' in metadata)
        contexts.setdefault(key, dict(gpu_admission_mode=selected_mode,
            tp_size=metadata.get('config', {}).get('tp_size', metadata.get('tp_size', 2)),
            mem_fraction_static=metadata.get('config', {}).get('mem_fraction_static', metadata.get('mem_fraction_static')),
            tp_memory_imbalance_check=metadata.get('tp_memory_imbalance_check'),
            shared_resource_environment=selected_mode == 'shared',
            gpu_resources_before_start=metadata.get('gpu_resources_before_start', [])))
        for path in sorted(root.glob('request-*.json')):
            record = read(path)
            if record['warmup']:
                continue
            for metric in ('request_ms', 'prefill_ttft_ms'):
                groups[(scheme, key, metric)].append(record[metric])
        for path in sorted(root.glob('xor-*.json')):
            record = read(path)
            if not record['byte_exact']:
                rejected.append(str(path))
                continue
            for metric in ('xor_ms', 'recovery_ms', 'validation_ms'):
                groups[(scheme, key, metric)].append(record[metric])
        for path in sorted(root.glob('rs-*.json')):
            record = read(path)
            if record.get('real_rdma') is not True:
                rejected.append(str(path))
                continue
            for metric in ('rs_encode_ms', 'remote_committed_ms', 'recovery_ms', 'server_recovery_ms'):
                if isinstance(record.get(metric), (int, float)):
                    groups[(scheme, key, metric)].append(record[metric])
        for directory in sorted(root.glob('trial-*')):
            if not directory.is_dir():
                continue
            path = directory / 'trial.json'
            try:
                trial = read(path)
                if 'gpu_admission_mode' in metadata:
                    if (trial.get('gpu_admission_mode') != selected_mode or
                            trial.get('functional_correctness_pass') is not True or
                            trial.get('isolated_performance_eligible') is not (selected_mode == 'strict')):
                        raise ValueError('GPU_ADMISSION_MODE_MISMATCH: trial labels')
                    validate_gpu_startups(directory, metadata['config'], scheme)
                if (trial['status'] != 'accepted' or trial.get('pass_fail') != 'PASS' or
                        (scheme == 'two-level' and trial['real_rdma_eligible'] is not True)):
                    raise ValueError(trial.get('failure_reason', 'not an accepted real trial'))
                per_trial = defaultdict(float)
                if scheme == 'two-level':
                    verdict = validate_trial(directory, trial['run_id'])
                else:
                    control = read(directory / 'reprefill.json')
                    if control.get('pass_reprefill') is not True or control.get('output_token_ids_match') is not True:
                        raise ValueError('reprefill_control_gate_failed')
                    for metric in ('fault_boundary_to_ready_ms', 'ready_to_first_token_ms',
                                   'ready_to_decode_complete_ms', 'fault_boundary_to_decode_complete_ms',
                                   'client_request_ms', 'client_prefill_ttft_ms'):
                        value = control.get(metric)
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                            raise ValueError('reprefill_metric_missing_or_invalid:' + metric)
                        per_trial[metric] += value
                fingerprint = (trial['run_id'], trial['verdict_sha256'])
                if fingerprint in seen:
                    raise ValueError('duplicate trial; cannot count same PASS twice')
                expected_digest = (digest(directory / 'stage-e-check.json') if scheme == 'two-level'
                                   else digest(directory / 'reprefill.json'))
                if trial['verdict_sha256'] != expected_digest:
                    raise ValueError('trial_verdict_digest_mismatch')
                stages = set()
                checkpoint_calls = defaultdict(list)
                for log in sorted(directory.glob('perf*/*.jsonl')):
                    for line in log.read_text().splitlines():
                        record = json.loads(line)
                        if record['run_id'] != trial['run_id'] or record['status'] != 'ok':
                            raise ValueError('foreign or failed observation: ' + str(log))
                        value = record['duration_ms']
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                            raise ValueError('invalid observation duration')
                        stages.add(record['stage'])
                        name = '/'.join(map(str, (record['stage'], record['role'], record['host'], record.get('tp_rank', '-'))))
                        per_trial[name] += value
                        if record['stage'] == 'checkpoint_load_validate_inclusive':
                            checkpoint_calls[name].append(dict(
                                call_index=record.get('call_index'), duration_ms=value,
                                span_id=record.get('span_id'), pid=record.get('pid'),
                                boot_id=record.get('boot_id'), operation_id=record.get('operation_id')))
                if checkpoint_calls:
                    for name, calls in checkpoint_calls.items():
                        per_trial[name + '/call_count'] = len(calls)
                        per_trial[name + '/cumulative_ms'] = sum(c['duration_ms'] for c in calls)
                        for index, call in enumerate(sorted(calls,
                                key=lambda c: c['call_index'] or 0), start=1):
                            per_trial[name + f'/call_{index}_ms'] = call['duration_ms']
                timing_path = directory / 'cold-comparison-timing.json'
                performance_measurement_eligible = False
                if timing_path.exists():
                    timing = read(timing_path)
                    from .cold_timing import validate_timing
                    validate_timing(timing, trial['run_id'], scheme)
                    timing_status = read(directory / 'cold-comparison-timing-status.json') \
                        if (directory / 'cold-comparison-timing-status.json').exists() else {}
                    if timing_status and (timing_status.get('operation_id') != trial['run_id'] or
                            timing_status.get('scheme') != scheme):
                        raise ValueError('cold_comparison_timing_status_identity_invalid')
                    required_timing = ('target_launch_to_ready_ms',
                        'fault_boundary_to_first_output_token_ms',
                        'fault_boundary_to_response_complete_ms',
                        'fault_boundary_to_validation_complete_ms', 'response_stream_ms')
                    for metric in required_timing:
                        per_trial['cold_comparison/' + metric] = timing[metric]
                    performance_measurement_eligible = timing.get(
                        'performance_measurement_eligible') is True
                elif (directory / 'cold-comparison-timing-status.json').exists():
                    timing_status = read(directory / 'cold-comparison-timing-status.json')
                    if (timing_status.get('operation_id') != trial['run_id'] or
                            timing_status.get('scheme') != scheme):
                        raise ValueError('cold_comparison_timing_status_identity_invalid')
                    performance_measurement_eligible = timing_status.get(
                        'performance_measurement_eligible') is True
                required = ({'gpu_kv_snapshot', 'gpu_to_host', 'local_xor_encode', 'gpu_kv_writeback',
                            'gpu_layer_readback_validation', 'scheduler_request_rebuild',
                            'model_weight_load', 'first_restored_decode_cuda',
                            'failure_to_decode_complete_harness', 'new_instance_launch_to_ready_harness'}
                            if scheme == 'two-level' else set())
                absent = sorted(required - stages)
                if absent:
                    missing.append(dict(trial=str(directory), stages=absent))
                    raise ValueError('missing required performance observations: ' + ','.join(absent))
                if scheme == 'reprefill':
                    accepted.append(str(directory))
                    contexts[key]['functional_correctness_pass'] = True
                    contexts[key].setdefault('performance_measurement_eligible_by_trial', {})[
                        trial['run_id']] = performance_measurement_eligible
                    seen.add(fingerprint)
                    for name, value in per_trial.items():
                        groups[(scheme, key, name)].append(value)
                    continue
                native = read(directory / 'native-metrics.json')
                if not native or 'rs_decode_ms' not in native:
                    raise ValueError('native metrics missing')
                for metric, value in native.items():
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid native metric: ' + metric)
                    per_trial[metric] = value
                # Commit observations only after the ENTIRE trial validates.
                for name, value in per_trial.items():
                    groups[(scheme, key, name)].append(value)
                accepted.append(str(directory))
                contexts[key]['functional_correctness_pass'] = True
                contexts[key].setdefault('performance_measurement_eligible_by_trial', {})[
                    trial['run_id']] = performance_measurement_eligible
                contexts[key].setdefault('recovery_by_trial', {})[trial['run_id']] = dict(
                    rank_correctness=(verdict or {}).get('rank_correctness', []),
                    rank_gpu_kv_mapping=(verdict or {}).get('rank_gpu_kv_mapping', []),
                    model_tp_prefix_identity=(verdict or {}).get('model_tp_prefix_identity'),
                    gpu_startup_evidence=trial.get('gpu_startup_evidence', {}))
                seen.add(fingerprint)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(trial=str(directory), reason=f'{type(exc).__name__}: {exc}'))
    rows = [dict(scheme=scheme, config_sha256=key, metric=metric,
                 tp_size=contexts[key]['tp_size'],
                 gpu_admission_mode=policies[key][0],
                 mem_fraction_static=contexts[key]['mem_fraction_static'],
                 tp_memory_imbalance_check=contexts[key]['tp_memory_imbalance_check'],
                 shared_resource_environment=contexts[key]['shared_resource_environment'],
                 functional_correctness_pass=contexts[key].get('functional_correctness_pass'),
                 isolated_performance_eligible=policies[key][1], **stats(values))
            for (scheme, key, metric), values in sorted(groups.items())]
    save(args.output, dict(schema='kv-ec-perf-summary-v2', rows=rows, rejected=rejected,
        gpu_admission_modes=sorted({p[0] for p in policies.values()}),
        cross_mode_pooling=False,
        gpu_admission_context=contexts,
        accepted_trials=accepted, missing_metrics=missing,
        quantile_policy='P50 n>=5; P95 n>=20; P99 n>=100; linear interpolation; sample stddev',
        overhead_ratio=None,
        overhead_ratio_reason='Stage E capture/export is offline sequential harness, not steady-state serving; '
            'matched online EC-on/off request latency is required before reporting protection overhead',
        warning='inclusive regions overlap; never sum into recovery E2E or subtract validation to invent production cost'))
    print(json.dumps(dict(output=str(args.output), series=len(rows), rejected=len(rejected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inventory')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=1)
    for name in ('inference', 'xor-only', 'rs-only', 'repeat'):
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--repeat', type=int, default=5)
        p.add_argument('--resource-interval', type=float, default=None if name == 'repeat' else 1.)
        if name == 'repeat':
            p.add_argument('--recovery-mode', choices=('two-level', 'reprefill'), default='two-level')
            p.add_argument('--paired-ready-reprefill', action='store_true',
                           help='run a token-bound Re-Prefill request on the same accepted Stage E target')
        if name == 'inference':
            p.add_argument('--tp', type=int, choices=(2, 4), default=2)
            p.add_argument('--scheme', choices=('no-ec', 'reprefill'), required=True)
            p.add_argument('--url', required=True)
            p.add_argument('--model', required=True)
            p.add_argument('--server-model')
            p.add_argument('--input-ids', type=Path, required=True)
            p.add_argument('--warmup', type=int, default=2)
            p.add_argument('--timeout', type=float, default=600)
            p.add_argument('--expected-model-sha256')
            p.add_argument('--expected-input-ids-sha256')
            p.add_argument('--model-identity-evidence', type=Path)
            p.add_argument('--ready-monotonic-ns', type=int,
                           help='same-host coordinator monotonic timestamp for cold-start control')
        elif name == 'xor-only':
            p.add_argument('--capture', type=Path, required=True)
            p.add_argument('--lost-rank', type=int, choices=(0, 1, 2, 3), default=1)
        elif name == 'rs-only':
            p.add_argument('--report', type=Path, required=True)
        else:
            p.add_argument('--config', type=Path, required=True)
    p = sub.add_parser('collect')
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('collect-inference')
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scheme', choices=('no-ec', 'reprefill'), required=True)
    p.add_argument('--expected-trials', type=int, default=1)
    args = parser.parse_args()
    if (getattr(args, 'repeat', 1) < 1 or getattr(args, 'warmup', 0) < 0 or
            getattr(args, 'expected_trials', 1) < 1):
        parser.error('repeat must be positive; warmup must be nonnegative')
    interval = getattr(args, 'resource_interval', None)
    if interval is not None and (not math.isfinite(interval) or interval < .5):
        parser.error('resource interval must be finite and >= 0.5 seconds')
    if args.command == 'inventory':
        save(args.output, dict(**identity(), resources=resource_snapshot()))
    elif args.command == 'preflight':
        from .perf_preflight import preflight
        report = preflight(read(args.config), args.output.resolve().parent, args.repeat)
        save(args.output, report)
        print(json.dumps(report), flush=True)
        if report['status'] != 'PASS':
            raise SystemExit(1)
    else:
        {'inference': inference, 'xor-only': local_xor, 'rs-only': rs_only,
         'repeat': repeat, 'collect': collect, 'collect-inference': collect_inference}[args.command](args)


if __name__ == '__main__':
    main()
