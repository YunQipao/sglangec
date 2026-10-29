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
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.glob('perf_*.py'))}
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
    info = http(args.url.rstrip('/') + '/server_info')
    required = dict(tp_size=2, disable_radix_cache=True, disable_overlap_schedule=True,
                    chunked_prefill_size=-1, max_running_requests=1, attention_backend='triton')
    if any(info.get(k) != v for k, v in required.items()):
        raise ValueError('baseline server flags do not match Stage E: ' + str(required))
    graph = info.get('cuda_graph_config', {})
    if any(graph.get(k, {}).get('backend') != 'disabled' for k in ('decode', 'prefill')):
        raise ValueError('baseline requires disabled CUDA graphs, matching Stage E')
    if info.get('model_path') != args.model:
        raise ValueError('baseline model_path mismatch')
    # This is exactly Stage D submit's policy (temperature=0 normalizes to
    # greedy top_k=1), 17 output IDs = one pending y0 plus 16 native decodes.
    sampling = dict(max_new_tokens=17, temperature=0, ignore_eos=True, top_p=1.,
                    top_k=1, min_p=0., min_new_tokens=0, n=1,
                    frequency_penalty=0., presence_penalty=0., repetition_penalty=1.)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = identity()
    model_files = [Path(args.model) / 'config.json', *sorted(Path(args.model).glob('*.safetensors.index.json'))]
    meta.update(scheme=args.scheme, model=args.model, model_files={str(p): digest(p)
        for p in model_files if p.exists()}, input_sha256=digest(args.input_ids), prefix_tokens=256,
        tp_size=2, generation_tokens=17, sampling=sampling, server_info=info,
        repeat=args.repeat, warmup=args.warmup, cache_policy='server radix disabled; cached_tokens must be zero',
        limitation='reprefill starts at explicit client resubmission, excludes detection and startup')
    save(args.output / 'metadata.json', meta)
    with Sampler(args.output / 'resources.jsonl', args.resource_interval):
        for i in range(args.warmup + args.repeat):
            request = dict(input_ids=ids, sampling_params=sampling, stream=True,
                           rid=f'perf-{uuid.uuid4().hex}', return_logprob=False)
            started, first, final = time.monotonic_ns(), None, None
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
                    if item.get('meta_info', {}).get('completion_tokens', 0) > 0 and first is None:
                        first = time.monotonic_ns()
                    final = item
            ended = time.monotonic_ns()
            if final is None or first is None:
                raise ValueError('missing streaming generation evidence')
            usage = final['meta_info']
            if usage.get('cached_tokens') != 0 or usage.get('prompt_tokens') != 256 or usage.get('completion_tokens') != 17:
                raise ValueError('cached/short/wrong-prefix inference cannot enter baseline statistics')
            output_ids = final.get('output_ids')
            if not isinstance(output_ids, list) or len(output_ids) != 17:
                raise ValueError('server must return all 17 output_ids for correctness comparison')
            save(args.output / f'request-{i:03d}.json', dict(warmup=i < args.warmup,
                scheme=args.scheme, start_ns=started, end_ns=ended,
                request_ms=(ended-started)/1e6, prefill_ttft_ms=(first-started)/1e6,
                source='client_monotonic_ns', response=final, output_token_ids=output_ids,
                correctness='raw token IDs recorded; compare against Stage E reference separately'))


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
    if meta['tp_size'] != 2 or state['token_count'] != 256:
        raise ValueError('real TP2/256 capture required')
    save(args.output / 'metadata.json', dict(**identity(), scheme='xor-only',
        capture=str(args.capture), capture_sha256=hashlib.sha256(source.load()).hexdigest(),
        runtime_identity=state['runtime_identity'], prefix_tokens=256, tp_size=2,
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
    from .perf_preflight import preflight, validate_config
    from .perf_stage_e import process_identity, signal_owned
    import signal
    config = read(args.config)
    validate_config(config)
    if Path.cwd().resolve() != Path(config['nodes'][0]['repo']).resolve():
        raise ValueError('repeat must run from configured node0 repo on 232')
    config['resource_interval'] = args.resource_interval or config['resource_interval']
    argv = [sys.executable, '-m', 'experiments.kv_ec.perf_stage_e', 'trial']
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'metadata.json', dict(**identity(), config=config,
         repeat=args.repeat, argv=argv, scheme='two-level', resource_interval=config['resource_interval']))
    save(args.output / 'config.json', config)
    admission = preflight(config, args.output.resolve(), args.repeat)
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
                   KV_EC_PERF_ROLE='coordinator-232')
        started = time.monotonic_ns()
        raw = dict(run_id=run_id, returncode=None, status='failed', pass_fail='FAIL',
                   commit_sha=admission['nodes'][0]['checks']['inventory']['commit_sha'],
                   scheme='two-level', real_rdma_eligible=False,
                   remote_root=str(Path(config['remote_root']) / run_id))
        try:
            # Recheck idle GPUs/ports/RS epoch and all *remaining* slots before
            # each trial, including fresh batches on a long-lived server.
            current = preflight(config, root, args.repeat - i, admission['rs']['server_epoch'])
            save(root / 'preflight.json', current)
            if current['status'] != 'PASS':
                raise ValueError('trial preflight failed: ' + current['error'])
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
            validate_trial(root, run_id)
            inventories = [read(root / f'inventory-{node}.json') for node in (232, 233)]
            if inventories[0]['host'] == inventories[1]['host']:
                raise ValueError('two physical hosts required')
            if any(x['commit_sha'] != inventories[0]['commit_sha'] or
                   x['tracked_diff_sha256'] != inventories[0]['tracked_diff_sha256'] or
                   x['benchmark_files'] != inventories[0]['benchmark_files'] for x in inventories):
                raise ValueError('two host source identities differ')
            raw.update(status='accepted', pass_fail='PASS', real_rdma_eligible=True,
                decode_correctness=True, verdict_sha256=digest(root / 'stage-e-check.json'))
        except (Exception, KeyboardInterrupt) as exc:
            raw['failure_reason'] = f'{type(exc).__name__}: {exc}'
        raw['trial_command_ms'] = (time.monotonic_ns()-started)/1e6
        save(root / 'trial.json', raw)
        print(json.dumps(dict(trial=str(root), **raw)), flush=True)
        if raw['status'] != 'accepted':
            raise SystemExit(1)


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
                       ('recomputed_prefix_tokens', 0), ('tp_size', 2)):
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
    for key in ('commit_sha', 'tracked_diff_sha256', 'benchmark_files'):
        if inventories[0][key] != metadata[key]:
            raise ValueError('trial_source_changed_since_batch_admission: ' + key)
    return verdict


def collect(args):
    groups, rejected, accepted, missing = defaultdict(list), [], [], []
    seen = set()
    for root in args.roots:
        metadata = read(root / 'metadata.json')
        # No pooling configurations or schemes. Each group has its own source.
        key = hashlib.sha256(json.dumps(dict(config=metadata.get('config', metadata.get('server_info',
            metadata.get('runtime_identity', {}))), source={k: metadata.get(k) for k in
            ('commit_sha', 'tracked_diff_sha256', 'benchmark_files')}), sort_keys=True).encode()).hexdigest()
        scheme = metadata['scheme']
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
                if (trial['status'] != 'accepted' or trial['real_rdma_eligible'] is not True or
                        trial.get('pass_fail') != 'PASS'):
                    raise ValueError(trial.get('failure_reason', 'not an accepted real trial'))
                validate_trial(directory, trial['run_id'])
                fingerprint = (trial['run_id'], trial['verdict_sha256'])
                if fingerprint in seen:
                    raise ValueError('duplicate trial; cannot count same PASS twice')
                if trial['verdict_sha256'] != digest(directory / 'stage-e-check.json'):
                    raise ValueError('trial_verdict_digest_mismatch')
                per_trial = defaultdict(float)
                stages = set()
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
                required = {'gpu_kv_snapshot', 'gpu_to_host', 'local_xor_encode', 'gpu_kv_writeback',
                            'gpu_layer_readback_validation', 'scheduler_request_rebuild',
                            'model_weight_load', 'first_restored_decode_cuda',
                            'failure_to_decode_complete_harness', 'new_instance_launch_to_ready_harness'}
                absent = sorted(required - stages)
                if absent:
                    missing.append(dict(trial=str(directory), stages=absent))
                    raise ValueError('missing required performance observations: ' + ','.join(absent))
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
                seen.add(fingerprint)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(trial=str(directory), reason=f'{type(exc).__name__}: {exc}'))
    rows = [dict(scheme=scheme, config_sha256=key, metric=metric, **stats(values))
            for (scheme, key, metric), values in sorted(groups.items())]
    save(args.output, dict(schema='kv-ec-perf-summary-v2', rows=rows, rejected=rejected,
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
        if name == 'inference':
            p.add_argument('--scheme', choices=('no-ec', 'reprefill'), required=True)
            p.add_argument('--url', required=True)
            p.add_argument('--model', required=True)
            p.add_argument('--input-ids', type=Path, required=True)
            p.add_argument('--warmup', type=int, default=2)
            p.add_argument('--timeout', type=float, default=600)
        elif name == 'xor-only':
            p.add_argument('--capture', type=Path, required=True)
            p.add_argument('--lost-rank', type=int, choices=(0, 1), default=1)
        elif name == 'rs-only':
            p.add_argument('--report', type=Path, required=True)
        else:
            p.add_argument('--config', type=Path, required=True)
    p = sub.add_parser('collect')
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, 'repeat', 1) < 1 or getattr(args, 'warmup', 0) < 0:
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
         'repeat': repeat, 'collect': collect}[args.command](args)


if __name__ == '__main__':
    main()
