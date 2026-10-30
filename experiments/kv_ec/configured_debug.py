"""Explicit, foreground steps for local XOR and two-node offline RS validation.

Never SSH, kill processes, substitute synthetic buffers for KV, or claim live
GPU-pool restore. Each step invokes an existing production CLI unchanged.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    cfg = json.loads(Path(path).read_text())
    required = {'tp_size', 'gpu_ids', 'gpu_mode', 'min_free_gpu_gb',
        'mem_fraction_static', 'context_length', 'model_path', 'tokenizer_path',
        'rs', 'node0_host', 'node1_host', 'rs_host', 'rs_port', 'data_port',
        'return_port', 'snapshot_port', 'server_port', 'transport',
        'rdma_device', 'rdma_gid_index', 'remote_root', 'output_root'}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError('config fields missing or unknown')
    tp, ids = cfg['tp_size'], cfg['gpu_ids']
    if type(tp) is not int or tp not in (2, 4):
        raise ValueError('tp_size must be integer 2 or 4')
    if (type(ids) is not list or len(ids) != tp or
            any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != tp):
        raise ValueError('gpu_ids must contain exactly TP distinct physical integer indices')
    if cfg['gpu_mode'] not in ('STRICT', 'SHARED'):
        raise ValueError('gpu_mode must be STRICT or SHARED')
    for name, low, high in [('min_free_gpu_gb', 0, 100000), ('mem_fraction_static', .05, .95)]:
        value = cfg[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be a finite scalar in [{low}, {high}]')
    if type(cfg['context_length']) is not int or not 784 <= cfg['context_length'] <= 131072:
        raise ValueError('context_length must fit the 768 input + 16 output request')
    if cfg['rs'] != [2, 1] or any(type(x) is not int for x in cfg['rs']):
        raise ValueError('only two-node RS(2,1) is supported')
    if cfg['transport'] not in ('tcp', 'rdma'):
        raise ValueError('transport must be tcp or rdma')
    if cfg['rs_host'] != cfg['node1_host'] or cfg['node0_host'] == cfg['node1_host']:
        raise ValueError('worker placement requires distinct nodes and rs_host == node1_host')
    for key in ('node0_host', 'node1_host', 'rs_host', 'rdma_device'):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f'{key} must be nonempty')
    for key in ('model_path', 'tokenizer_path', 'remote_root', 'output_root'):
        if not isinstance(cfg[key], str) or not Path(cfg[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path')
    ports = [cfg[k] for k in ('rs_port', 'data_port', 'return_port', 'snapshot_port', 'server_port')]
    if any(type(p) is not int or not 1 <= p <= 65535 for p in ports) or len(set(ports)) != len(ports):
        raise ValueError('ports must be distinct integers in [1,65535]')
    gid = cfg['rdma_gid_index']
    if gid is not None and (type(gid) is not int or gid < 0):
        raise ValueError('rdma_gid_index must be null or a nonnegative integer')
    if cfg['transport'] == 'rdma' and gid is None:
        raise ValueError('RDMA requires explicit on-site rdma_gid_index; no TCP fallback')
    return cfg


def write(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


def fingerprint(cfg):
    """Include untracked experiment/runtime source; exclude host-local GPU/path choices."""
    sources = {}
    for base in ('experiments/kv_ec', 'python/sglang/srt'):
        for path in sorted((ROOT / base).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.c', '.h', '.cc', '.cpp', '.cu'):
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # GID, physical GPUs and local model paths can legitimately differ by host.
    config = {k: v for k, v in cfg.items() if k not in
              ('gpu_ids', 'model_path', 'tokenizer_path', 'remote_root', 'output_root', 'rdma_gid_index')}
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        config=config, source_sha256=sources)


def check_model(cfg):
    root = Path(cfg['model_path'])
    if not (root / 'config.json').is_file() or not list(root.glob('*.safetensors')):
        raise ValueError(f'local_model_files_required: {root}; need config.json and *.safetensors')
    tok = Path(cfg['tokenizer_path'])
    if not (tok / 'tokenizer_config.json').is_file() or not any(
            (tok / name).is_file() for name in ('tokenizer.json', 'tokenizer.model', 'vocab.json')):
        raise ValueError(f'local_tokenizer_files_required: {tok}')
    config = json.loads((root / 'config.json').read_text())
    if any(config.get(k) != v for k, v in dict(model_type='qwen3', num_hidden_layers=36,
            num_key_value_heads=8, num_attention_heads=32, head_dim=128).items()):
        raise ValueError('unsupported model geometry: require Qwen3-8B 36 layers/8 KV heads/128 head_dim')


def preflight(cfg):
    check_model(cfg)
    from .debug_resources import admit
    return admit(list(map(str, cfg['gpu_ids'])), cfg['model_path'], cfg['tp_size'],
        cfg['context_length'], cfg['mem_fraction_static'],
        shared=cfg['gpu_mode'] == 'SHARED', minimum=cfg['min_free_gpu_gb'])


def tokenizer_digest(cfg):
    root = Path(cfg['tokenizer_path'])
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.name.startswith(('tokenizer', 'vocab', 'merges', 'added_tokens', 'special_tokens'))):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def require_pair(cfg, run, node):
    record = json.loads((run / f'source-match-node{node}.json').read_text())
    if record['fingerprint'] != fingerprint(cfg) or record['state'] != 'MATCH':
        raise ValueError('source/config changed after peer comparison; rerun preflight and compare')


def require_snapshot(cfg, run, node):
    from .multinode.kv_hierarchical import _load_node
    out = run / f'node{node}'
    manifest, *_ = _load_node(out / f'{run.name}-node{node}-chunk-1', f'node{node}')
    response = json.loads((out / 'response.json').read_text())
    identity = json.loads((out / f'{run.name}-node{node}-initialization.json').read_text())['identity']
    if (manifest['tp_size'] != cfg['tp_size'] or manifest['provenance'] != 'sglang_request' or
            manifest['request_id'] != hashlib.sha256(response['request_id'].encode()).hexdigest() or
            manifest['model_identifier'] != response['model_identifier'] or
            manifest['model_sha256'] != identity['model_sha256'] or
            manifest['token_ids_sha256'] != response['input_chunks'][0]['token_ids_sha256'] or
            len(response['output_ids']) != 16):
        raise ValueError('snapshot/request/model/TP mismatch; not real captured request evidence')
    pre = json.loads((run / f'preflight-node{node}.json').read_text())
    from .gpu_admission import normalized_uuid
    for rank, meta in enumerate(manifest['ranks']):
        if (meta.get('physical_gpu_index') != str(cfg['gpu_ids'][rank]) or
                meta.get('gpu_admission_mode') != cfg['gpu_mode'].lower() or
                normalized_uuid(meta['gpu_uuid']) != normalized_uuid(pre['resources'][rank]['gpu_uuid'])):
            raise ValueError('snapshot physical GPU/resource policy binding mismatch')


def compare_preflights(local, peer, current, node):
    if (peer['node'] != 1 - node or local['node'] != node or
            local['fingerprint'] != current or local['fingerprint'] != peer['fingerprint'] or
            local['model_identity']['model_sha256'] != peer['model_identity']['model_sha256'] or
            local['tokenizer_sha256'] != peer['tokenizer_sha256']):
        raise ValueError('two-node source/config/model/tokenizer mismatch')


def require_request_success(run, node):
    record = json.loads((run / f'request-node{node}-exit.json').read_text())
    if record['exit_code'] != 0 or record['memory_observation_errors']:
        raise ValueError('request failed or GPU memory observation incomplete')
    for kind in ('parity', 'capture'):
        path = run / f'{kind}-node{node}.log'
        if path.exists() and any(marker in path.read_text(errors='replace').lower()
                for marker in ('out of memory', 'cuda error:', 'nccl error')):
            raise ValueError('GPU runtime failure in server log; no correctness PASS')


def command(cfg, run, node, step):
    """Construct real CLI arguments; synthetic and snapshot paths remain distinct."""
    out = run / f'node{node}'
    snap, rid = f'{run.name}-node{node}', f'{run.name}-request{node}'
    base = [sys.executable, '-m']
    transport = ['--transport', cfg['transport']]
    if cfg['transport'] == 'rdma':
        transport += ['--rdma-device', cfg['rdma_device'], '--rdma-gid-index', str(cfg['rdma_gid_index'])]
    if step in ('parity', 'capture', 'baseline'):
        args = base + ['experiments.kv_ec.launch_debug', '--mode', step,
            '--tp', str(cfg['tp_size']), '--gpus', ','.join(map(str, cfg['gpu_ids'])),
            '--gpu-mode', cfg['gpu_mode'], '--min-free-gpu-gb', str(cfg['min_free_gpu_gb']),
            '--model', cfg['model_path'], '--tokenizer', cfg['tokenizer_path'],
            '--port', str(cfg['server_port']), '--context-length', str(cfg['context_length']),
            '--mem-fraction-static', str(cfg['mem_fraction_static']), '--output', str(out),
            '--snapshot-id', snap + ('-parity' if step == 'parity' else ''),
            '--request-id', rid, '--token-count', '256', '--layers', 'all']
        if step == 'capture':
            args += ['--chunk-size', '256']
        if step == 'parity':
            args += ['--protection-mode', 'FAST_PARITY', '--parity-socket', str(run / f'manager{node}.sock')]
        return args
    if step == 'manager':
        return base + ['experiments.kv_ec.parity_manager', '--socket', str(run / f'manager{node}.sock'),
                       '--timeout', '120', '--test-oracle']
    if step == 'request':
        return base + ['experiments.kv_ec.request_debug', 'request', '--url', f'http://127.0.0.1:{cfg["server_port"]}',
            '--model', cfg['model_path'], '--request-id', rid, '--prompt-tokens', '768', '--stream-metrics',
            '--output', str(out / 'response.json')]
    if step in ('worker', 'coordinator'):
        args = base + ['experiments.kv_ec.multinode.' + step] + transport + [
            '--control-port', str(cfg['rs_port']), '--data-port', str(cfg['data_port']),
            '--return-port', str(cfg['return_port']), '--d0-bytes', '1048576', '--d1-bytes', '786432',
            '--experiment-id', run.name, '--stripe-id', run.name + '-synthetic',
            '--output', str(run / ('synthetic-' + step))]
        return args + (['--return-host', cfg['node0_host']] if step == 'worker' else ['--node-b', cfg['rs_host']])
    if step in ('send', 'receive'):
        return base + ['experiments.kv_ec.multinode.kv_hierarchical', step] + transport + [
            '--gpu-mode', cfg['gpu_mode'],
            '--snapshot', str(out / (snap + '-chunk-1')), '--node-id', f'node{node}',
            '--stripe-id', run.name + '-real-kv', '--experiment-id', run.name,
            '--data-port', str(cfg['snapshot_port']), '--request-metrics', str(out / 'response.json'),
            '--output', str(run / ('snapshot-' + step))] + (
            ['--peer', cfg['rs_host']] if step == 'send' else ['--bind-host', '0.0.0.0'])
    if step == 'verify-rs':
        return base + ['experiments.kv_ec.multinode.verify_hierarchical', str(run / 'snapshot-receive'),
                      '--local-snapshot', str(out / (snap + '-chunk-1'))]
    raise ValueError('unsupported step')


def observe(cfg):
    from .gpu_memory_budget import read_gpu_memory, read_gpu_tasks
    rows, tasks = read_gpu_memory(), read_gpu_tasks()
    return [dict(gpu_index=str(i), **rows[str(i)], other_tasks=tasks.get(rows[str(i)]['uuid'], []))
            for i in cfg['gpu_ids']]


def execute(argv, cfg, run, step, node):
    """Foreground child, durable logs, periodic read-only GPU observations."""
    stem = run / f'{step}-node{node}'
    metadata = dict(command=argv, tp_size=cfg['tp_size'], gpu_mode=cfg['gpu_mode'],
        shared_gpu=cfg['gpu_mode'] == 'SHARED', isolated_performance_eligible=False,
        config=cfg, real_gpu_restore_verified=False)
    write(str(stem) + '-launch.json', metadata)
    gpu_step = step in ('parity', 'capture', 'baseline', 'request')
    with Path(str(stem) + '.log').open('x') as log, Path(str(stem) + '-memory.jsonl').open('x') as memory:
        errors = []
        def sample(phase):
            try:
                record = dict(time=time.time(), phase=phase, gpus=observe(cfg))
            except Exception as exc:
                errors.append(str(exc))
                record = dict(time=time.time(), phase=phase, observation_error=str(exc))
            memory.write(json.dumps(record) + '\n')
            memory.flush()
        if gpu_step:
            sample('before_child')
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        # Terminal Ctrl-C reaches this foreground process group. Never discover
        # or signal unrelated workers; retain logs even for interrupted trials.
        try:
            ready = False
            while child.poll() is None:
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if gpu_step:
                        sample('during')
                        if not ready and step in ('parity', 'capture', 'baseline'):
                            import urllib.request
                            try:
                                with urllib.request.urlopen(f'http://127.0.0.1:{cfg["server_port"]}/health', timeout=.5):
                                    ready = True
                                sample('after_server_ready')
                            except OSError:
                                pass
        except KeyboardInterrupt:
            child.wait()
        finally:
            if gpu_step:
                sample('after_child')
        code = child.returncode
    write(str(stem) + '-exit.json', dict(exit_code=code, memory_observation_errors=errors, **metadata))
    if code or errors:
        raise RuntimeError(f'{step} exited {code}, memory_errors={errors}; inspect {stem}.log; no PASS claimed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run', type=Path)
    p.add_argument('--node', type=int, choices=(0, 1), default=0)
    p.add_argument('--peer-file', type=Path, help='peer preflight JSON, copied explicitly by the operator')
    p.add_argument('step', choices=('init', 'fingerprint', 'preflight', 'compare', 'manager', 'parity',
        'capture', 'baseline', 'request', 'check-parity', 'worker', 'coordinator', 'send', 'receive', 'verify-rs'))
    args = p.parse_args()
    try:
        cfg = load(args.config)
        if args.step == 'fingerprint':
            print(json.dumps(fingerprint(cfg), sort_keys=True, indent=2))
            return
        if args.step == 'init':
            import tempfile
            parent = Path(cfg['output_root'])
            parent.mkdir(parents=True, exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'), dir=parent))
            write(run / 'config.json', cfg)
            print(run)
            return
        if args.run is None or not args.run.is_absolute():
            raise ValueError('--run must be an absolute new run directory from init')
        run = args.run
        if run.parent.resolve() != Path(cfg['output_root']).resolve():
            raise ValueError('--run must be a direct child of configured output_root')
        run.mkdir(parents=True, exist_ok=True)
        node = args.node
        session_path = run / f'config-node{node}.json'
        if session_path.exists():
            if json.loads(session_path.read_text()) != cfg:
                raise ValueError('config drift within a run; create a new run')
        else:
            write(session_path, cfg)
        if (args.step in ('worker', 'receive', 'verify-rs') and node != 1 or
                args.step in ('coordinator', 'send') and node != 0):
            raise ValueError('worker/receive/verify-rs require node1; coordinator/send require node0')
        if args.step == 'preflight':
            from .debug_capture import source_identity
            result = dict(tp_size=cfg['tp_size'], gpu_mode=cfg['gpu_mode'],
                shared_gpu=cfg['gpu_mode'] == 'SHARED', isolated_performance_eligible=False,
                node=node, resources=preflight(cfg), fingerprint=fingerprint(cfg),
                model_identity=source_identity(cfg['model_path']), tokenizer_sha256=tokenizer_digest(cfg))
            write(run / f'preflight-node{node}.json', result)
            print('GPU_PREFLIGHT_PASS=true (budget estimate; not a guarantee against dynamic OOM)')
            return
        if args.step == 'compare':
            if args.peer_file is None:
                raise ValueError('compare requires --peer-file from the other node preflight')
            local = json.loads((run / f'preflight-node{node}.json').read_text())
            peer = json.loads(args.peer_file.read_text())
            compare_preflights(local, peer, fingerprint(cfg), node)
            write(run / f'source-match-node{node}.json', dict(state='MATCH', fingerprint=local['fingerprint']))
            print('TWO_NODE_IDENTITY_MATCH=true')
            return
        if args.step == 'check-parity':
            require_request_success(run, node)
            from .parity_manager import submit_once
            snapshot = f'{run.name}-node{node}-parity'
            deadline = time.monotonic() + 60
            while True:
                status = submit_once(run / f'manager{node}.sock', {'kind': 'status', 'snapshot_id': snapshot}, b'')
                if status.get('state') in ('COMMITTED', 'FAILED', 'EXPIRED') or time.monotonic() >= deadline:
                    break
                time.sleep(.2)
            response = json.loads((run / f'node{node}' / 'response.json').read_text())
            identity = json.loads((run / f'node{node}' / (snapshot + '-initialization.json')).read_text())['identity']
            expected = 36 * 256 * (8 // cfg['tp_size']) * 128 * 2 * 2
            if (status.get('state') != 'COMMITTED' or status.get('tp_size') != cfg['tp_size'] or
                    status.get('provenance') != 'sglang_request' or
                    status.get('request_id') != hashlib.sha256(response['request_id'].encode()).hexdigest() or
                    status.get('token_ids_sha256') != response['input_chunks'][0]['token_ids_sha256'] or
                    status.get('model_sha256') != identity['model_sha256'] or
                    status.get('model_identifier') != response['model_identifier'] or
                    status.get('layers') != list(range(36)) or status.get('token_count') != 256 or
                    status.get('token_start') != 0 or status.get('dtype') != 'torch.bfloat16' or
                    len(response['output_ids']) != 16 or
                    status.get('parity_bytes') != expected or
                    set(status.get('rank_hashes', {})) != {str(i) for i in range(cfg['tp_size'])} or
                    status.get('oracle_checks') != [dict(lost_rank=i, byte_exact=True) for i in range(cfg['tp_size'])]):
                raise ValueError('local XOR incomplete or Oracle not verified; no PASS')
            from .gpu_admission import normalized_uuid
            pre = json.loads((run / f'preflight-node{node}.json').read_text())
            timings = sorted((run / f'node{node}' / snapshot).glob('rank_*.timing.json'))
            records = [json.loads(path.read_text()) for path in timings]
            if len(records) != cfg['tp_size'] or {r['tp_rank'] for r in records} != set(range(cfg['tp_size'])):
                raise ValueError('incomplete rank capture diagnostics')
            for record in records:
                rank = record['tp_rank']
                if (record['physical_gpu_index'] != str(cfg['gpu_ids'][rank]) or
                        normalized_uuid(record['gpu_uuid']) != normalized_uuid(pre['resources'][rank]['gpu_uuid']) or
                        record['gpu_admission_mode'] != cfg['gpu_mode'].lower() or
                        record['tp_size'] != cfg['tp_size'] or record['model_sha256'] != status['model_sha256'] or
                        record['request_id'] != status['request_id'] or
                        record['token_ids_sha256'] != status['token_ids_sha256'] or
                        record['parity_protection'] != 'ENQUEUED'):
                    raise ValueError('rank capture diagnostic identity/device mismatch')
            write(run / f'parity-commit-node{node}.json', dict(status,
                shared_gpu=cfg['gpu_mode'] == 'SHARED', gpu_mode=cfg['gpu_mode'],
                isolated_performance_eligible=False, gpu_restore_verified=False))
            print('LOCAL_XOR_COMMITTED=true; ALL_RANK_ORACLES_BYTE_EXACT=true; GPU_RESTORE_VERIFIED=false')
            return
        if args.step in ('parity', 'capture', 'baseline'):
            preflight(cfg)  # Recheck dynamic free memory immediately before spawn.
        if args.step in ('send', 'receive'):
            require_pair(cfg, run, node)
            require_request_success(run, node)
            require_snapshot(cfg, run, node)
        (run / f'node{node}').mkdir(exist_ok=True)
        argv = command(cfg, run, node, args.step)
        print('Foreground child logs:', run / f'{args.step}-node{node}.log', flush=True)
        execute(argv, cfg, run, args.step, node)
    except (ValueError, OSError, RuntimeError) as exc:
        if args.run is not None and args.run.is_dir():
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', prefix=f'failure-{args.step}-', suffix='.json',
                                              dir=args.run, delete=False) as log:
                json.dump(dict(step=args.step, error=str(exc), functional_correctness_pass=False), log)
        p.exit(1, f'FAIL: {exc}\n')


if __name__ == '__main__':
    main()
