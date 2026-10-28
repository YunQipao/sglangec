"""Opt-in 232 coordinator for the existing Stage E CLI, no new recovery path.

The RS server is explicitly started by the operator on 233 and is never stopped
here. Only process groups created by this trial's `service` supervisor stop.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import zipfile

from .perf_benchmark import Sampler, digest, read, save

E = 'experiments.kv_ec.verify_rs_cold_resume'
D = 'experiments.kv_ec.verify_gpu_cold_resume'
B = 'experiments.kv_ec.perf_benchmark'
SELF = 'experiments.kv_ec.perf_stage_e'


def service(root, argv, interval):
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'server.log').open('x') as log, Sampler(root / 'resources.jsonl', interval):
        started = time.monotonic_ns()
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'owned-process.json', dict(pid=proc.pid, argv=argv, start_ns=started))
        try:
            while proc.poll() is None and not (root / 'stop').exists():
                time.sleep(.2)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=10)
        save(root / 'exit.json', dict(returncode=proc.returncode, end_ns=time.monotonic_ns()))


class Host:
    def __init__(self, cfg, root, run_id, node):
        self.cfg, self.root, self.run_id, self.node = cfg, Path(root), run_id, node
        self.env = dict(PYTHONPATH=cfg['repo'] + '/python:' + cfg['repo'],
            KV_EC_PERF_DIR=str(self.root / 'perf'), KV_EC_PERF_RUN_ID=run_id,
            KV_EC_PERF_ROLE=f'node{node}-{232+node}')

    def argv(self, module, *args):
        cmd = [self.cfg['python'], '-m', module, *map(str, args)]
        if self.cfg.get('ssh'):
            shell = 'cd ' + shlex.quote(self.cfg['repo']) + ' && exec env ' + shlex.join(
                [f'{k}={v}' for k, v in self.env.items()] + cmd)
            return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.cfg['ssh'], shell]
        return ['env', *[f'{k}={v}' for k, v in self.env.items()], *cmd]

    def run(self, module, *args, log=None, timeout=1200):
        cmd = self.argv(module, *args)
        result = subprocess.run(cmd, cwd=self.cfg['repo'] if not self.cfg.get('ssh') else None,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'node{self.node} exit={result.returncode}: {shlex.join(cmd)}')


def parallel(*actions):
    with ThreadPoolExecutor(max_workers=len(actions)) as pool:
        futures = [pool.submit(action) for action in actions]
        # Inspect every independent result; no orphaned background trial work.
        errors = []
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def scp(host, remote, local, upload=False):
    endpoint = host.cfg['ssh'] + ':' + str(remote)
    args = [str(local), endpoint] if upload else [endpoint, str(local)]
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', *args], check=True, timeout=120)


def native_metrics(root, n0, n1, recovery, delivery):
    from .rs_recovery_source import native_stats
    values = {}
    for node, export in enumerate((n0, n1)):
        receipt = export['receipt']
        metric = receipt['metrics']
        native = metric['rdma_native']
        native_stats(native, receipt['node_descriptor']['length'])
        for name in ('rdma_upload_wall_ms', 'global_commit_wait_ms', 'server_rs_encode_ms', 'server_commit_ms'):
            values[f'node{node}/{name}'] = metric[name]
        for name in ('device_setup_ms', 'memory_registration_ms', 'qp_setup_ms', 'exchange_ms', 'completion_ms', 'elapsed_s'):
            if name in native:
                values[f'node{node}/upload_native/{name}'] = native[name] * (1000 if name == 'elapsed_s' else 1)
    transfer = recovery['transfer']
    values['rs_decode_ms'] = transfer['rs_reconstruct_ms']
    for leg in ('client_send_stats', 'receive_stats', 'send_stats', 'client_receive_stats'):
        native = transfer[leg]
        native_stats(native, native['bytes'])
        for name in ('device_setup_ms', 'memory_registration_ms', 'qp_setup_ms', 'exchange_ms', 'completion_ms', 'elapsed_s'):
            if name in native:
                values[f'recovery/{leg}/{name}'] = native[name] * (1000 if name == 'elapsed_s' else 1)
    native = delivery['send_stats']
    native_stats(native, delivery['archive_bytes'])
    for name in ('device_setup_ms', 'memory_registration_ms', 'qp_setup_ms', 'exchange_ms', 'completion_ms', 'elapsed_s'):
        if name in native:
            values[f'delivery/send/{name}'] = native[name] * (1000 if name == 'elapsed_s' else 1)
    save(root / 'native-metrics.json', values)


def trial():
    cfg = read(os.environ['KV_EC_BENCH_CONFIG'])
    local = Path(os.environ['KV_EC_BENCH_TRIAL'])
    rid = os.environ['KV_EC_PERF_RUN_ID']
    remote = Path(cfg.get('remote_root', '/tmp/kv-ec-perf')) / rid
    hosts = [Host(cfg['nodes'][0], local, rid, 0), Host(cfg['nodes'][1], remote, rid, 1)]
    a, b = hosts
    b.run(SELF, 'prepare', '--root', remote)
    services, logs = [], []
    def start(h, name, module, *args):
        log = (local / f'{name}-node{h.node}.log').open('x')
        logs.append(log)
        proc = subprocess.Popen(h.argv(SELF, 'service', '--root', h.root / name,
            '--resource-interval', cfg.get('resource_interval', 1), '--', h.cfg['python'], '-m', module, *map(str, args)),
            cwd=h.cfg['repo'] if not h.cfg.get('ssh') else None, stdout=log, stderr=subprocess.STDOUT)
        services.append((h, name, proc))
        return proc
    def stop(h, name):
        h.run(SELF, 'stop', '--root', h.root / name)
        owned = next(p for host, label, p in services if host is h and label == name)
        if owned.wait(timeout=90) != 0:
            raise RuntimeError('owned service supervisor failed: ' + name)
    rs = cfg['rs_server']
    def rdma(h):
        return ['--remote-host', rs['host'], '--remote-port', rs['port'],
                '--rdma-device', h.cfg.get('rdma_device', 'mlx5_0'),
                '--rdma-gid-index', h.cfg['gid_index'], '--rdma-port-num', 1, '--timeout', 600]
    try:
        parallel(*[lambda h=h: h.run(B, 'inventory', '--output', h.root / f'inventory-{232+h.node}.json') for h in hosts])
        scp(b, remote / 'inventory-233.json', local / 'inventory-233.json')
        for h in hosts:
            n = h.node
            start(h, 'origin', D, 'capture', '--model', h.cfg['model'],
                '--template-checkpoint', h.cfg['template'], '--input-ids', h.cfg['input_ids'],
                '--bundle', h.root / 'capture', '--output', h.root / f'node{n}-reference',
                '--gpus', h.cfg['gpus'], '--port', h.cfg['origin_port'],
                '--mem-fraction-static', cfg.get('mem_fraction_static', .60), '--timeout', 120)
        def capture(h):
            # Admission/source hashing precedes run.json; wait without submitting
            # a request or claiming readiness before the launcher writes it.
            h.run(SELF, 'wait-file', '--path', h.root / f'node{h.node}-reference/run.json')
            h.run(SELF, 'wait-health', '--url', f'http://127.0.0.1:{h.cfg["origin_port"]}',
                  '--service-root', h.root / 'origin')
            h.run(D, 'submit', '--output', h.root / f'node{h.node}-reference',
                  '--url', f'http://127.0.0.1:{h.cfg["origin_port"]}')
            h.run(D, 'reference', '--output', h.root / f'node{h.node}-reference')
            h.run(E, 'claim', '--capture', h.root / 'capture', '--output', h.root / f'claim{h.node}.json')
        parallel(*[lambda h=h: capture(h) for h in hosts])
        for h in hosts:
            stop(h, 'origin')
        scp(b, remote / 'claim1.json', local / 'claim1.json')
        a.run(E, 'plan', '--node0-claim', local / 'claim0.json', '--node1-claim', local / 'claim1.json',
              '--run-id', rid, '--output', local / 'plan.json')
        scp(b, remote / 'plan.json', local / 'plan.json', upload=True)
        parallel(*[lambda h=h: h.run(E, 'export', '--plan', h.root / 'plan.json',
            '--capture', h.root / 'capture', '--node-id', h.node, *rdma(h),
            '--output', h.root / f'export{h.node}') for h in hosts])
        scp(b, remote / 'export1/export.json', local / 'export1.json')
        a.run(E, 'protect', '--plan', local / 'plan.json', '--node0-export', local / 'export0/export.json',
            '--node1-export', local / 'export1.json', '--remote-host', rs['host'], '--remote-port', rs['port'],
            '--output', local / 'protection.json')
        # Starts after successful protection and explicit controlled D1 loss.
        # All E2E timestamps below are on this 232 coordinator. SSH/control and
        # independent checks remain included and are explicitly harness costs.
        recovery_start = time.monotonic_ns()
        restore_id = read(local / 'capture/request_state.json')['restore_id']
        a.run(E, 'recover', '--protection', local / 'protection.json',
            '--survivor', local / f'export0/{restore_id}/global_node.bin', *rdma(a),
            '--rdma-return-host', cfg['return_host'], '--rdma-return-port', cfg['return_port'],
            '--output', local / 'recovered-source')
        # Native receiver, not a model service. It exits after one transfer.
        receive = subprocess.Popen(b.argv(E, 'receive-source', '--remote-host', cfg['return_host'],
            '--remote-port', cfg['delivery_port'], '--rdma-device', b.cfg.get('rdma_device', 'mlx5_0'),
            '--rdma-gid-index', b.cfg['gid_index'], '--timeout', 180, '--output', remote / 'rs-source'))
        a.run(E, 'send-source', '--bundle', local / 'recovered-source', '--remote-host', rs['host'],
            '--remote-port', cfg['delivery_port'], '--rdma-device', a.cfg.get('rdma_device', 'mlx5_0'),
            '--rdma-gid-index', a.cfg['gid_index'], '--timeout', 180, '--output', local / 'delivery-send.json')
        if receive.wait(timeout=240) != 0:
            raise RuntimeError('native delivery receiver failed')
        scp(b, remote / 'delivery-send.json', local / 'delivery-send.json', upload=True)
        b.run(E, 'check-source', '--bundle', remote / 'rs-source', '--capture', remote / 'capture',
              '--send-receipt', remote / 'delivery-send.json')
        target_started = time.monotonic_ns()
        start(b, 'target', E, 'launch', '--model', b.cfg['model'], '--bundle', remote / 'capture',
            '--rs-source', remote / 'rs-source', '--reference', remote / 'node1-reference',
            '--ledger', remote / 'restore-ledger', '--output', remote / 'cold-new', '--gpus', b.cfg['gpus'],
            '--port', cfg['target_port'], '--mem-fraction-static', cfg.get('mem_fraction_static', .60), '--timeout', 120)
        b.run(SELF, 'wait-file', '--path', remote / 'cold-new/run.json')
        b.run(SELF, 'wait-health', '--url', f'http://127.0.0.1:{cfg["target_port"]}', '--service-root', remote / 'target')
        ready = time.monotonic_ns()
        b.run(E, 'restore', '--output', remote / 'cold-new', '--url', f'http://127.0.0.1:{cfg["target_port"]}')
        decoded = time.monotonic_ns()
        b.run(E, 'check', '--output', remote / 'cold-new')  # strict live Stage E checker
        scp(b, remote / 'cold-new/check.json', local / 'stage-e-check.json')
        verdict = read(local / 'stage-e-check.json')
        save(local / 'acceptance.json', dict(run_id=rid, target_run_id=verdict['run_id'], live_check_exit=0))
        from .perf_observer import emit
        emit('failure_to_decode_complete_harness', 'recovery', start_ns=recovery_start, end_ns=decoded,
             includes='native_recovery,delivery,independent_checks,SSH,target_startup,16_decodes,validation',
             failure_detection='excluded; controlled-loss boundary', production_recovery=False)
        emit('new_instance_launch_to_ready_harness', 'startup', start_ns=target_started, end_ns=ready,
             includes='SSH,admission,full_source_weight_hashing,model_load,engine_initialization,health_poll')
        native_metrics(local, read(local / 'export0/export.json'), read(local / 'export1.json'),
            read(local / 'recovered-source/protected/recovery.json'), read(local / 'delivery-send.json'))
        # Read-only Stage E results remain on 233. Only JSON/JSONL comes back.
        b.run(SELF, 'bundle-json', '--root', remote, '--output', remote / 'json-evidence.zip')
        scp(b, remote / 'json-evidence.zip', local / 'json-evidence-233.zip')
        destination = local / 'perf-233'
        destination.mkdir()
        with zipfile.ZipFile(local / 'json-evidence-233.zip') as archive:
            for name in archive.namelist():
                if name.startswith('perf/') and name.endswith('.jsonl'):
                    with (destination / Path(name).name).open('xb') as out:
                        out.write(archive.read(name))
    finally:
        for h, name, proc in reversed(services):
            if proc.poll() is None:
                stop(h, name)
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('trial')
    for name in ('prepare', 'stop', 'service', 'bundle-json'):
        q = sub.add_parser(name)
        q.add_argument('--root', type=Path, required=True)
        if name == 'service':
            q.add_argument('--resource-interval', type=float, default=1.)
            q.add_argument('argv', nargs=argparse.REMAINDER)
        if name == 'bundle-json':
            q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('wait-file')
    q.add_argument('--path', type=Path, required=True)
    q = sub.add_parser('wait-health')
    q.add_argument('--url', required=True)
    q.add_argument('--service-root', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'trial':
        trial()
    elif a.command == 'prepare':
        a.root.mkdir(parents=True, exist_ok=False)
    elif a.command == 'stop':
        if not (a.root / 'owned-process.json').exists():
            raise ValueError('only a service owned by this experiment can stop')
        with (a.root / 'stop').open('x'):
            pass
    elif a.command == 'service':
        service(a.root, a.argv[1:] if a.argv[:1] == ['--'] else a.argv, a.resource_interval)
    elif a.command == 'bundle-json':
        with zipfile.ZipFile(a.output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(a.root.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.jsonl'):
                    z.write(path, str(path.relative_to(a.root)))
    elif a.command == 'wait-file':
        end = time.monotonic() + 900
        while not a.path.exists():
            if time.monotonic() > end:
                raise TimeoutError('launcher did not publish ' + str(a.path))
            time.sleep(.5)
    else:
        from .perf_benchmark import http
        end = time.monotonic() + 900
        while True:
            if (a.service_root / 'exit.json').exists():
                raise RuntimeError('owned server exited before ready')
            try:
                http(a.url.rstrip('/') + '/server_info', timeout=5)
                break
            except (OSError, ValueError):
                if time.monotonic() > end:
                    raise TimeoutError('server not ready')
                time.sleep(.5)


if __name__ == '__main__':
    main()
