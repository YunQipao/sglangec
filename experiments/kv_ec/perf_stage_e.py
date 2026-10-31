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
import threading
import traceback
import re
import math
import urllib.request

from .perf_benchmark import Sampler, digest, identity, read, save
from .pidfd_compat import check_signal_support, pidfd_open, pidfd_send_signal

E = 'experiments.kv_ec.verify_rs_cold_resume'
D = 'experiments.kv_ec.verify_gpu_cold_resume'
B = 'experiments.kv_ec.perf_benchmark'
SELF = 'experiments.kv_ec.perf_stage_e'


def target_binding(verdict, run, protection, trial_id):
    """Bind the unmodified checker result to fresh plan, target and TP reports.

    Stage E has no top-level run_id. The launch run.json and both TP reports
    are the authoritative target identity; the online plan supplies trial ID.
    """
    from .rs_cold_source import validate_protection
    from .online_rs_identity import stripe_id
    identity = validate_protection(protection)
    if identity['run_id'] != trial_id:
        raise ValueError('trial_plan_identity_mismatch')
    receipts = protection['receipts']
    for key in ('stripe_id', 'handle', 'manifest_sha256'):
        if not verdict.get(key) or any(r[key] != verdict[key] for r in receipts):
            raise ValueError('trial_verdict_protection_mismatch: ' + key)
    if verdict['stripe_id'] != stripe_id(identity):
        raise ValueError('trial_stripe_identity_mismatch')
    ranks = verdict.get('ranks', [])
    tp_size = run.get('tp_size', 2)
    if (tp_size not in (2, 4) or verdict.get('tp_size', 2) != tp_size or
            len(ranks) != tp_size or {r['tp_rank'] for r in ranks} != set(range(tp_size)) or
            any(r.get('tp_size', 2) != tp_size for r in ranks) or
            any(r.get('run_id') != run['run_id'] for r in ranks) or
            run.get('restore_id') != verdict.get('restore_id') or
            run.get('rs_source_binding') != verdict.get('rs_source')):
        raise ValueError('trial_target_identity_mismatch')
    return dict(target_run_id=run['run_id'], restore_id=verdict['restore_id'],
                stripe_id=verdict['stripe_id'], handle=verdict['handle'],
                manifest_sha256=verdict['manifest_sha256'])


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=fields[0], start_ticks=int(fields[19]), pgid=int(fields[2]),
                session=int(fields[3]), sid=int(fields[3]), uid=proc.stat().st_uid,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def owned_members(owned):
    """Only the recorded new session, including children with a new PGID.

    A session ID cannot be recycled while its original members remain alive.
    This also handles launcher exit with children still in the original group.
    """
    if owned['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('owned_process_boot_mismatch')
    members = []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            current = process_identity(int(entry.name))
        except (OSError, ValueError):
            continue
        if current['pid'] == owned['pid'] and current['start_ticks'] != owned['start_ticks']:
            raise ValueError('owned_process_PID_reused; refusing signal')
        if current['session'] == owned['pid'] and current['state'] not in ('Z', 'X'):
            if (current['uid'] != owned['uid'] or owned['uid'] != os.getuid() or
                    owned['pgid'] != owned['pid'] or owned['session'] != owned['pid']):
                raise ValueError('owned_process_session_identity_mismatch')
            if owned.get('run_id'):
                try:
                    environment = (entry / 'environ').read_bytes().split(b'\0')
                except FileNotFoundError:
                    continue  # exited during enumeration; never signal by PID
                token = ('KV_EC_PERF_RUN_ID=' + owned['run_id']).encode()
                if token not in environment:
                    raise ValueError('owned_process_trial_environment_mismatch; refusing signal')
            members.append(current)
    return members


def verify_signal_identity(member, owned, exit_retries=50):
    current = process_identity(member['pid'])
    if any(current[k] != member[k] for k in
           ('pid', 'start_ticks', 'uid', 'pgid', 'session', 'sid', 'boot_id')):
        raise ValueError('owned_process_changed_before_or_after_signal')
    if not owned.get('run_id') or owned['run_id'] != os.environ.get('KV_EC_PERF_RUN_ID'):
        raise ValueError('cleanup_trial_ownership_mismatch')
    if current['state'] in ('Z', 'X'):
        return
    token = ('KV_EC_PERF_RUN_ID=' + owned['run_id']).encode()
    try:
        environment = (Path('/proc') / str(member['pid']) / 'environ').read_bytes().split(b'\0')
    except PermissionError:
        # Linux may deny environ during exit after stat was read. Only a
        # revalidated zombie/dead instance is exempt; live denial fails closed.
        if not exit_retries:
            raise
        time.sleep(.01)
        return verify_signal_identity(member, owned, exit_retries - 1)
    if token not in environment and current['state'] not in ('Z', 'X'):
        raise ValueError('owned_process_trial_environment_mismatch; refusing signal')


def signal_owned(owned, sig):
    members = owned_members(owned)
    for member in members:
        # pidfd pins a process instance across signal delivery; never signal a
        # reused numeric PID or a whole group with unverified members.
        try:
            fd = pidfd_open(member['pid'])
        except ProcessLookupError:
            continue
        try:
            verify_signal_identity(member, owned)
            pidfd_send_signal(fd, sig)
            verify_signal_identity(member, owned)
        except (ProcessLookupError, FileNotFoundError):
            pass
        finally:
            os.close(fd)
    return bool(members)


def cleanup_process(proc, owned, root):
    """Bounded verified cleanup shared by model services and SSH tunnels."""
    record = dict(owner=owned, status='FAIL', attempts=[])
    try:
        if not owned.get('run_id') or owned['run_id'] != os.environ.get('KV_EC_PERF_RUN_ID'):
            raise ValueError('cleanup_trial_ownership_mismatch')
        for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            proc.poll()  # reap a launcher that exited before its children
            if not signal_owned(owned, 0):
                break
            record['attempts'].append(dict(signal=int(sig), members=owned_members(owned)))
            signal_owned(owned, sig)
            end = time.monotonic() + grace
            while time.monotonic() < end:
                proc.poll()
                if not signal_owned(owned, 0):
                    break
                time.sleep(.2)
        proc.wait(timeout=5)
        remaining = owned_members(owned)
        if remaining:
            raise RuntimeError('OWNED_PROCESS_LEAK: ' + json.dumps(remaining))
        record.update(status='PASS', remaining=[], returncode=proc.returncode)
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save(Path(root) / 'process-cleanup.json', record)


def require_trial(root):
    marker = read(Path(root) / 'trial-owner.json')
    if marker['run_id'] != os.environ['KV_EC_PERF_RUN_ID']:
        raise ValueError('trial_ownership_mismatch')
    return marker


def require_service_alive(root):
    """Read-only ownership/liveness check; a foreign HTTP listener is not READY."""
    marker = require_trial(root.parent)
    if (root / 'exit.json').exists():
        raise RuntimeError(f'owned server exited before ready: {read(root / "exit.json")}; '
                           f'see {root / "server.log"}')
    if not (root / 'owned-process.json').exists():
        return False  # supervisor has not published the child identity yet
    owned = read(root / 'owned-process.json')
    if owned.get('run_id') != marker['run_id']:
        raise RuntimeError('owned service trial identity mismatch: ' + str(root))
    try:
        current = process_identity(owned['pid'])
    except OSError as exc:
        raise RuntimeError('owned server exited or identity unavailable before ready: ' + str(root)) from exc
    if any(current[k] != owned[k] for k in ('pid', 'start_ticks', 'pgid', 'session', 'uid', 'boot_id')):
        raise RuntimeError('owned service process identity mismatch before ready: ' + str(root))
    if current['state'] in ('Z', 'X'):
        raise RuntimeError('owned server exited before ready: ' + str(root))
    return True


class _NoHealthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never follow a redirect into /server_info or another control API.
        return None


def health_status(url, timeout):
    # /health returns an empty Response, not JSON. Existing Stage A launcher
    # sets SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 for both E launch modes.
    opener = urllib.request.build_opener(_NoHealthRedirect())
    with opener.open(url, timeout=timeout) as response:
        return response.status


def gpu_failure_reason(log):
    """Diagnostic only, never converts a failing subprocess into acceptance."""
    if not Path(log).exists():
        return None
    with Path(log).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        tail = stream.read().decode(errors='replace').lower()
    if 'out of memory' in tail or 'cuda error: memory allocation' in tail:
        return 'GPU_RUNTIME_OOM: resource competition or allocation exceeded budget; see server.log; new trial required'
    if 'loaded weights leave no gpu memory' in tail:
        return 'GPU_STATIC_KV_POOL_INSUFFICIENT: actual post-weight free memory cannot fit KV pool; see server.log'
    if 'insufficient_gpu_memory' in tail or 'gpu_static_kv_pool_insufficient' in tail:
        return 'GPU_ADMISSION_RECHECK_FAILED: GPU state changed or budget insufficient; see server.log'
    return None


def wait_health(url, service_root, timeout=900):
    from .perf_ports import assert_listener_owned
    from urllib.parse import urlsplit
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite health timeout required')
    root = Path(service_root)
    endpoint = url.rstrip('/') + '/health'
    end = time.monotonic() + timeout
    last_error = 'owned process identity not yet published'
    while True:
        alive = require_service_alive(root)
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'server not ready: {endpoint}; {last_error}; see {root / "server.log"}')
        if alive:
            status = None
            try:
                assert_listener_owned(urlsplit(url).port, read(root / 'owned-process.json'))
                status = health_status(endpoint, timeout=min(5., remaining))
                last_error = f'HTTP {status}, expected 200'
            except (OSError, ValueError) as exc:
                last_error = f'{type(exc).__name__}: {exc}'
            # A service can die or its PID can change while HTTP is in flight.
            if require_service_alive(root) and status == 200 and time.monotonic() < end:
                assert_listener_owned(urlsplit(url).port, read(root / 'owned-process.json'))
                return
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'server not ready: {endpoint}; {last_error}; see {root / "server.log"}')
        time.sleep(min(.5, remaining))


def service(root, argv, interval):
    marker = require_trial(root.parent)
    if root.exists() and sorted(p.name for p in root.iterdir()) == ['stop']:
        save(root / 'exit.json', dict(returncode=0, reason='stopped_before_launch'))
        return
    check_signal_support()
    root.mkdir(parents=True, exist_ok=False)
    os.environ['KV_EC_PERF_ROLE'] = os.environ.get('KV_EC_PERF_ROLE', 'unspecified') + '/' + root.name
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stopping.set())
    with (root / 'server.log').open('x') as log, Sampler(root / 'resources.jsonl', interval):
        started = time.monotonic_ns()
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned = None
        reason = 'process_exited'
        try:
            owned = dict(**process_identity(proc.pid), argv=argv, start_ns=started, run_id=marker['run_id'])
            save(root / 'owned-process.json', owned)
            while proc.poll() is None:
                if stopping.is_set() or (root / 'stop').exists():
                    reason = 'owned_stop_requested'
                    break
                if time.time() - (root.parent / 'lease').stat().st_mtime > (150 if root.name == 'ports' else 90):
                    reason = 'coordinator_lease_expired_SSH_or_driver_lost'
                    break
                time.sleep(.2)
        finally:
            if owned is None:
                # Popen's unreaped child still has a /proc identity, even if it
                # already exited. Never fall back to an unauthenticated kill.
                owned = dict(**process_identity(proc.pid), run_id=marker['run_id'])
            cleanup_process(proc, owned, root)
            from .perf_ports import assert_free
            flags = ('--port', '--rdma-return-port') + (('--remote-port',) if root.name == 'receiver' else ())
            bound_ports = [int(argv[argv.index(flag) + 1]) for flag in flags if flag in argv]
            assert_free(bound_ports, 'SERVICE_PORT_RELEASE_FAILED')
        save(root / 'exit.json', dict(returncode=proc.returncode, reason=reason,
                                     gpu_failure_reason=gpu_failure_reason(root / 'server.log'),
                                     end_ns=time.monotonic_ns()))
    if proc.returncode > 0 or (reason == 'process_exited' and proc.returncode):
        raise SystemExit(proc.returncode if proc.returncode > 0 else 1)
    if reason == 'coordinator_lease_expired_SSH_or_driver_lost':
        raise SystemExit(1)


class Host:
    def __init__(self, cfg, root, run_id, node, gpu_admission_mode='strict'):
        from .perf_preflight import cuda_environment
        from .gpu_admission import environment
        self.cfg, self.root, self.run_id, self.node = cfg, Path(root), run_id, node
        cuda, self.path_prefixes = cuda_environment(cfg)
        self.env = dict(PYTHONPATH=cfg['repo'] + '/python:' + cfg['repo'],
            KV_EC_PERF_DIR=str(self.root / 'perf'), KV_EC_PERF_RUN_ID=run_id,
            KV_EC_PERF_ROLE=f'node{node}-{232+node}', **cuda, **environment(gpu_admission_mode))

    def argv(self, module, *args):
        cmd = [self.cfg['python'], '-m', module, *map(str, args)]
        if self.cfg.get('ssh'):
            # Expand each suffix on its own host; never copy node0's paths to
            # node1. Empty suffixes must not add the current directory to PATH.
            paths = [key + '=' + shlex.quote(value) + '"${' + key + ':+:$' + key + '}"'
                     for key, value in self.path_prefixes.items()]
            shell = 'cd ' + shlex.quote(self.cfg['repo']) + ' && exec env ' + ' '.join(paths) + ' ' + shlex.join(
                [f'{k}={v}' for k, v in self.env.items()] + cmd)
            return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', self.cfg['ssh'], shell]
        paths = [f'{key}={value}' + (os.pathsep + os.environ[key] if os.environ.get(key) else '')
                 for key, value in self.path_prefixes.items()]
        return ['env', *paths, *[f'{k}={v}' for k, v in self.env.items()], *cmd]

    def run(self, module, *args, log=None, timeout=1200):
        cmd = self.argv(module, *args)
        started = time.monotonic_ns()
        record = dict(node=self.node, module=module, args=list(map(str, args)), status='FAIL')
        try:
            result = subprocess.run(cmd, cwd=self.cfg['repo'] if not self.cfg.get('ssh') else None,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            record['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'node{self.node} exit={result.returncode}: {shlex.join(cmd)}; see driver/server logs')
            record['status'] = 'PASS'
            return result
        except BaseException as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            record['harness_ms'] = (time.monotonic_ns() - started) / 1e6
            from .perf_observer import emit
            if args and args[0] != 'heartbeat':
                emit('harness_command_' + module.rsplit('.', 1)[-1] + '_' + str(args[0]),
                     'harness', duration_ms=record['harness_ms'],
                     status='ok' if record['status'] == 'PASS' else 'error', node=self.node,
                     includes='SSH_if_remote,command_work,validation; not isolated network overhead')
            path = os.environ.get('KV_EC_BENCH_TRIAL')
            if path:
                with (Path(path) / 'commands.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')


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


def check_rs_health(cfg, expected_epoch):
    from .remote_rs_client import RemoteRSClient
    health, payload = RemoteRSClient(cfg['rs_server']['host'], cfg['rs_server']['port'], timeout=10).call('health')
    if (payload or not expected_epoch or health.get('state') != 'READY' or
            health.get('server_epoch') != expected_epoch):
        raise RuntimeError('RS_NOT_READY_OR_EPOCH_CHANGED_AFTER_TRIAL')
    return health


def scp(host, remote, local, upload=False):
    endpoint = host.cfg['ssh'] + ':' + str(remote)
    args = [str(local), endpoint] if upload else [endpoint, str(local)]
    from .perf_observer import span
    with span('harness_scp_metadata', 'harness', upload=upload):
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                        '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', *args], check=True, timeout=120)


def local_ssh_tunnel(host, local_port, remote_port, root):
    from .perf_ports import assert_free, assert_listener_owned
    check_signal_support()
    root = Path(root)
    root.mkdir(exist_ok=False)
    assert_free([local_port])
    # Multiplexed -L forwarding can outlive its client in a persistent master.
    # This process must own the listener itself, in foreground, without user
    # configured LocalForward/RemoteForward entries or ControlPersist.
    argv = ['ssh', '-F', '/dev/null', '-N', '-S', 'none', '-o', 'ControlMaster=no',
        '-o', 'ControlPersist=no', '-o', 'ForkAfterAuthentication=no',
        '-o', 'ClearAllForwardings=no', '-o', 'BatchMode=yes',
        '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
        '-L', f'127.0.0.1:{local_port}:127.0.0.1:{remote_port}', host.cfg['ssh']]
    with (root / 'ssh.log').open('x') as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        owned = None
        try:
            owned = dict(**process_identity(proc.pid), argv=argv,
                         run_id=os.environ.get('KV_EC_PERF_RUN_ID'), local_port=local_port,
                         remote_port=remote_port, remote_host=host.cfg['ssh'])
            proc.kv_ec_owned, proc.kv_ec_root = owned, root
            save(root / 'owned-process.json', owned)
            deadline, delay = time.monotonic() + 30, .1
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('reprefill_ssh_tunnel_failed; see ' + str(root / 'ssh.log'))
                try:
                    assert_listener_owned(local_port, owned)
                    status = health_status(f'http://127.0.0.1:{local_port}/health', timeout=2)
                    if status == 200:
                        assert_listener_owned(local_port, owned)
                        return proc
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('reprefill_ssh_tunnel_timeout; see ' + str(root / 'ssh.log'))
                time.sleep(delay)
                delay = min(delay * 2, 1.)
        except BaseException:
            if owned is None:
                owned = dict(**process_identity(proc.pid), run_id=os.environ.get('KV_EC_PERF_RUN_ID'))
            cleanup_process(proc, owned, root)
            assert_free([local_port], 'TUNNEL_PORT_RELEASE_FAILED')
            raise


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
    if root is not None:
        save(root / 'native-metrics.json', values)
    return values


def trial():
    from .gpu_admission import config_mode, metadata, validate_evidence
    cfg = read(os.environ['KV_EC_BENCH_CONFIG'])
    from .perf_preflight import template_path, validate_config
    validate_config(cfg)
    selected_mode = config_mode(cfg)
    policy = metadata(selected_mode, cfg.get('mem_fraction_static', .6))
    policy.update(tp_size=cfg['tp_size'], prefix_tokens=cfg['prefix_tokens'])
    local = Path(os.environ['KV_EC_BENCH_TRIAL'])
    rid = os.environ['KV_EC_PERF_RUN_ID']
    recovery_mode = os.environ.get('KV_EC_PERF_RECOVERY_MODE', 'two-level')
    if recovery_mode not in ('two-level', 'reprefill'):
        raise ValueError('invalid_recovery_mode')
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,128}', rid) or local.name != rid:
        raise ValueError('invalid_trial_ID_or_directory_binding')
    remote = Path(cfg.get('remote_root', '/tmp/kv-ec-perf')) / rid
    hosts = [Host(cfg['nodes'][0], local, rid, 0, selected_mode),
             Host(cfg['nodes'][1], remote, rid, 1, selected_mode)]
    a, b = hosts
    # Check both configured interpreters before any remote trial/service exists.
    # The coordinator log is already inside the current benchmark trial.
    with (local / 'pidfd-preflight.log').open('x') as log:
        for h in hosts:
            h.run(SELF, 'check-pidfd', log=log, timeout=30)
    save(local / 'trial-owner.json', dict(run_id=rid, **policy))
    (local / 'lease').touch(exist_ok=False)
    b.run(SELF, 'prepare', '--root', remote)
    b.run(SELF, 'init-trial', '--root', remote)
    heartbeat_stop = threading.Event()
    heartbeat_errors = []
    def heartbeat():
        while not heartbeat_stop.wait(15):
            try:
                (local / 'lease').touch()
                b.run(SELF, 'heartbeat', '--root', remote, timeout=25)
            except Exception as exc:
                heartbeat_errors.append(str(exc))
                return  # supervisors expire their lease and release owned GPU groups
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    def interrupted(signum, _):
        raise InterruptedError(f'coordinator_signal_{signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    services, logs, tunnels = [], [], []
    def start(h, name, module, *args):
        if name in ('origin', 'target'):
            args = (*args, '--gpu-admission-mode', selected_mode)
        log = (local / f'{name}-node{h.node}.log').open('x')
        logs.append(log)
        proc = subprocess.Popen(h.argv(SELF, 'service', '--root', h.root / name,
            '--resource-interval', cfg.get('resource_interval', 1), '--', h.cfg['python'], '-m', module, *map(str, args)),
            cwd=h.cfg['repo'] if not h.cfg.get('ssh') else None, stdout=log, stderr=subprocess.STDOUT)
        services.append((h, name, proc))
        return proc
    def stop(h, name):
        # The service may not yet have published owned-process.json. Its
        # supervisor observes this authenticated stop marker even during launch.
        h.run(SELF, 'stop', '--root', h.root / name, timeout=30)
        owned = next(p for host, label, p in services if host is h and label == name)
        if owned.wait(timeout=90) != 0:
            raise RuntimeError('owned service supervisor failed: ' + name)
    rs = cfg['rs_server']
    def rdma(h):
        return ['--remote-host', rs['host'], '--remote-port', rs['port'],
                '--rdma-device', h.cfg.get('rdma_device', 'mlx5_0'),
                '--rdma-gid-index', h.cfg['gid_index'], '--rdma-port-num', 1, '--timeout', 600]
    def startup_evidence(path):
        record = read(path)
        validate_evidence(record, selected_mode, cfg['tp_size'])
        if record['mem_fraction_static'] != cfg.get('mem_fraction_static', .6):
            raise ValueError('GPU_ADMISSION_MODE_MISMATCH: startup memory fraction')
        return record
    try:
        from .perf_ports import trial_ports
        save(local / 'trial-ports.json', dict(run_id=rid, rs_server=cfg['rs_server'],
             nodes=[dict(node=h.node, ports=trial_ports(cfg, h.node),
                         roles=(dict(origin=h.cfg['origin_port'], tunnel=h.cfg['origin_port'],
                                     return_port=cfg['return_port']) if h.node == 0 else
                                dict(origin=h.cfg['origin_port'], target=cfg['target_port'],
                                     delivery=cfg['delivery_port'])),
                         host=h.cfg.get('ssh', 'local232')) for h in hosts]))
        # Both host-local guards hold their locks from here until reverse-order
        # cleanup has stopped every model/native service and the local tunnel.
        for h in hosts:
            start(h, 'ports', SELF, 'hold-ports', '--root', h.root,
                  '--ports', ','.join(map(str, trial_ports(cfg, h.node))))
            h.run(SELF, 'wait-file', '--path', h.root / 'ports-ready.json',
                  '--service-root', h.root / 'ports')
        parallel(*[lambda h=h: h.run(B, 'inventory', '--output', h.root / f'inventory-{232+h.node}.json') for h in hosts])
        scp(b, remote / 'inventory-233.json', local / 'inventory-233.json')
        for h in hosts:
            n = h.node
            start(h, 'origin', D, 'capture', '--model', h.cfg['model'],
                '--template-checkpoint', template_path(cfg, n), '--input-ids', h.cfg['input_ids'],
                '--bundle', h.root / 'capture', '--output', h.root / f'node{n}-reference',
                '--gpus', h.cfg['gpus'], '--port', h.cfg['origin_port'],
                '--mem-fraction-static', cfg.get('mem_fraction_static', .60), '--timeout', 120)
        def capture(h):
            # Admission/source hashing precedes run.json; wait without submitting
            # a request or claiming readiness before the launcher writes it.
            h.run(SELF, 'wait-file', '--path', h.root / f'node{h.node}-reference/run.json',
                  '--service-root', h.root / 'origin')
            h.run(SELF, 'wait-health', '--url', f'http://127.0.0.1:{h.cfg["origin_port"]}',
                  '--service-root', h.root / 'origin')
            h.run(D, 'submit', '--output', h.root / f'node{h.node}-reference',
                  '--url', f'http://127.0.0.1:{h.cfg["origin_port"]}')
            h.run(D, 'reference', '--output', h.root / f'node{h.node}-reference')
            h.run(E, 'claim', '--capture', h.root / 'capture', '--output', h.root / f'claim{h.node}.json')
        parallel(*[lambda h=h: capture(h) for h in hosts])
        scp(b, remote / 'node1-reference/run.json', local / 'origin-run-233.json')
        startup_evidence(local / 'node0-reference/run.json')
        startup_evidence(local / 'origin-run-233.json')
        if recovery_mode == 'reprefill':
            (local / 'node1-reference').mkdir(exist_ok=True)
            scp(b, remote / 'node1-reference/rank_0.json',
                local / 'node1-reference/rank_0.json')
            scp(b, remote / 'node1-reference/submitted_input_ids.json',
                local / 'node1-reference/submitted_input_ids.json')
        for h in hosts:
            stop(h, 'origin')
        scp(b, remote / 'claim1.json', local / 'claim1.json')
        for claim in (read(local / 'claim0.json'), read(local / 'claim1.json')):
            if claim['descriptor']['length'] + 1024**2 > cfg['max_node_bytes']:
                raise ValueError('CAPTURE_EXCEEDS_PREFLIGHT_NODE_RESERVATION; no stripe registered')
        a.run(E, 'plan', '--node0-claim', local / 'claim0.json', '--node1-claim', local / 'claim1.json',
              '--run-id', rid, '--output', local / 'plan.json')
        scp(b, remote / 'plan.json', local / 'plan.json', upload=True)
        protection_started = time.monotonic()
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
        if recovery_mode == 'reprefill':
            target_started = time.monotonic_ns()
            start(b, 'target', 'experiments.kv_ec.reprefill_server', 'launch',
                '--model', b.cfg['model'], '--output', remote / 'cold-reprefill',
                '--gpus', b.cfg['gpus'], '--tp', cfg['tp_size'], '--port', cfg['target_port'],
                '--mem-fraction-static', cfg.get('mem_fraction_static', .60))
            b.run(SELF, 'wait-file', '--path', remote / 'cold-reprefill/run.json',
                  '--service-root', remote / 'target')
            b.run(SELF, 'wait-health', '--url', f'http://127.0.0.1:{cfg["target_port"]}',
                  '--service-root', remote / 'target')
            ready = time.monotonic_ns()
            scp(b, remote / 'cold-reprefill/run.json', local / 'reprefill-target-run.json')
            target_admission = startup_evidence(local / 'reprefill-target-run.json')
            from .perf_preflight import free_ports
            free_ports([a.cfg['origin_port']])
            tunnel = local_ssh_tunnel(b, a.cfg['origin_port'], cfg['target_port'], local / 'tunnel')
            tunnels.append(tunnel)
            a.run(B, 'inference', '--scheme', 'reprefill', '--url',
                f'http://127.0.0.1:{a.cfg["origin_port"]}', '--model', a.cfg['model'],
                '--server-model', b.cfg['model'],
                '--input-ids', local / 'node1-reference/submitted_input_ids.json',
                '--output', local / 'reprefill-inference', '--tp', cfg['tp_size'], '--warmup', 0, '--repeat', 1,
                '--expected-model-sha256',
                read(local / 'node1-reference/rank_0.json')['runtime_identity']['model_sha256'],
                '--expected-input-ids-sha256', read(local / 'node1-reference/rank_0.json')['token_ids_sha256'],
                '--model-identity-evidence', local / 'reprefill-target-run.json',
                '--ready-monotonic-ns', ready)
            request = read(local / 'reprefill-inference/request-000.json')
            reference = read(local / 'node1-reference/rank_0.json')
            if request['output_token_ids'] != reference['output_token_ids']:
                raise ValueError('reprefill_output_token_ids_mismatch')
            if all(key in request for key in ('start_ns', 'first_output_token_ns',
                                               'full_output_ns', 'response_complete_ns')):
                validation_complete = time.monotonic_ns()
                from .cold_timing import build_timing, clock_identity
                client_identity = clock_identity()
                client = dict(operation_id=rid, **client_identity,
                    request_start_ns=request['start_ns'],
                    first_output_token_ns=request['first_output_token_ns'],
                    full_output_ns=request['full_output_ns'],
                    response_complete_ns=request['response_complete_ns'])
                save(local / 'cold-comparison-timing.json', build_timing(
                    rid, 'reprefill', recovery_start, target_started, ready, client,
                    validation_complete))
            save(local / 'reprefill.json', dict(schema='kv-ec-cold-reprefill-v1',
                **dict(policy, gpu_resources_before_start=target_admission['gpu_resources_before_start']),
                functional_correctness_pass=True,
                pass_reprefill=True, output_token_ids_match=True,
                commit_sha=identity()['commit_sha'], model_sha256=reference['runtime_identity']['model_sha256'],
                input_sha256=digest(local / 'node1-reference/submitted_input_ids.json'), target_run_id=rid,
                target_started_ns=target_started, ready_ns=ready,
                fault_boundary_to_ready_ms=(ready-recovery_start)/1e6,
                ready_to_first_token_ms=request['ready_to_first_token_ms'],
                ready_to_decode_complete_ms=request['ready_to_decode_complete_ms'],
                fault_boundary_to_decode_complete_ms=(request['end_ns']-recovery_start)/1e6,
                client_request_ms=request['request_ms'], client_prefill_ttft_ms=request['prefill_ttft_ms'],
                timing_note='all event timestamps use node0 coordinator monotonic; SSH tunnel targets the owned node1 service'))
            return
        restore_id = read(local / 'capture/request_state.json')['restore_id']
        # The synchronous return operation creates a native receiver in a
        # daemon thread. Supervise its entire session even if its launcher exits
        # before that helper; never leave it in the coordinator's own session.
        recovery = start(a, 'return', E, 'recover', '--protection', local / 'protection.json',
            '--survivor', local / f'export0/{restore_id}/global_node.bin', *rdma(a),
            '--rdma-return-host', cfg['return_host'], '--rdma-return-port', cfg['return_port'],
            '--output', local / 'recovered-source')
        if recovery.wait(timeout=1200) != 0:
            raise RuntimeError('owned return recovery failed; see return/server.log')
        # Native receiver, not a model service. It exits after one transfer.
        receive = start(b, 'receiver', E, 'receive-source', '--remote-host', cfg['return_host'],
            '--remote-port', cfg['delivery_port'], '--rdma-device', b.cfg.get('rdma_device', 'mlx5_0'),
            '--rdma-gid-index', b.cfg['gid_index'], '--timeout', 180, '--output', remote / 'rs-source')
        a.run(E, 'send-source', '--bundle', local / 'recovered-source', '--remote-host', rs['host'],
            '--remote-port', cfg['delivery_port'], '--rdma-device', a.cfg.get('rdma_device', 'mlx5_0'),
            '--rdma-gid-index', a.cfg['gid_index'], '--timeout', 180, '--output', local / 'delivery-send.json')
        if receive.wait(timeout=240) != 0:
            raise RuntimeError('native delivery receiver failed')
        scp(b, remote / 'delivery-send.json', local / 'delivery-send.json', upload=True)
        b.run(E, 'check-source', '--bundle', remote / 'rs-source', '--capture', remote / 'capture',
              '--send-receipt', remote / 'delivery-send.json')
        # claim_restore requires an existing, empty ledger. Never reuse a
        # failed checkpoint or remove its claim/commit markers.
        b.run(SELF, 'prepare', '--root', remote / 'restore-ledger')
        target_started = time.monotonic_ns()
        start(b, 'target', E, 'launch', '--model', b.cfg['model'], '--bundle', remote / 'capture',
            '--rs-source', remote / 'rs-source', '--reference', remote / 'node1-reference',
            '--ledger', remote / 'restore-ledger', '--output', remote / 'cold-new', '--gpus', b.cfg['gpus'],
            '--port', cfg['target_port'], '--mem-fraction-static', cfg.get('mem_fraction_static', .60), '--timeout', 120)
        b.run(SELF, 'wait-file', '--path', remote / 'cold-new/run.json', '--service-root', remote / 'target')
        b.run(SELF, 'wait-health', '--url', f'http://127.0.0.1:{cfg["target_port"]}', '--service-root', remote / 'target')
        ready = time.monotonic_ns()
        b.run(E, 'restore', '--output', remote / 'cold-new', '--url', f'http://127.0.0.1:{cfg["target_port"]}')
        decoded = time.monotonic_ns()
        b.run(E, 'check', '--output', remote / 'cold-new')  # strict live Stage E checker
        if heartbeat_errors:
            raise RuntimeError('SSH_HEARTBEAT_FAILED: ' + str(heartbeat_errors))
        if time.monotonic() - protection_started > cfg['protection_timeout']:
            raise TimeoutError('PROTECTION_DEADLINE_EXCEEDED: new trial required; old stripe cannot be reused')
        scp(b, remote / 'cold-new/check.json', local / 'stage-e-check.json')
        verdict = read(local / 'stage-e-check.json')
        scp(b, remote / 'cold-new/run.json', local / 'target-run.json')
        startup_evidence(local / 'target-run.json')
        binding = target_binding(verdict, read(local / 'target-run.json'),
                                 read(local / 'protection.json'), rid)
        save(local / 'acceptance.json', dict(run_id=rid, **binding, **policy,
            functional_correctness_pass=True, live_check_exit=0,
            verdict_sha256=digest(local / 'stage-e-check.json'),
            target_run_sha256=digest(local / 'target-run.json'),
            protection_sha256=digest(local / 'protection.json')))
        save(local / 'stage-e-timing.json', dict(
            schema='kv-ec-stage-e-coordinator-timing-v1', target_run_id=read(local / 'target-run.json')['run_id'],
            ready_to_decode_complete_ms=(decoded-ready)/1e6,
            fault_boundary_to_decode_complete_ms=(decoded-recovery_start)/1e6,
            fault_boundary_to_ready_ms=(ready-recovery_start)/1e6,
            timing_note='232 coordinator monotonic; includes remote command and Stage E validation'))
        save(local / 'cold-comparison-timing-status.json', dict(
            schema='kv-ec-cold-comparison-timing-status-v1', operation_id=rid,
            scheme='two-level', functional_correctness_pass=True,
            performance_measurement_eligible=False,
            unavailable_metrics=['fault_boundary_to_first_output_token_ms',
                'fault_boundary_to_response_complete_ms', 'response_stream_ms'],
            reason='two-level completion is observed by the restore hook and Stage E checker; '
                'no equivalent client SSE response is issued by this path'))
        if os.environ.get('KV_EC_PAIRED_READY_REPREFILL') == '1':
            b.run(SELF, 'wait-health', '--url', f'http://127.0.0.1:{cfg["target_port"]}',
                  '--service-root', remote / 'target')
            (local / 'node1-reference').mkdir(exist_ok=True)
            scp(b, remote / 'node1-reference/rank_0.json', local / 'node1-reference/rank_0.json')
            scp(b, remote / 'node1-reference/submitted_input_ids.json',
                local / 'node1-reference/submitted_input_ids.json')
            from .perf_preflight import free_ports
            free_ports([a.cfg['origin_port']])
            tunnel = local_ssh_tunnel(b, a.cfg['origin_port'], cfg['target_port'], local / 'tunnel')
            tunnels.append(tunnel)
            target_run = read(local / 'target-run.json')
            a.run(B, 'inference', '--scheme', 'reprefill', '--url',
                f'http://127.0.0.1:{a.cfg["origin_port"]}', '--model', a.cfg['model'],
                '--server-model', b.cfg['model'], '--input-ids',
                local / 'node1-reference/submitted_input_ids.json', '--output',
                local / 'paired-ready-reprefill', '--tp', cfg['tp_size'], '--warmup', 0, '--repeat', 1,
                '--expected-model-sha256', target_run['runtime_identity']['model_sha256'],
                '--expected-input-ids-sha256', read(local / 'node1-reference/rank_0.json')['token_ids_sha256'],
                '--model-identity-evidence', local / 'target-run.json',
                '--ready-monotonic-ns', ready)
            paired = read(local / 'paired-ready-reprefill/request-000.json')
            reference = read(local / 'node1-reference/rank_0.json')
            if paired['output_token_ids'] != reference['output_token_ids']:
                raise ValueError('paired_reprefill_output_token_ids_mismatch')
            save(local / 'paired-ready-reprefill.json', dict(
                **policy, functional_correctness_pass=True,
                schema='kv-ec-stage-e-same-target-reprefill-v1', pass_reprefill=True,
                output_token_ids_match=True, stage_e_verdict_sha256=digest(local / 'stage-e-check.json'),
                target_run_sha256=digest(local / 'target-run.json'), target_run_id=target_run['run_id'],
                model_sha256=target_run['runtime_identity']['model_sha256'],
                input_sha256=digest(local / 'node1-reference/submitted_input_ids.json'),
                request_ms=paired['request_ms'], prefill_ttft_ms=paired['prefill_ttft_ms'],
                ready_to_first_token_ms=paired['ready_to_first_token_ms'],
                ready_to_decode_complete_ms=paired['ready_to_decode_complete_ms'],
                cached_tokens=paired['response']['meta_info']['cached_tokens'],
                prompt_tokens=paired['response']['meta_info']['prompt_tokens'],
                completion_tokens=paired['response']['meta_info']['completion_tokens']))
        from .perf_observer import emit
        emit('failure_to_decode_complete_harness', 'recovery', start_ns=recovery_start, end_ns=decoded,
             includes='native_recovery,delivery,independent_checks,SSH,target_startup,16_decodes,validation',
             failure_detection='excluded; controlled-loss boundary', production_recovery=False)
        emit('new_instance_launch_to_ready_harness', 'startup', start_ns=target_started, end_ns=ready,
             includes='SSH,admission,full_source_weight_hashing,model_load,engine_initialization,health_poll')
        native_metrics(local, read(local / 'export0/export.json'), read(local / 'export1.json'),
            read(local / 'recovered-source/protected/recovery.json'), read(local / 'delivery-send.json'))
        scp(b, remote / 'rs-source/delivery.json', local / 'delivery-receive.json')
        from .rs_recovery_source import native_stats
        delivery_receive = read(local / 'delivery-receive.json')
        native_stats(delivery_receive['receive_stats'], delivery_receive['archive_bytes'])
        save(local / 'cq-evidence.json', dict(
            uploads=[read(local / 'export0/export.json')['receipt']['metrics']['rdma_native'],
                     read(local / 'export1.json')['receipt']['metrics']['rdma_native']],
            recovery=read(local / 'recovered-source/protected/recovery.json')['transfer'],
            delivery_send=read(local / 'delivery-send.json'), delivery_receive=delivery_receive))
    except BaseException as exc:
        save(local / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
            **policy, functional_correctness_pass=False,
            remote_root=str(remote), run_id=rid, evidence_preserved=True, retry_checkpoint_allowed=False))
        raise
    finally:
        errors = []
        for tunnel in tunnels:
            try:
                cleanup_process(tunnel, tunnel.kv_ec_owned, tunnel.kv_ec_root)
                from .perf_ports import assert_free
                assert_free([a.cfg['origin_port']], 'TUNNEL_PORT_RELEASE_FAILED')
            except Exception as exc:
                errors.append('tunnel cleanup: ' + str(exc))
        for h, name, proc in reversed(services):
            if proc.poll() is None:
                try:
                    stop(h, name)
                except Exception as exc:
                    errors.append(f'cleanup node{h.node}/{name}: {exc}; '
                                  f'lease expires within {150 if name == "ports" else 90}s')
            elif proc.returncode:
                errors.append(f'owned supervisor failed node{h.node}/{name}: exit={proc.returncode}')
        for h in hosts:
            try:
                h.run(SELF, 'check-ports', '--root', h.root,
                      '--ports', ','.join(map(str, trial_ports(cfg, h.node))), timeout=30)
            except Exception as exc:
                errors.append(f'port release node{h.node}: {exc}')
        try:
            health = check_rs_health(cfg, os.environ.get('KV_EC_RS_EXPECTED_EPOCH'))
            save(local / 'rs-after-cleanup.json', health)
        except Exception as exc:
            errors.append('RS health after cleanup: ' + str(exc))
        # Preserve remote logs on success AND failure. No KV payload is SCP'd.
        try:
            b.run(SELF, 'bundle-json', '--root', remote, '--output', remote / 'json-evidence.zip', timeout=120)
            scp(b, remote / 'json-evidence.zip', local / 'json-evidence-233.zip')
            destination = local / 'perf-233'
            destination.mkdir()
            with zipfile.ZipFile(local / 'json-evidence-233.zip') as archive:
                for name in archive.namelist():
                    if name.startswith('perf/') and name.endswith('.jsonl'):
                        with (destination / Path(name).name).open('xb') as out:
                            out.write(archive.read(name))
        except Exception as exc:
            errors.append('evidence collection: ' + str(exc))
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=30)
        for log in logs:
            log.close()
        save(local / 'cleanup.json', dict(errors=errors, status='FAIL' if errors else 'PASS',
                                        remote_root=str(remote), lease_timeout_s=90,
                                        port_guard_lease_timeout_s=150))
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError('; '.join(errors))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    trial_parser = sub.add_parser('trial')
    trial_parser.add_argument('--recovery-mode', choices=('two-level', 'reprefill'), default='two-level')
    sub.add_parser('check-pidfd')
    for name in ('prepare', 'init-trial', 'heartbeat', 'stop', 'service', 'bundle-json', 'hold-ports', 'check-ports'):
        q = sub.add_parser(name)
        q.add_argument('--root', type=Path, required=True)
        if name == 'service':
            q.add_argument('--resource-interval', type=float, default=1.)
            q.add_argument('argv', nargs=argparse.REMAINDER)
        if name == 'bundle-json':
            q.add_argument('--output', type=Path, required=True)
        if name in ('hold-ports', 'check-ports'):
            q.add_argument('--ports', required=True)
    q = sub.add_parser('wait-file')
    q.add_argument('--path', type=Path, required=True)
    q.add_argument('--service-root', type=Path, required=True)
    q = sub.add_parser('wait-health')
    q.add_argument('--url', required=True)
    q.add_argument('--service-root', type=Path, required=True)
    q.add_argument('--timeout', type=float, default=900)
    a = p.parse_args()
    if a.command == 'check-pidfd':
        print(json.dumps(check_signal_support()), flush=True)
    elif a.command == 'trial':
        os.environ['KV_EC_PERF_RECOVERY_MODE'] = a.recovery_mode
        trial()
    elif a.command == 'prepare':
        a.root.mkdir(parents=True, exist_ok=False)
    elif a.command == 'init-trial':
        save(a.root / 'trial-owner.json', dict(run_id=os.environ['KV_EC_PERF_RUN_ID']))
        (a.root / 'lease').touch(exist_ok=False)
    elif a.command == 'heartbeat':
        require_trial(a.root)
        (a.root / 'lease').touch()
    elif a.command == 'stop':
        require_trial(a.root.parent)
        if a.root.name not in ('origin', 'target', 'receiver', 'return', 'ports'):
            raise ValueError('invalid_owned_service_name')
        a.root.mkdir(exist_ok=True)  # startup race: supervisor must see stop
        (a.root / 'stop').touch(exist_ok=True)
    elif a.command == 'hold-ports':
        from .perf_ports import hold_ports
        hold_ports(a.root, [int(p) for p in a.ports.split(',')])
    elif a.command == 'check-ports':
        from .perf_ports import listeners
        require_trial(a.root)
        occupied = listeners([int(p) for p in a.ports.split(',')])
        save(a.root / 'port-check.json', dict(run_id=os.environ['KV_EC_PERF_RUN_ID'],
             status='FAIL' if occupied else 'PASS', listeners=occupied))
        if occupied:
            raise RuntimeError('PORT_RELEASE_FAILED: ' + json.dumps(occupied))
    elif a.command == 'service':
        require_trial(a.root.parent)
        try:
            service(a.root, a.argv[1:] if a.argv[:1] == ['--'] else a.argv, a.resource_interval)
        except BaseException as exc:
            # Parent trial is authenticated above. Publish even if startup
            # failed before service() created its directory.
            a.root.mkdir(exist_ok=True)
            if not (a.root / 'exit.json').exists():
                save(a.root / 'exit.json', dict(returncode=1, reason='supervisor_start_or_cleanup_failed',
                                               error=f'{type(exc).__name__}: {exc}',
                                               traceback=traceback.format_exc()))
            raise
    elif a.command == 'bundle-json':
        with zipfile.ZipFile(a.output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(a.root.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
                    z.write(path, str(path.relative_to(a.root)))
    elif a.command == 'wait-file':
        end = time.monotonic() + 900
        while not a.path.exists():
            if (a.service_root / 'exit.json').exists():
                raise RuntimeError('OWNED_SERVICE_START_FAILED: ' +
                                   json.dumps(read(a.service_root / 'exit.json')) +
                                   '; see ' + str(a.service_root / 'exit.json'))
            if time.monotonic() > end:
                raise TimeoutError('launcher did not publish ' + str(a.path))
            time.sleep(.5)
    else:
        wait_health(a.url, a.service_root, timeout=a.timeout)


if __name__ == '__main__':
    main()
