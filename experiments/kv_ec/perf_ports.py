"""Host-local Stage E port leases and read-only listener evidence.

Locks coordinate participating trials for the full service lifetime. They are
not socket reservations: kernel bind is authoritative for unrelated programs.
No function in this module signals a process or changes RS/RDMA configuration.
"""
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import time


def trial_ports(cfg, node):
    ports = ([cfg['nodes'][0]['origin_port'], cfg['return_port']] if node == 0 else
             [cfg['nodes'][1]['origin_port'], cfg['target_port'], cfg['delivery_port']])
    if cfg['rs_server']['port'] in ports:
        raise ValueError('RS_CONTROL_PORT_RESERVED')
    return sorted(set(ports))


def validate_reserved_ports(cfg, rs_process):
    base = rs_process['rdma_port_base']
    end = base + max(4, rs_process['max_stripes'] * 4)
    if end > 65536 or any(base <= p < end for n in (0, 1) for p in trial_ports(cfg, n)):
        raise ValueError('RS_NATIVE_PORT_RANGE_CONFLICT')


def listeners(ports):
    """Include unreadable/unidentified owners rather than assuming a free port."""
    from .perf_preflight import listening_inode_map
    from .perf_stage_e import process_identity
    ports = list(ports)
    inode_map = listening_inode_map(ports)
    records = {p: dict(port=p, inodes=sorted(inode_map.get(p, set())), owners=[]) for p in ports}
    records = {p: r for p, r in records.items() if r['inodes']}
    if not records:
        return []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            identity = process_identity(int(entry.name))
            held = set()
            for fd in (entry / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith('socket:['):
                    held.add(target[8:-1])
            for record in records.values():
                if held.intersection(record['inodes']):
                    record['owners'].append(identity)
        except (OSError, ValueError):
            continue
    return list(records.values())


def assert_free(ports, reason='PORT_OCCUPIED'):
    occupied = listeners(ports)
    if occupied:
        raise RuntimeError(reason + ': ' + json.dumps(occupied, sort_keys=True) +
                           '; refusing to signal any listener')


def assert_listener_owned(port, owned):
    records = listeners([port])
    if not records:
        raise ConnectionRefusedError('owned listener not ready')
    from .perf_stage_e import owned_members
    members = owned_members(owned)
    identities = {(p['pid'], p['start_ticks'], p['boot_id'], p['uid']) for p in members}
    for record in records:
        if not record['owners'] or any(
                (p['pid'], p['start_ticks'], p['boot_id'], p['uid']) not in identities
                for p in record['owners']):
            raise RuntimeError('FOREIGN_OR_UNVERIFIABLE_LISTENER: ' + json.dumps(record))


class PortLease:
    """Bounded advisory admission; never rotate ports to hide a leaked listener."""
    def __init__(self, ports, root, run_id, lock_root=None):
        if not ports or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
            raise ValueError('invalid_trial_ports')
        self.ports, self.root, self.run_id = sorted(set(ports)), Path(root), run_id
        # Shared across checkouts/TMPDIRs belonging to this Unix user on a host.
        self.lock_root = Path(lock_root) if lock_root else Path(f'/tmp/kv-ec-port-locks-{os.getuid()}')
        self.files = []

    def __enter__(self):
        from .perf_benchmark import save
        from .perf_stage_e import process_identity
        self.lock_root.mkdir(mode=0o700, exist_ok=True)
        info = self.lock_root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError('UNSAFE_PORT_LOCK_DIRECTORY: ' + str(self.lock_root))
        try:
            owner = dict(run_id=self.run_id, host=socket.gethostname(),
                         **process_identity(os.getpid()))
            for port in self.ports:
                fd = os.open(self.lock_root / f'{port}.lock',
                             os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
                stream = os.fdopen(fd, 'r+')
                self.files.append(stream)
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
                        info.st_nlink != 1 or info.st_mode & 0o077):
                    raise RuntimeError('UNSAFE_PORT_LOCK_FILE')
                for attempt in range(3):
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if attempt == 2:
                            stream.seek(0)
                            raise RuntimeError(f'PORT_LEASE_BUSY port={port} owner={stream.read()[:4096]}')
                        time.sleep(.1 * 2 ** attempt)
                stream.seek(0)
                stream.truncate()
                stream.write(json.dumps(owner))
                stream.flush()
            assert_free(self.ports)
            save(self.root / 'port-plan.json', dict(status='ACQUIRED', ports=self.ports,
                 owner=owner, allocation='legacy-configured-ports',
                 concurrency='advisory flock held through bind and cleanup; kernel bind authoritative'))
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        # Never unlink locks: another waiter may already hold the same inode.
        for stream in reversed(self.files):
            stream.close()
        self.files = []

    def __exit__(self, *exc):
        from .perf_benchmark import save
        try:
            occupied = listeners(self.ports)
            save(self.root / 'port-release.json', dict(status='FAIL' if occupied else 'PASS',
                 run_id=self.run_id, ports=self.ports, listeners=occupied))
            if occupied:
                raise RuntimeError('PORT_RELEASE_FAILED: ' + json.dumps(occupied))
        finally:
            self.close()


def hold_ports(root, ports):
    """Run under the normal lease supervisor on each corresponding host."""
    import signal
    import threading
    from .perf_stage_e import require_trial
    from .perf_benchmark import save
    marker = require_trial(Path(root))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    with PortLease(ports, root, marker['run_id']):
        save(Path(root) / 'ports-ready.json', dict(run_id=marker['run_id'], ports=ports))
        while not stop.wait(.2):
            # Keep admission locked past the model supervisors' 90s lease plus
            # their bounded 30+10+5+5s cleanup. Both guard and supervisor have
            # this fallback in case either one is lost independently.
            if time.time() - (Path(root) / 'lease').stat().st_mtime > 150:
                break
