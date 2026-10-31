"""CPU-only pidfd/Stage E lifecycle regressions; no real SSH, GPU or RDMA."""
import ctypes
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from . import pidfd_compat as compat
from . import perf_stage_e as stage
from .perf_benchmark import read, save
from . import perf_ports as ports


class CompatibilityTests(unittest.TestCase):
    def test_native_preferred(self):
        with patch.object(compat.os, 'pidfd_open', return_value=123, create=True) as op, \
                patch.object(compat.signal, 'pidfd_send_signal', create=True) as send, \
                patch.object(compat, '_libc_call') as fallback, patch.object(compat.os, 'close'):
            self.assertEqual(compat.check_signal_support()['pidfd_send_signal'], 'python')
            op.assert_called_once_with(os.getpid(), 0)
            send.assert_called_once_with(123, 0, None, 0)
            fallback.assert_not_called()

    def test_real_libc_without_python_bindings(self):
        with patch.object(compat.os, 'pidfd_open', None, create=True), \
                patch.object(compat.signal, 'pidfd_send_signal', None, create=True):
            self.assertEqual(compat.check_signal_support()['pidfd_send_signal'], 'libc')
            with self.assertRaises(OSError) as result:
                compat.pidfd_send_signal(-1, 0)
            self.assertEqual(result.exception.errno, errno.EBADF)

    def test_errno_preserved(self):
        for error in (errno.EBADF, errno.ESRCH, errno.ENOSYS, errno.EPERM):
            with self.subTest(error=error), patch.object(compat.ctypes, 'CDLL') as libc:
                libc.return_value.pidfd_send_signal.return_value = -1
                ctypes.set_errno(error)
                with self.assertRaises(OSError) as result:
                    compat._libc_call('pidfd_send_signal', (), 1, 0, None, 0)
                self.assertEqual(result.exception.errno, error)

    def test_missing_libc_and_unsupported_platform_fail_closed(self):
        with patch.object(compat.ctypes, 'CDLL', return_value=object()), \
                self.assertRaises(OSError) as result:
            compat._libc_call('pidfd_open', (), 1, 0)
        self.assertEqual(result.exception.errno, errno.ENOSYS)
        with patch.object(compat.sys, 'platform', 'darwin'), self.assertRaises(OSError):
            compat._libc_call('pidfd_open', (), 1, 0)

    def test_native_error_never_falls_back(self):
        with patch.object(compat.signal, 'pidfd_send_signal', create=True,
                          side_effect=PermissionError(errno.EPERM, 'denied')), \
                patch.object(compat, '_libc_call') as fallback:
            with self.assertRaises(PermissionError):
                compat.pidfd_send_signal(1, 0)
            fallback.assert_not_called()

    def test_remote_capability_failure_precedes_prepare_or_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'trial-cpu-test'
            root.mkdir()
            cfg = Path(__file__).with_name('perf_stage_e_232.example.json')
            calls = []
            def run(host, module, *args, **kwargs):
                calls.append((host.node, args[0]))
                if host.node == 1:
                    raise RuntimeError('PIDFD_SIGNAL_SUPPORT_REQUIRED')
            with patch.dict(os.environ, KV_EC_BENCH_CONFIG=str(cfg),
                            KV_EC_BENCH_TRIAL=str(root), KV_EC_PERF_RUN_ID=root.name), \
                    patch.object(stage.Host, 'run', run), \
                    patch.object(stage.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'PIDFD_SIGNAL_SUPPORT_REQUIRED'):
                    stage.trial()
            self.assertEqual(calls, [(0, 'check-pidfd'), (1, 'check-pidfd')])
            spawn.assert_not_called()
            self.assertTrue((root / 'pidfd-preflight.log').exists())

    def test_kernel_unsupported_fails_before_child(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KV_EC_PERF_RUN_ID='cpu-test'):
            root = Path(tmp)
            save(root / 'trial-owner.json', dict(run_id='cpu-test'))
            with patch.object(compat.os, 'pidfd_open', None, create=True), \
                    patch.object(compat.signal, 'pidfd_send_signal', None, create=True), \
                    patch.object(compat, '_libc_call', side_effect=OSError(errno.ENOSYS, 'unsupported')), \
                    patch.object(stage.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'PIDFD_SIGNAL_SUPPORT_REQUIRED'):
                    stage.service(root / 'ports', ['unused'], 1)
                spawn.assert_not_called()
                self.assertFalse((root / 'ports').exists())

    def test_identity_changes_refused_before_and_after_send(self):
        member = dict(pid=123, start_ticks=9, uid=os.getuid(), pgid=123,
                      session=123, sid=123, boot_id='boot', state='S')
        owner = dict(member, run_id='cpu-test')
        for key in ('pid', 'start_ticks', 'uid', 'pgid', 'session', 'sid', 'boot_id'):
            for after in (False, True):
                changed = dict(member, **{key: 'changed'})
                values = [member, changed] if after else [changed]
                with self.subTest(key=key, after=after), \
                        patch.dict(os.environ, KV_EC_PERF_RUN_ID='cpu-test'), \
                        patch.object(stage, 'owned_members', return_value=[member]), \
                        patch.object(stage, 'pidfd_open', return_value=123), \
                        patch.object(stage.os, 'close'), \
                        patch.object(stage, 'process_identity', side_effect=values), \
                        patch.object(Path, 'read_bytes', return_value=b'KV_EC_PERF_RUN_ID=cpu-test\0'), \
                        patch.object(stage, 'pidfd_send_signal') as send:
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        stage.signal_owned(owner, signal.SIGTERM)
                    self.assertEqual(send.call_count, int(after))

    def test_trial_token_change_refused(self):
        member = stage.process_identity(os.getpid())
        with patch.dict(os.environ, KV_EC_PERF_RUN_ID='cpu-test'), \
                patch.object(Path, 'read_bytes', return_value=b'KV_EC_PERF_RUN_ID=foreign\0'), \
                self.assertRaisesRegex(ValueError, 'trial_environment'):
            stage.verify_signal_identity(member, dict(member, run_id='cpu-test'))

    def test_supervisor_early_failure_wait_file_reports_traceback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KV_EC_PERF_RUN_ID='cpu-test'):
            root = Path(tmp)
            save(root / 'trial-owner.json', dict(run_id='cpu-test'))
            with patch.object(sys, 'argv', ['stage', 'service', '--root', str(root / 'ports'), '--', 'unused']), \
                    patch.object(stage, 'check_signal_support', side_effect=RuntimeError('injected pidfd failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    stage.main()
            self.assertIn('injected pidfd failure', read(root / 'ports/exit.json')['traceback'])
            started = time.monotonic()
            with patch.object(sys, 'argv', ['stage', 'wait-file', '--path', str(root / 'ports-ready.json'),
                                          '--service-root', str(root / 'ports')]):
                with self.assertRaisesRegex(RuntimeError, 'OWNED_SERVICE_START_FAILED.*injected pidfd failure'):
                    stage.main()
            self.assertLess(time.monotonic() - started, 2)


class LifecycleTests(unittest.TestCase):
    def test_three_round_reprefill_tunnel_cleanup(self):
        # Real session/pidfd cleanup; SSH transport and HTTP are CPU fixtures.
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KV_EC_PERF_RUN_ID='cpu-test'):
            for i in range(3):
                def spawn(argv, **kwargs):
                    self.assertIn('ControlMaster=no', argv)
                    return real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
                with patch.object(stage.subprocess, 'Popen', side_effect=spawn), \
                        patch.object(ports, 'assert_free'), patch.object(ports, 'assert_listener_owned'), \
                        patch.object(stage, 'health_status', return_value=200):
                    proc = stage.local_ssh_tunnel(NS(cfg=dict(ssh='fixture')), 30984, 30986, Path(tmp) / str(i))
                stage.cleanup_process(proc, proc.kv_ec_owned, proc.kv_ec_root)
                self.assertEqual(read(proc.kv_ec_root / 'process-cleanup.json')['status'], 'PASS')
                self.assertEqual(stage.owned_members(proc.kv_ec_owned), [])

    def test_three_round_two_level_services_and_guard_release(self):
        # Real supervisor/Port Guard and flock; listener scan mocked because
        # this sandbox forbids sockets. No model, SSH, RS or RDMA is started.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KV_EC_PERF_RUN_ID='cpu-test'):
            for i in range(3):
                root = Path(tmp) / str(i)
                root.mkdir()
                save(root / 'trial-owner.json', dict(run_id='cpu-test'))
                (root / 'lease').touch()
                for name in ('origin', 'target', 'return', 'receiver', 'ports'):
                    argv = [sys.executable, '-c', 'import time; time.sleep(60)']
                    if name == 'ports':
                        argv = [sys.executable, '-m', stage.SELF, 'hold-ports', '--root', str(root), '--ports', '30984']
                    def request_stop():
                        ready = root / ('ports-ready.json' if name == 'ports' else name + '/owned-process.json')
                        deadline = time.monotonic() + 5
                        while not ready.exists() and time.monotonic() < deadline:
                            time.sleep(.02)
                        (root / name / 'stop').touch()
                    stopper = threading.Thread(target=request_stop)
                    stopper.start()
                    with patch.object(ports, 'assert_free'), patch.object(ports, 'listeners', return_value=[]):
                        stage.service(root / name, argv, 1)
                    stopper.join(timeout=6)
                    self.assertEqual(read(root / name / 'process-cleanup.json')['status'], 'PASS')
                    self.assertFalse(stage.owned_members(read(root / name / 'owned-process.json')))
                self.assertTrue((root / 'ports-ready.json').exists())
                self.assertEqual(read(root / 'port-release.json')['status'], 'PASS')
                # Independent same-port lease reacquisition proves no held flock.
                recheck = root / 'recheck'
                recheck.mkdir()
                with patch.object(ports, 'listeners', return_value=[]):
                    with ports.PortLease([30984], recheck, 'cpu-test'):
                        pass


if __name__ == '__main__':
    unittest.main()
