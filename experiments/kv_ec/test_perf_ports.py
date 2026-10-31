"""CPU regressions for Stage E port ownership; never contacts either node."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from experiments.kv_ec import perf_stage_e as stage
from experiments.kv_ec import perf_ports as ports


class PortLeaseTests(unittest.TestCase):
    def test_guard_lease_outlives_service_cleanup_but_is_bounded(self):
        for age, waits in ((100, 2), (151, 1)):
            with self.subTest(age=age), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'trial-owner.json').write_text('{"run_id":"trial-test"}')
                (root / 'lease').touch()
                os.utime(root / 'lease', (0, 0))
                with patch.dict(os.environ, KV_EC_PERF_RUN_ID='trial-test'), \
                        patch.object(ports, 'PortLease'), patch.object(ports.time, 'time', return_value=age), \
                        patch('threading.Event') as event, patch('signal.signal'):
                    event.return_value.wait.side_effect = [False, True]
                    ports.hold_ports(root, [30984])
                    self.assertEqual(event.return_value.wait.call_count, waits)

    def test_three_trials_reuse_same_ports_and_keep_history(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ports, 'listeners', return_value=[]):
            for i in range(3):
                root = Path(tmp) / f'trial-{i}'
                root.mkdir()
                with ports.PortLease([30984, 30891], root, root.name, lock_root=Path(tmp) / 'locks'):
                    self.assertTrue((root / 'port-plan.json').exists())
                self.assertEqual(json.loads((root / 'port-release.json').read_text())['status'], 'PASS')
            self.assertEqual(len(list(Path(tmp).glob('trial-*/port-plan.json'))), 3)

    def test_exception_releases_locks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ports, 'listeners', return_value=[]):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with ports.PortLease([30984], root, 'trial-test', lock_root=root / 'locks'):
                    raise RuntimeError('injected')
            (root / 'second').mkdir()
            with ports.PortLease([30984], root / 'second', 'trial-test2', lock_root=root / 'locks'):
                pass

    def test_concurrent_trial_cannot_acquire_same_port(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ports, 'listeners', return_value=[]), \
                patch.object(ports.time, 'sleep') as sleep:
            root = Path(tmp)
            with ports.PortLease([30984], root, 'trial-first', lock_root=root / 'locks'):
                with self.assertRaisesRegex(RuntimeError, 'PORT_LEASE_BUSY.*trial-first'):
                    with ports.PortLease([30984], root, 'trial-second', lock_root=root / 'locks'):
                        self.fail('second owner admitted')
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [.1, .2])

    def test_foreign_listener_refused_without_signal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ports, 'listeners',
                return_value=[dict(port=30984, owners=[dict(pid=999, uid=999)])]), \
                patch.object(os, 'kill') as kill, patch.object(os, 'killpg') as killpg:
            with self.assertRaisesRegex(RuntimeError, 'PORT_OCCUPIED'):
                with ports.PortLease([30984], Path(tmp), 'trial-test', lock_root=Path(tmp) / 'locks'):
                    self.fail('foreign listener admitted')
            kill.assert_not_called()
            killpg.assert_not_called()

    def test_different_hosts_check_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(ports, 'listeners', return_value=[]):
                with ports.PortLease([30984], root, 'node0', lock_root=root / 'host0'):
                    with patch.object(ports, 'listeners', return_value=[dict(port=30984, owners=[])]):
                        with self.assertRaisesRegex(RuntimeError, 'PORT_OCCUPIED'):
                            with ports.PortLease([30984], root, 'node1', lock_root=root / 'host1'):
                                pass

    def test_leaked_listener_fails_release_instead_of_hiding_leak(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ports, 'listeners', return_value=[]) as scan:
            with self.assertRaisesRegex(RuntimeError, 'PORT_RELEASE_FAILED'):
                with ports.PortLease([30984], Path(tmp), 'trial-test', lock_root=Path(tmp) / 'locks'):
                    scan.return_value = [dict(port=30984, owners=[dict(run_id='previous-trial')])]
            self.assertEqual(json.loads((Path(tmp) / 'port-release.json').read_text())['status'], 'FAIL')

    def test_legacy_config_exact_ports_and_RS_untouched(self):
        cfg = dict(nodes=[dict(origin_port=30984), dict(origin_port=30985)],
                   return_port=30891, delivery_port=30892, target_port=30986,
                   rs_server=dict(host='192.168.2.245', port=30701))
        before = json.dumps(cfg, sort_keys=True)
        self.assertEqual(ports.trial_ports(cfg, 0), [30891, 30984])
        self.assertEqual(ports.trial_ports(cfg, 1), [30892, 30985, 30986])
        self.assertEqual(json.dumps(cfg, sort_keys=True), before)
        for node in (0, 1):
            self.assertNotIn(30701, ports.trial_ports(cfg, node))

    def test_RS_native_range_excluded_on_both_hosts(self):
        cfg = dict(nodes=[dict(origin_port=30721), dict(origin_port=30985)],
                   return_port=30891, delivery_port=30892, target_port=30986,
                   rs_server=dict(host='192.168.2.245', port=30701))
        with self.assertRaisesRegex(ValueError, 'RS_NATIVE_PORT_RANGE_CONFLICT'):
            ports.validate_reserved_ports(cfg, dict(rdma_port_base=30720, max_stripes=32))

    def test_symlink_lock_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'real').mkdir()
            (root / 'link').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, 'UNSAFE_PORT_LOCK_DIRECTORY'):
                with ports.PortLease([30984], root, 'trial-test', lock_root=root / 'link'):
                    pass


class TunnelTests(unittest.TestCase):
    def test_missing_safe_signal_support_refuses_before_spawn(self):
        with patch.object(stage, 'pidfd_open', side_effect=PermissionError('denied')), \
                patch.object(stage.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'PIDFD_SIGNAL_SUPPORT_REQUIRED'):
                stage.local_ssh_tunnel(NS(cfg=dict(ssh='mock233')), 30984, 30986, Path('/unused'))
            spawn.assert_not_called()

    def test_RS_check_is_health_only_and_rejects_restart(self):
        cfg = dict(rs_server=dict(host='192.168.2.245', port=30701))
        with patch('experiments.kv_ec.remote_rs_client.RemoteRSClient') as client:
            client.return_value.call.return_value = (dict(state='READY', server_epoch='same'), b'')
            self.assertEqual(stage.check_rs_health(cfg, 'same')['state'], 'READY')
            client.assert_called_once_with('192.168.2.245', 30701, timeout=10)
            client.return_value.call.assert_called_once_with('health')
            with self.assertRaisesRegex(RuntimeError, 'EPOCH_CHANGED'):
                stage.check_rs_health(cfg, 'old')

    def test_isolated_SSH_and_failed_readiness_still_cleans(self):
        proc = Mock(pid=12345)
        proc.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(stage.subprocess, 'Popen', return_value=proc) as popen, \
                patch.object(stage, 'process_identity', return_value=dict(pid=12345)), \
                patch.object(stage, 'cleanup_process') as cleanup, \
                patch.object(ports, 'assert_free'), \
                patch.object(ports, 'assert_listener_owned', side_effect=RuntimeError('foreign listener')):
            with self.assertRaisesRegex(RuntimeError, 'foreign listener'):
                stage.local_ssh_tunnel(NS(cfg=dict(ssh='mock233')), 30984, 30986, Path(tmp) / 'tunnel')
            cmd = popen.call_args.args[0]
            self.assertIn('ControlMaster=no', cmd)
            self.assertIn('ControlPersist=no', cmd)
            self.assertEqual(cmd[cmd.index('-S') + 1], 'none')
            cleanup.assert_called_once()

    def test_service_publish_failure_still_cleans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'trial-owner.json').write_text('{"run_id":"trial-test"}')
            with patch.dict(os.environ, KV_EC_PERF_RUN_ID='trial-test'), \
                    patch.object(stage, 'Sampler'), patch.object(stage.signal, 'signal'), \
                    patch.object(stage.subprocess, 'Popen', return_value=Mock(pid=12345)), \
                    patch.object(stage, 'process_identity', return_value=dict(pid=12345)), \
                    patch.object(stage, 'save', side_effect=OSError('publish failed')), \
                    patch.object(stage, 'cleanup_process') as cleanup:
                with self.assertRaisesRegex(OSError, 'publish failed'):
                    stage.service(root / 'origin', [sys.executable, '-c', 'pass'], 1.)
                cleanup.assert_called_once()

    def test_foreign_listener_is_not_contacted_for_health(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(stage, 'require_service_alive', return_value=True), \
                patch.object(stage, 'read', return_value={}), \
                patch.object(ports, 'assert_listener_owned', side_effect=RuntimeError('FOREIGN')), \
                patch.object(stage, 'health_status') as health:
            with self.assertRaisesRegex(RuntimeError, 'FOREIGN'):
                stage.wait_health('http://127.0.0.1:30984', Path(tmp), timeout=1)
            health.assert_not_called()


class ActualCPUProcessTests(unittest.TestCase):
    def test_launcher_exit_and_child_new_group_are_cleaned_by_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / 'child.json'
            child = ('import os,json,time; os.setpgrp(); '
                     f'open({str(ready)!r},"w").write(json.dumps(dict(pid=os.getpid()))); '
                     'time.sleep(30)')
            code = f'import subprocess,sys; subprocess.Popen([sys.executable,"-c",{child!r}])'
            proc = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                    env=dict(os.environ, KV_EC_PERF_RUN_ID='cpu-session-test'),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            owned = dict(**stage.process_identity(proc.pid), run_id='cpu-session-test')
            with patch.dict(os.environ, KV_EC_PERF_RUN_ID='cpu-session-test'):
                try:
                    end = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < end:
                        time.sleep(.02)
                    self.assertTrue(ready.exists())
                    proc.wait(timeout=5)
                    members = stage.owned_members(owned)
                    self.assertTrue(members)
                    self.assertTrue(all(m['pgid'] != owned['pgid'] for m in members))
                finally:
                    stage.cleanup_process(proc, owned, root)
            self.assertEqual(stage.owned_members(owned), [])
            self.assertEqual(json.loads((root / 'process-cleanup.json').read_text())['status'], 'PASS')

    def test_other_UID_session_member_refuses_every_signal(self):
        fake = dict(pid=101, pgid=101, session=101, sid=101, uid=os.getuid(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=10, state='S')
        foreign = dict(fake, pid=102, pgid=102, uid=os.getuid() + 1)
        with patch.object(Path, 'glob', return_value=[Path('/proc/101'), Path('/proc/102')]), \
                patch.object(stage, 'process_identity', side_effect=[fake, foreign]), \
                patch.object(stage, 'pidfd_open') as open_pid:
            with self.assertRaisesRegex(ValueError, 'session_identity_mismatch'):
                stage.signal_owned(fake, signal.SIGTERM)
            open_pid.assert_not_called()

    def test_same_SID_without_trial_token_is_not_adopted(self):
        fake = dict(pid=101, pgid=101, session=101, uid=os.getuid(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=10, state='S')
        with patch.object(Path, 'glob', return_value=[Path('/proc/102')]), \
                patch.object(stage, 'process_identity', return_value=dict(fake, pid=102)), \
                patch.object(Path, 'read_bytes', return_value=b'KV_EC_PERF_RUN_ID=other-trial\0'), \
                patch.object(stage, 'pidfd_open') as open_pid:
            with self.assertRaisesRegex(ValueError, 'trial_environment_mismatch'):
                stage.signal_owned(dict(fake, run_id='this-trial'), signal.SIGTERM)
            open_pid.assert_not_called()

    def test_three_actual_CPU_services_release_same_listener(self):
        try:
            probe = socket.socket()
            probe.bind(('127.0.0.1', 0))
        except PermissionError as exc:
            self.skipTest('sandbox denies TCP bind: ' + str(exc))
        port = probe.getsockname()[1]
        probe.close()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                root = Path(tmp) / f'trial-cpu-{i}'
                root.mkdir()
                (root / 'trial-owner.json').write_text(json.dumps(dict(run_id=root.name)))
                (root / 'lease').touch()
                ready = root / 'ready'
                code = ('import socket,sys,time; s=socket.socket(); '
                        's.bind(("127.0.0.1",int(sys.argv[-1]))); s.listen(); '
                        f'open({str(ready)!r},"w").close(); time.sleep(30)')
                with ports.PortLease([port], root, root.name, lock_root=Path(tmp) / 'locks'):
                    proc = subprocess.Popen([sys.executable, '-m', stage.SELF, 'service',
                        '--root', str(root / 'origin'), '--', sys.executable, '-c', code, '--port', str(port)],
                        env=dict(os.environ, KV_EC_PERF_RUN_ID=root.name),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    try:
                        end = time.monotonic() + 5
                        while not ready.exists() and proc.poll() is None and time.monotonic() < end:
                            time.sleep(.02)
                        self.assertTrue(ready.exists())
                    finally:
                        (root / 'origin').mkdir(exist_ok=True)
                        (root / 'origin/stop').touch()
                        out = proc.communicate(timeout=15)[0]
                    self.assertEqual(proc.returncode, 0, out.decode())
                    self.assertEqual(ports.listeners([port]), [])


if __name__ == '__main__':
    unittest.main()
