"""CPU ONLY: mock coordination/faults and admission; never RDMA/GPU evidence."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import zipfile

from . import perf_benchmark as bench
from . import perf_preflight as pf
from . import perf_stage_e as stage


EXAMPLE = Path(__file__).with_name('perf_stage_e_232.example.json')


class AdmissionTests(unittest.TestCase):
    def test_example_valid(self):
        pf.validate_config(bench.read(EXAMPLE))

    def test_invalid_parameters(self):
        changes = [('resource_interval', .1), ('resource_interval', float('nan')),
                   ('trial_timeout', 0), ('trial_timeout', 1.5),
                   ('max_node_bytes', True), ('mem_fraction_static', float('inf')),
                   ('remote_root', '/tmp/../root'), ('remote_root', '/tmp/x;bad'),
                   ('return_port', 30984), ('target_port', 30985), ('delivery_port', -1),
                   ('tp_size', 4), ('tp_size', 2.0), ('rs', [2, 2]), ('rs', [2, True])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                cfg = bench.read(EXAMPLE)
                cfg[key] = value
                with self.assertRaises(ValueError):
                    pf.validate_config(cfg)

    def test_invalid_node_parameters(self):
        for key, value in [('gpus', '0,0'), ('gpus', 'auto'), ('gid_index', -1),
                           ('repo', 'relative'), ('ssh', '-oProxyCommand=bad'),
                           ('python', '/tmp/python;bad')]:
            cfg = bench.read(EXAMPLE)
            cfg['nodes'][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pf.validate_config(cfg)

    def reports(self):
        return [dict(checks={}), dict(checks=dict(
            inventory=dict(commit_sha='commit'), rs_disk_free_bytes=100 * 1024**3,
            rs_process=dict(max_stripes=32, max_bytes=8 * 1024**3, ttl=7200)))]

    def metrics(self, stripes=0):
        return dict(stripes=stripes, server_epoch='epoch', resident_data_bytes=0,
                    environment_B=dict(git=dict(returncode=0, stdout='commit\n')))

    def test_capacity_five_and_ten_and_prior_smoke(self):
        cfg = bench.read(EXAMPLE)
        for count, used in [(5, 0), (10, 0), (5, 1), (10, 6)]:
            result = pf.capacity(cfg, self.reports(), dict(state='READY', server_epoch='epoch'),
                                 self.metrics(used), count)
            self.assertEqual(result['requested_slots'], count)

    def test_capacity_rejections(self):
        for key, value, message in [('max_stripes', 4, 'STRIPE_BUDGET'),
                                     ('max_bytes', 10, 'BYTE_BUDGET'), ('ttl', 300, 'TTL_TOO_SHORT')]:
            reports = self.reports()
            reports[1]['checks']['rs_process'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                pf.capacity(bench.read(EXAMPLE), reports, dict(state='READY', server_epoch='epoch'), self.metrics(), 5)
        with self.assertRaisesRegex(ValueError, 'STRIPE_BUDGET'):
            # Expired slots still count. Never assume TTL reclaims a slot.
            pf.capacity(bench.read(EXAMPLE), self.reports(), dict(state='READY', server_epoch='epoch'), self.metrics(30), 5)

    def test_server_restart_and_commit_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'RESTARTED'):
            pf.capacity(bench.read(EXAMPLE), self.reports(), dict(state='READY', server_epoch='old'), self.metrics(), 1)
        m = self.metrics()
        m['environment_B']['git']['stdout'] = 'other'
        with self.assertRaisesRegex(ValueError, 'DIFFERENT_COMMIT'):
            pf.capacity(bench.read(EXAMPLE), self.reports(), dict(state='READY', server_epoch='epoch'), m, 1)

    def test_preflight_only_readonly_RPC(self):
        cfg = bench.read(EXAMPLE)
        reports = self.reports()
        for i, report in enumerate(reports):
            report.update(status='PASS')
            report['checks'].update(inventory=dict(commit_sha='commit', host=f'h{i}', boot_id=f'b{i}',
                tracked_diff_sha256='diff', benchmark_files={}), model_identity=dict(model_sha256='model'),
                disk_free_bytes=100 * 1024**3)
        def run(argv, **kwargs):
            index = int(argv[argv.index('--node') + 1])
            return NS(returncode=0, stdout='KV_EC_PREFLIGHT ' + json.dumps(reports[index]), stderr='')
        with patch.object(pf.subprocess, 'run', side_effect=run), \
                patch.object(stage.Host, 'argv', lambda host, module, *args: [module, *map(str, args)]), \
                patch('experiments.kv_ec.remote_rs_client.RemoteRSClient') as client:
            client.return_value.call.side_effect = [(dict(state='READY', server_epoch='epoch'), b''), (self.metrics(), b'')]
            result = pf.preflight(cfg, '/tmp/no-directory-created', 5)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual([c.args[0] for c in client.return_value.call.call_args_list], ['health', 'get_metrics'])

    def test_cli_bad_interval_and_repeat(self):
        for flags in [('--repeat', '0'), ('--resource-interval', 'nan'), ('--resource-interval', '.1')]:
            with patch.object(sys, 'argv', ['perf', 'repeat', '--config', '/missing', '--output', '/missing', *flags]), \
                    self.assertRaises(SystemExit) as exc:
                bench.main()
            self.assertEqual(exc.exception.code, 2)

    def test_two_parallel_failures_are_both_observed(self):
        called = []
        def fail(i):
            called.append(i)
            raise RuntimeError(str(i))
        with self.assertRaises(RuntimeError):
            stage.parallel(lambda: fail(0), lambda: fail(1))
        self.assertEqual(sorted(called), [0, 1])


class BindingTests(unittest.TestCase):
    def documents(self):
        verdict = dict(restore_id='restore', stripe_id='stripe', handle='handle', manifest_sha256='manifest',
                       rs_source=dict(source_sha256='source'), ranks=[dict(tp_rank=r, run_id='cold-new') for r in (0, 1)])
        run = dict(run_id='cold-new', restore_id='restore', rs_source_binding=verdict['rs_source'])
        protection = dict(plan=dict(run_id='unique-trial'), receipts=[dict(stripe_id='stripe', handle='handle', manifest_sha256='manifest')] * 2)
        return verdict, run, protection

    def binding(self, verdict, run, protection):
        with patch('experiments.kv_ec.rs_cold_source.validate_protection', return_value=dict(run_id='unique-trial')), \
                patch('experiments.kv_ec.online_rs_identity.stripe_id', return_value='stripe'):
            return stage.target_binding(verdict, run, protection, 'unique-trial')

    def test_no_top_level_run_id_required(self):
        docs = self.documents()
        self.assertNotIn('run_id', docs[0])
        self.assertEqual(self.binding(*docs)['target_run_id'], 'cold-new')

    def test_wrong_rank_source_restore_and_stripe_rejected(self):
        for field, value in [('restore_id', 'old'), ('stripe_id', 'old'), ('handle', 'old'),
                              ('rs_source', {}), ('ranks', [dict(tp_rank=0, run_id='old')])]:
            v, r, p = self.documents()
            v[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.binding(v, r, p)

    def test_failed_or_mock_receipt_never_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.save(root / 'acceptance.json', dict(run_id='mock', live_check_exit=1))
            with self.assertRaises(ValueError):
                bench.validate_trial(root, 'mock')


class CoordinatorMockTests(unittest.TestCase):
    """Run the actual trial() ordering with mock Host/SSH/SCP/processes."""

    def run_trial(self, fail=None):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local = base / 'trial-mock-only'
            local.mkdir()
            cfg = bench.read(EXAMPLE)
            cfg['remote_root'] = str(base / 'remote')
            config = base / 'config.json'
            bench.save(config, cfg)
            processes, calls = {}, []
            def option(args, key):
                return Path(args[args.index('--' + key) + 1])
            def run(host, module, *args, **kwargs):
                args = list(map(str, args))
                action = args[0]
                calls.append((host.node, action))
                if action == fail:
                    raise RuntimeError('MOCK_FAILURE_' + action)
                if action == 'prepare':
                    option(args, 'root').mkdir(parents=True, exist_ok=False)
                elif action == 'init-trial':
                    bench.save(option(args, 'root') / 'trial-owner.json', dict(run_id='trial-mock-only'))
                elif action == 'inventory':
                    bench.save(option(args, 'output'), dict(host=f'mock{host.node}'))
                elif action == 'stop':
                    processes[host.node, option(args, 'root').name].done = True
                elif action == 'wait-file':
                    self.assertTrue(option(args, 'path').exists())
                elif action in ('claim', 'plan', 'protect'):
                    bench.save(option(args, 'output'), dict(descriptor=dict(length=100)))
                elif action == 'export':
                    bench.save(option(args, 'output') / 'export.json', dict(receipt=dict(metrics=dict(rdma_native={}))))
                elif action == 'recover':
                    bench.save(option(args, 'output') / 'protected/recovery.json', dict(transfer={}))
                elif action == 'send-source':
                    bench.save(option(args, 'output'), dict(send_stats={}, archive_bytes=100))
                elif action == 'check':
                    bench.save(option(args, 'output') / 'check.json', dict(schema='MOCK_ONLY', real_rdma=False))
                elif action == 'bundle-json':
                    with zipfile.ZipFile(option(args, 'output'), 'x'):
                        pass
            class Proc:
                def __init__(self, argv, **kwargs):
                    self.node = int(argv[1])
                    args = list(map(str, argv))
                    self.name = option(args, 'root').name
                    self.done = False
                    self.returncode = 0
                    if fail == 'launch' and self.name == 'target':
                        raise RuntimeError('MOCK_FAILURE_launch')
                    processes[self.node, self.name] = self
                    tail = args[args.index('--') + 1:]
                    if self.name == 'origin':
                        bench.save(option(tail, 'bundle') / 'request_state.json', dict(restore_id='mock-restore'))
                        bench.save(option(tail, 'output') / 'run.json', dict(run_id='mock-origin'))
                    elif self.name == 'target':
                        self_outer.assertTrue(option(tail, 'ledger').is_dir())
                        self_outer.assertEqual(list(option(tail, 'ledger').iterdir()), [])
                        bench.save(option(tail, 'output') / 'run.json', dict(run_id='mock-target'))
                    elif self.name == 'receiver':
                        bench.save(option(tail, 'output') / 'delivery.json', dict(receive_stats={}, archive_bytes=100))
                def poll(self):
                    return 0 if self.done else None
                def wait(self, **kwargs):
                    self.done = True
                    return 0
            self_outer = self
            def argv(host, module, *args):
                return ['MOCK', str(host.node), module, *map(str, args)]
            def scp(host, remote, local, upload=False):
                if fail == 'scp' and Path(remote).name == 'check.json':
                    raise RuntimeError('MOCK_FAILURE_scp')
                shutil.copy2(local, remote) if upload else shutil.copy2(remote, local)
            env = dict(KV_EC_BENCH_CONFIG=str(config), KV_EC_BENCH_TRIAL=str(local), KV_EC_PERF_RUN_ID=local.name)
            with patch.dict(os.environ, env), patch.object(stage.Host, 'run', run), \
                    patch.object(stage.Host, 'argv', argv), patch.object(stage, 'scp', scp), \
                    patch.object(stage.subprocess, 'Popen', Proc), patch.object(stage.signal, 'signal'), \
                    patch.object(stage, 'target_binding', return_value=dict(target_run_id='mock-target')), \
                    patch.object(stage, 'native_metrics'), patch('experiments.kv_ec.rs_recovery_source.native_stats'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'MOCK_FAILURE'):
                        stage.trial()
                    self.assertTrue((local / 'failure.json').exists())
                    self.assertFalse((local / 'acceptance.json').exists())
                else:
                    stage.trial()
                    self.assertEqual(bench.read(local / 'acceptance.json')['run_id'], local.name)
                self.assertTrue((local / 'json-evidence-233.zip').exists())
                self.assertTrue(all(p.done for p in processes.values()))
                # No RS process launch/stop action exists in the coordinator.
                self.assertNotIn('rs-server', [name for _, name in processes])
                self.assertTrue((local / 'cleanup.json').exists())
                return calls

    def test_complete_mock_flow_creates_ledger_before_target(self):
        calls = self.run_trial()
        self.assertLess(calls.index((0, 'protect')), calls.index((0, 'recover')))

    def test_capture_RDMA_delivery_launch_and_check_failures_preserve_scene(self):
        for stage_name in ('wait-file', 'export', 'recover', 'send-source', 'check-source', 'launch', 'check', 'scp'):
            with self.subTest(stage=stage_name):
                self.run_trial(stage_name)


class ProcessSafetyTests(unittest.TestCase):
    def test_ssh_failure_has_command_log_and_explicit_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KV_EC_BENCH_TRIAL=tmp):
            cfg = bench.read(EXAMPLE)['nodes'][1]
            host = stage.Host(cfg, '/tmp/mock', 'mock-trial', 1)
            with patch.object(stage.subprocess, 'run', return_value=NS(returncode=255)), \
                    self.assertRaisesRegex(RuntimeError, 'node1 exit=255'):
                host.run(stage.E, 'check', '--output', '/tmp/mock')
            records = [json.loads(line) for line in (Path(tmp) / 'commands.jsonl').read_text().splitlines()]
            self.assertEqual(records[0]['status'], 'FAIL')
            self.assertEqual(records[0]['returncode'], 255)

    def test_PID_reuse_refuses_to_signal(self):
        owned = stage.process_identity(os.getpid())
        owned['start_ticks'] -= 1
        with patch.object(stage.os, 'killpg') as kill, self.assertRaisesRegex(ValueError, 'PID_reused'):
            stage.signal_owned(owned, signal.SIGTERM)
        kill.assert_not_called()

    def test_foreign_trial_cannot_stop(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KV_EC_PERF_RUN_ID='new'):
            bench.save(Path(tmp) / 'trial-owner.json', dict(run_id='old'))
            with self.assertRaisesRegex(ValueError, 'ownership_mismatch'):
                stage.require_trial(tmp)

    def test_actual_owned_CPU_service_can_stop(self):
        # Exercise supervisor with only our short-lived CPU sleep, no GPU/RS.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.save(root / 'trial-owner.json', dict(run_id='cpu-test'))
            (root / 'lease').touch()
            env = dict(os.environ, KV_EC_PERF_RUN_ID='cpu-test')
            proc = subprocess.Popen([sys.executable, '-m', stage.SELF, 'service', '--root', str(root / 'origin'),
                '--', sys.executable, '-c', 'pass'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = proc.communicate(timeout=15)[0]
            self.assertEqual(proc.returncode, 0, output.decode())
            self.assertEqual(bench.read(root / 'origin/exit.json')['returncode'], 0)

    def test_expired_lease_stops_only_own_CPU_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.save(root / 'trial-owner.json', dict(run_id='cpu-lease-test'))
            (root / 'lease').touch()
            os.utime(root / 'lease', (time.time() - 100, time.time() - 100))
            env = dict(os.environ, KV_EC_PERF_RUN_ID='cpu-lease-test')
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            try:
                proc = subprocess.Popen([sys.executable, '-m', stage.SELF, 'service', '--root', str(root / 'origin'),
                    '--', sys.executable, '-c', 'import time; time.sleep(30)'],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output = proc.communicate(timeout=15)[0]
                self.assertNotEqual(proc.returncode, 0, output.decode())
                self.assertIn('lease_expired', bench.read(root / 'origin/exit.json')['reason'])
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()  # test owns this CPU process too
                unrelated.wait(timeout=5)


class RepeatMockTests(unittest.TestCase):
    def exercise(self, count=1, driver_code=0, gate_error=False, timeout=False):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = bench.read(EXAMPLE)
            cfg['nodes'][0]['repo'] = str(Path.cwd())
            bench.save(base / 'config.json', cfg)
            source = dict(commit_sha='commit', tracked_diff_sha256='diff', benchmark_files={})
            admission = dict(status='PASS', nodes=[dict(checks=dict(inventory=source))] * 2,
                             rs=dict(server_epoch='epoch'))
            class Proc:
                pid = 12345
                waited = False
                def __init__(self, argv, env, **kwargs):
                    trial = Path(env['KV_EC_BENCH_TRIAL'])
                    for node in (232, 233):
                        bench.save(trial / f'inventory-{node}.json', dict(**source, host=f'host{node}'))
                    bench.save(trial / 'stage-e-check.json', dict(schema='MOCK_ONLY', real_rdma=False))
                def wait(self, **kwargs):
                    if timeout and not self.waited:
                        self.waited = True
                        raise subprocess.TimeoutExpired('mock-driver', 1)
                    return driver_code
            args = NS(config=base / 'config.json', output=base / 'batch', repeat=count, resource_interval=None)
            with patch.object(bench, 'identity', return_value=source), \
                    patch.object(pf, 'preflight', return_value=admission) as checks, \
                    patch.object(bench, 'Sampler'), patch.object(bench.subprocess, 'Popen', Proc), \
                    patch.object(stage, 'process_identity', return_value=dict(pid=12345)), \
                    patch.object(stage, 'signal_owned') as stop, \
                    patch.object(bench, 'validate_trial', side_effect=ValueError('MOCK_GATE_REJECT') if gate_error else None) as gate:
                if driver_code or gate_error or timeout:
                    with self.assertRaises(SystemExit):
                        bench.repeat(args)
                else:
                    bench.repeat(args)
                records = [bench.read(p) for p in sorted(args.output.glob('trial-*/trial.json'))]
                self.assertEqual(len(records), 1 if driver_code or gate_error or timeout else count)
                self.assertEqual(len({r['run_id'] for r in records}), len(records))
                self.assertEqual([call.args[2] for call in checks.call_args_list],
                                 [count] + list(range(count, count - len(records), -1)))
                if timeout:
                    stop.assert_called_once_with(dict(pid=12345), signal.SIGTERM)
                if driver_code or timeout:
                    gate.assert_not_called()
                return records

    def test_five_mock_trials_get_unique_IDs_and_gate_propagation(self):
        records = self.exercise(count=5)
        self.assertTrue(all(r['pass_fail'] == 'PASS' for r in records))

    def test_zero_driver_exit_does_not_override_gate_failure(self):
        records = self.exercise(gate_error=True)
        self.assertEqual(records[0]['pass_fail'], 'FAIL')

    def test_driver_failure_stops_batch_and_preserves_trial(self):
        records = self.exercise(count=5, driver_code=1)
        self.assertEqual(records[0]['status'], 'failed')

    def test_timeout_signals_verified_owned_group_and_records_failure(self):
        records = self.exercise(timeout=True)
        self.assertIn('TimeoutExpired', records[0]['failure_reason'])


class CollectionTests(unittest.TestCase):
    def test_no_partial_statistics_or_duplicate_trials(self):
        # Acceptance itself is mocked here: test collector transactionality,
        # not GPU/RDMA authenticity. Temporary artifacts are discarded.
        stages = ['gpu_kv_snapshot', 'gpu_to_host', 'local_xor_encode', 'gpu_kv_writeback',
                  'gpu_layer_readback_validation', 'scheduler_request_rebuild', 'model_weight_load',
                  'first_restored_decode_cuda', 'failure_to_decode_complete_harness',
                  'new_instance_launch_to_ready_harness']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'batch'
            bench.save(root / 'metadata.json', dict(scheme='two-level', config={}))
            for i in (0, 1):
                trial = root / f'trial-{i}'
                bench.save(trial / 'trial.json', dict(status='accepted', real_rdma_eligible=True,
                    pass_fail='PASS', run_id=f'mock-{i}', verdict_sha256='mock-digest'))
                bench.save(trial / 'stage-e-check.json', {})
                log = trial / 'perf/mock.jsonl'
                log.parent.mkdir()
                log.write_text(''.join(json.dumps(dict(stage=s, run_id=f'mock-{i}', status='ok',
                    duration_ms=7, role='mock', host='mock', start_ns=999, end_ns=1)) + '\n' for s in stages))
                bench.save(trial / 'native-metrics.json', dict(rs_decode_ms=5 if i == 0 else -1))
            out = Path(tmp) / 'summary.json'
            with patch.object(bench, 'validate_trial'), patch.object(bench, 'digest', return_value='mock-digest'):
                bench.collect(NS(roots=[root, root], output=out))
            report = bench.read(out)
            self.assertEqual(len(report['accepted_trials']), 1)
            self.assertEqual(len(report['rejected']), 3)
            self.assertTrue(report['rows'])
            self.assertTrue(all(row['n'] == 1 for row in report['rows']))
            self.assertTrue(all(row['p50_ms'] is None for row in report['rows']))

    def test_failed_and_missing_trials_produce_no_success_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'batch'
            bench.save(root / 'metadata.json', dict(scheme='two-level', config={}))
            bench.save(root / 'trial-0/trial.json', dict(status='failed', failure_reason='RDMA failed'))
            (root / 'trial-1').mkdir()
            out = Path(tmp) / 'summary.json'
            bench.collect(NS(roots=[root], output=out))
            report = bench.read(out)
            self.assertEqual(report['rows'], [])
            self.assertEqual(len(report['rejected']), 2)

    def test_missing_observation_rejects_entire_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'batch'
            bench.save(root / 'metadata.json', dict(scheme='two-level', config={}))
            bench.save(root / 'trial-0/trial.json', dict(status='accepted', real_rdma_eligible=True,
                pass_fail='PASS', run_id='mock', verdict_sha256='digest'))
            bench.save(root / 'trial-0/stage-e-check.json', {})
            out = Path(tmp) / 'summary.json'
            with patch.object(bench, 'validate_trial'), patch.object(bench, 'digest', return_value='digest'):
                bench.collect(NS(roots=[root], output=out))
            report = bench.read(out)
            self.assertEqual(report['rows'], [])
            self.assertEqual(len(report['missing_metrics']), 1)


if __name__ == '__main__':
    unittest.main()
