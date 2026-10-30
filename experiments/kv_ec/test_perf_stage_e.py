"""CPU ONLY: mock coordination/faults and admission; never RDMA/GPU evidence."""

import json
import ast
import asyncio
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
from unittest.mock import MagicMock, Mock, patch
import zipfile

from . import perf_benchmark as bench
from . import perf_preflight as pf
from . import perf_stage_e as stage
from . import reprefill_server
from . import gpu_admission as admission_policy


def mock_gpu_policy(selected='strict', fraction=.6):
    """CPU fixture, not real GPU evidence."""
    return admission_policy.metadata(selected, fraction, [dict(
        gpu_index=str(i), gpu_uuid=f'GPU-MOCK-{i}', total_memory_bytes=100,
        free_memory_bytes=90, required_estimated_memory_bytes=80,
        other_tasks=[], remaining_safety_margin_bytes=10) for i in range(2)])


EXAMPLE = Path(__file__).with_name('perf_stage_e_232.example.json')


class GPUAdmissionModeTests(unittest.TestCase):
    def model(self, directory):
        from .test_gpu_memory_budget import GPUMemoryBudgetTests
        return str(GPUMemoryBudgetTests().model(directory))

    def rows(self):
        from .gpu_memory_budget import GIB
        return {str(i): dict(index=str(i), uuid=f'GPU-{i}', total_bytes=total * GIB,
                            free_bytes=free * GIB)
                for i, (total, free) in enumerate(((80, 64), (64, 50), (80, 60)))}

    def check(self, model, selected, rows=None, gpus='0,1', fraction=.35):
        from . import launch_online
        args = NS(gpus=gpus, model=model, tp=2, gpu_admission_mode=selected,
                  mem_fraction_static=fraction)
        tasks = {'GPU-0': [dict(pid='987', name='other-user-task')]}
        with patch.object(launch_online, 'read_gpu_memory', return_value=rows or self.rows()), \
                patch.object(launch_online, 'read_gpu_tasks', return_value=tasks):
            return args, launch_online.check_gpu_resources(args)

    def test_default_and_invalid_config(self):
        cfg = bench.read(EXAMPLE)
        cfg.pop('gpu_admission')
        pf.validate_config(cfg)
        self.assertEqual(admission_policy.config_mode(cfg), 'strict')
        for value in (None, [], 'shared', {}, {'mode': None}, {'mode': []}, {'mode': 'debug'},
                      {'mode': 'shared', 'extra': True}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'GPU_ADMISSION'):
                pf.validate_config(dict(cfg, gpu_admission=value))
        for value in (True, None, '0.35', [.35], float('nan')):
            with self.subTest(fraction=value), self.assertRaises(ValueError):
                pf.validate_config(dict(cfg, mem_fraction_static=value))
        cfg['nodes'][1]['gpu_admission_mode'] = 'shared'
        with self.assertRaisesRegex(ValueError, 'MISMATCH'):
            pf.validate_config(cfg)

    def test_strict_busy_rejected_shared_busy_and_imbalance_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(tmp)
            with self.assertRaisesRegex(ValueError, 'busy'):
                self.check(model, 'strict')
            _, resources = self.check(model, 'shared')
            self.assertEqual(resources[0]['other_tasks'][0]['pid'], '987')
            self.assertNotEqual(resources[0]['total_memory_bytes'], resources[1]['total_memory_bytes'])
            self.assertNotEqual(resources[0]['free_memory_bytes'], resources[1]['free_memory_bytes'])
            self.assertTrue(all(r['remaining_safety_margin_bytes'] > 0 for r in resources))

    def test_one_insufficient_gpu_never_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows['1']['free_bytes'] = 1
            with self.assertRaisesRegex(ValueError, 'GPU index=1'):
                self.check(self.model(tmp), 'shared', rows)

    def test_actual_min_free_static_pool_must_fit_weights(self):
        from .gpu_memory_budget import GIB
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            for row in rows.values():
                row.update(total_bytes=24 * GIB, free_bytes=20 * GIB)
            with self.assertRaisesRegex(ValueError, 'GPU_STATIC_KV_POOL_INSUFFICIENT'):
                self.check(self.model(tmp), 'shared', rows, fraction=.4)

    def test_missing_or_invalid_memory_and_model_rejected(self):
        from .gpu_memory_budget import parse_gpu_rows
        for text in ('0, GPU-a, 0, 0', '0, GPU-a, 24, 25', '0, GPU-a, N/A, 10'):
            with self.assertRaises(ValueError):
                parse_gpu_rows(text)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.check(tmp, 'shared')

    def test_auto_maximizes_minimum_margin_and_strict_uses_idle_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(tmp)
            # GPU0 has the largest margin, GPU2 is next. GPU1 is weakest.
            args, resources = self.check(model, 'shared', gpus='auto')
            self.assertEqual(set(args.gpus.split(',')), {'0', '2'})
            args, _ = self.check(model, 'strict', gpus='auto')
            self.assertEqual(set(args.gpus.split(',')), {'1', '2'})
            self.assertEqual(len(resources), 2)

    def test_host_local_and_SSH_override_parent_without_mutating_it(self):
        for selected in ('strict', 'shared'):
            with patch.dict(os.environ, SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK='unexpected'):
                for node in (0, 1):
                    cfg = bench.read(EXAMPLE)['nodes'][node]
                    host = stage.Host(cfg, '/tmp/cpu-mode', 'cpu-mode', node, selected)
                    command = host.argv('module')
                    self.assertIn('SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=' +
                                  ('1' if selected == 'strict' else '0'), ' '.join(command))
                    self.assertIn('KV_EC_GPU_ADMISSION_MODE=' + selected, ' '.join(command))
                self.assertEqual(os.environ['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK'], 'unexpected')

    def test_coordinator_launcher_and_startup_mismatches_reject(self):
        with patch.dict(os.environ, KV_EC_GPU_ADMISSION_MODE='shared'):
            with self.assertRaisesRegex(ValueError, 'MISMATCH'):
                admission_policy.args_mode(NS(gpu_admission_mode='strict'))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(gpu_admission={'mode': 'shared'}, mem_fraction_static=.6)
            for name in ('node0-reference/run.json', 'origin-run-233.json', 'target-run.json'):
                bench.save(Path(tmp) / name, mock_gpu_policy('shared'))
            bench.validate_gpu_startups(tmp, cfg, 'two-level')
            (Path(tmp) / 'origin-run-233.json').write_text(json.dumps(mock_gpu_policy('strict')))
            with self.assertRaisesRegex(ValueError, 'MISMATCH'):
                bench.validate_gpu_startups(tmp, cfg, 'two-level')

    def test_reprefill_rechecks_before_exec_and_sets_child_only_environment(self):
        for selected in ('strict', 'shared'):
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
                args = NS(output=Path(tmp) / 'new', gpus='0,1', model='/mock', tp=2,
                          mem_fraction_static=.35, gpu_admission_mode=selected,
                          port=30985, prefill_chunk_size=-1)
                with patch.object(reprefill_server, 'check_gpu_resources', return_value=mock_gpu_policy(selected)['gpu_resources_before_start']) as check, \
                        patch.object(reprefill_server, 'source_identity', return_value={}), \
                        patch.object(reprefill_server, 'check_port'), \
                        patch.object(reprefill_server.os, 'execve') as execute:
                    reprefill_server.launch(args)
                self.assertEqual(check.call_count, 2)
                env = execute.call_args.args[2]
                self.assertEqual(env['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK'], '1' if selected == 'strict' else '0')
                self.assertNotIn('SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK', os.environ)
                evidence = bench.read(args.output / 'run.json')
                self.assertEqual(evidence['gpu_admission_mode'], selected)
                self.assertFalse(evidence['isolated_performance_eligible'])

    def test_resource_change_during_hashing_refuses_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = NS(output=Path(tmp) / 'new', gpu_admission_mode='shared', port=30985, model='/mock')
            with patch.object(reprefill_server, 'check_gpu_resources', side_effect=[[], ValueError('INSUFFICIENT_GPU_MEMORY: changed')]), \
                    patch.object(reprefill_server, 'source_identity', return_value={}), \
                    patch.object(reprefill_server, 'check_port'), \
                    patch.object(reprefill_server.os, 'execve') as execute:
                with self.assertRaisesRegex(ValueError, 'changed'):
                    reprefill_server.launch(args)
            execute.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_restore_launch_records_mode_and_overrides_only_child_environment(self):
        from . import verify_gpu_restore as restore
        from . import launch_online
        for selected in ('strict', 'shared'):
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
                checkpoint = Path(tmp) / 'checkpoint'
                checkpoint.write_bytes(b'CPU_MOCK_NOT_REAL_KV')
                args = NS(checkpoint=checkpoint, output=Path(tmp) / 'new', gpus='0,1',
                    model='/mock', mem_fraction_static=.6, port=30985, timeout=120,
                    gpu_admission_mode=selected)
                with patch.object(restore, 'validate_checkpoint', return_value={'ranks': [{'model_sha256': 'mock'}]}), \
                        patch('experiments.kv_ec.debug_capture.source_identity', return_value={'model_sha256': 'mock'}), \
                        patch.object(launch_online, 'check_gpu_resources', return_value=mock_gpu_policy(selected)['gpu_resources_before_start']) as check, \
                        patch('experiments.kv_ec.launch_debug.check_port'), \
                        patch.object(restore.os, 'execve') as execute:
                    restore.launch(args)
                self.assertEqual(check.call_count, 2)
                self.assertEqual(execute.call_args.args[2]['KV_EC_GPU_ADMISSION_MODE'], selected)
                self.assertNotIn('KV_EC_GPU_ADMISSION_MODE', os.environ)
                admission_policy.validate_evidence(bench.read(args.output / 'run.json'), selected)

    def test_preflight_rejects_cross_node_mode_mismatch_before_RS(self):
        cfg = bench.read(EXAMPLE)
        reports = []
        for selected in ('strict', 'shared'):
            reports.append(dict(status='PASS', checks={}, **mock_gpu_policy(selected)))
        def run(argv, **kwargs):
            i = int(argv[argv.index('--node') + 1])
            return NS(returncode=0, stdout='KV_EC_PREFLIGHT ' + json.dumps(reports[i]), stderr='')
        with patch.object(pf.subprocess, 'run', side_effect=run), \
                patch.object(stage.Host, 'argv', lambda host, module, *args: [module, *map(str, args)]), \
                patch('experiments.kv_ec.remote_rs_client.RemoteRSClient') as client:
            result = pf.preflight(cfg, '/tmp/not-created-by-test', 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('GPU_ADMISSION_MODE_MISMATCH', result['error'])
        client.assert_not_called()

    def test_simulated_OOM_preserves_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.save(root / 'trial-owner.json', dict(run_id='cpu-oom'))
            (root / 'lease').touch()
            env = dict(os.environ, KV_EC_PERF_RUN_ID='cpu-oom')
            result = subprocess.run([sys.executable, '-m', stage.SELF, 'service',
                '--root', str(root / 'origin'), '--', sys.executable, '-c',
                'import sys; print("CUDA out of memory", flush=True); sys.exit(1)'],
                env=env, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('GPU_RUNTIME_OOM', bench.read(root / 'origin/exit.json')['gpu_failure_reason'])
            self.assertFalse((root / 'acceptance.json').exists())


class HostEnvironmentTests(unittest.TestCase):
    def test_local_and_remote_shell_inherit_node_PATH_in_children(self):
        # Execute the exact remote shell payload with a default-only PATH;
        # SSH transport is mocked by a local /bin/sh, with no real SSH/GPU work.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            (repo / 'env_probe.py').write_text(
                'import json, os, subprocess, sys\n'
                'keys = ["PATH", "PYTHONPATH", "KV_EC_PERF_DIR", "KV_EC_PERF_RUN_ID", "KV_EC_PERF_ROLE", '
                '"TMPDIR", "TMP", "TEMP", "KV_EC_GPU_ADMISSION_MODE", '
                '"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", '
                '"CUDA_HOME", "CUDA_PATH", "CUDACXX", "NVCC", "LIBRARY_PATH", "LD_LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH"]\n'
                'env = {k: os.environ[k] for k in keys}\n'
                'child = json.loads(subprocess.check_output([sys.executable, "-c", '
                '"import json,os; print(json.dumps(dict(os.environ)))"], text=True))\n'
                'print(json.dumps(dict(env=env, child=child, ninja=subprocess.check_output('
                '["ninja", "--version"], text=True).strip(), nvcc=subprocess.check_output('
                '["nvcc", "--version"], text=True).strip())))\n')
            for node in (0, 1):
                with self.subTest(node=node):
                    # Also exercise shell quoting of a prefix containing spaces/$.
                    prefix = Path(tmp) / f'conda node{node} $literal'
                    bin_dir = prefix / 'bin'
                    bin_dir.mkdir(parents=True)
                    ninja = bin_dir / 'ninja'
                    ninja.write_text('#!/bin/sh\nprintf "1.13.2\\n"\n')
                    ninja.chmod(0o755)
                    cfg = dict(repo=str(repo), python=sys.executable, conda_prefix=str(prefix),
                               cuda_home=str(Path(tmp) / f'cuda node{node} $literal'),
                               tmpdir=str(Path(tmp) / f'tmp node{node} $literal'))
                    cuda_bin = Path(cfg['cuda_home']) / 'bin'
                    cuda_bin.mkdir(parents=True)
                    Path(cfg['tmpdir']).mkdir(parents=True)
                    for binary, version in ((cuda_bin / 'nvcc', f'toolkit-node{node}'),
                                             (bin_dir / 'nvcc', 'WRONG_CONDA_NVCC')):
                        binary.write_text(f'#!/bin/sh\nprintf "{version}\\n"\n')
                        binary.chmod(0o755)
                    if node:
                        cfg['ssh'] = 'test@233'
                    host = stage.Host(cfg, Path(tmp) / 'trial', 'env-cpu-trial', node)
                    with patch.dict(os.environ, PATH='/usr/bin:/bin'):
                        argv = host.argv('env_probe')
                    if node:
                        self.assertEqual(argv[0], 'ssh')
                        cmd = ['/bin/sh', '-c', argv[-1]]
                        inherited_path = '/bin:/usr/bin'  # remote PATH differs from node0
                    else:
                        cmd = argv
                        inherited_path = '/usr/bin:/bin'
                    suffix = f'/node{node}/inherited'
                    process_env = dict(os.environ, PATH=inherited_path,
                                       CUDA_HOME=':/bad/cuda', CUDA_PATH=':/bad/cuda', CUDACXX='bad', NVCC='bad')
                    for key in ('LIBRARY_PATH', 'LD_LIBRARY_PATH', 'CPATH', 'CPLUS_INCLUDE_PATH'):
                        process_env[key] = suffix
                    # Local env assignments expand when argv is constructed;
                    # remote assignments expand in the receiving shell.
                    if not node:
                        with patch.dict(os.environ, process_env, clear=True):
                            cmd = host.argv('env_probe')
                    result = subprocess.run(cmd, cwd=repo, env=process_env,
                                            capture_output=True, text=True, check=True, timeout=10)
                    result = json.loads(result.stdout)
                    expected = dict(host.env, **{key: value + ':' + (inherited_path if key == 'PATH' else suffix)
                                                for key, value in host.path_prefixes.items()})
                    self.assertEqual(result['env'], expected)
                    self.assertEqual({k: result['child'][k] for k in expected}, expected)
                    self.assertEqual(result['ninja'], '1.13.2')
                    self.assertEqual(result['nvcc'], f'toolkit-node{node}')
                    with patch.dict(os.environ, result['env']):
                        report = pf.check_ninja(cfg)
                    self.assertEqual(report, dict(path=expected['PATH'], executable=str(ninja), version='1.13.2'))

    def test_remote_unset_search_paths_have_no_empty_current_directory_entry(self):
        cfg = bench.read(EXAMPLE)['nodes'][1]
        host = stage.Host(cfg, '/tmp/cpu-only', 'env-cpu-trial', 1)
        # Inspect the actual shell assignment parsing without starting SSH.
        shell = host.argv('ignored')[-1].split(' && exec env ', 1)[1]
        assignments = shell.rsplit(' ' + cfg['python'], 1)[0]
        command = 'exec env ' + assignments + ' /usr/bin/env'
        result = subprocess.run(['/bin/sh', '-c', command], env={'PATH': '/usr/bin:/bin'},
                                capture_output=True, text=True, check=True, timeout=10)
        env = dict(line.split('=', 1) for line in result.stdout.splitlines())
        for key in ('LIBRARY_PATH', 'LD_LIBRARY_PATH', 'CPATH', 'CPLUS_INCLUDE_PATH'):
            self.assertEqual(env[key], host.path_prefixes[key])
            self.assertNotIn('', env[key].split(':'))

    def test_preflight_rejects_missing_ninja_before_resource_checks(self):
        cfg = bench.read(EXAMPLE)
        node = cfg['nodes'][1]
        with patch.dict(os.environ, dict(admission_policy.environment('strict'), PATH=node['conda_prefix'] + '/bin:/usr/bin:/bin')), \
                patch.object(pf.sys, 'prefix', node['conda_prefix']), \
                patch.object(pf, 'identity', return_value={}), \
                patch.object(pf.shutil, 'which', return_value=None), \
                patch.object(pf, 'free_ports') as resource_check:
            report = pf.probe(cfg, 1, '/tmp/not-created')
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('NINJA_NOT_FOUND', report['error'])
        resource_check.assert_not_called()

    def test_preflight_rejects_wrong_PATH(self):
        with patch.dict(os.environ, PATH='/usr/bin:/bin'), \
                self.assertRaisesRegex(ValueError, 'CONDA_PATH_MISMATCH'):
            pf.check_ninja(dict(conda_prefix='/configured/conda'))

    def test_preflight_rejects_broken_or_hanging_ninja(self):
        for outcome in (NS(returncode=1, stdout='', stderr='broken'),
                        NS(returncode=0, stdout='', stderr=''),
                        FileNotFoundError('gone'), subprocess.TimeoutExpired('ninja', 10)):
            with self.subTest(outcome=outcome), \
                    patch.dict(os.environ, PATH='/configured/conda/bin:/usr/bin'), \
                    patch.object(pf.shutil, 'which', return_value='/configured/conda/bin/ninja'), \
                    patch.object(pf.subprocess, 'run', **({'side_effect': outcome} if isinstance(outcome, Exception)
                                                      else {'return_value': outcome})), \
                    self.assertRaisesRegex(ValueError, 'NINJA_(UNAVAILABLE|VERSION_FAILED)'):
                pf.check_ninja(dict(conda_prefix='/configured/conda'))


class CudaToolchainTests(unittest.TestCase):
    """Mock toolchain faults are CPU tests, not real CUDA/RDMA evidence."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        home = self.root / 'cuda'
        for name in ('bin', 'include', 'lib64'):
            (home / name).mkdir(parents=True)
        for name in ('cuda.h', 'cuda_runtime.h', 'cuda_runtime_api.h'):
            (home / 'include' / name).touch()
        (home / 'bin/nvcc').write_text('#!/bin/sh\nexit 0\n')
        (home / 'bin/nvcc').chmod(0o755)
        self.runtime = home / 'lib64/libcudart.so'
        self.runtime.touch()
        tmpdir = self.root / 'tmpdir'
        tmpdir.mkdir()
        self.node = dict(cuda_home=str(home), conda_prefix='/configured/conda', tmpdir=str(tmpdir))
        fixed, prefixes = pf.cuda_environment(self.node)
        env = patch.dict(os.environ, dict(fixed, **prefixes))
        env.start()
        self.addCleanup(env.stop)
        import torch
        torch_cuda = patch.object(torch.version, 'cuda', '13.0')
        torch_cuda.start()
        self.addCleanup(torch_cuda.stop)

    def mock_command(self, argv, **kwargs):
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(kwargs['env']['TMPDIR'], self.node['tmpdir'])
        self.assertEqual(kwargs['env']['TMP'], self.node['tmpdir'])
        self.assertEqual(kwargs['env']['TEMP'], self.node['tmpdir'])
        if argv[-1] == '--version':
            return NS(returncode=0, stdout='Cuda compilation tools, release 13.0, V13.0.88', stderr='')
        if '-c' in argv and argv[0].endswith('/nvcc'):
            source = Path(argv[argv.index('-c') + 2]).read_text()
            self.assertIn('__global__', source)
            self.assertNotIn('<<<', source)
        if argv[0] == 'c++':
            self.assertIn('-lcudart', argv)
            self.assertFalse(any(arg.startswith('-L') for arg in argv))
        data = dict(header_version=13000, runtime_paths=[str(self.runtime)])
        return NS(returncode=0, stdout=json.dumps(data) if argv[0] == sys.executable else '', stderr='')

    def check(self):
        return pf.check_cuda_toolchain(self.node, self.root)

    def test_mock_compile_link_and_fresh_dynamic_loader_and_own_temp_cleanup(self):
        with patch.object(pf.subprocess, 'run', side_effect=self.mock_command) as run:
            result = self.check()
        self.assertEqual(run.call_count, 4)
        self.assertTrue(result['nvcc_compile'] and result['host_link'] and result['dynamic_load'])
        self.assertFalse(result['gpu_context_created'] or result['cuda_api_called'])
        self.assertIsNone(result['compatibility_warning'])
        self.assertEqual(list(self.root.glob('kv-ec-cuda-probe-*')), [])

    def test_tmpdir_probe_requires_writable_space_and_creates_no_residue(self):
        result = pf.check_tmpdir(self.node, minimum_bytes=1)
        self.assertEqual(result['path'], self.node['tmpdir'])
        self.assertTrue(result['tempfile_probe'])
        with self.assertRaisesRegex(ValueError, 'TMPDIR_NOT_WRITABLE'):
            pf.check_tmpdir(dict(self.node, tmpdir=str(self.root / 'missing')))

    def test_invalid_CUDA_HOME_and_directory_config_rejected_before_launch(self):
        for value in (None, '', '/cuda:/other', ':/usr/local/cuda', '/cuda:', 'relative', '/', '/cuda/../other'):
            cfg = bench.read(EXAMPLE)
            cfg['nodes'][0]['cuda_home'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'CUDA_HOME_INVALID'):
                pf.validate_config(cfg)
        for key, value in (('cuda_library_dirs', '/cuda/lib64'), ('cuda_include_dirs', []),
                           ('cuda_library_dirs', ['/cuda/lib64/stubs'])):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'CUDA_CONFIG_INVALID'):
                pf.cuda_environment(dict(self.node, **{key: value}))

    def test_missing_header_nvcc_and_unversioned_runtime_rejected(self):
        for target, error in (('include/cuda.h', 'CUDA_HEADER_NOT_FOUND'),
                              ('bin/nvcc', 'CUDA_NVCC_INVALID'), ('lib64/libcudart.so', 'CUDA_RUNTIME_NOT_FOUND')):
            p = Path(self.node['cuda_home']) / target
            original = p.read_bytes()
            mode = p.stat().st_mode
            p.unlink()
            try:
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, error):
                    self.check()
            finally:
                p.write_bytes(original)
                p.chmod(mode)

    def test_wrong_process_environment_is_rejected(self):
        with patch.dict(os.environ, CUDA_HOME=':/usr/local/cuda'), \
                self.assertRaisesRegex(ValueError, 'CUDA_ENV_MISMATCH'):
            self.check()

    def test_preflight_toolchain_failure_rejects_before_resource_admission(self):
        cfg = bench.read(EXAMPLE)
        node = cfg['nodes'][1]
        with patch.dict(os.environ, admission_policy.environment('strict')), \
                patch.object(pf.sys, 'prefix', node['conda_prefix']), \
                patch.object(pf, 'identity', return_value={}), \
                patch.object(pf, 'check_ninja', return_value={}), \
                patch.object(pf, 'check_cuda_toolchain', side_effect=ValueError('CUDA_RUNTIME_NOT_FOUND')) as check, \
                patch.object(pf, 'free_ports') as resource_check:
            result = pf.probe(cfg, 1, str(self.root))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('CUDA_RUNTIME_NOT_FOUND', result['error'])
        check.assert_called_once_with(node, str(self.root))
        resource_check.assert_not_called()

    def test_only_versioned_runtime_and_wrong_runtime_major_are_rejected(self):
        versioned = self.runtime.with_name('libcudart.so.12.8.90')
        self.runtime.rename(versioned)
        with self.assertRaisesRegex(ValueError, 'CUDA_RUNTIME_NOT_FOUND'):
            self.check()
        self.runtime.symlink_to(versioned.name)
        with patch.object(pf.subprocess, 'run', side_effect=self.mock_command), \
                self.assertRaisesRegex(ValueError, 'CUDA_RUNTIME_VERSION_MISMATCH'):
            self.check()

    def test_compile_link_loader_errors_and_timeouts_are_explicit(self):
        for index, error in ((0, 'CUDA_NVCC_INVALID'), (1, 'CUDA_JIT_COMPILE_FAILED'),
                             (2, 'CUDA_JIT_LINK_FAILED'), (3, 'CUDA_JIT_LINK_FAILED')):
            for timeout in (False, True):
                calls = []
                def run(argv, **kwargs):
                    calls.append(argv)
                    if len(calls) == index + 1:
                        if timeout:
                            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
                        return NS(returncode=1, stdout='', stderr='mock missing compiler/runtime')
                    return self.mock_command(argv, **kwargs)
                with self.subTest(index=index, timeout=timeout), patch.object(pf.subprocess, 'run', side_effect=run), \
                        self.assertRaisesRegex(ValueError, error):
                    self.check()
                self.assertEqual(list(self.root.glob('kv-ec-cuda-probe-*')), [])

    def test_torch_major_mismatch_fails_and_minor_mismatch_is_reported(self):
        import torch
        with patch.object(pf.subprocess, 'run', side_effect=self.mock_command):
            with patch.object(torch.version, 'cuda', '12.8'), \
                    self.assertRaisesRegex(ValueError, 'CUDA_TORCH_VERSION_MISMATCH'):
                self.check()
            with patch.object(torch.version, 'cuda', '13.1'):
                self.assertIn('CUDA_MINOR_VERSION_MISMATCH', self.check()['compatibility_warning'])

    def test_loader_wrong_runtime_or_header_version_rejected(self):
        for data, error in ((dict(header_version=12080, runtime_paths=[str(self.runtime)]), 'CUDA_HEADER_VERSION_MISMATCH'),
                            (dict(header_version=13000, runtime_paths=['/other/libcudart.so']), 'CUDA_JIT_LINK_FAILED')):
            def run(argv, **kwargs):
                result = self.mock_command(argv, **kwargs)
                if argv[0] == sys.executable:
                    result.stdout = json.dumps(data)
                return result
            with self.subTest(data=data), patch.object(pf.subprocess, 'run', side_effect=run), \
                    self.assertRaisesRegex(ValueError, error):
                self.check()


class WaitHealthTests(unittest.TestCase):
    """No HTTP listener, GPU or Scheduler is launched by these tests."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'origin'
        self.owned = dict(**stage.process_identity(os.getpid()), run_id='health-cpu-trial')
        bench.save(self.root.parent / 'trial-owner.json', dict(run_id='health-cpu-trial'))
        bench.save(self.root / 'owned-process.json', self.owned)
        env = patch.dict(os.environ, KV_EC_PERF_RUN_ID='health-cpu-trial')
        env.start()
        self.addCleanup(env.stop)
        self.clock = 0.

    def advance(self, seconds):
        self.clock += seconds

    def wait(self, timeout=1):
        with patch.object(stage.time, 'monotonic', side_effect=lambda: self.clock), \
                patch.object(stage.time, 'sleep', side_effect=self.advance):
            stage.wait_health('http://127.0.0.1:30984/', self.root, timeout=timeout)

    def test_HTTP_200_empty_body_is_ready_and_only_health_is_requested(self):
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        with patch.object(stage.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = response
            self.wait(timeout=2)
            opener.return_value.open.assert_called_once_with('http://127.0.0.1:30984/health', timeout=2.)
        response.read.assert_not_called()  # the endpoint is not a JSON API

    def test_non_200_never_grants_ready(self):
        for status in (204, 301, 302, 404, 500, 503):
            with self.subTest(status=status), patch.object(stage, 'health_status', return_value=status):
                self.clock = 0.
                with self.assertRaisesRegex(TimeoutError, f'HTTP {status}, expected 200'):
                    self.wait()

    def test_starting_503_retries_until_200(self):
        with patch.object(stage, 'health_status', side_effect=[503, 200]) as probe:
            self.wait(timeout=2)
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(self.clock, .5)

    def test_service_exit_receipt_fails_before_HTTP(self):
        bench.save(self.root / 'exit.json', dict(returncode=1, reason='mock_startup_failure'))
        with patch.object(stage, 'health_status') as probe, self.assertRaisesRegex(RuntimeError, 'exited before ready'):
            self.wait()
        probe.assert_not_called()

    def test_dead_process_fails_without_exit_receipt(self):
        with patch.object(stage, 'process_identity', side_effect=ProcessLookupError('gone')), \
                patch.object(stage, 'health_status') as probe, self.assertRaisesRegex(RuntimeError, 'exited or identity unavailable'):
            self.wait()
        probe.assert_not_called()

    def test_recycled_PID_and_foreign_trial_cannot_be_ready(self):
        changed = dict(self.owned, start_ticks=self.owned['start_ticks'] + 1)
        with patch.object(stage, 'process_identity', return_value=changed), \
                patch.object(stage, 'health_status') as probe, self.assertRaisesRegex(RuntimeError, 'process identity mismatch'):
            self.wait()
        probe.assert_not_called()
        with patch.dict(os.environ, KV_EC_PERF_RUN_ID='foreign-trial'), \
                self.assertRaisesRegex(ValueError, 'trial_ownership_mismatch'):
            self.wait()

    def test_service_exit_during_HTTP_200_is_not_ready(self):
        def response(*args, **kwargs):
            bench.save(self.root / 'exit.json', dict(returncode=1))
            return 200
        with patch.object(stage, 'health_status', side_effect=response), \
                self.assertRaisesRegex(RuntimeError, 'exited before ready'):
            self.wait()

    def test_transport_errors_timeout_with_bounded_request_deadline(self):
        def timeout(url, timeout):
            self.assertLessEqual(timeout, 1 - self.clock)
            self.advance(timeout)
            raise TimeoutError('mock HTTP read timed out')
        with patch.object(stage, 'health_status', side_effect=timeout) as probe, \
                self.assertRaisesRegex(TimeoutError, 'mock HTTP read timed out'):
            self.wait()
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(self.clock, 1.)

    def test_missing_owned_identity_never_uses_HTTP_as_port_only_readiness(self):
        (self.root / 'owned-process.json').unlink()
        with patch.object(stage, 'health_status') as probe, self.assertRaisesRegex(TimeoutError, 'identity not yet published'):
            self.wait()
        probe.assert_not_called()

    def test_redirect_to_internal_control_API_is_not_followed(self):
        self.assertIsNone(stage._NoHealthRedirect().redirect_request(
            None, None, 302, 'Found', {}, 'http://127.0.0.1:30984/server_info'))

    def test_invalid_timeout_rejected(self):
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.wait(timeout)

    def test_actual_health_handler_with_generation_disabled_has_no_internal_RPC(self):
        # Execute only the repository's actual handler AST with CPU fake state;
        # avoid importing the full HTTP server or starting any model process.
        repo = Path(__file__).resolve().parents[2]
        tree = ast.parse((repo / 'python/sglang/srt/entrypoints/http_server.py').read_text())
        handler = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'health_generate')
        handler.decorator_list = []
        tokenizer = NS(gracefully_exit=False, server_status='Up',
                       get_internal_state=Mock(side_effect=AssertionError('internal state RPC forbidden')),
                       generate_request=Mock(side_effect=AssertionError('generation forbidden')))
        namespace = dict(Request=object, Response=lambda **kw: NS(**kw),
                         ServerStatus=NS(Starting='Starting'), _global_state=NS(tokenizer_manager=tokenizer),
                         envs=NS(SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=NS(get=lambda: False)))
        exec(compile(ast.Module(body=[handler], type_ignores=[]), '<actual-health-handler>', 'exec'), namespace)
        result = asyncio.run(namespace['health_generate'](NS(url=NS(path='/health'))))
        self.assertEqual(result.status_code, 200)
        tokenizer.get_internal_state.assert_not_called()
        tokenizer.generate_request.assert_not_called()
        # Both E capture and target launch reuse this unchanged Stage A launcher.
        launcher = ast.parse((repo / 'experiments/kv_ec/verify_gpu_restore.py').read_text())
        self.assertTrue(any(isinstance(node, ast.keyword) and node.arg == 'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION'
                            and isinstance(node.value, ast.Constant) and node.value.value == '0'
                            for node in ast.walk(launcher)))


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
            report.update(mock_gpu_policy())
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

    def run_trial(self, fail=None, selected='strict', recovery='two-level'):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local = base / 'trial-mock-only'
            local.mkdir()
            cfg = bench.read(EXAMPLE)
            cfg['remote_root'] = str(base / 'remote')
            cfg['gpu_admission'] = {'mode': selected}
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
                elif action == 'inference':
                    bench.save(option(args, 'output') / 'request-000.json', dict(output_token_ids=[7],
                        ready_to_first_token_ms=1, ready_to_decode_complete_ms=2,
                        end_ns=time.monotonic_ns(), request_ms=1, prefill_ttft_ms=1))
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
                        self_outer.assertEqual(tail[tail.index('--gpu-admission-mode') + 1], selected)
                        bench.save(option(tail, 'output') / 'run.json', dict(run_id='mock-origin', **mock_gpu_policy(selected)))
                        bench.save(option(tail, 'output') / 'rank_0.json', dict(output_token_ids=[7],
                            runtime_identity=dict(model_sha256='mock-model')))
                        bench.save(option(tail, 'output') / 'submitted_input_ids.json', [1] * 256)
                    elif self.name == 'target':
                        if recovery == 'two-level':
                            self_outer.assertTrue(option(tail, 'ledger').is_dir())
                            self_outer.assertEqual(list(option(tail, 'ledger').iterdir()), [])
                        self_outer.assertEqual(tail[tail.index('--gpu-admission-mode') + 1], selected)
                        bench.save(option(tail, 'output') / 'run.json', dict(run_id='mock-target', **mock_gpu_policy(selected)))
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
            env = dict(KV_EC_BENCH_CONFIG=str(config), KV_EC_BENCH_TRIAL=str(local),
                       KV_EC_PERF_RUN_ID=local.name, KV_EC_PERF_RECOVERY_MODE=recovery)
            with patch.dict(os.environ, env), patch.object(stage.Host, 'run', run), \
                    patch.object(stage.Host, 'argv', argv), patch.object(stage, 'scp', scp), \
                    patch.object(stage.subprocess, 'Popen', Proc), patch.object(stage.signal, 'signal'), \
                    patch.object(stage, 'identity', return_value=dict(commit_sha='mock-only')), \
                    patch.object(stage, 'target_binding', return_value=dict(target_run_id='mock-target')), \
                    patch.object(stage, 'native_metrics'), patch.object(stage, 'local_ssh_tunnel'), \
                    patch.object(pf, 'free_ports'), patch('experiments.kv_ec.rs_recovery_source.native_stats'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'MOCK_FAILURE'):
                        stage.trial()
                    self.assertTrue((local / 'failure.json').exists())
                    self.assertFalse((local / 'acceptance.json').exists())
                else:
                    stage.trial()
                    if recovery == 'two-level':
                        self.assertEqual(bench.read(local / 'acceptance.json')['run_id'], local.name)
                    else:
                        self.assertTrue(bench.read(local / 'reprefill.json')['pass_reprefill'])
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

    def test_shared_coordinator_propagates_to_origins_and_restore(self):
        self.run_trial(selected='shared')

    def test_shared_coordinator_propagates_to_cold_reprefill(self):
        self.run_trial(selected='shared', recovery='reprefill')


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
    def exercise(self, count=1, driver_code=0, gate_error=False, timeout=False, selected='strict'):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = bench.read(EXAMPLE)
            cfg['nodes'][0]['repo'] = str(Path.cwd())
            cfg['gpu_admission'] = {'mode': selected}
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
                    for name in ('node0-reference/run.json', 'origin-run-233.json', 'target-run.json'):
                        bench.save(trial / name, mock_gpu_policy(selected))
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

    def test_shared_functional_pass_not_isolated_performance(self):
        records = self.exercise(selected='shared')
        self.assertTrue(records[0]['functional_correctness_pass'])
        self.assertFalse(records[0]['isolated_performance_eligible'])
        self.assertEqual(records[0]['gpu_admission_mode'], 'shared')
        failed = self.exercise(selected='shared', driver_code=1)
        self.assertFalse(failed[0]['functional_correctness_pass'])

    def test_driver_failure_stops_batch_and_preserves_trial(self):
        records = self.exercise(count=5, driver_code=1)
        self.assertEqual(records[0]['status'], 'failed')

    def test_timeout_signals_verified_owned_group_and_records_failure(self):
        records = self.exercise(timeout=True)
        self.assertIn('TimeoutExpired', records[0]['failure_reason'])


class CollectionTests(unittest.TestCase):
    def test_shared_and_strict_summaries_never_pool_and_false_eligibility_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for selected in ('strict', 'shared'):
                root = Path(tmp) / selected
                roots.append(root)
                cfg = dict(gpu_admission={'mode': selected}, mem_fraction_static=.6)
                bench.save(root / 'metadata.json', dict(scheme='reprefill', config=cfg,
                    gpu_admission_mode=selected))
                directory = root / ('trial-' + selected)
                control = dict(pass_reprefill=True, output_token_ids_match=True)
                for metric in ('fault_boundary_to_ready_ms', 'ready_to_first_token_ms',
                               'ready_to_decode_complete_ms', 'fault_boundary_to_decode_complete_ms',
                               'client_request_ms', 'client_prefill_ttft_ms'):
                    control[metric] = 1 if selected == 'strict' else 100
                bench.save(directory / 'reprefill.json', control)
                bench.save(directory / 'trial.json', dict(status='accepted', pass_fail='PASS',
                    functional_correctness_pass=True, isolated_performance_eligible=selected == 'strict',
                    gpu_admission_mode=selected, run_id=directory.name,
                    verdict_sha256=bench.digest(directory / 'reprefill.json')))
                for name in ('node0-reference/run.json', 'origin-run-233.json', 'reprefill-target-run.json'):
                    bench.save(directory / name, mock_gpu_policy(selected))
            output = Path(tmp) / 'summary.json'
            bench.collect(NS(roots=roots, output=output))
            report = bench.read(output)
            self.assertEqual(len(report['accepted_trials']), 2)
            self.assertEqual(report['gpu_admission_modes'], ['shared', 'strict'])
            self.assertTrue(all(r['n'] == 1 for r in report['rows']))
            self.assertTrue(all(not r['isolated_performance_eligible'] for r in report['rows'] if r['gpu_admission_mode'] == 'shared'))
            bad = roots[1] / 'trial-shared/trial.json'
            value = bench.read(bad)
            value['isolated_performance_eligible'] = True
            bad.write_text(json.dumps(value))
            other = Path(tmp) / 'summary-rejected.json'
            bench.collect(NS(roots=roots, output=other))
            self.assertEqual(len(bench.read(other)['accepted_trials']), 1)

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


class RePrefillInterfaceTests(unittest.TestCase):
    def test_reprefill_contract_rejects_wrong_prefix_and_short_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = Path(tmp) / 'ids.json'
            ids.write_text(json.dumps([1] * 256))
            self.assertEqual(len(reprefill_server.read_ids(ids)), 256)
            ids.write_text(json.dumps([1] * 255))
            with self.assertRaisesRegex(ValueError, 'exactly_256_input_ids_required'):
                reprefill_server.read_ids(ids)
        valid = dict(meta_info=dict(cached_tokens=0, prompt_tokens=256, completion_tokens=17),
                     output_ids=list(range(17)))
        self.assertEqual(reprefill_server.validate_generation(valid), list(range(17)))
        valid['meta_info']['cached_tokens'] = 1
        with self.assertRaisesRegex(ValueError, 'reprefill_generation_boundary_invalid'):
            reprefill_server.validate_generation(valid)

    def test_inference_records_coordinator_ready_to_first_and_complete(self):
        import urllib.request
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids_path = root / 'ids.json'
            ids_path.write_text(json.dumps([2] * 256))
            output = root / 'run'
            result = dict(meta_info=dict(cached_tokens=0, prompt_tokens=256,
                                         completion_tokens=17), output_ids=[3] * 17)
            class Response:
                def __enter__(self): return self
                def __exit__(self, *_): return False
                def __iter__(self):
                    yield b'data: ' + json.dumps(result).encode() + b'\n'
                    yield b'data: [DONE]\n'
            info = dict(tp_size=2, disable_radix_cache=True, disable_overlap_schedule=True,
                chunked_prefill_size=-1, max_running_requests=1, attention_backend='triton',
                cuda_graph_config={k: {'backend': 'disabled'} for k in ('decode', 'prefill')},
                model_path='/model')
            args = NS(input_ids=ids_path, url='http://127.0.0.1:30000', model='/model',
                scheme='reprefill', output=output, repeat=1, warmup=0, timeout=10,
                resource_interval=1., expected_model_sha256=None,
                model_identity_evidence=None, ready_monotonic_ns=time.monotonic_ns()-10_000_000)
            with patch.object(bench, 'http', return_value=info), \
                    patch.object(urllib.request, 'urlopen', return_value=Response()):
                bench.inference(args)
            record = bench.read(output / 'request-000.json')
            self.assertEqual(record['output_token_ids'], [3] * 17)
            self.assertGreater(record['ready_to_first_token_ms'], 0)
            self.assertGreaterEqual(record['ready_to_decode_complete_ms'], record['ready_to_first_token_ms'])


if __name__ == '__main__':
    unittest.main()
