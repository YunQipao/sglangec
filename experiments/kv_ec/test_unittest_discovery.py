"""CPU-only import regression: discover tests without executing their bodies."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DiscoveryImportTests(unittest.TestCase):
    def check_imports(self, top_level):
        # A fresh interpreter avoids cached modules masking import-path defects.
        # Only load the suite: never run hardware acceptance or test bodies here.
        script = """
import sys
import unittest

loader = unittest.TestLoader()
suite = loader.discover('experiments/kv_ec', pattern='test_*.py',
                        top_level_dir=sys.argv[1] or None)
if loader.errors:
    raise SystemExit('\\n'.join(loader.errors))

def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item.id()

ids = list(cases(suite))
for name in ('test_gpu_cold_resume', 'test_gpu_kv_restore',
             'test_gpu_restore_hook', 'test_gpu_resume', 'test_online_chunk',
             'test_online_rs', 'test_perf_stage_e', 'test_rdma_upload_control',
             'test_rs_cold_resume', 'test_rs_recovery_source', 'test_tp4_stage_e',
             'test_kv_ec', 'test_shared_gpu_debug'):
    if not any(name in test_id.split('.') for test_id in ids):
        raise SystemExit('missing discovered module: ' + name)
print('Discovered', len(ids), 'tests without import errors')
"""
        env = dict(os.environ)
        # Match the reported command; no experiment-directory path workaround.
        env['PYTHONPATH'] = str(ROOT)
        result = subprocess.run(
            [sys.executable, '-c', script, top_level], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('without import errors', result.stdout)

    def test_default_discovery_imports_all_modules(self):
        self.check_imports('')

    def test_package_discovery_imports_all_modules(self):
        self.check_imports('.')


if __name__ == '__main__':
    unittest.main()
