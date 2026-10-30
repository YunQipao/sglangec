"""Opt-in TWO-HOST CUDA/native-RDMA acceptance; no mocks or CPU fallback.

Run on node0 after CPU regression and dedicated RS startup. Set
KV_EC_TP4_LIVE_CONFIG to an absolute, site-verified TP4 configuration. Every
test creates fresh captures/stripes/ledgers under its own retained directory.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('KV_EC_TP4_LIVE_CONFIG'), 'explicit two-host GPU opt-in required')
class TP4LiveAcceptance(unittest.TestCase):
    def run_trial(self, scheme):
        config = Path(os.environ['KV_EC_TP4_LIVE_CONFIG']).resolve()
        cfg = json.loads(config.read_text())
        self.assertEqual(cfg['tp_size'], 4)
        self.assertEqual(cfg['gpu_admission']['mode'], 'shared')
        parent = Path(tempfile.mkdtemp(prefix='kv-tp4-live-'))
        output = parent / scheme
        print(f'Live evidence retained: {output}', flush=True)
        subprocess.run([sys.executable, '-m', 'experiments.kv_ec.perf_benchmark',
            'repeat', '--config', str(config), '--output', str(output), '--repeat', '1',
            '--recovery-mode', scheme], check=True, cwd=cfg['nodes'][0]['repo'])
        files = list(output.glob('trial-*/trial.json'))
        self.assertEqual(len(files), 1)
        trial = json.loads(files[0].read_text())
        self.assertTrue(trial['functional_correctness_pass'])
        self.assertFalse(trial['isolated_performance_eligible'])
        self.assertEqual(trial['tp_size'], 4)
        return files[0].parent

    def test_new_instance_all_four_ranks_match_original_sixteen_decodes(self):
        root = self.run_trial('two-level')
        result = json.loads((root / 'stage-e-check.json').read_text())
        self.assertTrue(result['pass_stage_e'])
        self.assertTrue(result['real_rdma'])
        self.assertTrue(result['new_instance'])
        self.assertEqual(result['resume_decode_forward_count'], 16)
        self.assertEqual([r['tp_rank'] for r in result['ranks']], list(range(4)))
        self.assertEqual(len(result['comparisons']), 4)
        for comparison in result['comparisons']:
            self.assertTrue(comparison['output_token_ids_match'])
        self.assertTrue(all(r['cuda_executed'] and r['committed'] for r in result['ranks']))

    def test_cold_reprefill_same_model_input_and_original_tokens(self):
        root = self.run_trial('reprefill')
        result = json.loads((root / 'reprefill.json').read_text())
        self.assertTrue(result['pass_reprefill'])
        self.assertTrue(result['output_token_ids_match'])
        self.assertEqual(result['tp_size'], 4)
        inference = json.loads((root / 'reprefill-inference/metadata.json').read_text())
        target = json.loads((root / 'reprefill-target-run.json').read_text())
        self.assertEqual(inference['server_info']['tp_size'], 4)
        self.assertEqual(target['model_identity']['model_sha256'], result['model_sha256'])
        self.assertEqual(inference['input_sha256'], result['input_sha256'])


if __name__ == '__main__':
    unittest.main()
