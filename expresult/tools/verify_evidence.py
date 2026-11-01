#!/usr/bin/env python3
"""Verify this portable evidence bundle offline; does not import experiment code.

Run from anywhere: python expresult/tools/verify_evidence.py
SHA256SUMS is accidental-corruption detection, not a signed attestation.
"""
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = root / 'run-20261101.Lqol9F7L'
    manifest = read(bundle / 'COPIED_MANIFEST.json')
    for item in manifest:
        path = bundle / item['bundle_relative']
        assert path.stat().st_size == item['bytes'], path
        assert digest(path) == item['sha256'], path
    source = read(bundle / 'analysis/source_snapshot_identity.json')
    snapshots = 0
    for item in source['files']:
        if item['matches_historical']:
            assert digest(bundle / item['bundle_relative']) == item['historical_sha256'], item['path']
            snapshots += 1
    accepted = 0
    for batch in ('two-level-stable', 'cold-reprefill'):
        base = bundle / 'evidence' / batch
        metadata = read(base / 'metadata.json')
        for trial in sorted(base.glob('trial-*')):
            record = read(trial / 'trial.json')
            assert record['status'] == 'accepted' and record['pass_fail'] == 'PASS'
            assert read(trial / 'cleanup.json')['status'] == 'PASS'
            verdict_name = 'stage-e-check.json' if batch == 'two-level-stable' else 'reprefill.json'
            assert digest(trial / verdict_name) == record['verdict_sha256']
            nodes = [read(trial / ('inventory-' + node + '.json')) for node in ('232', '233')]
            assert nodes[0]['host'] != nodes[1]['host']
            for key in ('commit_sha', 'tracked_diff_sha256', 'benchmark_files'):
                assert nodes[0][key] == nodes[1][key] == metadata[key], key
            if batch == 'two-level-stable':
                verdict = read(trial / verdict_name)
                for key in ('pass_stage_e', 'real_rdma', 'gpu_restore_committed', 'new_instance',
                            'no_slot_leak', 'no_request_row_leak', 'output_token_ids_match'):
                    assert verdict[key] is True, key
                assert verdict['resume_decode_forward_count'] == 16
                assert verdict['recomputed_prefix_tokens'] == 0
                assert len(verdict['rank_correctness']) == 2
                for rank in verdict['rank_correctness']:
                    assert all(rank[k] is True for k in ('byte_exact', 'scheduler_state_consistent',
                                                         'output_token_ids_match'))
                acceptance = read(trial / 'acceptance.json')
                for key, filename in (('verdict_sha256', verdict_name), ('target_run_sha256', 'target-run.json'),
                                      ('protection_sha256', 'protection.json')):
                    assert acceptance[key] == digest(trial / filename), key
                cq = read(trial / 'cq-evidence.json')
                legs = cq['uploads'] + [cq['recovery'][k] for k in ('client_send_stats', 'receive_stats',
                    'send_stats', 'client_receive_stats')] + [cq['delivery_send']['send_stats'],
                    cq['delivery_receive']['receive_stats']]
                assert len(legs) == 8
                assert all(x['transport'] == 'rdma' and x['completion_status'] == 'SUCCESS' for x in legs)
                status = read(trial / 'cold-comparison-timing-status.json')
                assert status['performance_measurement_eligible'] is False
                assert not (trial / 'cold-comparison-timing.json').exists()
            else:
                request = read(trial / 'reprefill-inference/request-000.json')
                reference = read(trial / 'node1-reference/rank_0.json')
                assert request['output_token_ids'] == reference['output_token_ids']
                usage = request['response']['meta_info']
                assert (usage['cached_tokens'], usage['prompt_tokens'], usage['completion_tokens']) == (0, 256, 17)
                timing = read(trial / 'cold-comparison-timing.json')
                for metric, (start, end) in timing['intervals'].items():
                    assert timing[metric + '_ms'] == (timing['events_ns'][end] - timing['events_ns'][start]) / 1e6
                assert timing['algorithm_comparison_eligible'] is False
            accepted += 1
    assert accepted == 4
    audit = read(bundle / 'analysis/offline_audit.json')
    assert len(audit) == 4 and all(x['all_checks_pass'] for x in audit)
    json_count = csv_count = 0
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        assert path.suffix not in ('.bin', '.npy', '.zip'), path
        if path.suffix == '.json':
            read(path)
            json_count += 1
        if path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                json.loads(line)
        if path.suffix == '.csv':
            with path.open(newline='') as stream:
                list(csv.DictReader(stream))
            csv_count += 1
    sums = root / 'SHA256SUMS'
    checksum_count = 0
    expected_paths = set()
    for line in sums.read_text().splitlines():
        checksum, relative = line.split('  ', 1)
        path = root / relative
        assert path.resolve().is_relative_to(root.resolve()), relative
        assert digest(path) == checksum, relative
        expected_paths.add(relative)
        checksum_count += 1
    actual_paths = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                    and p != sums and '__pycache__' not in p.parts}
    assert expected_paths == actual_paths, 'SHA256SUMS file coverage mismatch'
    print(json.dumps(dict(status='PASS', scope='OFFLINE artifact integrity and stored acceptance only; NOT live GPU regression',
        accepted_trials=accepted, verbatim_files=len(manifest), source_snapshots=snapshots,
        json_files=json_count, csv_files=csv_count, checksummed_files=checksum_count), indent=2))


if __name__ == '__main__':
    main()
