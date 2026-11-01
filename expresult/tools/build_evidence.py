#!/usr/bin/env python3
"""Read-only source export into a NEW directory; never launches experiment code.

Creates text/JSON/CSV evidence, not a full replay bundle. Source data is never
modified. Existing destination files are refused. Execute from any directory.
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from datetime import datetime, timezone


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        out = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        out.writeheader()
        out.writerows(rows)


def run(source, destination, repo):
    if destination.exists():
        raise ValueError('Refusing existing destination: ' + str(destination))
    if not (source / 'two-level-stable/repeat-summary.json').is_file():
        raise ValueError('Expected source batch not found')
    destination.mkdir(parents=True)
    copied = []
    def copy(path):
        if not path.is_file():
            return
        relative = path.relative_to(source)
        target = destination / 'evidence' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return
        shutil.copyfile(path, target)
        copied.append(dict(source_relative=str(relative), bundle_relative=str(target.relative_to(destination)),
                           bytes=path.stat().st_size, sha256=sha(path)))

    for name in ('config.json', 'tests.log', 'tests-fixed.log'):
        copy(source / name)
    accepted_batches = ('two-level-stable', 'cold-reprefill')
    trial_files = ('trial.json', 'stage-e-check.json', 'acceptance.json', 'cleanup.json',
        'stage-e-timing.json', 'cold-comparison-timing.json', 'cold-comparison-timing-status.json',
        'native-metrics.json', 'cq-evidence.json', 'reprefill.json', 'inventory-232.json',
        'inventory-233.json', 'port-release.json',
        'port-check.json', 'rs-after-cleanup.json', 'target-run.json', 'origin-run-233.json',
        'reprefill-target-run.json', 'protection.json', 'export1.json', 'delivery-send.json',
        'delivery-receive.json', 'commands.jsonl', 'node0-reference/run.json',
        'export0/export.json', 'recovered-source/protected/recovery.json',
        'reprefill-inference/request-000.json', 'reprefill-inference/metadata.json',
        'node1-reference/submitted_input_ids.json', 'node1-reference/rank_0.json')
    for batch in accepted_batches:
        base = source / batch
        for name in ('config.json', 'metadata.json', 'repeat-summary.json'):
            copy(base / name)
        for trial in sorted(base.glob('trial-*')):
            for name in trial_files:
                copy(trial / name)
            for path in trial.glob('perf*/*.jsonl'):
                copy(path)
    for batch in ('xor-rank-0', 'xor-rank-1'):
        for path in (source / batch).glob('*.json'):
            copy(path)
    for batch in ('two-level-smoke', 'two-level-fixed', 'two-level-v2', 'two-level-final'):
        base = source / batch
        copy(base / 'repeat-summary.json')
        for trial in sorted(base.glob('trial-*')):
            for name in ('trial.json', 'failure.json', 'cleanup.json', 'stage-e-timing.json',
                         'cold-comparison-timing-status.json', 'acceptance.json'):
                copy(trial / name)
        errors = []
        candidates = [source / (batch + '.log'), *base.glob('trial-*/driver.log')]
        for path in candidates:
            if not path.exists():
                continue
            lines = path.read_text(errors='replace').splitlines()
            indices = set()
            needles = ('Traceback', 'Error:', 'AssertionError', 'FAILED', 'SOURCE_IDENTITY_MISMATCH',
                       'trial_source_changed_since_batch_admission', 'FOREIGN_OR_UNVERIFIABLE_LISTENER')
            for i, line in enumerate(lines):
                if any(n in line for n in needles):
                    indices.update(range(max(0, i - 2), min(len(lines), i + 5)))
            errors.append(dict(source_relative=str(path.relative_to(source)), source_sha256=sha(path),
                               total_lines=len(lines), lines=[dict(line=i + 1, text=lines[i][:6000])
                                                              for i in sorted(indices)][-100:]))
        write_json(destination / 'analysis/failures' / (batch + '.json'), errors)

    preflights = []
    for path in [source / 'preflight.json', *source.glob('*/preflight.json')]:
        if not path.exists():
            continue
        value = read(path)
        preflights.append(dict(source_relative=str(path.relative_to(source)), sha256=sha(path),
                               status=value.get('status'), error=value.get('error')))
    write_json(destination / 'analysis/preflight_catalog.json', preflights)
    catalog, timings, audit, resource, checkpoint, perf = [], [], [], [], [], []
    series = collections.defaultdict(list)
    observations = collections.defaultdict(list)
    for base in sorted(p for p in source.iterdir() if p.is_dir()):
        metadata = read(base / 'metadata.json') if (base / 'metadata.json').exists() else {}
        for trial in sorted(base.glob('trial-*')):
            if not (trial / 'trial.json').exists():
                continue
            t = read(trial / 'trial.json')
            scheme, rid = t.get('scheme', metadata.get('scheme')), trial.name
            accepted = t.get('status') == 'accepted' and t.get('pass_fail') == 'PASS'
            verdict = read(trial / 'stage-e-check.json') if (trial / 'stage-e-check.json').exists() else {}
            catalog.append(dict(batch=base.name, run_id=rid, scheme=scheme, accepted=accepted,
                pass_fail=t.get('pass_fail'), functional_correctness_pass=t.get('functional_correctness_pass'),
                stage_e_subcheck_pass=verdict.get('pass_stage_e'),
                isolated_performance_eligible=t.get('isolated_performance_eligible'),
                trial_command_ms=t.get('trial_command_ms'), failure_reason=t.get('failure_reason'),
                evidence_relative=str(trial.relative_to(source))))
            if not accepted or base.name not in accepted_batches:
                continue
            result = dict(batch=base.name, run_id=rid, scheme=scheme,
                          trial_command_ms=t['trial_command_ms'], timing_qualification=False)
            timing = read(trial / 'stage-e-timing.json') if (trial / 'stage-e-timing.json').exists() else {}
            result.update({k: v for k, v in timing.items() if k.endswith('_ms')})
            comparison = read(trial / 'cold-comparison-timing.json') if (trial / 'cold-comparison-timing.json').exists() else {}
            result.update({k: v for k, v in comparison.items() if k.endswith('_ms')})
            result['timing_qualification'] = comparison.get('performance_measurement_eligible', False)
            result['algorithm_comparison_eligible'] = comparison.get('algorithm_comparison_eligible', False)
            per_trial = collections.defaultdict(float)
            for path in sorted(trial.glob('perf*/*.jsonl')):
                for line in path.read_text().splitlines():
                    x = json.loads(line)
                    row = dict(batch=base.name, run_id=rid, source_relative=str(path.relative_to(source)),
                        stage=x['stage'], role=x.get('role'), host=x.get('host'), pid=x.get('pid'),
                        rank=x.get('tp_rank'), source=x.get('source'), clock_domain=x.get('clock_domain'),
                        operation_id=x.get('operation_id'), span_id=x.get('span_id'),
                        call_index=x.get('call_index'), start_ns=x.get('start_ns'), end_ns=x.get('end_ns'),
                        duration_ms=x['duration_ms'], status=x.get('status'))
                    perf.append(row)
                    if row['stage'] == 'checkpoint_load_validate_inclusive':
                        checkpoint.append(row)
                    if x.get('status') == 'ok':
                        key = (scheme, x['stage'], x.get('role', ''), x.get('tp_rank', ''))
                        per_trial[key] += x['duration_ms']
                        observations[key].append(x['duration_ms'])
                    if x['stage'] == 'new_instance_launch_to_ready_harness':
                        result['target_launch_to_ready_harness_ms'] = x['duration_ms']
            for key, value in per_trial.items():
                series[key].append(value)
            if (trial / 'native-metrics.json').exists():
                for metric, value in read(trial / 'native-metrics.json').items():
                    series[(scheme, metric, 'native-metrics', '')].append(value)
            timings.append(result)
            verdict_name = 'stage-e-check.json' if scheme == 'two-level' else 'reprefill.json'
            checks = dict(verdict_digest=sha(trial / verdict_name) == t['verdict_sha256'],
                          cleanup_pass=read(trial / 'cleanup.json')['status'] == 'PASS')
            inventories = [read(trial / ('inventory-' + node + '.json')) for node in ('232', '233')]
            identity_keys = ('commit_sha', 'tracked_diff_sha256', 'benchmark_files')
            checks['distinct_hosts'] = inventories[0]['host'] != inventories[1]['host']
            checks['node_source_identity_equal'] = all(inventories[0][k] == inventories[1][k] for k in identity_keys)
            checks['batch_source_identity_equal'] = all(inventories[0][k] == metadata[k] for k in identity_keys)
            detail = dict(batch=base.name, run_id=rid, checks=checks,
                          scope='offline stored evidence only; no live server query or GPU rerun')
            if scheme == 'two-level':
                acceptance = read(trial / 'acceptance.json')
                checks['acceptance_hashes'] = all(acceptance[k] == sha(trial / name) for k, name in (
                    ('verdict_sha256', 'stage-e-check.json'), ('target_run_sha256', 'target-run.json'),
                    ('protection_sha256', 'protection.json')))
                q = read(trial / 'cq-evidence.json')
                legs = q['uploads'] + [q['recovery'][k] for k in
                    ('client_send_stats', 'receive_stats', 'send_stats', 'client_receive_stats')] + [
                    q['delivery_send']['send_stats'], q['delivery_receive']['receive_stats']]
                checks['eight_successful_rdma_legs'] = len(legs) == 8 and all(
                    x.get('transport') == 'rdma' and x.get('completion_status') == 'SUCCESS' for x in legs)
                checks['rank_byte_scheduler_token_checks'] = len(verdict['rank_correctness']) == 2 and all(
                    all(v.get(k) is True for k in ('byte_exact', 'scheduler_state_consistent', 'output_token_ids_match'))
                    for v in verdict['rank_correctness'])
                checks['restore_invariants'] = all(verdict.get(k) is True for k in
                    ('pass_stage_e', 'real_rdma', 'gpu_restore_committed', 'new_instance',
                     'no_slot_leak', 'no_request_row_leak', 'output_token_ids_match')) and (
                    verdict['resume_decode_forward_count'] == 16 and verdict['recomputed_prefix_tokens'] == 0)
                detail['rank_correctness'] = verdict['rank_correctness']
                detail['rdma_legs'] = [{k: v.get(k) for k in ('transport', 'completion_status', 'bytes',
                    'rdma_device', 'gid_index', 'port_num')} for v in legs]
            else:
                request = read(trial / 'reprefill-inference/request-000.json')
                reference = read(trial / 'node1-reference/rank_0.json')
                checks['output_ids_equal_reference'] = request['output_token_ids'] == reference['output_token_ids']
                usage = request['response']['meta_info']
                checks['uncached_256_input_17_output'] = usage['cached_tokens'] == 0 and usage['prompt_tokens'] == 256 and usage['completion_tokens'] == 17
                if comparison:
                    checks['comparison_intervals_match_events'] = all(comparison[name + '_ms'] ==
                        (comparison['events_ns'][end] - comparison['events_ns'][start]) / 1e6
                        for name, (start, end) in comparison['intervals'].items())
                detail['output_token_ids'] = request['output_token_ids']
            detail['all_checks_pass'] = all(checks.values())
            audit.append(detail)
            for name in ('node0-reference/run.json', 'origin-run-233.json', 'target-run.json', 'reprefill-target-run.json'):
                path = trial / name
                if path.exists():
                    cfg = read(path)
                    for gpu in cfg.get('gpu_resources_before_start', []):
                        resource.append(dict(batch=base.name, run_id=rid, startup=name,
                            gpu_index=gpu.get('gpu_index'), gpu_uuid=gpu.get('gpu_uuid'),
                            free_memory_bytes=gpu.get('free_memory_bytes'), used_memory_bytes=gpu.get('used_memory_bytes'),
                            other_tasks=gpu.get('other_tasks'), scope='startup snapshot, not continuous process utilization'))

    write_json(destination / 'analysis/trial_catalog.json', catalog)
    write_json(destination / 'analysis/timing_trials.json', timings)
    write_json(destination / 'analysis/offline_audit.json', audit)
    write_json(destination / 'analysis/gpu_startup_context.json', resource)
    write_csv(destination / 'analysis/trial_catalog.csv', catalog, list(catalog[0]))
    write_csv(destination / 'analysis/timing_trials.csv', timings, sorted(set().union(*(x.keys() for x in timings))))
    cols = list(perf[0])
    write_csv(destination / 'analysis/perf_events.csv', perf, cols)
    write_csv(destination / 'analysis/checkpoint_calls.csv', checkpoint, cols)
    checkpoint_summary = []
    groups = collections.defaultdict(list)
    for row in checkpoint:
        groups[(row['batch'], row['run_id'], row['role'], row['host'])].append(row['duration_ms'])
    for (batch, rid, role, host), values in groups.items():
        checkpoint_summary.append(dict(batch=batch, run_id=rid, role=role, host=host, calls=len(values),
            cumulative_ms=sum(values), mean_call_ms=statistics.mean(values), min_call_ms=min(values), max_call_ms=max(values)))
    write_json(destination / 'analysis/checkpoint_summary.json', checkpoint_summary)
    stat_rows = []
    for (scheme, metric, role, rank), values in sorted(series.items(), key=lambda x: str(x[0])):
        stat_rows.append(dict(scheme=scheme, metric=metric, role=role, rank=rank,
            n_trials=len(values), mean_ms=statistics.mean(values),
            stddev_ms=statistics.stdev(values) if len(values) > 1 else None,
            median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
            aggregation='per-trial sum of matching spans; nested spans must not be added',
            call_observations=len(observations[(scheme, metric, role, rank)])))
    write_csv(destination / 'analysis/metric_statistics.csv', stat_rows, list(stat_rows[0]))
    write_json(destination / 'analysis/metric_statistics.json', stat_rows)
    xor = []
    for batch in ('xor-rank-0', 'xor-rank-1'):
        rows = [read(p) for p in sorted((source / batch).glob('xor-*.json'))]
        for metric in ('xor_ms', 'recovery_ms', 'validation_ms'):
            values = [x[metric] for x in rows]
            xor.append(dict(batch=batch, metric=metric, n=len(values), all_byte_exact=all(x['byte_exact'] for x in rows),
                mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                stddev_ms=statistics.stdev(values), min_ms=min(values), max_ms=max(values),
                scope='CPU codec on one authentic GPU Capture; not independent GPU/node recovery trials'))
    write_json(destination / 'analysis/xor_statistics.json', xor)

    # A compact source snapshot is included only when its bytes match the
    # historical successful batch identity; current mismatches are not relabeled.
    metadata = read(source / 'two-level-stable/metadata.json')
    selected = ('perf_observer.py', 'perf_benchmark.py', 'perf_stage_e.py', 'perf_preflight.py',
        'cold_timing.py', 'gpu_cold_resume_hook.py', 'gpu_restore_hook.py', 'gpu_kv_restore.py',
        'gpu_resume_hook.py', 'cold_resume_state.py', 'verify_gpu_cold_resume.py',
        'verify_gpu_restore.py', 'verify_rs_cold_resume.py', 'rs_cold_source.py',
        'reprefill_server.py', 'remote_rs_client.py', 'remote_rs_server.py', 'rs_codec.py',
        'online_chunk_manager.py', 'multinode/rdma_transport.py', 'multinode/verbs_helper.c',
        'test_perf_stage_e.py')
    snapshot = []
    for name in selected:
        path = repo / 'experiments/kv_ec' / name
        expected = metadata['benchmark_files'].get(name)
        actual = sha(path) if path.exists() else None
        entry = dict(path='experiments/kv_ec/' + name, historical_sha256=expected, current_sha256=actual,
                     matches_historical=actual is not None and actual == expected)
        if entry['matches_historical']:
            # Text suffix prevents archived test modules being collected by
            # pytest when this evidence directory is added to the repository.
            target = destination / 'source_snapshot/experiments/kv_ec' / (name + '.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            entry['bundle_relative'] = str(target.relative_to(destination))
        snapshot.append(entry)
    write_json(destination / 'analysis/source_snapshot_identity.json', dict(
        commit_sha=metadata['commit_sha'], tracked_diff_sha256=metadata['tracked_diff_sha256'],
        files=snapshot, limitation='selected verified files only, not complete runtime/model/environment'))

    inventory = []
    selected_paths = {x['source_relative'] for x in copied}
    for path in sorted(p for p in source.rglob('*') if p.is_file()):
        relative = str(path.relative_to(source))
        reason = ('copied verbatim' if relative in selected_paths else
                  'large KV payload intentionally omitted' if path.suffix == '.bin' else
                  'raw logits intentionally omitted' if path.suffix == '.npy' else
                  'archive duplicates selected evidence' if path.suffix == '.zip' else
                  'resource sampling summarized by startup context; raw samples omitted' if 'resources' in path.name else
                  'redundant/debug/control artifact omitted; relevant failure text extracted where applicable')
        inventory.append(dict(source_relative=relative, bytes=path.stat().st_size,
                              sha256=sha(path), copied=relative in selected_paths, disposition=reason))
    write_csv(destination / 'SOURCE_INVENTORY.csv', inventory, list(inventory[0]))
    write_json(destination / 'COPIED_MANIFEST.json', copied)
    # No claims about newly rerunning GPU/RS acceptance; these are offline checks.
    summary = dict(schema='kv-ec-lightweight-export-v1', source_root=str(source),
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_file_count=len(inventory), source_bytes=sum(x['bytes'] for x in inventory),
        copied_file_count=len(copied), copied_bytes=sum(x['bytes'] for x in copied),
        source_unmodified=True, new_gpu_or_rdma_runs=False, git_commit_or_push=False,
        all_four_offline_evidence_audits_pass=len(audit) == 4 and all(x['all_checks_pass'] for x in audit),
        approved_for_algorithm_speedup_claim=False,
        privacy='Exact evidence includes internal hosts/IPs/user paths/process IDs/token IDs; not redacted. Review before public upload.',
        limitations=['No original KV .bin, logits .npy, or zip archives',
          'No full runtime/model snapshot; selected source hashes match historical metadata',
          'Three successful Two-level trials and one Re-Prefill trial, not matched five-round comparison',
          'Two-level has no equivalent client SSE timing; algorithm speedup cannot be calculated',
          'Shared GPUs; startup process snapshots do not quantify causal contention'])
    write_json(destination / 'EXPORT_INFO.json', summary)
    if not summary['all_four_offline_evidence_audits_pass']:
        raise ValueError('Offline evidence audit failed; inspect analysis/offline_audit.json')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.source.resolve(), args.destination.resolve(), args.repo.resolve())
