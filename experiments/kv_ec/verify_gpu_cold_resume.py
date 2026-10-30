"""Stage D capture+reference, fresh-instance launch, submit and independent check.

capture runs the normal request to completion and seals its first post-prefill
boundary. Its remaining 16 Decode forwards are the fault-free reference. The
old --template-checkpoint is ONLY a model/layout/input admission template; every
byte of the NEW checkpoint comes from the origin's real completed CUDA forward.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import verify_gpu_restore as stage_a
from .cold_resume_state import (claim_restore, load_checkpoint, process_identity,
    require_new_process, require_origins_stopped, sealed_origin_processes, write_json)
from .gpu_kv_restore import _require
from .gpu_resume_hook import token_digest, validate_decode_observation
from .node_block_export import canonical, sha
from .verify_gpu_resume import (compare_outputs, reference_identity, save_verdict,
    validate_restore_evidence)


def read_evidence(output, mode):
    root = Path(output).resolve()
    config = json.loads((root / 'run.json').read_text())
    _require(config['experiment'] == 'gpu-cold-resume-stage-d' and
             config['cold_mode'] == mode and not list(root.glob('rank_*.failure.json')),
             'cold_run_or_late_failure_invalid')
    source, state, receipt, _ = load_checkpoint(config[
        'capture_bundle' if mode == 'capture' else 'state_checkpoint'], config.get('rs_source'))
    origins = sealed_origin_processes(receipt)
    if mode == 'capture':
        _require(config['origin_server_process'] == receipt['origin_server_process'],
                 'cold_capture_server_process_binding_failed')
    else:
        require_new_process(config['origin_server_process'], origins)
        require_origins_stopped(origins)
    from .gpu_restore_hook import validate_checkpoint
    meta = validate_checkpoint(source.load())
    reports, logits = [], []
    for rank in (0, 1):
        report = json.loads((root / f'rank_{rank}.json').read_text())
        _require(report['schema'] == 'kv-gpu-cold-resume-stage-d-v1' and
                 report['cold_mode'] == mode and report['tp_rank'] == rank and
                 report['tp_size'] == 2 and report['run_id'] == config['run_id'] and
                     report.get('failure_reason') is None and
                     report['runtime_identity'] == config['runtime_identity'] == state['runtime_identity'],
                 'cold_rank_identity_failed')
        _require(report.get('request_id') == state['request_id'],
                 'cold_request_id_binding_failed')
        _require(all(report.get(key) is True for key in ('cuda_executed', 'generation_unchanged',
            'no_slot_leak', 'all_rows_released', 'row_released', 'request_finished')) and
            report['generation_before'] == report['generation_after'] > 0 and
            report['free_slots_after'] == report['allocator_size'] and
            report['free_rows_after'] == report['request_pool_size'], 'cold_cleanup_failed')
        outputs = report['output_token_ids']
        _require(len(outputs) == 17 and outputs[0] == state['pending_token_id'] and
                 all(type(t) is int and 0 <= t < state['vocab_size'] for t in outputs) and
                 token_digest(outputs) == report['output_token_ids_sha256'] and
                 report['resume_decode_forward_count'] == report['resumed_generated_tokens'] == 16 and
                 report['recomputed_prefix_tokens'] == 0 and
                 report['token_ids_sha256'] == state['input_ids_sha256'], 'cold_output_state_invalid')
        if mode == 'capture':
            _require(report['pass_reference'] is True and report['reference_no_restore'] is True and
                     not report['committed'] and not report['restored_slots_used_by_decode'] and
                     report['original_prefill_tokens'] == 256 and
                     report['original_prefill_forward_count'] == 1 and
                     report['capture_state_sha256'] == sha(canonical(state)) and
                     report['captured_kv_sha256'] == sha(source.load()) and
                     report['process'] == receipt['origin_processes'][rank], 'cold_reference_invalid')
            prefix, start = report['old_slots'], 0
        else:
            _require(all(report.get(k) is True for k in ('local_cold_resume_complete',
                'committed', 'verified', 'initial_install', 'new_instance',
                'scheduler_injected', 'restored_slots_used_by_decode')) and
                report['transaction_state'] == 'COMMITTED' and report['old_slots'] == [] and
                report['restored_kv_tokens'] == 256 and report['mapping_version'] == 1 and
                report['original_prefill_tokens'] == report['original_prefill_forward_count'] == 0 and
                report['prefix_prefill_forward_count'] == 0 and
                report['state_sha256'] == config['state_sha256'] == sha(canonical(state)) and
                report['restore_id'] == state['restore_id'] and
                report['restored_request_state'] == state and
                report['checkpoint_sha256'] == sha(source.load()) and
                report['forward_ct_before_restore'] == report['forward_ct_after_restore'] == 0,
                'cold_restore_boundary_or_commit_invalid')
            require_new_process(report['process'], sealed_origin_processes(receipt))
            validate_restore_evidence(report, meta['ranks'][rank], rank)
            prefix, start = report['new_slots'], 1
        _require(len(prefix) == len(set(prefix)) == 256, 'cold_prefix_slots_invalid')
        observations = report['forward_observations']
        _require(len(observations) == len(report['logits']) == 17 - start,
                 'cold_forward_evidence_missing')
        arrays = []
        for step, (obs, logit) in enumerate(zip(observations, report['logits']), start):
            _require(obs['step'] == logit['step'] == step and
                     obs['req_pool_indices'] == [report['row']] and
                     obs['generation'] == report['generation_before'] and
                     obs['forward_ct'] == step + 1 - start and
                     obs['kv_length'] == 256 + step and
                     all(0 < x <= report['allocator_size'] for x in obs['mapping']) and
                     obs['mapping_version'] == (0 if mode == 'capture' else 1),
                     'cold_forward_identity_invalid')
            if step == 0:
                _require(obs['mode'] == 'EXTEND' and obs['positions'] == list(range(256)) and
                         obs['input_ids'] == state['origin_input_ids'] and
                         obs['mapping'] == prefix, 'cold_reference_prefill_invalid')
            else:
                _require(obs['mode'] == 'DECODE', 'cold_prefix_was_prefilled')
                validate_decode_observation(step=step, outputs=outputs[:step],
                    seq_len=obs['kv_length'], committed_len=obs['kv_length'],
                    input_ids=obs['input_ids'], positions=obs['positions'],
                    indices=obs['kv_indices'], mapping=obs['mapping'],
                    prefix_slots=prefix, out_slots=obs['out_cache_loc'])
            name = f'rank_{rank}.logits_{step:03d}.npy'
            _require(logit['file'] == name and sha((root / name).read_bytes()) == logit['sha256'],
                     'cold_logits_hash_mismatch')
            arr = np.load(root / name, allow_pickle=False)
            _require(list(arr.shape) == logit['shape'] == [1, state['vocab_size']] and
                     np.isfinite(arr).all() and int(np.argmax(arr[0])) == outputs[step],
                     'cold_logits_or_sampling_invalid')
            arrays.append(arr)
        events = [json.loads(line) for line in (root / f'rank_{rank}.events.jsonl').read_text().splitlines()]
        _require(all(e['run_id'] == config['run_id'] and e['tp_rank'] == rank
                     and e['failure_reason'] is None for e in events), 'cold_tp_event_identity_invalid')
        if mode == 'restore':
            phases = ['GLOBAL_COLD_ROW_OK']
            for phase in ('VERIFIED', 'MAPPED', 'COMMITTED', 'FINISHED'):
                phases += [phase, 'GLOBAL_' + phase + '_OK']
            phases += ['GLOBAL_COLD_REQUEST_READY_OK', 'GLOBAL_COMMIT']
            for step in range(1, 17):
                phases += [f'GLOBAL_READY_{step}_OK', f'GLOBAL_OBSERVED_{step - 1}_OK',
                    f'GLOBAL_COLD_FORWARD_STATE_{step}_OK',
                    'GLOBAL_RESULT_' + token_digest(outputs[:step + 1]) + '_OK']
            phases += ['GLOBAL_RESTORE_LEDGER_COMMITTED_OK', 'GLOBAL_RESTORE_LEDGER_VISIBLE_OK']
            phases += ['COMPLETE']
            _require([e['phase'] for e in events] == phases, 'cold_tp_gate_evidence_incomplete')
        else:
            _require(all(phase in [e['phase'] for e in events] for phase in (
                'GLOBAL_CAPTURE_RANK_KV_OK', 'GLOBAL_CAPTURE_STATE_PREPARED_OK',
                'GLOBAL_CAPTURE_STATE_' + sha(canonical(state)) + '_OK',
                'GLOBAL_CAPTURE_PUBLISHED_OK', 'GLOBAL_CAPTURE_SEALED_OK', 'COMPLETE')),
                'cold_capture_seal_votes_missing')
        reports.append(report)
        logits.append(arrays)
    for key in ('output_token_ids', 'output_token_ids_sha256', 'row', 'generation_before',
                'generation_after', 'resume_decode_forward_count', 'mapping_version'):
        _require(reports[0][key] == reports[1][key], 'cold_rank_disagreement:' + key)
    _require(reports[0]['process'] != reports[1]['process'], 'cold_tp_process_identity_invalid')
    return config, state, reports, logits


def check_results(output):
    config = json.loads((Path(output) / 'run.json').read_text())
    mode = config['cold_mode']
    cfg, state, reports, logits = read_evidence(output, mode)
    result = dict(pass_stage_d=False, pass_reference=mode == 'capture', cuda_executed=True,
        tp_size=2, restored_kv_tokens=256 if mode == 'restore' else 0,
        restored_slots_used_by_decode=mode == 'restore', resume_decode_forward_count=16,
        recomputed_prefix_tokens=0, generation_unchanged=True, no_slot_leak=True,
        no_request_row_leak=True, output_token_ids_match=False, ranks=reports)
    if mode == 'capture':
        return result
    marker = json.loads((Path(cfg['ledger']) /
        (state['restore_id'] + '.committed.json')).read_text())
    _require(marker == dict(restore_id=state['restore_id'], run_id=cfg['run_id'],
        output_token_ids_sha256=reports[0]['output_token_ids_sha256'],
        state='COMMITTED', tp_ranks=[0, 1]), 'cold_replay_commit_marker_invalid')
    path = cfg['reference']['path']
    _require(reference_identity(path) == cfg['reference'], 'cold_reference_changed')
    ref_cfg, ref_state, refs, ref_logits = read_evidence(path, 'capture')
    def argv(c):
        args = list(c['argv'])
        args[args.index('--port') + 1] = '<port>'
        return args
    _require(state == ref_state and cfg['runtime_identity'] == ref_cfg['runtime_identity'] and
             argv(cfg) == argv(ref_cfg), 'cold_reference_runtime_or_state_mismatch')
    comparisons = [compare_outputs(refs[r]['output_token_ids'][1:],
                                  reports[r]['output_token_ids'][1:],
                                  ref_logits[r][1:], logits[r],
                                  output_offset=1, step_offset=1)
                   for r in (0, 1)]
    passed = all(c['output_token_ids_match'] for c in comparisons)
    result.update(pass_stage_d=passed, new_instance=True, original_prefix_prefill_tokens=0,
        output_token_ids_match=passed,
        reference_token_ids_sha256=token_digest(refs[0]['output_token_ids']),
        resumed_token_ids_sha256=token_digest(reports[0]['output_token_ids']),
        comparisons=comparisons, reference=path, restore_id=state['restore_id'])
    return result


def check_failed_restore(output):
    root = Path(output).resolve()
    cfg = json.loads((root / 'run.json').read_text())
    _require(cfg['cold_mode'] == 'restore' and cfg.get('inject_rank_failure') in (0, 1),
             'cold_failure_injection_run_required')
    ledger = Path(cfg['ledger'])
    restore_id = cfg['restore_id']
    claim = json.loads((ledger / (restore_id + '.claimed.json')).read_text())
    _require(claim == dict(restore_id=restore_id, run_id=cfg['run_id'], state='CLAIMED') and
             not (ledger / (restore_id + '.committed.json')).exists(),
             'cold_failure_replay_ledger_invalid')
    reports = []
    for rank in (0, 1):
        path = root / f'rank_{rank}.json'
        _require(path.is_file(), 'cold_missing_rank_failure_report')
        r = json.loads(path.read_text())
        _require(r['tp_rank'] == rank and r['failure_reason'] and
            all(r.get(k) is True for k in ('no_slot_leak', 'all_rows_released',
                'generation_unchanged', 'row_released')) and
            r.get('committed') is False and r.get('scheduler_injected') is False and
            r.get('resume_decode_forward_count') == 0 and
            r.get('transaction_state') in ('ABORTED', 'None'),
            'cold_rank_failure_not_compensated')
        failure = json.loads((root / f'rank_{rank}.events.jsonl').read_text().splitlines()[-1])
        _require(failure['phase'] == 'FAILED', 'cold_failure_terminal_event_missing')
        reports.append(r)
    _require(any('cold_injected_rank_restore_failure' in item for item in
                 reports[cfg['inject_rank_failure']].get('failure_chain', [])),
             'cold_injected_rank_did_not_fail')
    return dict(pass_stage_d=False, negative_test_pass=True, cuda_executed=True,
        tp_size=2, transaction_state=[r['transaction_state'] for r in reports],
        decode_forward_count=0, recomputed_prefix_tokens=0, no_slot_leak=True,
        no_request_row_leak=True, ranks=reports)


def check_exit_success(command, result):
    """Return success only for the selected check's independently validated verdict."""
    return (result.get('pass_stage_d') is True or result.get('pass_reference') is True or
            (command == 'check-failure' and result.get('negative_test_pass') is True))


def launch(args, capture=False):
    from .gpu_admission import args_mode
    args_mode(args)  # reject coordinator mismatch before creating/claiming state
    _require(args.output.is_absolute(), 'cold_absolute_output_required')
    cfg = dict(experiment='gpu-cold-resume-stage-d', cold_mode='capture' if capture else 'restore',
               timeout=args.timeout, decode_steps=16, mode='reference' if capture else 'resume',
               origin_server_process=process_identity())
    if capture:
        args.checkpoint = args.template_checkpoint
        # New bundle is created by this process, never reuse a historical one.
        bundle = args.bundle.resolve()
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.mkdir(exist_ok=False)
        cfg['capture_bundle'] = str(bundle)
        cfg['input_ids'] = stage_a.read_input_ids(args.input_ids,
            stage_a.validate_checkpoint(args.checkpoint.read_bytes()))
        cfg['template_only_not_runtime_restore_source'] = True
    else:
        source, state, receipt, binding = load_checkpoint(args.bundle, args.rs_source)
        origins = sealed_origin_processes(receipt)
        require_new_process(process_identity(), origins)
        require_origins_stopped(origins)
        _, ref_state, _, _ = read_evidence(args.reference, 'capture')
        _require(state == ref_state, 'cold_reference_state_mismatch')
        # Use the reconstructed artifact itself in launch admission if provided.
        args.checkpoint = (args.rs_source / 'recovered_node.bin' if args.rs_source else
                           args.bundle / 'global_node.bin')
        _require(sha(args.checkpoint.read_bytes()) == sha(source.load()), 'cold_launch_source_mismatch')
        cfg.update(state_checkpoint=str(args.bundle.resolve()), state_sha256=sha(canonical(state)),
            input_ids=state['origin_input_ids'], reference=reference_identity(args.reference),
            rs_source=str(args.rs_source.resolve()) if args.rs_source else None,
            rs_source_binding=binding, inject_rank_failure=args.inject_rank_failure,
            ledger=str(args.ledger.resolve()), restore_id=state['restore_id'])
        claim_restore(args.ledger.resolve(), state['restore_id'], args.output.name)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    stage_a.launch(args, extra_config=cfg, extra_env=dict(
        SGLANG_KV_EC_GPU_COLD_RESUME='1', SGLANG_KV_EC_GPU_RESUME='0'))


def submit(args):
    cfg = json.loads((args.output / 'run.json').read_text())
    _require(cfg['experiment'] == 'gpu-cold-resume-stage-d', 'cold_run_required')
    if cfg['cold_mode'] == 'capture':
        request_id = cfg['run_id']
    else:
        _, state, _, _ = load_checkpoint(cfg['state_checkpoint'], cfg.get('rs_source'))
        request_id = state['request_id']
    args.input_ids = args.output / 'submitted_input_ids.json'
    write_json(args.input_ids, cfg['input_ids'])
    stage_a.submit(args, max_new_tokens=17, sampling_overrides=dict(
        top_p=1.0, top_k=1, min_p=0.0, min_new_tokens=0, n=1,
        frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0),
        request_id=request_id)
    import time
    deadline = time.monotonic() + args.timeout
    while True:
        reports = []
        for rank in (0, 1):
            try:
                report = json.loads((args.output / f'rank_{rank}.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            _require(report.get('failure_reason') is None, 'cold_rank_reported_failure')
            reports.append(report)
        if len(reports) == 2:
            break
        _require(time.monotonic() < deadline, 'cold_result_report_timeout')
        time.sleep(0.1)
    print('Both TP rank reports saved. Run check; HTTP success is not acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('capture', 'launch'):
        p = sub.add_parser(name)
        p.add_argument('--model', required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--gpus', default='auto')
        p.add_argument('--port', type=int, default=30485)
        p.add_argument('--mem-fraction-static', type=float, default=0.6)
        p.add_argument('--gpu-admission-mode', choices=('strict', 'shared'), default='strict')
        p.add_argument('--timeout', type=float, default=120)
        if name == 'capture':
            p.add_argument('--template-checkpoint', type=Path, required=True)
            p.add_argument('--input-ids', type=Path, required=True)
        else:
            p.add_argument('--reference', type=Path, required=True)
            p.add_argument('--ledger', type=Path, required=True)
            p.add_argument('--rs-source', type=Path)
            p.add_argument('--inject-rank-failure', type=int, choices=(0, 1))
    p = sub.add_parser('submit', aliases=['restore'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:30485')
    p.add_argument('--wait-ready', type=float, default=600)
    p.add_argument('--timeout', type=float, default=300)
    for name in ('check', 'check-failure', 'reference'):
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('inspect')
    p.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        launch(args, capture=True)
    elif args.command == 'launch':
        launch(args)
    elif args.command in ('submit', 'restore'):
        submit(args)
    elif args.command == 'inspect':
        _, state, receipt, _ = load_checkpoint(args.bundle)
        print(json.dumps(dict(state=state, receipt=receipt), indent=2))
    else:
        try:
            if args.command == 'check-failure':
                result = check_failed_restore(args.output)
            elif args.command == 'reference':
                _require(json.loads((args.output / 'run.json').read_text())['cold_mode'] == 'capture',
                         'cold_reference_requires_origin_capture_run')
                result = check_results(args.output)
            else:
                result = check_results(args.output)
        except Exception as exc:
            result = dict(pass_stage_d=False, pass_reference=False,
                          failure_reason=f'{type(exc).__name__}:{exc}')
        save_verdict(args.output, result)
        print(json.dumps(result, indent=2))
        if not check_exit_success(args.command, result):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
