"""Stage D v1: sealed post-prefill-result boundary, greedy sampling ONLY.

No pickles, Python objects, CUDA addresses or allocator ownership are persisted.
Hashes detect accidental corruption; these local experiment records are not
authenticated checkpoints from untrusted writers.
"""

import json
import os
from pathlib import Path
import socket

from .gpu_kv_restore import BytesCheckpointSource, _require
from .gpu_restore_hook import validate_checkpoint
from .gpu_resume_hook import token_digest
from .node_block_export import canonical, sha, unpack
from .perf_observer import observed

SCHEMA = "kv-cold-request-v1"
SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=1, min_p=0.0,
                min_new_tokens=0, sampling_seed=None, skip_special_tokens=True,
                spaces_between_special_tokens=True, no_stop_trim=False,
                stream_interval=None, ignore_eos=True, n=1, max_new_tokens=17,
                frequency_penalty=0.0, presence_penalty=0.0,
                repetition_penalty=1.0, stop_token_ids=[], stop_strs=[],
                stop_regex_strs=[])


def process_identity():
    # starttime prevents PID reuse; boot ID and hostname bind the same machine.
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(host=socket.gethostname(), boot_id=Path(
        '/proc/sys/kernel/random/boot_id').read_text().strip(),
        pid=os.getpid(), start_ticks=int(stat[19]))


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)


def validate_sampling(params):
    for key, value in SAMPLING.items():
        got = getattr(params, key)
        if key.startswith('stop_'):
            got = list(got or [])
        _require(got == value, 'cold_sampling_policy_mismatch:' + key)
    _require(not any(getattr(params, key, None) for key in (
        'json_schema', 'regex', 'ebnf', 'structural_tag', 'logit_bias',
        'custom_params', 'stop', 'stop_regex')) and getattr(params, 'min_new_tokens', 0) == 0,
        'cold_sampling_extension_unsupported')


def validate_state(state, meta, block_hash):
    required = {'schema', 'checkpoint_version', 'request_version', 'tp_size', 'boundary',
        'origin_input_ids', 'output_ids', 'materialized_kv_tokens', 'kv_allocated_len',
        'pending_token_id', 'pending_token_has_kv', 'next_decode_input_ids',
        'next_decode_position', 'next_decode_seq_len', 'sampling', 'rng_policy',
        'finished', 'extend_batch_idx', 'decode_batch_idx', 'decode_steps', 'vocab_size',
        'request_id', 'restore_id', 'capture_run_id', 'token_start', 'token_count',
        'node_id', 'input_ids_sha256', 'kv_sha256', 'runtime_identity', 'origin_row',
        'origin_generation'}
    _require(isinstance(state, dict) and required.issubset(state),
             'cold_state_required_field_missing')
    _require(state['schema'] == SCHEMA and state['checkpoint_version'] == 1 and
             state['request_version'] == 1 and state['tp_size'] in (2, 4) and
             state['tp_size'] == len(meta['ranks']) and
             state['boundary'] == 'post_prefill_result_before_decode',
             'cold_state_version_or_boundary_mismatch')
    ids, outputs = state['origin_input_ids'], state['output_ids']
    _require(isinstance(ids, list) and len(ids) == 256 and
             isinstance(outputs, list) and len(outputs) == 1 and
             all(type(x) is int and 0 <= x < state['vocab_size'] for x in ids + outputs),
             'cold_token_sequence_invalid')
    _require(type(state['pending_token_id']) is int and
             state['pending_token_id'] == outputs[0] and
             state['pending_token_has_kv'] is False and
             state['materialized_kv_tokens'] == state['kv_allocated_len'] == 256 and
             state['next_decode_input_ids'] == outputs and
             state['next_decode_position'] == 256 and state['next_decode_seq_len'] == 257,
             'cold_pending_token_or_kv_length_mismatch')
    _require(state['sampling'] == SAMPLING and state['rng_policy'] == 'greedy_no_rng' and
             state['finished'] is False and state['extend_batch_idx'] == 1 and
             state['decode_batch_idx'] == 0 and state['decode_steps'] == 16,
             'cold_scheduler_or_sampling_state_unsupported')
    _require(state['kv_sha256'] == block_hash and state['node_id'] == meta['node_id'] and
             state['token_start'] == 0 and state['token_count'] == 256 and
             state['input_ids_sha256'] == token_digest(ids) and bool(state['request_id']),
             'cold_state_kv_binding_mismatch')
    _require(state['origin_row'] > 0 and state['origin_generation'] > 0,
             'cold_origin_generation_missing')
    for rank in meta['ranks']:
        _require(all(state['runtime_identity'][key] == rank[key] for key in (
            'model_sha256', 'source_commit', 'core_diff_sha256')) and
            state['input_ids_sha256'] == rank['token_ids_sha256'] and
            state['restore_id'] == rank['snapshot_id'] == rank['checkpoint_version'] and
            sha(state['request_id'].encode()) == rank['request_id'] and
            state['capture_run_id'] == rank['request_version'] == rank['manager_epoch'] and
            state['origin_row'] == rank['req_pool_idx'] and
            state['origin_generation'] == rank['req_generation'],
            'cold_state_rank_identity_mismatch')
    return state


@observed('checkpoint_load_validate_inclusive', 'validation')
def load_checkpoint(directory, rs_source=None):
    root = Path(directory).resolve()
    receipt = json.loads((root / 'COMMITTED.json').read_text())
    state_bytes = (root / 'request_state.json').read_bytes()
    _require(receipt['schema'] == 'kv-cold-seal-v1' and
             receipt['state'] == 'GLOBAL_COMMITTED' and
             receipt['request_state_sha256'] == sha(state_bytes), 'cold_state_hash_or_seal_invalid')
    # RS input is optional and never falls back to the original capture file.
    if rs_source is not None:
        if (Path(rs_source) / 'stage_e.json').exists():
            from .rs_cold_source import load_cold_source
            source, _, meta, binding = load_cold_source(rs_source)
            _require(binding['request_state_sha256'] == sha(state_bytes) and
                     binding['capture_seal_sha256'] == sha((root / 'COMMITTED.json').read_bytes()),
                     'stage_e_cold_state_seal_mismatch')
        else:
            from .rs_recovery_source import load_bundle
            source, _, meta, binding = load_bundle(rs_source)
        block = source.load()
    else:
        block = (root / 'global_node.bin').read_bytes()
        meta = validate_checkpoint(block)
        binding = None
    _require(sha(block) == receipt['kv_sha256'], 'cold_kv_hash_mismatch')
    state = validate_state(json.loads(state_bytes), meta, sha(block))
    # RS restore validates the reconstructed rank bytes, never reads the lost
    # target's original rank payloads. Local Stage D retains its original checks.
    rank_payloads = unpack(block)[1] if rs_source is not None else None
    for rank in range(state['tp_size']):
        rank_meta = json.loads((root / f'rank_{rank}.meta.json').read_text())
        payload = rank_payloads[rank] if rank_payloads is not None else (root / f'rank_{rank}.bin').read_bytes()
        _require(sha(payload) == meta['ranks'][rank]['sha256'] and
                 sha(canonical(rank_meta)) == sha(canonical(meta['ranks'][rank])),
                 'cold_rank_payload_or_metadata_mismatch')
    _require(state['restore_id'] == receipt['restore_id'] and
             receipt['rank_state_sha256'] == [sha(canonical(state))] * state['tp_size'] and
             len(receipt['origin_processes']) == state['tp_size'] and
             len({canonical(p) for p in receipt['origin_processes']}) == state['tp_size'],
             'cold_tp_state_commit_mismatch')
    server_process = receipt.get('origin_server_process')
    _require(isinstance(server_process, dict) and
             {'host', 'boot_id', 'pid', 'start_ticks'}.issubset(server_process) and
             all(server_process != p for p in receipt['origin_processes']),
             'cold_origin_server_process_missing')
    for rank in range(state['tp_size']):
        proof_path = root / f'rank_{rank}.capture.json'
        _require(sha(proof_path.read_bytes()) == receipt['rank_capture_sha256'][rank],
                 'cold_capture_proof_changed')
        proof = json.loads(proof_path.read_text())
        _require(proof['cuda_executed'] is True and proof['tp_rank'] == rank and
                 proof['state_sha256'] == sha(canonical(state)) and
                 proof['process'] == receipt['origin_processes'][rank] and
                 proof['materialized_kv_tokens'] == 256 and proof['pending_token_id'] ==
                 state['pending_token_id'], 'cold_capture_proof_invalid')
    _require([sha(rank_payloads[rank] if rank_payloads is not None else
                  (root / f'rank_{rank}.bin').read_bytes()) for rank in range(state['tp_size'])] ==
             receipt['rank_payload_sha256'], 'cold_rank_payload_changed')
    return BytesCheckpointSource(block), state, receipt, binding


def require_new_process(current, origins):
    _require(all(current['host'] == p['host'] and current['boot_id'] == p['boot_id']
                 and current != p for p in origins), 'cold_requires_new_same_host_process')


def sealed_origin_processes(receipt):
    return [*receipt['origin_processes'], receipt['origin_server_process']]


def require_origins_stopped(origins):
    for origin in origins:
        try:
            stat = Path(f"/proc/{int(origin['pid'])}/stat").read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        _require(int(stat[19]) != origin['start_ticks'], 'cold_origin_process_still_alive')


def claim_restore(ledger, restore_id, run_id):
    """Persistent one-shot claim, before allocation; failed attempts need review.

    All experiments using a checkpoint must share the same explicit ledger.
    This is replay protection for this local experiment, not distributed fencing.
    """
    _require(len(restore_id) == 64 and all(c in '0123456789abcdef' for c in restore_id),
             'cold_restore_id_invalid')
    root = Path(ledger)
    _require(root.is_absolute() and root.is_dir(), 'cold_shared_ledger_required')
    claim = root / (restore_id + '.claimed.json')
    _require(not claim.exists(), 'cold_checkpoint_already_claimed')
    write_json(claim,
               dict(restore_id=restore_id, run_id=run_id, state='CLAIMED'))
