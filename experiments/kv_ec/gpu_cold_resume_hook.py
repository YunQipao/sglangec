"""Default-off Stage D: capture an aligned boundary or inject a native Req.

The injected Req is built by handle_generate_request. Only its queue admission
is intercepted, after ordinary validation and before prefill planning. Native
update_running_batch / prepare_for_decode / run_batch / result processing own
every subsequent step. There is no model-only forward or synthetic old owner.
"""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import time

import torch

from .cold_resume_state import (SAMPLING, SCHEMA, load_checkpoint, process_identity,
    require_new_process, require_origins_stopped, sealed_origin_processes,
    validate_sampling, write_json)
from .debug_capture import source_identity
from .gpu_kv_restore import (BytesCheckpointSource, RestoreRejected, _require,
    restore_to_gpu, validate_live_ownership)
from .gpu_restore_hook import coordinated_restore
from .gpu_resume_hook import GPUResumeHook, token_digest
from .node_block_export import canonical, pack, sha

logger = logging.getLogger(__name__)


def state_field(fields, name, actual, expected, matches=None):
    fields[name] = dict(actual=actual, expected=expected,
        matches=actual == expected if matches is None else matches)


def decode_scheduler_state(req, batch, state, step, phase='READY'):
    """Native eager Decode state, after prepare and before/after result.

    prepare_for_decode advances both counters and lengths *before* the CUDA
    forward. The fill-ID array is an append-lazy prefill/admission cache; Decode
    result processing appends output_ids without refreshing that array.
    """
    fields = {}
    ids = list(req.origin_input_ids) + list(req.output_ids)
    cached = list(req.full_untruncated_fill_ids)
    minimum = len(state['origin_input_ids']) + len(state['output_ids'])
    state_field(fields, 'step', step, dict(min=1, max=state['decode_steps']),
                1 <= step <= state['decode_steps'])
    state_field(fields, 'forward_mode', batch.forward_mode.name, 'DECODE')
    state_field(fields, 'extend_batch_idx', req.extend_batch_idx, state['extend_batch_idx'])
    state_field(fields, 'decode_batch_idx', req.decode_batch_idx, state['decode_batch_idx'] + step)
    state_field(fields, 'output_ids_len', len(req.output_ids), step + int(phase == 'RESULT'))
    state_field(fields, 'origin_ids_sha256', token_digest(req.origin_input_ids), state['input_ids_sha256'])
    state_field(fields, 'sampled_seed', list(req.output_ids[:len(state['output_ids'])]), state['output_ids'])
    # Keep checking every cached token and its bounds; do not force a native
    # lazy cache to contain tokens that only the result processor has appended.
    state_field(fields, 'cached_fill_len', len(cached), dict(min=minimum, max=len(ids)),
                minimum <= len(cached) <= len(ids))
    state_field(fields, 'cached_fill_sha256', token_digest(cached), token_digest(ids[:len(cached)]))
    state_field(fields, 'seq_lens_cpu', batch.seq_lens_cpu.tolist(), [256 + step])
    state_field(fields, 'kv_committed_len_scheduled', req.kv_committed_len, 256 + step)
    return dict(phase=phase, step=step, fields=fields)


def require_decode_state(report):
    failures = {k: v for k, v in report['fields'].items() if not v['matches']}
    _require(not failures, 'cold_decode_scheduler_state_mismatch:' +
             json.dumps(dict(phase=report['phase'], step=report['step'], fields=failures), sort_keys=True))


def pool_cleanup(scheduler, req, row, generation):
    a, rp = scheduler.token_to_kv_pool_allocator, scheduler.req_to_token_pool
    free = a.free_pages.cpu().tolist() + a.release_pages.cpu().tolist()
    rows = list(rp.free_slots)
    return dict(no_slot_leak=(len(free) == len(set(free)) == a.size and
        set(free) == set(range(1, a.size + 1)) and a.is_not_in_free_group and
        not a.free_group and not getattr(a, '_kv_ec_restore_active', False) and
        not getattr(rp, '_kv_ec_row_owners', {}) and
        a.slot_lease_registry.active_lease_count == 0),
        all_rows_released=(len(rows) == len(set(rows)) == rp.size and
                          set(rows) == set(range(1, rp._alloc_size))),
        generation_unchanged=int(rp.req_generation[row]) == generation,
        generation_after=int(rp.req_generation[row]),
        row_released=req.req_pool_idx is None and row in rows,
        allocator_size=a.size, free_slots_after=len(free),
        request_pool_size=rp.size, free_rows_after=len(rows))


def build_decode_batch(s, req, pending):
    """Rebuild only the single-request, non-spec eager Decode batch contract."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    batch = ScheduleBatch.init_new([req], s.req_to_token_pool,
        s.token_to_kv_pool_allocator, s.tree_cache, s.model_config,
        s.enable_overlap, s.spec_algorithm)
    batch.forward_mode = ForwardMode.DECODE
    row, device = req.req_pool_idx, s.req_to_token_pool.device
    batch.req_pool_indices_cpu = torch.tensor([row], dtype=torch.int64)
    batch.req_pool_indices = batch.req_pool_indices_cpu.to(device)
    batch.seq_lens_cpu = torch.tensor([256], dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.to(device)
    batch.orig_seq_lens = torch.tensor([256], dtype=torch.int32, device=device)
    batch.seq_lens_sum = 256
    batch.multimodal_inputs = [None]
    batch.prefix_lens, batch.extend_lens = [256], [0]
    batch.extend_num_tokens = 0
    batch.extend_logprob_start_lens = [0]
    batch.input_ids = None  # Native resolve_forward_inputs reads FutureMap.
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, s.model_config.vocab_size)
    _require(not batch.sampling_info.penalizer_orchestrator.is_required,
             'cold_penalizer_state_unsupported')
    s.future_map.output_tokens_buf[row] = pending
    return batch


def seed_request_from_state(req, state):
    """Restore the logical Req counters while keeping the pending token un-KV'd."""
    _require(req.rid == state['request_id'] and not req.output_ids and
             req.kv_committed_len == 0,
             'cold_native_request_not_fresh')
    req.output_ids.extend(state['output_ids'])
    req._refresh_fill_ids()
    req.set_extend_range(state['token_start'],
                         state['token_start'] + state['materialized_kv_tokens'])
    req.already_computed = state['materialized_kv_tokens']
    req.num_matched_prefix_tokens = state['materialized_kv_tokens']
    req.extend_batch_idx = state['extend_batch_idx']
    req.decode_batch_idx = state['decode_batch_idx']
    _require(req.kv_committed_len == 0 and len(req.output_ids) == 1 and
             int(req.output_ids[-1]) == state['pending_token_id'] and
             list(req.full_untruncated_fill_ids) ==
             list(req.origin_input_ids) + list(req.output_ids),
             'cold_seeded_request_state_mismatch')
    return state['pending_token_id']


def validate_decode_scheduler_state(req, batch, state, step):
    report = decode_scheduler_state(req, batch, state, step)
    require_decode_state(report)
    return report


class GPUColdResumeHook(GPUResumeHook):
    @classmethod
    def from_env(cls, scheduler):
        output = Path(os.environ['SGLANG_KV_EC_GPU_RESTORE_OUTPUT'])
        cfg = json.loads((output / 'run.json').read_text())
        _require(cfg['experiment'] == 'gpu-cold-resume-stage-d' and
                 cfg['cold_mode'] in ('capture', 'restore'), 'cold_run_config_invalid')
        source, state, receipt, binding = None, None, None, None
        if cfg['cold_mode'] == 'restore':
            source, state, receipt, binding = load_checkpoint(cfg['state_checkpoint'], cfg.get('rs_source'))
            _require(sha(source.load()) == cfg['checkpoint_sha256'] and
                     sha(canonical(state)) == cfg['state_sha256'], 'cold_launch_checkpoint_changed')
            origins = sealed_origin_processes(receipt)
            require_new_process(process_identity(), origins)
            require_origins_stopped(origins)
        hook = cls(scheduler, cfg['checkpoint'], output, cfg['run_id'], cfg['timeout'],
                   mode='reference' if state is None else 'resume', source=source,
                   source_name='stage-d-aligned-capture' if state is not None else None)
        hook.cold_mode, hook.cfg = cfg['cold_mode'], cfg
        hook.state, hook.receipt = state, receipt
        hook.capture_done = False
        hook.process = process_identity()
        hook.evidence.update(schema='kv-gpu-cold-resume-stage-d-v1',
            cold_mode=hook.cold_mode, process=hook.process, pass_stage_d=False,
            initial_install=state is not None, prefix_prefill_forward_count=0)
        if binding is not None:
            hook.rs_source_binding = binding
            hook.evidence.update(rs_source=binding, runtime_original_local_file_used=False)
        if state is not None:
            _require(state['runtime_identity'] == hook.evidence['runtime_identity'] and
                     state['vocab_size'] == scheduler.model_config.vocab_size,
                     'cold_runtime_identity_mismatch')
            _require(binding == cfg.get('rs_source_binding'), 'cold_rs_source_binding_changed')
            scheduler.kv_ec_source_identity = state['runtime_identity']
            hook.request_state_binding = dict(restore_id=state['restore_id'],
                request_state_sha256=cfg['state_sha256'])
            hook.prefill_seen = True
            hook.evidence.update(state_sha256=cfg['state_sha256'], restore_id=state['restore_id'],
                origin_processes=receipt['origin_processes'], original_prefill_tokens=0,
                original_prefill_forward_count=0, new_instance=True)
        return hook

    def before_inputs(self, inputs):
        inputs = list(inputs)
        for msg in inputs:
            if type(msg).__name__ == 'TokenizedGenerateReqInput':
                validate_sampling(msg.sampling_params)
                _require(not any(getattr(msg, k, None) for k in (
                    'return_logprob', 'return_hidden_states', 'return_routed_experts',
                    'return_indexer_topk', 'return_sampling_mask', 'require_reasoning',
                    'lora_id', 'custom_logit_processor', 'positional_embed_overrides',
                    'token_type_ids', 'session_params', 'cache_salt')),
                    'cold_plain_text_request_required')
        super().before_inputs(inputs)

    def after_inputs(self):
        if self.cold_mode != 'restore' or self.req is not None or not self.admitted:
            return
        s, state = self.scheduler, self.state
        self.deadline = time.monotonic() + self.timeout
        def gate(phase, action):
            error = None
            try:
                action()
            except Exception as exc:
                error = exc
            agreed = self._vote(phase, error is None)
            if error is not None or not agreed:
                raise RestoreRejected('cold_' + phase.lower() + '_failed') from error
        def allocate_row():
            from sglang.srt.managers.schedule_batch import ReqKvInfo
            _require(s.forward_ct == 0 and s.running_batch.is_empty() and
                     s.last_batch is None and s.chunked_req is None and
                     len(s.waiting_queue) == 1, 'cold_requires_pristine_scheduler')
            self.req = s.waiting_queue[0]
            _require(list(self.req.origin_input_ids) == state['origin_input_ids'] and
                     not self.req.output_ids and self.req.kv_committed_len == 0 and
                     self.req.req_pool_idx is None and not self.req.finished(),
                     'cold_native_request_not_fresh')
            _require(s.req_to_token_pool.available_size() == s.req_to_token_pool.size,
                     'cold_request_rows_not_empty')
            self.req.kv = ReqKvInfo(kv_allocated_len=0, swa_evicted_seqlen=0)
            _require(s.req_to_token_pool.alloc([self.req]) is not None, 'cold_request_row_unavailable')
            self.row = self.req.req_pool_idx
            self.generation = int(s.req_to_token_pool.req_generation[self.row])
            # Logical prefill is restored, but no target-instance EXTEND runs.
            self.sampled_seed = seed_request_from_state(self.req, state)
            self.evidence.update(row=self.row, generation_before=self.generation,
                old_slots=[], prefill_seed_token_id=self.sampled_seed,
                request_id=self.req.rid, restored_request_state=state,
                token_ids_sha256=state['input_ids_sha256'])
            self.pending_batch = build_decode_batch(s, self.req, self.sampled_seed)
        try:
            gate('COLD_ROW', allocate_row)
            self.tx = coordinated_restore(self._prepare_restore, self._vote, self._observe_tx)
            self.prefix_slots = self.tx.new_slots
            self.req.prefix_indices = torch.tensor(self.prefix_slots, dtype=torch.int64,
                device=s.req_to_token_pool.device)
            self.evidence.update(committed=True, restored_kv_tokens=256, mapping_version=1,
                transaction_state=self.tx.state, forward_ct_before_restore=0,
                forward_ct_after_restore=s.forward_ct)
            gate('COLD_REQUEST_READY', lambda: self._live(self.pending_batch))
            # Publish to the native planner only after all TP gates complete.
            s.waiting_queue.remove(self.req)
            s.running_batch = self.pending_batch
            self.evidence['scheduler_injected'] = True
            self._event('GLOBAL_COMMIT', self.tx)
        except Exception as exc:
            self._cleanup_failed_install()
            self._fail(exc)
            raise

    def _cleanup_failed_install(self):
        s, req = self.scheduler, self.req
        if req is None or req.req_pool_idx is None:
            return
        pool = s.model_worker.model_runner.token_to_kv_pool
        torch.cuda.synchronize(pool.k_buffer[0].device)
        if self.tx is not None and not self.tx.finalized and self.tx.state != 'ABORTED':
            self.tx.abort()
        if req.kv_committed_len:
            from sglang.srt.mem_cache.common import release_kv_cache
            release_kv_cache(req, s.tree_cache, is_insert=False)
        else:
            s.req_to_token_pool.free(req)
        self.evidence.update(pool_cleanup(s, req, self.row, self.generation),
            transaction_state=self.tx.state if self.tx is not None else 'ABORTED',
            committed=False, resume_decode_forward_count=0, scheduler_injected=False)

    def _observe_tx(self, phase, tx):
        if phase == 'VERIFIED':
            self.evidence.update(verified=True, new_slots=list(tx.new_slots),
                slots_disjoint=not tx.old_slots, stats=asdict(tx.stats))
        self._event(phase, tx)

    def _prepare_restore(self):
        if self.cold_mode != 'restore':
            return super()._prepare_restore()
        self.tx = restore_to_gpu(self.scheduler, self.req, self.source,
            initial_install=True, source_name=self.source_name)
        if self.cfg.get('inject_rank_failure') == self.rank:
            self.tx.abort()
            raise RestoreRejected('cold_injected_rank_restore_failure')
        return self.tx

    def before_forward(self, batch):
        if self.cold_mode == 'capture':
            return super().before_forward(batch)
        self.deadline = time.monotonic() + self.timeout
        def check():
            n = self.evidence['resume_decode_forward_count'] + 1
            _require(self.admitted and not self.done and self.evidence['committed'] and
                     self.evidence['scheduler_injected'], 'cold_decode_not_globally_committed')
            self._validate_decode_state(batch, n, 'READY')
            self._live(batch)
        self._barrier('READY_' + str(self.evidence['resume_decode_forward_count'] + 1), check)

    def _validate_decode_state(self, batch, step, phase):
        report = decode_scheduler_state(self.req, batch, self.state, step, phase)
        fields = report['fields']
        completed = self.evidence['resume_decode_forward_count']
        state_field(fields, 'observed_decode_forwards', completed,
                    step - int(phase == 'READY'))
        observations = self.evidence['forward_observations']
        materialized = observations[-1]['kv_length'] if observations else self.state['materialized_kv_tokens']
        state_field(fields, 'kv_materialized_len_observed', materialized, 256 + completed)
        state_field(fields, 'kv_allocated_len', self.req.kv.kv_allocated_len, 256 + step)
        state_field(fields, 'row', self.req.req_pool_idx, self.row)
        state_field(fields, 'generation', int(self.scheduler.req_to_token_pool.req_generation[self.row]), self.generation)
        if phase != 'FORWARD_COMPLETE':
            # READY: relay contains the input token with no KV yet. RESULT:
            # relay contains the next sampled token, also not materialized.
            state_field(fields, 'future_map_pending_token',
                int(self.scheduler.future_map.output_tokens_buf[self.row]), int(self.req.output_ids[-1]))
        else:
            # run_batch has already overwritten the relay with the next sample;
            # the synchronized model observer retains the actual consumed input.
            state_field(fields, 'forward_input_ids', observations[-1]['input_ids'], [int(self.req.output_ids[-1])])
            state_field(fields, 'forward_positions', observations[-1]['positions'], [255 + step])
        report.update(tp_rank=self.rank, forward_ct=self.scheduler.forward_ct)
        failed = any(not field['matches'] for field in fields.values())
        if step <= 3 or failed:
            self.evidence.setdefault('decode_state_diagnostics', []).append(report)
            with (self.output / f'rank_{self.rank}.decode_state.jsonl').open('a') as handle:
                handle.write(json.dumps(report, sort_keys=True) + '\n')
            logger.info('STAGE_D_DECODE_STATE %s', json.dumps(report, sort_keys=True))
        require_decode_state(report)

    def after_forward(self, batch):
        if self.cold_mode == 'restore':
            self._barrier('COLD_FORWARD_STATE_' + str(self.evidence['resume_decode_forward_count']),
                lambda: self._validate_decode_state(batch,
                    self.evidence['resume_decode_forward_count'], 'FORWARD_COMPLETE'))
            return
        if not self.prefill_seen:
            def stage():
                from .online_chunk import OnlineProtector, finish_metadata
                # Reuse the real online exporter, with no worker and no network.
                exporter = OnlineProtector.__new__(OnlineProtector)
                exporter.scheduler, exporter.run_id = self.scheduler, self.run_id
                exporter.identity = self.evidence['runtime_identity']
                exporter.max_bytes = 64 * 1024**2
                exporter.stream = torch.cuda.Stream()
                item = exporter._stage(self.req, 0, self.generation)
                item.done.synchronize()
                payload = item.host.numpy().tobytes()
                meta = finish_metadata(item.metadata, payload)
                bundle = Path(self.cfg['capture_bundle'])
                with (bundle / f'rank_{self.rank}.bin').open('xb') as handle:
                    handle.write(payload)
                write_json(bundle / f'rank_{self.rank}.meta.json', meta)
            self._barrier('CAPTURE_RANK_KV', stage)
        super().after_forward(batch)

    def _seal_capture(self):
        bundle = Path(self.cfg['capture_bundle'])
        def prepare():
            _require(self.req.kv_committed_len == self.req.kv.kv_allocated_len == 256 and
                     list(self.req.output_ids) == [int(self.scheduler.future_map.output_tokens_buf[self.row])]
                     and not self.req.finished() and self.req.decode_batch_idx == 0,
                     'cold_capture_boundary_not_consistent')
            validate_sampling(self.req.sampling_params)
            self._live()
            metas = [json.loads((bundle / f'rank_{r}.meta.json').read_text()) for r in (0, 1)]
            block = pack(self.meta['node_id'], [(bundle / f'rank_{r}.bin').read_bytes()
                for r in (0, 1)], metas)
            self.state = dict(schema=SCHEMA, checkpoint_version=1, request_version=1,
                boundary='post_prefill_result_before_decode', tp_size=2,
                origin_input_ids=list(self.req.origin_input_ids), output_ids=list(self.req.output_ids),
                materialized_kv_tokens=256, kv_allocated_len=256,
                pending_token_id=self.req.output_ids[0], pending_token_has_kv=False,
                next_decode_input_ids=list(self.req.output_ids), next_decode_position=256,
                next_decode_seq_len=257, sampling=SAMPLING, rng_policy='greedy_no_rng',
                finished=False, extend_batch_idx=1, decode_batch_idx=0, decode_steps=16,
                vocab_size=self.scheduler.model_config.vocab_size, request_id=self.req.rid,
                origin_row=self.row, origin_generation=self.generation,
                restore_id=metas[0]['snapshot_id'], capture_run_id=self.run_id,
                token_start=0, token_count=256, node_id=self.meta['node_id'],
                input_ids_sha256=token_digest(self.req.origin_input_ids), kv_sha256=sha(block),
                runtime_identity=source_identity(self.scheduler.server_args.model_path))
            self.evidence['request_id'] = self.req.rid
            self.captured_block = block
            self.evidence.update(capture_state_sha256=sha(canonical(self.state)),
                captured_kv_sha256=sha(block), capture_bundle=str(bundle))
            self.evidence['template_checkpoint_sha256'] = self.evidence['checkpoint_sha256']
            self.evidence['checkpoint_sha256'] = sha(block)
            self.evidence['checkpoint'] = str(bundle / 'global_node.bin')
            write_json(bundle / f'rank_{self.rank}.capture.json', dict(tp_rank=self.rank,
                process=self.process, cuda_executed=self.evidence['cuda_executed'],
                state_sha256=sha(canonical(self.state)), materialized_kv_tokens=256,
                pending_token_id=self.req.output_ids[0]))
        self._barrier('CAPTURE_STATE_PREPARED', prepare)
        self._barrier('CAPTURE_STATE_' + sha(canonical(self.state)), lambda: None)
        def publish():
            if self.rank == 0:
                with (bundle / 'global_node.bin').open('xb') as handle:
                    handle.write(self.captured_block)
                write_json(bundle / 'request_state.json', self.state)
                proofs = [json.loads((bundle / f'rank_{r}.capture.json').read_text()) for r in (0, 1)]
                write_json(bundle / 'COMMITTED.json', dict(schema='kv-cold-seal-v1',
                    state='GLOBAL_COMMITTED', restore_id=self.state['restore_id'],
                    request_state_sha256=sha((bundle / 'request_state.json').read_bytes()),
                    rank_state_sha256=[p['state_sha256'] for p in proofs],
                    rank_capture_sha256=[sha((bundle / f'rank_{r}.capture.json').read_bytes()) for r in (0, 1)],
                    rank_payload_sha256=[sha((bundle / f'rank_{r}.bin').read_bytes()) for r in (0, 1)],
                    kv_sha256=sha(self.captured_block), origin_processes=[p['process'] for p in proofs],
                    origin_server_process=self.cfg['origin_server_process']))
        self._barrier('CAPTURE_PUBLISHED', publish)
        self._barrier('CAPTURE_SEALED', lambda: load_checkpoint(bundle))
        self.capture_done = True

    def after_result(self, batch):
        if self.cold_mode == 'capture':
            super().after_result(batch)
            if not self.capture_done:
                self._seal_capture()
            return
        if self.done:
            return
        def check():
            outputs = list(self.req.output_ids)
            count = self.evidence['resume_decode_forward_count']
            # Final result processing releases the Req row and KV, so inspect
            # live runtime fields only while the request still owns them.
            if not self.req.finished():
                self._validate_decode_state(batch, count, 'RESULT')
            _require(len(outputs) == count + 1 and outputs[0] == self.sampled_seed and
                     self.req.extend_batch_idx == self.state['extend_batch_idx'],
                     'cold_result_or_prefill_mismatch')
            self.evidence.update(output_token_ids=outputs, output_token_ids_sha256=token_digest(outputs),
                resumed_generated_tokens=count)
            if not self.req.finished():
                self._live(batch)
                return
            torch.cuda.synchronize(self.scheduler.model_worker.model_runner.token_to_kv_pool.k_buffer[0].device)
            cleanup = pool_cleanup(self.scheduler, self.req, self.row, self.generation)
            self.evidence.update(cleanup, request_finished=True)
            _require(all(cleanup[k] for k in ('no_slot_leak', 'all_rows_released',
                'row_released', 'generation_unchanged')) and count == 16 and
                self.scheduler.forward_ct - self.forward_start == 16,
                'cold_cleanup_or_decode_count_failed')
        self._barrier('RESULT_' + token_digest(self.req.output_ids), check)
        if self.req.finished():
            marker = Path(self.cfg['ledger']) / (self.state['restore_id'] + '.committed.json')
            def publish_commit():
                if self.rank == 0:
                    write_json(marker, dict(restore_id=self.state['restore_id'],
                        run_id=self.run_id, output_token_ids_sha256=token_digest(self.req.output_ids),
                        state='COMMITTED', tp_ranks=[0, 1]))
            self._barrier('RESTORE_LEDGER_COMMITTED', publish_commit)
            self._barrier('RESTORE_LEDGER_VISIBLE',
                          lambda: _require(marker.exists(), 'cold_commit_marker_missing'))
            self.evidence.update(local_cold_resume_complete=True, transaction_state=self.tx.state)
            self._event('COMPLETE', self.tx)
            self._save()
            self.events.close()
            self.done = True
