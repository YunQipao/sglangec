"""Opt-in Stage B: preserve a live request, replace KV, then run native Decode.

No request reconstruction or cold failover is claimed. The sampled Prefill
token stays in the native FutureMap and has NO KV until the first Decode.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .gpu_kv_restore import (
    RestoreRejected, _metadata_identity, _read_layer, _require,
    restore_to_gpu, validate_live_ownership,
)
from .gpu_restore_hook import GPURestoreHook, coordinated_restore
from .node_block_export import sha, unpack


def token_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def require_cuda_forward(pool, fb, logits):
    _require(logits is not None, "stage_b_raw_logits_missing")
    tensors = [*pool.k_buffer, *pool.v_buffer, fb.input_ids, fb.positions,
               fb.seq_lens, fb.out_cache_loc, fb.req_pool_indices, logits]
    _require(all(t.is_cuda for t in tensors), "stage_b_requires_actual_cuda_tensors")


def validate_decode_observation(*, step, outputs, seq_len, committed_len,
                                input_ids, positions, indices, mapping,
                                prefix_slots, out_slots):
    """Validate actual ForwardBatch and kernel metadata, not HTTP success."""
    _require(step > 0 and len(outputs) == step, "resume_output_kv_off_by_one")
    _require(seq_len == committed_len == 256 + step, "resume_decode_length_mismatch")
    _require(input_ids == [outputs[-1]] and positions == [255 + step],
             "resume_decode_input_or_position_mismatch")
    _require(len(mapping) == len(set(mapping)) == seq_len and
             indices == mapping and indices[:256] == list(prefix_slots) and
             out_slots == [mapping[-1]], "resume_attention_slot_mismatch")


class GPUResumeHook(GPURestoreHook):
    input_requirement_reason = "stage_b_requires_plain_256_token_resume_request"

    def __init__(self, scheduler, checkpoint, output, run_id, timeout=120, *,
                 mode="resume", decode_steps=16, source=None, source_name=None):
        _require(mode in ("reference", "resume", "corrupt") and
                 16 <= decode_steps <= 128, "stage_b_invalid_mode_or_steps")
        self.mode, self.decode_steps = mode, decode_steps
        self.output_token_limit = 1 if mode == "corrupt" else decode_steps + 1
        super().__init__(scheduler, checkpoint, output, run_id, timeout,
                         source=source, source_name=source_name)
        runner = scheduler.model_worker.model_runner
        _require(type(runner.attn_backend).__name__ == "TritonAttnBackend" and
                 getattr(runner.attn_backend, "sliding_window_size", None) in (None, -1, 0) and
                 not getattr(runner.attn_backend, "use_sliding_window_kv_pool", False) and
                 not getattr(runner.attn_backend, "use_mla", False) and
                 scheduler.server_args.chunked_prefill_size == -1,
                 "stage_b_requires_eager_triton_unfragmented_prefill")
        from .debug_capture import source_identity
        identity = source_identity(scheduler.server_args.model_path)
        _require(identity["model_sha256"] == self.meta["ranks"][self.rank]["model_sha256"],
                 "stage_b_checkpoint_model_mismatch")
        self.evidence.update(schema="kv-gpu-resume-stage-b-v1", mode=mode,
            pass_stage_b=False, pass_reference=False, negative_test_pass=False,
            runtime_identity=identity, decode_steps=decode_steps,
            restored_kv_tokens=0, restored_slots_used_by_decode=False,
            resume_decode_forward_count=0, recomputed_prefix_tokens=0,
            output_token_ids_match=False, reference_token_ids_sha256=None,
            resumed_token_ids_sha256=None, mapping_version=0,
            forward_observations=[], logits=[], corruption_detected=False)
        self.prefill_seen = False
        self.req = None
        self.prefix_slots = ()
        self.original_slots = ()
        self.forward_start = scheduler.forward_ct
        self.sampled_seed = None
        runner.kv_ec_resume_observer = self.observe_model_forward

    @classmethod
    def from_env(cls, scheduler):
        source, source_name, rs = None, None, None
        if os.environ.get("SGLANG_KV_EC_RS_RESUME") == "1":
            from .rs_recovery_source import load_bundle
            config = json.loads((Path(os.environ["SGLANG_KV_EC_GPU_RESTORE_OUTPUT"]) /
                                 "run.json").read_text())
            _require(config.get("experiment_mode") == "rs-resume" and
                     config["mode"] == os.environ["SGLANG_KV_EC_GPU_RESUME_MODE"] == "resume",
                     "stage_c_requires_explicit_rs_resume")
            source, report, _, rs = load_bundle(config["rs_source"]["path"], config["rs_source"])
            _require(config["checkpoint_sha256"] == rs["block_sha256"],
                     "stage_c_run_source_hash_mismatch")
            source_name = "native-rdma-rs:" + report["receipt"]["stripe_id"]
        hook = cls(scheduler, os.environ["SGLANG_KV_EC_GPU_RESTORE_CHECKPOINT"],
                   os.environ["SGLANG_KV_EC_GPU_RESTORE_OUTPUT"],
                   os.environ["SGLANG_KV_EC_GPU_RESTORE_RUN_ID"],
                   float(os.environ.get("SGLANG_KV_EC_GPU_RESTORE_TIMEOUT", "120")),
                   mode=os.environ["SGLANG_KV_EC_GPU_RESUME_MODE"],
                   decode_steps=int(os.environ["SGLANG_KV_EC_GPU_RESUME_STEPS"]),
                   source=source, source_name=source_name)
        if rs is not None:
            hook.rs_source_binding = rs
            hook.evidence.update(experiment_mode="rs-resume", rs_source=rs,
                source_kind="native-rdma-rs-immutable-bytes",
                runtime_original_local_file_used=False)
        return hook

    def _fail(self, exc):
        self.evidence["pass_stage_b"] = False
        self.evidence["pass_reference"] = False
        self.evidence["negative_test_pass"] = False
        super()._fail(exc)

    def _barrier(self, phase, action):
        error, value = None, None
        try:
            value = action()
        except Exception as exc:
            error = exc
        try:
            unanimous = self._vote(phase, error is None)
            if error is not None or not unanimous:
                raise RestoreRejected("stage_b_" + phase.lower() + "_failed") from error
            return value
        except Exception as exc:
            self._fail(exc)
            raise

    def before_inputs(self, inputs):
        # Materialize once so validation and the base admission see the same batch.
        inputs = list(inputs)
        try:
            for msg in inputs:
                if type(msg).__name__ == "TokenizedGenerateReqInput":
                    p = msg.sampling_params
                    # SamplingParams.normalize rewrites temperature=0 to
                    # temperature=1, top_k=1 before scheduler admission.
                    _require(p.top_k == 1 and p.ignore_eos and p.n == 1 and
                             p.frequency_penalty == p.presence_penalty == 0 and
                             p.repetition_penalty == 1 and not p.logit_bias and
                             not p.json_schema and not p.regex and not p.ebnf and
                             not p.structural_tag and not p.stop_token_ids and
                             not p.stop_strs and not p.stop_regex_strs,
                             "stage_b_requires_plain_greedy_sampling")
        except Exception as exc:
            self.evidence["rejected_input"] = dict(
                message_type=type(msg).__name__, admitted=self.admitted)
            self._fail(exc)
            raise
        super().before_inputs(inputs)

    def _live(self, batch=None):
        _require((batch is None or (len(batch.reqs) == 1 and batch.reqs[0] is self.req)) and
                 not self.req.finished() and not self.req.is_retracted and
                 not self.req.to_finish and self.req.req_pool_idx == self.row and
                 int(self.scheduler.req_to_token_pool.req_generation[self.row]) == self.generation,
                 "stage_b_request_identity_changed")
        _require(token_digest(self.req.origin_input_ids) ==
                 self.meta["ranks"][self.rank]["token_ids_sha256"],
                 "stage_b_request_prefix_changed")
        mapping = validate_live_ownership(self.scheduler, self.req, 0,
                                          self.req.kv_committed_len)
        _require(mapping[:256] == self.prefix_slots, "stage_b_prefix_mapping_changed")
        return list(mapping)

    def before_forward(self, batch):
        self.deadline = time.monotonic() + self.timeout
        def check():
            _require(self.admitted and not self.done, "stage_b_unadmitted_forward")
            if not self.prefill_seen:
                _require(len(batch.reqs) == 1 and batch.forward_mode.name == "EXTEND" and
                         batch.prefix_lens == [0] and batch.extend_num_tokens == 256,
                         "stage_b_requires_one_original_prefill")
                self.req = batch.reqs[0]
                self.row = self.req.req_pool_idx
                self.generation = int(self.scheduler.req_to_token_pool.req_generation[self.row])
                _require(not self.req.output_ids and self.req.extend_batch_idx == 1,
                         "stage_b_invalid_initial_request")
                self.original_slots = validate_live_ownership(self.scheduler, self.req, 0, 256)
                self.prefix_slots = self.original_slots
                _metadata_identity(self.meta["ranks"][self.rank], self.scheduler, self.req,
                                   rank=self.rank, tp_size=self.tp_size, token_start=0, token_count=256)
                self.evidence.update(row=self.row, generation_before=self.generation,
                                     old_slots=list(self.original_slots))
            else:
                _require(self.mode != "corrupt" and batch.forward_mode.name == "DECODE",
                         "stage_b_unexpected_prefill_or_corrupt_decode")
                _require(self.mode == "reference" or self.evidence["committed"],
                         "stage_b_decode_before_global_commit")
                self._live(batch)
                n = self.evidence["resume_decode_forward_count"] + 1
                _require(n <= self.decode_steps and len(self.req.output_ids) == n and
                         int(batch.seq_lens_cpu[0]) == self.req.kv_committed_len == 256 + n,
                         "stage_b_decode_scheduler_state_mismatch")
        phase = "READY_" + str(len(self.evidence["forward_observations"]))
        self._barrier(phase, check)

    def observe_model_forward(self, fb, logits_output, can_run_graph):
        """Called inside TpModelWorker after CUDA forward, before sampling."""
        from .perf_observer import observed
        @observed('decode_forward_validation', 'validation')
        def observe():
            pool = self.scheduler.model_worker.model_runner.token_to_kv_pool
            require_cuda_forward(pool, fb, logits_output.next_token_logits)
            torch.cuda.synchronize(pool.k_buffer[0].device)
            _require(not can_run_graph and fb.batch_size == 1,
                     "stage_b_graph_or_batch_unsupported")
            step = self.evidence["resume_decode_forward_count"] + int(self.prefill_seen)
            mode = fb.forward_mode.name
            mapping = self._live()
            inputs = fb.input_ids.cpu().tolist()
            positions = fb.positions.cpu().tolist()
            rows = fb.req_pool_indices.cpu().tolist()
            out_slots = fb.out_cache_loc.cpu().tolist()
            _require(rows == [self.row], "stage_b_forward_request_row_mismatch")
            if self.prefill_seen:
                _require(mode == "DECODE", "stage_b_prefix_recomputed")
                # This checkout binds attention via ModelRunner's
                # ForwardContext; ForwardBatch has no attn_backend field.
                backend = self.scheduler.model_worker.model_runner.attn_backend
                _require(type(backend).__name__ == "TritonAttnBackend" and
                         backend.req_to_token is self.scheduler.req_to_token_pool.req_to_token and
                         backend.token_to_kv_pool is pool,
                         "stage_b_unexpected_attention_backend")
                md = backend.forward_metadata
                indptr = md.kv_indptr[:2].cpu().tolist()
                _require(indptr == [0, len(mapping)], "stage_b_kv_indptr_mismatch")
                indices = md.kv_indices[:indptr[-1]].cpu().tolist()
                validate_decode_observation(step=step, outputs=self.req.output_ids,
                    seq_len=int(fb.seq_lens[0]), committed_len=self.req.kv_committed_len,
                    input_ids=inputs, positions=positions, indices=indices,
                    mapping=mapping, prefix_slots=self.prefix_slots,
                    out_slots=out_slots)
                self.evidence["resume_decode_forward_count"] = step
                self.evidence["restored_slots_used_by_decode"] = self.mode == "resume"
            else:
                _require(mode == "EXTEND" and inputs == list(self.req.origin_input_ids) and
                         positions == list(range(256)), "stage_b_invalid_prefill_forward")
                indices = []  # Prefill uses its native extend kernel, not Decode indices.
            _require(logits_output.next_token_logits is not None, "stage_b_raw_logits_missing")
            logits = logits_output.next_token_logits.detach().float().cpu().numpy()
            _require(logits.ndim == 2 and logits.shape[0] == 1 and np.isfinite(logits).all(),
                     "stage_b_invalid_raw_logits")
            name = f"rank_{self.rank}.logits_{step:03d}.npy"
            with (self.output / name).open("xb") as handle:
                np.save(handle, logits, allow_pickle=False)
            self.evidence["logits"].append(dict(step=step, file=name,
                sha256=sha((self.output / name).read_bytes()), shape=list(logits.shape)))
            self.evidence["forward_observations"].append(dict(
                step=step, mode=mode, forward_ct=self.scheduler.forward_ct,
                input_ids=inputs, positions=positions, kv_length=len(mapping),
                kv_indices=indices, mapping=mapping,
                req_pool_indices=rows, out_cache_loc=out_slots,
                mapping_version=self.evidence["mapping_version"], generation=self.generation))
        self._barrier("OBSERVED_" + str(len(self.evidence["forward_observations"])), observe)

    def _prepare_restore(self):
        _require(sha(self.source.load()) == self.block_hash, "checkpoint_changed_after_startup")
        self.tx = restore_to_gpu(self.scheduler, self.req, self.source,
                                 source_name=self.source_name)
        try:
            pool = self.scheduler.model_worker.model_runner.token_to_kv_pool
            if self.mode == "corrupt" and self.rank == 0:
                # Flip exactly one byte in a PRIVATE, not-yet-mapped K slot.
                # Both ranks still hold their undo journals; no Decode is possible.
                pool.k_buffer[0][self.tx.new_slots[0]].view(torch.uint8).reshape(-1)[0].bitwise_xor_(1)
                torch.cuda.synchronize(pool.k_buffer[0].device)
                self.evidence["corruption_injected"] = True
            # Re-read the private slots immediately before the global VERIFIED vote.
            # The negative case exercises this gate without weakening Stage A checks.
            meta, payloads = unpack(self.source.load())
            slots = torch.tensor(self.tx.new_slots, dtype=torch.int64, device=pool.k_buffer[0].device)
            stream = torch.cuda.current_stream(pool.k_buffer[0].device)
            cursor = 0
            for record in meta["ranks"][self.rank]["layer_records"]:
                size = record["k_nbytes"] + record["v_nbytes"]
                got = _read_layer(pool, record["layer_id"], slots, stream)
                if got != payloads[self.rank][cursor:cursor + size]:
                    self.evidence["corruption_detected"] = self.mode == "corrupt" and self.rank == 0
                    raise RestoreRejected("stage_b_precommit_byte_mismatch")
                cursor += size
            return self.tx
        except Exception:
            self.tx.abort()
            raise

    def _request_state(self, batch):
        req = self.req
        return dict(row=req.req_pool_idx, generation=int(
            self.scheduler.req_to_token_pool.req_generation[self.row]),
            input_sha256=token_digest(req.origin_input_ids), output_ids=list(req.output_ids),
            kv_committed_len=req.kv_committed_len, kv_allocated_len=req.kv.kv_allocated_len,
            seq_lens=batch.seq_lens_cpu.tolist(), extend_batch_idx=req.extend_batch_idx,
            decode_batch_idx=req.decode_batch_idx,
            pending_token=self.scheduler.future_map.output_tokens_buf[self.row].reshape(-1).cpu().tolist())

    def after_forward(self, batch):
        if self.prefill_seen:
            return  # Actual Decode observations were captured in TpModelWorker.
        self.evidence.update(cuda_executed=True, original_prefill_tokens=256,
            original_prefill_forward_count=1, forward_ct_before_restore=self.scheduler.forward_ct)
        self.prefill_seen = True
        if self.mode == "reference":
            self.evidence["reference_no_restore"] = True
            return
        state_before = self._request_state(batch)
        self.evidence["request_state_before_restore"] = state_before
        def observe(phase, tx):
            if phase == "VERIFIED":
                self.evidence.update(verified=True, new_slots=list(tx.new_slots),
                    slots_disjoint=not bool(set(tx.old_slots) & set(tx.new_slots)), stats=asdict(tx.stats))
            self._event(phase, tx)
        try:
            self.tx = coordinated_restore(self._prepare_restore, self._vote, observe)
            _require(self.scheduler.forward_ct == self.evidence["forward_ct_before_restore"],
                     "stage_b_forward_during_restore")
            self.prefix_slots = self.tx.new_slots
            self.evidence.update(committed=True, restored_kv_tokens=256,
                mapping_version=1, transaction_state=self.tx.state,
                forward_ct_after_restore=self.scheduler.forward_ct)
            state_after = self._request_state(batch)
            self.evidence.update(request_state_after_restore=state_after,
                request_state_unchanged_by_restore=state_before == state_after)
            _require(state_before == state_after, "stage_b_request_state_changed_by_restore")
            self._live(batch)
            self._event("GLOBAL_COMMIT", self.tx)
        except Exception as exc:
            # Only the deliberately injected pre-commit mismatch is recoverable
            # for this negative experiment; all other failures remain fatal.
            if self.mode == "corrupt":
                def expected_abort():
                    _require(str(exc) == "tp_restore_verified_failed" and self.tx is not None and
                             self.tx.state == "ABORTED" and
                             (self.rank != 0 or self.evidence["corruption_detected"]),
                             "stage_b_unexpected_negative_failure")
                    self._live(batch)
                    self.evidence.update(expected_rejection=str(exc),
                        transaction_state="ABORTED", stats=asdict(self.tx.stats),
                        new_slots=list(self.tx.new_slots))
                self._barrier("NEGATIVE_ABORTED", expected_abort)
                return  # max_new_tokens=1; normal result cleanup frees OLD slots.
            self._fail(exc)
            raise

    def after_result(self, batch):
        if self.done:
            return
        def check():
            outputs = list(self.req.output_ids)
            _require(len(outputs) == len(self.evidence["forward_observations"]),
                     "stage_b_sample_count_mismatch")
            if self.sampled_seed is None:
                self.sampled_seed = outputs[0]
                self.evidence["prefill_seed_token_id"] = self.sampled_seed
            _require(outputs[0] == self.sampled_seed, "stage_b_seed_changed")
            self.evidence["output_token_ids"] = outputs
            if not self.req.finished():
                self._live(batch)
                _require(self.req.prefix_indices.cpu().tolist() == list(self.prefix_slots),
                         "stage_b_chunk_cache_prefix_stale")
                return
            s = self.scheduler
            torch.cuda.synchronize(s.model_worker.model_runner.token_to_kv_pool.k_buffer[0].device)
            a, rp = s.token_to_kv_pool_allocator, s.req_to_token_pool
            free = a.free_pages.cpu().tolist() + a.release_pages.cpu().tolist()
            free_rows = list(rp.free_slots)
            all_rows_released = (len(free_rows) == len(set(free_rows)) == rp.size and
                                 set(free_rows) == set(range(1, rp._alloc_size)))
            no_leak = (len(free) == len(set(free)) == a.size and
                set(free) == set(range(1, a.size + 1)) and a.is_not_in_free_group and
                not a.free_group and not getattr(a, "_kv_ec_restore_active", False) and
                not getattr(rp, "_kv_ec_row_owners", {}) and all_rows_released and
                a.slot_lease_registry.active_lease_count == 0)
            generation = int(rp.req_generation[self.row])
            released = self.req.req_pool_idx is None and self.row in rp.free_slots
            self.evidence.update(generation_after=generation,
                generation_unchanged=generation == self.generation, row_released=released,
                no_slot_leak=no_leak, allocator_size=a.size, free_slots_after=len(free),
                all_rows_released=all_rows_released, request_pool_size=rp.size,
                free_rows_after=len(free_rows),
                request_finished=True, resumed_generated_tokens=max(0, len(outputs) - 1),
                output_token_ids_sha256=token_digest(outputs))
            _require(no_leak and released and generation == self.generation and
                     len(outputs) == self.output_token_limit and
                     self.req.extend_batch_idx == 1 and
                     self.scheduler.forward_ct - self.forward_start == self.output_token_limit and
                     self.evidence["resume_decode_forward_count"] ==
                     (0 if self.mode == "corrupt" else self.decode_steps),
                     "stage_b_cleanup_or_forward_count_mismatch")
        # Include sampled tokens in the collective identity; peers cannot proceed
        # to another Decode if sampling or result processing diverged on one rank.
        digest = token_digest(self.req.output_ids)
        self._barrier("RESULT_" + digest, check)
        if self.req.finished():
            self.evidence["pass_reference"] = self.mode == "reference"
            self.evidence["negative_test_pass"] = self.mode == "corrupt"
            # Token comparison is performed by the independent CLI checker.
            # A local structural completion is NEVER a Stage B PASS by itself.
            self.evidence["local_resume_complete"] = self.mode == "resume"
            self._event("COMPLETE", self.tx)
            self._save()
            self.events.close()
            self.done = True
