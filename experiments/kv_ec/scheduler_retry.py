"""Default-off, one-request Qwen3 TP4 prompt replay in the existing pause gate.

Hashes are verification oracles only. No original KV bytes, parity or rollback
are retained for rebuilding. All allocation/free operations use normal APIs.
"""

import hashlib
import os
import time

import torch

from .scheduler_recovery_guard import read_scheduler_mapping
from .scheduler_shadow import (
    SchedulerShadowRecovery, checked_buffers, pack_chunk, zero_mapped_bytes,
)


class SchedulerPrefillRetry:
    evidence_key = "prefill_retry"
    online = False
    protects_prefill = False
    modified = False
    done = False

    def __init__(self, *, lost_rank=0, recover_tokens=256):
        if type(lost_rank) is not int or lost_rank not in range(4) or recover_tokens != 256:
            raise ValueError("retry_initial_scope_rank_0_to_3_and_256_tokens")
        self.lost_rank, self.recover_tokens = lost_rank, recover_tokens
        self.event = None
        self.fault_monotonic = None
        self.started = False

    @classmethod
    def from_env(cls):
        return cls(lost_rank=int(os.environ.get("SGLANG_KV_EC_LOST_RANK", "0")),
                   recover_tokens=int(os.environ.get("SGLANG_KV_EC_RECOVER_TOKENS", "256")))

    def check_admission(self, gate):
        # Reuse the model/source/Qwen3-8B/graph restrictions without creating
        # a parity client or a second recovery/protection state machine.
        SchedulerShadowRecovery.check_online_admission(self, gate)

    def validate(self, gate):
        if self.event is not None and not self.event.query():
            raise ValueError("retry_cuda_event_incomplete")

    def _fence(self, gate):
        self.event = torch.cuda.Event()
        self.event.record(torch.cuda.current_stream())
        while not self.event.query():
            self._remaining(gate)
            time.sleep(.001)
        gate.scheduler.token_to_kv_pool_allocator.attach_slot_lease_cuda_event(
            gate.owner, self.event)
        gate.event = self.event

    def _remaining(self, gate):
        if time.monotonic() >= gate.deadline:
            raise TimeoutError("retry_total_deadline")

    def _wait_control(self, gate, key):
        while True:
            status = gate._call("status")
            if status[key] == list(range(4)):
                return
            self._remaining(gate)
            time.sleep(.01)

    def _hash_prompt(self, gate):
        pool = gate.scheduler.model_worker.model_runner.token_to_kv_pool
        hashes = []
        for start in (0, 256, 512):
            payload, meta = pack_chunk(
                pool, gate.mapping, gate.req, gate.model_identity,
                snapshot_id=gate.identity["transaction_id"] + f"-oracle{start}",
                manager_epoch=gate.identity["epoch"], token_start=start)
            # The payload is deliberately discarded before injection/replay.
            hashes.append({key: meta[key] for key in (
                "sha256", "k_sha256", "v_sha256", "token_ids_sha256", "input_ids_sha256")})
            del payload
        return hashes

    def _check_history(self, req, history, stream_state):
        if tuple(req.output_ids) != history or any(
                getattr(req, key, None) != value for key, value in stream_state.items()):
            raise ValueError("retry_changed_output_history_or_stream_offsets")

    def _release_old(self, gate):
        """Only the all-rank handoff permits ending the old row/slot lease."""
        s = gate.scheduler
        status = gate._call("status")
        if status.get("retry_ready_ranks") != list(range(4)):
            raise ValueError("retry_release_before_four_rank_handoff")
        gate._revalidate()
        if not gate._vote(True):
            raise ValueError("retry_handoff_vote_failed")
        gate.state = "RETRY_PENDING"
        s.token_to_kv_pool_allocator.release_slot_lease(
            gate.owner, manager_epoch=gate.identity["epoch"])
        gate.released = True
        s.req_to_token_pool.release_kv_ec_row(
            gate.mapping.req_pool_idx, gate.mapping.req_generation, gate.owner)
        # No rank frees or reallocates until all four have released safely.
        if not gate._vote(True):
            raise ValueError("retry_release_vote_failed")
        gate._call("retry_released")
        self._wait_control(gate, "retry_released_ranks")

    def _rebuild(self, gate):
        from sglang.srt.managers.schedule_batch import ScheduleBatch, release_req

        s, req = gate.scheduler, gate.req
        if s.last_batch is None or s.last_batch.reqs != [req] or not s.running_batch.is_empty():
            raise ValueError("retry_requires_exclusive_prefill_last_batch")
        stage = time.monotonic()
        release_req(
            req=req, remaing_req_count=0, server_args=s.server_args,
            req_to_token_pool=s.req_to_token_pool,
            token_to_kv_pool_allocator=s.token_to_kv_pool_allocator,
            tree_cache=s.tree_cache, hisparse_coordinator=None, offload_kv=False)
        if req.req_pool_idx is not None or req.kv is not None or not req.is_retracted:
            raise ValueError("retry_normal_release_incomplete")
        released_at = time.monotonic()
        req.init_next_round_input(s.tree_cache)
        if len(req.prefix_indices) != 0:
            raise ValueError("retry_forbids_prefix_reuse")
        # Keep output_ids intact. Recompute only the original 768-token prompt;
        # the replayed first sample is verified and NOT processed or streamed.
        # The subsequent normal decode consumes the retained first token once.
        req.set_extend_range(0, 768)
        batch = ScheduleBatch.init_new(
            [req], s.req_to_token_pool, s.token_to_kv_pool_allocator,
            s.tree_cache, s.model_config, s.enable_overlap, s.spec_algorithm)
        batch.prepare_for_extend()
        if batch.prefix_lens != [0] or batch.extend_num_tokens != 768:
            raise ValueError("retry_must_recompute_exact_prompt")
        old = gate.mapping
        new = read_scheduler_mapping(s, req, token_start=0, token_count=768)
        if ((new.req_pool_idx, new.req_generation) == (old.req_pool_idx, old.req_generation)
                or (new.physical_slots, new.slot_generations) ==
                   (old.physical_slots, old.slot_generations)):
            raise ValueError("retry_requires_new_row_and_slot_allocation_identity")
        gate.mapping = new
        s.req_to_token_pool.protect_kv_ec_row(new.req_pool_idx, new.req_generation, gate.owner)
        s.token_to_kv_pool_allocator.begin_slot_lease(
            gate.owner, manager_epoch=gate.identity["epoch"], request_id=req.rid,
            request_version=new.request_version, slots=list(new.physical_slots))
        gate.released = False
        self._fence(gate)
        gate._revalidate()
        if not gate._vote(True):
            raise ValueError("retry_new_allocation_vote_failed")
        allocated_at = time.monotonic()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        result = s.run_batch(batch)
        end.record(torch.cuda.current_stream())
        self._fence(gate)
        forward_done = time.monotonic()
        if (not result.has_sampled_token_ids or result.delay_sample_func is not None
                or result.next_token_ids.detach().cpu().tolist() != list(req.output_ids)):
            raise ValueError("retry_replayed_first_token_mismatch")
        # Do not call process_batch_result: that would append/stream token 0
        # twice. run_batch has already republished it through the normal relay.
        s.tree_cache.cache_unfinished_req(req)
        s.last_batch = batch
        gate.state = "SHADOW_READY"
        gate._revalidate()
        return dict(
            normal_release_ms=(released_at - stage) * 1000,
            reallocation_ms=(allocated_at - released_at) * 1000,
            prefill_forward_ms=start.elapsed_time(end),
            prefill_forward_host_ms=(forward_done - allocated_at) * 1000,
            rebuilt_forward_done_monotonic=forward_done,
            original_request_version=old.request_version,
            rebuilt_request_version=new.request_version,
            rebuilt_row=new.req_pool_idx, rebuilt_request_generation=new.req_generation,
            rebuilt_slot_generations=list(new.slot_generations))

    def run(self, gate):
        if self.started or self.done:
            raise ValueError("retry_duplicate_recovery")
        self.started = True
        self.check_admission(gate)
        gate._revalidate()
        s, req = gate.scheduler, gate.req
        pool = s.model_worker.model_runner.token_to_kv_pool
        if not torch.cuda.is_available() or pool.k_buffer[0].device.type != "cuda":
            raise ValueError("retry_requires_real_cuda")
        if (len(req.origin_input_ids) != 768 or len(req.output_ids) != 1
                or req.sampling_params.max_new_tokens != 16 or req.sampling_params.top_k != 1
                or req.input_embeds is not None or req.grammar is not None
                or req.return_logprob or req.return_hidden_states
                or not s.tree_cache.is_chunk_cache()):
            raise ValueError("retry_unsupported_request")
        history = tuple(req.output_ids)
        stream_state = {k: getattr(req, k) for k in (
            "send_token_offset", "send_decode_id_offset", "send_output_token_logprobs_offset",
            "send_output_sampling_mask_offset", "surr_offset", "read_offset")}
        identity_hash = hashlib.sha256(req.rid.encode()).hexdigest()
        with torch.cuda.stream(s.forward_stream):
            s.forward_stream.wait_event(gate.event)
            before = self._hash_prompt(gate)
            slots = torch.tensor(gate.mapping.physical_slots[:256],
                                 device=pool.k_buffer[0].device, dtype=torch.int64)
            gate._revalidate()
            if not gate._vote(True):
                raise ValueError("retry_prefault_vote_failed")
            # Broadcast the target clock, not a client TTFT, on this one node.
            clock = torch.zeros(1, dtype=torch.float64)
            changed = False
            if gate.rank == self.lost_rank:
                clock[0] = time.monotonic()
                self.modified = True
                zero_mapped_bytes(pool, slots)
                self._fence(gate)
                for layer in range(36):
                    if any(torch.count_nonzero(t.index_select(0, slots)).item()
                           for t in checked_buffers(pool, layer, slots)):
                        raise ValueError("retry_fault_clear_incomplete")
                _, cleared = pack_chunk(
                    pool, gate.mapping, req, gate.model_identity,
                    snapshot_id=gate.identity["transaction_id"] + "-cleared",
                    manager_epoch=gate.identity["epoch"])
                changed = cleared["sha256"] != before[0]["sha256"]
                if not changed:
                    raise ValueError("retry_fault_did_not_change_bytes")
            work = torch.distributed.broadcast(
                clock, src=self.lost_rank, group=s.tp_cpu_group, async_op=True)
            from datetime import timedelta
            if work.wait(timeout=timedelta(seconds=max(.001, gate.deadline - time.monotonic()))) is False:
                raise TimeoutError("retry_fault_clock_vote_timeout")
            self.fault_monotonic = float(clock.item())
            gate._call("retry_ready", observation=dict(
                request_hash=identity_hash, request_version=gate.mapping.request_version,
                input_ids_sha256=before[0]["input_ids_sha256"],
                event_complete=gate.event.query(), lease_validated=True,
                fault_injected=gate.rank == self.lost_rank, fault_bytes_changed=changed))
            self._wait_control(gate, "retry_ready_ranks")
        # Normal allocator preparation runs on the scheduler stream; the
        # complete producer fence above makes the ownership handoff safe.
        self._release_old(gate)
        result = self._rebuild(gate)
        self._check_history(req, history, stream_state)
        with torch.cuda.stream(s.forward_stream):
            s.forward_stream.wait_event(gate.event)
            after = self._hash_prompt(gate)
            if before != after:
                raise ValueError("retry_rebuilt_prompt_not_byte_exact")
            self._fence(gate)
        gate._revalidate()
        if not gate._vote(True):
            raise ValueError("retry_rebuilt_validation_vote_failed")
        verified = time.monotonic()
        result.update(
            mode="prefill_retry", tp_rank=gate.rank, device=str(pool.k_buffer[0].device),
            fault_injected=gate.rank == self.lost_rank, fault_bytes_changed=changed,
            lost_rank=self.lost_rank, recover_tokens=256,
            fault_monotonic=self.fault_monotonic, rebuilt_state_monotonic=verified,
            fault_to_rebuilt_state_ms=(verified - self.fault_monotonic) * 1000,
            prompt_hashes_before=before, prompt_hashes_after=after,
            output_count_before=len(history), output_count_after=len(req.output_ids),
            stream_offsets_before=stream_state,
            stream_offsets_after={k: getattr(req, k) for k in stream_state},
            consensus=dict(
                request_hash=identity_hash, original_request_version=result["original_request_version"],
                input_ids_sha256=before[0]["input_ids_sha256"],
                byte_exact=True, retry_rebuilt=True, history_preserved=True,
                fresh_allocation=True, new_lease_validated=True,
                used_committed_parity=False, rollback_used_for_recovery=False,
                duplicate_output_emitted=False, fault_boundary_output_count=1,
                rebuilt_prefix_tokens=768))
        gate.evidence.update(retry_handoff_complete=True)
        return result
