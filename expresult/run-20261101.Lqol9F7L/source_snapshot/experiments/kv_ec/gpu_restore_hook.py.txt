"""Stage A only: blocking, single-forward GPU writeback in each TP scheduler.

There is deliberately no decode/resume protocol. The admitted request prefills
256 real tokens normally and samples ONE output. We replace its KV before
result processing, then let the normal finished-request path release the row
and the new slots. No second model forward can occur during this experiment.
"""

from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .gpu_kv_restore import (
    LocalCheckpointSource, RestoreRejected, _require, restore_to_gpu,
)
from .node_block_export import sha, unpack
from .online_chunk import validate_runtime


def coordinated_restore(prepare, vote, observe=lambda phase, tx: None):
    """Fixed-order all-TP-rank prepare / map / release / finish votes.

    All ranks keep the allocator undo journal while any peer could fail its
    mapping or old-slot release. A negative vote compensates even a locally
    committed rank. Timeouts/crashes fail closed; no forward is authorized.
    This is in-process compensation, not durable distributed crash recovery.
    """
    tx = None
    try:
        for phase in ("VERIFIED", "MAPPED", "COMMITTED", "FINISHED"):
            error = None
            try:
                if phase == "VERIFIED":
                    tx = prepare()
                    _require(tx.state == "VERIFIED", "local_restore_not_verified")
                elif phase == "MAPPED":
                    tx.install_mapping()
                elif phase == "COMMITTED":
                    tx.commit(retain_rollback=True)
                else:
                    tx.finish()
                observe(phase, tx)
            except Exception as exc:
                error = exc
            unanimous = vote(phase, error is None)
            if error is not None or not unanimous:
                raise RestoreRejected("tp_restore_" + phase.lower() + "_failed") from error
        return tx
    except BaseException:
        if tx is not None and tx.state != "ABORTED":
            # After FINISHED a rank may already have dropped its journal. That
            # state is quarantined by terminating the experiment, never decoded.
            if tx.state != "COMMITTED" or getattr(tx.allocator, "_kv_ec_restore_active", False):
                tx.abort()
        raise


def validate_checkpoint(block):
    meta, payloads = unpack(block)  # fixtures never accepted
    _require(len(meta['ranks']) in (2, 4) and len(payloads) == len(meta['ranks']),
             "stage_a_requires_complete_tp2_or_tp4_checkpoint")
    for rank in meta["ranks"]:
        _require(rank["token_start"] == 0 and rank["token_count"] == 256,
                 "stage_a_requires_first_256_token_chunk")
        _require(rank.get("chunk_id") == 0 and rank.get("lifetime") ==
                 "non_overlap_before_result_free_independent_gpu_staging",
                 "stage_a_requires_real_online_checkpoint")
    return meta


class GPURestoreHook:
    output_token_limit = 1
    input_requirement_reason = "stage_a_requires_plain_256_token_one_output_request"

    def __init__(self, scheduler, checkpoint, output, run_id, timeout=120, *,
                 source=None, source_name=None):
        validate_runtime(scheduler)
        _require(not any(os.environ.get(k) == "1" for k in (
            "SGLANG_KV_EC_ONLINE", "SGLANG_KV_EC_PAUSE_RESUME", "SGLANG_KV_EC_DEBUG")),
            "restore_conflicting_experiment")
        _require(torch.cuda.is_available(), "cuda_required_for_stage_a")
        self.tp_size = int(scheduler.model_worker.ps.tp_size)
        _require(self.tp_size in (2, 4), "stage_a_requires_tp2_or_tp4")
        _require(torch.distributed.get_world_size(scheduler.tp_cpu_group) == self.tp_size,
                 "stage_a_requires_complete_tp_cpu_group")
        self.scheduler = scheduler
        self.rank = int(scheduler.model_worker.ps.tp_rank)
        self.source = source if source is not None else LocalCheckpointSource(Path(checkpoint))
        self.source_name = source_name if source_name is not None else str(checkpoint)
        block = self.source.load()
        self.block_hash = sha(block)
        self.meta = validate_checkpoint(block)
        _require(len(self.meta['ranks']) == self.tp_size, 'checkpoint_tp_size_mismatch')
        if source is None:
            self.source = LocalCheckpointSource(Path(checkpoint), self.block_hash)
        self.output, self.run_id, self.timeout = Path(output), run_id, timeout
        _require(self.output.is_absolute() and self.output.is_dir() and bool(run_id)
                 and timeout > 0, "stage_a_output_run_or_timeout_invalid")
        self.events = (self.output / f"rank_{self.rank}.events.jsonl").open("x")
        self.admitted = False
        self.tx = None
        self.done = False
        self.deadline = None
        self.evidence = dict(
            schema="kv-gpu-restore-stage-a-v1", run_id=run_id, tp_rank=self.rank,
            tp_size=self.tp_size, checkpoint_sha256=self.block_hash, checkpoint=str(checkpoint),
            cuda_executed=False, verified=False, committed=False, pass_stage_a=False,
            failure_reason=None, decode_forward_count=0,
            restore_forward_count=0, original_prefill_tokens=0,
        )
        run_file = self.output / 'run.json'
        if run_file.is_file():
            config = json.loads(run_file.read_text())
            if config.get('rank_gpu_binding_required'):
                from .gpu_admission import runtime_rank_binding
                self.evidence['rank_gpu_binding'] = runtime_rank_binding(
                    scheduler, config['gpu_resources_before_start'])

    @classmethod
    def from_env(cls, scheduler):
        return cls(scheduler, os.environ["SGLANG_KV_EC_GPU_RESTORE_CHECKPOINT"],
                   os.environ["SGLANG_KV_EC_GPU_RESTORE_OUTPUT"],
                   os.environ["SGLANG_KV_EC_GPU_RESTORE_RUN_ID"],
                   float(os.environ.get("SGLANG_KV_EC_GPU_RESTORE_TIMEOUT", "120")))

    def _event(self, phase, tx=None):
        self.events.write(json.dumps(dict(
            phase=phase, tp_rank=self.rank, run_id=self.run_id,
            transaction_state=tx.state if tx else None,
            written_bytes=tx.stats.written_bytes if tx else 0,
            failure_reason=self.evidence.get("failure_reason"),
            monotonic=time.monotonic())) + "\n")
        self.events.flush()

    def _vote(self, phase, ready):
        remaining = self.deadline - time.monotonic()
        _require(remaining > 0, "stage_a_total_deadline")
        identity = dict(phase=phase, run_id=self.run_id, checkpoint=self.block_hash,
                        token_ids_sha256=self.meta["ranks"][0]["token_ids_sha256"],
                        row=self.evidence.get("row"), generation=self.evidence.get("generation_before"))
        if getattr(self, "rs_source_binding", None) is not None:
            identity["rs_source_binding"] = self.rs_source_binding
        if getattr(self, "request_state_binding", None) is not None:
            identity["request_state_binding"] = self.request_state_binding
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()
        value = torch.tensor([int(ready), *digest], dtype=torch.int32)
        gathered = [torch.empty_like(value) for _ in range(self.tp_size)]
        work = torch.distributed.all_gather(gathered, value,
            group=self.scheduler.tp_cpu_group, async_op=True)
        if work.wait(timeout=timedelta(seconds=remaining)) is False:
            raise TimeoutError("stage_a_tp_vote_timeout:" + phase)
        result = (all(int(v[0]) == 1 for v in gathered)
                  and all(torch.equal(gathered[0][1:], v[1:]) for v in gathered[1:]))
        self._event("GLOBAL_" + phase + ("_OK" if result else "_FAILED"))
        return result

    def _save(self):
        with (self.output / f"rank_{self.rank}.json").open("x") as handle:
            json.dump(self.evidence, handle, indent=2)

    def _fail(self, exc):
        causes = []
        current = exc
        while current is not None and len(causes) < 8:
            causes.append(f"{type(current).__name__}:{current}")
            if hasattr(current, "restore_stats"):
                self.evidence["stats"] = asdict(current.restore_stats)
            current = current.__cause__ or current.__context__
        self.evidence.update(failure_reason=f"{type(exc).__name__}:{exc}", pass_stage_a=False)
        self.evidence["failure_chain"] = causes
        if self.tx is not None:
            self.evidence["transaction_state"] = self.tx.state
            self.evidence["stats"] = asdict(self.tx.stats)
        if not self.events.closed:
            self._event("FAILED", self.tx)
        if (self.output / f"rank_{self.rank}.json").exists():
            with (self.output / f"rank_{self.rank}.failure.json").open("x") as handle:
                json.dump(self.evidence, handle, indent=2)
        else:
            self._save()
        # Propagate to the scheduler failure path. Never return to Decode.

    def before_inputs(self, inputs):
        # Synchronous scheduler-only barrier: no inputs are processed until this
        # returns. FreezeGCReq is sent even with --skip-server-warmup; its
        # scheduler handler only freezes Python GC and forwards to detokenizer.
        # Neither it nor health checks change request rows or KV ownership.
        # All other control RPCs remain denied.
        input_context = None
        try:
            for message in inputs:
                name = type(message).__name__
                input_context = dict(message_type=name, admitted=self.admitted)
                if name in ("HealthCheckReqInput", "FreezeGCReq"):
                    continue
                _require(name == "TokenizedGenerateReqInput",
                         "stage_a_unsupported_message_type:" + name)
                _require(not self.admitted, "stage_a_duplicate_generate_request")
                _require(len(message.input_ids) == 256 and
                         message.sampling_params.max_new_tokens == self.output_token_limit and
                         message.session_id is None and message.mm_inputs is None and
                         message.input_embeds is None,
                         self.input_requirement_reason)
                token_hash = hashlib.sha256(np.asarray(message.input_ids, dtype="<i8").tobytes()).hexdigest()
                _require(token_hash == self.meta["ranks"][0]["token_ids_sha256"],
                         "stage_a_checkpoint_request_token_mismatch")
                self.admitted = True
                self.evidence["token_ids_sha256"] = token_hash
        except Exception as exc:
            if input_context is not None:
                self.evidence["rejected_input"] = input_context
            self._fail(exc)
            raise

    def after_inputs(self):
        """Experimental cold admission overrides this; A/B/C are unchanged."""

    def before_forward(self, batch):
        """Stage A needs no pre-forward work; Stage B overrides this barrier."""

    def after_forward(self, batch):
        self.deadline = time.monotonic() + self.timeout
        req_holder = []

        def prepare():
            _require(self.admitted and not self.done and self.tx is None and
                     len(batch.reqs) == 1, "stage_a_unexpected_batch")
            req = batch.reqs[0]
            req_holder.append(req)
            _require(batch.forward_mode.name == "EXTEND" and batch.prefix_lens == [0]
                     and int(batch.seq_lens_cpu[0]) == req.kv_committed_len == 256
                     and len(req.origin_input_ids) == 256 and not req.output_ids
                     and req.sampling_params.max_new_tokens == 1
                     and req.extend_batch_idx == 1
                     and not req.finished() and not req.is_retracted
                     and not getattr(req, "to_finish", None), "stage_a_invalid_forward_boundary")
            _require(sha(self.source.load()) == self.block_hash, "checkpoint_changed_after_startup")
            self.evidence["original_prefill_tokens"] = 256
            self.evidence["original_prefill_forward_count"] = req.extend_batch_idx
            self.forward_ct = self.scheduler.forward_ct
            self.evidence["forward_ct_before_restore"] = self.forward_ct
            # Hash actual model bytes (prepared at launcher startup). Source
            # commit/diff may differ from historical online capture after fixes;
            # retain both provenances, require identical weights/configuration.
            from .debug_capture import source_identity
            identity = source_identity(self.scheduler.server_args.model_path)
            _require(identity["model_sha256"] == self.meta["ranks"][self.rank]["model_sha256"],
                     "stage_a_checkpoint_model_mismatch")
            self.evidence["runtime_identity"] = identity
            self.tx = restore_to_gpu(self.scheduler, req, self.source,
                                     source_name=self.source_name)
            return self.tx

        def observe(phase, tx):
            if phase == "VERIFIED":
                self.evidence.update(cuda_executed=True, verified=True,
                    row=tx.row, generation_before=tx.generation,
                    old_slots=list(tx.old_slots), new_slots=list(tx.new_slots),
                    slots_disjoint=not bool(set(tx.old_slots) & set(tx.new_slots)),
                    stats=asdict(tx.stats))
            self._event(phase, tx)

        try:
            self.tx = coordinated_restore(prepare, self._vote, observe)
            self.evidence["restore_forward_count"] = self.scheduler.forward_ct - self.forward_ct
            _require(self.scheduler.forward_ct == self.forward_ct,
                     "stage_a_forward_during_restore")
            matches = tuple(self.tx._mapping().cpu().tolist()) == self.tx.new_slots
            _require(matches, "stage_a_committed_mapping_mismatch")
            self.evidence.update(committed=True, transaction_state=self.tx.state,
                                 mapping_matches_new=matches,
                                 forward_ct_after_restore=self.scheduler.forward_ct)
            self.req = req_holder[0]
            self.generation = self.tx.generation
            self.row = self.tx.row
            self._event("GLOBAL_COMMIT", self.tx)
        except Exception as exc:
            self._fail(exc)
            raise

    def after_result(self, batch):
        if self.tx is None or self.done:
            return
        try:
            s, tx = self.scheduler, self.tx
            torch.cuda.synchronize(s.model_worker.model_runner.token_to_kv_pool.k_buffer[0].device)
            a, rp = s.token_to_kv_pool_allocator, s.req_to_token_pool
            free = a.free_pages.cpu().tolist() + a.release_pages.cpu().tolist()
            generation = int(rp.req_generation[self.row])
            row_released = self.req.req_pool_idx is None and self.row in rp.free_slots
            no_leak = (len(free) == len(set(free)) == a.size and
                       set(free) == set(range(1, a.size + 1)) and
                       a.is_not_in_free_group and not a.free_group and
                       not a._kv_ec_restore_active and not getattr(rp, "_kv_ec_row_owners", {}))
            self.evidence.update(generation_after=generation,
                generation_unchanged=generation == self.generation,
                row_released=row_released, allocator_size=a.size,
                free_slots_after=len(free), no_slot_leak=no_leak,
                request_finished=self.req.finished(),
                original_sampled_output_tokens=len(self.req.output_ids))
            local_ok = (generation == self.generation and row_released and no_leak
                        and self.req.finished() and len(self.req.output_ids) == 1
                        and s.forward_ct == self.forward_ct)
            global_ok = self._vote("CLEANUP", local_ok)
            _require(local_ok and global_ok, "stage_a_cleanup_failed")
            self.evidence["pass_stage_a"] = True
            self._event("PASS", tx)
            self._save()
            self.events.close()
            self.done = True
        except Exception as exc:
            self._fail(exc)
            raise
