"""Default-off event-loop gate. No KV reads/writes or model collectives.

The scheduler retains last_batch/running_batch unchanged while this gate is
active. Local failure is fail-stop with row/slot protections retained.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from datetime import timedelta

import torch

from .cross_rank_control import request
from .debug_capture import source_identity
from .scheduler_recovery_guard import read_scheduler_mapping


class PauseRejected(RuntimeError):
    pass


class SchedulerPauseGate:
    def __init__(self, scheduler, *, target, identity, socket, output, timeout_s=30,
                 ipc_timeout_s=0.5):
        if (not target or not identity.get("epoch") or not identity.get("transaction_id")
                or not math.isfinite(timeout_s) or timeout_s <= 0
                or not math.isfinite(ipc_timeout_s) or ipc_timeout_s <= 0):
            raise ValueError("invalid pause identity/deadline")
        if not Path(socket).is_absolute() or not Path(output).is_absolute():
            raise ValueError("pause socket/output must be absolute")
        self.scheduler = scheduler
        self.target, self.identity = target, dict(identity)
        self.socket, self.output = str(socket), Path(output)
        self.timeout_s, self.ipc_timeout_s = timeout_s, ipc_timeout_s
        self.state = "IDLE"
        self.req = self.mapping = self.event = None
        self.deadline = None
        self.next_poll = 0
        self.trace = []
        self.evidence = {}
        self.released = False
        self.rank = int(scheduler.model_worker.ps.tp_rank)
        self.owner = f"{identity['transaction_id']}:rank{self.rank}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / f"pause-rank-{self.rank}.json"
        self.pending_path = self.output / f"pause-rank-{self.rank}.pending.json"
        if self.path.exists():
            raise FileExistsError(self.path)

    @classmethod
    def from_env(cls, scheduler):
        # Reject unsupported execution loops at startup, before any request.
        args = scheduler.server_args
        if (not args.disable_overlap_schedule or getattr(args, "pp_size", 1) != 1
                or getattr(args, "disaggregation_mode", "null") != "null"):
            raise PauseRejected("pause_requires_normal_nonoverlap_scheduler")
        gate = cls(
            scheduler, target=os.environ["SGLANG_KV_EC_PAUSE_REQUEST_ID"],
            identity=json.loads(os.environ["SGLANG_KV_EC_PAUSE_IDENTITY"]),
            socket=os.environ["SGLANG_KV_EC_PAUSE_SOCKET"],
            output=os.environ["SGLANG_KV_EC_PAUSE_OUTPUT"],
            timeout_s=float(os.environ.get("SGLANG_KV_EC_PAUSE_TIMEOUT_S", "30")),
            ipc_timeout_s=float(os.environ.get("SGLANG_KV_EC_PAUSE_IPC_TIMEOUT_S", "0.5")),
        )
        # Fingerprint work is startup-only, not inside the pause deadline.
        gate.model_identity = source_identity(args.model_path)
        return gate

    @property
    def active(self):
        return self.state not in ("IDLE", "RESUMED")

    def _call(self, op, **values):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("pause_total_deadline")
        started = time.monotonic()
        message = dict(self.identity, op=op, tp_rank=self.rank, **values)
        response = request(self.socket, message,
                           timeout=min(self.ipc_timeout_s, remaining),
                           deadline=min(self.deadline, started + self.ipc_timeout_s))
        self.trace.append({
            "monotonic": started, "operation": op, "tp_rank": self.rank,
            "elapsed_ms": (time.monotonic() - started) * 1000,
            "ipc": response.get("_control_ipc_detail"),
        })
        if time.monotonic() >= self.deadline:
            raise TimeoutError("pause_total_deadline")
        if response.get("state") != "OK":
            raise PauseRejected(f"pause_control_rejected:{response.get('reason')}")
        status = response["status"]
        for key in ("epoch", "transaction_id"):
            if status.get(key) != self.identity[key]:
                raise PauseRejected("pause_coordinator_restarted")
        if status["state"] == "FAILED":
            raise PauseRejected(f"pause_failed:{status.get('failure_reason')}")
        return status

    def _fatal(self, exc):
        self.state = "ABORTED_FAIL_CLOSED"
        allocator = self.scheduler.token_to_kv_pool_allocator
        allocator.slot_lease_registry.close()
        failure = {"exception_type": type(exc).__name__, "exception_message": str(exc),
                   "traceback": traceback.format_exc(), "monotonic": time.monotonic()}
        self.trace.append(failure)
        try:
            if self.deadline and self.deadline > time.monotonic():
                self._call("fail", reason=f"{type(exc).__name__}:{exc}")
        except Exception:
            pass
        self._write_evidence(failure)
        raise PauseRejected(f"pause_fail_stop:{exc}") from exc

    def _write_evidence(self, failure=None, path=None):
        record = dict(
            self.evidence, state=self.state, failure=failure,
            transaction_id=self.identity["transaction_id"],
            coordinator_epoch=self.identity["epoch"], tp_rank=self.rank,
            trace=self.trace, online_kv_pool_modified=False,
        )
        # One immutable final record; logs never contain actual KV bytes.
        path = path or (self.path if failure is None else self.path.with_suffix(".error.json"))
        with path.open("x") as stream:
            json.dump(record, stream, indent=2)

    def check_inputs(self, messages):
        if not self.active:
            return
        # AbortReq is processed by the real scheduler, setting to_finish on
        # running/last_batch requests. The next tick fails before decode.
        if any(type(message).__name__ != "AbortReq" for message in messages):
            self._fatal(PauseRejected("control_mutation_while_paused"))

    def after_result(self, batch):
        if (
            self.state == "RESUMED"
            and batch is not None
            and batch.forward_mode.is_decode()
            and any(req.rid == self.target for req in batch.reqs)
        ):
            # RESUMED only means that all ranks were authorized to re-enter
            # scheduling. The success record is emitted only after the next
            # real decode result has completed on this rank.
            self.evidence.update(
                decode_observed=True,
                decode_output_count=len(self.req.output_ids),
                decode_forward_mode="decode",
                online_kv_pool_modified=False,
            )
            self._write_evidence()
            if self.pending_path.exists():
                self.pending_path.unlink()
            return
        if self.state != "IDLE" or batch is None or not batch.forward_mode.is_extend():
            return
        targets = [req for req in batch.reqs if req.rid == self.target]
        if not targets:
            return
        self.deadline = time.monotonic() + self.timeout_s
        self.req = targets[0]
        self.state = "PREPARE"
        try:
            scheduler, req = self.scheduler, self.req
            if req.finished() or req.is_retracted or req.to_finish is not None:
                raise PauseRejected("request_finished_or_cancelled")
            # Unlike capture, this is AFTER output append/cache bookkeeping;
            # the next decode allocation has not happened yet.
            if len(batch.reqs) != 1 or scheduler.waiting_queue:
                raise PauseRejected("single_request_only")
            # Check the prefill batch, not req.prefix_indices. ChunkCache may
            # populate req.prefix_indices during result processing even when
            # the request had no cached prefix at admission.
            if list(getattr(batch, "prefix_lens", [0])) != [0]:
                raise PauseRejected("prefill_prefix_not_empty")
            self._check_runtime()
            count = len(req.origin_input_ids)
            if count != 768 or len(req.output_ids) != 1:
                raise PauseRejected("pause_requires_completed_768_prefill")
            initial = read_scheduler_mapping(scheduler, req, token_start=0, token_count=count)
            allocator = scheduler.token_to_kv_pool_allocator
            allocator.enable_slot_leases(timeout_s=self.timeout_s, active_slots=initial.physical_slots)
            self.mapping = read_scheduler_mapping(scheduler, req, token_start=0, token_count=count)
            scheduler.req_to_token_pool.protect_kv_ec_row(
                self.mapping.req_pool_idx, self.mapping.req_generation, self.owner)
            allocator.begin_slot_lease(
                self.owner, manager_epoch=self.identity["epoch"], request_id=req.rid,
                request_version=self.mapping.request_version, slots=list(self.mapping.physical_slots),
            )
            # Normal non-overlap scheduling runs this code under
            # schedule_stream. Fence that producer stream before publishing
            # the allocator event on forward_stream.
            if torch.cuda.is_available():
                producer_stream = torch.cuda.current_stream()
                producer_event = scheduler.device_module.Event()
                producer_event.record(stream=producer_stream)
                scheduler.forward_stream.wait_event(producer_event)
            self.event = allocator.record_slot_lease_cuda_event(
                self.owner, stream=scheduler.forward_stream)
            self.state = "EVENT_PENDING"
            self.started = time.monotonic()
            self.evidence = {
                "request_hash": hashlib.sha256(req.rid.encode()).hexdigest(),
                "request_generation_before": self.mapping.req_generation,
                "row": self.mapping.req_pool_idx, "token_count": count,
                "slot_generations_before": self.mapping.slot_generations,
                "model_identity": self.model_identity,
                "cuda_event_stream": "schedule_stream->scheduler.forward_stream",
                "cuda_event_recorded": True, "cuda_event_complete": False,
                "scheduler_skipped_iterations": 0,
            }
        except Exception as exc:
            self._fatal(exc)

    def _check_runtime(self):
        s = self.scheduler
        a = s.server_args
        ps = s.model_worker.ps
        runner = s.model_worker.model_runner
        pool = runner.token_to_kv_pool
        # Exact types are important: inherited pools may have offload/layout
        # semantics the prototype does not understand.
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        if (type(pool) is not MHATokenToKVPool
                or type(s.req_to_token_pool) is not ReqToTokenPool
                or type(s.token_to_kv_pool_allocator) is not TokenToKVPoolAllocator):
            raise PauseRejected("plain_pool_allocator_required")
        if (ps.tp_size != 4 or ps.attn_tp_size != 4
                or ps.attn_cp_size != 1 or ps.attn_dcp_size != 1):
            raise PauseRejected("only_plain_tp4")
        for name in ("pp_size", "dp_size", "dcp_size", "nnodes"):
            if getattr(a, name, 1) != 1:
                raise PauseRejected(f"unsupported_parallel:{name}")
        for name in ("enable_dp_attention", "enable_pdmux", "enable_hierarchical_cache",
                     "enable_hisparse", "enable_unified_memory", "enable_lora",
                     "enable_mixed_chunk", "enable_page_major_kv_layout"):
            if getattr(a, name, False):
                raise PauseRejected(f"unsupported_runtime:{name}")
        if (not a.disable_overlap_schedule or not a.disable_radix_cache
                or a.chunked_prefill_size != -1 or a.max_running_requests != 1
                or getattr(a, "disaggregation_mode", "null") != "null"
                or not s.spec_algorithm.is_none() or getattr(a, "cpu_offload_gb", 0)
                or getattr(a, "quantization", None) is not None):
            raise PauseRejected("unsupported_scheduler")
        hf = runner.model_config.hf_config
        if (hf.model_type != "qwen3" or hf.num_hidden_layers != 36
                or hf.num_key_value_heads != 8 or hf.head_dim != 128
                or pool.dtype != torch.bfloat16 or pool.store_dtype != pool.dtype
                or pool.kv_cache_layout != "nhd" or pool.use_hnd
                or pool.page_size != 1 or pool.start_layer != 0 or pool.layer_num != 36
                or pool.is_quantized_kv_cache or pool.post_capture_active
                or pool.layer_transfer_counter is not None
                or runner.req_to_token_pool is not s.req_to_token_pool
                or s.token_to_kv_pool_allocator.get_kvcache() is not pool):
            raise PauseRejected("unsupported_qwen3_pool_layout")
        if (s.enable_overlap or s.enable_pdmux or s.enable_hisparse
                or s._engine_paused or self.req.session_id
                or self.req.multimodal_inputs is not None or len(self.req.prefix_indices)
                or self.req.inflight_middle_chunks):
            raise PauseRejected("unsupported_request_lifetime")

    def _revalidate(self):
        req, s = self.req, self.scheduler
        if req.finished() or req.is_retracted or req.to_finish is not None:
            raise PauseRejected("request_cancelled")
        mapping = read_scheduler_mapping(s, req, token_start=0, token_count=768)
        if mapping != self.mapping:
            raise PauseRejected("request_row_slot_generation_or_mapping_changed")
        if not self.event.query():
            raise PauseRejected("cuda_event_incomplete")
        if not self.released:
            s.token_to_kv_pool_allocator.validate_slot_lease(
                self.owner, manager_epoch=self.identity["epoch"], request_id=req.rid,
                request_version=mapping.request_version, slots=list(mapping.physical_slots))

    def _tick_local(self):
        if not self.active:
            return False
        if self.state == "ABORTED_FAIL_CLOSED":
            raise PauseRejected("failed_pause_cannot_resume")
        if self.state == "RESUME_READY":
            self._revalidate()
            return False
        try:
            if time.monotonic() >= self.deadline:
                raise TimeoutError("pause_total_deadline")
            if self.req.to_finish is not None or self.req.finished() or self.req.is_retracted:
                raise PauseRejected("request_cancelled")
            self.evidence["scheduler_skipped_iterations"] += 1
            now = time.monotonic()
            if now < self.next_poll:
                time.sleep(min(0.01, self.next_poll - now))
                return True
            self.next_poll = now + 0.05
            if self.state == "EVENT_PENDING":
                if not self.event.query():
                    return True
                self.evidence["cuda_event_complete"] = True
                self._revalidate()
                self._call("paused", observation={
                    "event_complete": True, "request_hash": self.evidence["request_hash"],
                    "request_version": self.mapping.request_version, "token_count": 768,
                    "model_identifier": self.model_identity["model_identifier"],
                    "model_sha256": self.model_identity["model_sha256"],
                })
                self.state = "PAUSED"
                return True
            status = self._call("status")
            state = status["state"]
            if self.state == "PAUSED" and state == "RESUME_REQUESTED":
                self._revalidate()
                self._call("validated")
                self.state = "VALIDATED"
            elif self.state == "VALIDATED" and state == "RELEASE_AUTHORIZED":
                self._revalidate()
                self.scheduler.token_to_kv_pool_allocator.release_slot_lease(
                    self.owner, manager_epoch=self.identity["epoch"])
                self.released = True
                self._call("released")
                self.state = "RELEASED"
            elif self.state == "RELEASED" and state == "ALL_RELEASED":
                self._revalidate()
                self._call("armed")
                self.state = "ARMED"
            elif self.state == "ARMED" and state == "RESUME_AUTHORIZED":
                self._revalidate()
                self.evidence["coordinator_final"] = status
                self.state = "RESUME_READY"
                return False
            return True
        except Exception:
            raise

    def _vote(self, ready):
        """Keep TP CPU receiver collectives aligned before any rank decodes.

        Work.wait has a bounded timeout. A missing rank never falls through
        into an attention/model collective with the other three ranks.
        """
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("pause_total_deadline")
        votes = torch.tensor([int(ready)], dtype=torch.int32, device="cpu")
        work = torch.distributed.all_reduce(
            votes, op=torch.distributed.ReduceOp.MIN,
            group=self.scheduler.tp_cpu_group, async_op=True)
        if work.wait(timeout=timedelta(seconds=remaining)) is False:
            raise TimeoutError("pause_rank_vote_timeout")
        return bool(votes.item())

    def tick(self):
        """True skips selection/forward, identically on every TP rank."""
        if self.state in ("IDLE", "RESUMED"):
            return False
        try:
            local_blocked = self._tick_local()
            if not self._vote(not local_blocked):
                return True
            self._revalidate()
            self.scheduler.req_to_token_pool.release_kv_ec_row(
                self.mapping.req_pool_idx, self.mapping.req_generation, self.owner)
            self.evidence.update(
                pause_ms=(time.monotonic() - self.started) * 1000,
                request_generation_after=self.mapping.req_generation,
                slot_generations_after=self.mapping.slot_generations,
                allocator_lease_released=True,
                tp_cpu_resume_vote=True,
            )
            # Row-release failure on one rank must not allow its peers to
            # start decode. Failed Gloo work is fail-stop.
            if not self._vote(True):
                raise PauseRejected("resume_final_vote_failed")
            self.state = "RESUMED"
            self.evidence.update(
                resume_authorized=True,
                decode_observed=False,
                online_kv_pool_modified=False,
            )
            # Authorization is not a successful resume. Wait for the next
            # real decode result before emitting the final evidence file.
            self._write_evidence(path=self.pending_path)
            return False
        except Exception as exc:
            self._fatal(exc)
