"""Same-process TP recovery for the real non-overlap SGLang scheduler.

The hook is intentionally idle until a fault is requested, but the mechanism is
created by the online launcher by default.  A fault may be requested at an
exact KV boundary, with ``SGLANG_KV_EC_FAULT_AFTER_DECODE``, or an atomic JSON
control file.  Recovery
never starts a model process and never sends output while the request is in a
prepared-but-uncommitted state.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time

import torch

from .gpu_kv_restore import (
    BytesCheckpointSource, LocalCheckpointSource, RestoreRejected,
    restore_to_gpu, restore_xor_to_gpu,
)
from .gpu_restore_hook import coordinated_restore
from .node_block_export import unpack, xor, sha
from .cuda_rs_codec import CudaRSCodec

logger = logging.getLogger(__name__)


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RecoveryPoint:
    request_id: str
    delivered_tokens: int
    computed_unsent_tokens: tuple[int, ...]
    next_generation_position: int
    kv_token_boundary: int
    kv_slots: tuple[int, ...]
    req_pool_idx: int
    req_generation: int
    recovery_version: int
    tp_rank: int
    tp_size: int
    created_ns: int

    @classmethod
    def capture(cls, scheduler, req, version: int):
        if req.req_pool_idx is None:
            raise RestoreRejected("live_recovery_request_has_no_row")
        row = int(req.req_pool_idx)
        boundary = int(req.kv_committed_len)
        delivered = int(req.send_token_offset)
        if not (0 <= delivered <= len(req.output_ids)):
            raise RestoreRejected("live_recovery_invalid_output_commit_boundary")
        slots = tuple(
            scheduler.req_to_token_pool.req_to_token[row, :boundary].cpu().tolist()
        )
        if len(slots) != boundary or len(set(slots)) != boundary or any(x <= 0 for x in slots):
            raise RestoreRejected("live_recovery_invalid_kv_mapping")
        return cls(
            request_id=str(req.rid),
            delivered_tokens=delivered,
            computed_unsent_tokens=tuple(req.output_ids[delivered:]),
            next_generation_position=boundary,
            kv_token_boundary=boundary,
            kv_slots=slots,
            req_pool_idx=row,
            req_generation=int(scheduler.req_to_token_pool.req_generation[row]),
            recovery_version=version,
            tp_rank=int(scheduler.model_worker.ps.tp_rank),
            tp_size=int(scheduler.model_worker.ps.tp_size),
            created_ns=time.monotonic_ns(),
        )

    def common_identity(self):
        # Physical slot numbers are rank-local and deliberately excluded.
        return dict(
            request_id=self.request_id,
            delivered_tokens=self.delivered_tokens,
            computed_unsent_tokens=self.computed_unsent_tokens,
            next_generation_position=self.next_generation_position,
            kv_token_boundary=self.kv_token_boundary,
            req_pool_idx=self.req_pool_idx,
            req_generation=self.req_generation,
            recovery_version=self.recovery_version,
            tp_size=self.tp_size,
        )


class HybridPolicy:
    """Same-boundary EWMA selector with a conservative first-sample model."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ewma_ms = {"EC": None, "RE_PREFILL": None}

    def observe(self, path, elapsed_ms):
        if path not in self.ewma_ms or not elapsed_ms > 0:
            raise ValueError("invalid_hybrid_sample")
        old = self.ewma_ms[path]
        self.ewma_ms[path] = elapsed_ms if old is None else self.alpha * elapsed_ms + (1 - self.alpha) * old

    def choose(self, *, ec_available, token_count, kv_bytes, prefill_tokens_per_s=None,
               ec_bytes_per_s=None):
        if not ec_available:
            return "RE_PREFILL", "ec_not_committed"
        ec = self.ewma_ms["EC"]
        rp = self.ewma_ms["RE_PREFILL"]
        if ec is None and ec_bytes_per_s:
            ec = kv_bytes / ec_bytes_per_s * 1000
        if rp is None and prefill_tokens_per_s:
            rp = token_count / prefill_tokens_per_s * 1000
        if ec is not None and rp is not None:
            return ("EC", "same_boundary_estimate") if ec <= rp else ("RE_PREFILL", "same_boundary_estimate")
        # A committed checkpoint is the only path with bounded recovery work
        # before online calibration.  The decision and missing samples are logged.
        return "EC", "uncalibrated_committed_ec"


class LiveRecoveryHook:
    """Scheduler-thread recovery coordinator; model and request stay resident."""

    def __init__(self, scheduler, *, checkpoint: bytes | None, control: Path | None,
                 timeout: float, fault_after_decode: int | None,
                 fault_at_kv_boundary: int | None,
                 xor_parity=None, xor_survivor=None, parity_sha256=None,
                 cuda_codec=None, source=None, hybrid_enabled=True,
                 same_process_switch=True, fused_enabled=True):
        self.scheduler = scheduler
        self.rank = int(scheduler.model_worker.ps.tp_rank)
        self.tp_size = int(scheduler.model_worker.ps.tp_size)
        if self.tp_size != 2:
            raise RestoreRejected("live_recovery_requires_tp2")
        if torch.distributed.get_world_size(scheduler.tp_cpu_group) != self.tp_size:
            raise RestoreRejected("live_recovery_incomplete_tp_group")
        self.source = source or (
            BytesCheckpointSource(checkpoint) if checkpoint is not None else None
        )
        self.control = control
        self.timeout = timeout
        self.fault_after_decode = fault_after_decode
        self.fault_at_kv_boundary = fault_at_kv_boundary
        self.policy = HybridPolicy()
        self.requested = False
        self.completed = False
        self.version = 0
        self.point = None
        self.xor_parity = xor_parity
        self.xor_survivor = xor_survivor
        self.parity_sha256 = parity_sha256
        self.cuda_codec = cuda_codec
        self.hybrid_enabled = bool(hybrid_enabled)
        self.same_process_switch = bool(same_process_switch)
        self.fused_enabled = bool(fused_enabled)
        self.lock = threading.Lock()
        self.events = []
        self.audit = None
        self.selected_action = None
        self.kv_ready_logged = False

    @classmethod
    def from_env(cls, scheduler):
        path = os.environ.get("SGLANG_KV_EC_LIVE_CHECKPOINT")
        hot_memory = os.environ.get("SGLANG_KV_EC_HOT_MEMORY", "1") == "1"
        checkpoint = Path(path).read_bytes() if path and hot_memory else None
        source = (
            BytesCheckpointSource(checkpoint) if checkpoint is not None else
            LocalCheckpointSource(
                Path(path), expected_sha256=sha(Path(path).read_bytes())
            ) if path else None
        )
        xor_parity = xor_survivor = parity_sha256 = cuda_codec = None
        # Each launcher switch changes the selected data path.  The optimized
        # path needs all four capabilities; disabling any one falls back to the
        # separately verified, non-fused checkpoint writeback.
        fused = all(os.environ.get(name, "1") == "1" for name in (
            "SGLANG_KV_EC_FUSED_XOR_WRITE", "SGLANG_KV_EC_STREAMING_RECOVERY",
            "SGLANG_KV_EC_PIPELINE", "SGLANG_KV_EC_CUDA_RS",
        ))
        if checkpoint is not None and fused:
            _, blocks = unpack(checkpoint)
            if len(blocks) != 2:
                raise RestoreRejected("fused_live_recovery_requires_tp2_checkpoint")
            rank = int(scheduler.model_worker.ps.tp_rank)
            xor_parity = xor(*blocks)
            xor_survivor = blocks[1 - rank]
            parity_sha256 = sha(xor_parity)
            device = scheduler.model_worker.model_runner.token_to_kv_pool.k_buffer[0].device
            cuda_codec = CudaRSCodec("cuda", device=device)
        control_raw = os.environ.get("SGLANG_KV_EC_LIVE_CONTROL")
        control = Path(control_raw) if control_raw else None
        fault_raw = os.environ.get("SGLANG_KV_EC_FAULT_AFTER_DECODE")
        boundary_raw = os.environ.get("SGLANG_KV_EC_FAULT_AT_KV_BOUNDARY")
        return cls(
            scheduler,
            checkpoint=checkpoint,
            control=control,
            timeout=float(os.environ.get("SGLANG_KV_EC_LIVE_TIMEOUT", "120")),
            fault_after_decode=int(fault_raw) if fault_raw else None,
            fault_at_kv_boundary=int(boundary_raw) if boundary_raw else None,
            xor_parity=xor_parity, xor_survivor=xor_survivor,
            parity_sha256=parity_sha256, cuda_codec=cuda_codec, source=source,
            hybrid_enabled=os.environ.get("SGLANG_KV_EC_HYBRID", "1") == "1",
            same_process_switch=(
                os.environ.get("SGLANG_KV_EC_SAME_PROCESS_SWITCH", "1") == "1"
            ),
            fused_enabled=fused,
        )

    def _event(self, kind, **fields):
        item = dict(kind=kind, tp_rank=self.rank, monotonic_ns=time.monotonic_ns(), **fields)
        self.events.append(item)
        logger.info("KV_EC_LIVE %s", json.dumps(item, sort_keys=True))

    def before_inputs(self, inputs):
        # Abort remains owned by the normal scheduler.  Weight/storage epoch
        # changes invalidate all prepared recovery material.
        if any(type(x).__name__.startswith(("UpdateWeight", "ReleaseMemoryOccupation", "ResumeMemoryOccupation")) for x in inputs):
            self.source = None
            self._event("checkpoint_invalidated", reason="model_or_storage_epoch_changed")

    def _control_requests_fault(self):
        if self.control is None or not self.control.is_file():
            return False
        try:
            data = json.loads(self.control.read_text())
        except (OSError, ValueError):
            return False
        return data.get("action") == "recover" and int(data.get("version", -1)) > self.version

    def after_result(self, batch):
        if self.audit is not None and self.audit.poll():
            self._event(
                "post_commit_async_audit_complete",
                layers=self.audit.layers_verified,
                recovery_version=self.version,
            )
            self.audit = None
        # Re-Prefill has no explicit restore transaction.  Its fair KV-ready
        # boundary is the first completed forward that has rebuilt the exact
        # recovery-point prefix.  Observe it here, immediately after the model
        # result is committed, using the same monotonic fault clock as EC.
        if (
            self.completed
            and self.selected_action == "RE_PREFILL"
            and not self.kv_ready_logged
            and self.point is not None
            and batch
        ):
            req = next(
                (x for x in batch.reqs if str(x.rid) == self.point.request_id),
                None,
            )
            if (
                req is not None
                and int(req.kv_committed_len) >= self.point.kv_token_boundary
            ):
                now_ns = time.monotonic_ns()
                elapsed_ms = (now_ns - self.point.created_ns) / 1_000_000
                self.policy.observe("RE_PREFILL", elapsed_ms)
                self.kv_ready_logged = True
                self._event(
                    "recovery_kv_ready",
                    action="RE_PREFILL",
                    recovery_version=self.point.recovery_version,
                    token_boundary=self.point.kv_token_boundary,
                    elapsed_ms=elapsed_ms,
                )
        if self.completed or self.requested or not batch or len(batch.reqs) != 1:
            return
        req = batch.reqs[0]
        if req.finished() or req.is_retracted or req.req_pool_idx is None:
            return
        decode_count = int(getattr(req, "decode_batch_idx", 0))
        if self._control_requests_fault() or (
            self.fault_at_kv_boundary is not None
            and int(req.kv_committed_len) >= self.fault_at_kv_boundary
        ) or (
            self.fault_after_decode is not None and decode_count >= self.fault_after_decode
        ):
            self.version += 1
            req.kv_ec_recovery_version = self.version
            req.kv_ec_output_paused = True
            req.kv_ec_forward_blocked = True
            self.point = RecoveryPoint.capture(self.scheduler, req, self.version)
            req.kv_ec_recovery_point = self.point
            req.kv_ec_fault_monotonic_ns = self.point.created_ns
            req.kv_ec_fault_output_count = len(req.output_ids)
            req.kv_ec_first_delivery_logged = False
            req.kv_ec_first_new_token_logged = False
            req.kv_ec_response_complete_logged = False
            self.requested = True
            self._event("fault_paused", point=asdict(self.point))

    def _vote(self, phase, ready, point):
        identity = dict(phase=phase, **point.common_identity())
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()
        value = torch.tensor([int(ready), *digest], dtype=torch.int32)
        gathered = [torch.empty_like(value) for _ in range(self.tp_size)]
        work = torch.distributed.all_gather(
            gathered, value, group=self.scheduler.tp_cpu_group, async_op=True
        )
        if work.wait(timeout=timedelta(seconds=self.timeout)) is False:
            raise TimeoutError("live_recovery_tp_vote_timeout:" + phase)
        return all(int(x[0]) == 1 for x in gathered) and all(
            torch.equal(gathered[0][1:], x[1:]) for x in gathered[1:]
        )

    def _validate_point(self, req, point, *, expected_slots=None):
        current = RecoveryPoint.capture(self.scheduler, req, point.recovery_version)
        if current != point:
            # created_ns differs; compare all state-bearing fields instead.
            a, b = asdict(current), asdict(point)
            a.pop("created_ns"); b.pop("created_ns")
            if expected_slots is not None:
                actual_slots = tuple(a.pop("kv_slots"))
                b.pop("kv_slots")
                if actual_slots != tuple(expected_slots):
                    raise RestoreRejected("live_recovery_committed_mapping_mismatch")
            if a != b:
                raise RestoreRejected("live_recovery_point_changed")
        if req.kv_ec_recovery_version != point.recovery_version or not req.kv_ec_output_paused:
            raise RestoreRejected("live_recovery_version_or_output_gate_lost")

    def _recover_ec(self, req, point):
        if self.source is None:
            raise RestoreRejected("live_recovery_checkpoint_unavailable")
        started = time.perf_counter()
        tx_holder = {}

        def prepare():
            self._validate_point(req, point)
            if self.cuda_codec is not None:
                tx = restore_xor_to_gpu(
                    self.scheduler, req, self.source, token_start=0,
                    parity=self.xor_parity, survivor=self.xor_survivor,
                    parity_sha256=self.parity_sha256, codec=self.cuda_codec,
                    source_name="immutable_host_memory_fused_xor",
                )
            else:
                tx = restore_to_gpu(
                    self.scheduler, req, self.source, token_start=0,
                    source_name="immutable_host_memory",
                )
            if tx.token_count != point.kv_token_boundary:
                tx.abort()
                raise RestoreRejected("live_checkpoint_boundary_mismatch")
            tx_holder["tx"] = tx
            return tx

        tx = coordinated_restore(
            prepare,
            lambda phase, ready: self._vote(phase, ready, point),
            lambda phase, _: self._event("tp_" + phase.lower(), recovery_version=point.recovery_version),
        )
        self._validate_point(req, point, expected_slots=tx.new_slots)
        factory = getattr(tx, "post_commit_audit_factory", None)
        if (
            factory is not None
            and os.environ.get("SGLANG_KV_EC_ASYNC_VERIFY", "1") == "1"
        ):
            self.audit = factory()
        elapsed = (time.perf_counter() - started) * 1000
        self.policy.observe("EC", elapsed)
        return tx, elapsed

    def _refresh_from_incremental_protection(self, req, point):
        """Install the exact committed live prefix as the in-memory EC source."""
        protector = getattr(self.scheduler, "_kv_ec_online", None)
        if protector is None or os.environ.get("SGLANG_KV_EC_INCREMENTAL", "1") != "1":
            return
        local = protector.snapshot_prefix(req, point.kv_token_boundary, timeout=self.timeout)
        gathered = [None for _ in range(self.tp_size)]
        torch.distributed.all_gather_object(
            gathered, local, group=self.scheduler.tp_cpu_group
        )
        rank_chunks = {rank: chunks for rank, chunks in enumerate(gathered)}
        token_ids = (req.origin_input_ids + req.output_ids)[:point.kv_token_boundary]
        if len(token_ids) != point.kv_token_boundary:
            raise RestoreRejected("incremental_recovery_token_boundary_unavailable")
        from .incremental_checkpoint import build_node_checkpoint

        checkpoint, blocks = build_node_checkpoint(
            int(os.environ.get("SGLANG_KV_EC_NODE_ID", "0")),
            rank_chunks, point.kv_token_boundary, token_ids,
        )
        self.source = BytesCheckpointSource(checkpoint)
        protection_mode = os.environ.get(
            "SGLANG_KV_EC_ONLINE_PROTECTION_MODE", "xor"
        )
        if self.fused_enabled and protection_mode in ("xor", "xor-rs"):
            self.xor_parity = xor(*blocks)
            self.xor_survivor = blocks[1 - self.rank]
            self.parity_sha256 = sha(self.xor_parity)
            device = self.scheduler.model_worker.model_runner.token_to_kv_pool.k_buffer[0].device
            self.cuda_codec = CudaRSCodec("cuda", device=device)
        else:
            self.xor_parity = self.xor_survivor = self.parity_sha256 = None
            self.cuda_codec = None
        self._event(
            "incremental_checkpoint_installed",
            recovery_version=point.recovery_version,
            token_boundary=point.kv_token_boundary,
            checkpoint_sha256=sha(checkpoint),
            failure_scope=os.environ.get("SGLANG_KV_EC_FAILURE_SCOPE", "node"),
            recovery_tier=(
                "local_xor_host_dram"
                if os.environ.get("SGLANG_KV_EC_FAILURE_SCOPE", "node") == "gpu-rank"
                else "remote_rs_global_commit"
            ),
        )

    def _re_prefill(self, req, point):
        """Retract in place and queue the same Req; weights/process stay hot."""
        from sglang.srt.managers.schedule_batch import retract_all

        s = self.scheduler
        # Immediately after ``process_batch_result`` the active decode request
        # is owned by ``last_batch`` until the next scheduling plan promotes
        # it back to ``running_batch``.  Fault recovery runs in exactly that
        # between-forwards window, so both scheduler-owned locations are live.
        batch = s.running_batch
        if batch is None or req not in batch.reqs:
            batch = getattr(s, "last_batch", None)
        if batch is None or req not in batch.reqs or len(batch.reqs) != 1:
            raise RestoreRejected("live_reprefill_requires_single_running_request")
        # reset_for_retract preserves output_ids and send_token_offset.  Thus
        # sent tokens are not emitted again and pending tokens are recomputed.
        retract_all(
            reqs=[req], server_args=s.server_args,
            req_to_token_pool=s.req_to_token_pool,
            token_to_kv_pool_allocator=s.token_to_kv_pool_allocator,
            tree_cache=s.tree_cache, hisparse_coordinator=s.hisparse_coordinator,
            offload_kv=False,
        )
        # A fault can be injected immediately after the final chunked-prefill
        # forward, before the normal scheduling path clears this cursor.  The
        # retracted Req has no extend_range, so retaining the stale cursor would
        # make get_next_batch_to_run dereference None on the next iteration.
        if getattr(s, "chunked_req", None) is req:
            s.chunked_req = None
        batch.filter_batch(keep_indices=[])
        req.kv_ec_output_paused = False
        req.kv_ec_forward_blocked = False
        req.kv_ec_active_version = point.recovery_version
        s._add_request_to_queue(req, is_retracted=True)
        self._event("reprefill_queued", recovery_version=point.recovery_version)

    def _choose_path(self, point):
        if not self.same_process_switch:
            return "RE_PREFILL", "same_process_switch_disabled"
        if not self.hybrid_enabled:
            return (
                ("EC", "hybrid_disabled_ec") if self.source is not None
                else ("RE_PREFILL", "hybrid_disabled_no_ec")
            )
        runner = self.scheduler.model_worker.model_runner
        hf = runner.model_config.hf_config
        kv_bytes = (
            point.kv_token_boundary * int(hf.num_hidden_layers)
            * int(hf.num_key_value_heads) * int(hf.head_dim) * 2 * 2
        )
        return self.policy.choose(
            ec_available=self.source is not None,
            token_count=point.kv_token_boundary,
            kv_bytes=kv_bytes,
            prefill_tokens_per_s=float(os.environ.get(
                "SGLANG_KV_EC_PREFILL_TOKENS_PER_S", "0"
            )) or None,
            ec_bytes_per_s=float(os.environ.get(
                "SGLANG_KV_EC_RESTORE_BYTES_PER_S", "0"
            )) or None,
        )

    def _choose_before_materializing_ec(self, point):
        """Use calibrated costs before assembling the protected prefix.

        Incremental protection can have a committed prefix while ``source`` is
        still unset: materializing it requires waiting for every chunk and
        packing a node checkpoint.  Paying that cost before choosing
        Re-Prefill defeats the hybrid policy.  Only make an early decision when
        both same-boundary calibration rates are explicit; otherwise preserve
        the conservative committed-EC behavior.
        """
        if not self.same_process_switch or not self.hybrid_enabled:
            return None
        protector = getattr(self.scheduler, "_kv_ec_online", None)
        if protector is None or os.environ.get("SGLANG_KV_EC_INCREMENTAL", "1") != "1":
            return None
        prefill_rate = float(os.environ.get("SGLANG_KV_EC_PREFILL_TOKENS_PER_S", "0"))
        restore_rate = float(os.environ.get("SGLANG_KV_EC_RESTORE_BYTES_PER_S", "0"))
        if prefill_rate <= 0 or restore_rate <= 0:
            return None
        runner = self.scheduler.model_worker.model_runner
        hf = runner.model_config.hf_config
        kv_bytes = (
            point.kv_token_boundary * int(hf.num_hidden_layers)
            * int(hf.num_key_value_heads) * int(hf.head_dim) * 2 * 2
        )
        return self.policy.choose(
            ec_available=True,
            token_count=point.kv_token_boundary,
            kv_bytes=kv_bytes,
            prefill_tokens_per_s=prefill_rate,
            ec_bytes_per_s=restore_rate,
        )

    def before_schedule(self):
        if not self.requested:
            return False
        req = getattr(self.point, "request_id", None)
        candidates = []
        if self.scheduler.running_batch is not None:
            candidates.extend(self.scheduler.running_batch.reqs)
        if self.scheduler.last_batch is not None:
            candidates.extend(self.scheduler.last_batch.reqs)
        candidates.extend(self.scheduler.waiting_queue)
        live = next((x for x in candidates if str(x.rid) == req), None)
        if live is None:
            raise RestoreRejected("live_recovery_request_disappeared")
        point = self.point
        early = self._choose_before_materializing_ec(point)
        if early is not None and early[0] == "RE_PREFILL":
            action, reason = early
            reason = "pre_materialization_" + reason
        else:
            try:
                self._refresh_from_incremental_protection(live, point)
                action, reason = self._choose_path(point)
            except (RestoreRejected, TimeoutError, ValueError) as exc:
                if not self.hybrid_enabled:
                    raise
                action, reason = "RE_PREFILL", "ec_materialization_failed"
                self._event(
                    "ec_materialization_fallback",
                    error=str(exc), recovery_version=point.recovery_version,
                )
        self._event("hybrid_choice", action=action, reason=reason, recovery_version=point.recovery_version)
        self.selected_action = action
        if action == "RE_PREFILL":
            self._re_prefill(live, point)
        else:
            tx, elapsed = self._recover_ec(live, point)
            # The mapping is already globally committed.  Only now release the
            # forward/output gates.  Pending output remains at send_token_offset.
            live.kv_ec_active_version = point.recovery_version
            live.kv_ec_output_paused = False
            live.kv_ec_forward_blocked = False
            self._event(
                "scheduler_ready", recovery_version=point.recovery_version,
                elapsed_ms=elapsed, transaction_state=tx.state,
                delivered_tokens=point.delivered_tokens,
                pending_tokens=len(point.computed_unsent_tokens),
            )
            self.kv_ready_logged = True
            self._event(
                "recovery_kv_ready",
                action="EC",
                recovery_version=point.recovery_version,
                token_boundary=point.kv_token_boundary,
                elapsed_ms=(time.monotonic_ns() - point.created_ns) / 1_000_000,
            )
        self.requested = False
        self.completed = True
        return True

    def close(self):
        if self.audit is not None:
            self.audit.close()
            self.audit = None
        self.source = None
        self.xor_parity = self.xor_survivor = None
