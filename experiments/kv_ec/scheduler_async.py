"""E-only adapter: blocking D2H under the existing gate, then CPU-only work."""

import json
import os
import resource
import time

import torch

from .async_protection import AsyncProtection, CPUSnapshot
from .scheduler_shadow import SchedulerShadowRecovery, pack_chunk


class SchedulerAsyncProtection:
    online = False
    modified = False
    protects_prefill = False
    evidence_key = "async_protection_only"

    def __init__(self):
        self.socket = os.environ["SGLANG_KV_EC_SHADOW_PARITY_SOCKET"]
        self.manager_epoch = os.environ["SGLANG_KV_EC_SHADOW_MANAGER_EPOCH"]
        self.lost_rank, self.recover_tokens = 0, 256
        if (os.environ.get("SGLANG_KV_EC_RECOVER_TOKENS", "256") != "256"
                or os.environ.get("SGLANG_KV_EC_LOST_RANK", "0") != "0"):
            raise ValueError("async_initial_scope_rank0_chunk0_256")
        self.done = False
        self.lifecycle_written = False
        self.worker = AsyncProtection(
            self.socket, timeout_s=float(os.environ["SGLANG_KV_EC_PAUSE_TIMEOUT_S"]),
            commit_delay_s=float(os.environ.get("SGLANG_KV_EC_ASYNC_COMMIT_DELAY_S", "0")),
            test_hold_state=(os.environ.get("SGLANG_KV_EC_M7_TEST_FAULT_STATE", "")
                             if os.environ.get("SGLANG_KV_EC_M7_TEST_FAULT_STATE", "")
                             != "UNPROTECTED" else ""))

    def check_online_admission(self, gate):
        SchedulerShadowRecovery.check_online_admission(self, gate)

    def validate(self, gate):
        if not gate.event.query():
            raise ValueError("async_source_event_incomplete")
        if self.done:
            meta = self.worker.metadata
            mapping = gate.mapping
            if (meta["request_version"] != mapping.request_version
                    or meta["req_pool_idx"] != mapping.req_pool_idx
                    or meta["req_generation"] != mapping.req_generation
                    or meta["physical_slots"] != list(mapping.physical_slots[:256])
                    or meta["slot_generation"] != list(mapping.slot_generations[:256])
                    or meta["manager_epoch"] != self.manager_epoch
                    or meta["checkpoint_version"] != gate.identity["transaction_id"] + "-chunk0"):
                raise ValueError("async_snapshot_binding_changed_before_release")

    def run(self, gate):
        self.validate(gate)
        gate._revalidate()
        self.check_online_admission(gate)
        pool = gate.scheduler.model_worker.model_runner.token_to_kv_pool
        if not torch.cuda.is_available() or pool.k_buffer[0].device.type != "cuda":
            raise ValueError("async_requires_real_cuda")
        started = time.monotonic()
        with torch.cuda.stream(gate.scheduler.forward_stream):
            gate.scheduler.forward_stream.wait_event(gate.event)
            payload, metadata = pack_chunk(
                pool, gate.mapping, gate.req, gate.model_identity,
                snapshot_id=gate.identity["transaction_id"] + "-chunk0",
                manager_epoch=self.manager_epoch)
            metadata = dict(metadata, lifetime="independent_immutable_cpu_snapshot",
                            completion_dependency="producer_event_then_blocking_d2h_then_capture_event")
            # Every .cpu() in pack_chunk is blocking. Retain the existing
            # producer/lease event and additionally fence the completed capture.
            copied = gate.scheduler.device_module.Event()
            copied.record(stream=gate.scheduler.forward_stream)
            copied.synchronize()
            snapshot = CPUSnapshot.freeze(
                payload, metadata, event_complete=copied.query(), started=started)
        self.worker.stage(snapshot)
        self.validate(gate)
        consensus = dict(
            snapshot_id=metadata["snapshot_id"], checkpoint_version=metadata["checkpoint_version"],
            request_hash=gate.evidence["request_hash"], request_version=metadata["request_version"],
            manager_epoch=self.manager_epoch, input_ids_sha256=metadata["input_ids_sha256"],
            snapshot_ready=True, copy_complete=True, independent_cpu_bytes=True,
            checkpoint_committed=False, used_committed_parity=False,
            online_kv_pool_modified=False, fault_injected=False)
        return dict(
            mode=self.evidence_key, consensus=consensus, metadata=metadata,
            snapshot_started_monotonic=started, snapshot_ready_monotonic=snapshot.ready,
            snapshot_ms=(snapshot.ready - started) * 1000,
            source_copy_event_complete=True, source_copy_event_synchronized=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            snapshot_cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

    def start_background(self, gate):
        # Called only after all ranks completed the existing lease release and
        # row-release vote. No foreground call will wait on worker.finished.
        self.resume = time.monotonic()
        gate.evidence[self.evidence_key].update(
            foreground_protection_blocking_ms=(self.resume - gate.started) * 1000,
            foreground_released_monotonic=self.resume,
            commit_wait_in_decode_path=False)
        path = gate.output / f"async-rank-{gate.rank}.json"
        rank, transaction = gate.rank, gate.identity["transaction_id"]
        prefill = gate.evidence["prefill_completed_monotonic"]
        released = self.resume
        def complete(record):
            committed = record["committed_observed_monotonic"]
            record.update(
                tp_rank=rank, transaction_id=transaction,
                foreground_released_monotonic=released,
                prefill_completed_monotonic=prefill,
                unprotected_window_ms=(
                    (committed - prefill) * 1000
                    if committed is not None else None),
                background_cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            # Publish atomically; polling readers must not see partial JSON.
            temporary = path.with_suffix(".writing")
            with temporary.open("x") as stream:
                json.dump(record, stream, indent=2)
            temporary.rename(path)
        self.worker.start(on_complete=complete)

    def mark_foreground_ready(self, gate):
        # Include thread startup and the pending-evidence write, but never wait
        # for background completion. This timestamp precedes the next selection.
        ready = time.monotonic()
        gate.evidence[self.evidence_key].update(
            foreground_ready_monotonic=ready,
            foreground_blocking_started_monotonic=gate.started,
            foreground_protection_blocking_ms=(ready - gate.started) * 1000)

    def observe_request_end(self, gate):
        if self.lifecycle_written:
            return
        req = gate.req
        reason = getattr(req, "finished_reason", None) or req.to_finish
        cancelled = req.is_retracted or type(reason).__name__ == "FINISH_ABORT"
        if not cancelled and not req.finished():
            return
        if cancelled:
            self.worker.cancel()
        record = dict(
            transaction_id=gate.identity["transaction_id"], tp_rank=gate.rank,
            request_end_monotonic=time.monotonic(), cancelled=cancelled,
            state_at_request_end=self.worker.record()["state"],
            pending_policy="bounded_off_path_drain" if not cancelled else "invalidate_fail_closed",
            foreground_waited_for_commit=False)
        path = gate.output / f"async-lifecycle-rank-{gate.rank}.json"
        temporary = path.with_suffix(".writing")
        with temporary.open("x") as stream:
            json.dump(record, stream, indent=2)
        temporary.rename(path)
        self.lifecycle_written = True
