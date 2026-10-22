"""Scheduler-path CPU tests for the no-write M6-B1 transaction."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from experiments.kv_ec.pause_control import PauseCoordinator
from experiments.kv_ec.scheduler_pause import PauseRejected
from experiments.kv_ec.test_scheduler_pause import fixture


class CompletedEvent:
    def synchronize(self):
        pass

    def query(self):
        return True

    def record(self, *args, **kwargs):
        pass


class Work:
    def wait(self, timeout=None):
        return True


class PostResumeSchedulerTests(unittest.TestCase):
    def setup_rank(self, root, coordinator, rank):
        gate, req, _, _ = fixture(root, coordinator, rank)
        gate.req = req
        allocator = gate.scheduler.token_to_kv_pool_allocator
        allocator.enable_slot_leases(active_slots=[])
        slots = gate.scheduler.req_to_token_pool.req_to_token[2, :256].tolist()
        allocator.slot_lease_registry.generation[slots] = 1
        gate.scheduler.forward_stream = Mock()
        gate.scheduler.tp_cpu_group = object()
        allocator.record_slot_lease_cuda_event = lambda lease_id, stream=None: (
            allocator.attach_slot_lease_cuda_event(lease_id, CompletedEvent()) or
            allocator.slot_lease_registry._leases[lease_id].cuda_event)
        gate.scheduler.model_worker.model_runner = NS(
            token_to_kv_pool=NS(k_buffer=[NS(device=NS(type="cuda"))]))
        gate.shadow = NS(
            manager_epoch="manager", worker=NS(state="COMMITTED", metadata={
                "checkpoint_version": "tx-chunk0", "manager_epoch": "manager"},
                require_recoverable=Mock(return_value={"state": "COMMITTED"})))
        gate.state = "RESUMED"
        gate.released = True
        gate.owner = f"tx:rank{rank}"
        gate.evidence.update(decode_observed=True)
        gate.deadline = 0
        gate.timeout_s = 3
        def fatal(exc):
            gate.state = "ABORTED_FAIL_CLOSED"
            raise PauseRejected(str(exc))
        gate._fatal = Mock(side_effect=fatal)
        return gate, req

    def collective(self, *, reject_rank=None):
        def gather(output, local, group=None, async_op=False):
            data = local.clone()
            if reject_rank is not None and int(local[0]) != 1:
                data[0] = 0
            for item in output:
                item.copy_(data)
            return Work()
        return gather

    def run_rank(self, gate):
        import experiments.kv_ec.post_resume_recovery as module

        def pack(pool, mapping, req, identity, **kwargs):
            return b"metadata-only", {
                "checkpoint_version": "tx-chunk0", "manager_epoch": "manager"}

        with patch.object(module.torch.distributed, "all_gather", side_effect=self.collective()), \
                patch.object(module.torch.cuda, "is_available", return_value=True), \
                patch.object(module.torch.cuda, "Event", CompletedEvent), \
                patch.object(module.torch.cuda, "current_stream", return_value=object()), \
                patch.object(module, "pack_chunk", side_effect=pack):
            return gate.tick()

    def test_scheduler_tick_runs_four_rank_transaction_and_fresh_lease(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            c = PauseCoordinator("tx")
            gates = [self.setup_rank(Path(tmp), c, rank)[0] for rank in range(4)]
            for gate in gates:
                gate.post_resume_enabled = True
                gate.post_resume = PostResumeRecovery()
            for gate in gates:
                self.assertFalse(self.run_rank(gate))
                ev = gate.evidence["post_resume_recovery"]
                self.assertTrue(ev["new_lease_created"] and ev["new_lease_released"])
                self.assertFalse(ev["old_lease_reused"] or ev["kv_written"])
                self.assertEqual(gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)
            self.assertTrue(all(g.post_resume.state == "RUNNING" for g in gates))

    def test_post_resume_recapture_preserves_async_checkpoint_identity(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            seen = {}
            import experiments.kv_ec.post_resume_recovery as module
            def pack(*args, **kwargs):
                seen.update(kwargs)
                return b"metadata-only", {
                    "checkpoint_version": "tx-chunk0", "manager_epoch": "manager"
                }
            with patch.object(module.torch.distributed, "all_gather", side_effect=self.collective()), \
                    patch.object(module.torch.cuda, "is_available", return_value=True), \
                    patch.object(module.torch.cuda, "Event", CompletedEvent), \
                    patch.object(module.torch.cuda, "current_stream", return_value=object()), \
                    patch.object(module, "pack_chunk", side_effect=pack):
                g.tick()
            self.assertEqual(seen["lifetime"], "independent_immutable_cpu_snapshot")
            self.assertEqual(seen["completion_dependency"],
                             "producer_event_then_blocking_d2h_then_capture_event")

    def test_repeated_entry_does_not_create_a_second_lease(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            def fail_closed(exc):
                g.state = "ABORTED_FAIL_CLOSED"
                g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.close()
                raise PauseRejected(str(exc))
            g._fatal = Mock(side_effect=fail_closed)
            self.run_rank(g)
            self.assertFalse(self.run_rank(g))
            self.assertEqual(len(g.post_resume.trace), 6)

    def test_request_cancelled_fails_closed_through_scheduler_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, req = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            req.to_finish = "cancelled"
            g.post_resume_enabled = True
            g.deadline = 10**20
            g.timeout_s = 3
            import experiments.kv_ec.post_resume_recovery as module
            with patch.object(module.torch.distributed, "all_gather", side_effect=self.collective()), \
                    patch.object(module.torch.cuda, "is_available", return_value=True):
                with self.assertRaises(PauseRejected):
                    g.tick()
            g._fatal.assert_called_once()

    def test_uncommitted_checkpoint_is_unanimously_deferred_before_transaction(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.shadow.worker.state = "COMMITTING"
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=self.collective()):
                self.assertFalse(g.tick())
            self.assertEqual(g.post_resume.state, "RUNNING")
            self.assertEqual(g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)
            self.assertNotIn("post_resume_recovery", g.evidence)

    def test_cuda_event_incomplete_fails_closed_and_releases_new_lease(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            def fail_closed(exc):
                g.state = "ABORTED_FAIL_CLOSED"
                g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.close()
                raise PauseRejected(str(exc))
            g._fatal = Mock(side_effect=fail_closed)
            import experiments.kv_ec.post_resume_recovery as module
            class PendingEvent(CompletedEvent):
                def query(self):
                    return False
            g.scheduler.token_to_kv_pool_allocator.record_slot_lease_cuda_event = (
                lambda lease_id, stream=None: (
                    g.scheduler.token_to_kv_pool_allocator.attach_slot_lease_cuda_event(
                        lease_id, PendingEvent()) or
                    g.scheduler.token_to_kv_pool_allocator.slot_lease_registry._leases[lease_id].cuda_event))
            def pack(*args, **kwargs):
                return b"metadata", {"checkpoint_version": "tx-chunk0", "manager_epoch": "manager"}
            with patch.object(module.torch.distributed, "all_gather", side_effect=self.collective()), \
                    patch.object(module.torch.cuda, "is_available", return_value=True), \
                    patch.object(module.torch.cuda, "Event", PendingEvent), \
                    patch.object(module.torch.cuda, "current_stream", return_value=object()), \
                    patch.object(module, "pack_chunk", side_effect=pack):
                with self.assertRaises(PauseRejected):
                    g.tick()
            self.assertEqual(g.state, "ABORTED_FAIL_CLOSED")
            # Never release a lease whose CUDA event is incomplete.
            registry = g.scheduler.token_to_kv_pool_allocator.slot_lease_registry
            self.assertEqual(registry.active_lease_count, 1)
            self.assertTrue(registry.closed)

    def test_changed_checkpoint_is_rejected_by_scheduler_path(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            import experiments.kv_ec.post_resume_recovery as module
            def pack(*args, **kwargs):
                return b"metadata", {"checkpoint_version": "wrong-checkpoint", "manager_epoch": "manager"}
            with patch.object(module.torch.distributed, "all_gather", side_effect=self.collective()), \
                    patch.object(module.torch.cuda, "is_available", return_value=True), \
                    patch.object(module.torch.cuda, "Event", CompletedEvent), \
                    patch.object(module.torch.cuda, "current_stream", return_value=object()), \
                    patch.object(module, "pack_chunk", side_effect=pack), \
                    patch.object(g.shadow.worker, "require_recoverable", side_effect=ValueError("checkpoint mismatch")):
                with self.assertRaises(PauseRejected):
                    g.tick()
            self.assertEqual(g.state, "ABORTED_FAIL_CLOSED")
            self.assertEqual(g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)

    def test_all_gather_timeout_fails_closed_before_any_lease(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            import experiments.kv_ec.post_resume_recovery as module
            class TimedOutWork:
                def wait(self, timeout=None):
                    return False
            with patch.object(module.torch.distributed, "all_gather", return_value=TimedOutWork()):
                with self.assertRaises(PauseRejected):
                    g.tick()
            self.assertEqual(g.state, "ABORTED_FAIL_CLOSED")
            self.assertEqual(g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)

    def test_default_off_preserves_resumed_scheduler_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_enabled = False
            self.assertFalse(g.tick())
            self.assertIsNone(g.post_resume)


if __name__ == "__main__":
    unittest.main()
