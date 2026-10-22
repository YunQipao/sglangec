"""Scheduler-path CPU tests for the no-write M6-B1 transaction."""

import tempfile
import time
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
            g.deadline = time.monotonic() + 3
            g.timeout_s = 3
            import experiments.kv_ec.post_resume_recovery as module
            with patch.object(module.torch.distributed, "all_gather", side_effect=self.collective()), \
                    patch.object(module.torch.cuda, "is_available", return_value=True):
                with self.assertRaises(PauseRejected):
                    g.tick()
            g._fatal.assert_called_once()

    def test_committing_without_fault_continues_decode(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.shadow.worker.state = "COMMITTING"
            g.post_resume_enabled = True
            g.post_resume_mode = "m6b2"
            g.post_resume = PostResumeRecovery()
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=self.collective()):
                self.assertFalse(g.tick())
            self.assertEqual(g.post_resume.state, "RUNNING")
            self.assertEqual(g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)
            self.assertFalse(g._fatal.called)
            self.assertFalse(g.evidence["recovery_readiness"]["fault_requested"])
            self.assertEqual(g.evidence["recovery_readiness"]["action"],
                             "WAIT_AND_CONTINUE_DECODE")

    def test_m7_committed_no_fault_control_continues_decode(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_enabled = True
            g.post_resume_mode = "m6b2"
            g.m7_test_no_fault = True
            g.post_resume = PostResumeRecovery()
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=self.collective()):
                self.assertFalse(g.tick())
            self.assertEqual(g.post_resume.state, "RUNNING")
            self.assertFalse(g.evidence.get("fault_requested", False))
            self.assertFalse(g.evidence.get("post_resume_recovery", {}).get("parity_used", False))

    def test_m7_no_fault_completed_request_records_unanimous_normal_completion(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, req = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            req.finished = Mock(return_value=True)
            g.post_resume_enabled = True
            g.post_resume_mode = "m6b2"
            g.m7_test_no_fault = True
            g.post_resume = PostResumeRecovery()
            g.deadline = time.monotonic() + 3
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=self.collective()):
                self.assertFalse(g.tick())
            record = g.evidence["post_resume_recovery"]
            self.assertEqual(record["state"], "NO_FAULT_DECODE_CONTINUED")
            self.assertEqual(record["rank_checkpoint_readiness"], [5, 5, 5, 5])
            self.assertFalse(record["fault_injected"] or record["parity_used"] or record["kv_written"])

    def test_m7_target_state_not_reached_is_explicit_not_pass(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, req = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            req.finished = Mock(return_value=True)
            g.post_resume_enabled = True
            g.post_resume_mode = "m6b2"
            g.m7_test_fault_state = "UNPROTECTED"
            g.post_resume = PostResumeRecovery()
            g.deadline = time.monotonic() + 3
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=self.collective()):
                self.assertFalse(g.tick())
            record = g.evidence["post_resume_recovery"]
            self.assertEqual(record["state"], "TEST_NOT_REACHED")
            self.assertFalse(record["target_state_reached"])
            self.assertEqual(record["rank_checkpoint_readiness"], [5, 5, 5, 5])

    def test_m7_negative_acceptance_rejects_missing_or_inconsistent_proof(self):
        from experiments.kv_ec.accept_pause import check_m7_fail_closed_evidence
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"transaction_id": "tx"}
            for rank in range(4):
                ranks = root / "ranks"
                ranks.mkdir(exist_ok=True)
                policy = dict(action="FAIL_CLOSED", recommended_action="PREFILL_RETRY",
                              use_parity=False, reason="parity_not_committed")
                record = dict(
                    state="FAILED", transaction_id=f"tx:m6b1:chunk{rank}",
                    fault_requested=True, fault_injected=False, parity_used=False,
                    kv_written=False, policy=policy, failure_reason="expected",
                    checkpoint_state="COMMITTING",
                    fault_boundary=dict(local_checkpoint_state="COMMITTING",
                                        rank_fault_intents=[1, 1, 1, 1]),
                    readiness_trace=[dict(checkpoint_state="COMMITTING",
                                          fault_requested=True)])
                (ranks / f"m6b2-rank-{rank}.json").write_text(__import__("json").dumps(record))
                error = dict(state="ABORTED_FAIL_CLOSED", failure=dict(
                    exception_message="m7_policy_fail_closed:parity_not_committed"))
                (ranks / f"pause-rank-{rank}.error.json").write_text(__import__("json").dumps(error))
            records, _ = check_m7_fail_closed_evidence(root, identity, "COMMITTING")
            self.assertEqual(len(records), 4)
            bad = __import__("json").loads((root / "ranks/m6b2-rank-2.json").read_text())
            bad["kv_written"] = True
            (root / "ranks/m6b2-rank-2.json").write_text(__import__("json").dumps(bad))
            with self.assertRaisesRegex(ValueError, "policy_proof_invalid"):
                check_m7_fail_closed_evidence(root, identity, "COMMITTING")

    def test_explicit_fault_before_commit_fails_closed_without_parity(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.shadow.worker.state = "COMMITTING"
            g.post_resume_enabled = True
            g.post_resume_mode = "m6b2"
            g.post_resume = PostResumeRecovery()
            g.evidence["fault_requested"] = True
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=self.collective()):
                with self.assertRaises(PauseRejected):
                    g.tick()
            self.assertEqual(g.post_resume.state, "FAILED")
            policy = g.evidence["recovery_policy_decision"]
            self.assertEqual(policy["action"], "FAIL_CLOSED")
            self.assertEqual(policy["recommended_action"], "PREFILL_RETRY")
            self.assertFalse(policy["use_parity"])
            self.assertFalse(g.evidence["post_resume_recovery"]["parity_used"])

    def test_m7_controlled_uncommitted_fault_states_fail_closed_without_parity(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        import experiments.kv_ec.post_resume_recovery as module

        for state in ("UNPROTECTED", "SNAPSHOT_READY", "COMMITTING"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
                g.shadow.worker.state = state
                g.post_resume_enabled = True
                g.post_resume_mode = "m6b2"
                g.m7_test_fault_state = state
                g.post_resume = PostResumeRecovery()
                with patch.object(module.torch.distributed, "all_gather",
                                  side_effect=self.collective()):
                    with self.assertRaises(PauseRejected):
                        g.tick()
                boundary = g.evidence["m7_fault_boundary"]
                self.assertTrue(boundary["local_fault_intent"])
                self.assertEqual(boundary["rank_fault_intents"], [1, 1, 1, 1])
                policy = g.evidence["recovery_policy_decision"]
                self.assertEqual(policy["action"], "FAIL_CLOSED")
                self.assertFalse(policy["use_parity"])
                self.assertFalse(g.evidence["post_resume_recovery"]["parity_used"])
                self.assertIn("prefill_retry_unsupported_after_decode_observed",
                              g.evidence["post_resume_recovery"]["failure_reason"])

    def test_m7_fault_intent_disagreement_fails_closed_before_policy_vote(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        import experiments.kv_ec.post_resume_recovery as module
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.shadow.worker.state = "COMMITTING"
            g.post_resume_enabled = True
            g.post_resume_mode = "m6b2"
            g.m7_test_fault_state = "COMMITTING"
            g.post_resume = PostResumeRecovery()

            def mixed_intent(output, local, group=None, async_op=False):
                for index, item in enumerate(output):
                    item.copy_(local)
                    item[1] = 1 if index in (0, 2) else 0
                return Work()

            with patch.object(module.torch.distributed, "all_gather",
                              side_effect=mixed_intent):
                with self.assertRaises(PauseRejected):
                    g.tick()
            self.assertEqual(g.evidence["m7_fault_boundary"]["rank_fault_intents"],
                             [1, 0, 1, 0])
            self.assertNotIn("recovery_policy_decision", g.evidence)
            self.assertFalse(g.evidence["post_resume_recovery"]["parity_used"])

    def test_mixed_committing_and_committed_ranks_wait_without_recovery(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.shadow.worker.state = "COMMITTING"
            g.post_resume_mode = "m6b2"
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            def mixed(output, local, group=None, async_op=False):
                for index, item in enumerate(output):
                    item.copy_(local)
                    item[0] = 1 if index in (1, 3) else 2
                return Work()
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=mixed):
                self.assertFalse(g.tick())
            self.assertEqual(g.post_resume.state, "RUNNING")
            self.assertFalse(g._fatal.called)
            self.assertEqual(g.evidence["recovery_readiness"]["rank_readiness"], [2, 1, 2, 1])
            self.assertEqual(g.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 0)

    def test_live_manager_epoch_change_still_fails_closed_while_waiting(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_mode = "m6b2"
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            g.shadow.manager_epoch = "restarted-manager"
            with patch("experiments.kv_ec.post_resume_recovery.torch.distributed.all_gather",
                       side_effect=self.collective()):
                with self.assertRaises(PauseRejected):
                    g.tick()
            self.assertEqual(g.evidence["readiness_failure"], "manager_epoch_changed")
            self.assertTrue(g._fatal.called)

    def test_row_generation_change_still_fails_closed_while_waiting(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_mode = "m6b2"
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            g.shadow.worker.state = "COMMITTING"
            g.shadow.worker.metadata.update(
                req_pool_idx=1, req_generation=2, request_version="request-v1",
                physical_slots=[8] * 256, slot_generation=[1] * 256)
            changed = NS(req_pool_idx=2, req_generation=2,
                         request_version="request-v1", physical_slots=tuple([8] * 256),
                         slot_generations=tuple([1] * 256),
                         slot_generation_available=True)
            import experiments.kv_ec.post_resume_recovery as module
            with patch.object(module, "read_scheduler_mapping", return_value=changed), \
                    patch.object(module.torch.distributed, "all_gather",
                                 side_effect=self.collective()):
                with self.assertRaises(PauseRejected):
                    g.tick()
            self.assertEqual(g.evidence["readiness_failure"], "row_slot_generation_changed")
            self.assertTrue(g._fatal.called)

    def test_all_committed_ready_ranks_enter_existing_b2_transaction(self):
        from experiments.kv_ec.post_resume_recovery import PostResumeRecovery
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_mode = "m6b2"
            g.post_resume_enabled = True
            g.post_resume = PostResumeRecovery()
            fake_recovery = dict(fault_injected=True, parity_used=True, kv_written=True,
                                 byte_exact=True, four_rank_vote=True, timings={})
            import experiments.kv_ec.post_resume_recovery as module
            with patch.object(module.torch.distributed, "all_gather",
                              side_effect=self.collective()), \
                    patch.object(module.torch.cuda, "is_available", return_value=True), \
                    patch.object(module.torch.cuda, "Event", CompletedEvent), \
                    patch.object(module.torch.cuda, "current_stream", return_value=object()), \
                    patch.object(module, "pack_chunk", return_value=(b"metadata", {
                        "checkpoint_version": "tx-chunk0", "manager_epoch": "manager"})), \
                    patch.object(g.post_resume, "_b2", return_value=fake_recovery) as b2:
                self.assertFalse(g.tick())
            b2.assert_called_once()
            self.assertEqual(g.evidence["recovery_policy_decision"]["action"], "EC_RECOVERY")

    def test_committed_state_records_unanimous_ec_policy(self):
        from experiments.kv_ec.post_resume_recovery import (
            EC_RECOVERY, PostResumeRecovery,
        )
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.setup_rank(Path(tmp), PauseCoordinator("tx"), 0)
            g.post_resume_mode = "m6b2"
            g.deadline = time.monotonic() + 3
            g.shadow.lost_rank = 0
            recovery = PostResumeRecovery()
            recovery.checkpoint_id = "tx-chunk0"
            recovery.epoch = "manager"
            import experiments.kv_ec.post_resume_recovery as module
            with patch.object(module.torch.distributed, "all_gather",
                              side_effect=self.collective()):
                decision = recovery._b2_policy(g, None, None)
            self.assertEqual(decision.action, EC_RECOVERY)
            self.assertTrue(decision.use_parity)

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
