"""CPU execution of the live Scheduler path; never hardware acceptance."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.kv_ec import scheduler_shadow as recovery
from experiments.kv_ec.accept_pause import check_online_evidence
from experiments.kv_ec.gpu_scratch_restore import restore_rank_to_gpu
from experiments.kv_ec.m3_performance import online_record
from experiments.kv_ec.parity_manager import ParityManager
from experiments.kv_ec.pause_control import PauseCoordinator
from experiments.kv_ec.scheduler_pause import PauseRejected
from experiments.kv_ec.test_accept_pause import completed_run
from experiments.kv_ec.test_online_restore import pool_fixture
from experiments.kv_ec.test_scheduler_pause import fixture


class M3LivePathTests(unittest.TestCase):
    def execute(self, root, *, lost, tokens, local=None, failure=None):
        """Only CUDA/IPC are substituted; packing/XOR/scatter/leases are real."""
        local = lost if local is None else local
        coordinator = PauseCoordinator("test", require_online=True, lost_rank=lost, recover_tokens=tokens)
        gate, req, batch, _ = fixture(root, coordinator, rank=local)
        gate.identity.update(lost_rank=lost, recover_tokens=tokens)
        manager = ParityManager(test_oracle=True, manager_epoch="manager")
        pool = pool_fixture()
        # Noncontiguous prompt order catches offset/physical-slot confusion.
        gate.scheduler.req_to_token_pool.req_to_token[2, :768] = torch.arange(768, 0, -1)
        proxy = NS(k_buffer=[NS(device=torch.device(f"cuda:{local}"))])
        gate.scheduler.model_worker.model_runner = NS(token_to_kv_pool=proxy)
        gate.scheduler.forward_stream = Mock()
        gate.scheduler.tp_cpu_group = object()
        gate.model_identity.update(source_commit="commit", core_diff_sha256="diff")
        online = recovery.SchedulerShadowRecovery("/tmp/parity.sock", "manager", online=True,
                                                   lost_rank=lost, recover_tokens=tokens)
        gate.shadow = online
        real_tensor, packer, checker = torch.tensor, recovery.pack_chunk, recovery.checked_buffers
        writer = recovery.write_mapped_bytes
        blocks = {}
        writes, gathers = [], []
        def control(path, message, **kwargs):
            return {"state": "OK", "status": coordinator.dispatch(message)}
        def tensor(*args, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return real_tensor(*args, **kwargs)
        def pack(*args, **kwargs):
            return packer(pool, *args[1:], **kwargs)
        def ipc(path, metadata, payload, **kwargs):
            if metadata.get("kind") == "status":
                return manager.status(metadata["snapshot_id"])
            self.assertFalse(online.write_started)
            snapshot = metadata["snapshot_id"]
            blocks[snapshot] = {}
            manager.submit(metadata, payload)
            blocks[snapshot][local] = payload.copy()
            for rank in range(4):
                if rank == local:
                    continue
                block, meta = np.bitwise_xor(payload, rank + 1), deepcopy(metadata)
                meta["tp_rank"] = rank
                derived = recovery.validate_payload(block, meta, verify_checksums=False)
                for key in ("sha256", "k_sha256", "v_sha256"):
                    meta[key] = derived[key]
                for layer in meta["layer_records"]:
                    for kind in ("k", "v"):
                        layer[kind + "_sha256"] = derived["layer_sha256"][(layer["layer_id"], kind)]
                blocks[snapshot][rank] = block
                manager.submit(meta, block)
            return manager.status(snapshot)
        def gather(contribution, *, gather_list, dst, **kwargs):
            self.assertEqual(dst, lost)
            gathers.append(online.metadata["token_start"])
            if local == lost:
                self.assertIsNone(online.survivor)
                self.assertTrue(all(c["survivor"] is None for c in online.chunks))
                self.assertEqual(torch.count_nonzero(contribution).item(), 0)
                for rank in range(4):
                    if rank != lost:
                        gather_list[rank].copy_(torch.from_numpy(blocks[online.metadata["snapshot_id"]][rank]))
            else:
                self.assertIsNone(gather_list)
                self.assertTrue(np.array_equal(contribution.numpy(), online.survivor))
            return NS(wait=lambda **kwargs: True)
        def fetch(path, metadata, **kwargs):
            if failure == "second_chunk_parity" and metadata["token_start"] == 256:
                raise ValueError("injected_second_chunk_parity_failure")
            status, raw = manager.export_parity(metadata["snapshot_id"], metadata)
            return status, np.frombuffer(raw, dtype=np.uint8).copy()
        def restore(payload, metadata, device, *, cuda_metrics):
            cuda_metrics["h2d_copy_event_ms"] = 0.
            return restore_rank_to_gpu(payload, metadata, torch.device("cpu"))
        def write(unused, slots, sources):
            if sources is online.recovered:
                self.assertNotEqual(id(sources), id(online.rollback))
                writes.append(tuple(slots.tolist()))
            return writer(pool, slots, sources)
        with patch("experiments.kv_ec.scheduler_pause.torch.cuda.is_available", return_value=False), patch(
                "experiments.kv_ec.scheduler_pause.request", side_effect=control):
            gate.after_result(batch)
        before = [tuple(t.view(torch.uint8).clone() for t in pool.get_kv_buffer(layer)) for layer in range(36)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(online, "check_online_admission"))
            for name, kwargs in (
                ("manager_hello", dict(return_value="manager")),
                ("torch.cuda.is_available", dict(return_value=True)),
                ("torch.cuda.stream", dict(return_value=nullcontext())),
                ("torch.cuda.Event", dict(side_effect=lambda: NS(record=Mock(), query=lambda: True))),
                ("torch.tensor", dict(side_effect=tensor)),
                ("pack_chunk", dict(side_effect=pack)),
                ("checked_buffers", dict(side_effect=lambda p, layer, slots: checker(pool, layer, slots))),
                ("submit_once", dict(side_effect=ipc)),
                ("torch.distributed.gather", dict(side_effect=gather)),
                ("fetch_parity", dict(side_effect=fetch)),
                ("restore_rank_to_gpu", dict(side_effect=restore)),
                ("write_mapped_bytes", dict(side_effect=write)),
            ):
                stack.enter_context(patch("experiments.kv_ec.scheduler_shadow." + name, **kwargs))
            stack.enter_context(patch("experiments.kv_ec.scheduler_pause.request", side_effect=control))
            for _ in range(10):
                if online.prepare(gate):
                    break
            else:
                self.fail("protection did not complete")
            self.assertEqual(len(online.chunks), tokens // 256)
            self.assertTrue(all(c["status"]["state"] == "COMMITTED" for c in online.chunks))
            self.assertTrue(all(not c.ranks for c in manager._checkpoints.values()))
            gate._tick_local()
            for rank in range(4):
                if rank != local:
                    coordinator.dispatch(dict(gate.identity, op="paused", tp_rank=rank,
                                               observation=deepcopy(coordinator.observations[local])))
            gate._vote = lambda ready: ready
            gate.state = "SHADOW_READY"
            if failure:
                with self.assertRaisesRegex(ValueError, failure):
                    online.run_online(gate)
                self.assertTrue(gate.evidence["online"]["rollback_byte_exact"])
                with self.assertRaises(PauseRejected):
                    gate._fatal(RuntimeError(failure))
                self.assertTrue(gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.closed)
                self.assertEqual(gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.active_lease_count, 1)
                self.assertFalse(coordinator.shadow)
                result = None
            else:
                result = online.run_online(gate)
                self.assertEqual(result["chunk_count"], tokens // 256)
                self.assertEqual([c["token_start"] for c in result["chunks"]], list(range(0, tokens, 256)))
                for rank in range(4):
                    coordinator.dispatch(dict(gate.identity, op="shadow_verified", tp_rank=rank,
                                               observation=result["consensus"]))
                self.assertEqual(coordinator.dispatch(dict(gate.identity, op="authorize"))["state"],
                                 "RESUME_REQUESTED")
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    online.run_online(gate)
            self.assertEqual(gathers, list(range(0, 512 if failure else tokens, 256)))
            if local == lost and not failure:
                self.assertEqual(len(writes), tokens // 256)
                self.assertEqual(len({slot for chunk in writes for slot in chunk}), tokens)
            elif local != lost:
                self.assertFalse(writes)
                self.assertFalse(online.modified)
                self.assertFalse(online.modified_chunks)
            for layer, pair in enumerate(before):
                for live, expected in zip(pool.get_kv_buffer(layer), pair):
                    self.assertTrue(torch.equal(live.view(torch.uint8), expected))
        return gate, result, manager

    def test_every_rank_and_token_range_executes_live_target_path(self):
        for lost in range(4):
            for tokens in (256, 512, 768):
                with self.subTest(lost=lost, tokens=tokens), tempfile.TemporaryDirectory() as tmp:
                    self.execute(Path(tmp), lost=lost, tokens=tokens)

    def test_survivors_never_inject_or_write_for_any_target(self):
        for lost in range(4):
            for local in range(4):
                if local == lost:
                    continue
                with self.subTest(lost=lost, local=local), tempfile.TemporaryDirectory() as tmp:
                    self.execute(Path(tmp), lost=lost, tokens=768, local=local)

    def test_second_chunk_failure_rolls_back_all_modified_ranges_and_keeps_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.execute(Path(tmp), lost=2, tokens=768, failure="second_chunk_parity")

    def test_m3_auditor_requires_each_chunk_and_actual_target_evidence(self):
        # Synthetic timing normalization tests only the offline validator.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate, online, manager = self.execute(root, lost=1, tokens=768)
            completed_run(root)
            identity = dict(transaction_id="test", epoch="test-epoch", lost_rank=1, recover_tokens=768)
            checkpoints = [manager.status(f"test-chunk{i}") for i in range(3)]
            target_record = None
            for rank in range(4):
                path = root / "ranks" / f"pause-rank-{rank}.json"
                record = json.loads(path.read_text())
                record.update(online_kv_pool_modified=rank == 1, physical_slots_before=list(range(768, 0, -1)),
                              physical_slots_after=list(range(768, 0, -1)), paused_observation_monotonic=2.,
                              decode_observed_monotonic=5., request_hash=gate.evidence["request_hash"],
                              request_generation_before=gate.mapping.req_generation,
                              request_generation_after=gate.mapping.req_generation,
                              slot_generations_before=list(gate.mapping.slot_generations),
                              slot_generations_after=list(gate.mapping.slot_generations))
                record["coordinator_final"].update(require_online=True, shadow_ranks=list(range(4)),
                                                    lost_rank=1, recover_tokens=768)
                record["protection"] = deepcopy(gate.evidence["protection"])
                record["online"] = deepcopy(online)
                record["online"]["lost_rank"] = 1
                record["online"]["timing_ms"]["fault_to_decode_ms"] = 2.
                for index, chunk in enumerate(record["online"]["chunks"]):
                    sha = checkpoints[index]["rank_hashes"][str(rank)]["sha256"]
                    chunk.update(tp_rank=rank, device=f"cuda:{rank}", original_sha256=sha,
                                 restored_sha256=sha, rollback_sha256=sha, fault_injected=rank == 1,
                                 fault_bytes_changed=rank == 1, online_kv_pool_modified=rank == 1,
                                 writeback_byte_exact=rank == 1, survivor_unchanged=rank != 1,
                                 fault_monotonic=3., decode_observed_monotonic=5.)
                    chunk["timing_ms"]["fault_to_decode_ms"] = 2.
                    protection = record["protection"]["chunks"][index]
                    protection.update(committed_monotonic=1., source_sha256=sha)
                path.write_text(json.dumps(record))
                if rank == 1:
                    target_record = record
            response = {"input_ids_sha256": online["input_ids_sha256"],
                        "client_metrics_ms": dict(request_total_ms=99., ttft_ms=1., tpot_ms=2.)}
            self.assertEqual(len(check_online_evidence(root / "ranks", identity, checkpoints, response)), 4)
            performance = online_record(target_record, response, config={"test": True})
            self.assertEqual(len(performance["chunks"]), 3)
            self.assertEqual(performance["recovery_ms"],
                             sum(c["timing_ms"]["chunk_recovery_ms"] for c in online["chunks"]))
            path = root / "ranks/pause-rank-1.json"
            original = json.loads(path.read_text())
            for mutation in ("missing_chunk", "wrong_slot", "wrong_target", "missing_layer", "rollback",
                             "late_commit", "bad_hash", "duplicate_checkpoint"):
                record = deepcopy(original)
                chunks = record["online"]["chunks"]
                if mutation == "missing_chunk":
                    chunks.pop()
                elif mutation == "wrong_slot":
                    chunks[1]["physical_slots_sha256"] = chunks[0]["physical_slots_sha256"]
                elif mutation == "wrong_target":
                    chunks[1]["recovered_rank"] = 0
                elif mutation == "missing_layer":
                    chunks[1]["layer_validation"].pop()
                elif mutation == "rollback":
                    chunks[1]["rollback_used_for_recovery"] = True
                elif mutation == "late_commit":
                    record["protection"]["chunks"][2]["committed_monotonic"] = 4.
                elif mutation == "bad_hash":
                    chunks[2]["restored_sha256"] = "bad"
                else:
                    chunks[1]["checkpoint"] = chunks[0]["checkpoint"]
                path.write_text(json.dumps(record))
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                    check_online_evidence(root / "ranks", identity, checkpoints, response)


if __name__ == "__main__":
    unittest.main()
