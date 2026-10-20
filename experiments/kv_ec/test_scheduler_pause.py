"""CPU ownership/state tests. Fake CUDA events are not GPU pause evidence."""

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from experiments.kv_ec.cross_rank_control import request
from experiments.kv_ec.pause_control import PauseCoordinator, PauseControlServer
from experiments.kv_ec.scheduler_pause import SchedulerPauseGate, PauseRejected

ROOT = Path(__file__).resolve().parents[2]


def request_pool_class():
    # Execute the actual class, avoiding imports of GPU inference dependencies.
    path = ROOT / "python/sglang/srt/mem_cache/memory_pool.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "ReqToTokenPool")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [node],
                        type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ReqToTokenPool"]


class Event:
    complete = True
    def query(self):
        return self.complete


def fixture(root, coordinator, rank=0):
    pool = request_pool_class().__new__(request_pool_class())
    pool._alloc_size = 3
    pool.free_slots = [1]
    pool.req_generation = torch.tensor([0, 0, 5], dtype=torch.int64)
    pool.req_to_token = torch.zeros((3, 1024), dtype=torch.int32)
    allocator = TokenToKVPoolAllocator(1024, torch.bfloat16, "cpu", NS(), need_sort=False)
    slots = allocator.alloc(768)
    pool.req_to_token[2, :768] = slots.to(torch.int32)
    req = NS(rid="artificial", req_pool_idx=2, origin_input_ids=[1] * 768,
             output_ids=[2], finished=lambda: False, is_retracted=False, to_finish=None)
    scheduler = NS(
        server_args=NS(disable_overlap_schedule=True, disable_radix_cache=True,
                       chunked_prefill_size=-1, max_running_requests=1),
        model_worker=NS(ps=NS(tp_rank=rank, tp_size=4, attn_tp_size=4)),
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
        waiting_queue=[], forward_stream=object(),
    )
    identity = {"epoch": coordinator.epoch, "transaction_id": coordinator.transaction_id}
    gate = SchedulerPauseGate(scheduler, target=req.rid, identity=identity,
                              socket="/tmp/unused-pause.sock", output=str(root / str(rank)))
    gate.model_identity = {"model_identifier": "fixture", "model_sha256": "fixture"}
    event = Event()
    def record(lease_id, *, stream):
        assert stream is scheduler.forward_stream
        allocator.attach_slot_lease_cuda_event(lease_id, event)
        return event
    allocator.record_slot_lease_cuda_event = record
    gate._check_runtime = lambda: None  # Scope/type checks need real SGLang/GPU.
    batch = NS(
        reqs=[req],
        prefix_lens=[0],
        forward_mode=NS(is_extend=lambda: True, is_decode=lambda: False),
    )
    return gate, req, batch, event


def observation():
    return dict(event_complete=True, request_hash="r", request_version="v1",
                token_count=768, model_identifier="fixture", model_sha256="fixture")


class PauseCoordinatorTests(unittest.TestCase):
    def test_incomplete_duplicate_failure_and_deadline_are_fail_closed(self):
        clock = [0.]
        c = PauseCoordinator("tx", timeout_s=1, clock=lambda: clock[0])
        def call(op, **values):
            return c.dispatch(dict(epoch=c.epoch, transaction_id="tx", op=op, **values))
        with self.assertRaisesRegex(ValueError, "before_all"):
            call("authorize")
        for rank in (0, 1, 2):
            call("paused", tp_rank=rank, observation=observation())
        call("paused", tp_rank=1, observation=observation())
        self.assertEqual(c.state, "PREPARE")
        with self.assertRaises(ValueError):
            call("validated", tp_rank=0)
        clock[0] = 2.
        self.assertEqual(call("paused", tp_rank=3, observation=observation())["state"], "FAILED")
        self.assertEqual(call("authorize")["state"], "FAILED")

    def test_changed_observation_and_epoch_rejected(self):
        c = PauseCoordinator("tx")
        message = dict(epoch=c.epoch, transaction_id="tx", op="paused",
                       tp_rank=0, observation=observation())
        c.dispatch(message)
        bad = deepcopy(message)
        bad["observation"]["request_version"] = "other"
        self.assertEqual(c.dispatch(bad)["state"], "FAILED")
        restarted = PauseCoordinator("tx")
        with self.assertRaisesRegex(ValueError, "identity_changed"):
            restarted.dispatch(message)

    def test_real_socket_concurrent_handshake_and_disconnect(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pause.sock"
            try:
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.bind(str(path))
            except PermissionError:
                self.skipTest("sandbox disallows Unix domain sockets")
            path.unlink()
            c = PauseCoordinator("tx")
            server = PauseControlServer(path, c)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            self.assertTrue(server.ready.wait(2))
            def call(op, **values):
                return request(path, dict(epoch=c.epoch, transaction_id="tx",
                                          op=op, **values), timeout=1)["status"]
            try:
                with ThreadPoolExecutor(max_workers=4) as workers:
                    list(workers.map(lambda rank: call("paused", tp_rank=rank,
                                                       observation=observation()), range(4)))
                self.assertEqual(call("status")["state"], "PAUSED")
                # Disconnect before a response. A subsequent request still works.
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(path))
                call("authorize")
                for op in ("validated", "released", "armed"):
                    with ThreadPoolExecutor(max_workers=4) as workers:
                        list(workers.map(lambda rank: call(op, tp_rank=rank), range(4)))
                self.assertEqual(call("status")["state"], "RESUME_AUTHORIZED")
            finally:
                server.stop()
                thread.join(3)
                self.assertFalse(thread.is_alive())


class SchedulerPauseTests(unittest.TestCase):
    def test_four_gates_hold_real_allocator_and_request_row_until_authorized(self):
        with tempfile.TemporaryDirectory() as root:
            coordinator = PauseCoordinator("tx")
            fixtures = [fixture(Path(root), coordinator, rank) for rank in range(4)]
            gates = [item[0] for item in fixtures]
            def dispatch(path, message, **kwargs):
                return {"state": "OK", "status": coordinator.dispatch(message)}
            with patch("experiments.kv_ec.scheduler_pause.request", side_effect=dispatch):
                for gate, req, batch, event in fixtures:
                    gate.after_result(batch)
                    gate._vote = lambda ready: ready and all(
                        item.state in ("RESUME_READY", "RESUMED") for item in gates)
                    self.assertTrue(gate.tick())
                    s = gate.scheduler
                    with self.assertRaisesRegex(RuntimeError, "row_protected"):
                        s.req_to_token_pool.free(req)
                    with self.assertRaisesRegex(RuntimeError, "row_protected"):
                        s.req_to_token_pool.clear()
                    with self.assertRaisesRegex(RuntimeError, "row_protected"):
                        s.req_to_token_pool.alloc([])
                    with self.assertRaisesRegex(RuntimeError, "row_protected"):
                        s.req_to_token_pool.write((2, 0), 999)
                    with self.assertRaisesRegex(RuntimeError, "protected_slot_free"):
                        s.token_to_kv_pool_allocator.free(torch.arange(1, 769))
                self.assertEqual(coordinator.state, "PAUSED")
                coordinator.dispatch(dict(epoch=coordinator.epoch, transaction_id="tx",
                                          op="authorize"))
                for _ in range(20):
                    for gate in gates:
                        gate.next_poll = 0
                        gate.tick()
                    if all(gate.state == "RESUMED" for gate in gates):
                        break
                self.assertTrue(all(gate.state == "RESUMED" for gate in gates))
                self.assertEqual(coordinator.state, "RESUME_AUTHORIZED")
                for gate, req, batch, event in fixtures:
                    self.assertEqual(gate.scheduler.token_to_kv_pool_allocator
                                     .slot_lease_registry.active_lease_count, 0)
                    decode_batch = NS(
                        reqs=[req],
                        forward_mode=NS(is_extend=lambda: False, is_decode=lambda: True),
                    )
                    gate.after_result(decode_batch)
                    evidence = (Path(root) / str(gate.rank) /
                                f"pause-rank-{gate.rank}.json")
                    self.assertTrue(evidence.is_file())
                    self.assertFalse(evidence.with_name(
                        f"pause-rank-{gate.rank}.pending.json").exists())
                    self.assertTrue(
                        json.loads(evidence.read_text())["decode_observed"]
                    )
                    gate.scheduler.req_to_token_pool.free(req)
                    self.assertIsNone(req.req_pool_idx)

    def test_failures_never_resume_or_reuse(self):
        cases = ("cancel", "slot_generation", "row_generation", "coordinator_restart",
                 "event_pending_timeout", "authorization", "rank_vote_timeout")
        for failure in cases:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                c = PauseCoordinator("tx")
                gate, req, batch, event = fixture(Path(root), c)
                def dispatch(path, message, **kwargs):
                    return {"state": "OK", "status": c.dispatch(message)}
                with patch("experiments.kv_ec.scheduler_pause.request", side_effect=dispatch):
                    gate.after_result(batch)
                    gate._vote = lambda ready: ready
                    self.assertTrue(gate.tick())
                    if failure == "cancel":
                        req.to_finish = "cancelled"
                    elif failure == "slot_generation":
                        gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry.generation[1] += 1
                        gate.state = "RESUME_READY"
                    elif failure == "row_generation":
                        gate.scheduler.req_to_token_pool.req_generation[2] += 1
                        gate.state = "RESUME_READY"
                    elif failure == "coordinator_restart":
                        c = PauseCoordinator("tx")
                    elif failure == "event_pending_timeout":
                        event.complete = False
                        gate.state = "EVENT_PENDING"
                        gate.deadline = time.monotonic() - 1
                    elif failure == "authorization":
                        c.state, c.failure_reason = "FAILED", "authorization_denied"
                    else:
                        gate._vote = lambda ready: (_ for _ in ()).throw(TimeoutError("rank_missing"))
                    gate.next_poll = 0
                    with self.assertRaises(PauseRejected):
                        gate.tick()
                self.assertEqual(gate.state, "ABORTED_FAIL_CLOSED")
                registry = gate.scheduler.token_to_kv_pool_allocator.slot_lease_registry
                self.assertTrue(registry.closed)
                self.assertEqual(registry.active_lease_count, 1)
                with self.assertRaises(RuntimeError):
                    gate.scheduler.req_to_token_pool.free(req)
                with self.assertRaises(RuntimeError):
                    gate.scheduler.token_to_kv_pool_allocator.alloc(1)
                with self.assertRaisesRegex(RuntimeError, "registry_closed"):
                    gate.scheduler.token_to_kv_pool_allocator.enable_slot_leases(timeout_s=999)

    def test_actual_event_loop_skips_selection_and_forward(self):
        path = ROOT / "python/sglang/srt/managers/scheduler.py"
        cls = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "event_loop_normal")
        method.decorator_list = []
        namespace = {"envs": NS(SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda: False))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        calls = []
        ticks = iter([True, True, False])
        s = NS(gracefully_exit=False, running_batch=None, last_batch=None,
               request_receiver=NS(recv_requests=lambda: []),
               process_input_requests=lambda messages: None,
               _engine_paused=False,
               _kv_ec_pause_gate=NS(check_inputs=lambda messages: None,
                                    tick=lambda: next(ticks),
                                    after_result=lambda batch: calls.append("after_result")),
               get_next_batch_to_run=lambda **kw: NS(running_batch=None, batch_to_run="batch"),
               run_batch=lambda batch: calls.append("forward"),
               process_batch_result=lambda batch, result: setattr(s, "gracefully_exit", True))
        namespace["event_loop_normal"](s)
        self.assertEqual(calls, ["forward", "after_result"])


if __name__ == "__main__":
    unittest.main()
