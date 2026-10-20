"""CPU ownership/state tests. Fake CUDA events are not GPU pause evidence."""

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, call, patch

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
             output_ids=[2], finished=lambda: False, is_retracted=False, to_finish=None,
             session_id=None, multimodal_inputs=None, prefix_indices=[],
             inflight_middle_chunks=0, cache_protected_len=0, extend_range=NS(end=768))
    scheduler = NS(
        server_args=NS(disable_overlap_schedule=True, disable_radix_cache=True,
                       chunked_prefill_size=-1, max_running_requests=1),
        model_worker=NS(ps=NS(tp_rank=rank, tp_size=4, attn_tp_size=4)),
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
        waiting_queue=[], forward_stream=object(),
        enable_overlap=False, enable_pdmux=False, enable_hisparse=False,
        _engine_paused=False,
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
    # Pool/layout checks need real SGLang/GPU; lifetime checks remain real.
    gate._check_runtime = gate._check_request_lifetime
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
    def setUp(self):
        # CPU ownership tests must not select CUDA paths based on host hardware.
        self.enterContext(patch(
            "experiments.kv_ec.scheduler_pause.torch.cuda.is_available",
            return_value=False,
        ))
        self.current_stream = self.enterContext(patch(
            "experiments.kv_ec.scheduler_pause.torch.cuda.current_stream",
            side_effect=AssertionError("CPU fixture must not access a CUDA stream"),
        ))

    def test_chunk_cache_post_prefill_prefix_is_not_historical_reuse(self):
        path = ROOT / "python/sglang/srt/mem_cache/chunk_cache.py"
        cls = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "ChunkCache")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "cache_unfinished_req")
        module = ast.Module(
            body=ast.parse("from __future__ import annotations").body + [method],
            type_ignores=[])
        namespace = {"torch": torch}
        exec(compile(module, str(path), "exec"), namespace)
        with tempfile.TemporaryDirectory() as root:
            gate, req, batch, event = fixture(Path(root), PauseCoordinator("tx"))
            cache = NS(req_to_token_pool=gate.scheduler.req_to_token_pool)
            namespace["cache_unfinished_req"](cache, req)
            self.assertEqual(len(req.prefix_indices), 768)
            self.assertEqual(batch.prefix_lens, [0])
            gate.after_result(batch)
            self.assertEqual(gate.state, "EVENT_PENDING")
            self.assertEqual(gate.evidence["post_prefill_prefix_indices_count"], 768)

    def test_historical_or_unknown_prefill_prefix_is_rejected(self):
        for prefix in ([12], None):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as root:
                gate, req, batch, event = fixture(Path(root), PauseCoordinator("tx"))
                batch.prefix_lens = prefix
                with self.assertRaisesRegex(PauseRejected, "prefill_prefix_not_empty:actual="):
                    gate.after_result(batch)
                self.assertEqual(gate.state, "ABORTED_FAIL_CLOSED")
                self.assertEqual(gate.evidence["runtime_checks"]
                                 ["prefill_prefix_not_empty"]["actual"], prefix)

    def test_lifetime_guards_have_specific_safe_diagnostics(self):
        cases = [
            ("scheduler", "enable_overlap", True, "scheduler.enable_overlap"),
            ("scheduler", "enable_pdmux", True, "scheduler.enable_pdmux"),
            ("scheduler", "enable_hisparse", True, "scheduler.enable_hisparse"),
            ("scheduler", "_engine_paused", True, "scheduler._engine_paused"),
            ("request", "session_id", "private-session", "session_present"),
            ("request", "multimodal_inputs", NS(private="data"), "multimodal_present"),
            ("request", "inflight_middle_chunks", 1, "inflight_middle_chunks"),
            ("request", "cache_protected_len", 1, "cache_protected_len"),
        ]
        for owner, name, value, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as root:
                gate, req, batch, event = fixture(Path(root), PauseCoordinator("tx"))
                setattr(gate.scheduler if owner == "scheduler" else req, name, value)
                with self.assertRaisesRegex(PauseRejected, reason + ":actual="):
                    gate.after_result(batch)
                evidence = json.loads(gate.path.with_suffix(".error.json").read_text())
                check = evidence["runtime_checks"]["unsupported_request_lifetime:" + reason]
                self.assertIn("actual", check)
                self.assertIn("expected", check)
                self.assertNotIn("private-session", json.dumps(evidence))
                self.assertNotIn('"private"', json.dumps(evidence))

    def test_full_runtime_guard_and_specific_rejection_reasons(self):
        with tempfile.TemporaryDirectory() as root:
            gate, req, batch, event = fixture(Path(root), PauseCoordinator("tx"))
            gate.req = req
            s = gate.scheduler
            pool_class = type("MHATokenToKVPool", (), {})
            pool = pool_class()
            pool.__dict__.update(
                dtype=torch.bfloat16, store_dtype=torch.bfloat16, kv_cache_layout="nhd",
                use_hnd=False, page_size=1, start_layer=0, layer_num=36,
                is_quantized_kv_cache=False, post_capture_active=False,
                layer_transfer_counter=None)
            hf = NS(model_type="qwen3", num_hidden_layers=36, num_key_value_heads=8, head_dim=128)
            s.model_worker.model_runner = NS(
                token_to_kv_pool=pool, req_to_token_pool=s.req_to_token_pool,
                model_config=NS(hf_config=hf))
            s.model_worker.ps.attn_cp_size = s.model_worker.ps.attn_dcp_size = 1
            s.spec_algorithm = NS(is_none=lambda: True)
            s.token_to_kv_pool_allocator._kvcache = pool
            s.server_args.pp_size = 1
            s.server_args.enable_lora = False
            s.server_args.quantization = None
            module = NS(MHATokenToKVPool=pool_class, ReqToTokenPool=type(s.req_to_token_pool))
            with patch.dict(sys.modules, {"sglang.srt.mem_cache.memory_pool": module}):
                SchedulerPauseGate._check_runtime(gate)
                cases = [
                    (s.model_worker.ps, "tp_size", 2, "only_plain_tp4:tp_size"),
                    (s.server_args, "pp_size", 2, "unsupported_parallel:pp_size"),
                    (s.server_args, "enable_lora", True, "unsupported_runtime:enable_lora"),
                    (s.server_args, "disable_radix_cache", False,
                     "unsupported_scheduler:disable_radix_cache"),
                    (s.server_args, "quantization", "unsupported",
                     "unsupported_scheduler:quantization_present"),
                    (hf, "num_hidden_layers", 32, "unsupported_qwen3_pool_layout:num_hidden_layers"),
                    (pool, "dtype", torch.float16, "unsupported_qwen3_pool_layout:dtype"),
                    (pool, "layer_transfer_counter", object(),
                     "unsupported_qwen3_pool_layout:layer_transfer_counter_present"),
                    (s, "enable_overlap", True, "unsupported_request_lifetime:scheduler.enable_overlap"),
                    (req, "session_id", "private-session", "unsupported_request_lifetime:session_present"),
                ]
                for owner, name, value, reason in cases:
                    with self.subTest(reason=reason):
                        old = getattr(owner, name)
                        setattr(owner, name, value)
                        try:
                            with self.assertRaisesRegex(PauseRejected, reason + ":actual="):
                                SchedulerPauseGate._check_runtime(gate)
                        finally:
                            setattr(owner, name, old)
                s.model_worker.model_runner.token_to_kv_pool = NS()
                with self.assertRaisesRegex(PauseRejected, "plain_pool_allocator_required:kv_pool"):
                    SchedulerPauseGate._check_runtime(gate)

    def test_producer_fence_precedes_slot_lease_event(self):
        with tempfile.TemporaryDirectory() as root:
            gate, req, batch, event = fixture(Path(root), PauseCoordinator("tx"))
            scheduler = gate.scheduler
            producer_stream = object()
            producer_event = Mock(spec=["record"])
            scheduler.device_module = NS(Event=Mock(return_value=producer_event))
            scheduler.forward_stream = Mock(spec=["wait_event"])
            allocator = scheduler.token_to_kv_pool_allocator
            calls = Mock()
            calls.attach_mock(scheduler.device_module.Event, "create")
            calls.attach_mock(producer_event.record, "record")
            calls.attach_mock(scheduler.forward_stream.wait_event, "wait")
            allocator.record_slot_lease_cuda_event = Mock(
                wraps=allocator.record_slot_lease_cuda_event)
            calls.attach_mock(allocator.record_slot_lease_cuda_event, "lease_event")
            with patch("experiments.kv_ec.scheduler_pause.torch.cuda.is_available",
                       return_value=True), patch(
                           "experiments.kv_ec.scheduler_pause.torch.cuda.current_stream",
                           return_value=producer_stream):
                gate.after_result(batch)
            self.assertEqual(calls.mock_calls, [
                call.create(),
                call.record(stream=producer_stream),
                call.wait(producer_event),
                call.lease_event(gate.owner, stream=scheduler.forward_stream),
            ])
            self.assertEqual(gate.state, "EVENT_PENDING")
            self.assertIs(gate.event, event)
            self.current_stream.assert_not_called()

    def test_producer_fence_failure_retains_protections(self):
        for failure in ("create", "record", "wait"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                coordinator = PauseCoordinator("tx")
                gate, req, batch, event = fixture(Path(root), coordinator)
                scheduler = gate.scheduler
                producer_event = Mock(spec=["record"])
                scheduler.device_module = NS(Event=Mock(return_value=producer_event))
                scheduler.forward_stream = Mock(spec=["wait_event"])
                operation = {
                    "create": scheduler.device_module.Event,
                    "record": producer_event.record,
                    "wait": scheduler.forward_stream.wait_event,
                }[failure]
                operation.side_effect = RuntimeError("producer_fence_failed")
                allocator = scheduler.token_to_kv_pool_allocator
                allocator.record_slot_lease_cuda_event = Mock(
                    wraps=allocator.record_slot_lease_cuda_event)
                with patch("experiments.kv_ec.scheduler_pause.torch.cuda.is_available",
                           return_value=True), patch(
                               "experiments.kv_ec.scheduler_pause.torch.cuda.current_stream",
                               return_value=object()), patch(
                                   "experiments.kv_ec.scheduler_pause.request",
                                   side_effect=lambda path, message, **kwargs: {
                                       "state": "OK", "status": coordinator.dispatch(message)}):
                    with self.assertRaisesRegex(PauseRejected, "producer_fence_failed"):
                        gate.after_result(batch)
                self.assertEqual(gate.state, "ABORTED_FAIL_CLOSED")
                self.assertTrue(allocator.slot_lease_registry.closed)
                self.assertEqual(allocator.slot_lease_registry.active_lease_count, 1)
                allocator.record_slot_lease_cuda_event.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "row_protected"):
                    scheduler.req_to_token_pool.free(req)
                with self.assertRaisesRegex(RuntimeError, "registry_closed"):
                    allocator.alloc(1)

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
                    req.output_ids.append(3)
                    gate.after_result(decode_batch)
                    evidence = (Path(root) / str(gate.rank) /
                                f"pause-rank-{gate.rank}.json")
                    self.assertTrue(evidence.is_file())
                    self.assertFalse(evidence.with_name(
                        f"pause-rank-{gate.rank}.pending.json").exists())
                    self.assertTrue(
                        json.loads(evidence.read_text())["decode_observed"]
                    )
                    original = evidence.read_bytes()
                    # Every subsequent decode must leave immutable evidence
                    # alone, rather than crashing while recreating the file.
                    for token in range(4, 18):
                        req.output_ids.append(token)
                        gate.after_result(decode_batch)
                    self.assertEqual(evidence.read_bytes(), original)
                    self.assertEqual(len(req.output_ids), 16)
                    self.assertEqual(gate.evidence["decode_output_count"], 2)
                    self.assertTrue(gate.evidence["row_protection_released"])
                    self.assertEqual(gate.evidence["active_slot_leases_after_release"], 0)
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
