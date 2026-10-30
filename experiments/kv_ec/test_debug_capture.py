"""Synthetic/CPU tests ONLY; no claim of real KV or CUDA completion validation."""

from copy import deepcopy
import ast
from dataclasses import dataclass, fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.kv_ec.debug_capture import (
    CaptureConfig, CaptureRejected, maybe_capture, select_raw_bytes, validate_lifetime,
)
from experiments.kv_ec.rank_snapshot import validate_payload, verify, write_rank_files
from experiments.kv_ec.snapshot import digest
from experiments.kv_ec import debug_capture


ROOT = Path(__file__).resolve().parents[2]


def result_processor_class():
    """Execute the checkout's real class/methods without GPU-heavy imports.

    Keep its actual fields, slots, frozen decorator and prefill method unchanged;
    external inference dependencies are supplied by each CPU test.
    """
    path = ROOT / "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "SchedulerBatchResultProcessor")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [node],
                        type_ignores=[])
    namespace = {"dataclass": dataclass, "__name__": __name__, "os": os}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["SchedulerBatchResultProcessor"], namespace


def frozen_processor(rank=0, tp=1):
    cls, namespace = result_processor_class()
    scheduler, batch, req, config = lifetime_fixture()
    values = {f.name: None for f in fields(cls)}
    values.update(server_args=scheduler.server_args, enable_overlap=False,
                  tree_cache=scheduler.tree_cache, draft_worker=None,
                  model_worker=NS(ps=NS(tp_rank=rank, tp_size=tp)))
    return cls(**values), batch, req, config, namespace


def lifetime_fixture():
    args = NS(disable_overlap_schedule=True, disable_radix_cache=True, chunked_prefill_size=-1,
              enable_mixed_chunk=False, max_running_requests=1, pp_size=1, dp_size=1,
              dcp_size=1, nnodes=1, enable_dp_attention=False, enable_unified_memory=False,
              enable_page_major_kv_layout=False, enable_hierarchical_cache=False,
              enable_hisparse=False, cpu_offload_gb=0, disaggregation_mode="null",
              quantization=None, enable_lora=False, kv_cache_dtype="auto", page_size=1,
              cuda_graph_config=NS(decode=NS(backend="disabled"), prefill=NS(backend="disabled")))
    scheduler = NS(server_args=args, enable_overlap=False, tree_cache=NS(disable=True),
                   spec_algorithm=NS(is_none=lambda: True), enable_pdmux=False)
    req = NS(rid="artificial", inflight_middle_chunks=0, output_ids=[], finished=lambda: False,
             is_retracted=False, session_id=None, multimodal_inputs=None, prefix_indices=[],
             origin_input_ids=[1] * 384, extend_range=NS(start=0, end=384), kv_committed_len=384,
             req_pool_idx=7)
    batch = NS(reqs=[req], forward_mode=NS(name="EXTEND"), decoding_reqs=[], prefix_lens=[0],
               seq_lens_cpu=torch.tensor([384]))
    config = CaptureConfig(Path("/tmp"), "test", "artificial", 0, 128, "0", 1024**2)
    return scheduler, batch, req, config


def rank_fixture(rank, tp, layers=(0,), count=4, heads=2):
    chunks, records = [], []
    offset = 0
    kh, vh = hashlib.sha256(), hashlib.sha256()
    for layer in layers:
        rng = np.random.default_rng(1000 + rank * 100 + layer)
        k, v = [rng.integers(0, 256, count * heads * 128 * 2, dtype=np.uint8) for _ in range(2)]
        shape = [count, heads, 128]
        records.append(dict(layer_id=layer, dtype="torch.bfloat16", original_shape={"k": [max(64, count), heads, 128], "v": [max(64, count), heads, 128]},
                            source_stride={"k": [heads * 128, 128, 1], "v": [heads * 128, 128, 1]}, packed_shape={"k": shape, "v": shape},
                            device="cpu", k_offset=offset, k_nbytes=k.size, k_sha256=digest(k),
                            v_offset=offset+k.size, v_nbytes=v.size, v_sha256=digest(v)))
        offset += k.size + v.size
        chunks.extend([k, v])
        kh.update(k)
        vh.update(v)
    payload = np.concatenate(chunks)
    meta = dict(format="kv-ec-rank-bin-v1", provenance="synthetic_fixture", snapshot_id="fixture",
                request_id="artificial-hash", request_id_encoding="sha256", model_identifier="fixture-model",
                model_sha256="fixture", source_commit="fixture", core_diff_sha256="fixture",
                tp_size=tp, tp_rank=rank, layers=list(layers), token_start=0, token_count=count,
                token_ids_sha256="fixture", dtype="torch.bfloat16", packed_byte_count=payload.size,
                pool_class="synthetic_fixture", lifetime="fixture", completion_dependency="fixture",
                snapshot_completion_status="complete", req_pool_idx=rank+1, req_generation=rank+3,
                physical_slots=[rank*10+i+1 for i in range(count)], source_unchanged=True,
                sha256=digest(payload), k_sha256=kh.hexdigest(), v_sha256=vh.hexdigest(), layer_records=records)
    return payload, meta


class LifetimeTests(unittest.TestCase):
    def test_shared_gpu_mode_is_opt_in_and_reaches_launcher_budget_check(self):
        from experiments.kv_ec import launch_debug
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, "argv", [
                    "launch_debug", "--mode", "baseline", "--tp", "4", "--gpus", "0,1,2,3",
                    "--output", directory, "--snapshot-id", "shared", "--layers", "all",
                    "--allow-shared-gpu-debug", "--mem-fraction-static", ".3",
                    "--context-length", "1024"]), \
                patch.dict(os.environ, {}, clear=True), \
                patch("experiments.kv_ec.launch_debug.check_port", return_value="available"), \
                patch("experiments.kv_ec.debug_resources.admit",
                      return_value=[{"gpu_index": str(i), 'gpu_uuid': f'GPU-test-{i}'} for i in range(4)]) as budget, \
                patch("experiments.kv_ec.launch_debug.os.execv") as execute:
            launch_debug.main()
            budget.assert_called_once_with(["0", "1", "2", "3"],
                                           "/media/sda1/iic_data/luotao/models/Qwen3-8B",
                                           4, 1024, .3, shared=True, minimum=0)
            argv = execute.call_args.args[1]
            self.assertEqual(argv[argv.index("--mem-fraction-static") + 1], "0.3")
            self.assertEqual(argv[argv.index("--context-length") + 1], "1024")

    def test_tp2_launcher_parity_modes_and_baseline(self):
        from experiments.kv_ec import launch_debug

        for mode, protection in (("parity", "FAST_PARITY"), ("parity", "DEBUG_SAFE"),
                                 ("baseline", "FAST_PARITY")):
            with self.subTest(mode=mode, protection=protection), tempfile.TemporaryDirectory() as directory:
                argv = ["launch_debug", "--mode", mode, "--protection-mode", protection,
                        "--tp", "2", "--gpus", "1,2", "--output", directory,
                        "--snapshot-id", "tp2", "--parity-socket", "/tmp/test-parity.sock",
                        "--mem-fraction-static", ".24", "--context-length", "1024"]
                with patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True), \
                     patch("experiments.kv_ec.debug_resources.admit", return_value=[
                         {'gpu_index': str(i), 'gpu_uuid': f'GPU-test-{i}'} for i in (1, 2)]), \
                     patch("experiments.kv_ec.launch_debug.socket.socket"), \
                     patch("experiments.kv_ec.parity_manager.manager_hello", return_value="test-epoch"), \
                     patch.object(debug_capture, "source_identity", return_value={}), \
                     patch("experiments.kv_ec.launch_debug.os.execv") as execute:
                    launch_debug.main()
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-test-1,GPU-test-2")
                    self.assertEqual(os.environ["SGLANG_KV_EC_DEBUG"], "0" if mode == "baseline" else "1")
                    if mode == "parity":
                        self.assertEqual(os.environ["SGLANG_KV_EC_PROTECTION_MODE"], protection)
                        self.assertEqual(os.environ["SGLANG_KV_EC_PARITY_SOCKET"], "/tmp/test-parity.sock")
                    else:
                        self.assertNotIn("SGLANG_KV_EC_PARITY_SOCKET", os.environ)
                    args = execute.call_args.args[1]
                    self.assertEqual(args[args.index("--tp-size") + 1], "2")
                    self.assertEqual(args[args.index("--mem-fraction-static") + 1], "0.24")
                    self.assertEqual(args[args.index("--context-length") + 1], "1024")

    def test_supported_guard(self):
        validate_lifetime(*lifetime_fixture())

    def test_unsafe_modes_rejected_before_pool_read(self):
        mutations = {"disable_overlap_schedule": False, "disable_radix_cache": False,
                     "chunked_prefill_size": 128, "max_running_requests": 2,
                     "enable_mixed_chunk": True, "pp_size": 2, "dp_size": 2,
                     "dcp_size": 2, "nnodes": 2, "enable_unified_memory": True,
                     "enable_hierarchical_cache": True, "enable_hisparse": True,
                     "enable_page_major_kv_layout": True, "page_size": 16,
                     "kv_cache_dtype": "fp8_e4m3", "enable_lora": True,
                     "disaggregation_mode": "prefill", "cpu_offload_gb": 1}
        for key, value in mutations.items():
            s, b, r, c = lifetime_fixture()
            setattr(s.server_args, key, value)
            with self.subTest(key=key), self.assertRaises(CaptureRejected):
                validate_lifetime(s, b, r, c)

    def test_invalid_lifetimes_and_ranges(self):
        for mutate in (lambda s,b,r: setattr(r,"req_pool_idx",None),
                       lambda s,b,r: setattr(r,"is_retracted",True),
                       lambda s,b,r: r.output_ids.append(9),
                       lambda s,b,r: setattr(r,"kv_committed_len",383),
                       lambda s,b,r: setattr(r.extend_range,"end",128),
                       lambda s,b,r: setattr(b.forward_mode,"name","MIXED"),
                       lambda s,b,r: setattr(s.server_args.cuda_graph_config.prefill,"backend","full")):
            s,b,r,c = lifetime_fixture()
            mutate(s,b,r)
            with self.assertRaises(CaptureRejected):
                validate_lifetime(s,b,r,c)
        s,b,r,c = lifetime_fixture()
        c = CaptureConfig(c.output,c.snapshot_id,c.target_rid,300,128,"0",1024**2)
        with self.assertRaises(CaptureRejected):
            validate_lifetime(s,b,r,c)

    def test_disabled_is_noop_and_only_target_is_captured(self):
        with patch.dict(os.environ, {"SGLANG_KV_EC_DEBUG":"0"}, clear=True):
            maybe_capture(None,None,None)
        with patch.dict(os.environ, {"SGLANG_KV_EC_DEBUG":"1", "SGLANG_KV_EC_REQUEST_ID":"other"}, clear=True):
            maybe_capture(NS(),None,NS(rid="not-target"))

    def test_runtime_bridge_failure_is_local_and_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
            "SGLANG_KV_EC_DEBUG": "1", "SGLANG_KV_EC_REQUEST_ID": "artificial",
            "SGLANG_KV_EC_SNAPSHOT_ID": "bridge-failure", "SGLANG_KV_EC_OUTPUT_DIR": d,
        }, clear=True):
            # No model worker is supplied: this deliberately exercises the
            # fail-closed bridge path, including rank-safe error reporting.
            maybe_capture(NS(model_worker=NS(ps=NS(tp_rank=0))), None, NS(rid="artificial"))

    def test_logical_slot_order_and_no_source_mutation(self):
        bits = torch.arange(64, dtype=torch.int16)
        tensor = bits.view(torch.bfloat16).reshape(8,2,4)
        before = bits.clone()
        slots = torch.tensor([6,2,7])
        raw = select_raw_bytes(tensor,slots)
        expected = torch.cat([tensor[i].view(torch.uint8).reshape(-1) for i in slots])
        self.assertTrue(torch.equal(raw, expected))
        self.assertTrue(torch.equal(bits,before))


class RealCallPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = {"SGLANG_KV_EC_DEBUG": "1", "SGLANG_KV_EC_REQUEST_ID": "artificial",
                    "SGLANG_KV_EC_SNAPSHOT_ID": "regression", "SGLANG_KV_EC_OUTPUT_DIR": self.tmp.name}
        self.env_patch = patch.dict(os.environ, self.env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.state_patch = patch.object(debug_capture, "_ATTEMPTED", set())
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def test_minimal_original_frozen_slots_failure(self):
        @dataclass(kw_only=True, frozen=True, slots=True)
        class Minimal:
            model_worker: object

        processor = Minimal(model_worker=NS(ps=NS(tp_rank=0)))
        # This is the original failing statement, requiring no CUDA/SGLang.
        with self.assertRaises(TypeError):
            processor._kv_ec_debug_attempted = True

    def test_real_frozen_fields_lifetime_and_success_delegation(self):
        s, b, r, c, _ = frozen_processor()
        self.assertFalse(hasattr(s, "ps"))
        self.assertFalse(hasattr(s, "__dict__"))
        validate_lifetime(s, b, r, c)
        with patch.object(debug_capture, "capture", return_value={}) as capture_mock:
            maybe_capture(s, b, r)
        capture_mock.assert_called_once()
        self.assertIs(capture_mock.call_args.args[0], s)
        self.assertEqual(capture_mock.call_args.args[3].snapshot_id, "regression")

    def test_chunk_plan_captures_contiguous_independent_snapshots(self):
        s, b, r, _, _ = frozen_processor()
        os.environ["SGLANG_KV_EC_TOKEN_COUNT"] = "256"
        os.environ["SGLANG_KV_EC_CHUNK_PLAN"] = json.dumps(
            {"artificial": ["prefill-chunk-1", "prefill-chunk-2", "prefill-chunk-3"]}
        )
        with patch.object(debug_capture, "capture", return_value={}) as capture_mock:
            maybe_capture(s, b, r)
        self.assertEqual(capture_mock.call_count, 3)
        configs = [call.args[3] for call in capture_mock.call_args_list]
        self.assertEqual(
            [(config.snapshot_id, config.token_start, config.token_count) for config in configs],
            [
                ("prefill-chunk-1", 0, 256),
                ("prefill-chunk-2", 256, 256),
                ("prefill-chunk-3", 512, 256),
            ],
        )

    def test_default_disabled_and_non_target_never_capture(self):
        s, b, r, _, _ = frozen_processor()
        with patch.object(debug_capture, "capture") as capture_mock:
            os.environ.pop("SGLANG_KV_EC_DEBUG")
            maybe_capture(None, None, None)
            os.environ["SGLANG_KV_EC_DEBUG"] = "1"
            r.rid = "not-target"
            maybe_capture(s, b, r)
        capture_mock.assert_not_called()
        self.assertFalse(debug_capture._ATTEMPTED)

    def test_one_attempt_per_snapshot_request_rank_and_process(self):
        s, b, r, _, _ = frozen_processor()
        with patch.object(debug_capture, "capture") as capture_mock:
            maybe_capture(s, b, r)
            maybe_capture(s, b, r)
            # A replacement result processor does not reset the protection.
            s2, _, _, _, _ = frozen_processor()
            maybe_capture(s2, b, r)
            self.assertEqual(capture_mock.call_count, 1)
            os.environ["SGLANG_KV_EC_SNAPSHOT_ID"] = "next"
            maybe_capture(s, b, r)
            os.environ["SGLANG_KV_EC_REQUEST_ID"] = r.rid = "next-request"
            maybe_capture(s, b, r)
            with patch.object(debug_capture.os, "getpid", return_value=-123):
                maybe_capture(s, b, r)
            self.assertEqual(capture_mock.call_count, 4)

    def test_tp1_tp4_failure_sidecars_and_no_retry(self):
        for tp in (1, 4):
            os.environ["SGLANG_KV_EC_SNAPSHOT_ID"] = f"tp{tp}"
            for rank in range(tp):
                s, b, r, _, _ = frozen_processor(rank, tp)
                with patch.object(debug_capture, "capture", side_effect=CaptureRejected("test_rejected")) as mock:
                    maybe_capture(s, b, r)
                    maybe_capture(s, b, r)
                mock.assert_called_once()
                path = Path(self.tmp.name) / f"tp{tp}" / f"rank_{rank}.error.json"
                self.assertEqual(json.loads(path.read_text()), {
                    "snapshot_completion_status": "failed", "reason": "test_rejected", "tp_rank": rank})

    def test_real_capture_guard_rejection_is_reported(self):
        s, b, r, _, _ = frozen_processor()
        s.server_args.disable_overlap_schedule = False
        # Not mocked: enters the existing real capture function, rejects before GPU imports.
        maybe_capture(s, b, r)
        error = json.loads((Path(self.tmp.name) / "regression/rank_0.error.json").read_text())
        self.assertEqual(error["reason"], "overlap_not_supported")

    def test_pdmux_rejected_from_server_args(self):
        s, b, r, c, _ = frozen_processor()
        s.server_args.enable_pdmux = True
        with self.assertRaisesRegex(CaptureRejected, "offload_or_pdmux_not_supported"):
            validate_lifetime(s, b, r, c)

    def test_existing_evidence_is_not_overwritten_and_inference_continues(self):
        s, b, r, _, _ = frozen_processor()
        directory = Path(self.tmp.name) / "regression"
        directory.mkdir()
        error_path = directory / "rank_0.error.json"
        error_path.write_text("prior evidence")
        with patch.object(debug_capture, "capture", side_effect=FileExistsError), self.assertLogs(
            debug_capture.logger, level="WARNING"
        ) as logs:
            maybe_capture(s, b, r)
        self.assertEqual(error_path.read_text(), "prior evidence")
        self.assertTrue(any("error sidecar failed" in line for line in logs.output))

    def test_source_identity_actual_checkout_path(self):
        model = Path(self.tmp.name) / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"synthetic weights")
        identity = debug_capture.source_identity(model)
        self.assertEqual(identity["model_identifier"], str(model))
        self.assertEqual(len(identity["model_sha256"]), 64)
        self.assertEqual(len(identity["source_commit"]), 40)

    def test_actual_prefill_method_continues_after_capture_error(self):
        s, b, r, _, namespace = frozen_processor()
        cls = type(s)
        # Load the actual bridge file without importing the heavy SGLang package.
        name = "sglang.srt.debug_utils.kv_ec"
        spec = importlib.util.spec_from_file_location(name, ROOT / "python/sglang/srt/debug_utils/kv_ec.py")
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        bridge._capture_module = debug_capture
        namespace.update(get_memory=lambda: NS(enable_hisparse=False),
                         maybe_cache_unfinished_req=Mock())
        values = {f.name: getattr(s, f.name) for f in fields(s)}
        allocator, streamer, reporter = Mock(), Mock(), Mock()
        values.update(is_generation=True, token_to_kv_pool_allocator=allocator,
                      output_streamer=streamer, metrics_reporter=reporter)
        s = cls(**values)
        r.time_stats = Mock()
        r.update_finish_state = Mock()
        r.return_sampling_mask = False
        r.grammar = None
        b.return_hidden_states = b.return_logprob = False
        b.prefill_stats = b.dp_cooperation_info = None
        result = NS(copy_done=None, routed_experts_output=None, indexer_topk_output=None,
                    logits_output=NS(), next_token_ids=torch.tensor([42]),
                    extend_input_len_per_req=None, extend_logprob_start_len_per_req=None,
                    can_run_cuda_graph=False)
        with patch.dict(sys.modules, {name: bridge}), patch.multiple(
            cls, move_logprobs_to_cpu=Mock(), _validate_pp_skip_output_comm=Mock(),
            _get_prefill_hidden_capture_mode=Mock(), _maybe_update_reasoning_tokens=Mock(),
            _maybe_collect_customized_info=Mock()
        ), patch.object(debug_capture, "capture", side_effect=CaptureRejected("test_rejected")):
            s.process_batch_result_prefill(b, result)
        self.assertEqual(r.output_ids, [42])
        r.update_finish_state.assert_called_once()
        allocator.free_group_end.assert_called_once()
        streamer.stream_output.assert_called_once()
        reporter.report_prefill_stats.assert_called_once()
        self.assertTrue((Path(self.tmp.name) / "regression/rank_0.error.json").is_file())


class RankFormatTests(unittest.TestCase):
    def test_tp1_and_tp4_layer0_and_36_layers(self):
        for tp in (1,4):
            for layers,count in (((0,),128),((0,),256),(tuple(range(36)),256)):
                with self.subTest(tp=tp,layers=len(layers),count=count), tempfile.TemporaryDirectory() as d:
                    for rank in range(tp):
                        payload,meta = rank_fixture(rank,tp,layers,count)
                        write_rank_files(d,payload,meta)
                    report=verify(d)
                    self.assertEqual(report["provenance"],"synthetic_fixture")
                    self.assertEqual(len(report["checks"]),tp if tp>1 else 0)
                    self.assertTrue(all(c["all_K_V_checksums_equal"] for c in report["checks"]))
                    self.assertEqual(json.loads((Path(d)/"manifest.json").read_text())["tp_size"],tp)
                    with self.assertRaises(FileExistsError):
                        verify(d)

    def test_reject_corruption_and_shape_metadata(self):
        payload,meta=rank_fixture(0,4)
        bad=payload.copy()
        bad[0]^=1
        with self.assertRaises(ValueError): validate_payload(bad,meta)
        badmeta=deepcopy(meta)
        badmeta["layer_records"][0]["packed_shape"]["k"][1]=3
        with self.assertRaises(ValueError): validate_payload(payload,badmeta)

    def test_reject_missing_rank_or_mismatched_capture(self):
        with tempfile.TemporaryDirectory() as d:
            p,m=rank_fixture(0,4)
            write_rank_files(d,p,m)
            with self.assertRaises(ValueError): verify(d)
            for rank in range(1,4):
                p,m=rank_fixture(rank,4)
                if rank==3:m["model_sha256"]="different"
                write_rank_files(d,p,m)
            with self.assertRaises(ValueError): verify(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
