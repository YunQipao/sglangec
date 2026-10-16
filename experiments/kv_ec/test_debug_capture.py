"""Synthetic/CPU tests ONLY; no claim of real KV or CUDA completion validation."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.kv_ec.debug_capture import (
    CaptureConfig, CaptureRejected, maybe_capture, select_raw_bytes, validate_lifetime,
)
from experiments.kv_ec.rank_snapshot import validate_payload, verify, write_rank_files
from experiments.kv_ec.snapshot import digest


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


def rank_fixture(rank, tp, layers=(0,), count=4):
    chunks, records = [], []
    offset = 0
    kh, vh = hashlib.sha256(), hashlib.sha256()
    for layer in layers:
        rng = np.random.default_rng(1000 + rank * 100 + layer)
        k, v = [rng.integers(0, 256, count * 2 * 128 * 2, dtype=np.uint8) for _ in range(2)]
        shape = [count, 2, 128]
        records.append(dict(layer_id=layer, dtype="torch.bfloat16", original_shape={"k": [64, 2, 128], "v": [64, 2, 128]},
                            source_stride={"k": [256, 128, 1], "v": [256, 128, 1]}, packed_shape={"k": shape, "v": shape},
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
