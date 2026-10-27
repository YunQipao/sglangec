"""CPU state/protocol tests only. Mocked CUDA NEVER establishes GPU acceptance."""

import ast
from contextlib import ExitStack, nullcontext
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.sampling.sampling_params import SamplingParams
from .gpu_kv_restore import (
    RestoreRejected, RestoreStats, RestoreTransaction, allocate_restore_slots,
    validate_live_ownership,
)
from .gpu_resume_hook import GPUResumeHook, token_digest, validate_decode_observation, require_cuda_forward
from .node_block_export import sha
from .test_scheduler_pause import request_pool_class
from .verify_gpu_resume import check_results, compare_outputs, read_run, reference_identity, save_verdict


class ResumeFixture:
    """Real CPU allocator/row/transaction, artificial model inputs and logits."""
    identity = dict(model_identifier="cpu-fixture", model_sha256="cpu-fixture",
                    source_commit="cpu-test", core_diff_sha256="cpu-test")

    def __init__(self, root, mode, rank=0):
        self.root, self.mode, self.rank = Path(root), mode, rank
        self.root.mkdir()
        self.checkpoint = self.root.parent / "cpu-checkpoint.bin"
        if not self.checkpoint.exists():
            self.checkpoint.write_bytes(b"CPU TEST FIXTURE - NOT A REAL CHECKPOINT")
        rp_cls = request_pool_class()
        rp = rp_cls.__new__(rp_cls)
        rp._alloc_size, rp.size, rp.device = 2, 1, "cpu"
        rp.free_slots, rp.req_generation = [], torch.tensor([0, 7])
        rp.req_to_token = torch.zeros((2, 512), dtype=torch.int32)
        pool = NS(k_buffer=[torch.zeros((1025, 1, 1), dtype=torch.bfloat16)],
                  v_buffer=[torch.zeros((1025, 1, 1), dtype=torch.bfloat16)], size=1024,
                  layer_num=36, kv_cache_layout="nhd", use_hnd=False,
                  is_quantized_kv_cache=False, dtype=torch.bfloat16, store_dtype=torch.bfloat16,
                  start_layer=0, page_size=1)
        a = TokenToKVPoolAllocator(1024, torch.bfloat16, "cpu", pool, False)
        old = a.alloc(256)
        rp.req_to_token[1, :256] = old.int()
        req = NS(req_pool_idx=1, kv_committed_len=256, kv=NS(kv_allocated_len=256),
                 output_ids=[], origin_input_ids=[1] * 256, extend_batch_idx=1,
                 decode_batch_idx=0, cache_protected_len=0, inflight_middle_chunks=0,
                 is_retracted=False, to_finish=None, _finished=False,
                 prefix_indices=torch.empty(0, dtype=torch.int64))
        req.finished = lambda: req._finished
        backend = type("TritonAttnBackend", (), {})()
        backend.req_to_token = rp.req_to_token
        backend.token_to_kv_pool = pool
        runner = NS(attn_backend=backend, token_to_kv_pool=pool,
                    model_config=NS(hf_config=NS(model_type="qwen3", num_key_value_heads=8,
                        head_dim=128, hidden_size=4096, num_attention_heads=32)))
        s = NS(req_to_token_pool=rp, token_to_kv_pool_allocator=a,
               model_worker=NS(model_runner=runner, ps=NS(tp_rank=rank, tp_size=2)),
               tp_cpu_group=None, enable_overlap=False, forward_ct=0,
               future_map=NS(output_tokens_buf=torch.zeros(2, dtype=torch.int64)),
               server_args=NS(model_path="cpu-fixture", chunked_prefill_size=-1,
                              disable_radix_cache=True))
        records = [dict(layer_id=i, k_sha256="k", v_sha256="v") for i in range(36)]
        self.meta = dict(node_id=0, ranks=[dict(tp_rank=r, tp_size=2, dtype="torch.bfloat16",
            layers=list(range(36)), token_start=0, token_count=256,
            snapshot_completion_status="complete", source_unchanged=True,
            model_sha256="cpu-fixture", token_ids_sha256=token_digest(req.origin_input_ids),
            packed_byte_count=18874368, layer_records=deepcopy(records)) for r in (0, 1)])
        with ExitStack() as stack:
            stack.enter_context(patch("experiments.kv_ec.gpu_restore_hook.validate_runtime"))
            stack.enter_context(patch("experiments.kv_ec.gpu_restore_hook.validate_checkpoint", return_value=self.meta))
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(patch("torch.distributed.get_world_size", return_value=2))
            stack.enter_context(patch("experiments.kv_ec.debug_capture.source_identity", return_value=self.identity))
            h = GPUResumeHook(s, self.checkpoint, self.root, self.root.name, mode=mode)
        self.hook, self.req, self.scheduler = h, req, s
        self.batch = NS(reqs=[req], forward_mode=NS(name="EXTEND"), prefix_lens=[0],
                        extend_num_tokens=256, seq_lens_cpu=torch.tensor([256]))
        def vote(phase, ready):
            h._event("GLOBAL_" + phase + ("_OK" if ready else "_FAILED"))
            return ready
        h._vote = vote
        self.config = dict(experiment="gpu-resume-stage-b", run_id=self.root.name,
            mode=mode, decode_steps=16, checkpoint=str(self.checkpoint),
            checkpoint_sha256=sha(self.checkpoint.read_bytes()), runtime_identity=self.identity,
            argv=["python", "-m", "server", "--port", "30482", "--tp-size", "2"])

    def transaction(self):
        h, s, req = self.hook, self.scheduler, self.req
        a, rp = s.token_to_kv_pool_allocator, s.req_to_token_pool
        old = tuple(rp.req_to_token[1, :256].tolist())
        a.enable_slot_leases(active_slots=old)
        rp.protect_kv_ec_row(1, 7, "cpu-test")
        new, journal = allocate_restore_slots(a, 256, old)
        stats = RestoreStats("cpu", 0, self.rank, 2, 0, 256, 256, 18874368,
            layers_verified=[dict(x, k_byte_exact=True, v_byte_exact=True)
                             for x in self.meta["ranks"][self.rank]["layer_records"]])
        tx = RestoreTransaction(a, rp, req, 1, 7, old, new, 0, 256, stats,
                                state="VERIFIED", savepoint=journal, guard_owner="cpu-test")
        tx.ownership_check = lambda: validate_live_ownership(s, req, 0, 256,
            reserved=tx.old_slots if tx.mapping_installed else tx.new_slots)
        h.tx = tx
        return tx

    def admission(self):
        msg = type("TokenizedGenerateReqInput", (), {})()
        msg.input_ids = self.req.origin_input_ids
        msg.sampling_params = SamplingParams(max_new_tokens=self.hook.output_token_limit,
                                            temperature=0, ignore_eos=True)
        msg.sampling_params.normalize(None)
        msg.session_id = msg.mm_inputs = msg.input_embeds = None
        return msg

    def run(self, divergence=None, mock_cuda=True):
        h, s, req, batch = self.hook, self.scheduler, self.req, self.batch
        h.before_inputs([type("FreezeGCReq", (), {})(), self.admission()])
        def prepare():
            tx = self.transaction()
            if self.mode == "corrupt" and self.rank == 0:
                h.evidence.update(corruption_injected=True, corruption_detected=True)
                tx.abort()
                raise RestoreRejected("stage_b_precommit_byte_mismatch")
            return tx
        h._prepare_restore = prepare
        count = h.output_token_limit
        with ExitStack() as stack:
            if mock_cuda:
                stack.enter_context(patch("torch.cuda.synchronize"))
                stack.enter_context(patch("experiments.kv_ec.gpu_resume_hook.require_cuda_forward"))
            for step in range(count):
                if step:
                    batch.forward_mode.name = "DECODE"
                    req.decode_batch_idx += 1
                    req.kv_committed_len += 1
                    req.kv.kv_allocated_len += 1
                    batch.seq_lens_cpu += 1
                    loc = s.token_to_kv_pool_allocator.alloc(1)
                    s.req_to_token_pool.req_to_token[1, 255 + step] = loc[0].int()
                h.before_forward(batch)
                s.forward_ct += 1
                mapping = s.req_to_token_pool.req_to_token[1, :256 + step].clone()
                backend = s.model_worker.model_runner.attn_backend
                backend.forward_metadata = NS(kv_indptr=torch.tensor([0, len(mapping)]),
                                               kv_indices=mapping)
                fb = NS(batch_size=1, forward_mode=batch.forward_mode,
                    req_pool_indices=torch.tensor([1]),
                    input_ids=torch.tensor(req.origin_input_ids if step == 0 else [req.output_ids[-1]]),
                    positions=torch.arange(256) if step == 0 else torch.tensor([255 + step]),
                    seq_lens=torch.tensor([256 + step]),
                    out_cache_loc=mapping if step == 0 else mapping[-1:])
                sampled = step + 1 if divergence != step else step + 2
                logits = torch.zeros((1, 64))
                logits[0, sampled] = 10
                h.observe_model_forward(fb, NS(next_token_logits=logits), False)
                s.future_map.output_tokens_buf[1] = sampled
                h.after_forward(batch)
                req.output_ids.append(sampled)
                if step == count - 1:
                    req._finished = True
                    slots = s.req_to_token_pool.req_to_token[1, :req.kv_committed_len].clone()
                    s.token_to_kv_pool_allocator.free(slots)
                    s.req_to_token_pool.free(req)
                    req.kv = None
                else:
                    req.prefix_indices = s.req_to_token_pool.req_to_token[1, :256].long().clone()
                h.after_result(batch)
        run = self.root / "run.json"
        if run.exists():
            assert json.loads(run.read_text()) == self.config
        else:
            with run.open("x") as f:
                json.dump(self.config, f)
        return h.evidence


class ResumeLifecycleTests(unittest.TestCase):
    def test_reference_and_resume_each_run_sixteen_decode_steps_and_free_slots(self):
        with tempfile.TemporaryDirectory() as root:
            for mode in ("reference", "resume"):
                f = ResumeFixture(Path(root) / mode, mode)
                evidence = f.run()
                self.assertTrue(evidence["no_slot_leak"])
                self.assertTrue(evidence["generation_unchanged"])
                self.assertEqual(evidence["resume_decode_forward_count"], 16)
                self.assertEqual(evidence["resumed_generated_tokens"], 16)
                self.assertEqual(len(evidence["output_token_ids"]), 17)
                self.assertEqual(evidence["recomputed_prefix_tokens"], 0)
                if mode == "resume":
                    self.assertTrue(evidence["restored_slots_used_by_decode"])
                    self.assertTrue(evidence["request_state_unchanged_by_restore"])
                    self.assertNotEqual(evidence["old_slots"], evidence["new_slots"])
                # GPU flag here is MOCKED; only real server runs are acceptance.

    def test_normalized_greedy_admission_and_stochastic_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            f = ResumeFixture(Path(root) / "admit", "resume")
            msg = f.admission()
            self.assertEqual(msg.sampling_params.temperature, 1)
            self.assertEqual(msg.sampling_params.top_k, 1)
            msg.sampling_params.top_k = 2
            with self.assertRaisesRegex(RestoreRejected, "plain_greedy"):
                f.hook.before_inputs([msg])
            self.assertFalse(f.hook.admitted)
            f.hook.events.close()

    def test_reprefill_and_generation_change_fail_before_decode(self):
        for change in ("mode", "generation", "commit"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                f = ResumeFixture(Path(root) / "fail", "resume")
                h = f.hook
                h.admitted, h.prefill_seen = True, True
                h.req, h.row, h.generation = f.req, 1, 7
                h.prefix_slots = tuple(range(1, 257))
                h.evidence["committed"] = change != "commit"
                f.req.output_ids = [1]
                if change != "mode":
                    f.batch.forward_mode.name = "DECODE"
                if change == "generation":
                    f.scheduler.req_to_token_pool.req_generation[1] = 8
                with self.assertRaises(RestoreRejected):
                    h.before_forward(f.batch)
                self.assertEqual(h.evidence["resume_decode_forward_count"], 0)
                self.assertEqual(f.scheduler.forward_ct, 0)
                h.events.close()

    def test_negative_mode_aborts_journal_without_running_decode(self):
        with tempfile.TemporaryDirectory() as root:
            f = ResumeFixture(Path(root) / "negative", "corrupt")
            r = f.run()
            self.assertTrue(r["negative_test_pass"])
            self.assertFalse(r["pass_stage_b"])
            self.assertEqual(r["transaction_state"], "ABORTED")
            self.assertEqual(r["resume_decode_forward_count"], 0)
            self.assertTrue(r["no_slot_leak"])

    def test_actual_byte_flip_is_rejected_and_new_slots_returned(self):
        with tempfile.TemporaryDirectory() as root:
            f = ResumeFixture(Path(root) / "corrupt", "corrupt")
            f.hook.req, f.hook.row, f.hook.generation = f.req, 1, 7
            tx = f.transaction()
            pool = f.scheduler.model_worker.model_runner.token_to_kv_pool
            payload = bytes(256 * 4)  # 256 BF16 K + V values, initially zero.
            meta = dict(ranks=[dict(layer_records=[dict(layer_id=0, k_nbytes=512, v_nbytes=512)])])
            def read(pool, layer, slots, stream):
                return b"".join(buf[slots].view(torch.uint8).numpy().tobytes()
                                for buf in (pool.k_buffer[0], pool.v_buffer[0]))
            with patch("experiments.kv_ec.gpu_resume_hook.restore_to_gpu", return_value=tx), \
                 patch("experiments.kv_ec.gpu_resume_hook.unpack", return_value=(meta, [payload])), \
                 patch("experiments.kv_ec.gpu_resume_hook._read_layer", side_effect=read), \
                 patch("torch.cuda.synchronize"), patch("torch.cuda.current_stream", return_value=None):
                with self.assertRaisesRegex(RestoreRejected, "precommit_byte_mismatch"):
                    f.hook._prepare_restore()
            self.assertEqual(tx.state, "ABORTED")
            self.assertTrue(f.hook.evidence["corruption_detected"])
            self.assertEqual(f.scheduler.token_to_kv_pool_allocator.available_size(), 768)
            self.assertEqual(tuple(tx._mapping().tolist()), tx.old_slots)
            f.hook.events.close()


class DecodeObservationTests(unittest.TestCase):
    def observation(self):
        mapping = list(range(257, 514))
        return dict(step=1, outputs=[23], seq_len=257, committed_len=257,
                    input_ids=[23], positions=[256], indices=list(mapping), mapping=mapping,
                    prefix_slots=mapping[:256], out_slots=[513])

    def test_last_sampled_token_is_next_decode_input_without_existing_kv(self):
        validate_decode_observation(**self.observation())

    def test_wrong_token_position_old_slots_and_length_are_rejected(self):
        for key, value in (("outputs", [23, 24]), ("committed_len", 256),
                           ("input_ids", [24]), ("positions", [257]),
                           ("indices", list(range(1, 258))), ("out_slots", [257])):
            with self.subTest(field=key):
                o = self.observation()
                o[key] = value
                with self.assertRaises(RestoreRejected):
                    validate_decode_observation(**o)


class ResumeTPSafetyTests(unittest.TestCase):
    def pair(self, root, mode, fail_commit=False):
        fixtures = [ResumeFixture(Path(root) / f"rank{rank}", mode, rank) for rank in (0, 1)]
        barrier = threading.Barrier(2, timeout=10)
        votes = {}
        for rank, f in enumerate(fixtures):
            def vote(phase, ready, rank=rank, hook=f.hook):
                votes[rank] = (phase, ready)
                barrier.wait()
                unanimous = votes[0][0] == votes[1][0] and all(v[1] for v in votes.values())
                barrier.wait()
                hook._event("GLOBAL_" + phase + ("_OK" if unanimous else "_FAILED"))
                return unanimous
            f.hook._vote = vote
        if fail_commit:
            allocator = fixtures[1].scheduler.token_to_kv_pool_allocator
            original = allocator.free
            def fail(slots):
                original(slots)
                raise RuntimeError("injected_rank1_old_slot_free_failure")
            allocator.free = fail
        def run(f):
            try:
                return f.run(mock_cuda=False)
            except RestoreRejected:
                return None
        # Patches are process-wide, so keep one enclosing CUDA fence mock while
        # both threads use real, separate CPU request pools and allocators.
        with patch("torch.cuda.synchronize"), \
             patch("experiments.kv_ec.gpu_resume_hook.require_cuda_forward"), \
             ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, fixtures))
        return fixtures, results

    def test_rank1_commit_failure_blocks_all_decode_and_compensates_both(self):
        with tempfile.TemporaryDirectory() as root:
            fixtures, results = self.pair(root, "resume", fail_commit=True)
            self.assertEqual(results, [None, None])
            for f in fixtures:
                self.assertEqual(f.hook.tx.state, "ABORTED")
                self.assertEqual(f.scheduler.forward_ct, 1)
                self.assertEqual(f.hook.evidence["resume_decode_forward_count"], 0)
                self.assertEqual(f.scheduler.token_to_kv_pool_allocator.available_size(), 768)
                self.assertEqual(tuple(f.hook.tx._mapping().tolist()), f.hook.tx.old_slots)
                f.hook.events.close()

    def test_negative_vote_aborts_both_ranks_and_cleanup_reclaims_all_slots(self):
        with tempfile.TemporaryDirectory() as root:
            fixtures, results = self.pair(root, "corrupt")
            self.assertTrue(all(r["negative_test_pass"] for r in results))
            for f in fixtures:
                self.assertEqual(f.hook.tx.state, "ABORTED")
                self.assertEqual(f.scheduler.forward_ct, 1)
                self.assertEqual(f.hook.evidence["resume_decode_forward_count"], 0)
                self.assertTrue(f.hook.evidence["no_slot_leak"])
            # Assemble immutable CPU fixture reports into one check directory.
            # This remains a mocked protocol test, never real GPU evidence.
            output = fixtures[0].root
            r1 = dict(results[1], run_id=results[0]["run_id"])
            (output / "rank_1.json").write_text(json.dumps(r1))
            events = [json.loads(line) for line in
                      (fixtures[1].root / "rank_1.events.jsonl").read_text().splitlines()]
            for event in events:
                event["run_id"] = results[0]["run_id"]
            (output / "rank_1.events.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events) + "\n")
            for logit in results[1]["logits"]:
                (output / logit["file"]).write_bytes((fixtures[1].root / logit["file"]).read_bytes())
            with patch("experiments.kv_ec.verify_gpu_resume.validate_checkpoint", return_value=fixtures[0].meta):
                verdict = check_results(output)
            self.assertTrue(verdict["negative_test_pass"])
            self.assertFalse(verdict["pass_stage_b"])

    def test_single_rank_ready_failure_prevents_both_forward_actions(self):
        with tempfile.TemporaryDirectory() as root:
            fixtures = [ResumeFixture(Path(root) / str(r), "resume", r) for r in (0, 1)]
            barrier, votes = threading.Barrier(2, timeout=10), {}
            def run(rank):
                h = fixtures[rank].hook
                def vote(phase, ready):
                    votes[rank] = ready
                    barrier.wait()
                    answer = all(votes.values())
                    barrier.wait()
                    return answer
                h._vote = vote
                def check():
                    if rank == 1:
                        raise RestoreRejected("injected_generation_changed")
                try:
                    h._barrier("READY_1", check)
                    fixtures[rank].scheduler.forward_ct += 1
                except RestoreRejected:
                    pass
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(run, (0, 1)))
            self.assertEqual([f.scheduler.forward_ct for f in fixtures], [0, 0])
            for f in fixtures:
                f.hook.events.close()


class ResumeCheckerTests(unittest.TestCase):
    def runs(self, root, divergence=None):
        fixtures = []
        for mode in ("reference", "resume"):
            output = Path(root) / mode
            for rank in (0, 1):
                if rank == 0:
                    fixture = ResumeFixture(output, mode, rank)
                else:
                    # Same evidence directory, one private scheduler per rank.
                    temp = ResumeFixture(Path(root) / (mode + "-rank1"), mode, rank)
                    temp.hook.events.close()
                    temp.hook.output = output
                    temp.hook.run_id = output.name
                    temp.hook.evidence["run_id"] = output.name
                    temp.hook.events = (output / "rank_1.events.jsonl").open("x")
                    temp.root = output
                    temp.config["run_id"] = output.name
                    fixture = temp
                if mode == "resume":
                    fixture.config["reference"] = reference_identity(Path(root) / "reference")
                # run.json is immutable; both ranks share the same config.
                fixture.run(divergence=divergence if mode == "resume" else None)
            fixtures.append(fixture)
        return fixtures

    def test_checker_requires_both_ranks_and_matching_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            fixtures = self.runs(root)
            with patch("experiments.kv_ec.verify_gpu_resume.validate_checkpoint", return_value=fixtures[0].meta):
                verdict = check_results(Path(root) / "resume")
                self.assertTrue(verdict["pass_stage_b"])
                self.assertTrue(verdict["output_token_ids_match"])
                self.assertEqual(verdict["resume_decode_forward_count"], 16)
                self.assertTrue(all(r["pass_stage_b"] for r in verdict["ranks"]))
                save_verdict(Path(root) / "resume", verdict)
                save_verdict(Path(root) / "resume", verdict)
                with self.assertRaisesRegex(RestoreRejected, "existing_verdict"):
                    save_verdict(Path(root) / "resume", dict(pass_stage_b=False))

    def test_missing_rank_or_late_failure_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            fixtures = self.runs(root)
            output = Path(root) / "resume"
            path = output / "rank_1.json"
            path.rename(output / "missing-rank.json")
            with patch("experiments.kv_ec.verify_gpu_resume.validate_checkpoint", return_value=fixtures[0].meta):
                with self.assertRaises(FileNotFoundError):
                    check_results(output)
                (output / "missing-rank.json").rename(path)
                (output / "rank_1.failure.json").write_text("{}")
                with self.assertRaisesRegex(RestoreRejected, "late_rank_failure"):
                    check_results(output)

    def test_mismatch_has_first_difference_logits_errors_and_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            fixtures = self.runs(root, divergence=4)
            with patch("experiments.kv_ec.verify_gpu_resume.validate_checkpoint", return_value=fixtures[0].meta):
                verdict = check_results(Path(root) / "resume")
            self.assertFalse(verdict["pass_stage_b"])
            self.assertEqual(verdict["first_difference"]["output_index"], 4)
            self.assertGreater(verdict["first_difference"]["logits_error"]["max_abs_error"], 0)

    def test_tampered_structural_fields_cannot_claim_pass(self):
        with tempfile.TemporaryDirectory() as root:
            fixtures = self.runs(root)
            path = Path(root) / "resume/rank_0.json"
            original = json.loads(path.read_text())
            for key, value in (("no_slot_leak", False), ("generation_after", 8),
                               ("resume_decode_forward_count", 0), ("recomputed_prefix_tokens", 256),
                               ("restored_slots_used_by_decode", False)):
                changed = deepcopy(original)
                changed[key] = value
                path.write_text(json.dumps(changed))
                with self.subTest(key=key), patch(
                        "experiments.kv_ec.verify_gpu_resume.validate_checkpoint", return_value=fixtures[0].meta):
                    with self.assertRaises(RestoreRejected):
                        read_run(Path(root) / "resume")
            path.write_text(json.dumps(original))
            logits = Path(root) / "resume/rank_0.logits_001.npy"
            logits.write_bytes(b"changed")
            with patch("experiments.kv_ec.verify_gpu_resume.validate_checkpoint", return_value=fixtures[0].meta):
                with self.assertRaisesRegex(RestoreRejected, "logits_changed"):
                    read_run(Path(root) / "resume")

    def test_equal_tokens_report_real_logit_errors_without_hiding_them(self):
        ref = [np.array([[0., 2.]])]
        result = compare_outputs([1], [1], ref, [ref[0] + 0.01])
        self.assertTrue(result["output_token_ids_match"])
        self.assertGreater(result["logits_errors"][0]["max_abs_error"], 0)


class IntegrationPlacementTests(unittest.TestCase):
    def test_native_decode_preparation_and_future_map_preserve_restored_prefix(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "python/sglang/srt/managers/schedule_batch.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "ScheduleBatch")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and
                      n.name == "prepare_for_decode")
        overlap_path = root / "python/sglang/srt/managers/overlap_utils.py"
        resolver = next(n for n in ast.parse(overlap_path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "resolve_forward_inputs")
        with tempfile.TemporaryDirectory() as directory:
            f = ResumeFixture(Path(directory) / "native", "resume")
            self.addCleanup(f.hook.events.close)
            tx = f.transaction()
            tx.commit()
            f.req.output_ids = [23]  # sampled, but still no KV for this Token
            s, b = f.scheduler, f.batch
            b.spec_algorithm = NS(is_none=lambda: True)
            b.sampling_info = NS(penalizer_orchestrator=NS(is_required=False))
            b.model_config = NS(is_encoder_decoder=False)
            b.seq_lens = torch.tensor([256])
            b.orig_seq_lens = torch.tensor([256])
            b.hisparse_coordinator = None
            b.input_ids = b.prefill_input_ids_cpu = b.mix_running_indices = None
            b.req_pool_indices = torch.tensor([1])
            b.enable_overlap = False
            def alloc(batch, token_per_req):
                self.assertEqual(token_per_req, 1)
                loc = s.token_to_kv_pool_allocator.alloc(1)
                s.req_to_token_pool.req_to_token[1, 256] = loc[0].int()
                f.req.kv.kv_allocated_len += 1
                return loc
            namespace = dict(torch=torch, alloc_for_decode=alloc,
                             ForwardMode=NS(DECODE=NS(name="DECODE")),
                             mamba_extra_buffer_enabled=lambda: False, _DEBUG_ASSERT=False)
            module = ast.Module(body=ast.parse("from __future__ import annotations").body +
                                [method, resolver], type_ignores=[])
            exec(compile(module, str(path), "exec"), namespace)
            namespace["prepare_for_decode"](b)
            self.assertEqual(f.req.kv_committed_len, 257)
            self.assertEqual(f.req.output_ids, [23])
            self.assertEqual(f.req.decode_batch_idx, 1)
            self.assertEqual(b.seq_lens_cpu.tolist(), [257])
            self.assertEqual(tuple(s.req_to_token_pool.req_to_token[1, :256].tolist()), tx.new_slots)
            self.assertIsNone(b.input_ids)
            s.future_map.output_tokens_buf[1] = 23
            s.future_map.spec_algo = b.spec_algorithm
            namespace["resolve_forward_inputs"](b, s.future_map)
            self.assertEqual(b.input_ids.tolist(), [23])
            f.hook.events.close()

    def test_cpu_tensors_cannot_establish_cuda_execution(self):
        t = torch.zeros(1)
        pool = NS(k_buffer=[t], v_buffer=[t])
        fb = NS(input_ids=t, positions=t, seq_lens=t, out_cache_loc=t, req_pool_indices=t)
        with self.assertRaisesRegex(RestoreRejected, "actual_cuda_tensors"):
            require_cuda_forward(pool, fb, t)

    def test_ready_vote_precedes_model_forward_and_observer_precedes_sampling(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "python/sglang/srt/managers/scheduler.py").read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        loop = ast.unparse(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "event_loop_normal"))
        self.assertLess(loop.index(".before_forward(batch)"), loop.index("self.run_batch(batch)"))
        source = (root / "python/sglang/srt/managers/tp_worker.py").read_text()
        start = source.index("kv_ec_observer =")
        self.assertLess(start, source.index("self.model_runner.sample(", start))


if __name__ == "__main__":
    unittest.main()
