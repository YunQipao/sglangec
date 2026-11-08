"""Synthetic CPU protocol tests, never GPU/native/network performance evidence."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from . import cpu_host_gate as gate, gpu_fault_recovery as gpu, transport_microbench as micro
from .evidence import save, sha
from .. import online_compact as compact
from ..workload import trace
from ... import cold_resume_state as state_module
from ...cold_resume_state import SAMPLING, load_checkpoint
from ...gpu_resume_hook import token_digest
from ...node_block_export import canonical, pack, unpack
from ...test_online_rs import fixture
from ...online_rs_exporter import OnlineRSExporter


def fake_process(root, pid, argv, env=b"", *, state="S"):
    directory = root/str(pid)
    directory.mkdir()
    (directory/"cmdline").write_bytes(b"\0".join(a.encode() for a in argv)+b"\0")
    (directory/"environ").write_bytes(env)
    fields = [state]+["0"]*18+["1234"]
    (directory/"stat").write_text(f"{pid} (test) "+" ".join(fields))
    return directory


class CPUAdmissionTests(unittest.TestCase):
    def test_shared_foreign_process_never_queries_gpu_or_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_process(root, 888880, ["python", "-m", "sglang.launch_server"])
            with patch.object(gate, "project_activity", wraps=lambda: gate_activity(root)), patch.object(gate.subprocess,"run") as run, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""), patch("os.kill") as kill:
                value = gate.cpu_gate("shared")
            self.assertEqual(len(value["shared_host_activity"]), 1)
            self.assertFalse(value["isolated_performance_eligible"])
            run.assert_not_called()
            kill.assert_not_called()

    def test_foreign_denied_environment_is_unknown_not_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            own = fake_process(root, 888881, ["python", "-m", "sglang.launch_server"])
            original = Path.read_bytes
            def read(path, *args, **kwargs):
                if path == own/"environ":
                    raise PermissionError("foreign user")
                return original(path, *args, **kwargs)
            with patch.object(Path, "read_bytes", read):
                activity = gate.project_activity(root)
            self.assertEqual(activity[0]["activity"], "unknown")

    def test_owned_bench_controller_and_sglang_are_rejected(self):
        for argv, env in ((["python","-m","experiments.kv_ec.perf_benchmark"],b""),
                          (["python","-m","experiments.kv_ec.perf_stage_e"],b""),
                          (["python","-m","sglang.launch_server"],b"KV_EC_PERF_RUN_ID=owned\0")):
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fake_process(root, 888882, argv, env)
                with self.assertRaisesRegex(RuntimeError, "PROJECT_ACTIVITY_PRESENT"), patch("os.kill") as kill:
                    gate.project_activity(root)
                kill.assert_not_called()

    def test_unreadable_identified_project_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = fake_process(root, 888883, ["python","-m","experiments.kv_ec.perf_benchmark"])
            (proc/"stat").write_text("unparseable")
            with self.assertRaisesRegex(RuntimeError,"PROJECT_ACTIVITY_UNKNOWN"):
                gate.project_activity(root)

    def test_missing_process_table_fails_closed(self):
        with self.assertRaises(OSError):
            gate.project_activity(Path("/nonexistent-ec-test-proc"))

    def test_cuda_must_be_explicitly_hidden(self):
        for env in ({}, {"CUDA_VISIBLE_DEVICES":"0"}):
            with patch.dict(os.environ,env,clear=True), self.assertRaisesRegex(RuntimeError,"CPU_ONLY"):
                gate.cpu_gate("shared")

    def test_strict_idle_remains_optional_and_rejects_foreign_gpu(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=""), patch.object(gate,"project_activity",return_value=[]), patch.object(gate.subprocess,"run",side_effect=[Mock(stdout="123\n"),Mock(stdout="0\n")]):
            with self.assertRaisesRegex(RuntimeError,"strict_idle"):
                gate.cpu_gate("strict_idle")

    def test_shared_policy_does_not_allow_fixture_or_test_idle_override(self):
        from . import cpu_ablation, failure_sensitivity
        for module in (cpu_ablation, failure_sensitivity):
            with self.assertRaisesRegex(ValueError,"test"):
                module.run([],"unused",cpu_host_mode="shared",idle_check=lambda:{})
        raw = pack(0,[b.tobytes() for b in fixture(0,tp=2)[0]],fixture(0,tp=2)[1],allow_fixture=True)
        with self.assertRaisesRegex(ValueError,"real SGLang"):
            unpack(raw)


gate_activity = gate.project_activity


def sealed_fixture(root):
    """Deliberately synthetic data/proofs; production validator is NOT weakened."""
    root.mkdir()
    blocks, metas = fixture(0,tp=2)
    ids, request, restore_id = [11]*256, "fixture-request", "a"*64
    runtime = {k:metas[0][k] for k in ("model_sha256","source_commit","core_diff_sha256")}
    for rank, meta in enumerate(metas):
        meta.update(snapshot_id=restore_id,checkpoint_version=restore_id,request_version="fresh-fixture",
            manager_epoch="fresh-fixture",request_id=sha(request.encode()),token_ids_sha256=token_digest(ids),
            req_pool_idx=3,req_generation=4)
    raw = pack(0,[b.tobytes() for b in blocks],metas,allow_fixture=True)
    state = dict(schema="kv-cold-request-v1",checkpoint_version=1,request_version=1,tp_size=2,
        boundary="post_prefill_result_before_decode",origin_input_ids=ids,output_ids=[7],
        materialized_kv_tokens=256,kv_allocated_len=256,pending_token_id=7,pending_token_has_kv=False,
        next_decode_input_ids=[7],next_decode_position=256,next_decode_seq_len=257,
        sampling=SAMPLING,rng_policy="greedy_no_rng",finished=False,extend_batch_idx=1,decode_batch_idx=0,
        decode_steps=16,vocab_size=32000,request_id=request,restore_id=restore_id,
        capture_run_id="fresh-fixture",token_start=0,token_count=256,node_id=0,
        input_ids_sha256=token_digest(ids),kv_sha256=sha(raw),runtime_identity=runtime,origin_row=3,origin_generation=4)
    origins=[dict(host="fixture-host",boot_id="fixture-boot",pid=900000+r,start_ticks=1234) for r in range(3)]
    (root/"global_node.bin").write_bytes(raw)
    save(root/"request_state.json",state)
    for rank in range(2):
        (root/f"rank_{rank}.bin").write_bytes(blocks[rank].tobytes())
        save(root/f"rank_{rank}.meta.json",metas[rank])
        save(root/f"rank_{rank}.capture.json",dict(cuda_executed=True,tp_rank=rank,state_sha256=sha(canonical(state)),
            process=origins[rank],materialized_kv_tokens=256,pending_token_id=7))
    save(root/"COMMITTED.json",dict(schema="kv-cold-seal-v1",state="GLOBAL_COMMITTED",restore_id=restore_id,
        request_state_sha256=sha((root/"request_state.json").read_bytes()),rank_state_sha256=[sha(canonical(state))]*2,
        rank_capture_sha256=[sha((root/f"rank_{r}.capture.json").read_bytes()) for r in range(2)],
        rank_payload_sha256=[sha(b) for b in blocks],kv_sha256=sha(raw),
        origin_processes=origins[:2],origin_server_process=origins[2]))
    return raw, state


@contextmanager
def fixture_codec():
    # The entire state/seal/proof/hash validator runs; only synthetic geometry
    # admission is replaced within this test. There is no fixture CLI option.
    with patch.object(state_module,"validate_checkpoint",side_effect=lambda b:unpack(b,allow_fixture=True)[0]), patch.object(gpu,"unpack",side_effect=lambda b:unpack(b,allow_fixture=True)), patch.object(gpu,"pack",side_effect=lambda n,b,m:pack(n,b,m,allow_fixture=True)):
        yield


class GPURecoveryTests(unittest.TestCase):
    def test_mocked_lifecycle_uses_fresh_capture_and_restored_only_launch(self):
        from ...cold_timing import clock_identity
        for baseline in gpu.BASELINES:
            with self.subTest(baseline=baseline),tempfile.TemporaryDirectory() as tmp,fixture_codec():
                root=Path(tmp)/("fresh-fixture-"+baseline)
                ledger=Path(tmp)/"ledger"
                ledger.mkdir()
                events=[]
                captured={}
                cfg=dict(argv=["same-debug-flags"],runtime_identity={})
                class Trial:
                    def __init__(self,*args):
                        self.root=root
                    @contextmanager
                    def owned(self):
                        root.mkdir()
                        with patch.dict(os.environ,KV_EC_PERF_RUN_ID=root.name):
                            yield self
                    def budget(self,value):
                        return value
                    def start(self,name,module,*args):
                        events.append((name,module,args))
                        output=Path(args[args.index("--output")+1])
                        output.mkdir()
                        if name=="origin":
                            _,state=sealed_fixture(root/"capture")
                            captured.update(state=state)
                            cfg["runtime_identity"]=state["runtime_identity"]
                            save(output/"run.json",dict(cfg,capture_bundle=str(root/"capture")))
                            for rank in range(2):
                                save(output/f"rank_{rank}.json",dict(synthetic=True))
                        else:
                            save(output/"run.json",dict(argv=cfg["argv"],model_identity=cfg["runtime_identity"]))
                            if baseline!="reprefill":
                                bundle=Path(args[args.index("--bundle")+1])
                                self_check=load_checkpoint(bundle)
                                assert self_check[1]==captured["state"]
                                assert not (root/"capture/global_node.bin").exists()
                    def stop(self,name):
                        events.append(("stopped",name,()))
                    def run(self,module,*args):
                        if args[0]=="check":
                            save(Path(args[args.index("--output")+1])/"check.json",dict(pass_stage_d=True,
                                restored_slots_used_by_decode=True,ranks=[dict(tp_rank=r,stats=dict(written_bytes=1,h2d_ms=1,verify_ms=1)) for r in range(2)]))
                def reference(*args):
                    return cfg,captured["state"],[dict(output_token_ids=[7]+list(range(16))) for _ in range(2)],[]
                def stream(url,ids,rid,operation,**kw):
                    now=time.monotonic_ns()
                    return dict(clock_identity(),operation_id=operation,span_id="fixture-client",request_id=rid,
                        request_start_ns=now,first_output_token_ns=now,full_output_ns=now,response_complete_ns=now,
                        first_observed_completion_tokens=1,output_token_ids=[7]+list(range(16)),
                        response=dict(output_ids=[7]+list(range(16)),meta_info=dict(cached_tokens=0,prompt_tokens=256,completion_tokens=17)))
                with patch.object(gpu,"LocalTrial",Trial),patch.object(gpu,"wait_health"),patch.object(gpu.cold,"read_evidence",side_effect=reference),patch.object(gpu,"stream_request",side_effect=stream),patch.object(gpu.cold_sse_bridge,"run") as bridge,patch.object(gpu,"validate_config",return_value=[11]*256):
                    result=gpu.one_trial(dict(model="fixture",gpus="0,2",ledger=str(ledger),template="fixture",input_ids="fixture"),root,root.name,baseline,time.monotonic()+10)
                self.assertEqual(result["status"],"PASS",result.get("error"))
                self.assertEqual(result["gpu_restore_verified"],baseline!="reprefill")
                target=next(event for event in events if event[0]=="target")
                self.assertEqual(events[1][0:2],("stopped","origin"))
                if baseline!="reprefill":
                    self.assertEqual(target[1],gpu.MODULE)
                    self.assertEqual(target[2][target[2].index("--bundle")+1],str(root/"restored"))
                    self.assertEqual([c.args[0] for c in bridge.call_args_list],["prepare","finish"])
                else:
                    self.assertEqual(target[1],"experiments.kv_ec.reprefill_server")
                    bridge.assert_not_called()

    def test_real_validator_rejects_synthetic_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"capture"
            sealed_fixture(source)
            with self.assertRaisesRegex(ValueError,"real SGLang"):
                gpu.prepare_owned_capture(source,"replication")

    def test_cpu_replica_and_xor_rebuild_exact_seals_without_original_files(self):
        for baseline in ("replication","local-xor"):
            with self.subTest(baseline=baseline), tempfile.TemporaryDirectory() as tmp,fixture_codec():
                root=Path(tmp)
                raw,state=sealed_fixture(root/"capture")
                seals={p.name:p.read_bytes() for p in (root/"capture").glob("*.json")}
                prepared=gpu.prepare_owned_capture(root/"capture",baseline)
                lost=prepared["active"][0]
                proof,rebuild=gpu.erase_and_rebuild(prepared,root/"capture",root/"restored",baseline)
                self.assertTrue(proof["zeroed"])
                self.assertFalse(np.any(lost))
                self.assertTrue(all(not (root/"capture"/name).exists() for name in ("global_node.bin","rank_0.bin","rank_1.bin")))
                self.assertEqual((root/"restored/global_node.bin").read_bytes(),raw)
                self.assertEqual(load_checkpoint(root/"restored")[1],state)
                self.assertTrue(rebuild["byte_exact"])
                for name,seal in seals.items():
                    self.assertEqual((root/"restored"/name).read_bytes(),seal)
                self.assertEqual(prepared["stable_replica_bytes"]>0,baseline=="replication")
                self.assertEqual(prepared["stable_local_parity_bytes"]>0,baseline=="local-xor")

    def test_reprefill_has_erasure_but_never_claims_gpu_restore(self):
        with tempfile.TemporaryDirectory() as tmp,fixture_codec():
            root=Path(tmp)
            sealed_fixture(root/"capture")
            prepared=gpu.prepare_owned_capture(root/"capture","reprefill")
            proof,rebuild=gpu.erase_and_rebuild(prepared,root/"capture",root/"restored","reprefill")
            self.assertIsNone(rebuild)
            self.assertIsNone(prepared["active"][0])
            self.assertTrue(proof["zeroed"])
            self.assertFalse(proof["gpu_memory_zeroization_claimed"])
            self.assertFalse((root/"restored").exists())

    def test_restore_corruption_fails_original_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp,fixture_codec():
            root=Path(tmp)
            sealed_fixture(root/"capture")
            prepared=gpu.prepare_owned_capture(root/"capture","local-xor")
            gpu.erase_and_rebuild(prepared,root/"capture",root/"restored","local-xor")
            (root/"restored/rank_0.capture.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"cold_capture_proof_changed"):
                load_checkpoint(root/"restored")

    def test_wrong_parity_does_not_fallback_to_lost_original(self):
        with tempfile.TemporaryDirectory() as tmp,fixture_codec():
            root=Path(tmp)
            sealed_fixture(root/"capture")
            prepared=gpu.prepare_owned_capture(root/"capture","local-xor")
            prepared["parity"][0]^=1
            with self.assertRaisesRegex(ValueError,"reconstructed ranks"):
                gpu.erase_and_rebuild(prepared,root/"capture",root/"restored","local-xor")
            self.assertFalse((root/"capture/global_node.bin").exists())

    def test_sealed_cpu_protection_can_be_prepared_before_origin_stop(self):
        with tempfile.TemporaryDirectory() as tmp,fixture_codec():
            root=Path(tmp)/"capture"
            sealed_fixture(root)
            with patch.object(gpu,"require_origins_stopped",side_effect=ValueError("still alive")) as stopped:
                prepared=gpu.prepare_owned_capture(root,"replication")
                stopped.assert_not_called()
                self.assertGreater(prepared["stable_replica_bytes"],0)
                self.assertLessEqual(prepared["protection_prepare_start_ns"],prepared["protection_prepare_complete_ns"])
                self.assertFalse(np.shares_memory(prepared["active"][0],prepared["replicas"][0]))

    def test_shared_hardlink_is_never_erased(self):
        with tempfile.TemporaryDirectory() as tmp,fixture_codec():
            root=Path(tmp)
            raw,_=sealed_fixture(root/"capture")
            prepared=gpu.prepare_owned_capture(root/"capture","replication")
            os.link(root/"capture/rank_0.bin",root/"shared.bin")
            with self.assertRaisesRegex(ValueError,"unsafe original"):
                gpu.erase_and_rebuild(prepared,root/"capture",root/"restored","replication")
            self.assertEqual((root/"capture/global_node.bin").read_bytes(),raw)

    def test_reference_view_changes_only_configuration_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=root/"reference"
            reference.mkdir()
            config=dict(capture_bundle="/old",cold_mode="capture",run_id="independent-capture")
            save(reference/"run.json",config)
            (reference/"rank_0.json").write_bytes(b"immutable")
            record=gpu.reference_view(reference,root/"restored",root/"view")
            relocated=gpu.read(root/"view/run.json")
            self.assertEqual({k:v for k,v in relocated.items() if k!="capture_bundle"},{k:v for k,v in config.items() if k!="capture_bundle"})
            self.assertEqual((root/"view/rank_0.json").read_bytes(),b"immutable")
            self.assertEqual(record["only_changed_field"],"capture_bundle")

    def test_five_rounds_reject_without_smoke_before_any_launch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(gpu,"validate_config"),patch.object(gpu,"one_trial") as launch:
            input_path=Path(tmp)/"input.json"
            input_path.write_text("[]")
            with self.assertRaisesRegex(ValueError,"smoke"):
                gpu.execute(dict(input_ids=str(input_path)),Path(tmp)/"new-run-001","new-run-001",rounds=5)
            launch.assert_not_called()

    def test_supervisor_stop_uses_authenticated_marker_never_direct_kill(self):
        config=dict(repo="/repo",python="/env/python",model="/model",gpus="0,2",cuda_home="/cuda",conda_prefix="/env")
        with tempfile.TemporaryDirectory() as tmp,patch("os.kill") as kill,patch.object(gpu.Host,"run") as run:
            trial=gpu.LocalTrial(config,Path(tmp),"new-run",time.monotonic()+10)
            child=Mock()
            child.wait.return_value=0
            trial.services["origin"]=child
            (Path(tmp)/"origin").mkdir()
            save(Path(tmp)/"origin/exit.json",dict(reason="owned_stop_requested"))
            trial.stop("origin")
            self.assertEqual(run.call_args.args[:2],(gpu.SELF,"stop"))
            child.wait.assert_called_once_with(timeout=90)
            child.kill.assert_not_called()
            kill.assert_not_called()


class OnlineBoundaryTests(unittest.TestCase):
    def test_actual_server_flags_and_listener_lifetime_are_checked(self):
        from ...launch_online import server_argv
        actual=dict(identity=dict(pid=123,start_ticks=17,uid=1000,boot_id="fixture",pgid=123,session=123),
            argv=server_argv(SimpleNamespace(model="/model",tp=2,port=32961,mem_fraction_static=.6,prefill_chunk_size=256)),
            environment=dict(KV_EC_GPU_ADMISSION_MODE="shared",CUDA_VISIBLE_DEVICES="0,2",SGLANG_KV_EC_ONLINE="0",
                SGLANG_KV_EC_PREPARED_IDENTITY=json.dumps(dict(model_sha256="a"*64))))
        for key in ("DEBUG","PAUSE_RESUME","GPU_RESTORE","GPU_RESUME","GPU_COLD_RESUME","RS_RESUME"):
            actual["environment"]["SGLANG_KV_EC_"+key]="0"
        spec=dict(mode="noec",pid=123,url="http://127.0.0.1:32961")
        owners=[dict(port=32961,owners=[dict(actual["identity"])])]
        with patch.object(compact,"process",side_effect=lambda _:deepcopy(actual)),patch("experiments.kv_ec.perf_ports.listeners",return_value=owners):
            result=compact.verify_runtime(spec,dict(model_sha256="a"*64))
            self.assertEqual(result["concurrency_semantics"],"queue_concurrency_not_true_batch")
            owners[0]["owners"][0]["start_ticks"]+=1
            with self.assertRaisesRegex(ValueError,"listener not owned"):
                compact.verify_runtime(spec,dict(model_sha256="a"*64))
            owners[0]["owners"][0]["start_ticks"]-=1
            actual["argv"][actual["argv"].index("--max-running-requests")+1]="2"
            with self.assertRaisesRegex(ValueError,"runtime activation"):
                compact.verify_runtime(spec,dict(model_sha256="a"*64))

    def test_launcher_discards_inherited_prepared_identity(self):
        import sys
        from ... import launch_online
        args=["launcher","--model","/reviewed-model","--gpus","1,3","--tp","2","--socket","/tmp/unused.sock","--run-id","fresh-launch","--baseline"]
        def identity(model):
            self.assertNotIn("SGLANG_KV_EC_PREPARED_IDENTITY",os.environ)
            return dict(model_sha256="fresh")
        with patch.object(sys,"argv",args),patch.dict(os.environ,SGLANG_KV_EC_PREPARED_IDENTITY="stale-or-invalid-json"),patch.object(launch_online,"check_gpu_resources"),patch.object(launch_online,"check_port"),patch.object(launch_online,"source_identity",side_effect=identity),patch.object(launch_online.os,"execve") as execute:
            launch_online.main()
        self.assertEqual(json.loads(execute.call_args.args[2]["SGLANG_KV_EC_PREPARED_IDENTITY"]),dict(model_sha256="fresh"))

    def test_publisher_status_and_manifest_are_atomic_and_receipt_is_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root/"manifest.json",dict(memory_ledger={},timing_ms={},protection_mode="rs-only",state="LOCAL_SNAPSHOT"))
            save(root/"global_status.json",dict(global_state="WAITING_COMMIT"))
            (root/"global_node.bin").write_bytes(b"fixture")
            job=dict(directory=root,created=time.monotonic(),status=dict(local_snapshot_id="fixture"))
            original=Path.write_text
            observations=[]
            def write(path,*args,**kw):
                old=json.loads((root/"global_status.json").read_text())
                manifest=json.loads((root/"manifest.json").read_text())
                observations.append((path.name,old["global_state"],manifest["state"]))
                return original(path,*args,**kw)
            with patch.object(Path,"write_text",write):
                OnlineRSExporter._save(job,"GLOBAL_COMMITTED",remote_outcome="COMMITTED",remote_parity_bytes=42)
            self.assertEqual(observations,[("manifest.global.tmp","WAITING_COMMIT","LOCAL_SNAPSHOT"),("global_status.tmp","WAITING_COMMIT","COMMITTED")])
            self.assertEqual(json.loads((root/"global_status.json").read_text())["global_state"],"GLOBAL_COMMITTED")

    def test_formal_online_requires_own_mode_smoke(self):
        value=dict(rounds=5,trace_sha256="trace")
        with self.assertRaisesRegex(ValueError,"smoke"):
            compact.require_smoke(value,"noec",None)

    def test_four_requests_are_fixed_total_for_c1_and_c4(self):
        value=compact.plan(trace(seed=3,requests=4,input_tokens=4096,output_tokens=17,vocab_size=100,model_sha256="a"*64),prefixes=(256,1024,4096),concurrencies=(1,4),rounds=5)
        self.assertEqual(len(value["rows"]),120)
        for concurrency in (1,4):
            self.assertEqual(sum(r["concurrency"]==concurrency for r in value["rows"]),60)

    def test_wrong_unix_peer_rejected_before_any_protocol_send(self):
        conn=Mock()
        conn.getsockopt.return_value=struct.pack("3i",999,1000,1)
        socket=Mock()
        socket.__enter__=Mock(return_value=conn)
        socket.__exit__=Mock(return_value=False)
        with patch.object(compact,"process",return_value=dict(identity=dict(uid=1000))),patch.object(compact.socket,"socket",return_value=socket):
            with self.assertRaisesRegex(ValueError,"Unix socket peer"):
                with compact.manager_connection(dict(manager_pid=123,manager_socket="unused"),5):
                    self.fail("foreign peer accepted")

    def test_normal_window_finishes_before_protection_drain(self):
        value=compact.plan(trace(seed=3,requests=4,input_tokens=256,output_tokens=17,vocab_size=100,model_sha256="a"*64),prefixes=(256,),concurrencies=(1,),rounds=1)
        calls=[]
        def measure(*args,**kw):
            calls.append("normal")
            return dict(output_ids=[1]*17,actual_output_tokens=17)
        def protection(*args,**kw):
            calls.append("drain")
            return []
        runtime=dict(identity=dict(pid=1),concurrency_semantics="queue_concurrency_not_true_batch")
        with tempfile.TemporaryDirectory() as tmp,patch.object(compact,"verify_runtime",return_value=runtime),patch.object(compact,"measure",side_effect=measure),patch.object(compact,"protection",side_effect=protection),patch("experiments.kv_ec.perf_benchmark.resource_snapshot",return_value={}),patch("experiments.kv_ec.perf_benchmark.command",return_value={}):
            result=compact.run(dict(mode="noec",pid=os.getpid(),url="unused"),value,Path(tmp)/"run")
        self.assertEqual(result["status"],"PASS",result.get("error"))
        self.assertEqual(calls,["normal"]*4+["drain"]*4)
        group=result["groups"][0]
        self.assertAlmostEqual(group["wall_s"],group["normal_response_window_s"]+group["protection_drain_window_s"])


class TransportAdapterTests(unittest.TestCase):
    def plan(self):
        rows=[dict(transport=t,port=micro.PORTS[t],mib=m,repeat=0,payload_bytes=m*1024**2,sha256="a"*64) for t in micro.PORTS for m in micro.SIZES]
        value=dict(schema="ec-transport-plan-v1",run_id="fresh-transport",rounds=1,rows=rows,source={})
        value["plan_sha256"]=micro.digest(value)
        return value

    def test_complete_matrix_and_fixed_ports(self):
        value=self.plan()
        micro.validate(value)
        value["rows"][0]["port"]=32965
        value["plan_sha256"]=micro.digest({k:v for k,v in value.items() if k!="plan_sha256"})
        with self.assertRaisesRegex(ValueError,"fixed port"):
            micro.validate(value)

    def test_source_too_small_never_repeats_payload(self):
        with patch.object(micro,"load_rank",return_value=(b"small",{})),self.assertRaisesRegex(ValueError,"shorter than 64MiB"):
            micro.prepare("unused","unused","fresh-transport")

    def test_native_cq_evidence_required_not_default_success(self):
        row=dict(payload_bytes=1024)
        with self.assertRaisesRegex(ValueError,"native CQ"):
            micro.native_check(dict(bytes=1024,completion_status="success"),row,True)
        stats=dict(bytes=1024,completion_status="SUCCESS",helper_stdout="KV_EC_RDMA_STATS bytes=1024",write_segments=1,rdma_write_cq_ms=2)
        micro.native_check(stats,row,True)
        with self.assertRaisesRegex(ValueError,"native CQ"):
            micro.native_check(stats,row,False)

    def test_receiver_requires_parent_supervisor_identity(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(micro,"require_trial",side_effect=ValueError("foreign trial")),patch.object(micro,"tcp_receive") as receive:
            with self.assertRaisesRegex(ValueError,"foreign trial"):
                micro.transfer(self.plan(),Path(tmp)/"received",sending=False,host="unused")
            receive.assert_not_called()

    def test_prebind_foreign_listener_fails_without_signal_or_receive(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(micro,"require_receiver_supervisor"),patch.object(micro,"PortLease") as lease,patch.object(micro,"assert_free",side_effect=RuntimeError("foreign owner")),patch.object(micro,"listeners",return_value=[dict(port=32963,owners=[])]),patch.object(micro,"tcp_receive") as receive,patch("os.kill") as kill:
            result=micro.transfer(self.plan(),Path(tmp)/"received",sending=False,host="unused")
            self.assertEqual(result["status"],"PARTIAL")
            self.assertIn("foreign owner",result["records"][0]["error"])
            receive.assert_not_called()
            kill.assert_not_called()

    def test_receiver_rejects_foreign_pid_even_with_matching_run_id(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,KV_EC_PERF_RUN_ID="fresh-transport"):
            root=Path(tmp)
            save(root/"trial-owner.json",dict(run_id="fresh-transport"))
            (root/"receiver").mkdir()
            identity=micro.process_identity(os.getpid())
            identity.update(pid=identity["pid"]+1,run_id="fresh-transport")
            save(root/"receiver/owned-process.json",identity)
            with self.assertRaisesRegex(RuntimeError,"exact parent-supervised"):
                micro.require_receiver_supervisor(root,"fresh-transport")

    def test_collect_requires_receiver_sha_not_sender_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=self.plan()
            for side in ("sender","receiver"):
                records=[dict(row,index=i,success=True,**({"sender_sha256":row["sha256"]} if side=="sender" else {"receiver_sha256":"wrong","receiver_sha_verified":False})) for i,row in enumerate(plan["rows"])]
                save(root/f"{side}.json",dict(schema="ec-transport-side-v1",side=side,run_id=plan["run_id"],plan_sha256=plan["plan_sha256"],records=records))
            value=micro.collect(plan,root/"sender.json",root/"receiver.json",root/"collected.json")
            self.assertEqual(value["status"],"PARTIAL")
            self.assertTrue(all(not r["success"] for r in value["records"]))

    def test_tcp_is_one_payload_plus_ack_and_has_separate_wall_stats(self):
        from ...multinode.tcp_transport import TCPTransport
        conn=Mock()
        conn.recv.return_value=b"\1"
        conn.__enter__=Mock(return_value=conn)
        conn.__exit__=Mock(return_value=False)
        with patch("socket.create_connection",return_value=conn):
            transport=TCPTransport(retry_timeout=1)
            transport.send("unused",32963,b"payload")
        self.assertEqual([c.args[0] for c in conn.sendall.call_args_list],[struct.pack("!Q",7),b"payload"])
        self.assertEqual(transport.last_stats["bytes"],7)
        self.assertIn("acknowledged_unidirectional_ms",transport.last_stats)

    def test_tcp_receiver_flushes_payload_before_ack(self):
        from ...multinode.tcp_transport import receive_file
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"received.bin"
            conn=Mock()
            conn.__enter__=Mock(return_value=conn)
            conn.__exit__=Mock(return_value=False)
            conn.recv.side_effect=[struct.pack("!Q",4),b"data"]
            def ack(raw):
                self.assertEqual(raw,b"\1")
                self.assertEqual(path.read_bytes(),b"data")
            conn.sendall.side_effect=ack
            server=Mock()
            server.__enter__=Mock(return_value=server)
            server.__exit__=Mock(return_value=False)
            server.accept.return_value=(conn,("fixture",32963))
            status={}
            with patch("socket.socket",return_value=server):
                receive_file("unused",32963,path,timeout=1,status=status)
            self.assertEqual(status["bytes"],4)
            self.assertIn("receive_payload_and_flush_ms",status)


if __name__ == "__main__":
    unittest.main()
