import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.kv_ec.design_study import online_compact as compact
from experiments.kv_ec.design_study.workload import trace
from experiments.kv_ec.design_study.extension_v2 import collector, cpu_ablation, failure_sensitivity
from experiments.kv_ec.design_study.extension_v2 import supervised_online
from experiments.kv_ec.design_study.extension_v2.evidence import save, sha
from experiments.kv_ec.node_block_export import pack, unpack
from experiments.kv_ec.online_chunk_manager import OnlineChunkManager
from experiments.kv_ec.online_rs_exporter import OnlineRSExporter
from experiments.kv_ec.remote_rs_server import StripeStore
from experiments.kv_ec.test_online_rs import fixture, MockRDMAClient


def nodes():
    return [pack(n,[b.tobytes() for b in fixture(n,tp=2)[0]],fixture(n,tp=2)[1],allow_fixture=True) for n in (0,1)]


class PipelineTests(unittest.TestCase):
    def test_real_rs_only_exporter_commits_without_local_xor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            store=StripeStore(allow_fixture=True,rdma_directory=root/"rdma")
            exporters=[]
            try:
                managers=[]
                keys=[]
                for n in (0,1):
                    blocks,entries=fixture(n,tp=2)
                    exporter=OnlineRSExporter(node_id=n,run_id="global",checkpoint_version="v1",
                        requests={entries[0]["request_id"]:"pair"},remote_host="mock",timeout=3,poll_interval=.005,
                        workers=1,max_pending=1,client_factory=lambda:MockRDMAClient(store),allow_fixture=True)
                    exporters.append(exporter)
                    m=OnlineChunkManager(root/f"node{n}",f"local{n}",protection_mode="rs-only",global_exporter=exporter)
                    managers.append(m)
                    with patch("experiments.kv_ec.online_chunk_manager.encode",side_effect=AssertionError("local XOR forbidden")):
                        for b,meta in zip(blocks,entries):
                            status=m.submit(meta,b)
                    self.assertEqual(status["state"],"LOCAL_SNAPSHOT")
                    keys.append(entries[0]["snapshot_id"])
                deadline=time.monotonic()+4
                for m,key in zip(managers,keys):
                    path=m.output/key/"manifest.json"
                    while time.monotonic()<deadline:
                        value=json.loads(path.read_text())
                        if value["state"]=="COMMITTED":
                            break
                        time.sleep(.005)
                    self.assertEqual(value["state"],"COMMITTED")
                    self.assertEqual(value["protection_state"],"GLOBAL_COMMITTED")
                    self.assertFalse((m.output/key/"parity.bin").exists())
                    self.assertGreater(value["memory_ledger"]["stable_remote_parity_bytes"],0)
                    self.assertGreater(value["timing_ms"]["global_commit"],0)
                    self.assertEqual(m.status(key)["state"],"COMMITTED")
            finally:
                for e in exporters:
                    e.close()

    def test_five_cpu_modes_rank_and_node_faults_have_honest_outcomes(self):
        raw=nodes()
        for mode in cpu_ablation.MODES:
            for rank in (0,None):
                adapter=cpu_ablation.ShardAdapter(raw,mode,allow_fixture=True)
                proof=adapter.erase(0,rank)
                self.assertTrue(all(c["zeroed"] for c in proof["actual_erasure"]))
                self.assertIsNone(adapter.active[0][0])
                expected=mode in ("rs-only","xor-rs") or rank is not None and mode in ("replication","xor")
                if expected:
                    self.assertTrue(adapter.recover(0,rank)["byte_exact"])
                else:
                    with self.assertRaises(cpu_ablation.Unrecoverable):
                        adapter.recover(0,rank)
                if mode=="replication":
                    self.assertEqual(adapter.ledger["stable_replica_bytes"],adapter.source_bytes)
                    self.assertEqual(adapter.ledger["stable_local_parity_bytes"],0)

    def test_cpu_ablation_complete_and_expected_failure_not_recovery_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths=[]
            for n,raw in enumerate(nodes()):
                path=root/f"node{n}.bin"
                path.write_bytes(raw)
                paths.append(path)
            value=cpu_ablation.run(paths,root/"run",rounds=1,allow_fixture=True,idle_check=lambda:{"fixture":True})
            self.assertEqual(value["status"],"PASS",value.get("error"))
            self.assertEqual(len(value["records"]),30)
            self.assertEqual(value["recovery_successes"],20)
            self.assertEqual(value["expected_unrecoverable"],10)
            self.assertGreater(value["rss"]["self_lifetime_ru_maxrss_bytes"],0)
            for r in value["records"]:
                if r.get("expected_unrecoverable"):
                    self.assertFalse(r["recovery_success"])
                    self.assertIsNone(r.get("fault_to_host_kv_ready_ms"))

    def test_cpu_request_rates_execute_actual_faults_and_keep_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths=[]
            for n,raw in enumerate(nodes()):
                path=root/f"node{n}.bin"
                path.write_bytes(raw)
                paths.append(path)
            value=failure_sensitivity.run(paths,root/"rates",requests=20,rounds=1,seed=1,
                allow_fixture=True,idle_check=lambda:{"fixture":True})
            self.assertEqual(value["status"],"CPU_ONLY",value.get("error"))
            self.assertEqual(len(value["records"]),300)
            selected=[r for r in value["records"] if r["selected"]]
            self.assertGreater(len(selected),0)
            self.assertTrue(all(r.get("erasure") for r in selected))
            self.assertTrue(any(not r["success"] for r in selected))
            self.assertTrue(all(not r["success"] for r in selected if r["mode"]=="noec"))
            self.assertTrue(all(r["success"] for r in selected if r["mode"]=="rs-only"))
            report=next(r for r in value["summary"] if r["mode"]=="noec" and r["actual_faults"])
            self.assertIsNone(report["mttr_all_faults_ms"])
            self.assertEqual(report["recovery_success_rate"],0)

    def test_compact_plan_same_trace_and_failed_records_survive(self):
        value=compact.plan(trace(seed=3,requests=4,input_tokens=1024,output_tokens=17,vocab_size=100,model_sha256="a"*64),
            prefixes=(256,1024),concurrencies=(1,4),rounds=1)
        self.assertEqual(len(value["rows"]),16)
        self.assertEqual(len({r["request_id"] for r in value["rows"]}),16)
        self.assertEqual(value["rows"][0]["input_ids"],value["rows"][8]["input_ids"][:256])
        runtime=dict(identity={"pid":os.getpid()},concurrency_semantics="queue_concurrency_not_true_batch")
        def measure(url,row,**kw):
            if row["request_id"].endswith("p1024-c4-r0"):
                raise TimeoutError("preserved failure")
            return dict(output_ids=[1]*17,actual_output_tokens=17,end_to_end_ms=10)
        with tempfile.TemporaryDirectory() as tmp,patch.object(compact,"verify_runtime",return_value=runtime),patch.object(compact,"measure",side_effect=measure),patch("experiments.kv_ec.perf_benchmark.resource_snapshot",return_value={}),patch("experiments.kv_ec.perf_benchmark.command",return_value={}):
            report=compact.run(dict(mode="noec",pid=os.getpid(),url="unused"),value,Path(tmp)/"run")
            self.assertEqual(report["status"],"PARTIAL")
            self.assertEqual(report["requests"],16)
            self.assertEqual(report["completed"],12)
            self.assertEqual(sum(r["timeout"] for r in report["records"]),4)
            self.assertTrue(all(g["concurrency_semantics"]=="queue_concurrency_not_true_batch" for g in report["groups"]))

    def test_compact_wrong_mode_and_uncommitted_chunk_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root/"service_manifest.json",dict(run_id="run"))
            with self.assertRaises(TimeoutError):
                compact.protection(dict(mode="rs-only",manager_output=str(root)),
                    dict(request_id="r",input_ids=[1]*256),timeout=.01)
        actual=dict(argv=["python","-m","sglang.launch_server","--tp-size","2","--dtype","bfloat16",
            "--attention-backend","triton","--chunked-prefill-size","256","--disable-radix-cache","--disable-overlap-schedule"],
            environment={"KV_EC_GPU_ADMISSION_MODE":"shared","SGLANG_KV_EC_ONLINE":"1","SGLANG_KV_EC_ONLINE_PROTECTION_MODE":"xor"})
        with patch("experiments.kv_ec.design_study.online_compact.process",return_value=actual):
            with self.assertRaises(ValueError):
                compact.verify_runtime(dict(mode="replication",pid=123),dict(model_sha256="a"*64))

    def test_collector_hash_gate_partial_null_and_no_smoke_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths=[]
            for repeat in range(5):
                path=root/f"smoke{repeat}.json"
                save(path,dict(schema="ec-cpu-ablation-v2",records=[dict(repeat=repeat,mode="replication",success=True,expected_unrecoverable=True,recovery_success=False)]))
                paths.append(dict(path=str(path),sha256=sha(path.read_bytes()),experiment="E5",evidence_class="historical_smoke"))
            paths.append(dict(paths[0],sha256="f"*64))
            save(root/"index.json",dict(inputs=paths))
            value=collector.collect(root/"index.json",root/"report")
            self.assertEqual(value["status"],"PARTIAL")
            self.assertEqual(len(value["rejected"]),1)
            self.assertEqual(value["coverage"]["E1"]["status"],"NOT_RUN")
            group=value["groups"][0]
            self.assertFalse(group["five_round_eligible"])
            self.assertEqual(group["recovery_successes"],0)
            self.assertIsNone(group["metrics"]["node_rto_ms"]["median"])
            self.assertTrue((root/"report/records.csv").is_file())
            self.assertTrue((root/"report/coverage.svg").is_file())

    def test_parent_wrapper_uses_owned_supervisor_and_stops_only_new_scope_on_failure(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=False):
            root=Path(tmp)
            value=compact.plan(trace(seed=3,requests=1,input_tokens=256,output_tokens=17,vocab_size=100,model_sha256="a"*64),
                prefixes=(256,),concurrencies=(1,),rounds=1)
            save(root/"plan.json",value)
            args=SimpleNamespace(plan=root/"plan.json",mode="noec",port=32961,rs_config=None,
                output=root/"fresh",reference=None,model="reviewed-model",gpus="0,1",capacity=4,
                mem_fraction_static=.6,startup_timeout=1,total_timeout=1)
            child=Mock(returncode=0)
            child.wait.return_value=0
            def start(argv,**kwargs):
                service=root/"fresh/origin"
                service.mkdir()
                save(service/"owned-process.json",dict(pid=99))
                self.assertIn("experiments.kv_ec.perf_stage_e",argv)
                return child
            with patch.object(supervised_online,"PortLease"),patch.object(supervised_online.subprocess,"Popen",side_effect=start),patch.object(supervised_online,"wait_health",side_effect=TimeoutError("readiness failed")):
                with self.assertRaises(TimeoutError):
                    supervised_online.execute(args)
            self.assertTrue((root/"fresh/origin/stop").exists())
            self.assertTrue((root/"fresh/failure.json").exists())
            child.wait.assert_called_once_with(timeout=60)
            child.kill.assert_not_called()
            child.terminate.assert_not_called()

    def test_cold_adapter_uses_sealed_source_then_erases_real_owned_bytes(self):
        raw=nodes()[0]
        source=Mock()
        source.load.return_value=raw
        row=dict(request_id="logical-cold",input_ids=[1]*256,output_tokens=17)
        state=dict(origin_input_ids=row["input_ids"])
        def measured(*args,**kwargs):
            return dict(first_output_ns=time.monotonic_ns(),output_ids=[9]*17)
        with patch("experiments.kv_ec.cold_resume_state.load_checkpoint",return_value=(source,state,{},None)),patch("experiments.kv_ec.node_block_export.unpack",side_effect=lambda b:unpack(b,allow_fixture=True)),patch("experiments.kv_ec.design_study.workload.measure",side_effect=measured):
            result=failure_sensitivity.cold_reprefill("unused","sealed-case",row,selected=True,expected_output=[9]*17)
        self.assertEqual(len(result["actual_erasure"]),2)
        self.assertTrue(all(r["zeroed"] and r["bytes"]>0 for r in result["actual_erasure"]))
        self.assertFalse(result["gpu_erased"])
        self.assertIsNone(result["fault_to_host_kv_ready_ms"])
        self.assertGreaterEqual(result["fault_to_first_sse_ms"],0)


if __name__=="__main__":
    unittest.main()
