import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from experiments.kv_ec.design_study.extension_v2.ready_timing import observe
from experiments.kv_ec import prefill_observer as p

class ReadyTests(unittest.TestCase):
    def tree(self,tmp):
        root=Path(tmp)/'trial-id';root.mkdir();(root/'perf').mkdir();target=root/'trial-id-restore';target.mkdir()
        client=dict(host='host',boot_id='boot',response_complete_ns=10000000)
        rows=[]
        for rank in (0,1):
            common=dict(schema='kv-ec-perf-v1',run_id=root.name,role='node/target',host='host',boot_id='boot',clock_domain='host_monotonic_ns',status='ok',tp_rank=rank,request_id='request',token_count=256)
            rows += [dict(common,stage='prefill_kv_ready',end_ns=3000000+rank*1000,cuda_completed=True,scheduler_ready=True,sampled_token_id=7),dict(common,stage='kv_first_decode',end_ns=5000000+rank*1000)]
            (target/f'rank_{rank}.events.jsonl').write_text(json.dumps(dict(phase='GLOBAL_COMMIT',run_id=target.name,tp_rank=rank,transaction_state='COMMITTED',written_bytes=100,monotonic=(3000000+rank*1000)/1e9))+'\n')
        (root/'perf/host.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return root,target,client

    def test_prefill_and_gpu_ready_use_same_two_rank_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,t,c=self.tree(tmp)
            a=observe(r,t,'reprefill',c,'request',7,1000000,2000000)
            verdict=dict(pass_stage_d=True,restored_slots_used_by_decode=True,ranks=[dict(tp_rank=n,process=dict(host='host',boot_id='boot')) for n in (0,1)])
            b=observe(r,t,'local-xor',c,'request',7,1000000,2000000,verdict)
            self.assertTrue(a['common_ready_boundary_observed']);self.assertTrue(b['common_ready_boundary_observed'])
            self.assertAlmostEqual(a['fault_to_kv_ready_ms'],b['fault_to_kv_ready_ms'],places=5)
            self.assertNotEqual(a['service_http_ready_ms'],a['fault_to_kv_ready_ms'])

    def test_missing_rank_or_foreign_clock_never_substitutes_http_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,t,c=self.tree(tmp);c['boot_id']='foreign'
            x=observe(r,t,'reprefill',c,'request',7,1000000,2000000)
            self.assertFalse(x['common_ready_boundary_observed']);self.assertIsNone(x['fault_to_kv_ready_ms'])
            self.assertIn('clock',x['kv_ready_unavailable_reason'])
            (r/'perf/host.jsonl').write_text('')
            x=observe(r,t,'reprefill',c,'request',7,1000000,2000000)
            self.assertIsNone(x['fault_to_kv_ready_ms'])

    def test_prefill_observer_default_off_has_no_cuda_or_state_access(self):
        with patch.dict('os.environ',{'SGLANG_KV_EC_PREFILL_OBSERVE':'0'}),patch.object(p,'emit') as emitted:
            p.prefill_kv_ready(object(),object(),object());p.before_decode(object(),object());emitted.assert_not_called()

    def test_optin_observer_fences_cuda_and_records_true_pending_token(self):
        args=NS(pp_size=1,chunked_prefill_size=-1,max_running_requests=1,disable_radix_cache=True,dtype='bfloat16',kv_cache_dtype='auto',cuda_graph_config=NS(decode=NS(backend='disabled'),prefill=NS(backend='disabled')))
        req=NS(origin_input_ids=[1]*256,req_pool_idx=1,finished=lambda:False,is_retracted=False,inflight_middle_chunks=0,kv_committed_len=256,output_ids=[7],extend_input_len=256,rid='request')
        sched=NS(server_args=args,is_generation=True,enable_overlap=False,model_worker=NS(ps=NS(tp_size=2,tp_rank=0)),req_to_token_pool=NS(device='cuda:0'))
        batch=NS(reqs=[req],extend_lens=[256],extend_num_tokens=256,spec_algorithm=NS(is_none=lambda:True),forward_mode=NS(is_extend_without_speculative=lambda:True,is_decode=lambda:True))
        with patch.dict('os.environ',{'SGLANG_KV_EC_PREFILL_OBSERVE':'1','KV_EC_PERF_DIR':'fake','KV_EC_PERF_RUN_ID':'fake'}),patch('torch.cuda.synchronize') as sync,patch.object(p,'emit') as emitted:
            p.prefill_kv_ready(sched,batch,NS(can_run_cuda_graph=False));sync.assert_called_once_with('cuda:0')
            self.assertEqual(emitted.call_args.args[0],'prefill_kv_ready');self.assertTrue(emitted.call_args.kwargs['cuda_completed'])
            p.before_decode(sched,batch);self.assertEqual(emitted.call_args.args[0],'kv_first_decode')

if __name__=='__main__':unittest.main()
