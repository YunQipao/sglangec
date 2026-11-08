"""Observed KV readiness from both TP ranks; HTTP readiness is separate."""
import json
from pathlib import Path


def observe(root, target, baseline, client, request_id, first_token, fault_ns, http_ns, verdict=None):
    result=dict(service_http_ready_ms=(http_ns-fault_ns)/1e6,
        legacy_ready_field_semantics='HTTP listener ready, not KV ready',
        fault_to_kv_ready_ms=None, kv_ready_ns=None, common_ready_boundary_observed=False,
        ready_semantics='all TP ranks: KV installed/recomputed and scheduler ready before first Decode',
        ready_clock_domain='host_monotonic_ns')
    second=client.get('first_new_decode_output_ns')
    result.update(fault_to_first_new_decode_sse_ms=(second-fault_ns)/1e6 if type(second) is int else None,
        first_new_decode_sse_semantics='first cumulative SSE event containing output token index 1 (second token); batching may coalesce tokens')
    try:
        host,boot=client['host'],client['boot_id']
        rows=[]
        for path in (Path(root)/'perf').glob('*.jsonl'):
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row.get('run_id')==Path(root).name and row.get('role','').endswith('/target'):
                    rows.append(row)
        def markers(stage):
            selected=[r for r in rows if r.get('stage')==stage and r.get('request_id')==request_id]
            if len(selected)!=2 or {r.get('tp_rank') for r in selected}!={0,1}:
                raise ValueError('exactly one bound '+stage+' marker per TP rank required')
            for r in selected:
                if (r.get('host')!=host or r.get('boot_id')!=boot or r.get('clock_domain')!='host_monotonic_ns'
                        or r.get('status')!='ok' or r.get('token_count')!=256 or type(r.get('end_ns')) is not int):
                    raise ValueError('marker clock/request/profile mismatch')
            return {r['tp_rank']:r for r in selected}
        first=markers('kv_first_decode')
        if baseline=='reprefill':
            ready=markers('prefill_kv_ready')
            for row in ready.values():
                if (row.get('cuda_completed') is not True or row.get('scheduler_ready') is not True
                        or row.get('sampled_token_id')!=first_token):
                    raise ValueError('prefill completion or sampled-token confirmation missing')
            times={r:ready[r]['end_ns'] for r in (0,1)}
            source='stock prefill result after explicit CUDA completion and scheduler bookkeeping'
        else:
            if not verdict or not verdict.get('pass_stage_d') or not verdict.get('restored_slots_used_by_decode'):
                raise ValueError('strict GPU restore/decode verdict required')
            times={}
            for rank in (0,1):
                proof=next(r for r in verdict['ranks'] if r['tp_rank']==rank)['process']
                if proof['host']!=host or proof['boot_id']!=boot:
                    raise ValueError('restore rank/client clock mismatch')
                events=[json.loads(l) for l in (Path(target)/f'rank_{rank}.events.jsonl').read_text().splitlines()]
                commits=[r for r in events if r.get('phase')=='GLOBAL_COMMIT' and r.get('run_id')==Path(target).name and r.get('tp_rank')==rank]
                if len(commits)!=1 or commits[0].get('transaction_state')!='COMMITTED' or commits[0].get('written_bytes',0)<=0:
                    raise ValueError('unique committed rank/scheduler marker required')
                times[rank]=int(commits[0]['monotonic']*1e9)
            source='existing cold GLOBAL_COMMIT after TP COLD_REQUEST_READY vote and scheduler install'
        for rank,t in times.items():
            if not fault_ns<=t<=first[rank]['end_ns']<=client['response_complete_ns']:
                raise ValueError('KV ready must precede its own rank first Decode in the same clock')
        observed=max(times.values())
        result.update(kv_ready_ns=observed,fault_to_kv_ready_ms=(observed-fault_ns)/1e6,
            common_ready_boundary_observed=True,kv_ready_observation_source=source,
            per_rank_kv_ready_ns=times,per_rank_first_decode_ns={r:first[r]['end_ns'] for r in (0,1)},
            first_sse_may_be_pending=baseline!='reprefill')
    except (OSError,ValueError,KeyError,TypeError,StopIteration) as exc:
        result['kv_ready_unavailable_reason']=f'{type(exc).__name__}: {exc}'
    return result
