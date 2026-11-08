"""One-shot remote evidence adapter; HTTP is issued only by the coordinator."""
import argparse
import json
import os
import time
from pathlib import Path

from .cold_resume_state import load_checkpoint, write_json
from .gpu_kv_restore import _require


def run(action, output, client_path=None):
    cfg = json.loads((output / 'run.json').read_text())
    _, state, _, _ = load_checkpoint(cfg['state_checkpoint'], cfg.get('rs_source'))
    ids = state['origin_input_ids']
    if action == 'prepare':
        write_json(output / 'submitted.json', dict(run_id=cfg['run_id'],
            request_id=state['request_id'], operation_id=os.environ['KV_EC_PERF_RUN_ID'],
            input_ids_sha256=state['input_ids_sha256'], token_count=len(ids), transport='coordinator-sse'))
        write_json(output / 'submitted_input_ids.json', ids)
        return
    _require((output / 'submitted.json').exists(), 'sse_submission_not_reserved')
    client = json.loads(client_path.read_text())
    submission = json.loads((output / 'submitted.json').read_text())
    _require(client['operation_id'] == submission['operation_id'] == os.environ['KV_EC_PERF_RUN_ID']
             and client['input_ids_sha256'] == submission['input_ids_sha256'] == state['input_ids_sha256']
             and client['response']['output_ids'] == client['output_token_ids'], 'sse_operation_or_input_binding_invalid')
    _require(client['request_id'] == state['request_id'] and
             client['response']['meta_info']['prompt_tokens'] == len(ids) and
             len(client['output_token_ids']) == 17 and
             client['output_token_ids'][0] == state['pending_token_id'],
             'sse_restored_request_binding_invalid')
    write_json(output / 'http_response.json', client['response'])
    write_json(output / 'client-sse.json', client)
    deadline = time.monotonic() + cfg['timeout']
    while True:
        reports = []
        for rank in range(cfg['tp_size']):
            try:
                report = json.loads((output / f'rank_{rank}.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            _require(report.get('failure_reason') is None, 'sse_rank_reported_failure')
            reports.append(report)
        if len(reports) == cfg['tp_size']:
            return
        _require(time.monotonic() < deadline, 'sse_rank_report_timeout')
        time.sleep(.1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'finish'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--client', type=Path)
    a = p.parse_args()
    run(a.action, a.output, a.client)
