"""Parent-run three-host source worker: reads only its local authentic KV.

Run the same fresh stripe ID on D0 and D1 concurrently. Each worker registers,
waits with a bounded deadline, submits once, and performs five recoveries of the
OTHER node using only its own surviving bytes. No SSH/service/process control.
"""
import argparse
import json
import time
from pathlib import Path

from ..node_block_export import descriptor, sha
from ..remote_rs_client import binding
from .admission import local_identity, placement
from .remote import client
from .three_host import ARTIFACTS, _read_node, _write_json


def run(config, role, physical_id, node_path, stripe_id, output, *, wait_s=120, connect=client):
    import re
    placement(config)
    if config['transport'] != 'tcp' or role not in ('D0', 'D1'):
        raise ValueError('explicit TCP data role required; RDMA fails closed')
    if type(wait_s) is not int or not 1 <= wait_s <= 300:
        raise ValueError('wait deadline must be in 1..300 seconds')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', stripe_id or ''):
        raise ValueError('fresh bounded stripe ID required')
    local_identity(config, role, physical_id)
    node = int(role[-1])
    raw, meta = _read_node(node_path, node)
    if 3 * len(raw) > config['max_bytes']:
        raise ValueError('local block alone exceeds stripe reservation budget')
    output = Path(output).resolve()
    if output.parent != ARTIFACTS.resolve():
        raise ValueError('new direct child of design_study/artifacts required')
    output.mkdir(exist_ok=False)
    report = dict(schema='kv-ec-source-trial-v1', state='RUNNING', role=role,
                  stripe_id=stripe_id, physical_id=physical_id, transport='TCP',
                  config_sha256=placement(config), source_descriptor=meta, operations=[],
                  missing_original_read=False, physical_failure=False,
                  scope='five sequential checked TCP CPU-DRAM recoveries; not disk/GPU/RDMA/RTO')
    deadline = time.monotonic() + wait_s

    def poll(wanted):
        while time.monotonic() < deadline:
            # Bound each RPC too: waiting for the other worker does not extend
            # the total rendezvous/recovery deadline indefinitely.
            peer.timeout = min(config['timeout_s'], max(0.001, deadline-time.monotonic()))
            status, _ = peer.call('get_stripe_status', stripe_id=stripe_id, transport='tcp')
            if status['state'] in wanted:
                return status
            if status['state'] not in ('REGISTERING', 'PENDING'):
                raise ValueError('stripe closed before rendezvous: ' + status['state'])
            time.sleep(min(0.2, max(0, deadline-time.monotonic())))
        raise TimeoutError('bounded source-worker rendezvous deadline exceeded')

    try:
        peer = connect(config)
        peer.call('register_node', stripe_id=stripe_id, node_id=node, descriptor=meta, transport='tcp')
        status = poll(('PENDING', 'COMMITTED'))
        key = binding(status)
        if status['manifest']['nodes'][node] != meta:
            raise ValueError('registered source binding mismatch')
        peer.call('submit_node_block', node_id=node, payload=raw, transport='tcp', **key)
        status = poll(('COMMITTED',))
        if binding(status) != key or status.get('retained_node_block_bytes') != 0:
            raise ValueError('committed binding/source-release mismatch')
        report['committed_status'] = status
        lost = 1-node
        for repeat in range(5):
            if time.monotonic() >= deadline:
                raise TimeoutError('source trial total deadline exceeded')
            peer.timeout = min(config['timeout_s'], deadline-time.monotonic())
            start = time.perf_counter_ns()
            returned, recovered = peer.call('reconstruct_missing_node', missing_nodes=[lost],
                                            payload=raw, transport='tcp', **key)
            elapsed = time.perf_counter_ns()-start
            if (binding(returned) != key or returned.get('recovered_node') != lost or
                    descriptor(recovered) != status['manifest']['nodes'][lost]):
                raise ValueError('recovery response content/request binding mismatch')
            report['operations'].append(dict(repeat=repeat, missing_node=lost,
                elapsed_ns=elapsed, recovery_input_node=node, recovered_sha256=sha(recovered),
                recovered_bytes=len(recovered), transfer=peer.transfers[-1]))
        report['state'] = 'PASS'
        return report
    except Exception as exc:
        report.update(state='FAIL', error=f'{type(exc).__name__}:{exc}')
        raise
    finally:
        report['transfers'] = getattr(locals().get('peer'), 'transfers', [])
        _write_json(output/'raw.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--role', choices=('D0', 'D1'), required=True)
    p.add_argument('--physical-id', required=True)
    p.add_argument('--node-block', type=Path, required=True)
    p.add_argument('--stripe-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-s', type=int, default=120)
    a = p.parse_args()
    result = run(json.loads(a.config.read_text()), a.role, a.physical_id,
                 a.node_block, a.stripe_id, a.output, wait_s=a.wait_s)
    print(json.dumps(dict(state=result['state'], transport='TCP', role=a.role)))


if __name__ == '__main__':
    main()
