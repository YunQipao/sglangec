# Stage E repeated-trial port lifecycle

## Diagnosis and evidence (2026-10-31)

Evidence batch:
`/media/sda1/iic/luotao/sglangec-env/phase1-smoke.0aN9Cu70/reprefill-stability`

- `trial-000-82a8c3a86c20497fa71d8d77ca2c2cf7/trial.json` reports
  `scheme=reprefill`, `pass_fail=PASS`, `returncode=0`.
- Its local `origin/owned-process.json` records PID/PGID/SID 890352, UID 1002,
  start_ticks 300817816, boot ID `5c91e5b6-d75d-43c9-990a-9f37aae1a4b0`.
  `origin/exit.json` reports `owned_stop_requested`, returncode 0. The server
  log ends with application shutdown complete and server process 890352
  finished. This establishes launcher shutdown, not proof that every descendant
  or later listener was gone.
- The archived 233 evidence also records returncode 0 for Origin PID 3567269
  and Re-Prefill Target PID 3568811, with `owned_stop_requested`.
- After stopping both Origins, Re-Prefill reuses **232 port 30984 as an SSH
  local forwarding listener** to 233 Target port 30986. The inference command
  in `commands.jsonl` confirms that address. It is not an RS native port.
- Local, connection-free `ssh -G iic@10.102.0.233` reports `controlmaster auto`,
  `controlpersist 600`, and a persistent control socket. The old tunnel launch
  inherited these settings. Its cleanup terminated only the Popen client and
  never checked listener release. Forwarding handled by a persistent SSH
  master can therefore outlive that client.
- `cleanup.json` nevertheless says PASS with no errors. The next trial's
  `trial.json` reports node0 `PORT_OCCUPIED port=30984` during preflight.
  Preflight checks LISTEN sockets, so TIME_WAIT does not explain this error.

The leading causal explanation is a forwarding listener retained by SSH
multiplexing, together with incomplete cleanup verification. The historical
logs do not record the PID holding 30984 after cleanup, and the inspection
environment has an isolated PID namespace. The exact historical listener
owner cannot be proven retrospectively. No existing listener was terminated.

Additional confirmed lifecycle gaps: cleanup searched only the launcher's
PGID rather than its whole SID; process identity publication preceded the
cleanup try/finally; tunnel startup exceptions could escape caller cleanup;
the synchronous Return operation could leave a native receiver child after
its launcher exited; preflight was only a snapshot and did not coordinate
concurrent trials. Two-level normally has no tunnel, while Re-Prefill and the
optional paired comparison do; both tunnel paths now use the same fix.

GPU admission did not cause this port-specific preflight failure. Historical
shared GPU occupancy remains recorded separately. A foreign experiment or
user may occupy any configured port; the historical logs cannot exclude that
alternative for the unidentified listener.

## Scope of the fix

Use the smaller lifecycle repair rather than a dynamic port allocator. Old
configuration files retain their exact ports, commands and connection
addresses; successful serial trials reuse them. `trial-ports.json` records
Origin, Target, Return, Delivery and tunnel roles per host. RS stays at the
configured `192.168.2.245:30701`; the RS algorithm, native transport and GIDs
are unchanged. Both hosts' port sets are checked against the existing RS
native range obtained by preflight from the running RS process.

Each host holds advisory `flock` locks for its entire configured trial port
set, from before service startup until reverse-order cleanup. Lock files live
under `/tmp/kv-ec-port-locks-<uid>/`, independent of checkout and TMPDIR. Files
are not unlinked: unlinking could give a waiter a different lock inode. A lock
file's presence is not evidence of a live owner; the kernel lock is decisive.
On coordinator loss, model supervisors expire after 90 seconds and guards
after 150 seconds, leaving time for the bounded model cleanup before dropping
admission locks. Normal reverse-order cleanup does not wait for lease expiry.
Acquisition has three attempts with 0.1/0.2-second backoff. Listener conflicts
are not retried or bypassed by rotating ports. Cooperating concurrent trials
fail explicitly on contention. These locks are **not socket reservations**:
an unrelated program can still bind, and kernel bind plus listener ownership
validation determine whether service startup succeeds.

The tunnel is a foreground SSH process with multiplexing and persistence
disabled. `-F /dev/null` excludes configured forwards and background commands;
the configured destination must be directly reachable using default SSH keys
or ssh-agent and known_hosts. Host key checking is not disabled. Tunnel
readiness requires an owned listener and `/health`, not `/server_info`.
Failed startup always attempts verified cleanup before propagating the error.

Process receipts include UID, PID, PGID, SID (`sid`, plus legacy `session`),
boot ID, start_ticks and trial ID. Cleanup validates session membership and
the inherited trial ID in each member's process environment, as well as
individual process identities, then uses pidfd signals to prevent a recycled
numeric PID from being signalled. A libc fallback supports Conda Python builds
that omit `os.pidfd_open`. Startup refuses before spawning children if safe
pidfd signalling is unavailable. Signals escalate only for verified trial members,
with bounded 30/10/5-second grace periods. New PGIDs in the original SID are
included. Unknown listeners and other UIDs are never terminated. An escaped,
unverifiable listener causes explicit failure rather than an unsafe kill.

Evidence includes `port-plan.json`, `ports/owned-process.json`, per-service
`process-cleanup.json`, `port-release.json`, `port-check.json`, tunnel logs and
identity, `rs-after-cleanup.json`, `cleanup.json`, and archived equivalents
from 233. A leaked process or listener prevents cleanup PASS. Each batch has
`repeat-summary.json` with independent trial results and the number not
started. Failure still stops the batch so a leak cannot be hidden. Output
directories and evidence files use exclusive creation and preserve history.

## CPU/mock validation

Run locally; no SSH, GPU launch or RS operation is needed:

```bash
cd /media/sda1/iic/luotao/sglangec
PY=/media/sda1/iic/miniconda3/envs/sglang-ec/bin/python
"$PY" -m unittest \
  experiments.kv_ec.test_perf_ports \
  experiments.kv_ec.test_perf_stage_e \
  experiments.kv_ec.test_gpu_cold_resume \
  experiments.kv_ec.test_rs_cold_resume \
  experiments.kv_ec.test_rs_recovery_source \
  experiments.kv_ec.test_rdma_transport -v
```

Coverage includes sequential reuse, concurrent lock contention, foreign
listeners, different host occupancy, exception release, legacy configuration,
RS exclusion/health, publication failure, failed tunnel readiness, a launcher
exiting before its child, and a child changing PGID. A real CPU TCP listener
test runs three rounds on socket-capable hosts; it explicitly skips if the
sandbox rejects TCP bind. That skip is not a successful TCP validation.

Local result on 2026-10-31: **153 tests, 149 passed, 4 skipped, no failures**.
The four skips include the sandbox-denied CPU TCP bind test
and three existing native/hardware prerequisites or opt-in tests. Log:
`/tmp/kv-ec-port-release-tests.log`. `git diff --check`, Python compilation,
and syntax checks of the documented shell/Python validation commands passed.
No SSH connection to 233, GPU experiment, RS restart or existing-service
termination was performed for this validation.

## Real two-node validation commands (operator-run only)

These commands contact 233. They have **not** been executed by the agent.
First synchronize the reviewed Python changes to the corresponding checkout
on 233, preserving any independent work. Both checkouts must have the same
HEAD, tracked diff and Python file hashes. The current local HEAD is
`ceaeae2b05cf431de01ba1f12e190d4f2825737b`. Do not change HEAD merely to deploy
this patch: the existing RS startup-commit guard must still pass. Do not
restart RS to work around a failed identity/capacity check.

```bash
cd /media/sda1/iic/luotao/sglangec
ssh -o BatchMode=yes iic@10.102.0.233 \
  'cd /media/sda1/iic_data/luotao/sglangec/sglang && git status --short && git rev-parse HEAD'

# After reviewing the remote checkout, transfer these six Python files.
scp experiments/kv_ec/{perf_ports,perf_stage_e,perf_preflight,perf_benchmark,test_perf_ports,test_perf_stage_e}.py \
  iic@10.102.0.233:/media/sda1/iic_data/luotao/sglangec/sglang/experiments/kv_ec/

# Read-only diagnostic for a pre-existing 232 listener; do not kill it.
ss -ltnp 'sport = :30984'

PY=/media/sda1/iic/miniconda3/envs/sglang-ec/bin/python
CFG=/media/sda1/iic/luotao/sglangec-env/phase1-smoke.0aN9Cu70/reprefill-stability/config.json
RUNROOT=$(mktemp -d /media/sda1/iic/luotao/sglangec-env/stage-e-port-validation.XXXXXXXX)
export PY CFG RUNROOT

# Require capacity for all six retained stripes; this is not an RS restart.
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$CFG" --output "$RUNROOT/preflight-six.json" --repeat 6
```

If that preflight passes, this local orchestrator runs the two batches and
polls only RS health every five seconds. It fails if health/epoch changes,
any round fails, or either host lacks successful release/cleanup evidence.
It never sends an RS stop/rebind command. The RS monitor is sampled evidence,
not proof against an arbitrarily brief outage between samples.

```bash
"$PY" - <<'PY'
import json, os, pathlib, subprocess, sys, threading, time, zipfile
from experiments.kv_ec.remote_rs_client import RemoteRSClient

root = pathlib.Path(os.environ['RUNROOT'])
cfg = json.loads(pathlib.Path(os.environ['CFG']).read_text())
assert cfg['tp_size'] == 2 and cfg['gpu_admission']['mode'] == 'shared'
assert cfg['rs_server'] == dict(host='192.168.2.245', port=30701)
baseline = json.loads((root / 'preflight-six.json').read_text())
assert baseline['status'] == 'PASS'
epoch = baseline['rs']['server_epoch']
stop, errors = threading.Event(), []

def monitor():
    with (root / 'rs-health.jsonl').open('x') as out:
        while True:
            try:
                health, payload = RemoteRSClient('192.168.2.245', 30701, timeout=10).call('health')
                assert not payload and health['state'] == 'READY' and health['server_epoch'] == epoch
                record = dict(time=time.time(), status='PASS', health=health)
            except Exception as exc:
                errors.append(repr(exc))
                record = dict(time=time.time(), status='FAIL', error=repr(exc))
            out.write(json.dumps(record) + '\n')
            out.flush()
            if stop.wait(5):
                return

thread = threading.Thread(target=monitor)
thread.start()
try:
    for mode in ('two-level', 'reprefill'):
        output = root / mode
        subprocess.run([sys.executable, '-m', 'experiments.kv_ec.perf_benchmark',
            'repeat', '--config', os.environ['CFG'], '--output', str(output),
            '--recovery-mode', mode, '--repeat', '3'], check=True)
        summary = json.loads((output / 'repeat-summary.json').read_text())
        assert summary['status'] == 'PASS' and len(summary['trials']) == 3
        for trial in sorted(output.glob('trial-*')):
            read = lambda name: json.loads((trial / name).read_text())
            assert read('trial.json')['functional_correctness_pass'] is True
            assert read('cleanup.json')['status'] == 'PASS'
            assert read('port-check.json')['status'] == 'PASS'
            assert read('port-release.json')['status'] == 'PASS'
            assert read('rs-after-cleanup.json')['server_epoch'] == epoch
            local_receipts = list(trial.glob('*/process-cleanup.json'))
            assert local_receipts
            assert all(json.loads(p.read_text())['status'] == 'PASS' for p in local_receipts)
            with zipfile.ZipFile(trial / 'json-evidence-233.zip') as archive:
                for name in ('port-check.json', 'port-release.json'):
                    assert json.loads(archive.read(name))['status'] == 'PASS'
                receipts = [n for n in archive.namelist() if n.endswith('/process-cleanup.json')]
                assert receipts
                assert all(json.loads(archive.read(n))['status'] == 'PASS' for n in receipts)
        assert not errors, errors
finally:
    stop.set()
    thread.join(timeout=15)
assert not thread.is_alive() and not errors, errors
health, payload = RemoteRSClient('192.168.2.245', 30701, timeout=10).call('health')
assert not payload and health['state'] == 'READY' and health['server_epoch'] == epoch
print('PASS: two-level 3/3; reprefill 3/3; cleanup checked on both hosts; RS epoch unchanged')
print(root)
PY
```

To validate five rounds later, use a new output root and `--repeat 5`; never
reuse a previous batch directory. Account for RS's retained stripe budget
before starting another batch. If 30984 is already held by a historical SSH
master, admission will refuse it; this patch intentionally does not terminate
that unidentified pre-existing process or restart a shared SSH master.
