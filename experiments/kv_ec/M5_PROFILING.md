# M5 Capture Profiling

## Status

Instrumentation and read-only evidence inventory are implemented. No
capture optimization, asynchronous protection, reduced safety checks or
speedup is claimed. Real M5 GPU measurements are blocked in this execution
environment: `nvidia-smi` cannot communicate with the NVIDIA driver.
Do not choose a production optimization from CPU fixtures or mixed timers.

The original M4 evidence `/tmp/kv-ec-abcd-PYvwwJ/` is unchanged.
The pre-M5 experiment sources are archived from commit
`ed40b4d604bd783ff4aa211a4e5866cf8ff83dcd`:
`/tmp/kv-ec-m5-baseline-cwtSkF/m4-source.tar`.
Archive SHA256:
`10b6540dfcce8786dd352abc8b4e3bf15315862ff7ac0472a5abb69147478478`.
The archive contains experiment source only, not model weights or KV bytes.
Use the saved M4 measured-source hashes to identify its GPU revision;
the archive commit includes later experiment-only work.

Current validation:

- Targeted CPU: 37 tests, OK (1 skipped).
- Full CPU: 231 tests in 154.612 seconds, OK (9 skipped).
- Full log:
  `/tmp/kv-ec-m5-profile-check-6CsSRH/smoke/cpu-regression.log`.
- GPU entry point attempted exactly one diagnostic smoke after CPU passed.
  A failed at NVIDIA driver preflight; no server, fault or B/C/D was started.
  Manifest: `/tmp/kv-ec-m5-profile-check-6CsSRH/smoke/trials.json`.
  A evidence:
  `/tmp/kv-ec-m5-profile-check-6CsSRH/smoke/repeat1/baseline/acceptance.json`.
  This is M5 GPU NOT VALIDATED, not a regression result for the old M4 run.
- Old M4 evidence re-audits PASS with the new read-only analyzer.
  Inventory:
  `/tmp/kv-ec-m5-baseline-cwtSkF/m4-bottleneck-inventory-v2.json`.

The seven new CPU tests cover default-off behavior, byte/metadata equality
of the actual 36-layer packer, missing CUDA measurements, wall/thread-CPU
separation, overlapping device intervals, launch correlation and rejection
of profiled smoke as performance-study admission. CPU-only profiler export
can log a device-query error in this driverless environment; no such event
is treated as a GPU measurement. Socket-restricted tests are skipped, not
Lease or Event checks.

## M4 Observations

Three independent runs, milliseconds. Protection rows use the median of
each run's maximum rank value; different rows can therefore select different
ranks. Manager XOR is the median checkpoint `xor_ms`.

| Component | M4 B | Meaning / Missing Measurement |
| --- | ---: | --- |
| Capture + D2H + hashes + validation | 241.850 | Enclosing host wall, not pure copy |
| GPU gather | unavailable | Previously mixed into capture |
| Pure device D2H | unavailable | Host `.cpu()` includes waits and allocation |
| CPU copy / concatenation | unavailable | Previously mixed into capture |
| Hash / metadata validation | unavailable | Includes repeated source verification; not isolated |
| Manager XOR encoding | 7.374 | CPU wall around encoding, excludes validation |
| Submission roundtrip | 174.062 | Includes serialization/copy, IPC, manager work and waiting |
| Manager commit interval | 89.981 | First rank submission through COMMITTED; overlaps submission |
| Control roundtrips | 16.917 | Per-rank sum of saved control IPC spans; excludes votes and gaps |
| Scheduler protection blocking | 966.505 | Encloses capture, commit, source recheck, coordination and authorization |

Do not sum these rows or interpret blocking minus listed timers as a
measured CPU bottleneck. Across-rank and enclosing intervals overlap.

M4 C medians: capture combined 238.756, survivor Gloo gather 76.701,
`cpu_xor_ms` 168.777, H2D host 12.362, writeback host 7.225,
validation 205.984 and fault-to-first-decode 954.655.
In the legacy schema `cpu_xor_ms` includes reconstruction validation and
hashing: it is NOT pure XOR arithmetic. Survivor gather here is CPU Gloo
communication of retained blocks, NOT the initial GPU index-select.
M4 D fault-to-first-decode median is 1345.999 and replay forward is 183.804.

A/B/C/D request total medians are respectively 890.925, 2001.912, 2779.919
and 2878.405. These are the old real GPU baseline, not M5 results.

## Implementation

`--capture-profile` is default-off and restricted to single-run diagnostic
studies in `m4_abcd.py`. It adds `capture-profiles/` in each B/C/D run:

- `capture-PID-SEQUENCE.trace.json`: Kineto CPU scopes and CUDA activities.
- `capture-PID-SEQUENCE.timing.json`: existing `CaptureTiming` exclusive
  host wall and thread-CPU regions, rank, snapshot identity and token offset.

Capture still performs the original 36-layer K/V selections, blocking CPU
copies, NumPy copies, concatenation, all hashes and full payload validation.
Only timing scopes and references separating those existing operations are
added. All row/slot leases, producer dependencies, events, four-rank votes,
COMMITTED validation and recovery behavior remain in place.

Scopes separate mapping/tensor checks, GPU gather launch, blocking D2H,
CPU copy, segment hashes, concatenation, payload hash and validation.
Remaining metadata work appears in unaccounted host wall. CUDA gather
kernels are attributed through launch correlation, not by adding every
capture kernel (mapping checks also launch kernels). Pure D2H comes from
device memcpy activities. Missing CUDA/CUPTI data remains `null`.

The sidecars distinguish CPU thread execution from host wall and waits;
do not call host `.cpu()` time a DMA measurement. The device activity
union handles overlap, while individual activity sums may overlap across
streams. Never add device durations to their enclosing CPU scopes.

Protection evidence also retains submission transport details and status
roundtrip counts; the Scheduler trace adds TP CPU vote waits only in the
profiling mode. Existing control trace spans remain unchanged.

Profiler start/stop/export blocks the foreground gate outside the measured
capture total. These diagnostic latencies must NOT be compared as optimized
inference latencies. Profiled smoke evidence cannot admit a three-repeat
performance study. No tensor values, raw KV or shape/stack dumps are exported.

## Commands

Run only when all four GPUs are idle. The existing driver runs the complete
CPU regression first, then applies normal GPU/port/model admission checks.
It stops at the first failure and never terminates unrelated tasks.

```bash
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
RUNS=$(mktemp -d /tmp/kv-ec-m5-profile-XXXXXX)
"$PY" -u -m experiments.kv_ec.m4_abcd \
  --gpus 0,1,2,3 --shadow-evidence /tmp/kv-ec-pause-accept-o1pg2zr8 \
  --repeat 1 --capture-profile --output "$RUNS/smoke" &&
"$PY" -m experiments.kv_ec.m5_analysis \
  --run "$RUNS/smoke" --output "$RUNS/profile-inventory.json"
```

Read-only legacy baseline inventory (output must not already exist):

```bash
"$PY" -m experiments.kv_ec.m5_analysis \
  --run /tmp/kv-ec-abcd-PYvwwJ/three \
  --output /tmp/kv-ec-m5-original-m4-inventory.json
```

## Next Gate

After a real profiled 4/4 PASS, identify the largest exclusive cost and
implement only one optimization. Candidates include batching the existing
72 transfers, pinned asynchronous capture with completed producer/copy
events, or avoiding measured redundant CPU materialization. None is yet
selected. Any asynchronous design must bound queued bytes/checkpoints,
retain source leases until reads complete, fail closed on worker errors
and never announce protection before COMMITTED.

Re-run full CPU and real GPU correctness, then compare an **unprofiled**
single-run smoke against M4 with identical model, GPU UUIDs, prompt, sampler
and output. No optimization-before/after GPU comparison exists yet.
