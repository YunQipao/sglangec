# M5 Hash Optimization: Read-Only Re-Audit

## Original Failure

Smoke `/tmp/kv-ec-m5-next-2Yab76/smoke` ran at
`1a4a027b85f5e662cb3c096d13c78c82e9e4de3b`.
Three `/tmp/kv-ec-m5-hash-three` ran at
`61d3df5b0a498cd252f730161b44f4c386d5799c`.
The latter commit added 39 evidence JSON files only. All measured sources
match their historical Git blobs; the entire inference `python` tree and
experiment Python module set are identical between these commits.

All twelve GPU mode runs completed, but the final comparison included the
Git commit while preflight compared only a partial source hash map.
The original `trials.json` remains FAIL with
`ValueError: configuration_changed_since_smoke`. No original summary or
acceptance file is created, replaced or relabeled by re-audit.

## Policy

- `MEASURED_SOURCES` conservatively fingerprints every non-test Python module
  in `experiments/kv_ec`, including control, leases, codecs, capture, manager,
  protocol, launch, request and audit dependencies.
- `compatible_config` is shared by preflight and final smoke admission.
  It compares all configuration fields except the two source provenance
  fields, then independently proves those fields using Git.
- Changed commits are accepted only for additions of evidence JSON/log files
  in root `kv-ec-*`/`kv_ec_*` directories. Runtime module SHA-256 maps and the
  inference Git tree must also match. Real code/dependency changes fail.
- Live preflight rejects tracked, staged and untracked non-evidence changes,
  fingerprints the actual model and synthetic input, and checks selected
  GPU UUIDs plus software versions before any server is launched.
- New studies also record and compare GPU selection, model path, ports and
  startup/pause timeouts. Legacy smoke missing these launch flags cannot
  admit a new live three-repeat run. Offline historical re-audit compares
  effective parameters from records and audited runtime checks; it does not
  invent unrecorded historical timeout values.
- Missing historical hashes are supplemented with the exact commit blob
  ID and SHA-256 for each module, not treated as equal by omission.
  Recorded anchors must match those blobs. Missing commit/dependency/CPU
  evidence produces INCONCLUSIVE. A recorded hash mismatch fails.
- Git supplementation establishes historical tree equivalence, not an
  independent attestation of unrecorded historical worktree modifications
  or external package contents. The limitation is explicit in the proof.
- A re-audit only recovers the specific terminal configuration failure,
  never failed or incomplete GPU trials. It checks the successful CPU log
  footer, required non-skipped safety tests, three completed distinct
  A/B/C/D rounds, real TP4 logs, input/output tokens, runtime configuration,
  parity recovery/writeback and retry proofs using the existing auditors.
- All original study and smoke artifacts are hashed before/after re-audit.
  Reports are exclusive new files outside both input directories.
- The comparison CLI re-executes the full audit and checks the independent
  report against freshly reconstructed values and artifact hashes. A saved
  `state: PASS` is not trusted as an override.

No GPU experiment, server, CUDA preflight or CPU suite is launched by the
offline CLI. The full CPU regression is a separate development action.

## Commands

Run from the repository root. Output files must not already exist.

```bash
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python

"$PY" -m experiments.kv_ec.m4_abcd \
  --verify-study /tmp/kv-ec-m5-hash-three \
  --smoke-evidence /tmp/kv-ec-m5-next-2Yab76/smoke \
  --output /tmp/kv-ec-m5-hash-reverified.json

"$PY" -m experiments.kv_ec.m5_compare \
  --baseline /tmp/kv-ec-abcd-PYvwwJ/three \
  --candidate /tmp/kv-ec-m5-hash-reverified.json \
  --expect-repeat 3 --require-positive \
  --output /tmp/kv-ec-m5-hash-vs-m4.json

"$PY" -m experiments.kv_ec.m5_compare \
  --baseline /tmp/kv-ec-m5-opt-NrBKnD/three \
  --candidate /tmp/kv-ec-m5-hash-reverified.json \
  --expect-repeat 3 --require-positive \
  --output /tmp/kv-ec-m5-hash-vs-slot.json
```

The comparison distinguishes `original_run_PASS` from
`completed_run_offline_reaudit_PASS`, preserving original manifest FAIL.
The report's reconstructed summary is derived from original records, not
the missing historical `summary.json`.

## Validation

- Targeted CPU suite: 46 tests, OK.
- Full CPU regression: 257 tests in 130.498 seconds, OK (9 sandbox socket
  skips; no lease/event/integrity checks skipped).
- Full CPU log: `/tmp/kv-ec-m5-reaudit-cpu-8y1xr3.log`.
- Independent re-audit: `/tmp/kv-ec-m5-hash-reverified.json`, PASS.
- Version proof: 39 historical experiment modules, 9 recorded SHA-256
  anchors and 30 Git-blob supplements per revision; 39 evidence-only
  additions, identical inference tree. Both actual commits are preserved.
- 241 original study/smoke artifacts hashed before and after audit.
- Original study manifest still FAIL; original summary still absent.
- Both formal comparison commands with `--require-positive` exit zero.
- No GPU experiment, server or new model inference was run for this work.

## Saved Unprofiled Results

Medians over three independent runs, milliseconds; improvements are positive.
B minus A is the difference of medians, not a paired-sample median.

| Metric | M4 | Slot M5 | Hash M5 | Hash vs Slot | Hash vs M4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A total | 890.925 | 893.396 | 884.173 | 1.03% | 0.76% |
| B total | 2001.912 | 1886.752 | 1779.278 | 5.70% | 11.12% |
| B protection blocking | 966.505 | 937.142 | 788.892 | 15.82% | 18.38% |
| C fault to first decode | 954.655 | 1002.045 | 954.380 | 4.76% | 0.03% |
| C total | 2779.919 | 2792.367 | 2500.774 | 10.44% | 10.04% |
| D fault to first decode | 1345.999 | 1381.977 | 1077.079 | 22.06% | 19.98% |
| B minus A | 1110.986 | 993.355 | 895.105 | 9.89% | 19.43% |

Both comparisons pass the existing B/C admission heuristics. Three runs
remain pilot evidence, not statistical proof of efficacy. Unprofiled CPU
thread Hash/validation time and GPU memory peaks were not recorded and
must remain unmeasured; no profiled/unprofiled request latency is mixed.

These results support starting M6 design/prototyping, not granting async
protection acceptance. M6 must separately define producer completion,
immutable payload ownership, backpressure, version binding and lease/event
lifetimes, including failure/cancellation and four-rank commit/resume gates.
