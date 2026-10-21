# M3 Recovery Status

## Implemented in this increment

- `m3_recovery.py` validates the requested 256-token chunk ranges, one lost
  TP rank, exact survivor sets, independent per-chunk checkpoint versions,
  manager epoch, request/slot generations, all-layer writeback, and byte-exact
  validation. Resume authorization is impossible until every requested chunk
  is present and valid. A failed proof permanently fails that plan.
- `scheduler_shadow.pack_chunk` can capture one 256-token chunk at an explicit
  256-aligned prompt offset. The default remains chunk 0, preserving M2.
- `m3_performance.py` aggregates per-trial TTFT, TPOT, total request time,
  protection/parity cost, and recovery-stage timings. It rejects mixed
  configurations and reports median/P95.

## Not yet enabled

The live `SchedulerShadowRecovery.run_online` path still injects a fault only
on Rank 0, gathers around that rank, and operates on chunk 0. The pause
coordinator also binds the online checkpoint and resume proof to one
`transaction_id-chunk0` snapshot. Therefore `--lost-rank 1/2/3` and
`--recover-tokens 512/768` are not accepted by the live GPU runner yet.

Do not treat `M3RecoveryPlan` proofs as hardware evidence: it is a CPU-side
fail-closed evidence validator. It does not inject GPU faults, perform
collectives, write KV, or attest to a running scheduler. No M3 GPU run or
A/B/C/D benchmark results are claimed by this increment.

## Required live integration before GPU acceptance

The next online implementation must extend the existing interlock without
replacing its lease state machine:

1. Bind `lost_rank`, recovery token range, and the ordered per-chunk snapshot
   identities into coordinator transaction identity and every rank vote.
2. Commit one independently versioned 256-token parity checkpoint per range
   before fault injection. Preserve the request-row and full slot lease until
   all chunks are restored and verified.
3. Have only the selected rank save local rollback bytes and inject corruption.
   Recovery must gather exactly the three non-target ranks; the target rank's
   pre-fault payload must never enter reconstruction.
4. Restore and verify every layer's K/V bytes for each chunk. Any failed write,
   generation/version check, timeout, manager restart, or failed rollback
   keeps the gate fail-closed and must not authorize decode.
5. Emit one rank-local JSON proof per rank and chunk, then authorize resume
   only after all four ranks attest to all chunks.

The coordinator restart case must remain terminal. A rollback that cannot be
proved byte-exact is not a recovery success, and neither rollback outcome may
release the protected row/slots after a recovery failure.

## CPU checks

Run the M3-only tests without GPU dependencies:

```bash
python -m unittest experiments.kv_ec.test_m3_recovery experiments.kv_ec.test_m3_performance -v
```

In the current interpreter, full `experiments/kv_ec` discovery is not
available: PyTorch and Transformers are absent. No dependencies were installed
or changed. Once the project test environment is active, run:

```bash
python -m unittest discover -s experiments/kv_ec -p 'test_*.py' -v
```

## Performance record format

Each JSON trial must include `mode` (`baseline`, `protected`,
`online_recovery`, or `prefill_retry`), a complete identical `config` object
(model/checkpoint, prompt token hash and length, output length, GPU UUIDs,
TP/configuration, and software revision), and whichever of these metrics apply:

`ttft_ms`, `tpot_ms`, `request_total_ms`, `protection_overhead_ms`,
`parity_commit_ms`, `recovery_ms`, `cpu_xor_ms`, `gather_ms`, `h2d_ms`,
`kv_writeback_ms`, `validation_ms`.

Use the same input IDs, sampling settings, model, output length, selected
devices, and server configuration in all four modes. Record algorithmic
durations from monotonic/performance clocks; exclude manual coordination,
debugger pauses, and deliberate sleeps. Keep separate trials and aggregate:

```bash
python -m experiments.kv_ec.m3_performance /path/to/trials/*.json \
  --config /path/to/expected-config.json
```

Do not run the GPU acceptance command until the live integration above is
implemented and the selected four GPUs have been independently confirmed idle.
