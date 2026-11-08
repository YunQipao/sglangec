# SGLangEC portable branch

This branch consolidates the SGLangEC online KV-cache protection and recovery
work on top of commit `21e79593c94ac6d92769a4dd4bc15391a5ec1c16`.

It intentionally contains source code, configuration, native sources, and
tests only. Historical experiment outputs, generated verification JSON,
machine-local backups, model weights, and compiled binaries are excluded.

## Clone on a new server

```bash
git clone --branch clean/sglangec-portable-20261108 \
  git@github.com:YunQipao/sglangec.git
cd sglangec
python -m pip install -e python
```

Use a CUDA/PyTorch environment compatible with the target server. Optional
RDMA paths also require the system libibverbs development/runtime packages and
an active RDMA device. Do not copy compiled `.so` files between servers; build
native helpers on the destination server.

## Read-only preflight

The following checks implementation contracts without starting a model:

```bash
PYTHONPATH=python:. python -m experiments.kv_ec.verify_live_optimizations \
  --root "$PWD"
```

On a CUDA host, require the CUDA RS and fused XOR/writeback checks as well:

```bash
PYTHONPATH=python:. python -m experiments.kv_ec.verify_live_optimizations \
  --root "$PWD" --require-cuda
```

Run the focused CPU-safe test set before launching a service:

```bash
PYTHONPATH=python:. python -m unittest \
  experiments.kv_ec.test_live_optimization_contract \
  experiments.kv_ec.test_live_fused_restore \
  experiments.kv_ec.test_online_rs \
  experiments.kv_ec.test_remote_rs \
  experiments.kv_ec.test_long_prefix \
  experiments.kv_ec.test_cold_sse_stream \
  experiments.kv_ec.test_control_backpressure \
  experiments.kv_ec.test_prefix_rs_identity
```

## Online entry point

`experiments.kv_ec.launch_online` is the guarded foreground launcher. All 15
optimizations are enabled by default and an individual optimization can be
disabled only with a repeated `--disable-opt NAME` argument. The launcher
requires an already running, owned online manager socket and a unique run ID;
it performs GPU-memory admission and port checks before starting SGLang.

Inspect the exact arguments on the destination host:

```bash
PYTHONPATH=python:. python -m experiments.kv_ec.launch_online --help
```

Use `--gpu-admission-mode strict` for publishable isolated measurements.
`shared` is intended for correctness/debug runs and its measurements must be
reported as non-isolated. The launcher never stops other users' processes.

## Scope and evidence

Passing the preflight proves code-path availability and byte-exact codec
checks. It does not by itself prove end-to-end fault recovery for a particular
model, GPU topology, or network. A new server must still run preflight, smoke,
then repeated end-to-end trials while preserving failed trials and recording
GPU/network state.
