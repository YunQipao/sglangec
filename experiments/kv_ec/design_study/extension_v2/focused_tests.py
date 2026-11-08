"""Offline regression suite with hard guards against service/network/GPU use."""
import argparse
import io
import os
from pathlib import Path
import socket
import subprocess
import unittest
from unittest.mock import patch

import torch

from .evidence import save

NAMES = (
    "experiments.kv_ec.design_study.extension_v2.test_recovery_adapters",
    "experiments.kv_ec.design_study.extension_v2.test_modes",
    "experiments.kv_ec.design_study.extension_v2.test_pipeline",
    "experiments.kv_ec.test_launch_online",
    "experiments.kv_ec.test_online_rs",
    "experiments.kv_ec.test_gpu_cold_resume",
    "experiments.kv_ec.test_cold_sse_stream",
    "experiments.kv_ec.test_online_chunk.ChunkTests",
    "experiments.kv_ec.test_online_chunk.ManagerTests",
    "experiments.kv_ec.test_online_chunk.AsyncWorkerTests.test_cancel_waits_for_d2h_without_reading_host_or_opening_socket",
    "experiments.kv_ec.test_online_chunk.AsyncWorkerTests.test_worker_completion_and_cpu_aggregation_with_mock_transport",
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO()
    def blocked(*args,**kwargs):
        raise AssertionError("offline focused suite forbids real Popen/bind/connect")
    original_popen=subprocess.Popen
    def readonly_metadata(argv,*args,**kwargs):
        if not kwargs.get("shell") and isinstance(argv,(list,tuple)) and tuple(argv) in (("git","rev-parse","HEAD"),("uname","-a")):
            return original_popen(argv,*args,**kwargs)
        return blocked()
    with patch.object(torch.cuda,"is_available",return_value=False),patch.object(torch.cuda,"_lazy_init",blocked),patch.object(socket.socket,"bind",blocked),patch.object(socket.socket,"connect",blocked),patch.object(subprocess,"Popen",readonly_metadata),patch.object(os,"kill",blocked),patch.object(os,"execve",blocked),patch.object(os,"execv",blocked):
        suite=unittest.defaultTestLoader.loadTestsFromNames(NAMES)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(total=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        skipped=len(result.skipped),failures=len(result.failures),errors=len(result.errors),client_date="2026-11-03",
        guards="only readonly git-rev-parse/uname children; bind/connect/kill/exec/CUDA init forbidden; CUDA availability mocked false",
        evidence_class="synthetic_fixture",hardware_performance_evidence=False)
    save(args.output/"counts.json",report)
    with (args.output/"unittest.log").open("x") as handle:
        handle.write(stream.getvalue())
    print(stream.getvalue())
    print(report)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=="__main__":
    main()
