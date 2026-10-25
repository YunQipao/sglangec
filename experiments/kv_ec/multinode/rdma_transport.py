"""Native libibverbs transport wrapper.

The C helper implements host buffers, PD/MR/CQ/RC QP setup, TCP QP metadata
exchange, RDMA Write, completion polling, and orderly resource release.
"""

import ctypes.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from .transport import Transport


class RDMATransportUnavailable(RuntimeError):
    pass


class RDMATransport(Transport):
    name = "rdma"

    def __init__(self, helper=None, *, device="-"):
        source = Path(__file__).with_name("verbs_helper.c")
        self._temporary_directory = None if helper else Path(tempfile.mkdtemp(prefix="kv-ec-rdma-"))
        self.helper = Path(helper) if helper else self._temporary_directory / "verbs-helper"
        self.device = device
        compiler = shutil.which("cc")
        if not compiler or not ctypes.util.find_library("ibverbs") or not Path("/usr/include/infiniband/verbs.h").exists():
            if self._temporary_directory:
                shutil.rmtree(self._temporary_directory, ignore_errors=True)
            raise RDMATransportUnavailable("native RDMA requires a C compiler, libibverbs, and verbs headers")
        result = subprocess.run([compiler, "-O2", "-Wall", "-Wextra", str(source), "-libverbs", "-o", str(self.helper)],
                                capture_output=True, text=True)
        if result.returncode:
            if self._temporary_directory:
                shutil.rmtree(self._temporary_directory, ignore_errors=True)
            raise RDMATransportUnavailable(f"verbs helper compilation failed: {result.stderr.strip()}")
        self.helper.chmod(0o700)

    def send(self, host, port, payload):
        payload_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="kv-ec-rdma-", delete=False) as stream:
                payload_path = Path(stream.name)
                stream.write(payload)
            started = time.perf_counter()
            result = subprocess.run([str(self.helper), "send", host, str(port),
                                     self.device, str(payload_path)],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RDMATransportUnavailable(result.stderr.strip() or "RDMA sender failed")
            return time.perf_counter() - started
        finally:
            if payload_path is not None:
                payload_path.unlink(missing_ok=True)
            if self._temporary_directory:
                shutil.rmtree(self._temporary_directory, ignore_errors=True)


def receive_file(port, destination, *, device="-", bind_host=None):
    """Block until one RDMA Write completes; only the verbs helper receives payload."""
    source = Path(__file__).with_name("verbs_helper.c")
    compiler = shutil.which("cc")
    if not compiler or not ctypes.util.find_library("ibverbs") or not Path("/usr/include/infiniband/verbs.h").exists():
        raise RDMATransportUnavailable("native RDMA requires a C compiler, libibverbs, and verbs headers")
    with tempfile.TemporaryDirectory(prefix="kv-ec-rdma-") as temporary_directory:
        helper = Path(temporary_directory) / "verbs-helper"
        result = subprocess.run([compiler, "-O2", "-Wall", "-Wextra", str(source), "-libverbs", "-o", str(helper)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RDMATransportUnavailable(f"verbs helper compilation failed: {result.stderr.strip()}")
        helper.chmod(0o700)
        transfer_started = time.perf_counter()
        completed = subprocess.run([str(helper), "recv", str(port), device, str(destination)],
                                   capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RDMATransportUnavailable(completed.stderr.strip() or "RDMA receiver failed")
        return time.perf_counter() - transfer_started
