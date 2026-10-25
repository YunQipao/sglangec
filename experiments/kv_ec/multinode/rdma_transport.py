"""Native libibverbs transport wrapper.

The C helper implements host buffers, PD/MR/CQ/RC QP setup, TCP QP metadata
exchange, RDMA Write, completion polling, and orderly resource release.
"""

import ctypes.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import math

from .transport import Transport


class RDMATransportUnavailable(RuntimeError):
    pass


class RDMATransport(Transport):
    name = "rdma"

    def __init__(self, helper=None, *, device="-", gid_index=None, port_num=1,
                 timeout=120):
        source = Path(__file__).with_name("verbs_helper.c")
        self._temporary_directory = None if helper else Path(tempfile.mkdtemp(prefix="kv-ec-rdma-"))
        self.helper = Path(helper) if helper else self._temporary_directory / "verbs-helper"
        self.device = device
        self.gid_index = gid_index
        self.port_num = int(port_num)
        self.timeout = float(timeout)
        if (not 1 <= self.port_num <= 255 or not math.isfinite(self.timeout)
                or self.timeout <= 0 or (gid_index is not None and int(gid_index) < 0)):
            if self._temporary_directory:
                shutil.rmtree(self._temporary_directory, ignore_errors=True)
            raise ValueError("invalid RDMA port/GID/timeout")
        self.last_stats = None
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

    def _environment(self):
        env = os.environ.copy()
        env["KV_EC_RDMA_PORT"] = str(self.port_num)
        if self.gid_index is not None:
            env["KV_EC_RDMA_GID_INDEX"] = str(self.gid_index)
        else:
            env.pop("KV_EC_RDMA_GID_INDEX", None)
        env.pop("KV_EC_RDMA_BIND_HOST", None)
        return env

    @staticmethod
    def available():
        """Return whether the native build prerequisites are present.

        This intentionally does not claim that a usable HCA is present.  A
        caller must still run the verbs operation and surface its error; in
        particular, this helper is never used to silently select TCP.
        """
        return bool(
            shutil.which("cc")
            and ctypes.util.find_library("ibverbs")
            and Path("/usr/include/infiniband/verbs.h").exists()
        )

    @staticmethod
    def _parse_stats(stdout, elapsed):
        values = {"elapsed_s": elapsed}
        line = next((x for x in stdout.splitlines() if x.startswith("KV_EC_RDMA_STATS")), "")
        for key, value in re.findall(r"([a-z_]+)=([^ ]+)", line):
            try:
                if value in ("SUCCESS", "FAILED", "ERROR"):
                    values[key] = value
                else:
                    values[key] = float(value) if "." in value else int(value)
            except ValueError:
                values[key] = value
        values.setdefault("completion_status", "success")
        return values

    def send_with_stats(self, host, port, payload):
        payload_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="kv-ec-rdma-", delete=False) as stream:
                payload_path = Path(stream.name)
                stream.write(payload)
            started = time.perf_counter()
            try:
                result = subprocess.run([str(self.helper), "send", host, str(port),
                                         self.device, str(payload_path)],
                                        capture_output=True, text=True,
                                        timeout=self.timeout, env=self._environment())
            except subprocess.TimeoutExpired as exc:
                raise RDMATransportUnavailable(
                    f"RDMA sender timeout after {self.timeout}s"
                ) from exc
            if result.returncode:
                raise RDMATransportUnavailable(result.stderr.strip() or "RDMA sender failed")
            stats = self._parse_stats(result.stdout, time.perf_counter() - started)
            stats.update(transport="rdma", rdma_device=self.device,
                         gid_index=self.gid_index, port_num=self.port_num,
                         bytes=len(payload), helper_stdout=result.stdout.strip())
            self.last_stats = stats
            return stats
        finally:
            if payload_path is not None:
                payload_path.unlink(missing_ok=True)

    def send(self, host, port, payload):
        return self.send_with_stats(host, port, payload)["elapsed_s"]

    def close(self):
        if self._temporary_directory:
            shutil.rmtree(self._temporary_directory, ignore_errors=True)
            self._temporary_directory = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def receive_file(port, destination, *, device="-", bind_host=None,
                 gid_index=None, port_num=1, timeout=120, return_stats=False):
    """Block until one RDMA Write completes; only the verbs helper receives payload."""
    if (not 1 <= int(port) <= 65535 or not 1 <= int(port_num) <= 255
            or not math.isfinite(float(timeout)) or float(timeout) <= 0
            or (gid_index is not None and int(gid_index) < 0)):
        raise ValueError("invalid RDMA listen port/GID/port number/timeout")
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
        env = os.environ.copy()
        env["KV_EC_RDMA_PORT"] = str(port_num)
        if gid_index is not None:
            env["KV_EC_RDMA_GID_INDEX"] = str(gid_index)
        else:
            env.pop("KV_EC_RDMA_GID_INDEX", None)
        if bind_host:
            env["KV_EC_RDMA_BIND_HOST"] = str(bind_host)
        else:
            env.pop("KV_EC_RDMA_BIND_HOST", None)
        try:
            completed = subprocess.run([str(helper), "recv", str(port), device, str(destination)],
                                       capture_output=True, text=True, timeout=timeout,
                                       env=env)
        except subprocess.TimeoutExpired as exc:
            raise RDMATransportUnavailable(
                f"RDMA receiver timeout after {timeout}s"
            ) from exc
        if completed.returncode:
            raise RDMATransportUnavailable(completed.stderr.strip() or "RDMA receiver failed")
        stats = RDMATransport._parse_stats(
            completed.stdout, time.perf_counter() - transfer_started
        )
        stats.update(transport="rdma", rdma_device=device, gid_index=gid_index,
                     port_num=port_num, bytes=Path(destination).stat().st_size,
                     helper_stdout=completed.stdout.strip())
        return stats if return_stats else stats["elapsed_s"]
