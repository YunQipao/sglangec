"""Opt-in CPU-manager handoff -> bounded workers -> native RDMA -> remote RS.

Only CPU-owned, locally committed rank arrays enter this queue. No scheduler,
CUDA, or pool references are accepted. Local node files are retained for this
finite experiment; remote parity is volatile and expires with the RS store TTL.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import queue
import re
import threading
import time

from .node_block_export import MAX_NODE_BYTES, descriptor, pack
from .online_rs_identity import identity_for, stripe_id
from .remote_rs_client import RemoteRSClient
from .rs_stripe_builder import manifest_sha, validate_manifest


class Cancelled(Exception):
    pass


class OnlineRSExporter:
    def __init__(
        self,
        *,
        node_id,
        run_id,
        checkpoint_version,
        requests,
        remote_host,
        remote_port=30300,
        rdma_device="mlx5_0",
        rdma_gid_index=None,
        rdma_port_num=1,
        timeout=180,
        control_timeout=10,
        rdma_timeout=60,
        poll_interval=0.1,
        workers=2,
        max_pending=2,
        max_bytes=256 * 1024**2,
        max_checkpoints=16,
        client_factory=None,
        allow_fixture=False,
    ):
        limits = (timeout, control_timeout, rdma_timeout, poll_interval)
        if (
            type(node_id) is not int
            or node_id not in (0, 1)
            or any(not math.isfinite(v) or v <= 0 for v in limits)
            or any(
                type(v) is not int or v < 1
                for v in (workers, max_pending, max_bytes, max_checkpoints)
            )
            or workers > max_pending
            or not remote_host
            or not 1 <= remote_port <= 65535
            or not rdma_device
            or not 1 <= rdma_port_num <= 255
            or (
                rdma_gid_index is not None
                and (type(rdma_gid_index) is not int or rdma_gid_index < 0)
            )
        ):
            raise ValueError("invalid online RS configuration/limits")
        if (
            not isinstance(requests, dict)
            or not 0 < len(requests) <= 1024
            or any(not re.fullmatch(r"[0-9a-f]{64}", k) for k in requests)
            or len(set(requests.values())) != len(requests)
        ):
            raise ValueError("unique request SHA256 -> protection ID map required")
        for value in (run_id, checkpoint_version, *requests.values()):
            if not isinstance(value, str) or not 0 < len(value) <= 128:
                raise ValueError("nonempty global identity strings <=128 required")
        if allow_fixture and client_factory is None:
            raise ValueError("fixtures require an explicitly injected mock client")
        self.node_id, self.run_id = node_id, run_id
        self.checkpoint_version, self.requests = checkpoint_version, dict(requests)
        self.timeout, self.control_timeout = timeout, control_timeout
        self.rdma_timeout, self.poll_interval = rdma_timeout, poll_interval
        self.rdma = dict(
            device=rdma_device, gid_index=rdma_gid_index, port_num=rdma_port_num
        )
        self.max_pending, self.max_bytes = max_pending, max_bytes
        self.max_checkpoints, self.accepted = max_checkpoints, 0
        self.allow_fixture = allow_fixture
        self.client_factory = client_factory or (
            lambda: RemoteRSClient(remote_host, remote_port, control_timeout)
        )
        self.queue = queue.Queue(max_pending)
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.active, self.bytes_pending = {}, 0
        self.threads = [
            threading.Thread(target=self._worker, daemon=True, name=f"kv-online-rs-{i}")
            for i in range(workers)
        ]
        for thread in self.threads:
            thread.start()

    @classmethod
    def from_config(cls, path):
        config = json.loads(Path(path).read_text())
        # The CLI has no mock/fixture or TCP-payload mode.
        if "allow_fixture" in config or "client_factory" in config:
            raise ValueError("mock options forbidden in online CLI")
        return cls(**config)

    @staticmethod
    def _save(job, state, **fields):
        job["status"].update(global_state=state, updated_at_unix=time.time(), **fields)
        directory = job["directory"]
        temporary = directory / "global_status.tmp"
        temporary.write_text(json.dumps(job["status"], indent=2, allow_nan=False))
        temporary.replace(directory / "global_status.json")

    def enqueue(self, directory, entries, blocks):
        """Nonblocking admission on the CPU manager, never packs/hashes/networks."""
        directory = Path(directory)
        first = entries[0]
        now = time.monotonic()
        job = dict(
            directory=directory,
            created=now,
            deadline=now + self.timeout,
            status=dict(
                format="kv-online-rs-status-v1",
                local_state="LOCAL_COMMITTED",
                local_snapshot_id=first["snapshot_id"],
                created_at_unix=time.time(),
                node_id=self.node_id,
                transport="CPU_MOCK" if self.allow_fixture else "rdma",
                remote_outcome="NOT_SUBMITTED",
                metrics={},
            ),
        )
        protection_id = self.requests.get(first["request_id"])
        if protection_id is None:
            self._save(job, "SKIPPED_UNMAPPED")
            return "SKIPPED_UNMAPPED"
        identity = identity_for(
            self.run_id, protection_id, self.checkpoint_version, first
        )
        sid = stripe_id(identity)
        job["status"].update(stripe_id=sid, global_protection=identity)
        size = sum(b.nbytes for b in blocks)
        with self.lock:
            if (directory / "global_status.json").exists():
                return "DUPLICATE"
            elif self.stop.is_set():
                state = "CANCELLED"
            elif sid in self.active:
                state = "FAILED_IDENTITY_CONFLICT"
            elif (
                len(self.active) >= self.max_pending
                or self.bytes_pending + size > self.max_bytes
                or self.accepted >= self.max_checkpoints
                or size + 1024**2 > MAX_NODE_BYTES
            ):
                state = "DROPPED_CAPACITY"
            else:
                # Ownership transfers by references to the manager's private
                # copies. The manager never mutates these arrays after commit.
                job.update(entries=deepcopy(entries), blocks=tuple(blocks), size=size)
                self._save(job, "QUEUED")
                self.active[sid] = job
                self.bytes_pending += size
                self.accepted += 1
                self.queue.put_nowait(job)
                return "QUEUED"
        self._save(job, state)
        return state

    def _check(self, job):
        if self.stop.is_set():
            raise Cancelled("exporter shutting down")
        remaining = job["deadline"] - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("online protection deadline exceeded")
        return remaining

    def _call(self, job, client, op, **fields):
        client.timeout = min(self.control_timeout, self._check(job))
        result = client.call(op, **fields)
        self._check(job)
        return result[0]

    def _pause(self, job):
        self.stop.wait(min(self.poll_interval, self._check(job)))

    def _validate_remote(self, status, sid, own, key=None):
        manifest = status["manifest"]
        validate_manifest(manifest, allow_fixture=self.allow_fixture)
        if (
            status["stripe_id"] != sid
            or manifest["stripe_id"] != sid
            or status["manifest_sha256"] != manifest_sha(manifest)
            or not status.get("handle")
            or manifest["nodes"][self.node_id] != own
            or (key and any(status.get(k) != v for k, v in key.items()))
        ):
            raise ValueError("remote binding/descriptor mismatch")

    def _export(self, job):
        metrics = job["status"]["metrics"]
        metrics["queue_wait_ms"] = (time.monotonic() - job["created"]) * 1000
        self._check(job)
        identity, sid = (job["status"][k] for k in ("global_protection", "stripe_id"))
        self._save(job, "PACKING")
        started = time.perf_counter()
        ranks = [dict(m, global_protection=identity) for m in job["entries"]]
        block = pack(
            self.node_id,
            [b.tobytes() for b in job["blocks"]],
            ranks,
            allow_fixture=self.allow_fixture,
        )
        own = descriptor(block, allow_fixture=self.allow_fixture)
        metrics["pack_validate_ms"] = (time.perf_counter() - started) * 1000
        self._check(job)
        started = time.perf_counter()
        # Persist D0/D1 on its source node before remote registration. P is on
        # the RS receiver. This does not introduce a third physical domain.
        path = job["directory"] / "global_node.bin"
        with path.open("xb") as stream:
            stream.write(block)
        metrics["node_persist_ms"] = (time.perf_counter() - started) * 1000
        job.pop("blocks")
        job.pop("entries")
        self._save(job, "REGISTERING", node_descriptor=own)
        client = self.client_factory()
        started = time.perf_counter()
        status = self._call(
            job,
            client,
            "register_node",
            stripe_id=sid,
            node_id=self.node_id,
            descriptor=own,
        )
        metrics["register_ms"] = (time.perf_counter() - started) * 1000
        self._save(job, "WAITING_PEER")
        started = time.perf_counter()
        while status["state"] == "REGISTERING":
            self._pause(job)
            status = self._call(job, client, "get_stripe_status", stripe_id=sid)
        metrics["peer_wait_ms"] = (time.perf_counter() - started) * 1000
        if status["state"] not in ("PENDING", "COMMITTED"):
            raise ValueError("remote stripe closed: " + status["state"])
        self._validate_remote(status, sid, own)
        key = {k: status[k] for k in ("handle", "manifest_sha256")}
        self._save(job, "UPLOADING", **key, remote_outcome="UNKNOWN")
        client.timeout = min(self.control_timeout, self._check(job))
        started = time.perf_counter()
        status, tx_stats, disposition = client.rdma_submit_node_block(
            sid,
            self.node_id,
            block,
            **key,
            **self.rdma,
            timeout=min(self.rdma_timeout, self._check(job)),
        )
        metrics["rdma_upload_wall_ms"] = (time.perf_counter() - started) * 1000
        metrics["rdma_native"] = tx_stats
        # upload_wall includes control, remote validation/polling and may include
        # RS encode. Native verbs timings are separate; do not add them together.
        self._check(job)
        self._validate_remote(status, sid, own, key)
        self._save(job, "WAITING_COMMIT", upload_disposition=disposition)
        started = time.perf_counter()
        while status["state"] == "PENDING":
            self._pause(job)
            status = self._call(job, client, "get_stripe_status", stripe_id=sid, **key)
        metrics["global_commit_wait_ms"] = (time.perf_counter() - started) * 1000
        self._validate_remote(status, sid, own, key)
        if (
            status["state"] != "COMMITTED"
            or status["submitted_nodes"] != [0, 1]
            or not re.fullmatch(r"[0-9a-f]{64}", status.get("parity_sha256", ""))
        ):
            raise ValueError("remote parity not committed")
        metrics["server_rs_encode_ms"] = status["rs_encode_ms"]
        metrics["server_commit_ms"] = status["commit_ms"]
        self._save(
            job,
            "GLOBAL_COMMITTED",
            remote_outcome="COMMITTED",
            parity_sha256=status["parity_sha256"],
            remote_retention="volatile_store_ttl",
            recovery_byte_exact="NOT_YET_VERIFIED",
        )

    def _worker(self):
        while True:
            try:
                job = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.stop.is_set():
                    return
                continue
            try:
                self._export(job)
            except Exception as exc:
                state = (
                    "CANCELLED"
                    if isinstance(exc, Cancelled)
                    else "EXPIRED" if isinstance(exc, TimeoutError) else "FAILED"
                )
                try:
                    self._save(job, state, error=f"{type(exc).__name__}: {exc}")
                except OSError as write_error:
                    # Local XOR remains committed even if global evidence disk fails.
                    print(f"online_rs_status_error: {write_error}", flush=True)
            finally:
                with self.lock:
                    self.active.pop(job["status"]["stripe_id"], None)
                    self.bytes_pending -= job["size"]
                self.queue.task_done()
                job.clear()

    def close(self, timeout=5):
        """Cancel queued work; active native calls have their own finite timeout."""
        self.stop.set()
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(max(0, deadline - time.monotonic()))
