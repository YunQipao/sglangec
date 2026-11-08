"""Bounded immutable CPU/remote replicas using the existing checked TCP frame.

No authentication: isolated trusted experiment LAN only. TTL tombstones prevent
ID reuse. CPU replicas are not valid evidence for a physical-node loss.
"""

from copy import deepcopy
import threading
import time
import uuid

from ..node_block_export import canonical, descriptor, sha


REQUEST_STATE_FIELDS = (
    "snapshot_id", "request_id", "request_id_encoding", "request_version",
    "checkpoint_version", "manager_epoch", "req_generation", "req_pool_idx",
    "token_start", "token_count", "token_ids_sha256", "lifetime",
    "completion_dependency", "pool_class",
)


def request_state(raw, *, allow_fixture=False):
    """Extract lifecycle identity separately from the full content descriptor."""
    meta = descriptor(raw, allow_fixture=allow_fixture)["metadata"]
    ranks = meta["ranks"]
    # Pool index and generation can be rank-local; preserve every rank's exact
    # lifecycle tuple rather than requiring artificial cross-rank equality.
    return {"ranks": [{key: rank[key] for key in REQUEST_STATE_FIELDS} for rank in ranks]}


class ReplicaStore:
    def __init__(self, *, max_bytes, max_objects, ttl, allow_fixture=False):
        if any(type(v) is not int or v <= 0 for v in (max_bytes, max_objects, ttl)):
            raise ValueError("positive bounded replica budgets required")
        self.max_bytes, self.max_objects, self.ttl = max_bytes, max_objects, ttl
        self.allow_fixture = allow_fixture
        self.objects = {}
        self.epoch = uuid.uuid4().hex
        self.lock = threading.RLock()
        # Compatible with existing serial Handler, without inventing bandwidth.
        self.metrics = {k: 0 for k in ("requests", "errors", "rx_stream_bytes", "tx_stream_bytes",
                                     "rx_payload_bytes", "tx_payload_bytes", "receive_ms", "send_ms")}

    def expire(self):
        for item in self.objects.values():
            if time.monotonic() >= item["deadline"]:
                item["raw"] = None

    def dispatch(self, request, payload):
        with self.lock:
            if request.get("transport", "tcp") != "tcp":
                raise ValueError("physical243 replica TCP only; RDMA fails closed")
            if len(payload) > self.max_bytes:
                raise ValueError("replica payload budget exhausted")
            self.expire()
            op = request["op"]
            if op == "health" and not payload:
                return dict(state="READY", mode="replication", server_epoch=self.epoch), b""
            key = request["object_id"]
            if not isinstance(key, str) or not 1 <= len(key) <= 256:
                raise ValueError("bounded object ID required")
            item = self.objects.get(key)
            if op == "put_replica":
                meta = descriptor(payload, allow_fixture=self.allow_fixture)
                if request.get("descriptor") != meta:
                    raise ValueError("replica descriptor/integrity mismatch")
                state = request_state(payload, allow_fixture=self.allow_fixture)
                state_sha = sha(canonical(state))
                digest = sha(canonical(meta))
                if item is not None:
                    if (item["raw"] is None or item["binding"]["descriptor_sha256"] != digest
                            or item["binding"]["request_state_sha256"] != state_sha):
                        raise ValueError("immutable replica conflict/expired")
                    if descriptor(item["raw"], allow_fixture=self.allow_fixture) != item["descriptor"]:
                        raise ValueError("stored replica corruption")
                else:
                    if len(self.objects) >= self.max_objects or sum(len(v["raw"] or b"") for v in self.objects.values()) + len(payload) > self.max_bytes:
                        raise ValueError("replica admission budget exhausted")
                    item = dict(raw=memoryview(payload).tobytes(), descriptor=deepcopy(meta),
                                request_state=state,
                                deadline=time.monotonic() + self.ttl,
                                binding=dict(object_id=key, handle=uuid.uuid4().hex,
                                             server_epoch=self.epoch, descriptor_sha256=digest,
                                             request_state_sha256=state_sha))
                    self.objects[key] = item
                return deepcopy(item["binding"]), b""
            if payload or op != "get_replica" or item is None or item["raw"] is None:
                raise ValueError("live replica required")
            if any(request.get(k) != v for k, v in item["binding"].items()):
                raise ValueError("stale replica binding")
            if descriptor(item["raw"], allow_fixture=self.allow_fixture) != item["descriptor"]:
                raise ValueError("stored replica corruption")
            return deepcopy(item["binding"]), item["raw"]


class LocalReplicaClient:
    """CPU baseline adapter: identical request bindings without TCP copying."""

    def __init__(self, store):
        if not isinstance(store, ReplicaStore):
            raise TypeError("ReplicaStore required")
        self.store = store

    def call(self, op, *, payload=b"", **fields):
        return self.store.dispatch(dict(op=op, **fields), payload)


def put_replica(client, object_id, raw, *, allow_fixture=False):
    result, _ = client.call("put_replica", object_id=object_id, payload=raw,
                            descriptor=descriptor(raw, allow_fixture=allow_fixture))
    return {k: result[k] for k in ("object_id", "handle", "server_epoch", "descriptor_sha256",
                                   "request_state_sha256")}


def get_replica(client, binding, expected_descriptor, *, allow_fixture=False):
    returned_binding, raw = client.call("get_replica", **binding)
    if any(returned_binding.get(k) != v for k, v in binding.items()):
        raise ValueError("replica response/request binding mismatch")
    if (sha(canonical(expected_descriptor)) != binding["descriptor_sha256"] or
            sha(canonical(request_state_from_descriptor(expected_descriptor))) != binding["request_state_sha256"] or
            descriptor(raw, allow_fixture=allow_fixture) != expected_descriptor):
        raise ValueError("replica response/request binding mismatch")
    return raw


def request_state_from_descriptor(value):
    meta = value.get("metadata", {})
    ranks = meta.get("ranks", [])
    if not ranks:
        raise ValueError("replica descriptor request state required")
    state = {"ranks": [{key: rank.get(key) for key in REQUEST_STATE_FIELDS} for rank in ranks]}
    if any(v is None or v == "" for row in state["ranks"] for v in row.values()):
        raise ValueError("replica descriptor request state incomplete/mixed")
    return state
