"""CPU equivalence and fail-closed checks, not GPU performance evidence."""

from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.kv_ec.scheduler_shadow import pack_chunk, checked_buffers
from experiments.kv_ec.snapshot import digest
from experiments.kv_ec.xor_codec import encode


def capture_fixture():
    raw = torch.arange(300 * 2 * 128 * 2, dtype=torch.int64).to(torch.uint8)
    k = raw.view(torch.bfloat16).reshape(300, 2, 128)
    v = raw.bitwise_xor(91).view(torch.bfloat16).reshape(300, 2, 128)
    pool = NS(k_buffer=[k] * 36, v_buffer=[v] * 36,
              get_kv_buffer=lambda layer: (pool.k_buffer[layer], pool.v_buffer[layer]))
    mapping = NS(physical_slots=tuple(range(256, 0, -1)), slot_generations=(7,) * 256,
                 req_pool_idx=1, req_generation=5, request_version="v1", tp_rank=0)
    return pool, mapping, NS(rid="fixture", origin_input_ids=[1] * 768)


def capture(pool, mapping, req):
    return pack_chunk(pool, mapping, req, {}, snapshot_id="fixture", manager_epoch="epoch")


class CaptureOptimizationTests(unittest.TestCase):
    def test_one_mapping_scan_and_identical_layout_hashes_and_parity(self):
        pool, mapping, req = capture_fixture()
        counts = dict(min=0, max=0, unique=0)
        with ExitStack() as stack:
            for name in counts:
                original = getattr(torch.Tensor, name)
                def counted(tensor, *args, _name=name, _original=original, **kwargs):
                    counts[_name] += 1
                    return _original(tensor, *args, **kwargs)
                stack.enter_context(patch.object(torch.Tensor, name, counted))
            payload, meta = capture(pool, mapping, req)
        self.assertEqual(counts, dict(min=1, max=1, unique=1))
        pieces = [tensor.view(torch.uint8)[list(mapping.physical_slots)].reshape(-1).numpy()
                  for layer in range(36) for tensor in pool.get_kv_buffer(layer)]
        reference = np.concatenate(pieces)
        np.testing.assert_array_equal(payload, reference)
        self.assertEqual(meta["sha256"], digest(reference))
        self.assertEqual(meta["k_sha256"], digest(np.concatenate(pieces[::2])))
        self.assertEqual(meta["v_sha256"], digest(np.concatenate(pieces[1::2])))
        for index, record in enumerate(meta["layer_records"]):
            for i, kind in enumerate(("k", "v")):
                self.assertEqual(record[kind + "_sha256"], digest(pieces[index * 2 + i]))
        np.testing.assert_array_equal(encode([payload] * 3 + [reference]),
                                      encode([reference] * 4))

    def test_mapping_is_revalidated_on_every_capture(self):
        pool, mapping, req = capture_fixture()
        capture(pool, mapping, req)
        for slots in ((1,) * 256, tuple(range(256)), tuple(range(256))[:-1]):
            mapping.physical_slots = slots
            with self.subTest(slots=len(slots)), patch(
                    "experiments.kv_ec.scheduler_shadow.select_raw_bytes") as select:
                with self.assertRaisesRegex(ValueError, "mapping|exceeds"):
                    capture(pool, mapping, req)
                select.assert_not_called()

    def test_late_layer_owner_layout_and_capacity_still_checked(self):
        for kind, change in (
                ("capacity", lambda t: t[:256]),
                ("dtype", lambda t: t.float()),
                ("layout", lambda t: t.transpose(1, 2))):
            pool, mapping, req = capture_fixture()
            pool.v_buffer[35] = change(pool.v_buffer[35])
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "layer=35:v"):
                capture(pool, mapping, req)
        pool, mapping, req = capture_fixture()
        original_get = pool.get_kv_buffer
        pool.get_kv_buffer = lambda layer: (
            (pool.k_buffer[layer].clone(), pool.v_buffer[layer])
            if layer == 35 else original_get(layer))
        with self.assertRaisesRegex(ValueError, "layer=35:k"):
            capture(pool, mapping, req)

    def test_recovery_buffer_checks_do_not_reuse_capture_validation(self):
        pool, mapping, req = capture_fixture()
        capture(pool, mapping, req)
        slots = torch.tensor(mapping.physical_slots)
        checked_buffers(pool, 0, slots)
        slots[0] = slots[1]
        with self.assertRaisesRegex(ValueError, "mapping"):
            checked_buffers(pool, 0, slots)
