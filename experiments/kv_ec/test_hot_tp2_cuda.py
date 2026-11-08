"""CPU tests for CUDA rejection/order checks; GPU body is explicit smoke only."""

import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from experiments.kv_ec.hot_tp2 import Options, Rejected, protect
from experiments.kv_ec.hot_tp2_cuda import LowPriorityCapture, VerifiedCudaSink
from experiments.kv_ec.hot_tp2_native import require_p2p
from experiments.kv_ec.hot_tp2_smoke import smoke


class CudaGateCpuTests(unittest.TestCase):
    def test_default_off_no_cuda_context_or_allocation(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA touched")):
            with self.assertRaisesRegex(Rejected, "disabled"):
                LowPriorityCapture()
            self.assertEqual(smoke()["status"], "DISABLED")

    def test_unavailable_cuda_is_skip_never_pass(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(smoke(enabled=True)["status"], "SKIP_CUDA_UNAVAILABLE")
            with self.assertRaisesRegex(Rejected, "real_cuda"):
                LowPriorityCapture(options=Options(enabled=True))

    def test_p2p_fails_closed_without_silent_fallback(self):
        def no_probe(*args):
            raise AssertionError("disabled feature probed hardware")
        with self.assertRaisesRegex(Rejected, "explicit_cpu_pinned_fallback"):
            require_p2p(0, 1, capability=no_probe)
        for target, source, can in ((0, 1, False), (0, 0, True), (-1, 1, True)):
            with self.assertRaises(Rejected):
                require_p2p(target, source, enabled=True, capability=lambda *args: can)
        require_p2p(0, 1, enabled=True, capability=lambda *args: True)

    def test_full_destination_hash_catches_middle_corruption_after_event(self):
        cp, _ = protect(b"abcdefg", b"gfedcba", checkpoint="hash-middle",
                        options=Options(enabled=True, chunk_mib=1))
        class Event:
            def __init__(self): self.waited = False
            def synchronize(self): self.waited = True
        class Readback:
            def __init__(self, event): self.event = event
            def numpy(self):
                self.assert_waited()
                import numpy as np
                return np.frombuffer(b"abcXefg", dtype=np.uint8)
            def assert_waited(self):
                if not self.event.waited:
                    raise AssertionError("readback before CUDA event")
        event = Event()
        sink = object.__new__(VerifiedCudaSink)
        sink.cp, sink.lost_rank = cp, 0
        slot = NS(index=0, size=7, validated=event, bad_host=torch.tensor(False),
                  readback=Readback(event))
        with self.assertRaisesRegex(Rejected, "full_cuda_destination"):
            sink._drain(slot)
        self.assertTrue(event.waited)

    def test_submit_stream_wait_requires_verified_target(self):
        sink = object.__new__(VerifiedCudaSink)
        sink.device, sink.verified = "fake-cuda0", False
        waited = []
        stream = NS(device="fake-cuda0", wait_event=waited.append)
        with self.assertRaisesRegex(Rejected, "full_validation"):
            sink.wait_before_submit(stream)
        self.assertEqual(waited, [])
        sink.verified, sink.ready = True, object()
        sink.wait_before_submit(stream)
        self.assertEqual(waited, [sink.ready])


@unittest.skipUnless(os.environ.get("KV_EC_HOT_TP2_CUDA_SMOKE") == "1",
                     "explicit short CUDA smoke is OFF")
class ExplicitCudaSmoke(unittest.TestCase):
    def test_scratch_full_validation_capture_and_optional_native(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable; CUDA/P2P NOT VERIFIED")
        result = smoke(enabled=True, native_library=os.environ.get("KV_EC_HOT_TP2_NATIVE_LIBRARY"))
        self.assertEqual(result["status"], "SYNTHETIC_CUDA_SMOKE_PASS")
        self.assertFalse(result["model_hot_switch_verified"])


if __name__ == "__main__":
    unittest.main()
