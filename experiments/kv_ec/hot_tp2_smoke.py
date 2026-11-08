"""Small explicit CUDA smoke. No model, remote host, server, or fault injection."""

import argparse
import json

from .hot_tp2 import MIB, Options, Rejected, protect, recover


def smoke(*, enabled=False, native_library=None, p2p=False):
    if enabled is not True:
        return dict(status="DISABLED", cuda_verified=False)
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        return dict(status="SKIP_CUDA_UNAVAILABLE", cuda_verified=False,
                    p2p_verified=False, model_hot_switch_verified=False)
    from .hot_tp2_cuda import LowPriorityCapture, VerifiedCudaSink
    from .hot_tp2_native import NativeCodec
    rng = np.random.default_rng(20261104)
    a, b = rng.bytes(2 * MIB + 17), rng.bytes(2 * MIB + 17)
    opts = Options(enabled=True, chunk_mib=1, depth=2, max_bytes=8 * MIB)
    cp, _ = protect(a, b, checkpoint="synthetic-cuda-smoke-only", options=opts)
    for lost, survivor, expected in ((0, b, a), (1, a, b)):
        sink = VerifiedCudaSink(cp, lost, options=opts)
        recover(cp, lost_rank=lost,
                fetch_survivor=lambda i, n: survivor[i * MIB:i * MIB + n],
                sink=sink, options=opts)
        sink.wait_before_submit(torch.cuda.current_stream(sink.device))
        if sink.target.cpu().numpy().tobytes() != expected:
            raise Rejected("smoke_full_destination_mismatch")
        def consume(index, raw):
            if raw != expected[index * MIB:index * MIB + len(raw)]:
                raise Rejected("capture_smoke_mismatch")
        LowPriorityCapture(options=opts).capture(sink.target, consume, validate_lease=lambda: None)
        del sink
    native_status = "NOT_REQUESTED"
    if native_library:
        codec = NativeCodec(native_library, enabled=True)
        # One small odd-sized chunk exercises the vector path and scalar tail.
        aa, bb = a[:65539], b[:65539]
        d0 = torch.from_numpy(np.frombuffer(aa, dtype=np.uint8).copy()).to("cuda:0")
        d1 = torch.from_numpy(np.frombuffer(bb, dtype=np.uint8).copy()).to("cuda:0")
        parity, target = torch.empty_like(d0), torch.empty_like(d0)
        stream = torch.cuda.Stream(device="cuda:0")
        ticket = codec.cuda_into(d0, d1, parity, operation="encode", stream=stream)
        ticket["done"].synchronize()
        from .multinode.rs_codec import encode
        reference = encode(np.frombuffer(aa, dtype=np.uint8), np.frombuffer(bb, dtype=np.uint8))[2]
        if not np.array_equal(parity.cpu().numpy(), reference):
            raise Rejected("cuda_encode_oracle_mismatch")
        for op, survivor, expected in (("recover0", d1, aa), ("recover1", d0, bb),
                                        ("xor", d0, bytes(x ^ y for x, y in zip(reference, aa)))):
            ticket = codec.cuda_into(parity, survivor, target, operation=op, stream=stream)
            ticket["done"].synchronize()
            if target.cpu().numpy().tobytes() != expected:
                raise Rejected("cuda_decode_oracle_mismatch")
        native_status = "SMOKE_PASS"
        if p2p:
            if torch.cuda.device_count() < 2:
                raise Rejected("p2p_smoke_requires_two_real_devices")
            remote = d1.to("cuda:1")
            for op, expected in (("recover0", aa),
                                 ("xor", bytes(x ^ y for x, y in zip(reference, bb)))):
                ticket = codec.cuda_into(parity, remote, target, operation=op, stream=stream,
                                         allow_p2p=True)
                ticket["done"].synchronize()
                if target.cpu().numpy().tobytes() != expected or remote.cpu().numpy().tobytes() != bb:
                    raise Rejected("p2p_direct_target_or_survivor_mismatch")
    elif p2p:
        raise Rejected("p2p_smoke_requires_explicit_cuda_built_native_library")
    return dict(status="SYNTHETIC_CUDA_SMOKE_PASS", cuda_verified=True,
                native_cuda=native_status, p2p_verified=bool(p2p),
                gpudirect=False, model_hot_switch_verified=False,
                scope="scratch byte tensors only; no SGLang/real KV/TP communicator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--native-library")
    parser.add_argument("--p2p", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke(enabled=args.enable, native_library=args.native_library, p2p=args.p2p),
                     indent=2))


if __name__ == "__main__":
    main()
