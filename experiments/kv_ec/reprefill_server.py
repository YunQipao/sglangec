"""Ordinary Stage-E-compatible SGLang target for cold Re-Prefill controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .debug_capture import source_identity
from .launch_online import check_gpu_resources, server_argv
from .launch_debug import check_port
from .gpu_admission import args_mode, environment, metadata


def read_ids(path):
    value = json.loads(Path(path).read_text())
    ids = value.get('input_ids') if isinstance(value, dict) else value
    if (not isinstance(ids, list) or len(ids) != 256 or
            any(type(x) is not int or x < 0 for x in ids)):
        raise ValueError('exactly_256_input_ids_required')
    return ids


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def sampling():
    return dict(max_new_tokens=17, temperature=0, ignore_eos=True, top_p=1.,
                top_k=1, min_p=0., min_new_tokens=0, n=1,
                frequency_penalty=0., presence_penalty=0., repetition_penalty=1.)


def validate_generation(result):
    meta = result.get('meta_info', {})
    output = result.get('output_ids')
    if (meta.get('cached_tokens') != 0 or meta.get('prompt_tokens') != 256 or
            meta.get('completion_tokens') != 17 or not isinstance(output, list) or
            len(output) != 17 or any(type(x) is not int for x in output)):
        raise ValueError('reprefill_generation_boundary_invalid')
    return output


def launch(args):
    if not args.output.is_absolute():
        raise ValueError('absolute_output_required')
    selected_mode = args_mode(args)
    check_gpu_resources(args)
    check_port(args.port)
    identity = source_identity(args.model)
    resources = check_gpu_resources(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = dict(schema='kv-ec-reprefill-target-v1', run_id=output.name,
               model=args.model, model_identity=identity, gpus=args.gpus,
               tp_size=2, prefix_tokens=256, generation_tokens=17,
               sampling=sampling(), argv=server_argv(args))
    cfg.update(metadata(selected_mode, args.mem_fraction_static, resources))
    (output / 'run.json').write_text(json.dumps(cfg, indent=2, sort_keys=True))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        SGLANG_KV_EC_ONLINE='0', SGLANG_KV_EC_DEBUG='0',
        SGLANG_KV_EC_PAUSE_RESUME='0', SGLANG_KV_EC_GPU_RESTORE='0',
        SGLANG_KV_EC_GPU_RESUME='0', SGLANG_KV_EC_GPU_COLD_RESUME='0',
        SGLANG_KV_EC_RS_RESUME='0', SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION='0',
        SGLANG_KV_EC_PREPARED_IDENTITY=json.dumps(identity))
    os.execve(sys.executable, server_argv(args), dict(os.environ, **environment(selected_mode)))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('launch')
    q.add_argument('--model', required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--gpus', default='auto')
    q.add_argument('--port', type=int, required=True)
    q.add_argument('--mem-fraction-static', type=float, default=.6)
    q.add_argument('--gpu-admission-mode', choices=('strict', 'shared'), default='strict')
    q.add_argument('--prefill-chunk-size', type=int, default=-1)
    q.add_argument('--tp', type=int, choices=(2,), default=2)
    args = parser.parse_args()
    launch(args)


if __name__ == '__main__':
    main()
