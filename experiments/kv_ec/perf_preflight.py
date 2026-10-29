"""Stage E admission. No CUDA initialization, protection or service launch.

CUDA admission compiles/loads a disposable shared object without launching a
kernel or calling a CUDA runtime API. Only its own temporary files are removed.

RS budgets come from the *listening process*, not operator assertions. Old
servers have no budget RPC: fail closed if /proc cannot identify that process.
"""

import argparse
import base64
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from .perf_benchmark import command, identity

MODULE = 'experiments.kv_ec.perf_preflight'


def cuda_environment(node):
    """Pure configuration: identical values on the coordinator and SSH node."""
    def path(value, error):
        if (not isinstance(value, str) or not value or ':' in value or
                any(c in value for c in ('\n', '\r', '\0')) or
                not Path(value).is_absolute() or '..' in Path(value).parts or value == '/'):
            raise ValueError(error + ': expected one absolute path, not a colon-separated list')
        return str(Path(value))
    home = path(node.get('cuda_home'), 'CUDA_HOME_INVALID')
    def dirs(key, defaults):
        values = node.get(key, defaults)
        if not isinstance(values, list) or not values:
            raise ValueError('CUDA_CONFIG_INVALID: ' + key + ' must be a nonempty list')
        return list(dict.fromkeys(path(v, 'CUDA_CONFIG_INVALID: ' + key) for v in values))
    includes = dirs('cuda_include_dirs', [home + '/include', home + '/targets/x86_64-linux/include'])
    libraries = dirs('cuda_library_dirs', [home + '/lib64', home + '/lib', home + '/targets/x86_64-linux/lib'])
    if any('stubs' in Path(p).parts for p in libraries):
        raise ValueError('CUDA_CONFIG_INVALID: runtime library directories must not contain stubs')
    fixed = dict(CUDA_HOME=home, CUDA_PATH=home, CUDACXX=home + '/bin/nvcc', NVCC=home + '/bin/nvcc')
    tmpdir = node.get('tmpdir', '/tmp')
    if (not isinstance(tmpdir, str) or not tmpdir or ':' in tmpdir or
            any(c in tmpdir for c in ('\n', '\r', '\0')) or
            not Path(tmpdir).is_absolute() or '..' in Path(tmpdir).parts or tmpdir == '/'):
        raise ValueError('TMPDIR_INVALID: expected one absolute path')
    fixed.update(TMPDIR=tmpdir, TMP=tmpdir, TEMP=tmpdir)
    # An installed Conda nvcc must not shadow a separately selected Toolkit.
    prefixes = dict(PATH=':'.join(dict.fromkeys([home + '/bin', str(Path(node['conda_prefix']) / 'bin')])),
                    LIBRARY_PATH=':'.join(libraries), LD_LIBRARY_PATH=':'.join(libraries),
                    CPATH=':'.join(includes), CPLUS_INCLUDE_PATH=':'.join(includes))
    return fixed, prefixes


def check_tmpdir(node, minimum_bytes=1):
    """Verify the configured per-host temporary directory without deleting user data."""
    tmpdir = Path(node.get('tmpdir', '/tmp'))
    if not tmpdir.is_dir() or not os.access(tmpdir, os.W_OK | os.X_OK):
        raise ValueError('TMPDIR_NOT_WRITABLE: ' + str(tmpdir))
    usage = shutil.disk_usage(tmpdir)
    if usage.free < minimum_bytes:
        raise ValueError(f'TMPDIR_INSUFFICIENT_SPACE: {tmpdir} free={usage.free} required={minimum_bytes}')
    try:
        fd, path = tempfile.mkstemp(prefix='kv-ec-preflight-', dir=tmpdir)
        os.close(fd)
        Path(path).unlink()
    except OSError as exc:
        raise ValueError(f'TMPDIR_TEMPFILE_FAILED: {tmpdir}: {exc}') from exc
    return dict(path=str(tmpdir), free_bytes=usage.free, required_bytes=minimum_bytes,
                writable=True, tempfile_probe=True)


def validate_config(cfg):
    if (type(cfg.get('prefix_tokens')) is not int or cfg['prefix_tokens'] != 256 or
            type(cfg.get('tp_size')) is not int or cfg['tp_size'] != 2 or
            cfg.get('rs') != [2, 1] or any(type(n) is not int for n in cfg['rs'])):
        raise ValueError('config_requires_256_tokens_TP2_RS_2_1')
    if len(cfg.get('nodes', [])) != 2:
        raise ValueError('config_requires_two_nodes')
    if cfg['nodes'][0].get('ssh') or not cfg['nodes'][1].get('ssh'):
        raise ValueError('run_on_232: node0 must be local, node1 must use SSH')
    for name, minimum in (('resource_interval', .5), ('trial_timeout', 1),
                          ('protection_timeout', 1), ('min_disk_bytes', 1),
                          ('max_node_bytes', 1)):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise ValueError('invalid_config_' + name)
    for name in ('min_disk_bytes', 'max_node_bytes', 'trial_timeout', 'protection_timeout'):
        if type(cfg[name]) is not int:
            raise ValueError('integer_config_required: ' + name)
    if not .05 <= cfg.get('mem_fraction_static', .6) <= .95:
        raise ValueError('invalid_memory_fraction')
    def port(value):
        if type(value) is not int or not 1024 <= value <= 65535:
            raise ValueError('invalid_unprivileged_port: ' + str(value))
        return value
    for name in ('remote_root',):
        if not Path(cfg[name]).is_absolute() or '..' in Path(cfg[name]).parts or cfg[name] == '/':
            raise ValueError('absolute_dedicated_path_required: ' + name)
    rs = cfg['rs_server']
    ipaddress.IPv4Address(rs['host'])
    ipaddress.IPv4Address(cfg['return_host'])
    port(rs['port'])
    for i, node in enumerate(cfg['nodes']):
        cuda_environment(node)
        for key in ('repo', 'python', 'model', 'template', 'input_ids', 'conda_prefix'):
            value = node.get(key, '')
            if not value or not Path(value).is_absolute() or '..' in Path(value).parts:
                raise ValueError(f'node{i}_absolute_path_required: {key}')
            # SCP legacy remote paths are shell-interpreted. Restrict paths.
            if not re.fullmatch(r'/[A-Za-z0-9_./-]+', value):
                raise ValueError(f'node{i}_unsafe_path: {key}')
        if node.get('ssh') and not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+', node['ssh']):
            raise ValueError('unsafe_ssh_destination')
        if not re.fullmatch(r'\d+,\d+', node.get('gpus', '')) or len(set(node['gpus'].split(','))) != 2:
            raise ValueError('two_explicit_distinct_gpu_indices_required')
        if type(node.get('gid_index')) is not int or node['gid_index'] < 0:
            raise ValueError('invalid_gid_index')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', node.get('rdma_device', '')):
            raise ValueError('invalid_rdma_device')
        port(node['origin_port'])
    for ports in ((cfg['nodes'][0]['origin_port'], cfg['return_port']),
                  (cfg['nodes'][1]['origin_port'], cfg['target_port'], cfg['delivery_port'], rs['port'])):
        for value in ports:
            port(value)
        if len(set(ports)) != len(ports):
            raise ValueError('conflicting_ports_on_same_host')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', cfg['remote_root']):
        raise ValueError('unsafe_remote_root')
    return cfg


def listening_inodes(port):
    result = set()
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].split(':')[1], 16) == port and fields[3] == '0A':
                result.add(fields[9])
    return result


def free_ports(ports):
    # Inspect kernel listeners, never connect to an unrelated HTTP service or
    # bind/steal a port during a read-only check. Launch admission rechecks bind.
    for port in ports:
        if listening_inodes(port):
            raise ValueError(f'PORT_OCCUPIED port={port}; choose another port, no process stopped')


def server_process(port):
    inodes = listening_inodes(port)
    matches = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if 'experiments.kv_ec.remote_rs_server' not in argv:
                continue
            if not any(fd.readlink().name in {f'socket:[{i}]' for i in inodes}
                       for fd in (proc / 'fd').iterdir()):
                continue
            parser = argparse.ArgumentParser(add_help=False)
            for key, cast in (('port', int), ('max-stripes', int), ('max-bytes', int),
                              ('ttl', float), ('rdma-port-base', int), ('rdma-timeout', float),
                              ('rdma-device', str), ('rdma-gid-index', int), ('rdma-directory', str),
                              ('rdma-port-num', int), ('host', str)):
                parser.add_argument('--' + key, type=cast, required=True)
            index = argv.index('experiments.kv_ec.remote_rs_server')
            args, _ = parser.parse_known_args(argv[index + 1:])
            if args.port != port:
                continue
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            matches.append(dict(pid=int(proc.name), start_ticks=int(stat[19]), argv=argv,
                                cwd=str((proc / 'cwd').resolve()), **vars(args)))
        except SystemExit as exc:
            raise ValueError('RS_SERVER_EXPLICIT_BUDGET_FLAGS_REQUIRED') from exc
        except (OSError, UnicodeError):
            continue
    if len(matches) != 1:
        raise ValueError('RS_LISTENER_IDENTITY_UNAVAILABLE: require readable /proc and explicit server budget flags')
    return matches[0]


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def check_ninja(node):
    """Inspect the actual Host.argv environment without building any kernels."""
    path = os.environ.get('PATH', '')
    conda_bin = str(Path(node['conda_prefix']) / 'bin')
    expected = list(dict.fromkeys([str(Path(node.get('cuda_home', node['conda_prefix'])) / 'bin'), conda_bin]))
    if path.split(os.pathsep)[:len(expected)] != expected:
        raise ValueError('CONDA_PATH_MISMATCH: expected CUDA/Conda PATH prefix ' + ':'.join(expected))
    ninja = shutil.which('ninja')
    if ninja is None:
        raise ValueError('NINJA_NOT_FOUND: PATH=' + path)
    try:
        result = subprocess.run([ninja, '--version'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f'NINJA_UNAVAILABLE: {ninja}: {exc}') from exc
    version = result.stdout.strip()
    if result.returncode or not version:
        raise ValueError(f'NINJA_VERSION_FAILED: {ninja}: exit={result.returncode}; {result.stderr.strip()}')
    return dict(path=path, executable=ninja, version=version)


def check_cuda_toolchain(node, output_root):
    """Compile device code and load a linked DSO; never call CUDA or use a GPU."""
    fixed, prefixes = cuda_environment(node)
    if not Path(fixed['CUDA_HOME']).is_dir():
        raise ValueError('CUDA_HOME_INVALID: Toolkit root directory not found: ' + fixed['CUDA_HOME'])
    for key, value in fixed.items():
        if os.environ.get(key) != value:
            raise ValueError(f'CUDA_ENV_MISMATCH: {key}, expected {value}')
    for key, value in prefixes.items():
        if os.environ.get(key, '').split(':')[:len(value.split(':'))] != value.split(':'):
            raise ValueError(f'CUDA_ENV_MISMATCH: {key}, expected prefix {value}')
    includes = prefixes['CPATH'].split(':')
    libraries = prefixes['LIBRARY_PATH'].split(':')
    for header in ('cuda.h', 'cuda_runtime.h', 'cuda_runtime_api.h'):
        if not any((Path(p) / header).is_file() for p in includes):
            raise ValueError(f'CUDA_HEADER_NOT_FOUND: {header}; searched {includes}')
    nvcc = fixed['NVCC']
    if not Path(nvcc).is_file() or not os.access(nvcc, os.X_OK):
        raise ValueError('CUDA_NVCC_INVALID: not executable: ' + nvcc)
    runtime = next((Path(p) / 'libcudart.so' for p in libraries if (Path(p) / 'libcudart.so').is_file()), None)
    if runtime is None:
        raise ValueError(f'CUDA_RUNTIME_NOT_FOUND: need linkable libcudart.so, not only a versioned runtime; searched {libraries}')

    def run(argv, error, timeout=120):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f'{error}: argv={argv!r}: {exc}') from exc
        if result.returncode:
            raise ValueError(f'{error}: argv={argv!r}; exit={result.returncode}; '
                             f'stdout={result.stdout[-4000:]}; stderr={result.stderr[-4000:]}')
        return result

    version_output = run([nvcc, '--version'], 'CUDA_NVCC_INVALID', timeout=10).stdout
    match = re.search(r'release (\d+)\.(\d+)', version_output)
    if not match:
        raise ValueError('CUDA_NVCC_INVALID: cannot parse version: ' + version_output)
    toolkit = tuple(map(int, match.groups()))
    import torch
    torch_cuda = torch.version.cuda
    match = re.match(r'^(\d+)\.(\d+)', torch_cuda or '')
    if not match:
        raise ValueError('CUDA_TORCH_VERSION_MISMATCH: PyTorch has no CUDA runtime')
    torch_pair = tuple(map(int, match.groups()))
    if toolkit[0] != torch_pair[0]:
        raise ValueError(f'CUDA_TORCH_VERSION_MISMATCH: nvcc={toolkit}, torch.version.cuda={torch_cuda}')
    runtime_match = re.search(r'\.so\.(\d+)', runtime.resolve().name)
    runtime_major = int(runtime_match.group(1)) if runtime_match else None
    if runtime_major is not None and runtime_major != toolkit[0]:
        raise ValueError(f'CUDA_RUNTIME_VERSION_MISMATCH: nvcc={toolkit}, runtime={runtime.resolve()}')
    warning = None if toolkit == torch_pair else (
        f'CUDA_MINOR_VERSION_MISMATCH: nvcc={toolkit}, torch.version.cuda={torch_cuda}; '
        'PyTorch permits a minor mismatch, but this probe does not certify every model JIT kernel')
    # No -L on the host link: exercise LIBRARY_PATH just as a JIT -lcudart
    # invocation does. A fresh Python loader exercises LD_LIBRARY_PATH without
    # accidentally reusing libcudart already imported by this process's torch.
    source = '''#include <cuda_runtime.h>
__global__ void stage_e_compile_only() {}
extern "C" {
int stage_e_header_version() { return CUDART_VERSION; }
cudaError_t (*stage_e_runtime_symbol)(int*) = &cudaRuntimeGetVersion;
}
'''
    loader = '''import ctypes,json,os,sys
lib = ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW)
lib.stage_e_header_version.restype = ctypes.c_int
paths = sorted({line.split(maxsplit=5)[-1].strip() for line in open('/proc/self/maps') if 'libcudart.so' in line})
print(json.dumps(dict(header_version=lib.stage_e_header_version(), runtime_paths=paths)))
'''
    # Use the node-local configured temporary filesystem.  The output root may
    # still be on /tmp (or a remote experiment volume), which is not suitable
    # when that filesystem is full.
    tmp_base = Path(node.get('tmpdir', '/tmp'))
    with tempfile.TemporaryDirectory(prefix='kv-ec-cuda-probe-', dir=tmp_base) as tmp:
        tmp = Path(tmp)
        src, obj, shared = tmp / 'probe.cu', tmp / 'probe.o', tmp / 'probe.so'
        src.write_text(source)
        compile_result = run([nvcc, '-c', '-Xcompiler=-fPIC', str(src), '-o', str(obj)], 'CUDA_JIT_COMPILE_FAILED')
        link_result = run(['c++', '-shared', str(obj), '-lcudart', '-o', str(shared)], 'CUDA_JIT_LINK_FAILED')
        loaded = run([sys.executable, '-c', loader, str(shared)], 'CUDA_JIT_LINK_FAILED', timeout=15)
        try:
            loaded = json.loads(loaded.stdout)
            header_pair = (loaded['header_version'] // 1000, loaded['header_version'] % 1000 // 10)
            resolved = {str(Path(p).resolve()) for p in loaded['runtime_paths']}
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError('CUDA_JIT_LINK_FAILED: invalid loader result') from exc
        if header_pair != toolkit:
            raise ValueError(f'CUDA_HEADER_VERSION_MISMATCH: nvcc={toolkit}, header={header_pair}')
        if resolved != {str(runtime.resolve())}:
            raise ValueError(f'CUDA_JIT_LINK_FAILED: runtime resolved to {resolved}, expected {runtime.resolve()}')
    return dict(status='PASS', environment={key: os.environ[key] for key in [*fixed, *prefixes]},
                nvcc=nvcc, nvcc_version='.'.join(map(str, toolkit)), nvcc_version_output=version_output.strip(),
                torch_version=torch.__version__, torch_cuda=torch_cuda, compatibility_warning=warning,
                libcudart=str(runtime), resolved_libcudart=str(runtime.resolve()),
                runtime_filename_major=runtime_major, probe_cuda_visible_devices='',
                header_version=loaded['header_version'], nvcc_compile=True, host_link=True, dynamic_load=True,
                gpu_context_created=False, cuda_api_called=False, temporary_files_removed=True,
                compiler_output=compile_result.stderr[-4000:], linker_output=link_result.stderr[-4000:])


def probe(cfg, node_index, output_root):
    """Executed with configured Conda Python on each host; returns only JSON."""
    from .debug_capture import source_identity
    from .gpu_restore_hook import validate_checkpoint
    from .verify_gpu_restore import read_input_ids
    from .launch_online import check_gpu_resources
    from types import SimpleNamespace
    node = cfg['nodes'][node_index]
    result = dict(node=node_index, status='FAIL', checks={})
    checks = result['checks']
    try:
        checks['inventory'] = identity()
        if Path(sys.prefix).resolve() != Path(node['conda_prefix']).resolve():
            raise ValueError('CONDA_PREFIX_MISMATCH: ' + sys.prefix)
        checks['conda_prefix'] = sys.prefix
        checks['tmpdir'] = check_tmpdir(node, cfg['min_disk_bytes'])
        checks['ninja'] = check_ninja(node)
        checks['cuda_toolchain'] = check_cuda_toolchain(node, output_root)
        addresses = command(['ip', '-j', '-4', 'addr', 'show'])
        if addresses.get('returncode') != 0:
            raise ValueError('LOCAL_IPV4_INSPECTION_FAILED: require iproute2')
        local_ips = {a['local'] for link in json.loads(addresses['stdout']) for a in link.get('addr_info', [])}
        checks['local_ipv4'] = sorted(local_ips)
        required_ip = cfg['return_host'] if node_index == 0 else cfg['rs_server']['host']
        if required_ip not in local_ips:
            raise ValueError('RDMA_RETURN_OR_RS_IP_NOT_LOCAL: ' + required_ip)
        ports = ([node['origin_port'], cfg['return_port']] if node_index == 0 else
                 [node['origin_port'], cfg['target_port'], cfg['delivery_port']])
        free_ports(ports)
        checks['free_ports'] = ports
        disk = shutil.disk_usage(nearest_existing(output_root))
        checks['disk_free_bytes'] = disk.free
        if disk.free < cfg['min_disk_bytes']:
            raise ValueError('INSUFFICIENT_DISK_SPACE')
        hca = Path('/sys/class/infiniband') / node['rdma_device'] / 'ports/1'
        state = (hca / 'state').read_text().strip()
        gid = (hca / f'gids/{node["gid_index"]}').read_text().strip()
        checks['hca'] = dict(state=state, gid=gid, device=node['rdma_device'], port=1,
                             gid_index=node['gid_index'])
        if not state.startswith('4:') or int(ipaddress.IPv6Address(gid)) == 0:
            raise ValueError('HCA_INACTIVE_OR_ZERO_GID')
        # Transport may build its native helper lazily, so check build inputs.
        from .multinode.rdma_transport import RDMATransport
        checks['native_build_available'] = RDMATransport.available()
        if not checks['native_build_available']:
            raise ValueError('RDMA_NATIVE_BUILD_REQUIREMENTS_MISSING: cc / libibverbs / verbs.h')
        model = Path(node['model'])
        if not (model / 'config.json').is_file() or not any(model.glob('*.safetensors')):
            raise ValueError('MODEL_PATH_INVALID: config.json/safetensors missing at ' + str(model))
        os.environ.pop('SGLANG_KV_EC_PREPARED_IDENTITY', None)
        checks['model_identity'] = source_identity(node['model'])
        template = Path(node['template'])
        meta = validate_checkpoint(template.read_bytes())
        if (meta['node_id'] != node_index or len(meta['ranks']) != 2 or
                any(r['token_start'] != 0 or r['token_count'] != 256 for r in meta['ranks'])):
            raise ValueError('TEMPLATE_NODE_TP_OR_TOKEN_RANGE_MISMATCH')
        read_input_ids(node['input_ids'], meta)
        checks['template_bytes'] = template.stat().st_size
        if template.stat().st_size + 1024**2 > cfg['max_node_bytes']:
            raise ValueError('NODE_RESERVATION_TOO_SMALL: template + 1MiB envelope margin')
        if meta['ranks'][0]['model_sha256'] != checks['model_identity']['model_sha256']:
            raise ValueError('TEMPLATE_MODEL_IDENTITY_MISMATCH')
        from .perf_benchmark import digest
        checks['template_sha256'] = digest(template)
        checks['input_sha256'] = digest(node['input_ids'])
        checks['gpu_admission'] = check_gpu_resources(SimpleNamespace(
            gpus=node['gpus'], model=node['model'], tp=2, allow_shared_gpu_debug=False,
            mem_fraction_static=cfg.get('mem_fraction_static', .6)))
        if node_index == 1:
            process = server_process(cfg['rs_server']['port'])
            if Path(process['cwd']).resolve() != Path(node['repo']).resolve():
                raise ValueError('RS_SERVER_REPO_MISMATCH')
            if (process['rdma_device'] != node['rdma_device'] or
                    process['rdma_gid_index'] != node['gid_index'] or process['rdma_port_num'] != 1):
                raise ValueError('RS_SERVER_HCA_MISMATCH')
            if process['host'] not in ('0.0.0.0', cfg['rs_server']['host']):
                raise ValueError('RS_SERVER_BIND_ADDRESS_MISMATCH')
            base, count = process['rdma_port_base'], max(4, process['max_stripes'] * 4)
            if base + count - 1 > 65535 or any(base <= p < base + count for p in ports + [cfg['rs_server']['port']]):
                raise ValueError('RS_NATIVE_PORT_RANGE_CONFLICT')
            free_ports(range(base, base + count))
            checks['rs_process'] = process
            checks['rs_disk_free_bytes'] = shutil.disk_usage(nearest_existing(process['rdma_directory'])).free
        result['status'] = 'PASS'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def capacity(cfg, reports, health, metrics, repeat):
    """Store reserves 3*max(node length) per live stripe, not resident parity.

    Conservatively reserve all historical slots, even expired, and all bytes
    per slot because the old metrics RPC does not expose reserved_bytes.
    """
    p = reports[1]['checks']['rs_process']
    if health.get('state') != 'READY' or not health.get('server_epoch') or health['server_epoch'] != metrics.get('server_epoch'):
        raise ValueError('RS_SERVER_NOT_READY_OR_RESTARTED')
    used = metrics['stripes']
    startup_git = metrics.get('environment_B', {}).get('git', {})
    if (startup_git.get('returncode') != 0 or startup_git.get('stdout', '').strip() !=
            reports[1]['checks']['inventory']['commit_sha']):
        raise ValueError('RS_SERVER_STARTED_WITH_DIFFERENT_COMMIT')
    reserve = 3 * cfg['max_node_bytes']
    if p['max_stripes'] - used < repeat:
        raise ValueError(f'RS_STRIPE_BUDGET: used={used} max={p["max_stripes"]} requested={repeat}; expired tombstones retain slots; operator must start a new dedicated server')
    # Old stripes may have reserved more than their current resident parity;
    # require a conservative upper bound from protocol MAX_NODE_BYTES instead.
    from .remote_rs_protocol import MAX_NODE_BYTES
    needed = used * 3 * MAX_NODE_BYTES + repeat * reserve
    if needed > p['max_bytes']:
        raise ValueError(f'RS_BYTE_BUDGET: conservative_reserved={needed} max={p["max_bytes"]}; use empty dedicated server or larger capacity')
    if p['ttl'] < cfg['protection_timeout']:
        raise ValueError('RS_TTL_TOO_SHORT: must cover protection through final live check')
    if reports[1]['checks']['rs_disk_free_bytes'] < cfg['min_disk_bytes']:
        raise ValueError('RS_INSUFFICIENT_DISK_SPACE')
    return dict(server_epoch=health['server_epoch'], used_slots=used,
                requested_slots=repeat, reserved_upper_bound_bytes=needed,
                ttl_s=p['ttl'], max_stripes=p['max_stripes'], max_bytes=p['max_bytes'])


def preflight(cfg, output_root, repeat, expected_epoch=None):
    from .perf_stage_e import Host, parallel
    from .remote_rs_client import RemoteRSClient
    validate_config(cfg)
    results = [None, None]
    encoded = base64.urlsafe_b64encode(json.dumps(cfg).encode()).decode()
    def check(i):
        root = str(output_root) if i == 0 else cfg['remote_root']
        h = Host(cfg['nodes'][i], root, 'preflight', i)
        cmd = h.argv(MODULE, 'probe', '--config-base64', encoded, '--node', i, '--output-root', root)
        completed = subprocess.run(cmd, cwd=cfg['nodes'][0]['repo'] if i == 0 else None,
                                   capture_output=True, text=True, timeout=900)
        lines = [line for line in completed.stdout.splitlines() if line.startswith('KV_EC_PREFLIGHT ')]
        if completed.returncode or not lines:
            raise RuntimeError(f'PREFLIGHT_NODE_{i}_SSH_OR_PROBE_FAILED: {completed.stderr[-4000:]} {completed.stdout[-4000:]}')
        results[i] = json.loads(lines[-1].split(' ', 1)[1])
    # Preserve both node errors rather than losing the second SSH result.
    try:
        parallel(lambda: check(0), lambda: check(1))
        if any(r['status'] != 'PASS' for r in results):
            raise ValueError('PREFLIGHT_NODE_FAILED: ' + json.dumps([r.get('error') for r in results]))
        a, b = [r['checks']['inventory'] for r in results]
        if a['host'] == b['host'] or a['boot_id'] == b['boot_id']:
            raise ValueError('TWO_PHYSICAL_HOSTS_REQUIRED')
        for key in ('commit_sha', 'tracked_diff_sha256', 'benchmark_files'):
            if a[key] != b[key]:
                raise ValueError('SOURCE_IDENTITY_MISMATCH: ' + key)
        if any(r['checks']['model_identity']['model_sha256'] != results[0]['checks']['model_identity']['model_sha256'] for r in results):
            raise ValueError('MODEL_IDENTITY_MISMATCH')
        client = RemoteRSClient(cfg['rs_server']['host'], cfg['rs_server']['port'], timeout=10)
        health, payload = client.call('health')
        metrics, payload2 = client.call('get_metrics')
        if payload or payload2:
            raise ValueError('PREFLIGHT_TCP_PAYLOAD_FORBIDDEN')
        budget = capacity(cfg, results, health, metrics, repeat)
        disk_required = max(cfg['min_disk_bytes'], repeat * cfg['max_node_bytes'] * 20)
        # Includes retained captures, references, sealed/recovered copies and
        # validation artifacts, with a conservative 20x per-node margin.
        if any(r['checks']['disk_free_bytes'] < disk_required for r in results):
            raise ValueError(f'BATCH_DISK_BUDGET: require {disk_required} free bytes per node')
        if expected_epoch and budget['server_epoch'] != expected_epoch:
            raise ValueError('RS_SERVER_EPOCH_CHANGED_DURING_BATCH')
        return dict(status='PASS', nodes=results, rs=budget, metrics=metrics,
                    creates_protection=False, read_only=False, read_only_services=True, temporary_cuda_probe=True)
    except Exception as exc:
        return dict(status='FAIL', nodes=results, error=f'{type(exc).__name__}: {exc}',
                    creates_protection=False, read_only=False, read_only_services=True, temporary_cuda_probe=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['probe', 'cuda-probe'])
    p.add_argument('--config-base64', required=True)
    p.add_argument('--node', type=int, choices=[0, 1], required=True)
    p.add_argument('--output-root', required=True)
    args = p.parse_args()
    cfg = validate_config(json.loads(base64.urlsafe_b64decode(args.config_base64)))
    if args.command == 'cuda-probe':
        try:
            result = dict(status='PASS', node=args.node, checks=dict(
                ninja=check_ninja(cfg['nodes'][args.node]),
                cuda_toolchain=check_cuda_toolchain(cfg['nodes'][args.node], args.output_root)))
        except Exception as exc:
            result = dict(status='FAIL', node=args.node, error=f'{type(exc).__name__}: {exc}')
    else:
        result = probe(cfg, args.node, args.output_root)
    print('KV_EC_PREFLIGHT ' + json.dumps(result), flush=True)
    if args.command == 'cuda-probe' and result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
