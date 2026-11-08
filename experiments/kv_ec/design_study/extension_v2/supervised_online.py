"""Parent-executed single-lifetime launcher using the existing perf supervisor.

This module is never imported to launch anything. Its explicit CLI starts only
new owned manager/serving processes, leases one fixed port, and asks the existing
exact-identity supervisor to clean up those processes. Never starts/restarts RS,
uses SSH, rotates occupied ports, or signals an existing process.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from ..admission import PROTECTED_PORTS
from ..online_compact import MODES, require_smoke, run, validate_plan
from ...perf_ports import PortLease
from ...perf_stage_e import require_service_alive, wait_health
from .evidence import save


def execute(args):
    value=json.loads(args.plan.read_text())
    validate_plan(value)
    proof = require_smoke(value,args.mode,getattr(args,"smoke",None))
    if proof is not None and proof.get("runtime",{}).get("comparison",{}).get("physical_gpus") != args.gpus:
        raise ValueError("formal GPU selection differs from accepted smoke")
    if args.mode != "noec" and not args.reference:
        raise ValueError("same-plan NoEC reference required before starting protection service")
    if args.port in PROTECTED_PORTS or not 1024<=args.port<=65535:
        raise ValueError("dedicated fixed port required")
    if (args.mode in ("rs-only","xor-rs")) != bool(args.rs_config):
        raise ValueError("RS config required only for global modes")
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    run_id="compact-"+uuid.uuid4().hex
    os.environ["KV_EC_PERF_RUN_ID"]=run_id
    os.environ["KV_EC_PERF_DIR"]=str(root/"events")
    save(root/"trial-owner.json",dict(run_id=run_id))
    (root/"lease").touch(exist_ok=False)
    socket_path=Path("/tmp")/(run_id+".sock")
    if socket_path.exists():
        raise FileExistsError("fresh private socket path already exists")
    heartbeat_stop=threading.Event()
    def heartbeat():
        while not heartbeat_stop.wait(2):
            (root/"lease").touch()
    pulse=threading.Thread(target=heartbeat,daemon=True)
    pulse.start()
    services=[]
    handles=[]
    report=None
    def launch(name,argv):
        log=(root/f"supervisor-{name}.log").open("x")
        handles.append(log)
        command=[sys.executable,"-m","experiments.kv_ec.perf_stage_e","service","--root",str(root/name),"--",*argv]
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        services.append((name,child))
        return child
    try:
        with PortLease([args.port],root,run_id):
            try:
                if args.mode!="noec":
                    manager=[sys.executable,"-m","experiments.kv_ec.online_chunk_manager","--socket",str(socket_path),
                        "--output",str(root/"checkpoints"),"--run-id",run_id,"--protection-mode",args.mode,
                        "--max-pending",str(args.max_pending),"--max-replica-bytes",str(args.max_local_bytes)]
                    if args.rs_config:
                        manager += ["--global-rs-config",str(args.rs_config.resolve())]
                    launch("target",manager)
                    end=time.monotonic()+30
                    while not (socket_path.exists() and require_service_alive(root/"target")):
                        if time.monotonic()>=end:
                            raise TimeoutError("owned manager not ready")
                        time.sleep(.1)
                command=[sys.executable,"-m","experiments.kv_ec.launch_online","--model",args.model,"--gpus",args.gpus,
                    "--tp","2","--socket",str(socket_path),"--run-id",run_id,"--port",str(args.port),
                    "--prefill-chunk-size","256","--capacity",str(args.capacity),
                    "--mem-fraction-static",str(args.mem_fraction_static),"--gpu-admission-mode","shared"]
                command += ["--baseline"] if args.mode=="noec" else ["--protection-mode",args.mode]
                if args.mode != "noec":
                    command += ["--manager-pid", str(json.loads((root/"target/owned-process.json").read_text())["pid"])]
                launch("origin",command)
                wait_health(f"http://127.0.0.1:{args.port}",root/"origin",timeout=args.startup_timeout)
                spec=dict(mode=args.mode,pid=json.loads((root/"origin/owned-process.json").read_text())["pid"],
                    owned_process=str(root/"origin/owned-process.json"),
                    url=f"http://127.0.0.1:{args.port}",release_after_validation=True)
                if args.mode!="noec":
                    spec.update(manager_pid=json.loads((root/"target/owned-process.json").read_text())["pid"],
                        manager_output=str(root/"checkpoints"),manager_socket=str(socket_path))
                save(root/"service.json",spec)
                report=run(spec,value,root/"measurements",reference=args.reference,total_timeout=args.total_timeout,
                    smoke=getattr(args,"smoke",None))
            finally:
                # Stop markers are inside brand-new owned service directories;
                # only perf_stage_e performs identity/pidfd-checked signalling.
                errors=[]
                for name,child in reversed(services):
                    (root/name).mkdir(exist_ok=True)
                    (root/name/"stop").touch(exist_ok=True)
                for name,child in reversed(services):
                    try:
                        child.wait(timeout=60)
                        if child.returncode:
                            errors.append(dict(service=name,returncode=child.returncode))
                    except subprocess.TimeoutExpired:
                        errors.append(dict(service=name,error="supervisor timeout; no unverified kill attempted"))
                save(root/"supervisor-exit.json",dict(status="FAIL" if errors else "PASS",errors=errors))
                if errors:
                    raise RuntimeError("owned cleanup failed; see supervisor-exit.json")
    except BaseException as exc:
        save(root/"failure.json",dict(error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        heartbeat_stop.set()
        pulse.join(3)
        for handle in handles:
            handle.close()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--gpus",required=True)
    p.add_argument("--port",type=int,default=32961)
    p.add_argument("--reference",type=Path)
    p.add_argument("--smoke",type=Path,help="same-mode smoke raw.json required for five-round plan")
    p.add_argument("--rs-config",type=Path)
    p.add_argument("--mem-fraction-static",type=float,default=.6)
    p.add_argument("--capacity",type=int,default=4)
    p.add_argument("--max-pending",type=int,default=32)
    p.add_argument("--max-local-bytes",type=int,default=4*1024**3)
    p.add_argument("--startup-timeout",type=float,default=900)
    p.add_argument("--total-timeout",type=float,default=7200)
    args=p.parse_args()
    result=execute(args)
    print(json.dumps(dict(status=result["status"],completed=result["completed"],requests=result["requests"])))
    raise SystemExit(0 if result["status"]=="PASS" else 1)


if __name__=="__main__":
    main()
