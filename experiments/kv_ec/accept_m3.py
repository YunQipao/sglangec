"""Ordered M3 GPU correctness matrix; stop at the first failed prerequisite/run."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ORDERED_CASES = (
    (0, 256),  # A: preserve M2
    (1, 256),  # B: dynamic target
    (0, 768),  # C: all three chunks
    (0, 512), (1, 512), (1, 768),
    (2, 256), (2, 512), (2, 768),
    (3, 256), (3, 512), (3, 768),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--shadow-evidence", required=True, type=Path)
    parser.add_argument("--model", default="/media/sda1/iic_data/luotao/models/Qwen3-8B")
    parser.add_argument("--baseline-port", type=int, default=30359)
    parser.add_argument("--pause-port", type=int, default=30360)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--pause-timeout", type=float, default=60)
    parser.add_argument("--smoke", action="store_true", help="only run A after the full CPU regression")
    parser.add_argument("--output", type=Path, help="exclusive new matrix evidence directory")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="kv-ec-m3-matrix-"))
    cases = ORDERED_CASES[:1] if args.smoke else ORDERED_CASES
    verdict = dict(M3_PASS=False, state="CPU_PENDING", run=str(root),
                   cases=[dict(lost_rank=rank, recover_tokens=tokens, state="NOT_RUN")
                          for rank, tokens in cases])
    print(f"MATRIX={root}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        with (root / "cpu-regression.log").open("x") as log:
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "experiments/kv_ec",
                 "-p", "test_*.py", "-v"], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        verdict["cpu_returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError("CPU regression failed; no GPU experiments launched")
        print("CPU_PASS=true", flush=True)
        verdict["state"] = "GPU_PENDING"
        for index, ((rank, tokens), case) in enumerate(zip(cases, verdict["cases"])):
            run = root / f"rank{rank}-tokens{tokens}"
            command = [sys.executable, "-u", "-m", "experiments.kv_ec.accept_pause",
                       "--online-restore", "--lost-rank", str(rank), "--recover-tokens", str(tokens),
                       "--gpus", args.gpus, "--shadow-evidence", str(args.shadow_evidence.resolve()),
                       "--model", args.model, "--baseline-port", str(args.baseline_port),
                       "--pause-port", str(args.pause_port),
                       "--startup-timeout", str(args.startup_timeout),
                       "--pause-timeout", str(args.pause_timeout), "--output", str(run)]
            case.update(state="RUNNING", run=str(run), command=command)
            print(f"CASE={index + 1}/{len(cases)} lost_rank={rank} recover_tokens={tokens}", flush=True)
            with (root / f"rank{rank}-tokens{tokens}.log").open("x") as log:
                result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            case["returncode"] = result.returncode
            acceptance = json.loads((run / "acceptance.json").read_text()) if (run / "acceptance.json").exists() else {}
            if result.returncode or acceptance.get("M3_PASS") is not True:
                case.update(state="FAIL", reason=acceptance.get("reason", "acceptance subprocess failed"))
                raise RuntimeError(f"rank{rank}/tokens{tokens}: {case['reason']}; remaining cases NOT_RUN")
            case.update(state="PASS", chunks=acceptance["chunks"])
            print(f"CASE_PASS=true lost_rank={rank} recover_tokens={tokens}", flush=True)
        verdict.update(state="PASS", M3_PASS=True)
    except (Exception, KeyboardInterrupt) as exc:
        verdict.update(state="FAIL", reason=f"{type(exc).__name__}: {exc}")
        for case in verdict["cases"]:
            if case["state"] == "RUNNING":
                case.update(state="FAIL", reason=verdict["reason"])
    finally:
        with (root / "matrix.json").open("x") as stream:
            json.dump(verdict, stream, indent=2)
    print(f"M3_PASS={str(verdict['M3_PASS']).lower()}\nEvidence: {root}", flush=True)
    return 0 if verdict["M3_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
