"""Reproducible CPU/Mock validation runner with exclusive JSON/CSV evidence.

Kernel TCP integration remains a separate reported capability, never promoted
to physical LAN or real KV acceptance. Run on A with the existing Torch env.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from .rs_experiment import environment, save_report


def run_command(argv, *, stdin=None):
    start = time.perf_counter()
    p = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=600)
    combined = p.stdout + p.stderr
    count = re.search(r"Ran (\d+) tests?", combined)
    skips = re.search(r"skipped=(\d+)", combined)
    return dict(
        argv=argv,
        returncode=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
        elapsed_s=time.perf_counter() - start,
        state="PASS" if p.returncode == 0 else "FAIL",
        tests=int(count[1]) if count else None,
        skipped=int(skips[1]) if skips else 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new validation evidence directory"
    )
    parser.add_argument("--full-regression", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        environment=environment(),
        stages={},
        verification={
            "GPU": "NOT VERIFIED",
            "physical_TCP_LAN": "NOT VERIFIED",
            "real_KV_end_to_end": "NOT VERIFIED",
            "online_remote_RS": "NOT IMPLEMENTED",
            "real_serving_impact": "NOT VERIFIED",
        },
    )
    groups = {
        "M1": [
            "experiments.kv_ec.test_dual_node_launch",
            "experiments.kv_ec.test_parity_manager",
        ],
        "M2": ["experiments.kv_ec.test_remote_rs"],
        "M3_M4": ["experiments.kv_ec.test_rs_experiments"],
    }
    for phase, tests in groups.items():
        result = run_command([sys.executable, "-m", "unittest", *tests, "-v"])
        report["stages"][phase] = result
        print(
            json.dumps(
                {
                    "phase": phase,
                    **{k: result[k] for k in ("state", "tests", "skipped")},
                }
            ),
            flush=True,
        )
    if args.full_regression:
        report["stages"]["full_regression"] = run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "experiments/kv_ec",
                "-p",
                "test_*.py",
                "-v",
            ]
        )
    report["stages"]["compile"] = run_command(
        [sys.executable, "-m", "compileall", "-q", "experiments/kv_ec"]
    )
    modules = (
        "launch_debug",
        "dual_node_check",
        "node_block_export",
        "remote_rs_server",
        "remote_rs_client",
        "dual_node_rs",
        "serving_impact",
    )
    report["stages"]["cli_help"] = {
        m: run_command([sys.executable, "-m", f"experiments.kv_ec.{m}", "--help"])
        for m in modules
    }
    doc = Path(__file__).with_name("DUAL_NODE_REMOTE_RS.md").read_text()
    blocks = re.findall(r"```bash\n(.*?)```", doc, re.S)
    report["stages"]["doc_bash_syntax"] = [
        run_command(["bash", "-n"], stdin=b) for b in blocks
    ]
    # B imports must work WITHOUT Torch: use sys.modules check, no mocking CUDA.
    report["stages"]["B_no_torch_import"] = run_command(
        [
            sys.executable,
            "-c",
            "import sys; import experiments.kv_ec.remote_rs_server; assert 'torch' not in sys.modules; print('PASS')",
        ]
    )

    def good(item):
        if isinstance(item, dict) and "returncode" in item:
            return item["returncode"] == 0
        return all(good(v) for v in (item.values() if isinstance(item, dict) else item))

    report["source_sha256_after"] = environment()["implementation_sha256"]
    report["source_changed_during_validation"] = (
        report["source_sha256_after"] != report["environment"]["implementation_sha256"]
    )
    report["state"] = "PASS" if good(report["stages"]) and not report["source_changed_during_validation"] else "FAIL"
    report["note"] = (
        "CPU/Mock only; inspect skipped tests. No random fixture is real KV acceptance."
    )
    save_report(args.output / "validation", report)
    print(json.dumps(dict(state=report["state"], output=str(args.output))), flush=True)
    if report["state"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
