"""Read-only provenance checks shared by study admission and offline audits."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[2]
CONFIG_KEYS = (
    "lost_rank", "recover_tokens", "model_identifier", "model_sha256",
    "source_commit", "core_diff_sha256", "gpu_uuids", "tp", "dtype", "layout",
    "page_size", "overlap", "prefix_cache", "input_tokens", "output_tokens",
    "input_ids_sha256", "sampling", "torch_version", "cuda_version",
    "experiment_source_sha256", "parity_test_oracle",
)
MINIMUM_SOURCES = (
    "scheduler_shadow.py", "scheduler_pause.py", "scheduler_retry.py",
    "pause_control.py", "rank_snapshot.py", "parity_manager.py",
    "checkpoint_protocol.py", "m3_recovery.py", "m3_performance.py",
    "accept_pause.py",
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class Inconclusive(ValueError):
    """Evidence is unavailable, rather than demonstrated incorrect."""


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def runtime_sources(directory=None):
    # Conservative superset of the experiment import/subprocess dependency
    # closure, including launch, control, validation, codecs and audit tools.
    return tuple(sorted(p.name for p in (directory or Path(__file__).parent).glob("*.py")
                        if not p.name.startswith("test_")))


def git(*args, repo=REPO):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise Inconclusive(f"git_evidence_unavailable:{args}") from exc


def historical_sources(config, recorded=None, *, repo=REPO):
    commit = config.get("source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Inconclusive("missing_or_invalid_source_commit")
    if config.get("core_diff_sha256") != EMPTY_SHA256:
        raise Inconclusive("historical_dirty_inference_patch_unavailable")
    paths = git("ls-tree", "-r", "--name-only", commit, "--", "experiments/kv_ec",
                repo=repo).decode().splitlines()
    paths = [p for p in paths if p.endswith(".py")
             and not Path(p).name.startswith("test_")]
    names = {Path(p).name for p in paths}
    if not set(MINIMUM_SOURCES) <= names:
        raise Inconclusive("historical_runtime_closure_incomplete")
    saved = config.get("experiment_source_sha256")
    if not isinstance(saved, dict) or not saved:
        raise Inconclusive("missing_recorded_source_anchors")
    if recorded is not None and recorded != saved:
        raise ValueError("manifest_and_runtime_source_hashes_disagree")
    result = {}
    for path in paths:
        name = Path(path).name
        raw = git("show", f"{commit}:{path}", repo=repo)
        digest = hashlib.sha256(raw).hexdigest()
        if name in saved and saved[name] != digest:
            raise ValueError(f"recorded_source_differs_from_commit:{name}")
        result[name] = dict(
            path=path, sha256=digest,
            git_blob=git("rev-parse", f"{commit}:{path}", repo=repo).decode().strip(),
            evidence="recorded_hash_and_git_blob" if name in saved else "git_blob_supplement")
    if not set(saved) <= names:
        raise Inconclusive("recorded_source_missing_from_commit_tree")
    return dict(
        state="PASS", source_commit=commit, files=result,
        inference_tree=git("rev-parse", f"{commit}:python", repo=repo).decode().strip(),
        historical_limit="Git supplements missing historical hashes; it cannot attest "
                         "unrecorded historical worktree modifications or external packages.")


def evidence_path(path):
    parts = Path(path).parts
    return (len(parts) > 1 and parts[0].startswith(("kv-ec-", "kv_ec_"))
            and Path(path).suffix in (".json", ".log"))


def compare_config_fields(before, after):
    excluded = {"source_commit", "experiment_source_sha256"}
    for key in sorted((set(before) | set(after) | set(CONFIG_KEYS)) - excluded):
        if key not in before or key not in after:
            raise Inconclusive(f"missing_configuration:{key}")
        if json.dumps(before[key], sort_keys=True) != json.dumps(after[key], sort_keys=True):
            raise ValueError(f"incompatible_config:{key}")


def compatible_config(before, after, *, before_recorded=None, after_recorded=None,
                      before_parameters=None, after_parameters=None, repo=REPO):
    """Same function for preflight and final smoke admission; no commit rewrite."""
    compare_config_fields(before, after)
    if before_parameters is not None or after_parameters is not None:
        if before_parameters is None or after_parameters is None:
            raise Inconclusive("missing_launch_parameters")
        if json.dumps(before_parameters, sort_keys=True) != json.dumps(after_parameters, sort_keys=True):
            raise ValueError("launch_parameters_changed_since_smoke")
    left = historical_sources(before, before_recorded, repo=repo)
    right = historical_sources(after, after_recorded, repo=repo)
    left_hashes = {k: v["sha256"] for k, v in left["files"].items()}
    right_hashes = {k: v["sha256"] for k, v in right["files"].items()}
    if left_hashes != right_hashes or left["inference_tree"] != right["inference_tree"]:
        raise ValueError("runtime_source_or_dependency_changed_since_smoke")
    changed = git("diff", "--name-status", "--no-renames",
                  before["source_commit"], after["source_commit"], repo=repo).decode().splitlines()
    if any(not line.startswith("A\t") or not evidence_path(line[2:]) for line in changed):
        raise ValueError("non_evidence_commit_difference")
    return dict(state="PASS", before=left, after=right,
                commit_change="evidence_only_additions" if changed else "identical_tree",
                changed_files=changed, strict_configuration_equal=True,
                launch_parameters=after_parameters,
                launch_parameter_evidence=(
                    "recorded_equal" if after_parameters is not None else
                    "legacy_launch_flags_not_recorded; audit_effective_runtime_parameters"))


def check_live_sources(*, repo=REPO):
    """Reject tracked, staged and untracked source changes before GPU launch."""
    changes = git("status", "--porcelain", "--untracked-files=all", repo=repo).decode().splitlines()
    changes = [line for line in changes if not evidence_path(line[3:])]
    if changes:
        raise ValueError("dirty_or_untracked_runtime_sources:" + repr(changes))
    return git("rev-parse", "HEAD", repo=repo).decode().strip()
