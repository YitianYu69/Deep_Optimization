"""Explicit compatibility identity plus best-effort source provenance."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess


def validate_identity(value):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("experiment_identity must be a dictionary with string keys")
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        # JSON-compatible primitives only; no arbitrary Python objects/tensors.
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("experiment_identity must contain finite JSON-compatible values") from exc
    return result


def compatibility_config(config):
    result = deepcopy(config)
    if "training" in result:
        result["training"] = {key: value for key, value in result["training"].items()
                              if key not in ("epochs", "batch_size", "checkpoint_every_steps")}
    if "runtime" in result:
        result["runtime"] = {key: value for key, value in result["runtime"].items()
                             if key not in ("checks", "execution_mode", "diagnostics", "mode_schedule", "non_blocking")}
    if not result.get("experiment_identity"):
        result.pop("experiment_identity", None)
    return result


@lru_cache(maxsize=8)
def git_identity(directory):
    result = {"git_commit": None, "working_tree_dirty": None}
    try:
        def git(*arguments):
            return subprocess.run(["git", *arguments], cwd=directory, capture_output=True, text=True,
                                  timeout=2, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}).stdout.strip()
        result["git_commit"] = git("rev-parse", "HEAD")
        result["working_tree_dirty"] = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def snapshot_identity(config, required):
    encoded = json.dumps(compatibility_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"required": validate_identity(required),
            "observed": {**git_identity(str(Path.cwd())),
                         "config_fingerprint": hashlib.sha256(encoded.encode()).hexdigest()}}
