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


def normalized_bf16_settings(settings):
    """Canonical identity; absent historical policy means only torch_amp."""
    from deepopt._bf16_policy import resolve_bf16_policy

    result = deepcopy(settings)
    policy = resolve_bf16_policy(result.get("bf16_policy"))
    result["bf16_policy"] = policy.as_dict()
    from deepopt._activation_memory import resolve_activation_memory

    result["activation_memory"] = resolve_activation_memory(result.get("activation_memory")).as_dict()
    from deepopt._packed_offload import resolve_packed_offload

    # Historical checkpoints predate transport configuration. Their absent
    # policy is exactly disabled; enabled policies remain part of identity.
    result["packed_activation_offload"] = resolve_packed_offload(result.get("packed_activation_offload")).as_dict()
    return result


def model_bf16_policy(model, config):
    """Owned preparation plans outrank ambient scope for exact resume."""
    from deepopt._bf16_policy import resolve_bf16_policy

    policies = []
    for module in model.modules():
        owned = getattr(module, "_deepopt_bf16_policy", None)
        if owned is not None:
            policies.append(resolve_bf16_policy(owned))
        plan = getattr(module, "plan", None)
        if isinstance(plan, dict) and "bf16_policy" in plan:
            policies.append(resolve_bf16_policy(plan["bf16_policy"]))
    if policies and any(policy != policies[0] for policy in policies):
        raise ValueError("Prepared model has inconsistent BF16 policies")
    return policies[0] if policies else resolve_bf16_policy(config.get("runtime", {}).get("bf16_policy"))


def model_activation_memory(model, config):
    """Resolve exact resume identity from actual prepared invocation owners."""
    from deepopt._activation_memory import resolve_activation_memory

    plans = []
    for module in model.modules():
        owned = getattr(module, "_deepopt_activation_memory_plan", None)
        if owned is not None:
            plans.append(resolve_activation_memory(owned))
        operation = getattr(module, "plan", None)
        if isinstance(operation, dict) and "activation_memory" in operation:
            plans.append(resolve_activation_memory(operation["activation_memory"]))
    nondefault = [plan for plan in plans if not plan.is_default]
    if nondefault and any(plan != nondefault[0] for plan in nondefault):
        raise ValueError("Prepared model has inconsistent activation memory recipes")
    return (nondefault[0] if nondefault else plans[0] if plans else
            resolve_activation_memory(config.get("runtime", {}).get("activation_memory")))


def compatibility_config(config):
    result = deepcopy(config)
    if "training" in result:
        result["training"] = {key: value for key, value in result["training"].items()
                              if key not in ("epochs", "batch_size", "checkpoint_every_steps")}
        if not result["training"].get("ema", {}).get("enabled", False):
            result["training"].pop("ema", None)
    result.setdefault("runtime", {})
    if "runtime" in result:
        result["runtime"] = {key: value for key, value in normalized_bf16_settings(result["runtime"]).items()
                             if key not in ("checks", "execution_mode", "diagnostics", "mode_schedule", "non_blocking",
                                            "fill_uninitialized_memory")}
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
