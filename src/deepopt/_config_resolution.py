"""Resolve immutable-in-use configuration snapshots without initializing Torch."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from ._bf16_policy import resolve_bf16_policy
from ._packed_offload import resolve_packed_offload
from ._memory_units import normalize_packed_offload_units

from .config import (
    ConfigError, ExperimentConfig, TrainingConfig, _read_config,
    config_from_dict, get_config, training_config_from_dict,
)


_UNSET = object()
_scope = ContextVar("deepopt_config_scope", default=_UNSET)
_ROOT_FIELDS = {"name", "seed", "training", "runtime", "application",
                "experiment_identity", "schedules", "loader", "logging"}
_TRAINING_REQUIRED = ("epochs", "batch_size", "learning_rate")


class ResolvedConfig(dict):
    """An owned snapshot marker: passing it never consults current defaults."""


def _merge(base, values):
    result = deepcopy(dict(base))
    for key, value in values.items():
        if value is _UNSET:
            continue
        if isinstance(value, Mapping):
            previous = result.get(key)
            if key == "packed_activation_offload":
                # Normalize the explicit table before inheriting canonical byte
                # defaults. Conflicts in one table still fail; GB overrides of
                # an inherited byte budget replace that quantity as intended.
                try:
                    normalized = normalize_packed_offload_units(value)
                except ValueError as error:
                    raise ConfigError(f"packed_activation_offload: {error}") from error
                if ("host_pinned_gb" not in value and "host_pinned_bytes" not in value
                        and isinstance(previous, Mapping)
                        and type(previous.get("host_pinned_bytes")) is int
                        and previous["host_pinned_bytes"] > 0):
                    # A default for a newly enabled GB route cannot replace an
                    # inherited, explicitly configured positive host budget.
                    normalized.pop("host_pinned_bytes", None)
                value = normalized
            if (key == "kernels" and value.get("backend") in ("inherit", "triton", "tilelang")
                    and isinstance(previous, Mapping) and previous.get("backend") == "mixed"):
                # An explicit mode switch replaces inherited Mixed routes.
                # Partial changes within Mixed retain the normal deep merge.
                previous = {}
            result[key] = _merge(previous if isinstance(previous, Mapping) else {}, value)
        else:
            result[key] = deepcopy(value)
    return result


def _source_dict(source):
    if source is None:
        return {}
    if isinstance(source, ExperimentConfig):
        return asdict(source)
    if isinstance(source, Mapping):
        return _merge({}, source)
    if isinstance(source, (str, Path)):
        return asdict(_read_config(source))
    raise ConfigError("Configuration must be an ExperimentConfig, mapping, TOML path, or builtin:NAME")


def _validate_snapshot(raw, require_complete):
    if not raw and not require_complete:
        return ResolvedConfig()
    training = raw.get("training", {})
    missing = [key for key in ("name", "seed") if key not in raw]
    missing.extend(f"training.{key}" for key in _TRAINING_REQUIRED
                   if not isinstance(training, Mapping) or key not in training)
    if require_complete and missing:
        raise ConfigError("A complete experiment configuration is required; missing fields: " + ", ".join(missing))

    # Reuse the authoritative schema for optional defaults and cross-field
    # validation, then remove the scaffold's required values from the snapshot.
    scaffold = {"name": "_partial", "seed": 0,
                "training": {"epochs": 1, "batch_size": 1, "learning_rate": 1.0}}
    validation_raw = raw
    runtime = raw.get("runtime", {})
    # Partial policy snapshots need not choose a device or AMP dtype yet.  The
    # autocast/execution consumer checks the effective dtype before entering.
    # Explicit fields still use the complete schema's cross-field validation.
    try:
        policy = resolve_bf16_policy(runtime.get("bf16_policy")) if isinstance(runtime, Mapping) else None
    except ValueError as exc:
        raise ConfigError(f"runtime.bf16_policy: {exc}") from exc
    scaffold_runtime = {}
    if policy is not None and policy.storage_enabled and not require_complete:
        if "precision" not in runtime:
            scaffold_runtime["precision"] = "bf16"
        if "device" not in runtime:
            scaffold_runtime["device"] = "cuda"
        validation_raw = _merge({"runtime": scaffold_runtime}, raw)
    normalized = asdict(config_from_dict(_merge(scaffold, validation_raw)))
    # Descriptor-v1 snapshots retain their historical config wire shape;
    # descriptor-v2 must carry its explicit physical-budget accounting version.
    normalized["runtime"]["packed_activation_offload"] = resolve_packed_offload(
        normalized["runtime"]["packed_activation_offload"]).as_config()
    for field in scaffold_runtime:
        normalized["runtime"].pop(field, None)
    for key in ("name", "seed"):
        if key not in raw:
            normalized.pop(key)
    for key in _TRAINING_REQUIRED:
        if not isinstance(training, Mapping) or key not in training:
            normalized["training"].pop(key)
    # Absent execution policies are not explicit disabling requests. Complete
    # sources contain their dataclass defaults; partial inputs keep provenance.
    for section in ("runtime", "loader", "logging", "schedules", "application", "experiment_identity"):
        if section not in raw:
            normalized.pop(section)
    if "runtime" in normalized:
        for policy in ("fp8", "activation_compression", "activation_memory", "packed_activation_offload", "compile", "ddp", "cuda_graph", "precision", "attention", "kernels", "bf16_policy"):
            if policy not in raw["runtime"]:
                if policy == "packed_activation_offload" and policy in raw["runtime"].get("compile", {}).get("options", {}):
                    continue
                normalized["runtime"].pop(policy, None)
    # JSON round-trip gives snapshots independent mutable containers and turns
    # schema tuples into their portable checkpoint representation.
    return ResolvedConfig(json.loads(json.dumps(normalized, allow_nan=False)))


def resolve_config(config=None, *, overrides=None, base=_UNSET, require_complete=False):
    """Resolve defaults, a source, and explicit overrides into one owned snapshot.

    Partial mappings inherit a scoped/global base. Typed experiments and paths
    are complete sources. Explicit ``base=None`` disables inherited defaults.
    Lists replace lists; False and None override values; _UNSET leaves them alone.
    """
    if isinstance(config, (ResolvedConfig, ExperimentConfig, str, Path)):
        raw = _source_dict(config)
    else:
        inherited = base
        if inherited is _UNSET:
            inherited = _scope.get()
            if inherited is _UNSET:
                inherited = get_config()
        raw = _source_dict(inherited)
        if config is not None:
            if not isinstance(config, Mapping):
                raise ConfigError("Partial configuration must be a mapping")
            raw = _merge(raw, config)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ConfigError("Configuration overrides must be a mapping")
        raw = _merge(raw, overrides)
    return _validate_snapshot(raw, require_complete)


def resolve_section(config, section, *, default=None):
    """Extract an owned section using a dotted path such as runtime.fp8."""
    value = resolve_config(config)
    for key in section.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return deepcopy(default)
        value = value[key]
    return deepcopy(value)


def resolve_training_config(config=None, *, defaults=None):
    """Resolve training settings, with optional consumer-local unused defaults.

    Defaults apply only to root/snapshot sources. Explicit legacy training
    tables retain their required fields, and effective snapshots are untouched.
    """
    if isinstance(config, TrainingConfig):
        return config
    source = config
    legacy_section = (isinstance(config, Mapping) and not isinstance(config, ResolvedConfig)
                      and not (_ROOT_FIELDS & config.keys()))
    if legacy_section:
        source = {"training": config}
    snapshot = resolve_config(source)
    training = dict(snapshot.get("training", {}))
    if defaults is not None and not legacy_section:
        training = {**defaults, **training}
    return training_config_from_dict(training)


@contextmanager
def config_scope(snapshot):
    """Keep factory descendants on one snapshot without changing process defaults."""
    token = _scope.set(resolve_config(snapshot, base=None))
    try:
        yield
    finally:
        _scope.reset(token)
