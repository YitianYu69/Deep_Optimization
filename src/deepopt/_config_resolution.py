"""Resolve immutable-in-use configuration snapshots without initializing Torch."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

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
    normalized = asdict(config_from_dict(_merge(scaffold, raw)))
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
        for policy in ("fp8", "compile", "ddp", "cuda_graph", "precision"):
            if policy not in raw["runtime"]:
                normalized["runtime"].pop(policy)
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
