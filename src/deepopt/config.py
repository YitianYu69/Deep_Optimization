"""Validated experiment settings; this module does not import PyTorch."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any
import tomllib


class ConfigError(ValueError):
    """A readable configuration contains invalid settings."""


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    accumulation_steps: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    momentum: float = 0.9
    lr_gamma: float = 1.0  # Multiplicative decay at the selected scheduler interval.
    grad_clip_norm: float | None = None
    monitor: str | None = "loss"
    monitor_mode: str = "min"
    checkpoint_every_steps: int = 0  # Opt in with a positive optimizer-update interval.
    scheduler_interval: str = "epoch"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    precision: str = "fp32"
    deterministic: bool = True
    checks: bool = True
    execution_mode: str | None = None
    diagnostics: dict[str, Any] | None = None
    mode_schedule: dict[str, str] | None = None
    non_blocking: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    training: TrainingConfig
    # Application-owned settings, recorded for reproducibility. The framework
    # does not prescribe dataset, model, or task fields inside this table.
    application: dict[str, Any] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment_identity: dict[str, Any] = field(default_factory=dict)


def _require_keys(table: Any, expected: set[str], location: str,
                  optional: set[str] | None = None) -> None:
    if not isinstance(table, dict):
        raise ConfigError(f"{location} must be a TOML table")
    missing = expected - table.keys()
    unknown = table.keys() - expected - (optional or set())
    if missing:
        raise ConfigError(f"{location}: missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{location}: unknown fields: {', '.join(sorted(unknown))}")


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float, *, positive=False) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ConfigError(f"{name} is too large") from exc
    if not isfinite(number) or number < minimum or (positive and number == minimum):
        operator = ">" if positive else ">="
        raise ConfigError(f"{name} must be a finite number {operator} {minimum}")
    return number


def _application_value(value: Any, location: str) -> Any:
    """Copy user options into JSON/checkpoint-safe data without owning a schema."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return value
    if isinstance(value, list):
        return [_application_value(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _application_value(item, f"{location}.{key}") for key, item in value.items()}
    raise ConfigError(f"{location} must contain only strings, booleans, finite numbers, lists, tables, or null; "
                      "encode paths and dates as strings")


def config_from_dict(raw: dict) -> ExperimentConfig:
    """Shared validation for TOML input and checkpoint configuration."""
    if isinstance(raw, dict) and "data" in raw:
        raise ConfigError("The fixed [data] schema was removed. Supply datasets through dataset_factory(); "
                          "store caller-owned options under [application]. For the synthetic demo only, "
                          "move the old [data] fields to [application.synthetic].")
    _require_keys(raw, {"name", "seed", "training"}, "config", {"application", "runtime", "experiment_identity"})
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("name must be a non-empty string")
    seed = _integer(raw["seed"], "seed", 0)
    if seed > 2**32 - 1:
        raise ConfigError("seed must be <= 2**32 - 1")

    t = raw["training"]
    _require_keys(t, {"epochs", "batch_size", "learning_rate"}, "training",
                  {"accumulation_steps", "optimizer", "weight_decay", "momentum", "lr_gamma", "grad_clip_norm",
                   "monitor", "monitor_mode", "checkpoint_every_steps", "scheduler_interval"})
    optimizer = t.get("optimizer", "adamw")
    if optimizer not in ("adamw", "sgd", "custom"):
        raise ConfigError("training.optimizer must be adamw, sgd, or custom")
    momentum = _number(t.get("momentum", 0.9), "training.momentum", 0)
    gamma = _number(t.get("lr_gamma", 1.0), "training.lr_gamma", 0, positive=True)
    if momentum >= 1 or gamma > 1:
        raise ConfigError("momentum must be < 1 and lr_gamma must be <= 1")
    clip = t.get("grad_clip_norm")
    if clip is not None:
        clip = _number(clip, "training.grad_clip_norm", 0, positive=True)
    monitor, monitor_mode = t.get("monitor", "loss"), t.get("monitor_mode", "min")
    if monitor is False:  # TOML has no null; false explicitly disables best selection.
        monitor = None
    if monitor is not None and (not isinstance(monitor, str) or not monitor.strip() or
                                monitor in ("samples", "batches", "seconds", "samples_per_second")):
        raise ConfigError("training.monitor must name a loss or task metric, or be false/null")
    if monitor_mode not in ("min", "max"):
        raise ConfigError("training.monitor_mode must be min or max")
    scheduler_interval = t.get("scheduler_interval", "epoch")
    if scheduler_interval not in ("epoch", "step"):
        raise ConfigError("training.scheduler_interval must be epoch or step")
    training = TrainingConfig(
        epochs=_integer(t["epochs"], "training.epochs", 1),
        batch_size=_integer(t["batch_size"], "training.batch_size", 1),
        learning_rate=_number(t["learning_rate"], "training.learning_rate", 0, positive=True),
        accumulation_steps=_integer(t.get("accumulation_steps", 1), "training.accumulation_steps", 1),
        optimizer=optimizer, weight_decay=_number(t.get("weight_decay", 0), "training.weight_decay", 0),
        momentum=momentum, lr_gamma=gamma, grad_clip_norm=clip,
        monitor=monitor, monitor_mode=monitor_mode,
        checkpoint_every_steps=_integer(t.get("checkpoint_every_steps", 0), "training.checkpoint_every_steps", 0),
        scheduler_interval=scheduler_interval,
    )
    application = raw.get("application", {})
    if not isinstance(application, dict):
        raise ConfigError("application must be a TOML table")
    application = _application_value(application, "application")
    r = raw.get("runtime", {})
    _require_keys(r, set(), "runtime", {"device", "precision", "deterministic", "checks", "execution_mode",
                                          "diagnostics", "mode_schedule", "non_blocking"})
    device, precision = r.get("device", "cpu"), r.get("precision", "fp32")
    if not isinstance(device, str) or not (
        device in ("cpu", "cuda", "auto") or
        (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ConfigError("runtime.device must be cpu, auto, cuda, or cuda:N")
    if precision not in ("fp32", "bf16"):
        raise ConfigError("runtime.precision must be fp32 or bf16; fp16 is unsupported")
    if device == "cpu" and precision != "fp32":
        raise ConfigError("This core supports CPU fp32 and CUDA fp32/bf16")
    deterministic = r.get("deterministic", True)
    if type(deterministic) is not bool:
        raise ConfigError("runtime.deterministic must be a boolean")
    checks = r.get("checks", True)
    if type(checks) is not bool:
        raise ConfigError("runtime.checks must be a boolean")
    mode = r.get("execution_mode")
    modes = ("max_speed", "balanced", "analyze")
    if mode is not None and mode not in modes:
        raise ConfigError("runtime.execution_mode must be max_speed, balanced, or analyze")
    diagnostics = r.get("diagnostics")
    if diagnostics is not None:
        if mode is not None:
            raise ConfigError("Choose runtime.execution_mode or runtime.diagnostics")
        _require_keys(diagnostics, set(), "runtime.diagnostics",
                      {"mode", "loss_finite_every", "gradient_finite_every", "inspect_weights", "inspect_activations"})
        if diagnostics.get("mode", "balanced") not in modes:
            raise ConfigError("Invalid diagnostics mode")
        for key in ("loss_finite_every", "gradient_finite_every"):
            if key in diagnostics and diagnostics[key] is not None:
                _integer(diagnostics[key], f"runtime.diagnostics.{key}", 1)
        for key in ("inspect_weights", "inspect_activations"):
            if key in diagnostics and type(diagnostics[key]) is not bool:
                raise ConfigError(f"runtime.diagnostics.{key} must be a boolean")
        if diagnostics.get("mode") == "max_speed" and any(diagnostics.get(key) for key in
                ("loss_finite_every", "gradient_finite_every", "inspect_weights", "inspect_activations")):
            raise ConfigError("max_speed disables passive diagnostics")
        diagnostics = dict(diagnostics)
    schedule = r.get("mode_schedule")
    if schedule is not None:
        if not isinstance(schedule, dict):
            raise ConfigError("runtime.mode_schedule must be a table")
        for key, value in schedule.items():
            if not isinstance(key, str) or not key.isdigit() or int(key) < 1 or value not in modes:
                raise ConfigError("Mode schedule keys must be positive update numbers and values must be execution modes")
        if len({int(key) for key in schedule}) != len(schedule):
            raise ConfigError("Mode schedule contains duplicate update numbers")
    non_blocking = r.get("non_blocking", False)
    if type(non_blocking) is not bool:
        raise ConfigError("runtime.non_blocking must be a boolean")
    identity = raw.get("experiment_identity", {})
    if not isinstance(identity, dict):
        raise ConfigError("experiment_identity must be a table")
    identity = _application_value(identity, "experiment_identity")
    return ExperimentConfig(name, seed, training, application=application,
                            runtime=RuntimeConfig(device, precision, deterministic, checks, mode, diagnostics, schedule, non_blocking),
                            experiment_identity=identity)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as file:
        return config_from_dict(tomllib.load(file))
