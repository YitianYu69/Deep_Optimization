"""Validated experiment settings; this module does not import PyTorch."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
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
    lr_schedule: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompileConfig:
    enabled: bool = False
    backend: str = "inductor"
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FP8Config:
    enabled: bool = False
    backend: str = "reference"
    trace_mode: str = "modules"
    activation_dtype: str = "e4m3fn"
    weight_dtype: str = "e4m3fn"
    grad_dtype: str = "e4m3fn"
    block_shapes: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {role: None for role in ("activation", "weight", "grad")}
    )
    num_stages: int = 3
    out_dtype: str | None = None


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
    compile: CompileConfig = field(default_factory=CompileConfig)
    fp8: FP8Config = field(default_factory=FP8Config)
    ddp: dict[str, Any] = field(default_factory=dict)
    cuda_graph: dict[str, Any] = field(default_factory=dict)


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
    schedules: dict[str, Any] = field(default_factory=dict)
    loader: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)


def _require_keys(table: Any, expected: set[str], location: str,
                  optional: set[str] | None = None) -> None:
    if not isinstance(table, Mapping):
        raise ConfigError(f"{location} must be a TOML table")
    if any(not isinstance(key, str) for key in table):
        raise ConfigError(f"{location} keys must be strings")
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


def _copy_tables(value: Any) -> Any:
    """Own mapping/list containers while retaining schema-specific value types."""
    if isinstance(value, Mapping):
        return {key: _copy_tables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tables(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tables(item) for item in value)
    return value


def training_config_from_dict(raw: Mapping[str, Any]) -> TrainingConfig:
    """Validate a training table independently of experiment name and seed."""
    t = _copy_tables(raw)
    _require_keys(t, {"epochs", "batch_size", "learning_rate"}, "training",
                  {"accumulation_steps", "optimizer", "weight_decay", "momentum", "lr_gamma", "grad_clip_norm",
                   "monitor", "monitor_mode", "checkpoint_every_steps", "scheduler_interval", "lr_schedule"})
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
    from .scheduling.lr_config import normalize_lr_schedule
    try:
        lr_schedule = normalize_lr_schedule(t.get("lr_schedule"))
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
    if lr_schedule is not None and gamma != 1.:
        raise ConfigError("Choose training.lr_schedule or legacy lr_gamma, not both")
    return TrainingConfig(
        epochs=_integer(t["epochs"], "training.epochs", 1),
        batch_size=_integer(t["batch_size"], "training.batch_size", 1),
        learning_rate=_number(t["learning_rate"], "training.learning_rate", 0, positive=True),
        accumulation_steps=_integer(t.get("accumulation_steps", 1), "training.accumulation_steps", 1),
        optimizer=optimizer, weight_decay=_number(t.get("weight_decay", 0), "training.weight_decay", 0),
        momentum=momentum, lr_gamma=gamma, grad_clip_norm=clip,
        monitor=monitor, monitor_mode=monitor_mode,
        checkpoint_every_steps=_integer(t.get("checkpoint_every_steps", 0), "training.checkpoint_every_steps", 0),
        scheduler_interval=scheduler_interval, lr_schedule=lr_schedule,
    )


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Shared validation for mapping, TOML, and checkpoint configuration."""
    if isinstance(raw, Mapping) and "data" in raw:
        raise ConfigError("The fixed [data] schema was removed. Supply datasets through dataset_factory(); "
                          "store caller-owned options under [application]. For the synthetic demo only, "
                          "move the old [data] fields to [application.synthetic].")
    _require_keys(raw, {"name", "seed", "training"}, "config",
                  {"application", "runtime", "experiment_identity", "schedules", "loader", "logging"})
    raw = _copy_tables(raw)
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("name must be a non-empty string")
    seed = _integer(raw["seed"], "seed", 0)
    if seed > 2**32 - 1:
        raise ConfigError("seed must be <= 2**32 - 1")
    training = training_config_from_dict(raw["training"])
    from .scheduling.factory import normalize_schedules
    try:
        schedules = normalize_schedules(raw.get("schedules", {}))
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
    application = raw.get("application", {})
    if not isinstance(application, dict):
        raise ConfigError("application must be a TOML table")
    application = _application_value(application, "application")
    r = raw.get("runtime", {})
    _require_keys(r, set(), "runtime", {"device", "precision", "deterministic", "checks", "execution_mode",
                                          "diagnostics", "mode_schedule", "non_blocking", "compile", "fp8", "ddp", "cuda_graph"})
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
        raise ConfigError(f"runtime.execution_mode must be max_speed, balanced, or analyze; got {mode!r}")
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
    compilation = compile_config_from_dict(r.get("compile", {}))
    fp8 = fp8_config_from_dict(r.get("fp8", {}))
    ddp = _ddp_config_from_dict(r.get("ddp", {}))
    cuda_graph = _cuda_graph_config_from_dict(r.get("cuda_graph", {}))
    loader = _loader_config_from_dict(raw.get("loader", {}))
    logging = _logging_config_from_dict(raw.get("logging", {}))
    identity = raw.get("experiment_identity", {})
    if not isinstance(identity, dict):
        raise ConfigError("experiment_identity must be a table")
    identity = _application_value(identity, "experiment_identity")
    return ExperimentConfig(name, seed, training, application=application,
                            runtime=RuntimeConfig(device, precision, deterministic, checks, mode, diagnostics, schedule,
                                                  non_blocking, compilation, fp8, ddp, cuda_graph),
                            experiment_identity=identity, schedules=schedules, loader=loader, logging=logging)


def _ddp_config_from_dict(raw: Mapping[str, Any]) -> dict:
    _require_keys(raw, set(), "runtime.ddp", {
        "find_unused_parameters", "broadcast_buffers", "gradient_as_bucket_view",
        "bucket_cap_mb", "static_graph", "validation_sync_buffers",
    })
    result = dict(raw)
    for key, value in result.items():
        if key == "bucket_cap_mb":
            result[key] = _number(value, f"runtime.ddp.{key}", 0, positive=True)
        elif type(value) is not bool:
            raise ConfigError(f"runtime.ddp.{key} must be a boolean")
    return result


def _cuda_graph_config_from_dict(raw: Mapping[str, Any]) -> dict:
    _require_keys(raw, set(), "runtime.cuda_graph", {
        "device", "warmup", "allow_unused_input", "autograd", "capture_error_mode",
    })
    result = dict(raw)
    device = result.get("device")
    if device is not None and not (isinstance(device, str) and
            (device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()))):
        raise ConfigError("runtime.cuda_graph.device must be cuda, cuda:N, or null")
    if "warmup" in result:
        _integer(result["warmup"], "runtime.cuda_graph.warmup", 1)
    for key in ("allow_unused_input", "autograd"):
        if key in result and type(result[key]) is not bool:
            raise ConfigError(f"runtime.cuda_graph.{key} must be a boolean")
    if result.get("capture_error_mode", "global") not in ("global", "thread_local", "relaxed"):
        raise ConfigError("runtime.cuda_graph.capture_error_mode must be global, thread_local, or relaxed")
    return result


def _loader_config_from_dict(raw: Mapping[str, Any]) -> dict:
    _require_keys(raw, set(), "loader", {
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "drop_last",
    })
    result = dict(raw)
    workers = _integer(result.get("num_workers", 0), "loader.num_workers", 0)
    for key in ("pin_memory", "persistent_workers", "drop_last"):
        if key in result and type(result[key]) is not bool:
            raise ConfigError(f"loader.{key} must be a boolean")
    prefetch = result.get("prefetch_factor")
    if prefetch is not None:
        _integer(prefetch, "loader.prefetch_factor", 1)
        if workers == 0:
            raise ConfigError("loader.prefetch_factor requires num_workers > 0")
    if result.get("persistent_workers", False) and workers == 0:
        raise ConfigError("loader.persistent_workers requires num_workers > 0")
    return result


def _logging_config_from_dict(raw: Mapping[str, Any]) -> dict:
    import logging

    _require_keys(raw, set(), "logging", {"level", "file_level"})
    result = dict(raw)
    for key, value in result.items():
        if key == "level" and value is None:
            continue
        if isinstance(value, str):
            if value.upper() not in logging.getLevelNamesMapping():
                raise ConfigError(f"logging.{key} must be a logging level name or nonnegative integer")
            result[key] = value.upper()
        elif type(value) is not int or value < 0:
            raise ConfigError(f"logging.{key} must be a logging level name or nonnegative integer")
    return result


def compile_config_from_dict(raw: Mapping[str, Any]) -> CompileConfig:
    """Validate compiler settings without importing Torch or loading backends."""
    _require_keys(raw, set(), "runtime.compile", {"enabled", "backend", "mode", "fullgraph", "dynamic", "options"})
    raw = _copy_tables(raw)
    for key in ("enabled", "fullgraph"):
        if type(raw.get(key, False)) is not bool:
            raise ConfigError(f"runtime.compile.{key} must be a boolean")
    dynamic = raw.get("dynamic")
    if dynamic is not None and type(dynamic) is not bool:
        raise ConfigError("runtime.compile.dynamic must be a boolean or null")
    backend = raw.get("backend", "inductor")
    if not isinstance(backend, str) or not backend.strip():
        raise ConfigError("runtime.compile.backend must be a nonempty string")
    mode = raw.get("mode")
    if mode not in (None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        raise ConfigError("runtime.compile.mode must be default, reduce-overhead, max-autotune, or max-autotune-no-cudagraphs")
    options = raw.get("options", {})
    if not isinstance(options, dict) or any(not isinstance(key, str) or not key for key in options):
        raise ConfigError("runtime.compile.options must be a table with nonempty string keys")
    if any(type(value) not in (str, bool, int, float) or (type(value) is float and not isfinite(value))
           for value in options.values()):
        raise ConfigError("runtime.compile.options values must be strings, booleans or finite numbers")
    if mode is not None and options:
        raise ConfigError("Choose runtime.compile.mode or runtime.compile.options, not both")
    return CompileConfig(raw.get("enabled", False), backend, mode, raw.get("fullgraph", False), dynamic, dict(options))


def fp8_config_from_dict(raw: Mapping[str, Any]) -> FP8Config:
    """Validate FP8 settings without importing Torch or loading a GPU backend."""
    _require_keys(raw, set(), "runtime.fp8", {
        "enabled", "backend", "trace_mode", "activation_dtype", "weight_dtype",
        "grad_dtype", "block_shapes", "num_stages", "out_dtype",
    })
    raw = _copy_tables(raw)
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ConfigError("runtime.fp8.enabled must be a boolean")
    backend = raw.get("backend", "reference")
    if backend not in ("reference", "triton"):
        raise ConfigError("runtime.fp8.backend must be reference or triton")
    trace_mode = raw.get("trace_mode", "modules")
    if trace_mode not in ("full", "modules", "auto"):
        raise ConfigError("runtime.fp8.trace_mode must be full, modules, or auto")
    roles = ("activation", "weight", "grad")
    dtypes = {role: raw.get(f"{role}_dtype", "e4m3fn") for role in roles}
    for role, dtype in dtypes.items():
        if dtype not in ("e4m3fn", "e5m2"):
            raise ConfigError(f"runtime.fp8.{role}_dtype must be e4m3fn or e5m2")
    stages = _integer(raw.get("num_stages", 3), "runtime.fp8.num_stages", 1)
    if stages > 4:
        raise ConfigError("runtime.fp8.num_stages must be 1, 2, 3, or 4")
    out_dtype = raw.get("out_dtype")
    if out_dtype not in (None, "fp32", "bf16", "fp16"):
        raise ConfigError("runtime.fp8.out_dtype must be fp32, bf16, fp16, or null")
    blocks = raw.get("block_shapes", {})
    _require_keys(blocks, set(), "runtime.fp8.block_shapes", set(roles))
    block_shapes = {}
    for role in roles:
        shape = blocks.get(role)
        if shape is not None:
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ConfigError(f"runtime.fp8.block_shapes.{role} must contain two positive integers or be null")
            shape = tuple(_integer(size, f"runtime.fp8.block_shapes.{role}", 1) for size in shape)
        block_shapes[role] = shape
    if backend == "triton":
        if len(set(dtypes.values())) != 1:
            raise ConfigError("Triton FP8 requires the same activation, weight, and grad dtype")
        if any(size % 32 for shape in block_shapes.values() if shape is not None for size in shape):
            raise ConfigError("Triton FP8 block dimensions must both be multiples of 32")
    return FP8Config(enabled=enabled, backend=backend, trace_mode=trace_mode,
                     activation_dtype=dtypes["activation"], weight_dtype=dtypes["weight"],
                     grad_dtype=dtypes["grad"], block_shapes=block_shapes,
                     num_stages=stages, out_dtype=out_dtype)


def normalize_config(value: ExperimentConfig | Mapping[str, Any]) -> ExperimentConfig:
    """Keep an existing typed configuration, or validate and copy a mapping."""
    return value if isinstance(value, ExperimentConfig) else config_from_dict(value)


def normalize_training_config(value: TrainingConfig | Mapping[str, Any]) -> TrainingConfig:
    """Keep typed training settings, or parse an independently supplied table."""
    return value if isinstance(value, TrainingConfig) else training_config_from_dict(value)


def _read_config(path: str | Path | ExperimentConfig | Mapping[str, Any]) -> ExperimentConfig:
    """Read a TOML path, explicit builtin preset, or supplied configuration."""
    if isinstance(path, (ExperimentConfig, Mapping)):
        return normalize_config(path)
    if isinstance(path, str) and path.startswith("builtin:"):
        from .configs import load_preset

        return load_preset(path.removeprefix("builtin:"))
    with Path(path).open("rb") as file:
        return config_from_dict(tomllib.load(file))


def load_config(path: str | Path | ExperimentConfig | Mapping[str, Any]) -> ExperimentConfig:
    """Validate and install process-local defaults, returning an independent copy."""
    from ._config_state import install_config

    validated = config_from_dict(asdict(_read_config(path)))
    install_config(validated)
    return deepcopy(validated)


def get_config() -> ExperimentConfig | None:
    """Return a copy of the active process-global configuration, if one was loaded."""
    from ._config_state import get_config as current_config

    return current_config()


def reset_config() -> None:
    """Clear process-global defaults; already constructed objects keep their snapshots."""
    from ._config_state import reset_config as clear_config

    clear_config()
