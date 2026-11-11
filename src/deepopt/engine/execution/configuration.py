"""Resolve execution defaults and prepare models before optimizer/DDP ownership."""

from collections.abc import Mapping
from dataclasses import asdict

from deepopt._config_resolution import _UNSET, ResolvedConfig, resolve_config
from deepopt.config import FP8Config, RuntimeConfig, compile_config_from_dict
from deepopt.opt.policy import (prepare_fp8_model, resolve_activation_compression_config,
                               resolve_fp8_config)
from .compilation import compile_model
from .policy import DiagnosticPolicy, ModeController


def runtime_signature(snapshot):
    """Compare effective runtime ownership, including omitted native defaults."""
    runtime = resolve_config(
        ResolvedConfig({"runtime": asdict(RuntimeConfig())}),
        overrides={"runtime": snapshot.get("runtime", {})},
    )["runtime"]
    runtime.pop("cuda_graph", None)
    runtime.pop("deterministic", None)
    modes = ModeController(runtime.pop("execution_mode"), runtime.pop("diagnostics"),
                           {int(step): mode for step, mode in (runtime.pop("mode_schedule") or {}).items()})
    runtime["diagnostic_policy"] = modes.configuration()
    runtime["ddp"] = {"find_unused_parameters": False, "broadcast_buffers": True,
                      "gradient_as_bucket_view": True, "static_graph": False,
                      "validation_sync_buffers": True, **runtime["ddp"]}
    return runtime, snapshot.get("training", {}).get("grad_clip_norm")


def execution_config(config=None, *, device=_UNSET, precision=_UNSET, checks=_UNSET,
                     grad_clip_norm=_UNSET, execution_mode=_UNSET, diagnostics=_UNSET,
                     mode_schedule=_UNSET, non_blocking=_UNSET, fp8=_UNSET,
                     model=None, ddp_options=_UNSET, training_overrides=None):
    """Explicit serializable arguments override a construction snapshot."""
    if device is not _UNSET and device is not None:
        device = str(device)
    if isinstance(diagnostics, DiagnosticPolicy):
        diagnostics = diagnostics.as_dict()
    if isinstance(mode_schedule, Mapping):
        mode_schedule = {str(step): mode for step, mode in mode_schedule.items()}
    runtime = {key: value for key, value in {
        "device": device, "precision": precision,
        "checks": checks if type(checks) is bool else _UNSET,
        "execution_mode": execution_mode, "diagnostics": diagnostics,
        "mode_schedule": mode_schedule, "non_blocking": non_blocking,
    }.items() if value is not _UNSET}
    if fp8 is not _UNSET:
        # Explicit None preserves an already-prepared model's policy; for an
        # unprepared model it selects the ordinary disabled defaults.
        selected = getattr(model, "_deepopt_fp8_config", None) if fp8 is None else fp8
        if selected is None:
            selected = FP8Config()
        if isinstance(selected, FP8Config):
            selected = asdict(selected)
        elif type(selected) is bool:
            selected = {"enabled": selected}
        runtime["fp8"] = selected
    if ddp_options is not _UNSET:
        runtime["ddp"] = {} if ddp_options is None else ddp_options
    overrides = {"runtime": runtime} if runtime else {}
    training = {key: value for key, value in (training_overrides or {}).items() if value is not _UNSET}
    if grad_clip_norm is not _UNSET:
        training["grad_clip_norm"] = grad_clip_norm
    if training:
        overrides["training"] = training
    return resolve_config(config, overrides=overrides)


def prepare_execution_model(model, snapshot, fp8=_UNSET):
    """Reuse matching preparation while keeping parameter objects unchanged."""
    settings = resolve_fp8_config(model, fp8, config=snapshot)
    compression = resolve_activation_compression_config(model, config=snapshot)
    model, report = prepare_fp8_model(model, settings, config=snapshot, activation_compression=compression)
    if compression.enabled or hasattr(model, "_deepopt_activation_compression_config"):
        snapshot.setdefault("runtime", {})["activation_compression"] = asdict(compression)
    compilation = compile_config_from_dict(snapshot.get("runtime", {}).get("compile", {}))
    if compilation.enabled:
        previous = getattr(model, "_deepopt_compile_config", None)
        if previous != asdict(compilation):
            compile_model(model, compilation, config=ResolvedConfig())
    return model, settings, report
