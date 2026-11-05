"""Opt-in model compilation during experiment construction, outside the cores."""

from collections.abc import Mapping
from dataclasses import asdict

from torch import nn

from deepopt.config import CompileConfig, compile_config_from_dict
from deepopt._config_resolution import _UNSET, resolve_config


def compile_model(model: nn.Module, settings: CompileConfig | Mapping = _UNSET, *, config=None) -> nn.Module:
    """Compile in place, retaining model type, parameters and checkpoint names.

    Compilation is lazy: backend tracing/kernel generation happens on first
    execution. The runtime, optimizer, diagnostics and DDP reducer stay eager.
    Omitted settings inherit runtime.compile from config/current defaults.
    Explicit section settings override that selection; a CompileConfig is a
    complete replacement. The global configuration is never activated here.
    """
    if isinstance(settings, CompileConfig):
        raw = asdict(settings)
    else:
        overrides = (None if settings is _UNSET or settings is None else
                     {"runtime": {"compile": dict(settings) if isinstance(settings, Mapping) else settings}})
        snapshot = resolve_config(config, overrides=overrides)
        raw = snapshot.get("runtime", {}).get("compile", {})
    settings = compile_config_from_dict(raw)
    if not settings.enabled:
        return model
    if any(getattr(module.forward, "_deepopt_cuda_capture", False) for module in model.modules()):
        raise ValueError("Compile the model before CUDA Graph capture, not after capture")
    if getattr(model, "_compiled_call_impl", None) is not None:
        raise ValueError("Model is already compiled; disable runtime.compile or return an eager model from model_factory")
    options = {"backend": settings.backend, "fullgraph": settings.fullgraph, "dynamic": settings.dynamic}
    if settings.mode is not None:
        options["mode"] = settings.mode
    if settings.options:
        options["options"] = dict(settings.options)
    model.compile(**options)
    model._deepopt_compile_config = asdict(settings)
    return model
