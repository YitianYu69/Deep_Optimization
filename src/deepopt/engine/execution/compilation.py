"""Opt-in model compilation during experiment construction, outside the cores."""

from collections.abc import Mapping
from dataclasses import asdict

from torch import nn

from deepopt.config import CompileConfig, compile_config_from_dict


def compile_model(model: nn.Module, settings: CompileConfig | Mapping) -> nn.Module:
    """Compile in place, retaining model type, parameters and checkpoint names.

    Compilation is lazy: backend tracing/kernel generation happens on first
    execution. The runtime, optimizer, diagnostics and DDP reducer stay eager.
    A dictionary accepts the same fields and defaults as CompileConfig.
    """
    raw = asdict(settings) if isinstance(settings, CompileConfig) else settings
    settings = compile_config_from_dict(dict(raw) if isinstance(raw, Mapping) else raw)
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
    return model
