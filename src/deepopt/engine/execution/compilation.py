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
    if settings.backend != "deepopt" and hasattr(model, "_deepopt_compile_report"):
        from deepopt.compiler.capture import remove_capture

        remove_capture(model)
        del model._deepopt_compile_report
    if settings.backend == "deepopt":
        from deepopt.compiler import BackendConfig, CompilationReport, register_torch_backend
        from deepopt.compiler.backend import make_backend
        from deepopt.compiler.capture import prepare_capture, remove_capture, prepare_precision_body, bind_compiled_precision

        if settings.dynamic is True:
            raise ValueError("The deepopt compiler currently requires dynamic=False (static guarded specialization)")
        from deepopt.kernel_policy import resolve_kernel_policy
        snapshot = resolve_config(config)
        runtime = snapshot.get("runtime", {})
        owned_policy = getattr(model, "_deepopt_kernel_policy", None)
        kernel_options = owned_policy or runtime.get("kernels")
        backend_options = dict(settings.options)
        from deepopt._packed_offload import resolve_owned_packed_offload, resolve_packed_offload

        offload = resolve_owned_packed_offload(getattr(model, "_deepopt_packed_activation_offload_policy", None),
                                             runtime.get("packed_activation_offload"))
        if "packed_activation_offload" in backend_options and resolve_packed_offload(backend_options["packed_activation_offload"]) != offload:
            raise ValueError("Compiler packed activation offload conflicts with the prepared/runtime policy")
        backend_options["packed_activation_offload"] = offload
        from deepopt.opt.policy import resolve_model_activation_memory
        from deepopt._activation_memory import resolve_activation_memory

        memory = resolve_model_activation_memory(model, config=snapshot)
        if "activation_memory" in backend_options and resolve_activation_memory(backend_options["activation_memory"]) != memory:
            raise ValueError("Compiler activation memory conflicts with the prepared/runtime recipe")
        backend_options["activation_memory"] = memory
        from deepopt._bf16_policy import resolve_bf16_policy

        bf16_policy = resolve_bf16_policy(getattr(model, "_deepopt_bf16_policy", runtime.get("bf16_policy")))
        if "bf16_policy" in backend_options and resolve_bf16_policy(backend_options["bf16_policy"]) != bf16_policy:
            raise ValueError("Compiler BF16 policy conflicts with the prepared/runtime BF16 policy")
        backend_options["bf16_policy"] = bf16_policy
        if kernel_options is not None:
            selected = resolve_kernel_policy(kernel_options,
                fp8_backend=runtime.get("fp8", {}).get("backend") if runtime.get("fp8", {}).get("enabled") else None)
            if "kernel_backend" in backend_options and backend_options["kernel_backend"] != selected.backend:
                raise ValueError("Compiler kernel_backend conflicts with the prepared/runtime kernel policy")
            if "kernel_policy" in backend_options and resolve_kernel_policy(backend_options["kernel_policy"]) != selected:
                raise ValueError("Compiler kernel_policy conflicts with the prepared/runtime kernel policy")
            backend_options.update(kernel_backend=selected.backend, kernel_policy=selected.as_dict())
        backend_config = BackendConfig.from_options(backend_options, mode=settings.mode)
        register_torch_backend()
        remove_capture(model)
        report = CompilationReport()
        options["backend"] = make_backend(backend_config, report)
        options["dynamic"] = False
        model._deepopt_compile_report = report
    if settings.mode is not None:
        options["mode"] = settings.mode
    if settings.options:
        options["options"] = dict(settings.options)
    try:
        precision_forward = prepare_precision_body(model) if settings.backend == "deepopt" else None
        if settings.backend == "deepopt" and backend_config.capture_deepopt:
            owned_compression = getattr(model, "_deepopt_activation_compression_config", None)
            if owned_compression is not None and not isinstance(owned_compression, Mapping):
                owned_compression = asdict(owned_compression)
            prepare_capture(model, native_compression=owned_compression or runtime.get("activation_compression"))
        model.compile(**options)
        if settings.backend == "deepopt":
            bind_compiled_precision(model, precision_forward)
    except BaseException:
        if settings.backend == "deepopt":
            from deepopt.compiler.capture import remove_capture

            remove_capture(model)
            del model._deepopt_compile_report
        raise
    model._deepopt_compile_config = asdict(settings)
    if settings.backend == "deepopt":
        model._deepopt_packed_activation_offload_policy = backend_config.packed_activation_offload
    return model
