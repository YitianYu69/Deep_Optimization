"""Translate serializable FP8 configuration into model preparation settings."""

from dataclasses import asdict

from deepopt.config import (ActivationCompressionConfig, FP8Config,
                            activation_compression_config_from_dict, fp8_config_from_dict)
from deepopt._config_resolution import _UNSET, ResolvedConfig, resolve_config
from deepopt._bf16_policy import resolve_bf16_policy
from deepopt._activation_memory import resolve_activation_memory, validate_activation_memory_execution


def normalize_bf16_policy(value=_UNSET, *, config=None):
    snapshot = resolve_config(config)
    return resolve_bf16_policy(snapshot.get("runtime", {}).get("bf16_policy")
                               if value is _UNSET else value)


def resolve_model_bf16_policy(model, *, config=None):
    snapshot = resolve_config(config)
    previous = getattr(model, "_deepopt_bf16_policy", None)
    selected = snapshot.get("runtime", {}).get("bf16_policy", previous)
    policy = resolve_bf16_policy(selected)
    if previous is not None and resolve_bf16_policy(previous) != policy:
        raise ValueError("BF16 policy differs from the prepared model; reconstruct the original model first")
    return policy


def resolve_model_activation_memory(model, *, config=None):
    """Keep a prepared model's owned recipe when ambient settings are absent."""
    snapshot = resolve_config(config)
    runtime = snapshot.get("runtime", {})
    previous = getattr(model, "_deepopt_activation_memory_plan", None)
    selected = runtime.get("activation_memory", previous)
    options = runtime.get("compile", {})
    plan = resolve_activation_memory(selected, compiler_options=(
        options.get("options") if options.get("backend") == "deepopt" else None))
    if previous is not None and resolve_activation_memory(previous) != plan:
        raise ValueError("Activation memory differs from the prepared model; reconstruct the original model first")
    return plan


def normalize_fp8_config(value=_UNSET, *, config=None) -> FP8Config:
    """Resolve omitted/partial policy fields against selected configuration.

    A typed FP8Config is a complete replacement. Explicit None disables FP8;
    the prepared-model helper separately preserves its None/inherit contract.
    """
    if isinstance(value, FP8Config):
        return fp8_config_from_dict(asdict(value))
    if value is None:
        value = {"enabled": False}
    elif type(value) is bool:
        value = {"enabled": value}
    overrides = None if value is _UNSET else {"runtime": {"fp8": value}}
    snapshot = resolve_config(config, overrides=overrides)
    return fp8_config_from_dict(snapshot.get("runtime", {}).get("fp8", {}))


def normalize_activation_compression_config(value=_UNSET, *, config=None) -> ActivationCompressionConfig:
    """Resolve independent saved-activation settings from an owned snapshot."""
    if isinstance(value, ActivationCompressionConfig):
        return activation_compression_config_from_dict(asdict(value))
    if value is None:
        value = {"enabled": False}
    elif type(value) is bool:
        value = {"enabled": value}
    overrides = None if value is _UNSET else {"runtime": {"activation_compression": value}}
    snapshot = resolve_config(config, overrides=overrides)
    return activation_compression_config_from_dict(snapshot.get("runtime", {}).get("activation_compression", {}))


def resolve_activation_compression_config(model, settings=_UNSET, *, config=None):
    """Retain a prepared projection's codec policy and reject conflicting reuse."""
    previous = getattr(model, "_deepopt_activation_compression_config", None)
    if settings is None:
        settings = normalize_activation_compression_config(previous, config=ResolvedConfig({}))
    elif isinstance(settings, ActivationCompressionConfig):
        settings = normalize_activation_compression_config(settings)
    else:
        snapshot = resolve_config(config)
        if (settings is _UNSET and previous is not None
                and "activation_compression" not in snapshot.get("runtime", {})):
            settings = previous
        settings = normalize_activation_compression_config(settings, config=snapshot)
    if previous is not None and previous != settings:
        raise ValueError("Activation compression settings differ from the prepared model; reconstruct the original model first")
    return settings


def fp8_plan(settings=_UNSET, *, config=None, activation_compression=_UNSET) -> dict:
    """Resolve dtype names only when an FP8 computation plan is requested."""
    settings = normalize_fp8_config(settings, config=config)
    # An owned ResolvedConfig is already validated; re-resolving it here would
    # repeat schema work in autocast's hot path.
    snapshot = config if isinstance(config, ResolvedConfig) else resolve_config(config)
    if resolve_bf16_policy(snapshot.get("runtime", {}).get("bf16_policy")).storage_enabled:
        raise ValueError("bf16_storage requires FP8 disabled")
    compression = normalize_activation_compression_config(activation_compression, config=config)
    from deepopt.kernel_policy import resolve_kernel_policy
    kernel_options = snapshot.get("runtime", {}).get("kernels")
    kernels = resolve_kernel_policy(kernel_options, fp8_backend=settings.backend)
    backend = settings.backend if settings.backend == "reference" else kernels.backend
    extra = ({"kernel_policy": kernels.as_dict()} if kernel_options is not None
             or settings.backend in ("inherit", "tilelang") else {})
    import torch

    fp8_types = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    output_types = {None: None, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    from deepopt.opt.nn.functional import validate_fp8_plan

    return validate_fp8_plan({
        "fp8_training": {
            "fp8_dtypes": {role: fp8_types[getattr(settings, f"{role}_dtype")]
                           for role in ("activation", "weight", "grad")},
            "backend": backend,
            **extra,
            "num_stages": settings.num_stages,
            "autotune": settings.autotune,
            "out_dtype": output_types[settings.out_dtype],
            "scaling_recipe": settings.scaling_recipe,
            "delayed_scaling": asdict(settings.delayed_scaling),
            "backward_backend": settings.backward_backend,
            "backward_precision": settings.backward_precision,
            "convolution_policy": settings.convolution_policy,
            "activation_compression": asdict(compression),
        },
        "block_shapes": dict(settings.block_shapes),
        **({"activation_memory": resolve_activation_memory(snapshot["runtime"]["activation_memory"]).as_dict()}
           if "activation_memory" in snapshot.get("runtime", {}) else {}),
    })


def linear_plan(fp8=_UNSET, *, activation_compression=_UNSET, config=None) -> dict:
    """Resolve independent optimizations into an owned immutable linear plan.

    FP8 alone selects the existing FP8 path. Compression alone selects native
    compute with packed saved inputs. Enabling both selects fused FP8 packing.
    """
    snapshot = resolve_config(config)
    settings = normalize_fp8_config(fp8, config=snapshot)
    compression = normalize_activation_compression_config(activation_compression, config=snapshot)
    from deepopt.opt.nn.functional import _prepare_linear_plan

    if settings.enabled:
        return _prepare_linear_plan(fp8_plan(settings, config=snapshot, activation_compression=compression))
    values = {"fp8": False, "activation_compression": asdict(compression),
              "bf16_policy": normalize_bf16_policy(config=snapshot).as_dict()}
    if "activation_memory" in snapshot.get("runtime", {}):
        values["activation_memory"] = resolve_activation_memory(snapshot["runtime"]["activation_memory"]).as_dict()
    if "kernels" in snapshot.get("runtime", {}):
        from deepopt.kernel_policy import resolve_kernel_policy
        values["kernel_policy"] = resolve_kernel_policy(snapshot["runtime"]["kernels"]).as_dict()
    return _prepare_linear_plan(values)


def conv_plan(fp8=_UNSET, *, activation_compression=_UNSET, config=None) -> dict:
    """Use the shared independent precision/storage policy for spatial convolutions."""
    return linear_plan(fp8, activation_compression=activation_compression, config=config)


def normalization_plan(fp8=_UNSET, *, activation_compression=_UNSET, config=None) -> dict:
    """Use the shared compute policy while saving normalized activations."""
    return linear_plan(fp8, activation_compression=activation_compression, config=config)


def pooling_plan(fp8=_UNSET, *, activation_compression=_UNSET, config=None) -> dict:
    """Shared precision policy with exact routing or preprocessed pooling state."""
    return linear_plan(fp8, activation_compression=activation_compression, config=config)


def relu_plan(fp8=_UNSET, *, activation_compression=_UNSET, config=None) -> dict:
    """Use the shared precision policy; ReLU compresses its exact mask to one bit."""
    return linear_plan(fp8, activation_compression=activation_compression, config=config)


def activation_plan(fp8=_UNSET, *, activation_compression=_UNSET, config=None) -> dict:
    """Shared activation policy; smooth functions save local derivatives, not raw x."""
    return linear_plan(fp8, activation_compression=activation_compression, config=config)


def resolve_fp8_config(model, settings=_UNSET, *, config=None) -> FP8Config:
    """Retain a prepared model's known policy, or reject conflicting settings."""
    previous = getattr(model, "_deepopt_fp8_config", None)
    if settings is None:
        # Explicit None retains the prepared-model contract independently of
        # configuration loaded after that model was created.
        settings = normalize_fp8_config(previous, config=ResolvedConfig({}))
    elif isinstance(settings, FP8Config):
        settings = normalize_fp8_config(settings)
    elif settings is _UNSET:
        snapshot = resolve_config(config)
        if previous is not None and "fp8" not in snapshot.get("runtime", {}):
            settings = previous
        settings = normalize_fp8_config(settings, config=snapshot)
    else:
        settings = normalize_fp8_config(settings, config=config)
    if previous is not None and previous != settings:
        raise ValueError("FP8 settings differ from the prepared model; reconstruct the original model first")
    return settings


def prepare_fp8_model(model, settings=_UNSET, *, config=None, activation_compression=_UNSET):
    """Prepare an eager model once; use the returned model for subsequent calls.

    Module mode keeps an ordinary container root, converting supported children.
    An exact registered projection at the root is replaced by its DO adapter.
    Full/auto mode can return a GraphModule. Original parameters remain shared.
    """
    settings = resolve_fp8_config(model, settings, config=config)
    compression = resolve_activation_compression_config(model, activation_compression, config=config)
    bf16_policy = resolve_model_bf16_policy(model, config=config)
    activation_memory = resolve_model_activation_memory(model, config=config)
    snapshot = resolve_config(config)
    runtime = snapshot.get("runtime", {})
    # A fully disabled policy stays a no-op: no precision, storage, memory,
    # attention, or kernel policy is selected and the model carries no marker
    # from an earlier preparation. Preserve that contract so callers can supply
    # an eager model (or a placeholder) without inspection or mutation.
    if (not settings.enabled and not compression.enabled and not bf16_policy.storage_enabled
            and activation_memory.is_default
            and runtime.get("attention", {}).get("backend", "native") == "native"
            and runtime.get("kernels") is None
            and getattr(model, "_deepopt_attention_config", None) is None
            and getattr(model, "_deepopt_bf16_policy", None) is None
            and getattr(model, "_deepopt_kernel_policy", None) is None):
        return model, None
    from deepopt._bf16_policy import STORAGE_PRIMITIVE_RECIPE_VERSION

    if (bf16_policy.storage_enabled and getattr(model, "_deepopt_bf16_primitive_recipe", STORAGE_PRIMITIVE_RECIPE_VERSION)
            != STORAGE_PRIMITIVE_RECIPE_VERSION):
        raise ValueError("BF16 primitive recipe differs from the prepared model; reconstruct the original model first")
    if any(getattr(module, "_deepopt_attention_recipe_version", 2) != 2 for module in model.modules()):
        raise ValueError("Attention preparation recipe changed; reconstruct the original model first")
    if bf16_policy.storage_enabled and settings.enabled:
        raise ValueError("bf16_storage requires FP8 disabled")
    from deepopt.kernel_policy import resolve_kernel_policy
    kernel_options = runtime.get("kernels")
    previous_kernel_policy = getattr(model, "_deepopt_kernel_policy", None)
    if previous_kernel_policy is not None and (kernel_options is None or kernel_options.get("backend", "inherit") == "inherit"):
        kernel_options = previous_kernel_policy
    kernel_policy = resolve_kernel_policy(kernel_options, fp8_backend=settings.backend if settings.enabled else None)
    if previous_kernel_policy is not None and previous_kernel_policy != kernel_policy.as_dict():
        raise ValueError("Kernel policy differs from the prepared model; reconstruct the original model first")
    validate_activation_memory_execution(activation_memory, kernel_policy=kernel_policy, model=model)
    def attention(prepared):
        from .precision.bf16 import bind_bf16_model_policy
        if not activation_memory.is_default:
            from .nn.functional import _prepare_linear_plan

            for module in prepared.modules():
                plan = getattr(module, "plan", None)
                if isinstance(plan, dict) and ("fp8_training" in plan or plan.get("fp8") is False):
                    module.plan = _prepare_linear_plan({**plan, "activation_memory": activation_memory.as_dict()})
        if kernel_options is not None or kernel_policy.backend != "triton":
            from .nn.functional import _prepare_linear_plan
            prepared._deepopt_kernel_policy = kernel_policy.as_dict()
            for module in prepared.modules():
                from .scaling.delayed import DelayedScalingState
                if isinstance(module, DelayedScalingState):
                    module.kernel_policy = kernel_policy
                plan = getattr(module, "plan", None)
                if isinstance(plan, dict):
                    if "fp8_training" in plan:
                        plan = {**plan, "fp8_training": {**plan["fp8_training"], "kernel_policy": kernel_policy.as_dict()}}
                    elif plan.get("fp8") is False:
                        plan = {**plan, "kernel_policy": kernel_policy.as_dict()}
                    else:
                        continue
                    module.plan = _prepare_linear_plan(plan)
        snapshot = resolve_config(config)
        backend = snapshot.get("runtime", {}).get("attention", {}).get("backend", "native")
        if backend == "native" and getattr(prepared, "_deepopt_attention_config", None) is None:
            return bind_bf16_model_policy(prepared, bf16_policy, activation_memory=activation_memory)
        from .nn.attention_backend import prepare_attention_model
        return bind_bf16_model_policy(
            prepare_attention_model(prepared, config=config, compression=compression), bf16_policy,
            activation_memory=activation_memory)

    if not settings.enabled:
        if compression.enabled or bf16_policy.storage_enabled:
            from .nn.prepare import prepare_native_operations

            previous = getattr(model, "_deepopt_native_normalization_report", None)
            if previous is not None:
                return attention(model), previous
            model, report = prepare_native_operations(model, compression, settings.exclude_modules,
                                                       bf16_policy=bf16_policy)
            if report is not None:
                model._deepopt_activation_compression_config = compression
                model._deepopt_fp8_config = settings
                model._deepopt_native_normalization_report = report
            return attention(model), report
        return attention(model), None

    previous = getattr(model, "_deepopt_fp8_config", None)
    if previous is not None:
        return attention(model), model._deepopt_fp8_report

    for module in model.modules():
        if (getattr(module, "_compiled_call_impl", None) is not None
                or getattr(module.forward, "_torchdynamo_orig_callable", None) is not None):
            raise ValueError("Prepare FP8 before compilation; supply an eager model")
        if getattr(module.forward, "_deepopt_cuda_capture", False):
            raise ValueError("Prepare FP8 before CUDA Graph capture; supply an eager model")

    from .fx.registry import default_registry
    from .transform import TransformReport, transform_model, excluded_module_ids

    registry = default_registry()
    plan = fp8_plan(settings, config=config, activation_compression=compression)
    excluded_ids, excluded_names = excluded_module_ids(model, settings.exclude_modules)
    from .nn.convolution_transpose.modules import CONVERSIONS as TRANSPOSE_CONVERSIONS

    # A root transposed module has a public runtime output_size argument.
    # Keep its adapter API in every mode rather than tracing/specializing that
    # optional argument into an FX assertion against the default None.
    root_rule = (registry.lookup(model) if (settings.trace_mode == "modules" or type(model) in TRANSPOSE_CONVERSIONS)
                 and id(model) not in excluded_ids else None)
    if root_rule is not None:
        prepared = registry.convert(model, plan=plan, excluded_ids=excluded_ids)
        report = TransformReport(
            trace_mode="modules",
            original_type=f"{type(model).__module__}.{type(model).__qualname__}",
            converted_modules=("<root>",), function_calls=(),
            excluded_modules=excluded_names,
        )
    else:
        prepared, report = transform_model(model, plan=plan, trace_mode=settings.trace_mode,
                                           inplace=True, registry=registry, exclude_modules=settings.exclude_modules)
    prepared._deepopt_fp8_config = settings
    prepared._deepopt_activation_compression_config = compression
    prepared._deepopt_fp8_report = report
    from .precision import prepare_precision_plan

    from deepopt.kernel_policy import kernel_policy_scope
    with kernel_policy_scope(kernel_policy):
        prepare_precision_plan(prepared, settings)
    return attention(prepared), report
