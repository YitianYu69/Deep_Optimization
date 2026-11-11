"""Translate serializable FP8 configuration into model preparation settings."""

from dataclasses import asdict

from deepopt.config import (ActivationCompressionConfig, FP8Config,
                            activation_compression_config_from_dict, fp8_config_from_dict)
from deepopt._config_resolution import _UNSET, ResolvedConfig, resolve_config


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
    compression = normalize_activation_compression_config(activation_compression, config=config)
    import torch

    fp8_types = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    output_types = {None: None, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    from deepopt.opt.nn.functional import validate_fp8_plan

    return validate_fp8_plan({
        "fp8_training": {
            "fp8_dtypes": {role: fp8_types[getattr(settings, f"{role}_dtype")]
                           for role in ("activation", "weight", "grad")},
            "backend": settings.backend,
            "num_stages": settings.num_stages,
            "out_dtype": output_types[settings.out_dtype],
            "scaling_recipe": settings.scaling_recipe,
            "delayed_scaling": asdict(settings.delayed_scaling),
            "backward_backend": settings.backward_backend,
            "backward_precision": settings.backward_precision,
            "activation_compression": asdict(compression),
        },
        "block_shapes": dict(settings.block_shapes),
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
    return _prepare_linear_plan({"fp8": False, "activation_compression": asdict(compression)})


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
    if not settings.enabled:
        if compression.enabled:
            from .nn.prepare import prepare_native_operations

            previous = getattr(model, "_deepopt_native_normalization_report", None)
            if previous is not None:
                return model, previous
            model, report = prepare_native_operations(model, compression, settings.exclude_modules)
            if report is not None:
                model._deepopt_activation_compression_config = compression
                model._deepopt_fp8_config = settings
                model._deepopt_native_normalization_report = report
            return model, report
        return model, None

    previous = getattr(model, "_deepopt_fp8_config", None)
    if previous is not None:
        return model, model._deepopt_fp8_report

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

    prepare_precision_plan(prepared, settings)
    return prepared, report
