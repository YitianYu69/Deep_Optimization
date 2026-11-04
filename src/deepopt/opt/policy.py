"""Translate serializable FP8 configuration into model preparation settings."""

from dataclasses import asdict

from deepopt.config import FP8Config, fp8_config_from_dict


def normalize_fp8_config(value: FP8Config | dict | bool | None) -> FP8Config:
    """Normalize direct Python convenience forms using the configuration schema."""
    if value is None:
        value = {}
    elif type(value) is bool:
        value = {"enabled": value}
    elif isinstance(value, FP8Config):
        value = asdict(value)
    return fp8_config_from_dict(value)


def fp8_plan(settings: FP8Config | dict | bool | None) -> dict:
    """Resolve dtype names only when an FP8 computation plan is requested."""
    settings = normalize_fp8_config(settings)
    import torch

    fp8_types = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    output_types = {None: None, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    return {
        "fp8_training": {
            "fp8_dtypes": {role: fp8_types[getattr(settings, f"{role}_dtype")]
                           for role in ("activation", "weight", "grad")},
            "backend": settings.backend,
            "num_stages": settings.num_stages,
            "out_dtype": output_types[settings.out_dtype],
        },
        "block_shapes": dict(settings.block_shapes),
    }


def resolve_fp8_config(model, settings) -> FP8Config:
    """Retain a prepared model's known policy, or reject conflicting settings."""
    previous = getattr(model, "_deepopt_fp8_config", None)
    settings = normalize_fp8_config(previous if settings is None else settings)
    if previous is not None and previous != settings:
        raise ValueError("FP8 settings differ from the prepared model; reconstruct the original model first")
    return settings


def prepare_fp8_model(model, settings: FP8Config | dict | bool | None):
    """Prepare an eager model once; use the returned model for subsequent calls.

    Module mode keeps an ordinary container root, converting supported children.
    An exact registered projection at the root is replaced by its DO adapter.
    Full/auto mode can return a GraphModule. Original parameters remain shared.
    """
    settings = resolve_fp8_config(model, settings)
    if not settings.enabled:
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
    from .transform import TransformReport, transform_model

    registry = default_registry()
    plan = fp8_plan(settings)
    root_rule = registry.lookup(model) if settings.trace_mode == "modules" else None
    if root_rule is not None:
        prepared = root_rule.converter(model, plan=plan)
        report = TransformReport(
            trace_mode="modules",
            original_type=f"{type(model).__module__}.{type(model).__qualname__}",
            converted_modules=("<root>",), function_calls=(),
        )
    else:
        prepared, report = transform_model(model, plan=plan, trace_mode=settings.trace_mode,
                                           inplace=True, registry=registry)
    prepared._deepopt_fp8_config = settings
    prepared._deepopt_fp8_report = report
    return prepared, report
