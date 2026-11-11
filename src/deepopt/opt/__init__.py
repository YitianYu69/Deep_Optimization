from importlib import import_module
from os.path import join as _join

# Resolve former package names lazily while keeping canonical modules first.
__path__.append(_join(__path__[0], "_legacy"))
del _join

__all__ = [
    "transform_model",
    "TransformReport",
    "TransformationTraceError",
    "ConversionRegistry",
    "default_registry",
    "autocast",
    "FP8Config",
    "DelayedScalingConfig", "fp8_scaling_scope", "fp8_checkpoint",
]

_EXPORTS = {
    "autocast": "._autocast",
    "FP8Config": "..config",
    "DelayedScalingConfig": "..config",
    "fp8_scaling_scope": ".scaling.delayed",
    "fp8_checkpoint": ".scaling.delayed",
    "transform_model": ".transform",
    "TransformReport": ".transform",
    "TransformationTraceError": ".transform",
    "ConversionRegistry": ".fx.registry",
    "default_registry": ".fx.registry",
}


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)

    if module_path is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module = import_module(module_path, __name__)
    value = getattr(module, name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
