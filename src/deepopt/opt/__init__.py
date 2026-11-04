from importlib import import_module

__all__ = [
    "transform_model",
    "TransformReport",
    "TransformationTraceError",
    "ConversionRegistry",
    "default_registry",
    "autocast",
    "FP8Config",
]

_EXPORTS = {
    "autocast": "._autocast",
    "FP8Config": "..config",
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
