from importlib import import_module


_EXPORTS = {
    "DOTracer": ".tracer",
    "ModuleConverter": ".registry",
    "ModuleConversion": ".registry",
    "ConversionRegistry": ".registry",
    "default_registry": ".registry",
    "match_graph": ".matcher",
    "rewrite_graph": ".rewriter",
}


__all__ = list(_EXPORTS)



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