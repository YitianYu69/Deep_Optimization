"""Reference backend implementations grouped by operation."""

from importlib import import_module

__all__ = ["gemm_nt"]


def __getattr__(name):
    if name != "gemm_nt":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(".gemm", __name__).gemm_nt
    globals()[name] = value
    return value
