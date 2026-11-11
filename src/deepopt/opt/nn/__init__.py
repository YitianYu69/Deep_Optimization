"""Projection adapters and eager autograd entry points."""

from importlib import import_module
from os.path import dirname as _dirname, join as _join

__path__.append(_join(_dirname(__path__[0]), "_legacy", "nn"))
del _dirname, _join

__all__ = ["functional"]


def __getattr__(name):
    if name != "functional":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".functional", __name__)
    globals()[name] = module
    return module
