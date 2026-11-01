"""Parent package for distributed execution cores.

DDPCore is currently the only distributed core. Single-process execution lives
in deepopt.engine.core.TrainingCore; process-group mechanics are internal helpers.
"""

from importlib import import_module


__all__ = ["DDPCore"]


def __getattr__(name):
    if name != "DDPCore":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(".ddp", __name__).DDPCore
    globals()[name] = value
    return value
