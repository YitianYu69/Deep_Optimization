"""Reserved for DeepOpt optimizer algorithms; legacy factories remain lazy aliases."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer_factory import build_optimizer, OptimizerApplyFunction
    from .scheduling.lr import build_scheduler

__all__ = ["build_optimizer", "build_scheduler", "OptimizerApplyFunction"]


def __getattr__(name):
    if name in ("build_optimizer", "OptimizerApplyFunction"):
        from . import optimizer_factory
        return getattr(optimizer_factory, name)
    if name == "build_scheduler":
        from .scheduling.lr import build_scheduler
        return build_scheduler
    raise AttributeError(name)
