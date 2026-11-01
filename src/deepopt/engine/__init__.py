"""Training loops and checkpoint lifecycle, independent of experiment factories."""

from deepopt.engine.training.trainer import Trainer, TrainingState
from deepopt.engine.execution.runtime import TrainingRuntime
from deepopt.engine.execution.optimization import OptimizationWindow
from deepopt.engine.core import TrainingCore

__all__ = ["Trainer", "TrainingState", "TrainingRuntime", "OptimizationWindow", "TrainingCore"]

from ._compat import register_aliases

register_aliases()
del register_aliases
