"""Training-only coefficient schedules; curve/config imports do not load Torch."""

from .base import Schedule
from .curves import Constant, LinearWarmup, LinearDecay, CosineDecay, ExponentialDecay, Piecewise, SequentialSchedule

__all__ = ["Schedule", "Constant", "LinearWarmup", "LinearDecay", "CosineDecay", "ExponentialDecay",
           "Piecewise", "SequentialSchedule", "ScheduleController"]


def __getattr__(name):
    if name == "ScheduleController":
        from .controller import ScheduleController
        return ScheduleController
    raise AttributeError(name)
