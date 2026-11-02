"""Pure-Python, deterministic schedule contracts."""

from abc import ABC, abstractmethod
from math import isfinite


def position(value, name="position", minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name="value"):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def interval(value):
    if value not in ("update", "epoch"):
        raise ValueError("schedule interval must be update or epoch; validation scheduling is unsupported")
    return value


class Schedule(ABC):
    """Values use one-based update/epoch positions; evaluation stays outside graphs.

    Custom schedules must be deterministic across ranks. Mutable custom state
    must be represented by state_dict/load_state_dict, with immutable identity
    in get_config. The default implementations describe stateless schedules.
    """

    interval = "update"

    @abstractmethod
    def value_at(self, position: int) -> float:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state != {}:
            raise ValueError("Stateless schedules require empty state")
