"""Finite, one-based curves. Bounded curves hold their final value."""

from bisect import bisect_right
from dataclasses import dataclass, asdict
from math import cos, pi

from .base import Schedule, finite, interval, position


@dataclass(frozen=True)
class Constant(Schedule):
    value: float
    interval: str = "update"

    def __post_init__(self):
        finite(self.value)
        interval(self.interval)

    def value_at(self, step):
        position(step)
        return finite(self.value)

    def get_config(self):
        return asdict(self)


@dataclass(frozen=True)
class LinearWarmup(Schedule):
    start: float
    end: float
    duration: int
    interval: str = "update"

    def __post_init__(self):
        finite(self.start, "start")
        finite(self.end, "end")
        position(self.duration, "duration")
        interval(self.interval)

    def value_at(self, step):
        position(step)
        alpha = 1. if self.duration == 1 else min((step - 1) / (self.duration - 1), 1.)
        # A weighted sum avoids overflow in end-start for large opposite signs.
        return finite((1 - alpha) * self.start + alpha * self.end)

    def get_config(self):
        return asdict(self)


@dataclass(frozen=True)
class LinearDecay(LinearWarmup):
    """Same endpoint interpolation as warmup; the caller chooses the direction."""


@dataclass(frozen=True)
class CosineDecay(LinearWarmup):
    def value_at(self, step):
        position(step)
        alpha = 1. if self.duration == 1 else min((step - 1) / (self.duration - 1), 1.)
        weight = (1 - cos(pi * alpha)) / 2
        return finite((1 - weight) * self.start + weight * self.end)


@dataclass(frozen=True)
class ExponentialDecay(Schedule):
    start: float
    gamma: float
    duration: int | None = None
    interval: str = "update"

    def __post_init__(self):
        finite(self.start, "start")
        if not 0 < finite(self.gamma, "gamma") <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if self.duration is not None:
            position(self.duration, "duration")
        interval(self.interval)

    def value_at(self, step):
        position(step)
        step = min(step, self.duration) if self.duration is not None else step
        return finite(self.start * self.gamma ** (step - 1))

    def get_config(self):
        return asdict(self)


@dataclass(frozen=True)
class Piecewise(Schedule):
    milestones: tuple[int, ...]
    values: tuple[float, ...]
    interval: str = "update"

    def __post_init__(self):
        object.__setattr__(self, "milestones", tuple(self.milestones))
        object.__setattr__(self, "values", tuple(self.values))
        interval(self.interval)
        for milestone in self.milestones:
            position(milestone, "milestone", minimum=2)
        if any(a >= b for a, b in zip(self.milestones, self.milestones[1:])):
            raise ValueError("milestones must be strictly increasing without duplicates")
        if len(self.values) != len(self.milestones) + 1:
            raise ValueError("Piecewise needs one more value than milestones")
        for value in self.values:
            finite(value)

    def value_at(self, step):
        position(step)
        return finite(self.values[bisect_right(self.milestones, step)])

    def get_config(self):
        return {"milestones": list(self.milestones), "values": list(self.values), "interval": self.interval}


@dataclass(frozen=True)
class SequentialSchedule(Schedule):
    schedules: tuple[Schedule, ...]
    durations: tuple[int, ...]
    interval: str = "update"

    def __post_init__(self):
        object.__setattr__(self, "schedules", tuple(self.schedules))
        object.__setattr__(self, "durations", tuple(self.durations))
        interval(self.interval)
        if not self.schedules or len(self.schedules) != len(self.durations):
            raise ValueError("SequentialSchedule needs one positive duration per child")
        for schedule, duration in zip(self.schedules, self.durations):
            position(duration, "duration")
            if not isinstance(schedule, Schedule) or schedule.interval != self.interval:
                raise ValueError("Sequential children must be schedules with the same interval")

    def value_at(self, step):
        position(step)
        for schedule, duration in zip(self.schedules, self.durations):
            if step <= duration:
                return finite(schedule.value_at(step))
            step -= duration
        return finite(self.schedules[-1].value_at(self.durations[-1]))

    def get_config(self):
        from .factory import describe
        return {"schedules": [describe(child) for child in self.schedules],
                "durations": list(self.durations), "interval": self.interval}

    def state_dict(self):
        return {"children": [child.state_dict() for child in self.schedules]}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"children"} or len(state["children"]) != len(self.schedules):
            raise ValueError("Invalid sequential schedule state")
        for child, saved in zip(self.schedules, state["children"]):
            child.load_state_dict(saved)
