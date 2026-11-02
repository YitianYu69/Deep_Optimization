"""Boundary-driven coefficient values, without optimizer or trainer ownership."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from types import MappingProxyType

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .base import Schedule, finite, interval, position
from .factory import describe, serializable, schedule_from_config, stateless


_fake_reads = ContextVar("deepopt_schedule_fake_reads", default=None)


@contextmanager
def fake_tensor_reads(mode):
    """Let the isolated memory estimator read coefficients without live cache writes."""
    token = _fake_reads.set(mode)
    try:
        yield
    finally:
        _fake_reads.reset(token)


class ScheduleController:
    """Prepare once per accumulation window; commit only after a successful step.

    Trainer supplies its authoritative next optimizer-step count. In manual
    loops the caller supplies that count. The controller records the last
    committed boundary for validation/restoration, never advances on reads,
    and never increments a separate clock on its own.
    """

    def __init__(self, schedules):
        if not isinstance(schedules, Mapping) or any(not isinstance(n, str) or not n.strip() for n in schedules):
            raise ValueError("schedules must map nonempty names to Schedule instances")
        if any(not isinstance(s, Schedule) for s in schedules.values()):
            raise TypeError("schedules must contain Schedule instances")
        if len({id(s) for s in schedules.values()}) != len(schedules):
            raise ValueError("Each coefficient must own a separate schedule instance")
        for schedule in schedules.values():
            interval(schedule.interval)
        self._schedules = dict(schedules)
        self._definitions = {name: describe(schedule) for name, schedule in self._schedules.items()}
        self._values = {name: finite(deepcopy(schedule).value_at(1)) for name, schedule in self._schedules.items()}
        self._positions = {name: 0 for name in self._schedules}
        self._completed_update = self._epoch = 0
        self._pending = self._rollback = None
        self._readonly = 0
        self._tensors = {}

    @classmethod
    def from_config(cls, config):
        return cls({name: schedule_from_config(value) for name, value in config.items()})

    def __len__(self):
        return len(self._schedules)

    def __getitem__(self, name):
        return self._values[name]

    @property
    def values(self):
        return MappingProxyType(self._values)

    @property
    def completed_update(self):
        return self._completed_update

    def get_config(self):
        if {name: describe(schedule) for name, schedule in self._schedules.items()} != self._definitions:
            raise ValueError("Schedule configurations are immutable; construct a new experiment to change them")
        return deepcopy(self._definitions)

    def _mutable(self):
        if self._readonly:
            raise RuntimeError("Validation may read schedules but cannot change them")

    @contextmanager
    def read_only(self):
        self._readonly += 1
        try:
            yield self
        finally:
            self._readonly -= 1

    def tensor(self, name, *, device=None, dtype=torch.float32):
        """Borrow a stable scalar tensor; callers must not mutate it.

        Bind before graph capture/compilation and pass the tensor to compiled
        computation. Schedule evaluation and fill_ remain eager, outside graphs.
        """
        value = self._values[name]
        device = torch.device("cpu" if device is None else device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type not in ("cpu", "cuda") or not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("Coefficient tensors require CPU/CUDA and a floating dtype")
        key = (name, device, dtype)
        mode = _fake_reads.get()
        if mode is not None:
            # Step closures keep their original controller during fake tracing.
            # Never store a fake tensor in that live controller's binding cache.
            if key in self._tensors:
                return mode.from_tensor(self._tensors[key])
            with mode:
                return torch.tensor(value, device=device, dtype=dtype)
        if key not in self._tensors:
            # A lazily materialized read cache is not schedule progress. This
            # also supports evaluation-only processes after checkpoint restore.
            if abs(value) > torch.finfo(dtype).max:
                raise ValueError("Coefficient value is not representable in the requested dtype")
            tensor = torch.tensor(value, device=device, dtype=dtype)
            if isinstance(tensor, FakeTensor):
                raise ValueError("Bind coefficient tensors outside arbitrary fake execution; use DeepOpt's memory estimator")
            self._tensors[key] = tensor
        return self._tensors[key]

    def _publish(self, values):
        for (name, _, dtype) in self._tensors:
            if abs(values[name]) > torch.finfo(dtype).max:
                raise ValueError(f"Coefficient {name!r} is not representable in a bound tensor dtype")
        if self._tensors:
            with torch.no_grad():
                for (name, _, _), tensor in self._tensors.items():
                    tensor.fill_(values[name])
        self._values = dict(values)

    def prepare_update(self, next_update, *, epoch=1):
        self._mutable()
        position(next_update, "next_update")
        position(epoch, "epoch")
        if self._pending == (next_update, epoch):
            return
        if self._pending is not None or next_update != self._completed_update + 1 or epoch < self._epoch:
            raise ValueError("Prepare the next uncommitted update in monotonic epoch order")
        states = {name: deepcopy(schedule.state_dict()) for name, schedule in self._schedules.items()}
        rollback = (dict(self._values), dict(self._positions), states)
        values, positions = dict(self._values), dict(self._positions)
        try:
            for name, schedule in self._schedules.items():
                step = next_update if schedule.interval == "update" else epoch
                if positions[name] != step:
                    values[name] = finite(schedule.value_at(step), name)
                    positions[name] = step
            self._publish(values)
        except BaseException:
            for name, schedule in self._schedules.items():
                schedule.load_state_dict(states[name])
            raise
        self._positions = positions
        self._rollback, self._pending = rollback, (next_update, epoch)

    def complete_update(self, update):
        self._mutable()
        position(update, "update")
        if self._pending is None or self._pending[0] != update:
            raise ValueError("Complete only the prepared update after optimizer success")
        self._completed_update, self._epoch = self._pending
        self._pending = self._rollback = None

    def abort_update(self):
        """Discard preparation after a failed update; optimizer recovery is caller-owned."""
        self._mutable()
        if self._rollback is not None:
            values, positions, states = self._rollback
            for name, schedule in self._schedules.items():
                schedule.load_state_dict(states[name])
            self._publish(values)
            self._positions = positions
        self._pending = self._rollback = None

    def state_dict(self):
        if self._pending is not None:
            raise RuntimeError("Schedule checkpoints require a completed optimizer boundary")
        return {"definitions": self.get_config(), "completed_update": self._completed_update,
                "epoch": self._epoch, "positions": dict(self._positions), "values": dict(self._values),
                "states": {name: serializable(schedule.state_dict()) for name, schedule in self._schedules.items()}}

    def validate_state_dict(self, state, *, completed_update=None):
        required = {"definitions", "completed_update", "epoch", "positions", "values", "states"}
        if not isinstance(state, dict) or set(state) != required or state["definitions"] != self._definitions:
            raise ValueError("Resume requires the same schedule names, types, intervals and immutable configurations")
        count = position(state["completed_update"], "completed_update", 0)
        epoch = position(state["epoch"], "epoch", 0)
        if completed_update is not None and count != completed_update:
            raise ValueError("Schedule progress must match TrainingState.optimizer_steps")
        if (count == 0) != (epoch == 0):
            raise ValueError("Invalid schedule epoch/update progress")
        for key in ("positions", "values", "states"):
            if not isinstance(state[key], dict) or state[key].keys() != self._schedules.keys():
                raise ValueError("Invalid saved schedule state names")
        for name, schedule in self._schedules.items():
            expected = count if schedule.interval == "update" else epoch
            if type(state["positions"][name]) is not int or state["positions"][name] != expected:
                raise ValueError("Invalid saved schedule position")
            finite(state["values"][name], name)
            clone = deepcopy(schedule)
            clone.load_state_dict(deepcopy(state["states"][name]))
            # Stateless schedules are reconstructed from progress, not trusted
            # cached values. Stateful customs restore their last applied value.
            if stateless(clone) and finite(clone.value_at(max(1, expected))) != state["values"][name]:
                raise ValueError("Saved schedule value differs from its progress/configuration")
        serializable(state)

    def load_state_dict(self, state, *, completed_update=None):
        self._mutable()
        self.validate_state_dict(state, completed_update=completed_update)
        for name, schedule in self._schedules.items():
            schedule.load_state_dict(deepcopy(state["states"][name]))
        self._publish(state["values"])
        self._positions = dict(state["positions"])
        self._completed_update, self._epoch = state["completed_update"], state["epoch"]
        self._pending = self._rollback = None
