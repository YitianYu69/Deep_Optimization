"""Immutable diagnostic presets and host-side update scheduling.

These policies never change the objective, precision, optimizer, or DDP safety
protocol. Omitting a policy preserves the legacy checks=True/False behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from deepopt._diagnostics import CheckPhase, FiniteChecks, execution_check_scope


_CHECKS_ON = FiniteChecks(True, True, True, True)
_CHECKS_OFF = FiniteChecks(False, False, False, False)


class ExecutionMode(StrEnum):
    MAX_SPEED = "max_speed"
    BALANCED = "balanced"
    ANALYZE = "analyze"


@dataclass(frozen=True)
class DiagnosticPolicy:
    mode: ExecutionMode = ExecutionMode.BALANCED
    loss_finite_every: int | None = 100
    gradient_finite_every: int | None = 100
    inspect_weights: bool = False
    inspect_activations: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ExecutionMode(self.mode))
        for name in ("loss_finite_every", "gradient_finite_every"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        for name in ("inspect_weights", "inspect_activations"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.mode == ExecutionMode.MAX_SPEED and (self.loss_finite_every is not None
                or self.gradient_finite_every is not None or self.inspect_weights or self.inspect_activations):
            raise ValueError("max_speed disables passive diagnostics; use DiagnosticPolicy.for_mode('max_speed')")

    @classmethod
    def for_mode(cls, mode: ExecutionMode | str) -> DiagnosticPolicy:
        mode = ExecutionMode(mode)
        if mode == ExecutionMode.MAX_SPEED:
            return cls(mode, None, None)
        if mode == ExecutionMode.ANALYZE:
            return cls(mode, 1, 1, True, True)
        return cls(mode)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "mode": self.mode.value}

    @classmethod
    def from_dict(cls, options: Mapping[str, Any]) -> DiagnosticPolicy:
        """Apply the selected preset's defaults, then validate explicit fields."""
        defaults = cls.for_mode(options.get("mode", ExecutionMode.BALANCED)).as_dict()
        return cls(**{**defaults, **options})


class ModeController:
    """Schedule diagnostics by optimizer update, outside compiled forward graphs."""

    def __init__(self, execution_mode: ExecutionMode | str | None = None,
                 diagnostics: DiagnosticPolicy | Mapping[str, Any] | None = None,
                 mode_schedule: Mapping[int, ExecutionMode | str] | None = None) -> None:
        if execution_mode is not None and diagnostics is not None:
            raise ValueError("Choose execution_mode or diagnostics, not both")
        if isinstance(diagnostics, Mapping):
            diagnostics = DiagnosticPolicy.from_dict(diagnostics)
        if diagnostics is not None and not isinstance(diagnostics, DiagnosticPolicy):
            raise TypeError("diagnostics must be a DiagnosticPolicy or mapping")
        self.base_policy = diagnostics if execution_mode is None else DiagnosticPolicy.for_mode(execution_mode)
        self.policy = self.base_policy
        if mode_schedule is not None and not isinstance(mode_schedule, Mapping):
            raise TypeError("mode_schedule must map optimizer update numbers to execution modes")
        self.schedule: dict[int, DiagnosticPolicy] = {}
        for step, mode in (mode_schedule or {}).items():
            if type(step) is not int or step < 1:
                raise ValueError("Mode schedule steps must be positive integers")
            self.schedule[step] = DiagnosticPolicy.for_mode(mode)
        self.completed_updates = 0

    def configuration(self) -> dict[str, Any]:
        return {"policy": self.base_policy.as_dict() if self.base_policy else None,
                "mode_schedule": {str(step): policy.mode.value for step, policy in sorted(self.schedule.items())}}

    def prepare_update(self) -> bool:
        policy = self.base_policy
        for step, candidate in sorted(self.schedule.items()):
            if step > self.completed_updates + 1:
                break
            policy = candidate
        changed = self.policy != policy
        self.policy = policy
        return changed

    def permits(self, kind: str) -> bool:
        policy = self.policy
        if policy is None:
            return True
        if policy.mode == ExecutionMode.MAX_SPEED:
            return False
        update = self.completed_updates + 1
        if kind in ("loss", "report"):
            return policy.loss_finite_every is not None and update % policy.loss_finite_every == 0
        if kind == "gradients":
            return policy.gradient_finite_every is not None and update % policy.gradient_finite_every == 0
        if kind == "weights":
            return policy.inspect_weights
        if kind == "activation":
            return policy.inspect_activations
        return kind == "metrics" or policy.mode == ExecutionMode.ANALYZE

    def finite_checks(self, *, legacy_checks: bool = True) -> FiniteChecks:
        """Use the same optimizer-update cadence inside lower-level operations."""
        policy = self.policy
        if policy is None:
            return _CHECKS_ON if legacy_checks else _CHECKS_OFF
        if policy.mode == ExecutionMode.MAX_SPEED:
            return _CHECKS_OFF
        loss, gradients = self.permits("loss"), self.permits("gradients")
        return FiniteChecks(loss or policy.inspect_weights or policy.inspect_activations,
                            gradients, loss, self.permits("report"))

    @contextmanager
    def scope(self, name: str, *, legacy_checks: bool = True,
              phase: CheckPhase = "forward") -> Iterator[None]:
        with execution_check_scope(self.finite_checks(legacy_checks=legacy_checks), phase,
                                   self.permits if self.policy is not None else None):
            if self.policy is not None and self.policy.mode == ExecutionMode.ANALYZE:
                from torch.profiler import record_function
                with record_function(name):
                    yield
            else:
                yield
