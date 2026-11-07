"""Context-local passive value checks shared by execution and computation."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal


CheckPhase = Literal["forward", "backward", "loss", "report"]


@dataclass(frozen=True)
class FiniteChecks:
    forward: bool
    backward: bool
    loss: bool
    report: bool


_active_checks: ContextVar[tuple[FiniteChecks, CheckPhase, Callable[[str], bool] | None] | None] = ContextVar(
    "deepopt_execution_checks", default=None)


def finite_checks_enabled(kind: CheckPhase | None = None) -> bool:
    active = _active_checks.get()
    return True if active is None else bool(getattr(active[0], kind or active[1]))


def has_execution_checks() -> bool:
    return _active_checks.get() is not None


def passive_checks_enabled(kind: str, *, collective: bool = False) -> bool:
    active = _active_checks.get()
    if active is None:
        return True
    # A missing gate is the legacy all-on/all-off policy.
    # Legacy check collections may differ by rank. Local opt-out must not
    # prevent a rank from joining an observer requested by another rank.
    return (collective or active[0].loss) if active[2] is None else active[2](kind)


@contextmanager
def execution_check_scope(checks: FiniteChecks, phase: CheckPhase,
                          permits: Callable[[str], bool] | None = None) -> Iterator[None]:
    token = _active_checks.set((checks, phase, permits))
    try:
        yield
    finally:
        _active_checks.reset(token)
