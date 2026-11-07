"""Thread/context-local quantization diagnostics, separate from numerical policy."""

from contextlib import contextmanager
from contextvars import ContextVar

from deepopt._diagnostics import finite_checks_enabled as diagnostic_finite_checks
from deepopt._diagnostics import has_execution_checks


_check_finite = ContextVar("deepopt_fp8_check_finite", default=None)


def finite_checks_enabled():
    override = _check_finite.get()
    return diagnostic_finite_checks() if override is None else override


def backward_finite_check_policy():
    override = _check_finite.get()
    if override is not None:
        return override
    # Unscoped forwards can defer to runtime.backward(); fully standalone
    # backward still defaults to strict checks at the quantization entry point.
    return diagnostic_finite_checks("backward") if has_execution_checks() else None


@contextmanager
def finite_check_scope(enabled):
    if type(enabled) is not bool:
        raise TypeError("check_finite must be a boolean")
    token = _check_finite.set(enabled)
    try:
        yield
    finally:
        _check_finite.reset(token)
