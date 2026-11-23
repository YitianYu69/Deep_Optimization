"""Invocation-scoped numerical observation outside normal compiled execution.

Observers use an artifact's existing observed wrapper. They do not change
kernel selection, differentiation ownership or the ordinary wrapper source.
"""

from contextlib import contextmanager
from contextvars import ContextVar


_observer = ContextVar("deepopt_native_boundary_observer", default=None)


def current_native_boundary_observer():
    return _observer.get()


@contextmanager
def native_boundary_observation(observer):
    if not callable(getattr(observer, "run_artifact", None)):
        raise TypeError("A native boundary observer requires run_artifact(artifact, *args, phase=...)")
    token = _observer.set(observer)
    try:
        yield observer
    finally:
        _observer.reset(token)
