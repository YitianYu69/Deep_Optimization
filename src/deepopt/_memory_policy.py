"""Scoped allocation filling; allocator configuration remains a startup choice."""

from contextlib import contextmanager
from threading import Lock, get_ident

import torch


_lock = Lock()
_owners = {}
_previous = None


def resolve_fill(mode=None, explicit=None):
    if explicit is not None and type(explicit) is not bool:
        raise ValueError("fill_uninitialized_memory must be a boolean or None")
    return explicit if explicit is not None else False if mode == "max_speed" else None


def configured_fill(config):
    runtime = config.get("runtime", {})
    mode = runtime.get("execution_mode") or (runtime.get("diagnostics") or {}).get("mode")
    return resolve_fill(mode, runtime.get("fill_uninitialized_memory"))


def fill_spec():
    """Capture relevant effective behavior without reading device tensors."""
    return (torch.are_deterministic_algorithms_enabled(),
            bool(torch.are_deterministic_algorithms_enabled()
                 and torch.utils.deterministic.fill_uninitialized_memory))


@contextmanager
def capture_memory_scope(config):
    # An enclosing runtime owns mode schedules. A standalone capture resolves
    # its construction config before setup, rather than waiting for a runtime.
    with _lock:
        inherited = get_ident() in _owners or (bool(_owners) and torch._C._current_graph_task_id() >= 0)
    with allocation_fill_scope(None if inherited else configured_fill(config)) as value:
        yield value


@contextmanager
def allocation_fill_scope(enabled=None):
    """Restore process state and reject incompatible overlapping owned scopes.

    This protects DeepOpt scopes, not concurrent unscoped third-party changes
    to Torch's global flag. Matching scopes can overlap across threads.
    """
    global _previous
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("fill_uninitialized_memory must be a boolean or None")
    owner = get_ident()
    with _lock:
        value = torch.utils.deterministic.fill_uninitialized_memory if enabled is None else enabled
        if any(any(flag != value for flag in stack) for thread, stack in _owners.items() if thread != owner):
            raise RuntimeError("Incompatible overlapping allocation-fill policies; use separate processes")
        if not _owners:
            _previous = torch.utils.deterministic.fill_uninitialized_memory
        _owners.setdefault(owner, []).append(value)
        torch.utils.deterministic.fill_uninitialized_memory = value
    try:
        yield value
    finally:
        with _lock:
            stack = _owners[owner]
            stack.pop()
            if not stack:
                del _owners[owner]
            restore = (stack[-1] if stack else next(iter(_owners.values()))[-1] if _owners else _previous)
            torch.utils.deterministic.fill_uninitialized_memory = restore
            if not _owners:
                _previous = None
