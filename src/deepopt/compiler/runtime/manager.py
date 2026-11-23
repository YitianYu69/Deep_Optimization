"""Live private compiled-graph runtime owners and metadata-only diagnostics."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
import threading
import weakref

from .packed_offload import PackedOffloadRuntime


_managers = weakref.WeakSet()
_lock = threading.RLock()
_owners = weakref.WeakKeyDictionary()
_current = ContextVar("deepopt_compiled_runtime_scope", default=None)
_scopes = count(1)


class CompiledGraphRuntimeManager(PackedOffloadRuntime):
    def __init__(self, policy, *, backend=None, start_worker=True):
        super().__init__(policy, backend=backend, start_worker=start_worker)
        with _lock:
            _managers.add(self)

    def begin_forward(self, plan, compute_stream=None, execution_scope=None):
        current = _current.get()
        if execution_scope is None and current is not None and current[0] is self:
            execution_scope = current[1]
        return super().begin_forward(plan, compute_stream=compute_stream, execution_scope=execution_scope)


@contextmanager
def compiled_runtime_scope(policy, owner):
    """TrainingRuntime-owned scope; regional pools share one declared budget."""
    if not policy.enabled:
        yield None
        return
    current = _current.get()
    if current is not None and current[2]() is owner:
        if current[0].policy != policy:
            raise ValueError("Nested compiled runtime scope changed the offload policy")
        yield current[0]
        return
    with _lock:
        choices = _owners.setdefault(owner, {})
        manager = choices.get(policy)
        if manager is None:
            manager = choices[policy] = CompiledGraphRuntimeManager(policy)
            # A weak-key registry alone is insufficient if a live pump thread
            # owns the manager. The finalizer captures only the manager.
            weakref.finalize(owner, _close_owner_manager, manager)
    token = _current.set((manager, ("runtime", next(_scopes)), weakref.ref(owner)))
    try:
        yield manager
    finally:
        _current.reset(token)


def current_runtime_manager(policy):
    current = _current.get()
    if current is None:
        return None
    if current[0].policy != policy:
        raise ValueError("Compiled graph and live runtime own different packed-offload policies")
    return current[0]


def _close_owner_manager(manager):
    with manager._lock:
        manager._owner_gone = True
        manager._maybe_close_owner()


@contextmanager
def compiled_forward_scope(plans, default_manager):
    """Generated outer wrapper: reserve all regional saves before producers."""
    plans = tuple(plans)
    if not plans:
        yield None
        return
    current = _current.get()
    manager = current_runtime_manager(plans[0].policy) if current is not None else default_manager
    if manager is None:
        raise ValueError("Compiled offload forward has no runtime owner")
    scope_id = current[1] if current is not None else ("compiled", next(_scopes))
    token = _current.set((manager, scope_id, current[2] if current is not None else lambda: None))
    try:
        with manager.reserve_scope(plans, scope_id):
            yield manager
    finally:
        _current.reset(token)


def packed_offload_runtime_snapshot():
    """No tensor inspection, transfers or synchronization; call outside timing."""
    with _lock:
        snapshots = tuple(manager.snapshot() for manager in tuple(_managers))
    totals = {}
    for section in ("counters", "host", "device", "lifecycle"):
        values = Counter()
        seen_pinned_domains = set()
        for snapshot in snapshots:
            fields = snapshot[section]
            domain = fields.get("pinned_allocator_domain_id") if section == "host" else None
            duplicate_domain = domain is not None and domain in seen_pinned_domains
            if domain is not None:
                seen_pinned_domains.add(domain)
            values.update({key: value for key, value in fields.items()
                           if type(value) in (int, float) and not key.endswith("_id")
                           and not (duplicate_domain and key.startswith("pinned_"))})
        totals[section] = dict(values)
    totals["lifecycle"]["quiescent"] = all(snapshot["lifecycle"]["quiescent"] for snapshot in snapshots)
    return dict(schema_version=1, managers=snapshots, totals=totals, metadata_only=True,
                frozen=all(snapshot["profile_frozen"] for snapshot in snapshots),
                accounting="physical pinned allocator domains deduplicated; other regional peaks are upper sums, not simultaneous")


def _owned_managers(owner=None):
    with _lock:
        return tuple(_managers) if owner is None else tuple(_owners.get(owner, {}).values())


def drain_packed_offload_runtime(owner=None):
    """Drain only work owned by live offload managers; propagate failures."""
    managers = _owned_managers(owner)
    first = None
    for manager in managers:
        try:
            manager.drain()
        except BaseException as error:
            first = first or error
    if first is not None:
        raise first
    return packed_offload_runtime_snapshot()


def quiesce_packed_offload_runtime(owner=None):
    return drain_packed_offload_runtime(owner)


def close_packed_offload_runtime(owner=None):
    first = None
    for manager in _owned_managers(owner):
        try:
            manager.close()
        except BaseException as error:
            first = first or error
    if first is not None:
        raise first
    return packed_offload_runtime_snapshot()


def freeze_packed_offload_profiles(owner=None):
    """Warmup boundary only: collect completed GPU costs and stop learning."""
    for manager in _owned_managers(owner):
        manager.freeze_profiles()
    return packed_offload_runtime_snapshot()


def invalidate_packed_offload_runtime(owner):
    """Abort owned graph frames without recycling their graph/DMA leases."""
    first = None
    for manager in _owned_managers(owner):
        for frame in tuple(manager._active_frames):
            frame.abort()
        try:
            manager.drain()
        except BaseException as error:
            first = first or error
        manager._event("runtime_invalidated")
    if first is not None:
        raise first
