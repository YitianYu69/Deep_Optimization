"""Private packed-save transport and autograd leases.

Only already encoded, immutable codec leaves cross the link.  Events and tensors
belong to a live invocation, never to a compiler/cache descriptor.  The explicit
CPU backend below is a state-machine testing aid, not an enabled CPU execution
route.
"""

from collections import Counter, deque
from bisect import bisect_right
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import count
from math import prod
import threading
import time
import weakref

import torch
from torch.multiprocessing.reductions import StorageWeakRef


_managed_backward = ContextVar("deepopt_packed_offload_backward", default=None)
_invocations = count(1)
_tasks = count(1)
_HOOK_ERROR = "Packed activation offload cannot nest external saved-tensor hooks"


class OffloadAdmissionError(RuntimeError):
    """A declared managed capacity cannot admit this invocation/consumer."""


class OffloadProtocolError(RuntimeError):
    """Generated actions or runtime operands contradict the frozen plan."""


def _dtype(value):
    result = value if isinstance(value, torch.dtype) else getattr(torch, str(value).removeprefix("torch."), None)
    if result not in (torch.int32, torch.float32):
        raise OffloadProtocolError("Packed transport accepts int32 payload and FP32 metadata only")
    return result


def _component_bytes(component):
    return prod(component.shape) * 4


def _storage(tensor):
    return str(tensor.device), tensor.untyped_storage()._cdata


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError as error:
        raise OffloadProtocolError("Packed offload requires ordinary versioned tensors") from error


def _check_hooks():
    # Public guard detects an existing hook pair without replacing it.  Exit
    # before installing our own narrowly scoped adapter.
    with torch.autograd.graph.disable_saved_tensors_hooks(_HOOK_ERROR):
        pass


class _CpuEvent:
    def __init__(self, complete=True):
        self._complete = threading.Event()
        if complete:
            self._complete.set()

    def query(self):
        return self._complete.is_set()

    def synchronize(self):
        self._complete.wait()

    def complete(self):
        self._complete.set()


class TestingTransportBackend:
    """Explicit, synchronous CPU-copy backend for lifecycle tests only."""

    testing_only = True
    device = torch.device("cpu")

    def __init__(self, *, auto_complete=True):
        self.auto_complete = auto_complete
        self.events = []
        self.allocations = []
        self.external_allocated_bytes = 0
        self._allocated = 0

    def check(self, device):
        if torch.device(device).type != "cpu":
            raise OffloadProtocolError("The testing backend only accepts explicit CPU fixtures")

    def event(self, stream=None, *, timing=False):
        result = _CpuEvent(self.auto_complete)
        self.events.append(result)
        return result

    def copy(self, destinations, sources, *, direction, dependency=None, chunk_bytes=0, measure=False):
        for destination, source in zip(destinations, sources):
            destination.copy_(source)
        return self.event()

    def wait(self, event):
        # Simulate a submitted device wait, not a host synchronization. Copies
        # are synchronous in this explicit test backend; events model leases.
        return None

    def synchronize_consumer(self):
        return None

    def allocate(self, component):
        result = torch.empty(component.shape, dtype=_dtype(component.dtype), device="cpu")
        self.allocations.append(weakref.ref(result))
        size = result.numel() * result.element_size()
        self._allocated += size
        weakref.finalize(result, self._release_allocation, size)
        return result

    def _release_allocation(self, size):
        self._allocated -= size

    def allocated_bytes(self):
        return self.external_allocated_bytes + self._allocated

    def host(self, component, *, pinned):
        return torch.empty(component.shape, dtype=_dtype(component.dtype), device="cpu")

    def host_arena(self, capacity, *, pinned):
        # Explicit CPU state-machine aid; never evidence of physical pinning.
        return torch.empty(capacity, dtype=torch.uint8, device="cpu")


class _CudaEventLease:
    """The actual event cannot be reused while a published handle exists."""

    def __init__(self, backend, event, timing=False):
        self.backend, self.event, self.timing = backend, event, timing
        self.start = None

    def query(self):
        return self.event.query()

    def synchronize(self):
        self.event.synchronize()

    def elapsed_ms(self):
        if self.start is not None and self.query() and self.start.query():
            return self.start.event.elapsed_time(self.event)
        return None

    def __del__(self):
        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.retire_event(self.event, self.timing)


class _CudaBackend:
    testing_only = False
    pinned_arena_domain = "Torch_CUDA_pinned_allocator_v2"

    def __init__(self, device):
        self.device = torch.device(device)
        self.check(device)
        with torch.cuda.device(self.device):
            self.d2h = torch.cuda.Stream(device=self.device)
            self.h2d = torch.cuda.Stream(device=self.device)
        self._event_lock = threading.RLock()
        self._events_free, self._events_pending = [], []
        self._events_allocated = 0
        self.event_capacity = 4096

    def check(self, device):
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise OffloadProtocolError("Enabled packed offload requires an available CUDA device")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise OffloadProtocolError("Packed activation offload does not support CUDA Graph capture")

    def event(self, stream=None, *, timing=False):
        with self._event_lock, torch.cuda.device(self.device):
            for event, kind in tuple(self._events_pending):
                if event.query():
                    self._events_pending.remove((event, kind))
                    self._events_free.append((event, kind))
            index = next((i for i, (_, kind) in enumerate(self._events_free) if kind == timing), None)
            if index is not None:
                event, _ = self._events_free.pop(index)
            else:
                if self._events_allocated >= self.event_capacity:
                    raise OffloadAdmissionError("Bounded packed transport event capacity exceeded")
                event = torch.cuda.Event(enable_timing=timing)
                self._events_allocated += 1
            event.record(stream or torch.cuda.current_stream(self.device))
        return _CudaEventLease(self, event, timing)

    def retire_event(self, event, timing):
        with self._event_lock:
            self._events_pending.append((event, timing))

    def copy(self, destinations, sources, *, direction, dependency=None, chunk_bytes=0, measure=False):
        stream = self.d2h if direction == "d2h" else self.h2d
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            if dependency is not None:
                stream.wait_event(dependency.event)
            start = self.event(stream, timing=True) if measure else None
            # Group-aligned bounded commands allow an urgent transfer to be
            # submitted between bundles. They do not claim panel consumption.
            elements = max(256, chunk_bytes // 4)
            try:
                for destination, source in zip(destinations, sources):
                    target, origin = destination.view(-1), source.view(-1)
                    for offset in range(0, source.numel(), elements):
                        target[offset:offset + elements].copy_(origin[offset:offset + elements], non_blocking=True)
                    if source.device.type == "cuda":
                        source.record_stream(stream)
                    if destination.device.type == "cuda":
                        destination.record_stream(stream)
                ready = self.event(stream, timing=measure)
                ready.start = start
                return ready
            except BaseException:
                stream.synchronize()
                raise

    def wait(self, event):
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).wait_event(event.event)

    def allocate(self, component):
        # Strict deterministic allocation can enqueue a fill kernel. Order it
        # on the copy stream; allocator lifetime tracking alone is insufficient.
        with torch.cuda.device(self.device), torch.cuda.stream(self.h2d):
            return torch.empty(component.shape, dtype=_dtype(component.dtype), device=self.device)

    def allocated_bytes(self):
        return torch.cuda.memory_allocated(self.device)

    def synchronize_consumer(self):
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).synchronize()

    def host(self, component, *, pinned):
        return torch.empty(component.shape, dtype=_dtype(component.dtype), device="cpu", pin_memory=pinned)

    def host_arena(self, capacity, *, pinned):
        from .pinned_arena import PinnedArenaAllocation

        statistics = getattr(torch.cuda.memory, "host_memory_stats", None)
        if not pinned or statistics is None:
            raise OffloadProtocolError("Physical pinned capacity requires Torch host allocator statistics")
        before = statistics()
        names = ("active_bytes.current", "allocated_bytes.current")
        if not isinstance(before, dict) or any(type(before.get(name)) is not int for name in names):
            raise OffloadProtocolError("Torch host allocator cannot verify physical pinned capacity")
        try:
            tensor = torch.empty(capacity, dtype=torch.uint8, device="cpu", pin_memory=True)
        except BaseException as error:
            try:
                after = statistics()
                verified = all(type(after.get(name)) is int for name in names)
                charge = max(0, after.get(names[1], before[names[1]]) - before[names[1]])
            except BaseException:
                charge, verified = capacity, False
            # Keep the original failure. A cache allocation left behind by a
            # failed attempt is still offload-induced physical capacity.
            error._deepopt_pinned_unowned_bytes = charge
            error._deepopt_pinned_failure_verified = verified
            raise
        try:
            after = statistics()
        except BaseException:
            return PinnedArenaAllocation(tensor, capacity, False, "post_allocation_statistics_unavailable")
        if not isinstance(after, dict) or any(type(after.get(name)) is not int for name in names):
            return PinnedArenaAllocation(tensor, capacity, False, "post_allocation_statistics_malformed")
        active = after.get(names[0], 0) - before[names[0]]
        allocated = after.get(names[1], 0) - before[names[1]]
        # An unexpected larger allocator block, or concurrent outside-owner
        # activity, invalidates the strict claim. Return ownership to the pool
        # even on disagreement; it must keep this capacity charged.
        verified = active == capacity and allocated <= capacity
        return PinnedArenaAllocation(tensor, max(capacity, active, allocated), verified,
            "power_of_two_arena_reconciled_with_Torch_pinned_allocator_counters")


class _HostLease:
    def __init__(self, pool, tensors, keys, size):
        self.pool, self.tensors, self.keys, self.bytes = pool, tuple(tensors), tuple(keys), size
        self.released = False

    def release(self):
        if not self.released:
            self.released = True
            self.pool.release(self)
            self.tensors = ()


class PinnedOffloadPool:
    """Bound active *and cached* host storage; reserve groups atomically."""

    def __init__(self, budget, backend, *, pinned=True):
        if type(budget) is not int or budget < 0:
            raise ValueError("Host pool budget must be a nonnegative integer")
        self.budget, self.backend, self.pinned = budget, backend, pinned
        self.active = self.cached = self.peak = 0
        self.allocations = 0
        self._free = []
        self._lock = threading.RLock()

    def acquire(self, components):
        components = tuple(components)
        size = sum(_component_bytes(item) for item in components)
        with self._lock:
            if self.active + size > self.budget:
                return None
            available = list(self._free)
            selected, missing = [], []
            for component in components:
                key = tuple(component.shape), _dtype(component.dtype)
                index = next((i for i, (other, _) in enumerate(available) if other == key), None)
                if index is None:
                    selected.append((key, None))
                    missing.append(component)
                else:
                    selected.append(available.pop(index))
            # Evict free incompatible capacities before allocating. No pool
            # entry can be active and free at once.
            needed = sum(_component_bytes(item) for item in missing)
            retained = sum(t.numel() * t.element_size() for _, t in available)
            while self.active + size + retained > self.budget and available:
                _, tensor = available.pop(0)
                retained -= tensor.numel() * tensor.element_size()
            tensor = None
            self._free = available
            self.cached = retained
            self.active += size
            tensors = []
            try:
                pending = iter(missing)
                for key, tensor in selected:
                    if tensor is None:
                        tensor = self.backend.host(next(pending), pinned=self.pinned)
                        self.allocations += 1
                    tensors.append(tensor)
            except BaseException:
                self.active -= size
                tensors.clear()
                for key, tensor in selected:
                    if tensor is not None:
                        self._free.append((key, tensor))
                        self.cached += tensor.numel() * tensor.element_size()
                tensor = None
                raise
            self.peak = max(self.peak, self.active + self.cached)
            assert self.active + self.cached <= self.budget and needed <= size
            return _HostLease(self, tensors, [key for key, _ in selected], size)

    def release(self, lease):
        with self._lock:
            if lease.bytes > self.active:
                raise OffloadProtocolError("Host lease was released more than once")
            self.active -= lease.bytes
            self._free.extend(zip(lease.keys, lease.tensors))
            self.cached += lease.bytes

    def trim(self):
        with self._lock:
            self._free.clear()
            self.cached = 0

    def snapshot(self):
        with self._lock:
            return dict(active_bytes=self.active, cached_bytes=self.cached, peak_bytes=self.peak,
                        budget_bytes=self.budget, allocations=self.allocations, pinned=self.pinned)


@dataclass
class _D2H:
    record: object
    producer_event: object
    ready: object
    stage: object = None
    enqueue_ms: float = 0.0


@dataclass
class _Staging:
    record: object
    tensors: tuple
    ready: object
    host_stage: object = None
    taken: bool = False
    dma_done: bool = False
    retired: bool = False
    completion: object = None
    records: tuple = ()
    bytes: int = 0
    enqueue_ms: float = 0.0
    prefetched: bool = False
    consumed: bool = False
    waited_consumer: str | None = None
    last_consumer_event: object = None

    def __post_init__(self):
        self.records = self.records or (self.record,)
        self.bytes = self.bytes or self.record.bundle.bytes


class _BundleRecord:
    def __init__(self, frame, bundle, host):
        self.frame = weakref.ref(frame)
        self.manager, self.bundle, self.host = frame.manager, bundle, host
        self.generation = frame.invocation
        self.plan_signature = frame.plan.signature
        self.refs = Counter(forward=1)
        self.ready = None
        self.host_ready = False
        self.last_forward_use = False
        self.published = False
        self.closed = False
        self.versions = None

    def acquire(self, owner):
        with self.manager._lock:
            if self.closed:
                raise OffloadProtocolError("Saved bundle lease is already retired")
            self.refs[owner] += 1

    def release(self, owner):
        with self.manager._lock:
            if self.refs[owner] <= 0:
                raise OffloadProtocolError("Saved bundle lease underflow")
            self.refs[owner] -= 1
            if not any(self.refs.values()):
                self.closed = True
                self.host.release()
                self.ready = None
                self.versions = None
                self.manager._records.discard(self)
                frame = self.frame()
                if frame is not None:
                    frame._maybe_retired()
            self.manager._condition.notify_all()

    def validate(self, *, require_ready=True):
        validate_lease = getattr(self.host.pool, "validate_lease", None)
        if validate_lease is not None:
            validate_lease(self.host)
        if require_ready and not self.host_ready:
            raise OffloadProtocolError("CPU saved bytes are not ready")
        if self.versions is None:
            raise OffloadProtocolError("CPU saved versions have not been published")
        if any(_version(value) != version for value, version in zip(self.host.tensors, self.versions)):
            raise RuntimeError("A packed CPU snapshot needed for gradient computation was modified by an inplace operation")

    def wait_host(self):
        while not self.host_ready:
            self.manager._check_error()
            ready = self.ready
            if ready is None:
                raise OffloadProtocolError("Packed producer has not published a recorded D2H event")
            ready.synchronize()  # Own dependency only; never device-wide.
            self.manager.poll()
        self.validate()


class _ResidentToken:
    def __init__(self, tensor):
        self.value = tensor.detach()
        self.original = weakref.ref(tensor)
        self.version = _version(tensor)
        self.storage = _storage(self.value)
        self.metadata = tuple(self.value.shape), tuple(self.value.stride()), self.value.dtype, self.value.device

    def unpack(self):
        if (_version(self.value) != self.version or _storage(self.value) != self.storage
                or (tuple(self.value.shape), tuple(self.value.stride()), self.value.dtype, self.value.device) != self.metadata):
            raise RuntimeError("A tensor needed for gradient computation was modified by an inplace operation")
        original = self.original()
        if original is not None:
            if _version(original) != self.version:
                raise RuntimeError("A tensor needed for gradient computation was modified by an inplace operation")
            # Stock SavedVariable follows unversioned .data rebinding of a
            # still-live input/parameter. Keep that behavior, while the detached
            # owner avoids a strong reference cycle through a saved output.
            return original
        return self.value


class _OffloadedToken:
    def __init__(self, record, component):
        self.record, self.component, self.generation = record, component, record.generation
        record.acquire("graph")

    def unpack(self):
        record = self.record
        if record.generation != self.generation or record.closed:
            raise OffloadProtocolError("Stale packed saved-token generation")
        scope = _managed_backward.get()
        if scope is not None and scope.frame() is record.frame():
            scope.acquire(record)
            return record.host.tensors[self.component]
        # Autograd inspection must not expose pooled mutable storage or read
        # unfinished DMA. An external holder owns an ordinary independent copy.
        record.wait_host()
        value = record.host.tensors[self.component]
        result = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=False)
        result.copy_(value)
        return result

    def __del__(self):
        record = getattr(self, "record", None)
        if record is not None:
            record.release("graph")


class _BackwardScope:
    def __init__(self, frame):
        self.frame = weakref.ref(frame)
        self.records = set()
        self.task = None

    def acquire(self, record):
        if record not in self.records:
            record.acquire("task")
            self.records.add(record)

    def release(self):
        for record in tuple(self.records):
            record.release("task")
        self.records.clear()


class ForwardFrame:
    def __init__(self, manager, plan, hosts, *, fallback=False, execution_scope=None):
        self.manager, self.plan = manager, plan
        self.invocation = next(_invocations)
        self.execution_scope = execution_scope if execution_scope is not None else ("direct", self.invocation)
        self.resident_fallback = bool(fallback or plan.bundles and not hosts)
        self.records = {bundle.bundle_id: _BundleRecord(self, bundle, hosts[bundle.bundle_id])
                        for bundle in plan.bundles if bundle.bundle_id in hosts} if not fallback else {}
        self._slots = manager._index(plan)["slots"]
        self._host_ids = {id(value): (record, index) for record in self.records.values()
                          for index, value in enumerate(record.host.tensors)}
        self._resident = {}
        self._exported = set()
        self._closed = self.finished = self.aborted = False
        self._tokens_installed = False
        manager._records.update(self.records.values())

    @contextmanager
    def saved_token_scope(self):
        _check_hooks()
        if self._tokens_installed:
            raise OffloadProtocolError("A forward frame cannot install its saved-token adapter twice")
        self._tokens_installed = True

        def pack(tensor):
            pair = self._host_ids.get(id(tensor))
            return _OffloadedToken(*pair) if pair is not None else _ResidentToken(tensor)

        def unpack(token):
            return token.unpack()

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                yield self
        finally:
            self._close_forward()

    def _close_forward(self):
        with self.manager._lock:
            if not self._closed:
                self._closed = True
                self._host_ids.clear()
                self._resident.clear()
                for record in self.records.values():
                    record.release("forward")
                self._maybe_retired()
                self.manager._maybe_close_owner()

    def _maybe_retired(self):
        if self._closed and all(record.closed for record in self.records.values()):
            self.manager._active_frames.discard(self)

    def pack_ready(self, bundle_id, components):
        self.manager._check_error()
        bundle = self.manager._bundle(self.plan, bundle_id)
        components = tuple(components)
        self.manager._validate_components(bundle, components)
        if bundle_id not in self.records:
            self._resident[bundle_id] = components
            return
        record = self.records[bundle_id]
        if record.published or self.finished or self.aborted:
            raise OffloadProtocolError("Packed producer may publish exactly once before forward completion")
        self.manager._publish(record, components)

    def last_forward_use(self, bundle_id):
        if bundle_id not in self.records:
            return
        record = self.records[bundle_id]
        if not record.published or record.last_forward_use:
            raise OffloadProtocolError("Invalid packed last-forward-use action")
        record.last_forward_use = True
        # No CUDA source tensor is retained here. record_stream protects its
        # allocator lifetime until the already submitted D2H reader finishes.

    def export_slot(self, saved_index, resident_tensor=None):
        pair = self._slots.get(saved_index)
        if pair is None:
            return resident_tensor
        bundle_id, component = pair
        if bundle_id not in self.records:
            return self._resident[bundle_id][component]
        record = self.records[bundle_id]
        if not record.published:
            raise OffloadProtocolError("Saved export precedes its packed producer")
        self._exported.add(saved_index)
        return record.host.tensors[component]

    def finish_forward(self):
        if self.aborted:
            raise OffloadProtocolError("An aborted frame cannot finish")
        if not self.resident_fallback and any(not r.published or not r.last_forward_use for r in self.records.values()):
            raise OffloadProtocolError("Forward did not publish and retire every selected bundle")
        self.finished = True

    def abort(self, error=None):
        self.aborted = True
        self.manager._event("forward_aborted", invocation=self.invocation,
                            error=None if error is None else type(error).__name__)
        self._close_forward()

    @contextmanager
    def backward_scope(self):
        if self.aborted or not self.finished:
            raise OffloadProtocolError("Backward requires a successfully completed forward")
        scope = _BackwardScope(self)
        # Bind before ctx.saved_tensors is unpacked/cleared. Closed records
        # indicate released graph tokens, not a permission to resurrect saves.
        try:
            with self.manager._lock:
                for record in self.records.values():
                    scope.acquire(record)
        except BaseException:
            scope.release()
            raise
        token = _managed_backward.set(scope)
        primary = None
        try:
            yield scope
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                if scope.task is not None:
                    scope.task.finish(error=primary)
            finally:
                _managed_backward.reset(token)
                scope.release()

    def begin_backward(self, saved_exports, *, seeds=()):
        scope = _managed_backward.get()
        if scope is None or scope.frame() is not self or scope.task is not None:
            raise OffloadProtocolError("Begin backward inside exactly one matching backward_scope")
        task = BackwardTask(self, scope, saved_exports)
        scope.task = task
        return task


class BackwardTask:
    def __init__(self, frame, scope, saved):
        self.frame, self.manager, self.plan, self.scope = frame, frame.manager, frame.plan, scope
        self._index = self.manager._index(self.plan)
        self._active_consumers = {item.consumer_id: replace(item,
            bundle_ids=tuple(bid for bid in item.bundle_ids if bid in frame.records),
            bytes=sum(self.manager._bundle(self.plan, bid).bytes for bid in item.bundle_ids if bid in frame.records))
            for item in self.plan.consumers}
        self.task_id = next(_tasks)
        self.thread_id = threading.get_ident()
        self._saved = list(saved)
        self._stages = {}
        self._requests = {}
        self._progress = []
        self._panel_stages = []
        self.current_boundary = None
        self.finished = False
        self._consumer = None
        self._cost_previous = (0, self.manager.backend.event(timing=True)) if self.manager.learning else None
        if not frame.resident_fallback:
            for index, (bundle_id, component) in frame._slots.items():
                if bundle_id not in frame.records:
                    continue
                if index >= len(self._saved) or self._saved[index] is not frame.records[bundle_id].host.tensors[component]:
                    raise OffloadProtocolError("Saved CPU export does not belong to this invocation")
        with self.manager._lock:
            if len(self.manager._active_tasks) >= self.plan.policy.max_backward_tasks:
                raise OffloadAdmissionError("Maximum live packed backward tasks exceeded")
            self.manager._active_tasks.add(self)

    def _consumer_for(self, consumer):
        match = (self._index["consumer_by_id"].get(consumer) if isinstance(consumer, str)
                 else self._index["consumer_by_boundary"].get(consumer))
        if match is None:
            raise OffloadProtocolError(f"Unknown packed consumer {consumer!r}")
        return match

    def advance(self, boundary_id):
        with self.manager._lock:
            self.manager._check_error()
            if self.finished:
                raise OffloadProtocolError("Backward task is finished")
            if self.current_boundary is not None and boundary_id < self.current_boundary:
                raise OffloadProtocolError("Packed backward boundaries must advance monotonically")
            self.current_boundary = boundary_id
            self._consumer = self._index["consumer_by_boundary"].get(boundary_id)
            self.manager.poll()
            if self.manager.learning:
                marker = self.manager.backend.event(timing=True)
                if self._cost_previous is not None and boundary_id > self._cost_previous[0]:
                    previous, event = self._cost_previous
                    self.manager._compute_jobs.append((self.plan.signature, previous, boundary_id, event, marker))
                self._cost_previous = boundary_id, marker
            if not self.frame.resident_fallback and self.plan.policy.policy == "memory_first_overlap":
                # Recording on the caller stream is not completion. Only query
                # success lets the pump reserve a future destination.
                ready = self._index["requests_by_boundary"].get(boundary_id, ())
                if not self._index["has_requests"]:
                    position = bisect_right(self._index["boundaries"], boundary_id)
                    future = self._index["ordered_consumers"][position] if position < len(self._index["boundaries"]) else None
                    pairs = (tuple((bid, future.boundary) for bid in future.bundle_ids)
                             if future is not None and future.panel is None and not self.plan.policy.physical_host_budget else ())
                else:
                    pairs = tuple((request.bundle_id, request.consumer_boundary) for request in ready)
                pairs = tuple(pair for pair in pairs if pair[0] in self.frame.records and pair[0] not in self._stages)
                if pairs:
                    gate = self.manager.backend.event()
                    self._progress.append(gate)
                    for bundle_id, deadline in pairs:
                        self._requests.setdefault(bundle_id, (gate, deadline))
                self.manager._wake()

    def _ensure_consumer(self, consumer):
        consumer = self._active_consumers[consumer.consumer_id]
        if not consumer.bundle_ids:
            return
        if getattr(consumer, "panel", None) is not None:
            raise OffloadProtocolError("Named panel consumers cannot use a whole-bundle restore")
        required = set(consumer.bundle_ids)
        if self.frame.resident_fallback:
            return
        self.manager._admit_consumer(self, consumer)
        for bundle_id in required:
            self._requests.pop(bundle_id, None)

    def take(self, bundle_id, consumer_id=None):
        with self.manager._lock:
            self.manager._check_error()
            if self.finished:
                raise OffloadProtocolError("Backward task is finished")
            consumer = self._consumer_for(consumer_id) if consumer_id is not None else self._consumer
            if consumer is None or bundle_id not in consumer.bundle_ids:
                raise OffloadProtocolError("Bundle read is not declared by the current consumer")
            self._ensure_consumer(consumer)
            if bundle_id not in self.frame.records:
                bundle = self.manager._bundle(self.plan, bundle_id)
                return tuple(self._saved[item.saved_index] for item in bundle.components)
            staging = self._stages[bundle_id]
            staging.record.validate(require_ready=bool(self.plan.policy.host_pageable_bytes))
            if staging.waited_consumer != consumer.consumer_id:
                self.manager.backend.wait(staging.ready)
                self.manager.counters["consumer_bundle_reads"] += 1
                if staging.prefetched and not staging.consumed:
                    self.manager.counters["prefetch_consumed"] += 1
                    if staging.dma_done:
                        self.manager.counters["prefetch_ready_observed_on_use"] += 1
                elif staging.consumed:
                    self.manager.counters["shared_restore_reuses"] += 1
                staging.waited_consumer = consumer.consumer_id
            staging.consumed = True
            staging.taken = True
            return staging.tensors

    def resolve(self, saved_index):
        if self.finished:
            raise OffloadProtocolError("Backward task is finished")
        pair = self.frame._slots.get(saved_index)
        if pair is None or pair[0] not in self.frame.records:
            return self._saved[saved_index]
        bundle_id, component = pair
        return self.take(bundle_id)[component]

    def release_after(self, bundle_id, consumer_id=None):
        consumer = self._consumer_for(consumer_id) if consumer_id is not None else self._consumer
        if consumer is None or bundle_id not in consumer.bundle_ids:
            raise OffloadProtocolError("Undeclared packed release")
        if bundle_id not in self.frame.records:
            return
        with self.manager._lock:
            stage = self._stages.get(bundle_id)
            if stage is None or not stage.taken:
                raise OffloadProtocolError("Packed release requires an acquired consumer lease")
            position = bisect_right(self._index["boundaries"], consumer.boundary)
            near = self._index["ordered_consumers"][position:position + 2]
            if (self.plan.policy.mode == "predictive" and any(bundle_id in item.bundle_ids for item in near)
                    and self.manager.staging_bytes <= self.manager._staging_limit_bytes):
                # Retire against the final numerical reader if pressure later
                # evicts this retained immutable copy. The original H2D event
                # alone does not cover a previously enqueued GPU consumer.
                stage.last_consumer_event = self.manager.backend.event()
                stage.taken = False
                self.manager.counters["short_reuse_retained"] += 1
                return
            self._stages.pop(bundle_id)
            self.manager._retire(stage)

    def run_panel_consumer(self, consumer_id, native_callable, nonpacked_args):
        consumer = self._consumer_for(consumer_id)
        panel = getattr(consumer, "panel", None)
        adapter = getattr(native_callable, "packed_offload_panel", None)
        if not any(bid in self.frame.records for bid in consumer.bundle_ids):
            args = tuple(nonpacked_args) + tuple(self._saved[component.saved_index]
                for bid in consumer.bundle_ids for component in self.manager._bundle(self.plan, bid).components)
            return native_callable(*args)
        if panel is None or adapter is None or len(nonpacked_args) != 1 or len(consumer.bundle_ids) != 3:
            raise OffloadProtocolError("A proven named three-bundle panel consumer is required")
        if not self.manager.backend.testing_only:
            from ..autodiff.swiglu_fusion import SwiGLUPackedOffloadPanel

            if type(adapter) is not SwiGLUPackedOffloadPanel:
                raise OffloadProtocolError("Untrusted packed-panel numerical adapter")
        names = tuple(component.name for bid in consumer.bundle_ids
                      for component in self.manager._bundle(self.plan, bid).components)
        if (adapter.kind != panel.kind or tuple(adapter.shape) != panel.shape or adapter.group_size != 1024
                or adapter.bits != 2 or names != panel.component_names):
            raise OffloadProtocolError("Packed-panel adapter contradicts the selected consumer ABI")
        records = tuple(self.frame.records[bid] for bid in consumer.bundle_ids)
        total_groups = records[0].bundle.components[0].shape[0]
        if any(record.bundle.components[0].shape[0] != total_groups for record in records):
            raise OffloadProtocolError("Packed-panel operand groups differ")
        elements = prod(panel.shape)
        if total_groups != (elements + 1023) // 1024:
            raise OffloadProtocolError("Packed-panel logical geometry differs from its encoded groups")
        with self.manager._lock:
            self.manager._discard_prefetches()
            self.manager._target_allows(elements * 4 + min(total_groups, panel.groups_per_panel) * 792, demand=True)
        outputs = adapter.allocate_outputs(nonpacked_args[0])
        for start in range(0, total_groups, panel.groups_per_panel):
            groups = min(panel.groups_per_panel, total_groups - start)
            with self.manager._lock:
                stage = self.manager._restore_panel(self, records, start, groups)
                self._panel_stages.append(stage)
                self.manager.backend.wait(stage.ready)
                stage.taken = True
            primary = None
            try:
                triplets = tuple(stage.tensors[index:index + 3] for index in (0, 3, 6))
                adapter.run(nonpacked_args[0], outputs, triplets, start * 1024,
                            min(groups * 1024, elements - start * 1024))
            except BaseException as error:
                primary = error
                raise
            finally:
                # A Python local must not retain the previous window after
                # completion returns its managed credits to the allocator.
                if "triplets" in locals():
                    del triplets
                with self.manager._lock:
                    try:
                        self.manager._retire(stage, suppress_errors=primary is not None)
                    finally:
                        self._panel_stages.remove(stage)
            self.manager.counters["panels_consumed"] += 1
        return outputs

    def finish(self, error=None):
        with self.manager._lock:
            if self.finished:
                return
            self.finished = True
            secondary = None
            try:
                self._requests.clear()
                for stage in (*self._stages.values(), *self._panel_stages):
                    try:
                        self.manager._retire(stage, suppress_errors=error is not None)
                    except BaseException as cleanup:
                        secondary = secondary or cleanup
                if self.manager.learning and self._cost_previous is not None:
                    previous, event = self._cost_previous
                    try:
                        marker = self.manager.backend.event(timing=True)
                        self.manager._compute_jobs.append((self.plan.signature, previous, previous + 1, event, marker))
                    except BaseException as cleanup:
                        self.manager._cleanup_error(cleanup, "cost_marker")
                        secondary = secondary or cleanup
            finally:
                self._panel_stages.clear()
                self._stages.clear()
                self._saved.clear()
                self._progress.clear()
                self._cost_previous = None
                self.manager._active_tasks.discard(self)
                self.manager._event("backward_finished", invocation=self.frame.invocation,
                                    task=self.task_id, error=None if error is None else type(error).__name__)
                self.manager._wake()
            if error is None and secondary is not None:
                raise secondary


class PackedOffloadRuntime:
    """Live transport owner. Public configuration never exposes this protocol."""

    def __init__(self, policy, *, backend=None, start_worker=True):
        self.policy, self.backend = policy, backend
        self._staging_limit_bytes = policy.device_staging_bytes
        self._d2h_limit_bytes = policy.d2h_inflight_bytes
        self._injected_backend = backend is not None
        self._start_worker = start_worker
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._active_frames = weakref.WeakSet()
        self._active_tasks = set()
        self._records = set()
        self._d2h, self._h2d, self._retiring = [], [], []
        self._source_ids = {}
        self._plans = {}
        self._indexes = {}
        self._reservations = {}
        self._admitting = 0
        self.learning = True
        self._cost_models, self._compute_durations, self._effective_prefetches, self._profiles = {}, {}, {}, {}
        self._compute_observations = Counter()
        self._compute_jobs = []
        self._hardware_identity = None
        self._events = deque(maxlen=2048)
        self._cleanup_errors = deque(maxlen=64)
        self._failed_stages = []
        self._admissions = deque(maxlen=2048)
        self._admission_counts = {}
        self.counters = Counter()
        self.staging_bytes = self.staging_peak = self.pending_retire = 0
        self.inflight_bytes = self.inflight_peak = 0
        self._thread = None
        self._error = None
        self._closed = False
        self._owner_gone = False
        self.pinned = self.pageable = None

    def _event(self, kind, **metadata):
        self._events.append(dict(kind=kind, **metadata))

    def _check_error(self):
        if self._closed:
            raise OffloadProtocolError("Packed runtime is closed")
        if self._error is not None:
            raise self._error

    def _index(self, plan):
        index = self._indexes.get(plan.signature)
        if index is None or index["plan"] is not plan:
            ordered = tuple(sorted(plan.consumers, key=lambda item: item.boundary))
            index = dict(plan=plan, bundle_by_id={bundle.bundle_id: bundle for bundle in plan.bundles},
                consumer_by_id={item.consumer_id: item for item in ordered},
                consumer_by_boundary={item.boundary: item for item in ordered}, ordered_consumers=ordered,
                boundaries=tuple(item.boundary for item in ordered),
                device=plan.bundles[0].components[0].device if plan.bundles else "cuda:0",
                slots={component.saved_index: (bundle.bundle_id, offset)
                       for bundle in plan.bundles for offset, component in enumerate(bundle.components)})
            self._indexes[plan.signature] = index
            self._request_index(index, getattr(plan, "prefetches", ()))
        return index

    @staticmethod
    def _request_index(index, requests):
        groups = {}
        for request in requests:
            if index["consumer_by_id"][request.consumer_id].panel is None:
                groups.setdefault(request.submit_boundary, []).append(request)
        index["has_requests"] = bool(requests)
        index["requests_by_boundary"] = {boundary: tuple(sorted(items, key=lambda item: item.consumer_boundary))
                                        for boundary, items in groups.items()}

    def _bundle(self, plan, bundle_id):
        result = self._index(plan)["bundle_by_id"].get(bundle_id)
        if result is None:
            raise OffloadProtocolError(f"Unknown packed bundle {bundle_id!r}")
        return result

    def _register_plan(self, plan):
        # Register every selected region before outer forward. Keep the maxima
        # across specializations because older graphs can still own leases.
        if plan.signature not in self._plans:
            if not self.policy.device_staging_bytes:
                self._staging_limit_bytes = max(self._staging_limit_bytes,
                    plan.required_staging_bytes * self.policy.max_backward_tasks)
            if not self.policy.d2h_inflight_bytes:
                self._d2h_limit_bytes = max(self._d2h_limit_bytes, plan.required_d2h_inflight_bytes)
        self._plans[plan.signature] = plan

    def _initialize(self, plan):
        previous = self._plans.get(plan.signature)
        if previous is not None and previous is not plan and previous != plan:
            raise OffloadProtocolError("Packed plan signature reused with contradictory immutable metadata")
        if previous is plan and self.pinned is not None:
            self.backend.check(self._index(plan)["device"])
            return
        if previous is None:
            d2h_limit = self.policy.d2h_inflight_bytes
            staging_limit = self.policy.device_staging_bytes
            slots, ids = set(), set()
            for bundle in plan.bundles:
                if bundle.bundle_id in ids:
                    raise OffloadProtocolError("Duplicate packed bundle identity")
                ids.add(bundle.bundle_id)
                groups = bundle.components[0].shape[0]
                shapes = ((groups, 64), (groups,), (groups,))
                strides = ((64, 1), (1,), (1,))
                for index, component in enumerate(bundle.components):
                    if (groups < 1 or tuple(component.shape) != shapes[index]
                            or tuple(component.stride) != strides[index]
                            or component.nbytes != _component_bytes(component)
                            or _dtype(component.dtype) != (torch.int32 if index == 0 else torch.float32)
                            or component.saved_index in slots):
                        raise OffloadProtocolError("Invalid packed triplet consumer ABI metadata")
                    slots.add(component.saved_index)
                if d2h_limit and bundle.bytes > d2h_limit:
                    raise OffloadAdmissionError(
                        f"Full packed bundle requires {bundle.bytes} D2H inflight bytes with a limit of {d2h_limit}")
                if self.policy.host_pageable_bytes and bundle.bytes > self.policy.host_pinned_bytes:
                    raise OffloadAdmissionError("Full packed bundle exceeds pageable pinned stage capacity")
                if self.policy.physical_host_budget and self.policy.host_pageable_bytes:
                    from .pinned_arena import plan_pinned_placement

                    if plan_pinned_placement(bundle.components, budget_bytes=self.policy.host_pinned_bytes) is None:
                        raise OffloadAdmissionError("Full packed bundle exceeds physical pinned staging capacity before forward")
            for consumer in plan.consumers:
                if any(bid not in ids for bid in consumer.bundle_ids):
                    raise OffloadProtocolError("Packed consumer names an unknown bundle")
                if consumer.bytes != sum(self._bundle(plan, bid).bytes for bid in consumer.bundle_ids):
                    raise OffloadProtocolError("Packed consumer byte count is not its complete operand set")
                panel = getattr(consumer, "panel", None)
                if staging_limit and consumer.staging_bytes > staging_limit:
                    raise OffloadAdmissionError(
                        f"Complete packed consumer requires {consumer.staging_bytes} staging bytes with a limit of {staging_limit}")
                if panel is not None and self.policy.physical_host_budget and self.policy.host_pageable_bytes:
                    from .pinned_arena import plan_pinned_placement

                    descriptions = []
                    for bid in consumer.bundle_ids:
                        for index, component in enumerate(self._bundle(plan, bid).components):
                            shape = (panel.groups_per_panel, 64) if index == 0 else (panel.groups_per_panel,)
                            descriptions.append(replace(component, shape=shape,
                                nbytes=prod(shape) * 4, stride=(64, 1) if index == 0 else (1,)))
                    if plan_pinned_placement(descriptions, budget_bytes=self.policy.host_pinned_bytes) is None:
                        raise OffloadAdmissionError("Panel operands exceed physical pinned staging capacity before forward")
        devices = {str(component.device) for bundle in plan.bundles for component in bundle.components}
        if len(devices) > 1:
            raise OffloadProtocolError("Packed runtime supports one device per process scope")
        device = next(iter(devices), "cuda:0")
        if self.backend is None:
            self.backend = _CudaBackend(device)
        self.backend.check(device)
        if torch.device(device) != self.backend.device:
            raise OffloadProtocolError("Packed runtime cannot mix different CUDA devices")
        if not self.backend.testing_only:
            # Descriptor-bound allowance, not unbounded allocation. Leases are
            # still reused only after their dependent submissions/completions.
            known = tuple(index["plan"] for index in self._indexes.values())
            events = (64 + self.policy.max_live_forwards * 3 * sum(len(item.bundles) for item in known)
                      + self.policy.max_backward_tasks * sum(
                          3 * len(item.consumers) + len(item.boundary_ids) + 8 for item in known))
            self.backend.event_capacity = max(self.backend.event_capacity, events)
        if not self.backend.testing_only and torch.device(device).type != "cuda":
            raise OffloadProtocolError("Enabled CPU offload execution is unsupported")
        if self.pinned is None:
            if self.policy.physical_host_budget:
                from .pinned_arena import PinnedArenaPool

                self.pinned = PinnedArenaPool(self.policy.host_pinned_bytes, self.backend)
            else:
                self.pinned = PinnedOffloadPool(self.policy.host_pinned_bytes, self.backend)
            self.pageable = PinnedOffloadPool(self.policy.host_pageable_bytes, self.backend, pinned=False)
            if not self.backend.testing_only:
                properties = torch.cuda.get_device_properties(self.backend.device)
                from importlib.metadata import PackageNotFoundError, version

                try:
                    triton = version("triton")
                except PackageNotFoundError:
                    triton = None
                self._hardware_identity = dict(device=str(self.backend.device), uuid=str(getattr(properties, "uuid", None)),
                    capability=(properties.major, properties.minor), torch=str(torch.__version__),
                    cuda=torch.version.cuda, triton=triton)
            else:
                self._hardware_identity = dict(backend="explicit_cpu_testing", cuda_execution=False)

    def begin_forward(self, plan, compute_stream=None, execution_scope=None):
        _check_hooks()
        with self._lock:
            self._check_error()
            if not self.policy.enabled or plan.policy != self.policy:
                raise OffloadProtocolError("Enabled runtime requires its exact frozen policy")
            self._initialize(plan)
            self._relieve_history_pressure(execution_scope, plan.selected_bytes)
            live_scopes = {frame.execution_scope for frame in self._active_frames} | set(self._reservations)
            if execution_scope not in live_scopes and len(live_scopes) >= self.policy.max_live_forwards:
                if self.policy.admission != "resident_fallback" or plan.selection_mode == "all_eligible":
                    raise OffloadAdmissionError("Maximum live packed forwards exceeded")
                self._register_plan(plan)
                frame = ForwardFrame(self, plan, {}, fallback=True, execution_scope=execution_scope)
                self._active_frames.add(frame)
                self.counters["resident_fallbacks"] += 1
                self._record_admission(frame)
                return frame
            self._register_plan(plan)
            reserved = self._reservations.get(execution_scope)
            if reserved is not None:
                queue = reserved.get(plan.signature)
                if not queue:
                    raise OffloadProtocolError("Regional plan is absent from its atomic forward reservation")
                hosts = queue.popleft()
                frame = ForwardFrame(self, plan, hosts or {}, fallback=hosts is None, execution_scope=execution_scope)
                self._active_frames.add(frame)
                if hosts is None:
                    self.counters["resident_fallbacks"] += 1
                self._record_admission(frame)
                return frame
            components = tuple(item for bundle in plan.bundles for item in bundle.components)
            pool = self.pageable if self.policy.host_pageable_bytes else self.pinned
            history = pool.acquire(components)
            if history is None:
                if self.policy.admission != "resident_fallback" or plan.selection_mode == "all_eligible":
                    raise OffloadAdmissionError(
                        f"Whole-forward host admission requires {plan.selected_bytes} bytes; "
                        f"available {pool.budget - pool.active}")
                self.counters["resident_fallbacks"] += 1
                frame = ForwardFrame(self, plan, {}, fallback=True, execution_scope=execution_scope)
            else:
                # Split one atomic history reservation into independent leases
                # without changing the pool's active accounting.
                offset, hosts = 0, {}
                for bundle in plan.bundles:
                    n = len(bundle.components)
                    hosts[bundle.bundle_id] = _HostLease(pool, history.tensors[offset:offset + n],
                        history.keys[offset:offset + n], bundle.bytes)
                    offset += n
                history.released = True
                history.tensors = ()
                frame = ForwardFrame(self, plan, hosts, execution_scope=execution_scope)
            self._active_frames.add(frame)
            self._event("forward_admitted", invocation=frame.invocation, signature=plan.signature,
                        bytes=plan.selected_bytes, resident_fallback=frame.resident_fallback)
            self._record_admission(frame)
            return frame

    def _record_admission(self, frame):
        plan = frame.plan
        region_id = plan.forward_signature + ":" + plan.backward_signature
        record = self._admission_counts.setdefault(region_id, dict(region_id=region_id,
            selected_bundle_ids=tuple(bundle.bundle_id for bundle in plan.bundles),
            offloaded_forwards=0, resident_fallback_forwards=0))
        record["resident_fallback_forwards" if frame.resident_fallback else "offloaded_forwards"] += 1
        actual = tuple(bundle.bundle_id for bundle in plan.bundles if bundle.bundle_id in frame.records)
        if actual and len(actual) < len(plan.bundles):
            record["partial_resident_forwards"] = record.get("partial_resident_forwards", 0) + 1
            self.counters["partial_regional_admissions"] += 1
        self._admissions.append(dict(region_id=region_id,
            selected_bundle_ids=actual, selected_bytes=sum(r.bundle.bytes for r in frame.records.values()),
            resident_fallback=frame.resident_fallback,
            partial_resident=bool(actual) and len(actual) < len(plan.bundles)))

    def _reserve_physical_history(self, plans, pool, *, over_scope=False):
        """Fast atomic reservation, with group-level fallback under pressure."""
        hosts = [{} for _ in plans]
        all_components = tuple(c for plan in plans for bundle in plan.bundles for c in bundle.components)
        lease = None if over_scope else pool.acquire(all_components)

        def assign(owned, groups):
            offset = 0
            for region, bundle in groups:
                n = len(bundle.components)
                hosts[region][bundle.bundle_id] = _HostLease(pool, owned.tensors[offset:offset + n],
                    owned.keys[offset:offset + n], bundle.bytes)
                offset += n
            owned.released, owned.tensors = True, ()

        if lease is not None:
            assign(lease, tuple((region, bundle) for region, plan in enumerate(plans) for bundle in plan.bundles))
        elif self.policy.admission != "resident_fallback" or any(p.selection_mode == "all_eligible" for p in plans):
            raise OffloadAdmissionError("Whole-model physical pinned capacity is unavailable before forward")
        elif not over_scope:
            # Complete named panel operand groups stay indivisible. This cold
            # pressure path does not change the immutable compiled inventory.
            units = [{(region, bundle.bundle_id)} for region, plan in enumerate(plans) for bundle in plan.bundles]
            by_key = {(region, bundle.bundle_id): bundle for region, plan in enumerate(plans) for bundle in plan.bundles}
            for region, plan in enumerate(plans):
                for consumer in plan.consumers:
                    if consumer.panel is None:
                        continue
                    group = {(region, bid) for bid in consumer.bundle_ids}
                    for unit in tuple(units):
                        if unit & group:
                            units.remove(unit)
                            group.update(unit)
                    units.append(group)
            try:
                for unit in sorted(units, key=lambda unit: (-sum(by_key[key].bytes for key in unit), tuple(sorted(unit)))):
                    groups = tuple((region, by_key[(region, bid)]) for region, bid in sorted(unit))
                    owned = pool.acquire(tuple(c for _, bundle in groups for c in bundle.components))
                    if owned is not None:
                        assign(owned, groups)
            except BaseException:
                for region_hosts in hosts:
                    for owned in region_hosts.values():
                        owned.release()
                raise
        queues = {}
        for plan, region_hosts in zip(plans, hosts):
            self._register_plan(plan)
            queues.setdefault(plan.signature, deque()).append(region_hosts or None)
        return queues, hosts

    @contextmanager
    def reserve_scope(self, plans, execution_scope):
        """Reserve the whole model's selected regional history before math."""
        plans = tuple(plans)
        _check_hooks()
        with self._lock:
            self._check_error()
            if execution_scope in self._reservations:
                raise OffloadProtocolError("Compiled forward scope already has an active reservation")
            for plan in plans:
                if plan.policy != self.policy:
                    raise OffloadProtocolError("Regional offload plans changed scope policy")
                self._initialize(plan)
            self._relieve_history_pressure(execution_scope, sum(plan.selected_bytes for plan in plans))
            live = {frame.execution_scope for frame in self._active_frames} | set(self._reservations)
            over_scope = execution_scope not in live and len(live) >= self.policy.max_live_forwards
            if over_scope and self.policy.admission != "resident_fallback":
                raise OffloadAdmissionError("Maximum live packed forwards exceeded")
            pool = self.pageable if self.policy.host_pageable_bytes else self.pinned
            if self.policy.physical_host_budget:
                queues, hosts = self._reserve_physical_history(plans, pool, over_scope=over_scope)
                self._reservations[execution_scope] = queues
                self._event("scope_admitted", selected_regions=sum(bool(items) for items in hosts),
                    resident_regions=sum(not items for items in hosts),
                    bytes=sum(owned.bytes for items in hosts for owned in items.values()),
                    model_scope_signatures=tuple(plan.model_scope_signature for plan in plans),
                    physical_capacity=True)
            else:
                queues = self._reserve_legacy_scope(plans, pool, over_scope)
                self._reservations[execution_scope] = queues
        try:
            yield self
        finally:
            with self._lock:
                queues = self._reservations.pop(execution_scope, {})
                for queue in queues.values():
                    for hosts in queue:
                        if hosts is not None:
                            for host in hosts.values():
                                host.release()

    def _reserve_legacy_scope(self, plans, pool, over_scope):
        """Preserve version-one whole-region logical-budget admission."""
        with self._lock:
            remaining = pool.budget - pool.active
            selected = []
            for plan in plans:
                fits = not over_scope and plan.selected_bytes <= remaining
                if not fits and self.policy.admission != "resident_fallback":
                    required = sum(item.selected_bytes for item in plans)
                    raise OffloadAdmissionError(f"Whole-forward scope admission requires {required} bytes; available {pool.budget - pool.active}")
                selected.append(fits)
                if fits:
                    remaining -= plan.selected_bytes
            descriptors = tuple(component for plan, keep in zip(plans, selected) if keep
                                for bundle in plan.bundles for component in bundle.components)
            lease = pool.acquire(descriptors)
            if lease is None:
                raise OffloadAdmissionError("Atomic forward reservation lost its declared host capacity")
            queues, offset = {}, 0
            for plan, keep in zip(plans, selected):
                self._register_plan(plan)
                hosts = None
                if keep:
                    hosts = {}
                    for bundle in plan.bundles:
                        n = len(bundle.components)
                        hosts[bundle.bundle_id] = _HostLease(pool, lease.tensors[offset:offset + n],
                            lease.keys[offset:offset + n], bundle.bytes)
                        offset += n
                queues.setdefault(plan.signature, deque()).append(hosts)
            lease.released, lease.tensors = True, ()
            self._event("scope_admitted", selected_regions=sum(selected), resident_regions=len(selected) - sum(selected),
                        bytes=sum(plan.selected_bytes for plan, keep in zip(plans, selected) if keep))
            return queues

    def _validate_components(self, bundle, tensors):
        if len(tensors) != 3 or len(bundle.components) != 3:
            raise OffloadProtocolError("Packed transport requires a complete triplet")
        identities = []
        for index, (component, tensor) in enumerate(zip(bundle.components, tensors)):
            expected_dtype = torch.int32 if index == 0 else torch.float32
            if (type(tensor) is not torch.Tensor or tensor.requires_grad or tensor.layout != torch.strided
                    or tensor.dtype != expected_dtype or tensor.dtype != _dtype(component.dtype)
                    or tuple(tensor.shape) != tuple(component.shape)
                    or tuple(tensor.stride()) != tuple(component.stride)
                    or not tensor.is_contiguous() or str(tensor.device) != str(component.device)
                    or tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != component.nbytes):
                raise OffloadProtocolError("Packed source contradicts its private contiguous consumer ABI")
            _version(tensor)
            identities.append(_storage(tensor))
        if len(set(identities)) != 3:
            raise OffloadProtocolError("Packed components require independent physical storages")
        # Serialize the entire bundle's storage registration with the expiry sweep in poll().
        with self._lock:
            for identity in identities:
                previous = self._source_ids.get(identity)
                if previous is not None and not previous[0].expired():
                    raise OffloadProtocolError("Distinct canonical bundles overlap live source storage")
            self._source_ids.update({
                identity: (StorageWeakRef(tensor.untyped_storage()), bundle.bundle_id)
                for identity, tensor in zip(identities, tensors)
            })

    def _relieve_history_pressure(self, execution_scope, required):
        """Wait only already enqueued DMA-only history, never a retained graph."""
        pool = self.pageable if self.policy.host_pageable_bytes else self.pinned
        while True:
            self.poll()
            live = {frame.execution_scope for frame in self._active_frames} | set(self._reservations)
            count_pressure = execution_scope not in live and len(live) >= self.policy.max_live_forwards
            byte_pressure = pool.active + min(required, pool.budget) > pool.budget
            if not count_pressure and not byte_pressure:
                return
            records = {record for frame in self._active_frames if frame.execution_scope != execution_scope
                       and all(record.closed or not any(record.refs[k] for k in ("forward", "graph", "task"))
                               for record in frame.records.values())
                       for record in frame.records.values() if not record.closed}
            pending = next((job.ready for job in self._d2h if job.record in records), None)
            if pending is None:
                pending = next((stage.ready for stage in self._h2d if any(record in records for record in stage.records)), None)
            if pending is None:
                return
            self.counters["history_backpressure_waits"] += 1
            pending.synchronize()

    def _wait_capacity(self, predicate, *, concurrent_tasks=False):
        while not predicate():
            self._check_error()
            self.poll()
            if predicate():
                return
            pending = next((item.ready for item in self._d2h), None)
            if pending is None:
                pending = next((item.completion for item in self._retiring), None)
            if pending is None:
                pending = next((item.ready for item in self._h2d), None)
            if pending is None:
                if concurrent_tasks and any(task.thread_id != threading.get_ident()
                                            for task in self._active_tasks):
                    self._condition.wait(timeout=0.01)
                    continue
                raise OffloadAdmissionError("Managed capacity cannot progress without a future consumer")
            self.counters["host_dependency_waits"] += 1
            pending.synchronize()

    def _pinned_can_acquire(self, components):
        can_acquire = getattr(self.pinned, "can_acquire", None)
        return (can_acquire(components) if can_acquire is not None else
                self.pinned.active + sum(_component_bytes(item) for item in components) <= self.pinned.budget)

    def _publish(self, record, tensors):
        size = record.bundle.bytes
        with self._lock:
            if size > self._d2h_limit_bytes:
                raise OffloadAdmissionError("A full bundle exceeds D2H inflight capacity")
            self._wait_capacity(lambda: self.inflight_bytes + size <= self._d2h_limit_bytes)
            stage = None
            if self.policy.host_pageable_bytes:
                while stage is None:
                    stage = self.pinned.acquire(record.bundle.components)
                    if stage is None:
                        self._wait_capacity(lambda: self._pinned_can_acquire(record.bundle.components))
                destinations = stage.tensors
            else:
                destinations = record.host.tensors
            record.acquire("dma")
            try:
                producer = self.backend.event()
                start = time.perf_counter() if self.learning else None
                ready = self.backend.copy(destinations, tensors, direction="d2h", dependency=producer,
                                          chunk_bytes=self.policy.copy_chunk_bytes, measure=self.learning)
                enqueue = (time.perf_counter() - start) * 1000 if start is not None else 0.0
            except BaseException:
                record.release("dma")
                if stage is not None:
                    stage.release()
                raise
            record.ready, record.published = ready, True
            if not self.policy.host_pageable_bytes:
                # copy_ increments the counter at enqueue, not completion;
                # checking it requires no read of the pending CPU payload.
                record.versions = tuple(_version(value) for value in record.host.tensors)
            self._d2h.append(_D2H(record, producer, ready, stage, enqueue))
            self.inflight_bytes += size
            self.inflight_peak = max(self.inflight_peak, self.inflight_bytes)
            self.counters.update(d2h_bytes=size, d2h_copies=3)
            self._event("d2h_submitted", invocation=record.generation, bundle=record.bundle.bundle_id, bytes=size)
            self._wake()

    def _restore(self, task, bundle_id, *, prefetch=False):
        self._check_error()
        if bundle_id in task._stages:
            return task._stages[bundle_id]
        record = task.frame.records[bundle_id]
        if not self._target_allows(record.bundle.bytes, demand=not prefetch):
            return None
        if self.policy.host_pageable_bytes:
            record.wait_host()
        else:
            record.validate(require_ready=False)
        stage = None
        if self.policy.host_pageable_bytes:
            stage = self.pinned.acquire(record.bundle.components)
            if stage is None:
                return None
            for destination, source in zip(stage.tensors, record.host.tensors):
                destination.copy_(source)
            self.counters["cpu_copy_bytes"] += record.bundle.bytes
            sources = stage.tensors
        else:
            sources = record.host.tensors
        record.acquire("dma")
        tensors = ()
        try:
            tensors = tuple(self.backend.allocate(component) for component in record.bundle.components)
            if len({_storage(value) for value in tensors}) != 3:
                raise OffloadProtocolError("Restored packed components must have independent storage")
            start = time.perf_counter() if self.learning else None
            ready = self.backend.copy(tensors, sources, direction="h2d", dependency=record.ready,
                                      chunk_bytes=self.policy.copy_chunk_bytes, measure=self.learning)
            enqueue = (time.perf_counter() - start) * 1000 if start is not None else 0.0
        except BaseException:
            record.release("dma")
            if stage is not None:
                stage.release()
            raise
        result = _Staging(record, tensors, ready, stage, enqueue_ms=enqueue, prefetched=prefetch)
        task._stages[bundle_id] = result
        self._h2d.append(result)
        self.staging_bytes += record.bundle.bytes
        self.staging_peak = max(self.staging_peak, self.staging_bytes)
        self.counters.update(h2d_bytes=record.bundle.bytes, h2d_copies=3)
        self.counters["prefetches" if prefetch else "demand_misses"] += 1
        self._event("h2d_submitted", invocation=record.generation, task=task.task_id,
                    bundle=bundle_id, bytes=record.bundle.bytes, prefetch=prefetch)
        return result

    def _admit_consumer(self, task, consumer):
        with self._lock:
            self._check_error()
            missing = [bid for bid in consumer.bundle_ids if bid not in task._stages]
            if not missing:
                return
            required = sum(self._bundle(task.plan, bid).bytes for bid in missing)
            if consumer.bytes > self._staging_limit_bytes:
                raise OffloadAdmissionError("The complete packed consumer operand set exceeds device staging")
            # Future unused prefetches may be retired. Never wait for a
            # not-yet-enqueued consumer to free its current operands.
            if self.staging_bytes + required > self._staging_limit_bytes:
                self._discard_prefetches(task=task, keep=consumer.bundle_ids, required_bytes=required)
            self._admitting += 1
            try:
                self._wait_capacity(lambda: self.staging_bytes + required <= self._staging_limit_bytes,
                                    concurrent_tasks=True)
                self._target_allows(required, demand=True)
                for bid in missing:
                    while self._restore(task, bid) is None:
                        self._wait_capacity(lambda: self._pinned_can_acquire(self._bundle(task.plan, bid).components))
                for bid in consumer.bundle_ids:
                    task._stages[bid].taken = True
            finally:
                self._admitting -= 1

    def _discard_prefetches(self, *, task=None, keep=(), required_bytes=None):
        candidates = [(other, bid, stage) for other in self._active_tasks for bid, stage in other._stages.items()
                      if not stage.taken and (other is not task or bid not in keep)]
        def next_use(item):
            other, bid, _ = item
            return min((c.boundary for c in other.plan.consumers if bid in c.bundle_ids
                        and c.boundary > (other.current_boundary or -1)), default=float("inf"))
        retired_bytes = 0
        for other, bid, stage in sorted(candidates, key=next_use, reverse=True):
            if (required_bytes is not None and
                    self.staging_bytes - retired_bytes + required_bytes <= self._staging_limit_bytes):
                break
            del other._stages[bid]
            self._retire(stage)
            retired_bytes += stage.bytes
            self.counters["shared_reuse_evictions" if stage.consumed else "prefetch_evictions"] += 1
            if stage.prefetched and not stage.consumed:
                self.counters["prefetch_evicted_before_use_bytes"] += stage.bytes

    def _cleanup_error(self, error, role):
        self._cleanup_errors.append(dict(role=role, exception_class=type(error).__name__, message=str(error)[:512]))

    def _retire(self, stage, *, suppress_errors=False):
        if stage.retired:
            return
        # If a prefetched tensor was never read, dependency on its submitted
        # copy is sufficient. Otherwise record after the numerical consumer.
        failure = None
        try:
            if stage.taken:
                if not stage.consumed:
                    # Admission may fail a metadata guard before take() has
                    # inserted its dependency. Retirement must still cover DMA.
                    self.backend.wait(stage.ready)
                stage.completion = self.backend.event()
            else:
                stage.completion = stage.last_consumer_event or stage.ready
        except BaseException as error:
            failure = error
            self._cleanup_error(error, "consumer_retirement_event")
            try:
                stage.ready.synchronize()
                self.backend.synchronize_consumer()
                stage.completion = _CpuEvent()
            except BaseException as fatal:
                self._cleanup_error(fatal, "consumer_retirement_owned_stream_drain")
                self._failed_stages.append(stage)
                self._error = fatal
                stage.retired = True
                if not suppress_errors:
                    raise failure
                return
        stage.retired = True
        self._retiring.append(stage)
        self.pending_retire += stage.bytes
        self._wake()
        if failure is not None and not suppress_errors:
            raise failure

    def _restore_panel(self, task, records, first_group, groups):
        """Stage only group rows, never a hidden whole packed allocation."""
        self._check_error()
        descriptions, sources = [], []
        for record in records:
            if self.policy.host_pageable_bytes:
                record.wait_host()
            else:
                record.validate(require_ready=False)
            for component, tensor in zip(record.bundle.components, record.host.tensors):
                shape = (groups, 64) if len(component.shape) == 2 else (groups,)
                descriptions.append(replace(component, shape=shape, nbytes=prod(shape) * 4))
                sources.append(tensor.narrow(0, first_group, groups))
        size = sum(component.nbytes for component in descriptions)
        self._discard_prefetches()
        self._wait_capacity(lambda: self.staging_bytes + size <= self._staging_limit_bytes,
                            concurrent_tasks=True)
        self._target_allows(size, demand=True)
        host_stage = None
        if self.policy.host_pageable_bytes:
            while host_stage is None:
                host_stage = self.pinned.acquire(descriptions)
                if host_stage is None:
                    self._wait_capacity(lambda: self._pinned_can_acquire(descriptions))
            for destination, source in zip(host_stage.tensors, sources):
                destination.copy_(source)
            self.counters["cpu_copy_bytes"] += size
            sources = host_stage.tensors
        for record in records:
            record.acquire("dma")
        try:
            tensors = tuple(self.backend.allocate(component) for component in descriptions)
            if len({_storage(value) for value in tensors}) != 9:
                raise OffloadProtocolError("Panel leaves require nine independent device storages")
            # Every D2H event is already recorded. H2D commands are ordered on
            # one copy stream; wait each source before submitting the panel.
            if not self.backend.testing_only:
                with torch.cuda.device(self.backend.device), torch.cuda.stream(self.backend.h2d):
                    for record in records:
                        self.backend.h2d.wait_event(record.ready.event)
            start = time.perf_counter() if self.learning else None
            ready = self.backend.copy(tensors, sources, direction="h2d", dependency=records[-1].ready,
                                      chunk_bytes=self.policy.copy_chunk_bytes, measure=self.learning)
            enqueue = (time.perf_counter() - start) * 1000 if start is not None else 0.0
        except BaseException:
            for record in records:
                record.release("dma")
            if host_stage is not None:
                host_stage.release()
            raise
        stage = _Staging(records[0], tensors, ready, host_stage, records=records, bytes=size, enqueue_ms=enqueue)
        self._h2d.append(stage)
        self.staging_bytes += size
        self.staging_peak = max(self.staging_peak, self.staging_bytes)
        self.counters.update(h2d_bytes=size, h2d_copies=9, panels_submitted=1)
        self._event("panel_h2d_submitted", task=task.task_id, first_group=first_group, groups=groups, bytes=size)
        return stage

    def _target_allows(self, size, *, demand):
        target = self.policy.peak_budget_bytes
        if target is None:
            return True
        allocated = self.backend.allocated_bytes()
        if allocated + size <= target:
            return True
        self.counters["peak_target_demand_misses" if demand else "peak_target_prefetch_deferrals"] += 1
        if demand:
            self._event("peak_target_unattainable", allocated_bytes=allocated, required_additional_bytes=size,
                        peak_budget_bytes=target, vendor_workspace_known=False)
            if self.policy.admission == "error":
                raise OffloadAdmissionError(
                    f"Current CUDA allocation {allocated} plus required packed operands {size} "
                    f"exceeds peak target {target}; vendor workspace remains unknown")
            return True
        return False

    def poll(self):
        with self._lock:
            self._source_ids = {identity: entry for identity, entry in self._source_ids.items()
                                if not entry[0].expired()}
            for job in tuple(self._d2h):
                if job.ready.query():
                    record = job.record
                    self._observe_copy(record.plan_signature, "D2H", record.bundle.bytes, job.ready, job.enqueue_ms)
                    if job.stage is not None:
                        start = time.perf_counter() if self.learning else None
                        for destination, source in zip(record.host.tensors, job.stage.tensors):
                            destination.copy_(source)
                        self.counters["cpu_copy_bytes"] += record.bundle.bytes
                        if start is not None:
                            self.counters["cpu_copy_seconds"] += time.perf_counter() - start
                        job.stage.release()
                    if job.stage is not None:
                        record.versions = tuple(_version(value) for value in record.host.tensors)
                    record.host_ready = True
                    self.inflight_bytes -= record.bundle.bytes
                    self._d2h.remove(job)
                    record.release("dma")
            for stage in tuple(self._h2d):
                if stage.ready.query():
                    self._observe_copy(stage.record.plan_signature, "H2D", stage.bytes, stage.ready, stage.enqueue_ms)
                    stage.dma_done = True
                    if stage.host_stage is not None:
                        stage.host_stage.release()
                        stage.host_stage = None
                    for record in stage.records:
                        record.release("dma")
                    self._h2d.remove(stage)
            for stage in tuple(self._retiring):
                if stage.dma_done and stage.completion.query():
                    self._retiring.remove(stage)
                    self.staging_bytes -= stage.bytes
                    self.pending_retire -= stage.bytes
                    stage.tensors = ()
                    stage.ready = stage.completion = None
            for job in tuple(self._compute_jobs):
                signature, first, last, start, end = job
                if end.query():
                    if not self.backend.testing_only and self.learning:
                        duration = start.event.elapsed_time(end.event)
                        key = signature, first, last
                        self._compute_observations[key] += 1
                        # The first execution can contain lazy kernel
                        # compilation/host idle time; never freeze it as steady
                        # compute slack. One warmup therefore stays cold.
                        if self._compute_observations[key] > 1:
                            values = self._compute_durations.setdefault(signature, {})
                            values[first] = max(values.get(first, 0), duration)
                            for index in range(first + 1, last):
                                values.setdefault(index, 0.0)
                    self._compute_jobs.remove(job)
            for task in (() if self._admitting else tuple(self._active_tasks)):
                for bid, (gate, _) in sorted(task._requests.items(), key=lambda item: item[1][1]):
                    if (not gate.query() or bid in task._stages
                            or self.staging_bytes + self._bundle(task.plan, bid).bytes > self._staging_limit_bytes):
                        continue
                    record = task.frame.records[bid]
                    if record.host_ready:
                        if self._restore(task, bid, prefetch=True) is not None:
                            del task._requests[bid]
            self._condition.notify_all()
            self._maybe_close_owner()

    def _wake(self):
        with self._condition:
            self._condition.notify_all()
            if self._start_worker and self._thread is None and not self._closed:
                self._thread = threading.Thread(target=self._pump, name="deepopt-packed-offload", daemon=True)
                self._thread.start()

    def _pump(self):
        try:
            while True:
                with self._condition:
                    self.poll()
                    pending = (self._d2h or self._h2d or self._retiring or self._compute_jobs
                               or any(t._requests for t in self._active_tasks))
                    if self._closed or not pending:
                        self._thread = None
                        return
                    self._condition.wait(timeout=0.002)
        except BaseException as error:
            with self._condition:
                self._error = error
                self._thread = None
                self._condition.notify_all()

    def drain(self):
        """Terminal/checkpoint wait for submitted owned work, not the device."""
        with self._lock:
            if self._active_tasks:
                raise OffloadProtocolError("Quiescence requires callers to finish their live backward tasks")
            while self._d2h or self._h2d or self._retiring or self._compute_jobs:
                pending = (self._d2h[0].ready if self._d2h else self._h2d[0].ready if self._h2d
                           else self._retiring[0].completion if self._retiring else self._compute_jobs[0][-1])
                pending.synchronize()
                self.poll()
            if self._error is not None:
                raise self._error
            self._maybe_close_owner()

    def _maybe_close_owner(self):
        if (self._owner_gone and not self._records and not self._active_tasks and not self._reservations
                and not (self._d2h or self._h2d or self._retiring or self._compute_jobs or self._failed_stages)):
            self._closed = True
            if self.pinned is not None:
                self.pinned.trim()
                self.pageable.trim()
            self._condition.notify_all()

    def quiesce(self):
        self.drain()
        return self.snapshot()

    def _observe_copy(self, signature, direction, nbytes, event, enqueue_ms):
        if not self.learning or self.backend.testing_only:
            return
        elapsed = event.elapsed_ms()
        if elapsed is not None and elapsed > 0:
            from .offload_cost import TransferCostModel

            self._cost_models.setdefault(signature, TransferCostModel()).observe(direction, nbytes, elapsed,
                                                                               enqueue_ms=enqueue_ms, load="compute")

    def freeze_profiles(self):
        self.drain()
        with self._lock:
            if not self.learning:
                return self.snapshot()["profiles"]
            from .offload_cost import TransferCostModel, plan_prefetches

            for signature, plan in self._plans.items():
                identity = dict(hardware=self._hardware_identity, forward=plan.forward_signature,
                                backward=plan.backward_signature, source_profile=plan.profile_identity,
                                policy=self.policy.as_dict())
                model = self._cost_models.setdefault(signature, TransferCostModel())
                profile = model.freeze(profile_identity=identity)
                effective = replace(plan, profile_identity=profile.identity)
                requests = plan_prefetches(effective, self._compute_durations.get(signature, {}), profile)
                self._effective_prefetches[signature] = requests
                self._request_index(self._index(plan), requests)
                self._profiles[signature] = dict(profile=profile.as_dict(), requests=tuple(
                    dict(bundle_id=item.bundle_id, consumer_id=item.consumer_id, submit_boundary=item.submit_boundary,
                         calibrated=item.calibrated, estimated_copy_ms=item.estimated_copy_ms,
                         ready_deadline_ms=item.ready_deadline_ms) for item in requests),
                    calibrated=bool(requests) and all(item.calibrated for item in requests),
                    compute_scope="GPU elapsed between coarse boundaries, including exposed waits; first execution excluded; not per-kernel timings",
                    queue_cost_measured=False, enqueue_cost_measured=True,
                    deadline_qualification=False)
            self.learning = False
            return self.snapshot()["profiles"]

    def close(self):
        self.drain()
        with self._lock:
            self._closed = True
            self._condition.notify_all()
            if self.pinned is not None:
                self.pinned.trim()
                self.pageable.trim()

    def snapshot(self):
        with self._lock:
            host = {}
            for name, pool in (("pinned", self.pinned), ("pageable", self.pageable)):
                values = pool.snapshot() if pool is not None else dict(active_bytes=0, cached_bytes=0, peak_bytes=0)
                host.update({f"{name}_{key}": value for key, value in values.items()})
            refs = Counter()
            for record in self._records:
                refs.update(record.refs)
            plans = tuple(dict(signature=plan.signature, region_id=plan.forward_signature + ":" + plan.backward_signature,
                               forward_signature=plan.forward_signature,
                               backward_signature=plan.backward_signature, selected_bytes=plan.selected_bytes,
                               selected_bundle_ids=tuple(bundle.bundle_id for bundle in plan.bundles),
                               bundle_bytes={bundle.bundle_id: bundle.bytes for bundle in plan.bundles},
                               required_staging_bytes=plan.required_staging_bytes,
                               required_d2h_inflight_bytes=plan.required_d2h_inflight_bytes,
                               model_scope_signature=plan.model_scope_signature, selection_mode=plan.selection_mode,
                               consumers=tuple(dict(consumer_id=item.consumer_id, boundary=item.boundary,
                                                    bundle_ids=item.bundle_ids) for item in plan.consumers))
                          for plan in self._plans.values())
            lifecycle = dict(live_forwards=len({frame.execution_scope for frame in self._active_frames}),
                             regional_frames=len(self._active_frames), backward_tasks=len(self._active_tasks),
                             graph_leases=refs["graph"], task_leases=refs["task"], dma_leases=refs["dma"])
            lifecycle.update(owner_gone=self._owner_gone, closed=self._closed)
            lifecycle["quiescent"] = not (self._d2h or self._h2d or self._retiring or self._active_tasks
                                          or self._compute_jobs or self._failed_stages)
            return dict(schema_version=1, policy=self.policy.as_dict(), plans=plans, counters=dict(self.counters),
                        transport_limits=dict(device_staging_bytes=self._staging_limit_bytes,
                                              d2h_inflight_bytes=self._d2h_limit_bytes),
                        host=host, device=dict(staging_bytes=self.staging_bytes, staging_peak_bytes=self.staging_peak,
                            pending_retire_bytes=self.pending_retire, d2h_inflight_bytes=self.inflight_bytes,
                            peak_d2h_inflight_bytes=self.inflight_peak), lifecycle=lifecycle,
                        events=tuple(self._events), testing_backend=bool(self.backend and self.backend.testing_only),
                        profiles=dict(self._profiles), profile_learning=self.learning,
                        profile_frozen=not self.learning,
                        calibrated=bool(self._profiles) and all(item["calibrated"] for item in self._profiles.values()),
                        cold=not self._profiles or not all(item["calibrated"] for item in self._profiles.values()),
                        admission_counts=tuple(dict(value) for value in self._admission_counts.values()),
                        admission_history=tuple(self._admissions),
                        admission_history_truncated=sum(value["offloaded_forwards"] + value["resident_fallback_forwards"]
                            for value in self._admission_counts.values()) > self._admissions.maxlen,
                        cleanup_errors=tuple(self._cleanup_errors),
                        copy_count_scope="component copies; commands may be chunked; counters alone do not prove overlap",
                        host_accounting="managed Tensor storage; underlying pinned allocator cache/page rounding not intercepted",
                        copy_queue_scope="chunk commands within one bundle are submitted together; no within-bundle urgent preemption",
                        cuda_execution_observed=bool(self.backend and not self.backend.testing_only and self.counters["d2h_bytes"]),
                        performance_qualification=False,
                        error=None if self._error is None else type(self._error).__name__)
