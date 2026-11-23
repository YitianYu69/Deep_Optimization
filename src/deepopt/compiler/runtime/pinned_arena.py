"""Private, physically charged host arenas for descriptor-v2 transport.

Placement contains metadata only. CPU bindings use ``set_`` on a fresh Tensor,
not ordinary views: disjoint leaves therefore have independent version
counters. Arenas stay privately owned and charged after logical trim. The
runtime's graph/task/DMA leases decide when a binding may be released.
"""

from dataclasses import asdict, dataclass
import hashlib
from itertools import count
import json
from math import prod
import threading
import weakref


_ALIGNMENT = 64
_MIN_ARENA = 4096
_ids = count(1)
_registry_lock = threading.RLock()
_pinned_repositories = {}
_testing_repositories = weakref.WeakKeyDictionary()


def _aligned(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _capacity(value):
    return 1 << (max(value, _MIN_ARENA) - 1).bit_length()


def _description(component, index):
    shape = tuple(component.shape)
    dtype = str(component.dtype)
    if (not shape or any(type(x) is not int or x < 1 for x in shape)
            or dtype not in ("torch.int32", "torch.float32")):
        raise ValueError("Pinned arenas require complete int32/FP32 component shapes")
    size = prod(shape) * 4
    if getattr(component, "nbytes", size) != size:
        raise ValueError("Pinned component byte count contradicts its shape")
    stride, running = [], 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    stride = tuple(reversed(stride))
    if getattr(component, "stride", stride) not in (None, stride):
        raise ValueError("Pinned arena components must be contiguous whole leaves")
    return dict(component_index=index, name=str(getattr(component, "name", index)), shape=shape,
                dtype=dtype, stride=stride, nbytes=size, role=str(getattr(component, "role", "value")))


@dataclass(frozen=True)
class PinnedArenaSpan:
    component_index: int
    arena_index: int
    byte_offset: int
    nbytes: int
    name: str
    shape: tuple
    dtype: str
    stride: tuple
    role: str

    def __post_init__(self):
        if (any(type(x) is not int or x < 0 for x in (self.component_index, self.arena_index, self.byte_offset))
                or type(self.nbytes) is not int or self.nbytes < 1
                or type(self.shape) is not tuple or not self.shape
                or any(type(x) is not int or x < 1 for x in self.shape)
                or self.nbytes != prod(self.shape)*4 or type(self.stride) is not tuple
                or self.dtype not in ("torch.int32", "torch.float32")
                or any(type(x) is not str for x in (self.name, self.dtype, self.role))):
            raise ValueError("Invalid immutable pinned component span")
        expected, running = [], 1
        for dim in reversed(self.shape):
            expected.append(running)
            running *= dim
        if any(type(x) is not int for x in self.stride) or self.stride != tuple(reversed(expected)):
            raise ValueError("Pinned spans must describe complete contiguous leaves")


@dataclass(frozen=True)
class PinnedArenaPlacement:
    arena_capacities: tuple[int, ...]
    spans: tuple[PinnedArenaSpan, ...]
    alignment: int = _ALIGNMENT

    def __post_init__(self):
        if (type(self.arena_capacities) is not tuple or type(self.spans) is not tuple
                or any(type(x) is not int or x < _MIN_ARENA or x & (x-1) for x in self.arena_capacities)
                or type(self.alignment) is not int or self.alignment < 4 or self.alignment & (self.alignment-1)
                or any(type(x) is not PinnedArenaSpan for x in self.spans)
                or sorted(x.component_index for x in self.spans) != list(range(len(self.spans)))):
            raise ValueError("Invalid immutable pinned arena placement")
        ends = {}
        for span in sorted(self.spans, key=lambda x: (x.arena_index, x.byte_offset)):
            if (span.arena_index >= len(self.arena_capacities) or span.byte_offset % self.alignment
                    or span.byte_offset < ends.get(span.arena_index, 0)
                    or span.byte_offset+span.nbytes > self.arena_capacities[span.arena_index]):
                raise ValueError("Pinned arena spans overlap or exceed their physical capacity")
            ends[span.arena_index] = span.byte_offset+span.nbytes

    @property
    def physical_bytes(self):
        return sum(self.arena_capacities)

    @property
    def logical_bytes(self):
        return sum(span.nbytes for span in self.spans)

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def as_dict(self):
        return {"version": 1, **asdict(self), "physical_bytes": self.physical_bytes,
                "logical_bytes": self.logical_bytes, "signature": self.signature}


def plan_pinned_placement(components, *, budget_bytes, alignment=_ALIGNMENT,
                          arena_capacities=(), free_ranges=None):
    """Deterministic whole-leaf placement, or None if valid requests cannot fit.

    Existing capacities remain charged even when completely free. New arenas
    have page-sized power-of-two capacities, so Torch's host rounding does not
    multiply the charge for each component. A leaf never crosses arenas.
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("Pinned physical budget must be a nonnegative integer")
    if type(alignment) is not int or alignment < 4 or alignment & (alignment - 1):
        raise ValueError("Pinned alignment must be a power of two of at least four bytes")
    capacities = list(arena_capacities)
    if any(type(x) is not int or x < _MIN_ARENA or x & (x - 1) for x in capacities):
        raise ValueError("Arena capacities must be page-sized powers of two")
    if sum(capacities) > budget_bytes:
        return None
    descriptions = [_description(value, index) for index, value in enumerate(components)]
    ranges = ([[(0, size)] for size in capacities] if free_ranges is None
              else [[tuple(pair) for pair in values] for values in free_ranges])
    if len(ranges) != len(capacities):
        raise ValueError("Pinned free ranges must match existing arenas")
    for size, values in zip(capacities, ranges):
        previous = 0
        for start, end in values:
            if any(type(x) is not int for x in (start, end)) or not 0 <= previous <= start < end <= size:
                raise ValueError("Invalid or overlapping pinned free ranges")
            previous = end
    spans, missing = [], []

    def place(description, indices):
        for arena_index in indices:
            for range_index, (start, end) in enumerate(ranges[arena_index]):
                offset = _aligned(start, alignment)
                finish = offset + description["nbytes"]
                if finish <= end:
                    ranges[arena_index][range_index:range_index + 1] = (
                        ([(start, offset)] if start < offset else [])
                        + ([(finish, end)] if finish < end else []))
                    spans.append(PinnedArenaSpan(arena_index=arena_index, byte_offset=offset, **description))
                    return True
        return False

    for description in sorted(descriptions, key=lambda x: (-x["nbytes"], x["component_index"])):
        if not place(description, range(len(capacities))):
            missing.append(description)
    if missing:
        available = budget_bytes - sum(capacities)
        extent = 0
        for item in missing:
            extent = _aligned(extent, alignment) + item["nbytes"]
        needed = _capacity(extent)
        if needed <= available:
            additions = [needed]
        else:
            additions = []
            remaining = available
            while remaining >= _MIN_ARENA:
                size = 1 << (remaining.bit_length() - 1)
                additions.append(size)
                remaining -= size
        first = len(capacities)
        capacities.extend(additions)
        ranges.extend([[(0, size)] for size in additions])
        if any(not place(item, range(first, len(capacities))) for item in missing):
            return None
        used = {span.arena_index for span in spans}
        keep = list(range(first)) + [i for i in range(first, len(capacities)) if i in used]
        remap = {old: new for new, old in enumerate(keep)}
        capacities = [capacities[i] for i in keep]
        spans = [PinnedArenaSpan(**{**asdict(span), "arena_index": remap[span.arena_index]}) for span in spans]
    return PinnedArenaPlacement(tuple(capacities), tuple(sorted(spans, key=lambda x: x.component_index)), alignment)


@dataclass(frozen=True)
class PinnedSpanKey:
    arena_id: int
    byte_offset: int
    nbytes: int
    generation: int

    def __post_init__(self):
        if (any(type(x) is not int or x < 1 for x in (self.arena_id, self.nbytes, self.generation))
                or type(self.byte_offset) is not int or self.byte_offset < 0 or self.byte_offset % 4):
            raise ValueError("Invalid immutable pinned span generation")


@dataclass
class PinnedArenaAllocation:
    tensor: object
    physical_bytes: int
    accounting_verified: bool = True
    accounting: str = "page_aligned_power_of_two_capacity"


class _Repository:
    def __init__(self):
        self.lock = threading.RLock()
        self.arenas = []
        self.live = {}
        self.pending = {}
        self.physical_peak = self.allocation_attempts = 0
        self.failed_capacity = 0
        self.cleanup_failures = 0
        self.accounting_verified = True
        self.identity = next(_ids)

    @property
    def physical(self):
        return sum(arena["physical_bytes"] for arena in self.arenas) + self.failed_capacity

    def release(self, key):
        with self.lock:
            record = self.pending.pop(key, None)
            if record is None:
                return
            arena = next(value for value in self.arenas if value["id"] == key.arena_id)
            arena["free"].append((key.byte_offset, key.byte_offset + key.nbytes))
            merged = []
            for start, end in sorted(arena["free"]):
                if merged and merged[-1][1] == start:
                    merged[-1] = merged[-1][0], end
                else:
                    merged.append((start, end))
            arena["free"] = merged


def _reserve_range(arena, start, nbytes):
    end = start + nbytes
    index = next(i for i, (first, last) in enumerate(arena["free"]) if first <= start and end <= last)
    first, last = arena["free"][index]
    arena["free"][index:index + 1] = ([(first, start)] if first < start else []) + ([(end, last)] if end < last else [])


def _repository(backend):
    with _registry_lock:
        if backend.testing_only:
            return _testing_repositories.setdefault(backend, _Repository())
        domain = getattr(backend, "pinned_arena_domain", None)
        if type(domain) is not str or not domain:
            raise ValueError("Physical pinned arenas require a qualified allocator domain")
        return _pinned_repositories.setdefault(domain, _Repository())


class _ArenaLease:
    def __init__(self, pool, tensors, keys, size):
        self.pool, self.tensors, self.keys, self.bytes = pool, tuple(tensors), tuple(keys), size
        self.released = False

    def release(self):
        if not self.released:
            self.pool.release(self)
            self.released = True
            self.tensors = ()

    def __del__(self):
        if not getattr(self, "released", True):
            try:
                self.release()
            except BaseException:
                # Destruction must not replace a primary admission failure.
                # Preserve the capacity and invalidate any strict cap claim.
                self.pool._state.cleanup_failures += 1
                self.pool._state.accounting_verified = False


class PinnedArenaPool:
    """Stable arenas shared within one allocator domain and never uncharged.

    Logical release waits for binding objects to disappear as well as the
    caller's graph/task/DMA union lease. Trim keeps private arena ownership;
    it does not claim the backing pinned allocation was returned to Torch.
    """

    def __init__(self, budget, backend, *, pinned=True):
        if type(budget) is not int or budget < 0 or pinned is not True:
            raise ValueError("Pinned arenas require a nonnegative physical budget")
        self.budget, self.backend, self.pinned = budget, backend, True
        self._state = _repository(backend)

    @property
    def active(self):
        return sum(key.nbytes for key in (*self._state.live, *self._state.pending))

    @property
    def cached(self):
        return max(0, self._state.physical - self.active)

    @property
    def peak(self):
        return self._state.physical_peak

    @property
    def allocations(self):
        return self._state.allocation_attempts

    def _placement(self, components):
        state = self._state
        if not state.accounting_verified or state.failed_capacity:
            return None
        return plan_pinned_placement(components, budget_bytes=self.budget,
            arena_capacities=tuple(arena["capacity"] for arena in state.arenas),
            free_ranges=tuple(arena["free"] for arena in state.arenas))

    def can_acquire(self, components):
        with self._state.lock:
            return self._placement(tuple(components)) is not None

    def acquire(self, components):
        import torch

        components = tuple(components)
        if torch.is_inference_mode_enabled():
            raise ValueError("Pinned saved bindings require ordinary versioned tensors, not inference mode")
        state = self._state
        with state.lock:
            placement = self._placement(components)
            if placement is None:
                return None
            # Retain every successfully allocated block if a later allocation
            # fails. Dropping its Tensor would leave untracked Torch cache.
            for capacity in placement.arena_capacities[len(state.arenas):]:
                state.allocation_attempts += 1
                try:
                    allocation = self.backend.host_arena(capacity, pinned=True)
                except BaseException as error:
                    state.failed_capacity += getattr(error, "_deepopt_pinned_unowned_bytes", 0)
                    state.accounting_verified &= getattr(error, "_deepopt_pinned_failure_verified",
                                                         bool(self.backend.testing_only))
                    state.physical_peak = max(state.physical_peak, state.physical)
                    raise
                if not isinstance(allocation, PinnedArenaAllocation):
                    allocation = PinnedArenaAllocation(allocation, capacity,
                        accounting_verified=bool(self.backend.testing_only))
                tensor = allocation.tensor
                valid_charge = type(allocation.physical_bytes) is int and allocation.physical_bytes >= capacity
                record = dict(id=next(_ids), capacity=capacity,
                    physical_bytes=allocation.physical_bytes if valid_charge else capacity,
                    tensor=tensor, free=[(0, capacity)], verified=allocation.accounting_verified)
                state.arenas.append(record)
                state.accounting_verified &= allocation.accounting_verified and valid_charge
                state.physical_peak = max(state.physical_peak, state.physical)
                valid = (allocation.physical_bytes == capacity and state.physical <= self.budget
                    and isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.uint8
                    and tuple(tensor.shape) == (capacity,) and tensor.stride() == (1,)
                    and tensor.storage_offset() == 0 and tensor.untyped_storage().nbytes() == capacity
                    and (self.backend.testing_only or tensor.is_pinned()))
                if not valid or not state.accounting_verified:
                    state.accounting_verified = False
                    raise RuntimeError("Physical pinned arena allocation cannot be reconciled with its budget")
            tensors, keys = [], []
            before_ranges = [list(arena["free"]) for arena in state.arenas]
            try:
                for span in placement.spans:
                    arena = state.arenas[span.arena_index]
                    dtype = getattr(torch, span.dtype.removeprefix("torch."))
                    tensor = torch.empty(0, dtype=dtype, device="cpu")
                    tensor.set_(arena["tensor"].untyped_storage(), span.byte_offset // 4, span.shape, span.stride)
                    key = PinnedSpanKey(arena["id"], span.byte_offset, span.nbytes, next(_ids))
                    state.live[key] = dict(span=span, tensor=weakref.ref(tensor))
                    keys.append(key)
                    tensors.append(tensor)
                    if (tensor._base is not None or tensor.storage_offset() * 4 != span.byte_offset
                            or not tensor.is_contiguous() or tuple(tensor.shape) != span.shape
                            or (not self.backend.testing_only and not tensor.is_pinned())):
                        raise RuntimeError("Pinned arena binding violated its independent whole-leaf ABI")
                for span in placement.spans:
                    _reserve_range(state.arenas[span.arena_index], span.byte_offset, span.nbytes)
            except BaseException:
                for arena, ranges in zip(state.arenas, before_ranges):
                    arena["free"] = ranges
                # A traceback can retain already-created bindings. Preserve
                # their ranges until those owners disappear, even on failure.
                for key, tensor in zip(keys, tensors):
                    record = state.live.pop(key)
                    state.pending[key] = record
                    arena = next(value for value in state.arenas if value["id"] == key.arena_id)
                    _reserve_range(arena, key.byte_offset, key.nbytes)
                    weakref.finalize(tensor, state.release, key)
                raise
            return _ArenaLease(self, tensors, keys, placement.logical_bytes)

    def release(self, lease):
        state = self._state
        with state.lock:
            if len(lease.keys) != len(lease.tensors) or len(set(lease.keys)) != len(lease.keys):
                raise RuntimeError("Invalid pinned arena lease split")
            if any(key not in state.live for key in lease.keys):
                raise RuntimeError("Pinned arena span was released more than once")
            if any(state.live[key]["tensor"]() is not tensor for key, tensor in zip(lease.keys, lease.tensors)):
                raise RuntimeError("Pinned arena lease rebound its component")
            for key, tensor in zip(lease.keys, lease.tensors):
                record = state.live.pop(key)
                state.pending[key] = record
                weakref.finalize(tensor, state.release, key)

    def validate_lease(self, lease):
        """Check bindings without reading payloads or sharing Tensor owners."""
        state = self._state
        with state.lock:
            if len(lease.keys) != len(lease.tensors):
                raise RuntimeError("Pinned arena lease changed component arity")
            for key, tensor in zip(lease.keys, lease.tensors):
                record = state.live.get(key)
                if record is None or record["tensor"]() is not tensor:
                    raise RuntimeError("Pinned arena lease is stale or rebound")
                span = record["span"]
                arena = next(x for x in state.arenas if x["id"] == key.arena_id)
                if (tensor.device.type != "cpu" or str(tensor.dtype) != span.dtype
                        or tuple(tensor.shape) != span.shape or tuple(tensor.stride()) != span.stride
                        or tensor.storage_offset() * 4 != key.byte_offset
                        or tensor.untyped_storage()._cdata != arena["tensor"].untyped_storage()._cdata
                        or tensor.untyped_storage().nbytes() != arena["capacity"]
                        or (not self.backend.testing_only and not tensor.is_pinned())):
                    raise RuntimeError("Pinned arena component storage or metadata was modified")

    def trim(self):
        # Strong repository ownership deliberately preserves and charges the
        # arena. Reuse is qualified; physical return is not assumed.
        return None

    def snapshot(self):
        state = self._state
        with state.lock:
            pending_ids = {key.arena_id for key in state.pending}
            active_ids = {key.arena_id for key in state.live}
            pending = sum(x["physical_bytes"] for x in state.arenas if x["id"] in pending_ids)
            active = sum(x["physical_bytes"] for x in state.arenas if x["id"] in active_ids - pending_ids)
            arenas = [dict(arena_id=x["id"], address=(x["tensor"].data_ptr() if hasattr(x["tensor"], "data_ptr") else None),
                capacity_bytes=x["capacity"],
                physical_bytes=x["physical_bytes"], state=("pending_retirement" if x["id"] in pending_ids else
                    "active" if x["id"] in active_ids else "cached")) for x in state.arenas[:1024]]
            spans = [dict(arena_id=key.arena_id, byte_offset=key.byte_offset, nbytes=key.nbytes,
                generation=key.generation, name=value["span"].name, shape=value["span"].shape,
                dtype=value["span"].dtype, role=value["span"].role, state=kind)
                for kind, records in (("active", state.live), ("pending_retirement", state.pending))
                for key, value in list(records.items())[:1024]]
            return dict(active_bytes=self.active, cached_bytes=self.cached, peak_bytes=self.peak,
                budget_bytes=self.budget, allocations=self.allocations, pinned=True,
                physical_capacity_bytes=state.physical, physical_peak_bytes=state.physical_peak,
                physical_active_bytes=active, physical_pending_retirement_bytes=pending,
                physical_cached_bytes=state.physical - active - pending,
                physical_unowned_retired_bytes=state.failed_capacity,
                cleanup_failures=state.cleanup_failures,
                accounting_verified=state.accounting_verified and state.physical <= self.budget,
                physical_return_observed=False, allocator_domain_id=state.identity,
                accounting="rounded_private_arena_capacity_including_cached_and_retired_owners",
                arenas=arenas, spans=spans, omitted_arenas=max(0, len(state.arenas)-1024),
                omitted_spans=max(0, len(state.live)+len(state.pending)-len(spans)))
