"""Opt-in final-schedule physical receipts, using the existing storage ledger.

No payload retention, allocator interception, or cross-invocation scratch pool.
Ordinary wrappers do not call these hooks. Observed bytes are backing-storage
bytes and must not be added to saved/allocator category peaks.
"""

from copy import deepcopy
from itertools import count
import weakref

import torch

from deepopt.memory.ledger import StorageLedger


_invocations = count(1)


class PhysicalPlanObservation:
    def __init__(self, artifact, *, max_records=50000, ledger=None):
        self.invocation = next(_invocations)
        self._artifact = weakref.ref(artifact)
        self._source = artifact.source
        self.ledger = StorageLedger(max_records=max_records) if ledger is None else ledger
        self.schedule = deepcopy(artifact.final_schedule)
        self.logical = {row["name"]: row for row in artifact.logical_lifetimes}
        self.bindings = []
        self.bindings_truncated = False
        self.finished = False
        self.exception = None
        self._stream = None
        self.current_index = None

    def begin(self, index):
        """Optional observed-wrapper metadata hook before numerical execution."""
        self.current_index = index

    def boundary(self, index, values):
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.device.type == "cuda":
                with torch.cuda.device(value.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("Physical observation is outside CUDA Graph capture")
                    stream = (str(value.device), torch.cuda.current_stream(value.device).cuda_stream)
            else:
                stream = (str(value.device), "host_ordered")
            if self._stream is None:
                self._stream = stream
            plan = self.logical.get(name, {})
            owner = plan.get("owner", "unknown")
            self.ledger.advance(f"schedule:{index}:{name}")
            ids = self.ledger.observe(value, owner=owner, consumer=f"schedule:{index}", producer=name)
            if len(self.bindings) < self.ledger.max_records:
                try:
                    version = value._version
                except RuntimeError:
                    version = None
                self.bindings.append(dict(name=name, index=index, storage_ids=ids, version=version,
                    shape=tuple(value.shape), strides=tuple(value.stride()), dtype=str(value.dtype),
                    device=str(value.device), storage_offset=value.storage_offset(), stream=stream,
                    owner=owner, escaped=plan.get("escaped", True), requires_grad=value.requires_grad,
                    stream_matches_invocation=stream == self._stream))
            else:
                self.bindings_truncated = True

    def finish(self, error=None):
        self.finished = True
        self.exception = None if error is None else type(error).__name__
        self.ledger.collect()

    def run(self, artifact, *args, **runtime_kwargs):
        if self.finished:
            raise RuntimeError("Physical observation belongs to one invocation")
        if (self._artifact() is not artifact or artifact.source != self._source
                or tuple(artifact.final_schedule) != tuple(self.schedule)):
            raise ValueError("Physical observation schedule identity mismatch")
        return artifact.observed_fn(*args, _physical_observer=self, **runtime_kwargs)

    def as_dict(self):
        snapshot = self.ledger.snapshot()
        groups = {}
        for binding in self.bindings:
            for identity in binding["storage_ids"]:
                groups.setdefault(identity, set()).add(binding["name"])
        return dict(invocation=self.invocation, finished=self.finished, exception=self.exception,
            final_schedule=self.schedule, bindings=deepcopy(self.bindings), ledger=snapshot,
            bindings_truncated=self.bindings_truncated,
            alias_groups=tuple(dict(storage_id=key, names=tuple(sorted(names))) for key, names in groups.items()),
            reuse_enabled=False, capture_observation=False,
            qualification="observed_boundaries_only",
            unknown_native_workspaces=True)


def reuse_candidates(observation):
    """Conservative physical feasibility receipts; never an active allocator.

    Unknown saves, escaped/caller/model buffers, aliases, version changes,
    multiple streams and differentiable values are excluded before considering
    shape/capacity. Actual generated-buffer reuse needs a qualified executor.
    """
    report = observation.as_dict()
    if (not report["finished"] or report["exception"] or report["bindings_truncated"]
            or report["ledger"]["metadata_truncated"]):
        return {"status": "unqualified", "candidates": (), "changes_execution": False}
    aliases = {row["storage_id"]: row["names"] for row in report["alias_groups"]}
    by_name = {}
    for binding in report["bindings"]:
        by_name.setdefault(binding["name"], []).append(binding)
    storages = {row["storage_id"]: row for row in report["ledger"]["storages"]}
    eligible = []
    for name, bindings in by_name.items():
        first = bindings[0]
        ids = first["storage_ids"]
        plan = observation.logical.get(name, {})
        if (len(ids) != 1 or first["owner"] != "invocation" or first["escaped"]
                or any(row["requires_grad"] or not row["stream_matches_invocation"] for row in bindings)
                or len({row["version"] for row in bindings}) != 1
                or any(row["storage_ids"] != ids for row in bindings)):
            continue
        storage = storages[ids[0]]
        if storage["saved_ever"] or len(aliases[ids[0]]) != 1 or first["version"] is None:
            continue
        eligible.append((name, plan, first, storage))
    candidates = []
    for source, plan, binding, storage in eligible:
        for target, target_plan, other, target_storage in eligible:
            if (plan.get("last_use", float("inf")) < target_plan.get("birth", -1)
                    and binding["dtype"] == other["dtype"] and binding["device"] == other["device"]
                    and binding["strides"] == other["strides"] and binding["shape"] == other["shape"]
                    and binding["stream"] == other["stream"]
                    and storage["storage_bytes"] >= target_storage["storage_bytes"]):
                candidates.append(dict(source=source, target=target,
                    storage_generation=storage["generation"], version=binding["version"],
                    bytes=storage["storage_bytes"], status="physical_feasibility_only"))
    return dict(status="recommendation_only", candidates=tuple(candidates), changes_execution=False,
                missing_gates=("qualified_generated_executor", "autograd_saved_ownership_receipt",
                               "target_runtime_and_peak_acceptance"))


class InvocationReuseArena:
    """Explicit experimental reuse restricted to compiler-generated buffers.

    Requires an earlier physical observation of this exact artifact. Only pure
    generated pointwise values whose complete consumer set is also generated
    are eligible. No opaque calls, autograd regions or cross-call pool shares.
    """
    def __init__(self, artifact, observation):
        if (observation._artifact() is not artifact or observation._source != artifact.source
                or observation.schedule != artifact.final_schedule
                or not observation.finished or observation.exception):
            raise ValueError("Reuse requires a completed matching physical observation")
        if getattr(artifact, "reuse_fn", None) is None:
            raise ValueError("Artifact has no generated-buffer reuse entry point")
        report = reuse_candidates(observation)
        if report["status"] == "unqualified":
            raise ValueError("Reuse requires complete untruncated physical observation")
        allowed = frozenset(artifact.reusable_values)
        self.allowed = {}
        for row in report["candidates"]:
            if row["source"] in allowed and row["target"] in allowed:
                self.allowed.setdefault(row["target"], set()).add(row["source"])
        self.artifact, self.pool, self.active = artifact, {}, {}
        self._observed_bindings = tuple(observation.bindings)
        self.stream = None
        self.events = []
        self.finished = False
        self._generation = 0
        self.ledger = StorageLedger()

    def acquire(self, name, shape, *, dtype, device):
        device = torch.device(device)
        if device.type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Experimental physical reuse is not qualified for CUDA Graph capture")
                stream = (str(device), torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = (str(device), "host_ordered")
        if self.stream is None:
            self.stream = stream
        if not any(row["device"] == str(device) and row["stream"] == stream
                   for row in self._observed_bindings):
            raise RuntimeError("Invocation reuse device/stream lacks physical observation")
        if stream != self.stream:
            raise RuntimeError("Invocation reuse cannot cross streams/devices")
        tensor, source = None, None
        for candidate in sorted(self.allowed.get(name, ())):
            entry = self.pool.get(candidate)
            if entry is None:
                continue
            value, version = entry
            if value._version != version:
                raise RuntimeError("Retired buffer version changed before reuse")
            if tuple(value.shape) == tuple(shape) and value.dtype == dtype and value.device == device:
                tensor, source = value, candidate
                del self.pool[candidate]
                break
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        self._generation += 1
        self.active[name] = weakref.ref(tensor), self._generation, None
        self.ledger.advance(f"acquire:{name}")
        ids = self.ledger.observe({name: tensor}, owner="invocation", producer=name)
        self.events.append(dict(kind="reuse" if source else "allocate", name=name, source=source,
                                storage_ids=ids, lease_generation=self._generation,
                                bytes=tensor.untyped_storage().nbytes()))
        return tensor

    def complete(self, name, tensor):
        entry = self.active.get(name)
        if entry is not None:
            self.active[name] = entry[0], entry[1], tensor._version

    def retire(self, name, tensor):
        entry = self.active.pop(name, None)
        if entry is None:
            return
        if entry[0]() is not tensor or entry[2] != tensor._version or tensor.requires_grad or tensor.grad_fn is not None:
            raise RuntimeError("Generated buffer ownership/version changed before retirement")
        self.pool[name] = tensor, tensor._version
        self.events.append(dict(kind="retire", name=name, lease_generation=entry[1], version=tensor._version))

    def finish(self):
        self.pool.clear()
        self.active.clear()
        self.finished = True

    def as_dict(self):
        return dict(events=deepcopy(self.events), ledger=self.ledger.snapshot(), finished=self.finished,
                    default_enabled=False, qualification="experimental_generated_buffer_reuse",
                    performance_qualified=False)

    def run(self, *args):
        if self.finished:
            raise RuntimeError("Reuse arena belongs to one invocation")
        values = args[0] if len(args) == 1 and isinstance(args[0], list) else args
        if any(isinstance(value, torch.Tensor) and value.requires_grad for value in values):
            raise ValueError("Generated buffer reuse excludes differentiable invocation inputs")
        module = self.artifact.reuse_fn.__globals__["_gm"]
        if any(value.requires_grad for value in (*module.parameters(), *module.buffers())):
            raise ValueError("Generated buffer reuse excludes differentiable model state")
        return self.artifact.reuse_fn(*args, _reuse_arena=self)
