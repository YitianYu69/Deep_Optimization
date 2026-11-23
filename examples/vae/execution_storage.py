"""Bounded weak physical-storage attribution for diagnostic executions only.

Imports remain Torch-free until a diagnostic explicitly creates an observer.
The observer runs inside already-compiled artifacts, outside Dynamo capture.
Only scalar/code metadata is retained; no numerical payload is read or saved.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import heapq
import json
from pathlib import Path
import time
import threading
import weakref


MAX_RECORDS = 50000
MAX_SAMPLES = 20000
MAX_JSON_BYTES = 64 * 2**20
MAX_ALLOCATOR_EVENTS = 50000
NATIVE_SCOPE_FIELDS = ("native_region_index", "native_scope_identity", "native_forward_signature",
    "native_backward_signature", "native_recipe_identity_sha256")


def _compact_storage(record):
    keys = ("storage_id", "device", "generation", "storage_bytes", "storage_implementation", "physical_address",
        "physical_address_status", "physical_byte_range", "birth_observed_at", "birth_phase", "producer", "owners",
        "last_use", "release_observed_at", "allocation_end_reason", "saved_reference_count", "saved_ever")
    return {key: deepcopy(record[key]) for key in keys if key in record}


def _owner_role(name, plan, phase):
    if plan.get("provenance") == "parameter" or plan.get("owner") == "model":
        return "model_state"
    if plan.get("provenance") == "user_input" or plan.get("owner") == "caller":
        return "caller_input_or_cotangent"
    if "codec_step" in name or "codec_minimum" in name:
        return "packed_codec_metadata"
    if plan.get("representation") == "packed":
        return "packed_payload"
    if phase == "backward":
        if "weight" in name and "grad" in name:
            return "parameter_gradient"
        return "native_adjoint_or_temporary"
    return "forward_activation_or_temporary"


def _category(record):
    owners = record.get("owners", ())
    for prefix in ("model_parameter", "model_buffer", "optimizer_state", "caller_input"):
        if any(owner.startswith(prefix) for owner in owners):
            return prefix
    roles = sorted(owner[5:] for owner in owners if owner.startswith("role:"))
    for role in ("packed_payload", "packed_codec_metadata", "parameter_gradient", "model_state"):
        if role in roles:
            return role
    return roles[-1] if roles else "unknown_visible_storage"


class NativeStorageTrace:
    """Transport-aware metadata observer sharing one weak ledger per update."""

    metadata_only = True
    supports_packed_transport = True

    def __init__(self, torch, device, *, recorder=None, runtime_snapshot=None,
                 max_records=MAX_RECORDS, max_samples=MAX_SAMPLES):
        from deepopt.memory.ledger import StorageLedger

        self.torch, self.device = torch, device
        self.recorder, self.runtime_snapshot = recorder, runtime_snapshot
        self.ledger = StorageLedger(max_records=max_records)
        self.max_samples = max_samples
        self._lock = threading.RLock()
        self._context = ContextVar("deepopt_diagnostic_storage_scope", default=None)
        self.samples, self.boundaries, self.errors = [], [], []
        self.samples_omitted = self.boundaries_omitted = 0
        self.invocations = 0
        self.context = {"phase": "setup", "operation": "persistent_state", "module": None, "source_node": None}
        self.peak_window = None
        self.baseline = None
        self.finished = False
        self._peak_score = -1
        self._model = self._optimizer = self._images = None

    @property
    def context(self):
        return self._context.get() or {"phase": "setup", "operation": "persistent_state", "module": None, "source_node": None}

    @context.setter
    def context(self, value):
        self._context.set(value)

    def observe_persistent(self, model=None, optimizer=None, images=None):
        with self._lock:
            return self._observe_persistent(model, optimizer, images)

    def _observe_persistent(self, model=None, optimizer=None, images=None):
        if model is not None:
            self._model = weakref.ref(model)
        if optimizer is not None:
            self._optimizer = weakref.ref(optimizer)
        if images is not None:
            self._images = weakref.ref(images)
        model = self._model() if self._model is not None else None
        optimizer = self._optimizer() if self._optimizer is not None else None
        images = self._images() if self._images is not None else None
        self.ledger.advance("persistent_state")
        if model is not None:
            for name, value in model.named_parameters():
                self.ledger.observe(value, owner="model_parameter:" + name)
                if value.grad is not None:
                    self.ledger.observe(value.grad, owner="role:parameter_gradient", producer=name)
            for name, value in model.named_buffers():
                self.ledger.observe(value, owner="model_buffer:" + name)
        if optimizer is not None:
            self.ledger.observe(optimizer.state, owner="optimizer_state")
        if images is not None:
            self.ledger.observe(images, owner="caller_input")
        sample = self.sample("persistent_state")
        if self.baseline is None:
            self.baseline = {**sample, "storages": [_compact_storage(record) for record in self.ledger.live_storage_records()]}

    def sample(self, trigger):
        with self._lock:
            return self._sample(trigger)

    def _sample(self, trigger):
        self.ledger.collect()
        live = self.ledger._sample()[0]
        row = {"sequence": self.ledger.sequence, "monotonic_ns": time.monotonic_ns(),
            "trigger": trigger, **self.context, "live_visible_bytes_by_device": live,
            "allocator_allocated_bytes": None, "allocator_reserved_bytes": None,
            "allocator_peak_allocated_bytes": None, "unknown_allocated_remainder_bytes": None}
        if self.device.type == "cuda":
            row.update(allocator_allocated_bytes=self.torch.cuda.memory_allocated(self.device),
                allocator_reserved_bytes=self.torch.cuda.memory_reserved(self.device),
                allocator_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device))
            visible = sum(size for device, size in live.items() if device.startswith("cuda"))
            row["unknown_allocated_remainder_bytes"] = row["allocator_allocated_bytes"] - visible
        value = row["allocator_allocated_bytes"]
        score = sum(live.values()) if value is None else value
        new_peak = score > self._peak_score
        if new_peak:
            self._peak_score = score
            self.peak_window = {**row, "storages": [_compact_storage(record) for record in self.ledger.live_storage_records()],
                "scope": "largest_sampled_current_allocation_not_unsampled_historical_peak"}
            if self.runtime_snapshot is not None:
                try:
                    self.peak_window["runtime_leases_and_capacity"] = self.runtime_snapshot()
                except Exception as error:
                    self.errors.append({"field": "peak_runtime_snapshot", "error_type": type(error).__name__,
                                        "error": str(error)[:4096]})
        if trigger in ("native_boundary", "persistent_state") or new_peak:
            if len(self.samples) < self.max_samples:
                self.samples.append(row)
            else:
                self.samples_omitted += 1
        return row

    @contextmanager
    def capture(self):
        from deepopt.compiler.runtime import native_boundary_observation

        with native_boundary_observation(self):
            yield self

    def run_artifact(self, artifact, *args, phase, **runtime_kwargs):
        from deepopt.compiler.runtime.physical_memory import PhysicalPlanObservation
        from deepopt.memory.ledger import StorageObserver

        trace = self
        if self.finished:
            raise RuntimeError("Storage trace belongs to one diagnostic update window")
        with self._lock:
            self.invocations += 1
            invocation = self.invocations
        source_hash = hashlib.sha256(artifact.source.encode()).hexdigest()
        if self.recorder is not None:
            self.recorder.record_artifact(artifact, phase=phase)

        class Observation(PhysicalPlanObservation):
            def __init__(self):
                super().__init__(artifact, ledger=trace.ledger)
                self.range = None
                self.metadata = {row["index"]: row for row in getattr(artifact, "boundary_metadata", ())}

            def begin(self, index):
                super().begin(index)
                self.close_range()
                nodes = self.metadata.get(index, {}).get("nodes", ())
                node = next((node for node in nodes if node.get("module")), nodes[0] if nodes else {})
                name = f"deepopt.storage.{phase}.{source_hash[:12]}.{invocation}.{index}"
                trace.context = {"phase": phase, "invocation": invocation, "index": index,
                    "range_name": name, "artifact_source_sha256": source_hash,
                    **{field: getattr(artifact, field, None) for field in NATIVE_SCOPE_FIELDS},
                    "operation": node.get("operation"), "module": node.get("module"),
                    "source_node": node.get("source_node")}
                with trace._lock:
                    trace.ledger.phase = name
                    trace.ledger.advance("native_begin")
                self.range = trace.torch.profiler.record_function(name)
                self.range.__enter__()
                with trace._lock:
                    if len(trace.boundaries) < trace.ledger.max_records:
                        trace.boundaries.append({**trace.context, "sequence": trace.ledger.sequence,
                            "monotonic_ns": time.monotonic_ns()})
                    else:
                        trace.boundaries_omitted += 1

            def close_range(self):
                if self.range is not None:
                    active, self.range = self.range, None
                    active.__exit__(None, None, None)

            def boundary(self, index, values):
                with trace._lock:
                    trace.ledger.phase = trace.context.get("range_name", phase)
                    super().boundary(index, values)
                    for name, value in values.items():
                        role = _owner_role(name, self.logical.get(name, {}), phase)
                        trace.ledger.observe(value, owner="role:" + role, consumer=f"native:{invocation}:{index}")
                    trace.sample("native_boundary")
                self.close_range()

            def finish(self, error=None):
                self.close_range()
                with trace._lock:
                    super().finish(error)

        class Dispatch(StorageObserver):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                with trace._lock:
                    trace.ledger.phase = trace.context.get("range_name", phase)
                    trace.ledger.advance(str(function))
                    trace.ledger.observe((args, kwargs), consumer=phase + ":" + str(function))
                # Only metadata is serialized across Python threads. Never
                # hold the observer lock across the numerical operation.
                value = function(*args, **kwargs)
                with trace._lock:
                    trace.ledger.phase = trace.context.get("range_name", phase)
                    trace.ledger.observe((args, kwargs))
                    operation = str(trace.context.get("operation", ""))
                    role = ("visible_convolution_backward_temporary" if phase == "backward" and "conv" in operation.lower()
                            else "visible_native_temporary")
                    trace.ledger.observe(value, owner="role:" + role, producer=phase + ":" + str(function))
                    trace.sample("dispatch:" + str(function))
                return value

        previous, previous_phase = self.context, self.ledger.phase
        observation = Observation()
        try:
            # Installing a dispatch observer around Dynamo itself changes its
            # capture contract. This scope starts only at artifact invocation.
            with Dispatch(self.ledger):
                return observation.run(artifact, *args, **runtime_kwargs)
        finally:
            observation.close_range()
            self.context = previous
            with self._lock:
                self.ledger.phase = previous_phase

    def finish(self):
        with self._lock:
            self.observe_persistent()
            self.ledger.collect()
            self.finished = True

    def as_dict(self):
        with self._lock:
            return self._as_dict()

    def _as_dict(self):
        self.ledger.collect()
        live, _, _ = self.ledger._sample()
        # Consumer name arrays are redundant with native boundary IDs and can
        # dominate even a tiny model. Export the complete bounded physical and
        # logical binding tables, not repeated per-dispatch consumer strings.
        logical_keys = ("value_id", "storage_id", "component", "shape", "dtype", "strides", "storage_offset",
            "logical_bytes", "version", "birth_observed_at", "release_observed_at", "binding_end_observed_at",
            "binding_end_reason", "storage_byte_span", "physical_byte_span")
        ledger = {"live_visible_bytes_by_device": live,
            "peak_visible_bytes_by_device": self.ledger.peak_bytes_by_device,
            "storages": [_compact_storage(record) for record in self.ledger._records],
            "logical_values": [{key: deepcopy(record[key]) for key in logical_keys if key in record}
                for record in self.ledger._values],
            "metadata_truncated": self.ledger._truncated,
            "max_records_per_collection": self.ledger.max_records,
            "unknown_storage": sorted(self.ledger._unknown),
            "format": "compact_physical_and_logical_bindings_without_repeated_consumer_arrays"}
        categories = defaultdict_bytes(self.ledger.live_storage_records())
        runtime = None
        if self.runtime_snapshot is not None:
            try:
                runtime = self.runtime_snapshot()
            except Exception as error:
                self.errors.append({"field": "final_runtime_snapshot", "error_type": type(error).__name__, "error": str(error)[:4096]})
        return {"schema_version": 1, "status": "complete_observed_boundaries" if not ledger["metadata_truncated"]
            and not self.samples_omitted and not self.boundaries_omitted and not self.errors else "incomplete",
            "implemented": True, "finished": self.finished, "metadata_only": True, "tensor_files_written": 0,
            "cuda_execution_evidence": self.device.type == "cuda" and self.invocations > 0,
            "scope": "weak_native_and_dispatch_visible_storages_not_all_allocator_or_vendor_storage",
            "attribution_complete": False, "unknown_vendor_workspace": True,
            "native_invocations": self.invocations, "baseline": deepcopy(self.baseline),
            "peak_window": deepcopy(self.peak_window), "live_category_bytes_by_device": categories,
            "ledger": ledger, "boundaries": deepcopy(self.boundaries), "allocator_samples": deepcopy(self.samples),
            "omitted_allocator_samples": self.samples_omitted, "omitted_boundaries": self.boundaries_omitted,
            "sample_policy": "native boundaries and new visible/current-allocator high-water marks; profiler keeps allocation events",
            "runtime_snapshot": runtime,
            "timing_note": "Host allocation/observation times; CUDA completion requires separate stream/event trace.",
            "category_note": "Each visible backing storage counted once. Reference-owner category is descriptive; native internal temporary role remains provisional.",
            "errors": deepcopy(self.errors)}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        report = self.as_dict()
        data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
        if len(data) > MAX_JSON_BYTES:
            report = {"schema_version": 1, "implemented": True, "status": "incomplete", "attribution_complete": False,
                "omission": "storage_ledger_byte_limit", "required_bytes": len(data), "limit_bytes": MAX_JSON_BYTES,
                "tensor_files_written": 0, "native_invocations": self.invocations}
            data = (json.dumps(report, indent=2) + "\n").encode()
        path = directory / "storage-ledger.json"
        with path.open("xb") as stream:
            stream.write(data)
        return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "status": report["status"], "attribution_complete": False, "omission": report.get("omission"),
            "tensor_files_written": 0}


def defaultdict_bytes(records):
    result = {}
    for record in records:
        device = result.setdefault(record["device"], {})
        category = _category(record)
        device[category] = device.get(category, 0) + record["storage_bytes"]
    return result


def join_allocator_trace(path, storage):
    """Join bounded allocator events to weak storage ranges/native host scopes."""
    value = json.loads(Path(path).read_text())
    ranges = {event["name"]: (event["ts"], event["ts"] + event.get("dur", 0))
        for event in value.get("traceEvents", ()) if str(event.get("name", "")).startswith("deepopt.storage.")
        and event.get("ph") == "X"}
    records = storage.get("ledger", {}).get("storages", ())
    boundary_context = {row["range_name"]: row for row in storage.get("boundaries", ()) if "range_name" in row}
    by_thread, memory_by_thread, event_ranges = {}, {}, {}
    for event in value.get("traceEvents", ()):
        if event.get("name") in ranges and event.get("ph") == "X":
            start, end = ranges[event["name"]]
            by_thread.setdefault(event.get("tid"), []).append((start, end, event["name"]))
        elif event.get("name") == "[memory]" and type(event.get("ts")) in (int, float):
            memory_by_thread.setdefault(event.get("tid"), []).append(event)
    # Correlate host allocator events on their own thread. A sweep avoids an
    # allocation-count times boundary-count scan on the full target VAE.
    for thread, events in memory_by_thread.items():
        intervals = sorted(by_thread.get(thread, ()))
        cursor, active = 0, []
        for event in sorted(events, key=lambda item: item["ts"]):
            timestamp = event["ts"]
            while cursor < len(intervals) and intervals[cursor][0] <= timestamp:
                start, end, name = intervals[cursor]
                heapq.heappush(active, (end - start, end, name))
                cursor += 1
            while active and active[0][1] < timestamp:
                heapq.heappop(active)
            event_ranges[id(event)] = active[0][2] if active else None
    by_address = {}
    for record in records:
        if record["device"].startswith("cuda") and record.get("physical_address") is not None:
            by_address.setdefault(record["physical_address"], []).append(record)
    rows, omitted = [], 0
    peak = None
    for event in value.get("traceEvents", ()):
        if event.get("name") != "[memory]":
            continue
        args = event.get("args", {})
        if args.get("Device Type") not in (1, "CUDA", "cuda"):
            continue
        address, size, timestamp = args.get("Addr"), args.get("Bytes"), event.get("ts")
        if type(address) is str:
            try:
                address = int(address, 0)
            except ValueError:
                address = None
        matches = []
        for record in by_address.get(address, ()):
            interval = ranges.get(record.get("birth_phase"))
            if interval is not None and timestamp is not None and interval[0] <= timestamp <= interval[1]:
                matches.append(record)
        active_range = event_ranges.get(id(event))
        native = boundary_context.get(active_range)
        row = {"timestamp_us": timestamp, "address": address, "bytes": size,
            "total_allocated_bytes": args.get("Total Allocated"), "total_reserved_bytes": args.get("Total Reserved"),
            "device_index": args.get("Device Id"), "storage_generations": [r["generation"] for r in matches],
            "categories": sorted({_category(record) for record in matches}),
            "native_boundary": None if native is None else {key: native.get(key) for key in
                (*NATIVE_SCOPE_FIELDS, "phase", "index", "module", "source_node", "operation", "range_name")},
            "join_status": "matched_address_generation_native_host_range" if matches else "unattributed_allocator_event"}
        if type(row["total_allocated_bytes"]) is int and (peak is None or row["total_allocated_bytes"] > peak["total_allocated_bytes"]):
            peak = dict(row)
        if len(rows) < MAX_ALLOCATOR_EVENTS:
            rows.append(row)
        else:
            omitted += 1
    return {"schema_version": 1, "scope": "CUDA_host_allocation_events_joined_to_visible_native_storage",
        "events": rows, "omitted_events": omitted, "event_limit": MAX_ALLOCATOR_EVENTS,
        "peak_allocation_event": peak, "matched_allocation_events": sum(row["bytes"] > 0 and bool(row["storage_generations"])
            for row in rows if type(row["bytes"]) is int),
        "unattributed_events": sum(not row["storage_generations"] for row in rows),
        "attribution_complete": False, "vendor_workspace_complete": False,
        "note": "Address plus observed native host range disambiguates recorded generations; unmatched/hidden storage remains explicit. Host timestamps do not imply GPU completion."}
