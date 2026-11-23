"""Invocation-local, payload-free storage evidence for diagnostic executions.

The ledger samples tensor dispatch boundaries. It is not an allocator trace:
native-library workspaces and allocations hidden inside an operator are unknown.
Only weak storage/tensor references are retained. Saved-tensor hooks remain the
owner of their normal autograd payloads, independently of this diagnostic.
"""

from contextlib import contextmanager
from copy import deepcopy
import weakref

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_components(value, path="value"):
    """Visit physical FP8 components instead of the wrapper's logical storage."""
    from deepopt.opt.quantization.activation_tensor import FP8Activation

    if isinstance(value, FP8Activation):
        yield from tensor_components(value._data, path + ".data")
        yield from tensor_components(value._scale_inv, path + ".scale_inv")
        if value._valid_linear_packed():
            packed = value._linear_packed
            for name in ("data", "step", "minimum"):
                yield from tensor_components(getattr(packed, name), path + ".linear_packed." + name)
    elif isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for index, (name, child) in enumerate(value.items()):
            # Optimizer state is keyed by Parameters. Formatting such a key
            # would inspect values and can copy payload data into metadata.
            label = name if isinstance(name, (str, int)) else f"entry_{index}"
            yield from tensor_components(child, f"{path}.{label}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensor_components(child, f"{path}.{index}")


class StorageLedger:
    """Deduplicate physical storage while recording logical aliases and owners.

    Storage generations are invocation-local monotonic IDs. Expired weak
    references are removed before identities are reused, including reuse of a
    storage implementation address. Historical metadata is bounded; live weak
    storage references remain tracked so truncation never understates peaks.
    """

    def __init__(self, *, max_records=50000):
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self.max_records = max_records
        self.sequence = 0
        self.phase = "invocation"
        self.operation = None
        self._generation = self._logical_generation = 0
        self._live = {}
        self._logical = {}
        self._records, self._values, self._timeline = [], [], []
        self._peak, self._saved_peak, self._saved_nonpersistent_peak = {}, 0, 0
        self._saved_unique_bytes = 0
        self._saved_refs = {}
        self._truncated = False
        self._unknown = set()

    @contextmanager
    def scope(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield self
        finally:
            self.phase = previous

    def _append(self, collection, item, *, detail=False):
        limit = min(self.max_records, 64) if detail else self.max_records
        if len(collection) < limit:
            collection.append(item)
        else:
            self._truncated = True

    def _event(self, kind, **details):
        self._append(self._timeline, {"sequence": self.sequence, "phase": self.phase,
                                      "operation": self.operation, "kind": kind, **details})

    def advance(self, operation=None):
        self.sequence += 1
        self.operation = operation
        self.collect()

    def collect(self):
        """Observe release; its exact time lies since the previous observation."""
        for identity, entry in tuple(self._live.items()):
            if entry["reference"].expired():
                record = entry["record"]
                record["release_observed_at"] = self.sequence
                self._event("storage_release", storage_id=record["storage_id"])
                del self._live[identity]
        for identity, (reference, _) in tuple(self._logical.items()):
            if reference() is None:
                self._logical[identity][1]["release_observed_at"] = self.sequence
                del self._logical[identity]

    @staticmethod
    def _physical_address(value, storage):
        # Fake/meta tensors describe geometry without a backing allocation.
        # Querying their data pointer warns, and cannot supply physical proof.
        if value.device.type == "meta" or getattr(value, "fake_mode", None) is not None:
            return None
        try:
            return int(storage.data_ptr())
        except RuntimeError:
            return None

    def observe(self, values, *, owner=None, consumer=None, producer=None):
        """Record visible physical components without taking tensor ownership."""
        self.collect()
        ids = []
        for path, value in tensor_components(values):
            if value.layout != torch.strided or value.is_nested:
                self._unknown.add(f"unsupported tensor storage: {value.layout}")
                continue
            storage = value.untyped_storage()
            identity = str(value.device), storage._cdata
            address = self._physical_address(value, storage)
            entry = self._live.get(identity)
            if (entry is not None and address is not None
                    and entry["record"]["physical_address"] not in (None, address)):
                # resize_ can replace the allocation while retaining the same
                # StorageImpl. Preserve the old generation instead of moving
                # its historical address onto the new backing allocation.
                entry["record"].update(release_observed_at=self.sequence,
                    allocation_end_reason="backing_address_changed")
                self._event("storage_reallocation", storage_id=entry["record"]["storage_id"])
                del self._live[identity]
                entry = None
            if entry is None:
                self._generation += 1
                record = {"storage_id": self._generation, "device": identity[0],
                          "generation": self._generation, "storage_bytes": int(storage.nbytes()),
                          "storage_implementation": int(storage._cdata), "physical_address": address,
                          "physical_address_status": "available" if address is not None else "unavailable_metadata_only",
                          "birth_observed_at": self.sequence, "birth_phase": self.phase,
                          "producer": producer, "owners": [], "logical_values": [],
                          "consumers": [], "last_use": None, "release_observed_at": None,
                          "saved_reference_count": 0, "saved_ever": False, "saved_max_bytes": 0}
                entry = {"reference": StorageWeakRef(storage), "record": record}
                self._live[identity] = entry
                self._append(self._records, record)
                self._event("storage_birth", storage_id=self._generation,
                            storage_bytes=record["storage_bytes"])
            record = entry["record"]
            # resize_ changes the byte count without necessarily changing storage.
            record["storage_bytes"] = int(storage.nbytes())
            record["physical_byte_range"] = (None if address is None else
                [address, address + record["storage_bytes"]])
            if record["saved_ever"] and record["storage_bytes"] > record["saved_max_bytes"]:
                self._saved_unique_bytes += record["storage_bytes"] - record["saved_max_bytes"]
                record["saved_max_bytes"] = record["storage_bytes"]
            if owner is not None and owner not in record["owners"]:
                self._append(record["owners"], owner, detail=True)
            if consumer is not None:
                record["last_use"] = self.sequence
                if not record["consumers"] or record["consumers"][-1] != consumer:
                    self._append(record["consumers"], consumer, detail=True)
            previous = self._logical.get(id(value))
            if previous is not None and previous[0]() is value:
                logical = previous[1]
                # A set_ can give the same Tensor object a different backing.
                if (logical["storage_id"] != record["storage_id"]
                        or logical["version"] != self._version(value)):
                    logical["binding_end_observed_at"] = self.sequence
                    logical["binding_end_reason"] = ("storage_rebound" if logical["storage_id"] != record["storage_id"]
                                                     else "version_changed")
                    previous = None
            if previous is None and len(self._values) < self.max_records:
                self._logical_generation += 1
                logical = {"value_id": self._logical_generation, "storage_id": record["storage_id"],
                           "component": path, "shape": list(value.shape), "dtype": str(value.dtype),
                           "strides": list(value.stride()), "storage_offset": value.storage_offset(),
                           "logical_bytes": value.numel() * value.element_size(),
                           "version": self._version(value), "birth_observed_at": self.sequence,
                           "consumers": [], "last_use": None, "release_observed_at": None,
                           "binding_end_observed_at": None, "binding_end_reason": None}
                offset = value.storage_offset() * value.element_size()
                span = (sum((dimension - 1) * stride for dimension, stride in zip(value.shape, value.stride())) + 1
                        if value.numel() else 0) * value.element_size()
                logical["storage_byte_span"] = [offset, offset + span]
                logical["physical_byte_span"] = None if address is None else [address + offset, address + offset + span]
                self._logical[id(value)] = weakref.ref(value), logical
                self._append(self._values, logical)
                self._append(record["logical_values"], logical["value_id"], detail=True)
            elif previous is None:
                self._truncated = True
                logical = None
            if logical is not None:
                logical["version"] = self._version(value)
                if consumer is not None:
                    logical["last_use"] = self.sequence
                    if not logical["consumers"] or logical["consumers"][-1] != consumer:
                        self._append(logical["consumers"], consumer, detail=True)
            ids.append(record["storage_id"])
        self._sample()
        return tuple(dict.fromkeys(ids))

    @staticmethod
    def _version(value):
        try:
            return value._version
        except RuntimeError:  # Inference tensors do not have version counters.
            return None

    def save(self, value):
        ids = self.observe(value, owner="saved_for_backward")
        for entry in self._live.values():
            record = entry["record"]
            if record["storage_id"] in ids:
                record["saved_reference_count"] += 1
                if not record["saved_ever"]:
                    record["saved_ever"] = True
                    self._saved_unique_bytes += record["storage_bytes"]
                    record["saved_max_bytes"] = record["storage_bytes"]
                self._saved_refs[record["storage_id"]] = record
        self._event("saved_acquire", storage_ids=list(ids))
        self._sample()
        return ids

    def unpack(self, ids):
        for identity in ids:
            record = self._saved_refs[identity]
            record["last_use"] = self.sequence
            self._append(record["consumers"], f"{self.phase}:saved_unpack", detail=True)
        self._event("saved_unpack", storage_ids=list(ids))

    def release_saved(self, ids):
        for identity in ids:
            record = self._saved_refs[identity]
            record["saved_reference_count"] -= 1
            if not record["saved_reference_count"]:
                del self._saved_refs[identity]
        self._event("saved_release", storage_ids=list(ids))
        self._sample()

    @staticmethod
    def _persistent(record):
        return any(owner.startswith(("model_parameter", "model_buffer", "optimizer_state"))
                   for owner in record["owners"])

    def _sample(self):
        totals = {}
        for entry in self._live.values():
            record = entry["record"]
            device = record["device"]
            totals[device] = totals.get(device, 0) + record["storage_bytes"]
        for device, size in totals.items():
            self._peak[device] = max(self._peak.get(device, 0), size)
        saved = sum(record["storage_bytes"] for record in self._saved_refs.values())
        nonpersistent = sum(record["storage_bytes"] for record in self._saved_refs.values()
                            if not self._persistent(record))
        self._saved_peak = max(self._saved_peak, saved)
        self._saved_nonpersistent_peak = max(self._saved_nonpersistent_peak, nonpersistent)
        return totals, saved, nonpersistent

    def saved_snapshot(self):
        _, saved, nonpersistent = self._sample()
        return {"live_unique_bytes": saved, "live_nonpersistent_bytes": nonpersistent,
                "peak_live_unique_bytes": self._saved_peak,
                "peak_live_nonpersistent_bytes": self._saved_nonpersistent_peak,
                "cumulative_unique_storage_bytes": self._saved_unique_bytes}

    @property
    def peak_bytes_by_device(self):
        return dict(self._peak)

    def live_storage_records(self):
        """Bounded metadata copies; never expose storage weakrefs or tensors."""
        self.collect()
        return deepcopy([entry["record"] for entry in self._live.values()][:self.max_records])

    def snapshot(self):
        self.collect()
        totals, _, _ = self._sample()
        return {"live_visible_bytes_by_device": totals,
                "peak_visible_bytes_by_device": dict(self._peak),
                "saved_storage_lifetimes": self.saved_snapshot(),
                "storages": deepcopy(self._records), "logical_values": deepcopy(self._values),
                "events": deepcopy(self._timeline), "metadata_truncated": self._truncated,
                "max_records_per_collection": self.max_records,
                "max_details_per_record": min(self.max_records, 64),
                "opaque_workspace_bytes": None, "unknown_storage": sorted(self._unknown),
                "coverage": "Dispatch-boundary tensor storage only; releases are observation bounds. "
                            "Native workspaces and intra-operator temporaries are unknown. Category peaks overlap "
                            "and must not be added. Owners describe references, not additive allocations."}


class StorageObserver(TorchDispatchMode):
    """Opt-in diagnostic sampling; ordinary execution does not install a mode."""

    def __init__(self, ledger):
        super().__init__()
        self.ledger = ledger

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = str(function)
        self.ledger.advance(name)
        self.ledger.observe((args, kwargs), consumer=f"{self.ledger.phase}:{name}")
        result = function(*args, **kwargs)
        # Outputs coexist with the argument backing allocations at this point.
        self.ledger.observe((args, kwargs))
        self.ledger.observe(result, producer=f"{self.ledger.phase}:{name}")
        return result
