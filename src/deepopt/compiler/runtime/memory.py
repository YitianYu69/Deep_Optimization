from __future__ import annotations

from collections import defaultdict
from typing import Any

from math import prod

import torch

from ..ir import InputNode, OutputNode, TensorValue, model_storage_source
from ..scheduler import Schedule, scheduled_inputs, scheduled_outputs, scheduled_nodes, scratch_requirements


class MemoryManager:
    """Emit local reference cleanup after each scheduled item's last use."""

    def __init__(self, schedule: Schedule, *, runtime_exports=frozenset()):
        self.schedule = schedule
        # Offloaded saves are exported by a runtime lease, rather than by a
        # Python reference in the wrapper's final GPU tuple. Their numerical
        # readers still participate in ordinary liveness accounting.
        self.runtime_exports = frozenset(runtime_exports)
        self.remaining_users: dict[TensorValue, int] = {}
        self._signature: tuple | None = None

    def consumed_values(self, item: Any) -> list[TensorValue]:
        values = scheduled_inputs(item)
        return ([value for value in values if value not in self.runtime_exports]
                if isinstance(item, OutputNode) else values)

    def collect_schedule_level_remaining_users(self) -> None:
        self.schedule.validate()
        remaining = defaultdict(int)
        for item in self.schedule.items:
            for value in self.consumed_values(item):
                remaining[value] += 1
        self.remaining_users = dict(remaining)
        self._signature = self.schedule.signature()

    def validate_current(self) -> None:
        if self._signature != self.schedule.signature():
            raise ValueError("Schedule changed after liveness analysis; rebuild use counts and code generation")

    def lifetime_plan(self) -> tuple[dict, ...]:
        """Logical reference intervals, deliberately not physical allocations.

        Alias backing storage, runtime codec layouts and opaque native workspace
        require the invocation storage ledger. Summing these logical intervals
        would double count views and must not be reported as a memory peak.
        """
        self.validate_current()
        births, consumers, owners = {}, defaultdict(list), {}
        escaped = set()
        for index, item in enumerate(self.schedule.items):
            for value in scheduled_outputs(item):
                births[value] = index
                provenance = getattr(value, "provenance", None)
                owners[value] = ("model" if model_storage_source(value) is not None else
                                 "autograd_saved" if provenance == "saved_state" else
                                 "caller" if isinstance(item, InputNode) else "invocation")
            for value in self.consumed_values(item):
                consumers[value].append(index)
                if isinstance(item, OutputNode):
                    escaped.add(value)
        records = []
        for value, birth in births.items():
            metadata = value.meta
            shape, dtype = getattr(metadata, "shape", None), getattr(metadata, "dtype", None)
            storage_dtype = getattr(metadata, "storage_dtype", None) or dtype
            size = (prod(shape) * torch.empty((), device="meta", dtype=dtype).element_size()
                    if shape is not None and dtype is not None else None)
            uses = tuple(consumers[value])
            records.append({
                "name": value.name, "birth": birth, "consumers": uses,
                "last_use": uses[-1] if uses else birth,
                "local_release_after": None if value in escaped else uses[-1] if uses else birth,
                "owner": owners[value], "escaped": value in escaped,
                "provenance": getattr(value, "provenance", None),
                "source_name": getattr(value, "source_name", None),
                "storage_source": model_storage_source(value),
                "shape": shape, "stride": getattr(metadata, "stride", None),
                "dtype": str(dtype) if dtype is not None else None,
                "storage_dtype": str(storage_dtype) if storage_dtype is not None else None,
                "representation": getattr(metadata, "representation", None),
                "logical_bytes": size, "physical_bytes": None,
                "alias_group": None, "storage_generation": None,
            })
        return tuple(records)

    def scratch_plan(self) -> tuple[dict, ...]:
        """Declared per-call intervals; not allocator observations or a peak sum.

        Physical reuse is owned by the executor and must stay within one
        invocation/stream. A call-scoped buffer cannot alias escaped outputs.
        """
        self.validate_current()
        return tuple(dict(requirement, birth=index, last_use=index,
                          local_release_after=index, physical_bytes_observed=None,
                          accounting="explicit_scratch_contract")
                     for index, item in enumerate(self.schedule.items)
                     for node in scheduled_nodes(item)
                     for requirement in scratch_requirements(node))

    def emit_del(
        self,
        item: Any,
        lines: list[str],
        runtime_names: dict[TensorValue, str],
        *,
        skip_values=frozenset(),
    ) -> None:
        for value in self.consumed_values(item):
            remaining = self.remaining_users.get(value, 0)
            if remaining <= 0:
                continue
            remaining -= 1
            self.remaining_users[value] = remaining
            runtime_name = runtime_names.get(value)
            if remaining == 0 and runtime_name is not None and value not in skip_values:
                lines.append(f"    del {runtime_name}")
        # Auxiliary multi-results and unused inputs otherwise survive until
        # function exit even though they have no scheduled consumers.
        for value in scheduled_outputs(item):
            runtime_name = runtime_names.get(value)
            if self.remaining_users.get(value, 0) == 0 and runtime_name is not None and value not in skip_values:
                lines.append(f"    del {runtime_name}")
