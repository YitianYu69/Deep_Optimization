"""Static, metadata-only cost estimates for compiler-generated regions.

Every value here is a prediction derived from declared tensor geometry and a
bandwidth model. Nothing in this module observes device execution, allocator
state or CUDA events, so an estimate must never be reported as measured time or
as an observed launch count.

The model exists to let the compiler decline a transformation it cannot win.
Two rules are deliberate:

* A region that fuses exactly one operation has no fusion gain. It replaces one
  eager ATen launch with one Triton launch and the same traffic. ATen's
  elementwise kernels are already vectorized, so admitting such a region can
  only lose.
* When no bandwidth is available the model reports 'unestimated' and never
  declines work. An unknown device must not silently change code generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

# 4.6 us is the measured range of a single small Triton launch on recent
# data-center GPUs. It is a model parameter, not an observation of this run.
DEFAULT_LAUNCH_OVERHEAD_MS = 4.6e-3
# An eager ATen elementwise launch is cheaper than a Triton launch from Python.
# Charging the Triton constant to both sides would bias the model toward
# admitting marginal fusions.
DEFAULT_EAGER_LAUNCH_OVERHEAD_MS = 2.5e-3
# A region must beat eager by this fraction before it is admitted.
DEFAULT_ADMIT_MARGIN = 0.10
# Below this many elements a single operation cannot amortize a new kernel.
DEFAULT_MIN_NUMEL = 4096
# Theoretical bandwidth is not attainable; this factor converts the vendor
# clock/bus product into a usable streaming estimate.
ACHIEVABLE_BANDWIDTH_FRACTION = 0.85

_ITEMSIZE = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


def dtype_itemsize(dtype) -> int | None:
    """Return the storage width of dtype in bytes, or None when unknown."""
    if dtype is None:
        return None
    return _ITEMSIZE.get(dtype)


def numel_of(shape) -> int | None:
    """Return the static element count of shape, or None when symbolic."""
    if shape is None:
        return None
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        return None
    return prod(shape)


def device_bandwidth_bytes_per_s(device) -> float | None:
    """Streaming bandwidth estimate for device, or None when unavailable.

    Only reads declared device properties. Returns None rather than guessing so
    that an unknown device disables estimation instead of changing code.
    """
    try:
        if not torch.cuda.is_available():
            return None
        index = None
        if device is not None:
            parsed = torch.device(device)
            if parsed.type != "cuda":
                return None
            index = parsed.index
        properties = torch.cuda.get_device_properties(
            torch.cuda.current_device() if index is None else index)
        clock_khz = getattr(properties, "memory_clock_rate", None)
        bus_bits = getattr(properties, "memory_bus_width", None)
        if not clock_khz or not bus_bits:
            return None
        # DDR transfers two words per clock; bus width is in bits.
        theoretical = float(clock_khz) * 1000.0 * 2.0 * (float(bus_bits) / 8.0)
        if not theoretical > 0:
            return None
        return theoretical * ACHIEVABLE_BANDWIDTH_FRACTION
    except Exception:
        # Estimation must never break compilation.
        return None


@dataclass(frozen=True)
class RegionCost:
    """Predicted cost of one pointwise partition versus its eager fallback."""

    numel: int
    inputs: int
    operations: int
    itemsize: int
    fused_bytes: int
    eager_bytes: int
    launches_fused: int
    launches_eager: int
    bandwidth_bytes_per_s: float | None = None
    fused_ms: float | None = None
    eager_ms: float | None = None
    predicted_speedup: float | None = None
    decision: str = "unestimated"
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "numel": self.numel,
            "inputs": self.inputs,
            # Operation names live in the estimate row; this is the chain depth.
            "operation_count": self.operations,
            "itemsize": self.itemsize,
            "fused_bytes": self.fused_bytes,
            "eager_bytes": self.eager_bytes,
            "launches_fused": self.launches_fused,
            "launches_eager": self.launches_eager,
            "predicted_fused_ms": self.fused_ms,
            "predicted_eager_ms": self.eager_ms,
            "predicted_speedup": self.predicted_speedup,
            "decision": self.decision,
            "reason": self.reason,
            "basis": "static_estimate_not_measurement",
        }


def estimate_region(
    *,
    numel: int | None,
    inputs: int,
    operations: int,
    dtype=None,
    bandwidth_bytes_per_s: float | None = None,
    launch_overhead_ms: float = DEFAULT_LAUNCH_OVERHEAD_MS,
    eager_launch_overhead_ms: float = DEFAULT_EAGER_LAUNCH_OVERHEAD_MS,
    admit_margin: float = DEFAULT_ADMIT_MARGIN,
    min_numel: int = DEFAULT_MIN_NUMEL,
) -> RegionCost:
    """Predict whether fusing this partition beats its eager fallback."""
    itemsize = dtype_itemsize(dtype)
    operations = max(1, int(operations))
    inputs = max(1, int(inputs))
    if numel is None:
        # Symbolic or missing geometry is not an empty region. Never decline
        # work the model cannot see.
        return RegionCost(0, inputs, operations, itemsize or 1, 0, 0, 0, operations,
                          decision="unestimated", reason="unknown_geometry")
    count = int(numel)
    if itemsize is None:
        return RegionCost(count, inputs, operations, 1, 0, 0, 0, operations,
                          decision="unestimated", reason="unknown_dtype")
    # A fused region reads its inputs once and writes one result. The eager
    # chain materializes every intermediate: n_ops - 1 extra writes and
    # n_ops - 1 extra reads.
    fused_bytes = (inputs + 1) * count * itemsize
    eager_bytes = (inputs + 1 + 2 * (operations - 1)) * count * itemsize
    if count == 0:
        return RegionCost(0, inputs, operations, itemsize, 0, 0, 1, operations,
                          decision="declined", reason="empty_region")
    if operations == 1:
        # Same traffic, same launch count, worse vectorization than ATen.
        return RegionCost(count, inputs, operations, itemsize, fused_bytes, eager_bytes,
                          1, 1, bandwidth_bytes_per_s, decision="declined",
                          reason="single_operation_region_has_no_fusion_gain")
    if bandwidth_bytes_per_s is None:
        return RegionCost(count, inputs, operations, itemsize, fused_bytes, eager_bytes,
                          1, operations, None,
                          decision="unestimated", reason="no_bandwidth_for_device")
    if count < min_numel:
        return RegionCost(count, inputs, operations, itemsize, fused_bytes, eager_bytes,
                          1, operations, bandwidth_bytes_per_s, decision="declined",
                          reason="below_minimum_region_size")
    seconds_per_byte = 1.0 / float(bandwidth_bytes_per_s)
    fused_ms = fused_bytes * seconds_per_byte * 1e3 + launch_overhead_ms
    eager_ms = eager_bytes * seconds_per_byte * 1e3 + operations * eager_launch_overhead_ms
    speedup = (eager_ms / fused_ms) if fused_ms > 0 else None
    if speedup is None or speedup < (1.0 + admit_margin):
        return RegionCost(count, inputs, operations, itemsize, fused_bytes, eager_bytes,
                          1, operations, bandwidth_bytes_per_s, fused_ms, eager_ms, speedup,
                          decision="declined", reason="predicted_gain_below_margin")
    return RegionCost(count, inputs, operations, itemsize, fused_bytes, eager_bytes,
                      1, operations, bandwidth_bytes_per_s, fused_ms, eager_ms, speedup,
                      decision="admitted", reason="predicted_gain_above_margin")


def partition_geometry(partition) -> tuple[int | None, int, int, object]:
    """Return (numel, inputs, operations, dtype) for a schedule partition."""
    from .scheduler import external_values, scheduled_nodes

    nodes = scheduled_nodes(partition)
    output = partition.output_value
    meta = getattr(output, "meta", None)
    numel = numel_of(getattr(meta, "shape", None))
    dtype = getattr(meta, "dtype", None)
    return numel, len(external_values(nodes)), len(nodes), dtype


def evaluate_partition(partition, *, bandwidth_bytes_per_s=None, **kwargs) -> RegionCost:
    """Estimate one partition using its declared metadata only."""
    numel, inputs, operations, dtype = partition_geometry(partition)
    if bandwidth_bytes_per_s is None:
        device = getattr(getattr(partition.output_value, "meta", None), "device", None)
        bandwidth_bytes_per_s = device_bandwidth_bytes_per_s(device)
    return estimate_region(numel=numel, inputs=inputs, operations=operations,
                           dtype=dtype, bandwidth_bytes_per_s=bandwidth_bytes_per_s,
                           **kwargs)


def apply(schedule, *, mode: str, bandwidth_bytes_per_s=None, **kwargs):
    """Attach estimates to schedule and optionally decline losing regions.

    mode='advisory' records predictions without changing execution.
    mode='enforce' replaces a declined partition with its plain IR nodes, so
    they run on their original eager callable. Returns (schedule, estimates).
    """
    from .scheduler import FusionGroup, SinglePointwiseOp

    if mode not in ("off", "advisory", "enforce"):
        raise ValueError("cost_model must be off, advisory, or enforce")
    if mode == "off":
        return schedule, ()
    estimates = []
    declined = []
    for index, item in enumerate(schedule.items):
        if not isinstance(item, (FusionGroup, SinglePointwiseOp)):
            continue
        cost = evaluate_partition(item, bandwidth_bytes_per_s=bandwidth_bytes_per_s, **kwargs)
        nodes = item.nodes if isinstance(item, FusionGroup) else [item.node]
        applied = mode == "enforce" and cost.decision == "declined"
        estimates.append({"index": index,
                          "schedule_position": "pre_admission",
                          "operations": tuple(getattr(node, "op", type(node).__name__)
                                              for node in nodes),
                          "applied": applied,
                          **cost.as_dict()})
        if applied:
            declined.append(id(item))
    if not declined:
        return schedule, tuple(estimates)
    return decline_partitions(schedule, frozenset(declined)), tuple(estimates)


def decline_partitions(schedule, declined: frozenset[int]):
    """Replace declined partitions with their plain nodes, preserving order."""
    from .scheduler import Schedule, scheduled_nodes

    items = []
    for item in schedule.items:
        if id(item) in declined:
            items.extend(scheduled_nodes(item))
        else:
            items.append(item)
    rebuilt = Schedule(items, effect_epochs=schedule.effect_epochs,
                       effect_barriers=schedule.effect_barriers)
    rebuilt.validate()
    return rebuilt
