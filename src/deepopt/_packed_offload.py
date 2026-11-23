"""Strict, tensor-free policy for compiler-owned packed-save transport.

This policy describes residency, never a different quantization recipe.  It
does not expose streams, saved tensors or lifecycle operations to callers.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite

from ._memory_units import normalize_packed_offload_units


@dataclass(frozen=True)
class PackedActivationOffloadPolicy:
    enabled: bool = False
    policy: str = "memory_first_overlap"
    host_pinned_bytes: int = 0
    # Zero defers transport sizing to the selected compiled operands.
    device_staging_bytes: int = 0
    d2h_inflight_bytes: int = 0
    copy_chunk_bytes: int = 16 * 1024 * 1024
    max_live_forwards: int = 1
    max_backward_tasks: int = 1
    peak_budget_bytes: int | None = None
    latency_target_percent: float = 1.0
    admission: str = "resident_fallback"
    host_pageable_bytes: int = 0
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version not in (1, 2):
            raise ValueError("Unsupported packed_activation_offload descriptor version")
        if type(self.enabled) is not bool:
            raise ValueError("packed_activation_offload.enabled must be a boolean")
        if type(self.policy) is not str or self.policy not in ("demand", "memory_first_overlap"):
            raise ValueError("packed_activation_offload.policy must be demand or memory_first_overlap")
        for name in ("host_pinned_bytes", "device_staging_bytes", "d2h_inflight_bytes", "host_pageable_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"packed_activation_offload.{name} must be a nonnegative integer")
        for name in ("copy_chunk_bytes", "max_live_forwards", "max_backward_tasks"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"packed_activation_offload.{name} must be a positive integer")
        if self.peak_budget_bytes is not None and (type(self.peak_budget_bytes) is not int or self.peak_budget_bytes < 1):
            raise ValueError("packed_activation_offload.peak_budget_bytes must be a positive integer or None")
        try:
            finite_latency = type(self.latency_target_percent) in (int, float) and isfinite(self.latency_target_percent)
        except OverflowError:
            finite_latency = False
        if not finite_latency or self.latency_target_percent < 0:
            raise ValueError("packed_activation_offload.latency_target_percent must be finite and nonnegative")
        object.__setattr__(self, "latency_target_percent", float(self.latency_target_percent))
        if type(self.admission) is not str or self.admission not in ("resident_fallback", "error"):
            raise ValueError("packed_activation_offload.admission must be resident_fallback or error")
        if self.enabled and self.host_pinned_bytes < 1:
            raise ValueError("Enabled packed_activation_offload requires a positive pinned host budget")

    @property
    def mode(self):
        return "demand" if self.policy == "demand" else "predictive"

    @property
    def canonical_host_bytes(self):
        """Total retained history; a pinned staging ring is not that history."""
        return self.host_pageable_bytes or self.host_pinned_bytes

    @property
    def physical_host_budget(self):
        """V2 charges rounded arenas, including retained allocator capacity."""
        return self.version == 2

    def as_config(self):
        result = asdict(self)
        if self.version == 1:
            # Historical configs and signatures omit the descriptor version.
            result.pop("version")
        return result

    def as_dict(self):
        return {"version": self.version, **self.as_config()}


# Both the runtime table and standalone compiler use the same frozen value.
PackedActivationOffloadConfig = PackedActivationOffloadPolicy
_DEFAULT = PackedActivationOffloadPolicy()


def resolve_packed_offload(value=None):
    if value is None:
        return _DEFAULT
    if isinstance(value, PackedActivationOffloadPolicy):
        return value
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError("packed_activation_offload must be a configuration table or descriptor")
    raw = normalize_packed_offload_units(value)
    version = raw.pop("version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported packed_activation_offload descriptor version")
    unknown = set(raw) - {item.name for item in fields(PackedActivationOffloadPolicy)}
    if unknown:
        raise ValueError(f"packed_activation_offload: unknown fields: {sorted(unknown)}")
    return PackedActivationOffloadPolicy(**raw, version=version)


def validate_packed_offload_execution(value, *, device_type=None, compiler_options=None,
                                      cuda_graph=False, fp8=False):
    """Reject unsupported execution owners before numerical or RNG effects."""
    policy = resolve_packed_offload(value)
    if not policy.enabled:
        return policy
    if device_type is not None and str(device_type).split(":", 1)[0] != "cuda":
        raise ValueError("Packed activation offload requires native compiled CUDA execution")
    if cuda_graph or fp8:
        raise ValueError("Packed activation offload does not support CUDA Graph capture or FP8 state")
    options = compiler_options or {}
    for name in ("enable_aot_autodiff", "use_native_autodiff"):
        if options.get(name, True) is not True:
            raise ValueError("Packed activation offload requires DeepOpt native AutoDiff")
    return policy


def resolve_owned_packed_offload(owned=None, selected=None):
    """An existing compiled model cannot silently acquire another residency owner."""
    owned = resolve_packed_offload(owned) if owned is not None else None
    selected = resolve_packed_offload(selected) if selected is not None else None
    if owned is not None and selected is not None and owned != selected:
        raise ValueError("Packed activation offload differs from the owned compiled-model policy")
    return owned or selected or _DEFAULT
