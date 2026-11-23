"""Activation storage and rematerialization policies for AOT autodiff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..ir import GetAttrNode, IRNode, PointwiseNode, TensorValue


@dataclass(frozen=True)
class SavedUse:
    """A consumer's permission to approximate one backward coefficient.

    Omitting this request is exact, including for extension VJPs. ``codec`` is
    an immutable explicit recipe used by established semantic operators (MHA).
    It never grants permission to encode model-owned state.

    Exact semantic roles (such as ``exp.forward_result_coefficient`` in the
    ``forward_result_dtype`` rounding domain) describe the forward contract.
    Neither a role nor exact retention alone authorizes a gate exemption.
    """

    role: str = "coefficient"
    allow_approximation: bool = False
    consumer: str | None = None
    rounding_domain: str = "operand_dtype"
    codec: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self):
        if not self.role or not self.rounding_domain:
            raise ValueError("Saved uses require a role and rounding domain")
        if self.codec is not None:
            object.__setattr__(self, "codec", tuple(sorted(self.codec)))
            if not self.allow_approximation:
                raise ValueError("An encoded recipe requires approximation permission")
            recipe = dict(self.codec)
            if (len(self.codec) != 2 or set(recipe) != {"bits", "group_size"}
                    or type(recipe["bits"]) is not int or recipe["bits"] not in (1, 2, 4, 8)
                    or type(recipe["group_size"]) is not int or recipe["group_size"] <= 0):
                raise ValueError("Saved codec recipe requires valid bits and group_size")


@dataclass(frozen=True)
class SavedActivationPolicy:
    """Resolved forward-time affine recipe for generic eligible coefficients."""

    enabled: bool = False
    bits: int = 2
    group_size: int = 512

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.bits) is not int or self.bits not in (1, 2, 4, 8):
            raise ValueError("Invalid saved activation policy")
        if type(self.group_size) is not int or self.group_size <= 0:
            raise ValueError("Saved activation group_size must be positive")

    @property
    def recipe(self):
        return (("bits", self.bits), ("group_size", self.group_size))

    def select(self, value, use):
        import torch
        from math import prod

        if not use.allow_approximation:
            return None, "consumer_requires_exact"
        if (value.meta.representation != "dense"
                or getattr(value.meta, "storage_dtype", None) in (torch.float8_e4m3fn, torch.float8_e5m2)):
            return None, "preserve_physical_representation"
        if value.provenance in {"parameter", "persistent_buffer", "constant"} or isinstance(value.producer, GetAttrNode):
            return None, "model_owned_state"
        meta = value.meta
        if meta.shape is None or meta.dtype not in (torch.bfloat16, torch.float32):
            return None, "ineligible_dtype_or_auxiliary"
        if not prod(meta.shape):
            return None, "empty_coefficient"
        if use.codec is not None:
            return use.codec, None
        if not self.enabled:
            return None, "compression_disabled"
        if meta.dtype != torch.bfloat16:
            return None, "preserve_operand_rounding_domain"
        count = prod(meta.shape)
        words = (self.group_size + 32 // self.bits - 1) // (32 // self.bits)
        encoded_bytes = ((count + self.group_size - 1) // self.group_size) * (words * 4 + 8)
        footprint = count
        if meta.stride is not None and any(stride == 0 and size > 1 for size, stride in zip(meta.shape, meta.stride)):
            footprint = min(count, 1 + sum((size - 1) * stride for size, stride in zip(meta.shape, meta.stride)))
        if count < self.group_size or encoded_bytes >= footprint * meta.dtype.itemsize:
            return None, "small_coefficient"
        return self.recipe, None


def capture_saved_activation_policy(device_type):
    """Snapshot the effective scope, preserving an explicit disabled override."""
    from deepopt.opt._autocast import _get_activation_compression

    options = _get_activation_compression(device_type)
    if options is None:
        return SavedActivationPolicy()
    return SavedActivationPolicy(bool(options["enabled"]), options.get("bits", 2), options["group_size"])


def activation_device_type(values):
    """A host scalar must not select policy/AMP for a CUDA activation region."""
    import torch

    devices = [torch.device(value.meta.device).type for value in values
               if value.meta is not None and value.meta.device is not None]
    return next((device for device in devices if device != "cpu"), devices[0] if devices else "cpu")


class ActivationSavePolicy(Protocol):
    """Return whether a requested forward value should be rematerialized."""

    name: str

    def should_recompute(self, producer: IRNode | None, value: TensorValue) -> bool: ...


def pure_pointwise_replay(producer):
    """Only built-in deterministic pointwise recipes have a replay contract."""
    import operator
    import torch
    from ..trace.register_ops import get_trace_op_node

    if not isinstance(producer, PointwiseNode):
        return False
    try:
        definition = get_trace_op_node(producer.op)
    except NotImplementedError:
        return False
    if definition.node_type is not PointwiseNode:
        return False
    aliases = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
               "div": operator.truediv, "neg": operator.neg}
    modules = {"relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid, "tanh": torch.nn.Tanh,
               "gelu": torch.nn.GELU, "silu": torch.nn.SiLU, "leaky_relu": torch.nn.LeakyReLU,
               "hardtanh": torch.nn.Hardtanh, "hardswish": torch.nn.Hardswish,
               "hardsigmoid": torch.nn.Hardsigmoid, "softplus": torch.nn.Softplus, "mish": torch.nn.Mish}
    if producer.op not in aliases and producer.op not in modules:
        return False
    target = producer.target
    return (target is definition.target or target is aliases.get(producer.op)
            or type(target) is modules.get(producer.op))


@dataclass(frozen=True)
class SaveAllPolicy:
    """Store each activation explicitly requested by a backward rule."""

    name: str = "save"

    def should_recompute(self, producer: IRNode | None, value: TensorValue) -> bool:
        return False


@dataclass(frozen=True)
class RecomputePointwisePolicy:
    """Recompute pure pointwise producers and store graph leaves/other ops."""

    name: str = "recompute"

    def should_recompute(self, producer: IRNode | None, value: TensorValue) -> bool:
        return pure_pointwise_replay(producer)


@dataclass(frozen=True)
class SelectivePolicy:
    """Recompute cheap operations only when the saved value is large enough.

    ``minimum_bytes`` is an explicit policy threshold, not a performance claim.
    It is useful to avoid recomputation for tiny activations where little memory
    is recovered. Workloads should tune it with their own memory measurements.
    """

    minimum_bytes: int = 64 * 1024
    recompute_ops: frozenset[str] = frozenset({"add", "sub", "neg", "relu", "sigmoid"})
    name: str = "selective"

    def __post_init__(self) -> None:
        if self.minimum_bytes < 0:
            raise ValueError("minimum_bytes must be non-negative")

    def should_recompute(self, producer: IRNode | None, value: TensorValue) -> bool:
        if (
            not pure_pointwise_replay(producer)
            or producer.op not in self.recompute_ops
        ):
            return False
        if value.meta is None:
            return False
        element_size = getattr(value.meta.dtype, "itemsize", None)
        if element_size is None:
            try:
                import torch

                element_size = torch.empty((), dtype=value.meta.dtype).element_size()
            except (TypeError, RuntimeError):
                return False
        element_count = 1
        for dimension in value.meta.shape:
            element_count *= dimension
        return element_count * element_size >= self.minimum_bytes


def resolve_activation_policy(
    policy: str | ActivationSavePolicy | None,
) -> ActivationSavePolicy:
    if policy is None:
        return SaveAllPolicy()
    if isinstance(policy, str):
        if policy in ("save", "store", "save_all"):
            return SaveAllPolicy()
        if policy in ("recompute", "rematerialize"):
            return RecomputePointwisePolicy()
        if policy in ("selective", "auto"):
            return SelectivePolicy()
        raise ValueError(
            f"Unknown activation save policy {policy!r}; choose save, recompute, or selective"
        )
    if isinstance(policy, type) or not callable(
        getattr(policy, "should_recompute", None)
    ):
        raise TypeError(
            "Custom activation policy must implement should_recompute(producer, value)"
        )
    if not isinstance(getattr(policy, "name", None), str) or not policy.name:
        raise ValueError("Custom activation policy must define a non-empty name")
    return policy
