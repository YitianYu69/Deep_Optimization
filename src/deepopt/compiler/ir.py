from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from deepopt._bf16_policy import BF16Policy


# ================
# Tensor Metadata
# ================


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device | None | str = None
    stride: tuple[int, ...] | None = None
    requires_grad: bool = False
    representation: str = "dense"
    storage_dtype: torch.dtype | None = None
    block_shape: tuple[int, ...] | None = None
    layout: str | None = None

    def __post_init__(self):
        if any(not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ValueError(
                f"Tensor dimensions must be non-negative integers: {self.shape!r}"
            )
        if self.stride is not None and len(self.stride) != len(self.shape):
            raise ValueError("Tensor stride rank must match tensor shape rank")


# ==========
# IR values
# ==========


@dataclass(frozen=True)
class FreshAllocationContract:
    """A built-in executor's proof for a fresh first-order backward result.

    This describes an invocation-owned result, never model-storage provenance
    or permission to overwrite a saved/caller tensor. A consuming transform
    must still prove sole use, no escape/alias and legal effect placement.
    """

    meta: TensorMeta
    producer_op: str
    owner: str = "backward_invocation"

    def __post_init__(self):
        meta = self.meta
        if (self.owner != "backward_invocation" or not self.producer_op
                or meta.requires_grad or meta.representation != "dense" or meta.storage_dtype is not None
                or not meta.shape or any(size <= 0 for size in meta.shape)
                or meta.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                or meta.device is None or str(meta.device).split(":", 1)[0] not in ("cpu", "cuda")):
            raise ValueError("Fresh backward allocation requires a concrete dense invocation-owned result")
        expected, step = [], 1
        for size in reversed(meta.shape):
            expected.insert(0, step)
            step *= size
        if meta.stride != tuple(expected):
            raise ValueError("Fresh backward allocation requires the executor's proven contiguous layout")


@dataclass(eq=False)
class TensorValue:
    """
    Represent a tensor produced by some IR ops
    """

    name: str
    meta: TensorMeta | None = None

    producer: IRNode | None = None
    users: list[IRNode] = field(default_factory=list)
    save_codec: dict | None = None
    # Ownership survives lifting a model attribute into a joint-region input.
    provenance: str | None = None
    source_name: str | None = None
    # Provenance describes origin; this names a proven alias of model storage.
    # A clone/cast can keep model provenance without sharing its allocation.
    storage_source: str | None = None
    model_state_copy: bool = False
    model_state_origin: tuple[str, TensorMeta] | None = None
    model_state_transforms: tuple[str, ...] = ()
    fresh_allocation: FreshAllocationContract | None = None

    def __repr__(self):
        if self.meta is None:
            return f"TensorValue({self.name})"

        return (
            f"TensorValue("
            f"{self.name}, "
            f"shape={self.meta.shape}, "
            f"dtype={self.meta.dtype}, "
            f"device={self.meta.device}, "
            f"stride={self.meta.stride}, "
            f"requires_grad={self.meta.requires_grad})"
        )


@dataclass(frozen=True)
class AuxiliaryTensorMeta:
    """Non-differentiable kernel state with a backend-owned tensor layout.

    RNG/cumulative-sequence descriptors can vary with the selected native
    attention kernel. An optional slot preserves an undefined backend tensor
    as ``None`` through the saved-variable boundary; an empty tensor remains
    a present tensor. Ordinary dense tensors never use this presence contract.
    Unknown layout fields remain explicit rather than guessed from Q/K/V.
    """

    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    device: torch.device | str | None = None
    stride: tuple[int, ...] | None = None
    requires_grad: bool = False
    representation: str = "auxiliary"
    presence: str = "required"

    def __post_init__(self):
        if self.presence not in ("required", "optional"):
            raise ValueError("Auxiliary tensor presence must be required or optional")
        if self.requires_grad:
            raise ValueError("Auxiliary kernel state cannot require gradients")
        if self.shape is not None and any(not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ValueError("Auxiliary tensor dimensions must be non-negative integers")
        if self.stride is not None and (self.shape is None or len(self.stride) != len(self.shape)):
            raise ValueError("Auxiliary tensor stride requires a matching shape")


# =============
# Base IR Node
# =============


@dataclass(eq=False)
class IRNode:
    """
    Base class for every computation in our IR.

    ``outputs`` is the canonical graph interface and can describe any number
    of tensor results. ``output`` is a compatibility helper for compiler
    stages that still require a single result.
    """

    backward_rule: str | None = field(default=None, init=False)
    bf16_policy: BF16Policy = field(default_factory=BF16Policy, init=False)
    saved_tensors_for_bwd: tuple[TensorValue, ...] = field(default=(), init=False)
    activation_save_policy: str | None = field(default=None, init=False)
    # Replacement kernels declare ordering intent; the final scheduler owns
    # placement after checking data dependencies and effect barriers.
    scheduling_constraints: dict = field(default_factory=dict, init=False)
    saved_operand_uses: dict[int, Any] = field(default_factory=dict, init=False)

    def inputs(self) -> list[TensorValue]:
        return []

    def outputs(self) -> tuple[TensorValue, ...]:
        """Return all tensor values produced by this node.

        The fallback supports existing custom nodes that override the former
        single-output ``output()`` method. New multi-output nodes should
        override ``outputs()`` directly.
        """
        legacy_output = type(self).output
        if legacy_output is IRNode.output:
            return ()
        value = legacy_output(self)
        return () if value is None else (value,)

    def output(self) -> TensorValue | None:
        """Return the sole result, or raise when this node has several."""
        values = self.outputs()
        if len(values) > 1:
            raise NotImplementedError(
                f"{type(self).__name__} has {len(values)} outputs; "
                "this compiler path only supports one output"
            )
        return values[0] if values else None


# =======
# Inputs
# =======


@dataclass
class InputNode(IRNode):
    index: int
    value: TensorValue

    def outputs(self) -> tuple[TensorValue, ...]:
        return (self.value,)


# ==================
# Module Attributes
# ==================


@dataclass
class GetAttrNode(IRNode):
    # Target attributes from the target model
    target: str
    value: TensorValue

    def outputs(self) -> tuple[TensorValue, ...]:
        return (self.value,)


# ======================
# Pointwise Computation
# ======================


@dataclass
class PointwiseNode(IRNode):
    op: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    value: TensorValue
    # Keep original op around so that we can fail back to
    # native execution
    target: Any

    def inputs(self):
        return list(find_tensor_values((self.args, self.kwargs)))

    def outputs(self) -> tuple[TensorValue, ...]:
        return (self.value,)


# ======================
# Multi-output IR node
# ======================


@dataclass
class SplitNode(IRNode):
    """IR-only placeholder for an operation with two tensor results.

    The graph can record both results and their producer. FX lowering,
    scheduling, code generation, runtime execution, and autodiff do not yet
    implement this node.
    """

    op: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    values: tuple[TensorValue, TensorValue]

    def inputs(self) -> list[TensorValue]:
        return list(find_tensor_values((self.args, self.kwargs)))

    def outputs(self) -> tuple[TensorValue, ...]:
        return self.values


# ==============================
# Native / External Computation
# ==============================


@dataclass
class ExternKernelNode(IRNode):
    """
    The op that executed by an external implementation

    Can be PyTorch, cuBLAS, cuDNN, FlashAttention, CUTLASS, ...
    """

    op: str
    target: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    value: TensorValue | None
    # Parameters owned by a call_module target are implicit in the forward
    # callable, but must be explicit inputs to an AOT autograd boundary.
    autograd_inputs: tuple[TensorValue, ...] = ()
    autograd_input_names: tuple[str, ...] = ()
    autograd_buffers: tuple[TensorValue, ...] = ()
    autograd_buffer_names: tuple[str, ...] = ()
    autograd_owner: str = "pytorch"
    stateful: bool = False

    def inputs(self):
        values: list[TensorValue] = []
        seen: set[int] = set()
        for value in (
            *find_tensor_values((self.args, self.kwargs)),
            *self.autograd_inputs,
            *self.autograd_buffers,
        ):
            if id(value) not in seen:
                values.append(value)
                seen.add(id(value))
        return values

    def outputs(self) -> tuple[TensorValue, ...]:
        return () if self.value is None else (self.value,)


@dataclass(eq=False)
class CallNode(IRNode):
    """An explicit numerical operation with a fixed tensor-result schema.

    Auxiliary results are ordinary IR values. Their metadata marks codec
    payloads and forward statistics as non-differentiable, while keeping them
    visible to saved-state planning, scheduling and storage liveness.
    """

    op: str
    target: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    values: tuple[TensorValue, ...]
    specification: Any = None

    def inputs(self):
        return list(find_tensor_values((self.args, self.kwargs)))

    def outputs(self):
        return self.values


# ==============================
# Graph Output
# ==============================


@dataclass
class OutputNode(IRNode):
    result: Any

    def inputs(self):
        return list(find_tensor_values(self.result))


# ================
# Entire IR Graph
# ================


@dataclass
class IRGraph:
    nodes: list[IRNode] = field(default_factory=list)

    def add(self, node: IRNode):
        if not isinstance(node, IRNode):
            raise TypeError(
                f"IRGraph accepts IRNode instances, got {type(node).__name__}"
            )
        if any(existing is node for existing in self.nodes):
            raise ValueError("The same IR node cannot be added to a graph twice")
        outputs = tuple(node.outputs())
        if len({id(value) for value in outputs}) != len(outputs):
            raise ValueError(
                "A node cannot list the same TensorValue as multiple outputs"
            )
        for output in outputs:
            if not isinstance(output, TensorValue):
                raise TypeError(
                    f"IRNode outputs must be TensorValue instances, got {type(output).__name__}"
                )
            if output.producer is not None and output.producer is not node:
                raise ValueError(f"TensorValue {output.name!r} already has a producer")
        for output in outputs:
            output.producer = node
            if output.provenance is None:
                output.provenance = ("persistent_buffer" if isinstance(node, GetAttrNode)
                                     else "user_input" if isinstance(node, InputNode) else "activation")
            if output.source_name is None:
                output.source_name = output.name
        _inherit_model_ownership(node)
        register_uses(node)
        self.nodes.append(node)

    def refresh_model_ownership(self):
        """Propagate ownership revealed by lifted captured operation operands.

        Dynamo can introduce a buffer cast before the opaque operation that
        identifies the original input as a model buffer. Refresh in topological
        order after lowering, before selecting any approximate saved values.
        """
        for node in self.nodes:
            _inherit_model_ownership(node)

    def validate(self) -> None:
        """Check producer, use, and topological invariants for this graph."""
        positions = {id(node): index for index, node in enumerate(self.nodes)}
        if len(positions) != len(self.nodes):
            raise ValueError("IR graph contains a node more than once")

        for index, node in enumerate(self.nodes):
            outputs = tuple(node.outputs())
            if len({id(value) for value in outputs}) != len(outputs):
                raise ValueError(
                    f"{type(node).__name__} lists the same TensorValue more than once"
                )
            for output in outputs:
                if not isinstance(output, TensorValue):
                    raise TypeError(
                        f"IRNode outputs must be TensorValue instances, "
                        f"got {type(output).__name__}"
                    )
                if output.producer is not node:
                    raise ValueError(
                        f"Output {output.name!r} of {type(node).__name__} has an invalid producer"
                    )
            for value in node.inputs():
                if not isinstance(value, TensorValue):
                    raise TypeError(
                        f"IRNode input must be TensorValue, got {type(value).__name__} "
                        f"in {type(node).__name__}"
                    )
                producer = value.producer
                if producer is None:
                    raise ValueError(
                        f"Input value {value.name!r} used by {type(node).__name__} has no producer"
                    )
                producer_index = positions.get(id(producer))
                if producer_index is None:
                    raise ValueError(
                        f"Producer for value {value.name!r} is not in this graph"
                    )
                if producer_index >= index:
                    raise ValueError(
                        f"IR graph is not topologically ordered at value {value.name!r}"
                    )
                if not any(user is node for user in value.users):
                    raise ValueError(
                        f"Use list for value {value.name!r} is missing a consumer"
                    )

        output_nodes = [node for node in self.nodes if isinstance(node, OutputNode)]
        if len(output_nodes) > 1:
            raise ValueError(
                f"IR graph has {len(output_nodes)} output nodes; expected at most one"
            )

    def dump(self):
        print()
        print("=" * 8, "DO IR", "=" * 8)
        for i, node in enumerate(self.nodes):
            print(f"{i:03d}: {node}")
        print("=" * 21)
        print()


# ========
# Helpers
# ========


def _inherit_model_ownership(node):
    op = getattr(node, "op", "").removeprefix("autodiff_primitive_")
    if op in {"reshape", "view", "transpose", "t", "permute", "flatten", "unflatten", "squeeze",
              "unsqueeze", "expand", "expand_as", "narrow", "getitem", "slice", "detach", "to",
              "float", "bfloat16", "half", "double", "type_as", "clone", "contiguous"}:
        inputs = node.inputs()
        if inputs and inputs[0].provenance in {"parameter", "persistent_buffer", "constant"}:
            for output in node.outputs():
                output.provenance = inputs[0].provenance
                output.model_state_copy = True
                output.model_state_origin = model_state_origin(inputs[0])
                output.model_state_transforms = (*inputs[0].model_state_transforms, op)
                source_meta, output_meta = inputs[0].meta, output.meta
                same_storage_contract = (source_meta is not None and output_meta is not None
                    and source_meta.shape == output_meta.shape and source_meta.dtype == output_meta.dtype
                    and str(source_meta.device) == str(output_meta.device)
                    and source_meta.stride == output_meta.stride)
                # No-op dtype/device conversions return the operand. Explicit
                # copies (including positional copy=True) never inherit it.
                arguments = getattr(node, "args", ())
                kwargs = getattr(node, "kwargs", {})
                noop_cast = (op in {"to", "float", "bfloat16", "half", "double", "type_as", "contiguous"}
                    and same_storage_contract and not kwargs.get("copy", False)
                    and not any(argument is True for argument in arguments[1:]))
                if noop_cast or op in {"view", "transpose", "t", "permute", "squeeze", "unsqueeze",
                                       "expand", "expand_as", "narrow", "detach"}:
                    output.storage_source = model_storage_source(inputs[0])
                    output.model_state_copy = inputs[0].model_state_copy


def model_storage_source(value):
    """Return only statically proven model aliases, never derived copies."""
    if value.model_state_copy:
        return None
    if value.storage_source is not None:
        return value.storage_source
    if (value.provenance in {"parameter", "persistent_buffer", "constant"}
            and isinstance(value.producer, (GetAttrNode, InputNode))):
        return value.source_name or value.name
    return None


def inherit_model_alias(value, source):
    """Use only for an executor slot that returns the unchanged operand."""
    value.provenance = source.provenance
    value.source_name = source.source_name or source.name
    value.storage_source = model_storage_source(source)
    value.model_state_copy = source.model_state_copy
    value.model_state_origin = model_state_origin(source)
    value.model_state_transforms = source.model_state_transforms


def model_state_origin(value):
    if value.model_state_origin is not None:
        return value.model_state_origin
    source = model_storage_source(value)
    return (source, value.meta) if source is not None else None


def find_tensor_values(x):
    """
    Recursively find TensorValue objects contained inside:

        tuple
        list
        dict
    """

    if isinstance(x, TensorValue):
        yield x
    elif isinstance(x, (tuple, list)):
        for item in x:
            yield from find_tensor_values(item)
    elif isinstance(x, dict):
        for item in x.values():
            yield from find_tensor_values(item)


def register_uses(node: IRNode):
    """
    Record dependency information

    If:

        z = x + y

    then the AddNode is a user of both x and y
    """

    for value in node.inputs():
        if not any(user is node for user in value.users):
            value.users.append(node)
