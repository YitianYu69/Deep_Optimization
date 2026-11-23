from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .ir import IRGraph, IRNode, PointwiseNode, TensorValue, InputNode, GetAttrNode
from .registry import BackendRegistry


def _renderable(value) -> bool:
    if (
        isinstance(value, TensorValue)
        or value is None
        or isinstance(value, (bool, int, str))
    ):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (tuple, list)):
        return all(_renderable(item) for item in value)
    if isinstance(value, dict):
        return all(
            _renderable(key) and _renderable(item) for key, item in value.items()
        )
    return False


@dataclass
class FusionGroup:
    """Pure elementwise IR nodes emitted together as one Triton kernel."""

    nodes: list[IRNode] = field(default_factory=list)

    @property
    def output_value(self) -> TensorValue:
        if not self.nodes:
            raise ValueError("A fusion group cannot be empty")
        output = self.nodes[-1].output()
        if output is None:
            raise TypeError(f"{type(self.nodes[-1]).__name__} has no tensor output")
        return output

    def __repr__(self) -> str:
        ops = ", ".join(
            str(getattr(node, "op", type(node).__name__)) for node in self.nodes
        )
        return f"FusionGroup([{ops}])"


@dataclass
class SinglePointwiseOp:
    """A one-node Triton partition (the name is kept for API compatibility)."""

    node: IRNode

    @property
    def output_value(self) -> TensorValue:
        output = self.node.output()
        if output is None:
            raise TypeError(f"{type(self.node).__name__} has no tensor output")
        return output

    def __repr__(self) -> str:
        return (
            f"SinglePointwiseOp({getattr(self.node, 'op', type(self.node).__name__)})"
        )


@dataclass
class Schedule:
    items: list[IRNode | FusionGroup | SinglePointwiseOp] = field(default_factory=list)
    # Source-derived ordering authority, retained across legal partitioning.
    effect_epochs: tuple[tuple[int, int], ...] = field(default=(), repr=False)
    effect_barriers: frozenset[int] = field(default=frozenset(), repr=False)

    def add(self, item: IRNode | FusionGroup | SinglePointwiseOp) -> None:
        self.items.append(item)

    def nodes(self) -> list[IRNode]:
        """The final transformed operations, in executable order."""
        return [node for item in self.items for node in scheduled_nodes(item)]

    def signature(self) -> tuple:
        """Detect rewrites that invalidate previously computed use counts."""
        return tuple(
            (id(item), tuple((id(node), getattr(node, "op", None), id(getattr(node, "target", None)),
                              repr(node.scheduling_constraints), repr(getattr(node, "bf16_policy", None)),
                              repr(scratch_requirements(node)),
                              tuple(id(value) for value in node.inputs()),
                              tuple((id(value), id(value.meta)) for value in node.outputs()))
                             for node in scheduled_nodes(item)))
            for item in self.items
        )

    def validate(self) -> None:
        # Only explicitly constrained fusion replacements may change source
        # order. Unknown effects and alias/version constraints remain barriers.
        IRGraph(self.nodes()).validate()
        if self.effect_epochs and _effect_epochs(self.nodes(), self.effect_barriers) != self.effect_epochs:
            raise ValueError("Schedule changed the source effect/alias barrier order")
        for item in self.items:
            if isinstance(item, FusionGroup) and not valid_single_output_group(item.nodes):
                raise ValueError("A scheduled fusion group has an escaping internal value")
            if isinstance(item, (FusionGroup, SinglePointwiseOp)) and not valid_same_shape_region(scheduled_nodes(item)):
                raise ValueError("A scheduled pointwise partition violates its layout contract")

    def evidence(self) -> tuple[dict, ...]:
        """Static dispatch evidence; a row is not a measured device launch."""
        self.validate()
        return tuple({
            "index": index,
            "operations": tuple(getattr(node, "op", type(node).__name__) for node in scheduled_nodes(item)),
            "inputs": tuple(value.name for value in scheduled_inputs(item)),
            "outputs": tuple(value.name for value in scheduled_outputs(item)),
            "ordering": "final_transformed_graph",
            "effect_epochs": tuple(dict(self.effect_epochs).get(id(node)) for node in scheduled_nodes(item)),
            "constraints": tuple(dict(node.scheduling_constraints) for node in scheduled_nodes(item)
                                 if node.scheduling_constraints),
            "workspace_bytes": None,
            "scratch_requirements": tuple(requirement for node in scheduled_nodes(item)
                                          for requirement in scratch_requirements(node)),
            "observed_cuda_launches": None,
        } for index, item in enumerate(self.items))

    def dump(self) -> None:
        print("\n======== Scheduler ========")
        for index, item in enumerate(self.items):
            if isinstance(item, FusionGroup):
                ops = ", ".join(
                    str(getattr(node, "op", type(node).__name__)) for node in item.nodes
                )
                print(f"{index:03d}: FusionGroup([{ops}]) -> {item.output_value.name}")
            elif isinstance(item, SinglePointwiseOp):
                print(f"{index:03d}: {item!r} -> {item.output_value.name}")
            else:
                print(f"{index:03d}: {type(item).__name__}")
        print("===========================\n")


def scratch_requirements(node: IRNode) -> tuple[dict, ...]:
    """Static, pointer-free call scratch. Unknown native workspace stays unknown."""
    requirements = node.scheduling_constraints.get("scratch_requirements")
    if requirements is None:
        requirements = getattr(getattr(node, "target", None), "scratch_requirements", ())
    result = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            raise ValueError("Scratch requirements must be metadata dictionaries")
        name, size = requirement.get("name"), requirement.get("bytes")
        if not isinstance(name, str) or not name or (size is not None and (type(size) is not int or size < 0)):
            raise ValueError("Scratch requires a name and nonnegative or unknown byte count")
        if requirement.get("lifetime", "call") != "call" or requirement.get("owner", "invocation") != "invocation":
            raise ValueError("Kernel scratch must be invocation-owned and released at call completion")
        reuse = requirement.get("reuse", "none")
        capture_safe = requirement.get("capture_safe_after_warmup")
        if (type(reuse) is not str or reuse not in ("none", "invocation_stream_ordered")
                or capture_safe is not None and type(capture_safe) is not bool):
            raise ValueError("Scratch reuse/capture contract must contain scalar metadata")
        result.append(dict(name=name, bytes=size, lifetime="call", owner="invocation",
                           reuse=reuse, capture_safe_after_warmup=capture_safe))
    return tuple(result)


def _node_output(node: IRNode) -> TensorValue | None:
    outputs = node.outputs()
    output = outputs[0] if len(outputs) == 1 else None
    return output if isinstance(output, TensorValue) else None


def external_values(nodes: list[IRNode]) -> list[TensorValue]:
    node_ids = {id(node) for node in nodes}
    result: list[TensorValue] = []
    seen: set[int] = set()
    for node in nodes:
        for value in node.inputs():
            if value.producer is not None and id(value.producer) in node_ids:
                continue
            if id(value) not in seen:
                result.append(value)
                seen.add(id(value))
    return result


def scheduled_nodes(item) -> list[IRNode]:
    if isinstance(item, FusionGroup):
        return item.nodes
    if isinstance(item, SinglePointwiseOp):
        return [item.node]
    if isinstance(item, IRNode):
        return [item]
    raise TypeError(f"Unsupported schedule item: {type(item).__name__}")


def _effect_epochs(nodes, barriers):
    result, epoch = [], 0
    for node in nodes:
        if isinstance(node, (InputNode, GetAttrNode)):
            continue
        if id(node) in barriers:
            epoch += 1
        result.append((id(node), epoch))
    # Operations inside a pure interval may reorder; barrier crossings may not.
    return tuple(sorted(result))


def scheduled_inputs(item) -> list[TensorValue]:
    # Count each operand once per dispatch even in expressions such as x*x.
    return external_values(scheduled_nodes(item))


def scheduled_outputs(item) -> tuple[TensorValue, ...]:
    if isinstance(item, (FusionGroup, SinglePointwiseOp)):
        return (item.output_value,)
    return tuple(item.outputs())


def _is_contiguous(meta) -> bool:
    if meta is None or meta.stride is None:
        return False
    expected = 1
    for size, stride in zip(reversed(meta.shape), reversed(meta.stride)):
        if size > 1 and stride != expected:
            return False
        expected *= max(size, 1)
    return True


def _is_triton_device(meta) -> bool:
    if meta is None or meta.device is None:
        return False
    try:
        return str(meta.device).split(":", 1)[0] == "cuda"
    except (TypeError, ValueError):
        return False


def _is_supported_dtype(meta) -> bool:
    if meta is None or meta.representation != "dense":
        return False
    return str(meta.dtype).split(".")[-1] in {
        "float16",
        "bfloat16",
        "float32",
    }


def valid_same_shape_region(nodes: list[IRNode]) -> bool:
    """Current Triton pointwise ABI uses flat, contiguous, non-broadcast inputs."""
    if not nodes:
        return False
    outputs = [_node_output(node) for node in nodes]
    if any(
        value is None
        or value.meta is None
        or not _is_contiguous(value.meta)
        or not _is_triton_device(value.meta)
        or not _is_supported_dtype(value.meta)
        for value in outputs
    ):
        return False
    output_shape = outputs[-1].meta.shape
    output_meta = outputs[-1].meta
    if any(
        value.meta.shape != output_shape
        or value.meta.dtype != output_meta.dtype
        or torch.device(value.meta.device) != torch.device(output_meta.device)
        for value in outputs
    ):
        return False
    inputs = external_values(nodes)
    if not inputs:
        return False
    return all(
        value.meta is not None
        and value.meta.shape == output_shape
        and value.meta.dtype == output_meta.dtype
        and _is_contiguous(value.meta)
        and _is_triton_device(value.meta)
        and _is_supported_dtype(value.meta)
        and torch.device(value.meta.device) == torch.device(output_meta.device)
        for value in inputs
    )


def escaping_values(nodes: list[IRNode]) -> list[TensorValue]:
    node_ids = {id(node) for node in nodes}
    escaping: list[TensorValue] = []
    for node in nodes:
        value = _node_output(node)
        if value is not None and any(id(user) not in node_ids for user in value.users):
            escaping.append(value)
    return escaping


def valid_single_output_group(nodes: list[IRNode]) -> bool:
    if not nodes:
        return False
    output = _node_output(nodes[-1])
    if output is None:
        return False
    return all(value is output for value in escaping_values(nodes))


def _has_codegen(node: IRNode, registry: BackendRegistry, backend="triton") -> bool:
    extension = registry.for_node(node)
    if extension is None:
        if not isinstance(node, PointwiseNode):
            if backend != "tilelang":
                return False
            from .codegen.tilelang.expressions import BACKWARD_OPS

            if getattr(node, "op", None) not in BACKWARD_OPS:
                return False
        if backend == "tilelang":
            from .codegen.tilelang.expressions import SUPPORTED_OPS

            if getattr(node, "op", None) not in SUPPORTED_OPS:
                return False
    elif not extension.pure or getattr(extension, f"{backend}_codegen", None) is None:
        return False
    args = getattr(node, "args", None)
    kwargs = getattr(node, "kwargs", None)
    return args is not None and kwargs is not None and _renderable((args, kwargs))


def _pair_can_fuse(
    producer: IRNode, consumer: IRNode, registry: BackendRegistry
) -> bool:
    left = _node_output(producer)
    right = _node_output(consumer)
    if left is None or right is None or left.meta is None or right.meta is None:
        return False
    if left.meta.shape != right.meta.shape:
        return False
    if not any(value is left for value in consumer.inputs()):
        return False
    for node in (producer, consumer):
        extension = registry.for_node(node)
        if extension is not None and extension.can_fuse is not None:
            if not extension.can_fuse(producer, consumer):
                return False
    return True


class Scheduler:
    """Build contiguous pointwise partitions with explicit fusion constraints."""

    def __init__(
        self,
        graph: IRGraph,
        *,
        registry: BackendRegistry | None = None,
        max_pointwise_fusion_length: int = 128,
        unsafe_values: frozenset[TensorValue] = frozenset(),
        kernel_policy=None,
        backward: bool = False,
    ):
        if max_pointwise_fusion_length < 1:
            raise ValueError("max_pointwise_fusion_length must be at least one")
        self.graph = graph
        self.registry = registry if registry is not None else BackendRegistry()
        self.max_pointwise_fusion_length = max_pointwise_fusion_length
        self.unsafe_values = unsafe_values
        from deepopt.kernel_policy import resolve_kernel_policy

        self.kernel_policy = resolve_kernel_policy(kernel_policy)
        self.backward = backward

    def backend_for(self, node):
        from deepopt.kernel_policy import kernel_backend, select_kernel_backend

        role = "compiler.backward" if self.backward else f"compiler.{getattr(node, 'op', 'pointwise')}"
        primary = kernel_backend("compiler_pointwise", backend=self.kernel_policy, role=role)
        extension = self.registry.for_node(node)
        output = _node_output(node)
        if (self.kernel_policy.backend == "mixed" and extension is not None and extension.pure
                and _is_triton_device(getattr(output, "meta", None))
                and (extension.triton_codegen is not None or extension.tilelang_codegen is not None)):
            eligibility = {
                backend: (None if getattr(extension, f"{backend}_codegen") is not None
                          else f"extension {extension.name!r} has no {backend} expression emitter")
                for backend in ("triton", "tilelang")
            }
            return select_kernel_backend("compiler_pointwise", eligibility,
                                         policy=self.kernel_policy, role=role)
        return primary

    def _has_codegen(self, node: IRNode) -> bool:
        output = _node_output(node)
        if output in self.unsafe_values:
            return False
        return _has_codegen(node, self.registry, self.backend_for(node))

    def _partition_region(self, region: list[IRNode]) -> list[list[IRNode]]:
        result: list[list[IRNode]] = []
        start = 0
        while start < len(region):
            chosen_end: int | None = None
            max_end = min(start + self.max_pointwise_fusion_length, len(region))
            for end in range(max_end, start, -1):
                candidate = region[start:end]
                if any(not self._has_codegen(node) for node in candidate):
                    continue
                if len({self.backend_for(node) for node in candidate}) != 1:
                    continue
                if any(
                    not _pair_can_fuse(left, right, self.registry)
                    for left, right in zip(candidate, candidate[1:])
                ):
                    continue
                if not valid_same_shape_region(
                    candidate
                ) or not valid_single_output_group(candidate):
                    continue
                chosen_end = end
                break
            if chosen_end is None:
                chosen_end = start + 1
            result.append(region[start:chosen_end])
            start = chosen_end
        return result

    def run(self) -> Schedule:
        self.graph.validate()
        barriers = frozenset(id(node) for node in self.graph.nodes if not self._reorderable(node))
        epochs = _effect_epochs(self.graph.nodes, barriers)
        self._resolve_fusion_constraints()
        schedule = Schedule(effect_epochs=epochs, effect_barriers=barriers)
        nodes = self.graph.nodes
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if not self._has_codegen(node):
                schedule.add(node)
                index += 1
                continue

            end = index
            region: list[IRNode] = []
            while end < len(nodes) and self._has_codegen(nodes[end]):
                region.append(nodes[end])
                end += 1

            for group in self._partition_region(region):
                if len(group) == 1:
                    if valid_same_shape_region(group):
                        partition = SinglePointwiseOp(group[0])
                        partition.kernel_backend = self.backend_for(group[0])
                        if self.backward:
                            partition.kernel_role = "compiler.backward"
                        schedule.add(partition)
                    else:
                        schedule.add(group[0])
                else:
                    partition = FusionGroup(group)
                    partition.kernel_backend = self.backend_for(group[0])
                    if self.backward:
                        partition.kernel_role = "compiler.backward"
                    schedule.add(partition)
            index = end
        schedule.validate()
        return schedule

    def _resolve_fusion_constraints(self):
        """Place replacement kernels early without crossing opaque effects.

        A fusion may consume a saved argument first declared later in the
        backward IR. Input bindings are freely movable; numerical dependencies
        and external operations remain barriers. No requested order overrides
        a real data dependency.
        """
        nodes = self.graph.nodes
        for node in list(nodes):
            constraints = node.scheduling_constraints
            preferred = set(constraints.get("prefer_before", ()))
            if not preferred:
                continue
            # This is deliberately a narrow physical-kernel contract. An
            # arbitrary extension cannot opt into reordering using a label.
            if getattr(node, "op", "") not in {"bf16.gemm_dx_native_derivative", "bf16.gemm_dx_packed_derivative"}:
                constraints["placement"] = "unsupported_reordering_contract"
                continue
            if not self._reorderable(node):
                constraints["placement"] = "unsafe_alias_or_version_constraint"
                continue
            current = nodes.index(node)
            targets = [index for index, item in enumerate(nodes[:current])
                       if any(value.name in preferred for value in item.outputs())]
            if not targets:
                constraints["placement"] = "no_preference_target"
                continue
            position = min(targets)
            names = {}
            for item in nodes:
                for value in item.outputs():
                    names.setdefault(value.name, []).append(item)
            declared = constraints.get("dependencies", ())
            if any(len(names.get(name, ())) != 1 for name in declared):
                constraints["placement"] = "unknown_or_ambiguous_dependency"
                continue
            inputs = [value.producer for value in node.inputs()]
            inputs += [names[name][0] for name in declared]
            dependencies = [nodes.index(item) for item in inputs if not isinstance(item, InputNode)]
            position = max(position, max(dependencies, default=-1) + 1)
            for index in range(position, current):
                item = nodes[index]
                if not self._reorderable(item):
                    position = index + 1
                    constraints["effect_barrier"] = getattr(item, "op", type(item).__name__)
            if position >= current:
                constraints["placement"] = "dependency_or_effect_limited"
                continue
            bindings = [item for item in nodes[position:current] if isinstance(item, InputNode)
                        and any(item is dependency for dependency in inputs)]
            moved = {id(item) for item in (*bindings, node)}
            prefix = [item for item in nodes[:position] if id(item) not in moved]
            suffix = [item for item in nodes[position:] if id(item) not in moved]
            nodes[:] = [*prefix, *bindings, node, *suffix]
            constraints["placement"] = "earliest_legal"
        self.graph.validate()

    def _reorderable(self, node):
        if any(value in self.unsafe_values for value in (*node.inputs(), *node.outputs())):
            return False
        if isinstance(node, (InputNode, GetAttrNode)):
            return True
        if isinstance(node, PointwiseNode):
            # Canonical out-of-place expressions only; RNG, view writes and
            # arbitrary Python targets have no commutation contract here.
            from .lowering import _SUPPORTED_POINTWISE

            return (node.op in _SUPPORTED_POINTWISE and not node.kwargs.get("inplace", False)
                    and "out" not in node.kwargs)
        return getattr(node, "op", "") in {
            "bf16.gemm_dx", "bf16.gemm_dw", "bf16.bias_reduce", "bf16.prepare_gradient",
            "bf16.gemm_dx_native_derivative", "bf16.gemm_dx_packed_derivative",
        }
