"""Runtime glue between the FX compiler and the project's AOT autodiff.

Generated Triton partitions and supported native layer calls are autograd
boundaries. Triton backward is traced by :class:`AOTAutoDiff`; native layer
backward IR contains registered direct ATen backward calls. Other native
operations retain their ordinary eager autograd behavior.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from .autodiff import AOTAutoDiff
from .ir import (
    ExternKernelNode,
    IRGraph,
    InputNode,
    IRNode,
    OutputNode,
    PointwiseNode,
    TensorValue,
)
from .registry import BackendRegistry
from .scheduler import FusionGroup, Scheduler, SinglePointwiseOp, external_values
from .codegen.runtime import launch_pointwise


def _partition_nodes(partition: SinglePointwiseOp | FusionGroup) -> list[IRNode]:
    if isinstance(partition, SinglePointwiseOp):
        return [partition.node]
    return list(partition.nodes)


def _map_tree(value: Any, values: dict[TensorValue, TensorValue]) -> Any:
    if isinstance(value, TensorValue):
        try:
            return values[value]
        except KeyError as exc:
            raise ValueError(
                f"Unmapped TensorValue {value.name!r} in Triton partition"
            ) from exc
    if isinstance(value, tuple):
        return tuple(_map_tree(item, values) for item in value)
    if isinstance(value, list):
        return [_map_tree(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _map_tree(item, values) for key, item in value.items()}
    return value


def _build_partition_graph(
    partition: SinglePointwiseOp | FusionGroup,
) -> tuple[IRGraph, list[TensorValue]]:
    """Copy one scheduled local region into a self-contained single-output IR."""
    nodes = _partition_nodes(partition)
    inputs = external_values(nodes)
    graph = IRGraph()
    value_map: dict[TensorValue, TensorValue] = {}

    for index, value in enumerate(inputs):
        if value.meta is None:
            raise ValueError(
                f"Cannot autodiff input {value.name!r} without tensor metadata"
            )
        local_value = TensorValue(name=value.name, meta=value.meta)
        graph.add(InputNode(index=index, value=local_value))
        value_map[value] = local_value

    for node in nodes:
        output = node.output()
        if output is None or output.meta is None:
            raise NotImplementedError(
                f"AOT autodiff requires one metadata-bearing output per Triton node; "
                f"got {type(node).__name__}"
            )
        cloned = copy.copy(node)
        if (
            not hasattr(cloned, "args")
            or not hasattr(cloned, "kwargs")
            or not hasattr(cloned, "value")
        ):
            raise NotImplementedError(
                f"Triton node {type(node).__name__} must expose args, kwargs, and value "
                "to participate in AOT autodiff"
            )
        cloned.args = _map_tree(node.args, value_map)
        cloned.kwargs = _map_tree(node.kwargs, value_map)
        cloned.value = TensorValue(name=output.name, meta=output.meta)
        cloned.backward_rule = None
        cloned.saved_tensors_for_bwd = ()
        cloned.activation_save_policy = None
        graph.add(cloned)
        value_map[output] = cloned.value

    graph.add(OutputNode(result=value_map[partition.output_value]))
    graph.validate()
    return graph, inputs


def _map_native_args(value: Any, values: dict[TensorValue, torch.Tensor]) -> Any:
    if isinstance(value, TensorValue):
        return values[value]
    if isinstance(value, tuple):
        return tuple(_map_native_args(item, values) for item in value)
    if isinstance(value, list):
        return [_map_native_args(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _map_native_args(item, values) for key, item in value.items()}
    return value


@dataclass
class NativeAutogradRegion:
    """A native layer forward paired with an ATen-backed AOT backward graph."""

    target: Any
    args_template: Any
    kwargs_template: Any
    input_values: tuple[TensorValue, ...]
    backward_fn: Any | None
    backward_graph: IRGraph | None
    saved_input_indices: tuple[int, ...]
    gradient_target_input_indices: tuple[int, ...]
    buffer_input_indices: tuple[int, ...]
    input_count: int
    kernel_policy: Any = None

    def _forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self.input_count:
            raise TypeError(
                f"Native layer region expects {self.input_count} tensor inputs, "
                f"got {len(inputs)}"
            )
        values = dict(zip(self.input_values, inputs))
        args = _map_native_args(self.args_template, values)
        kwargs = _map_native_args(self.kwargs_template, values)
        output = self.target(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError(
                "AOT native-layer regions currently require a single tensor output"
            )
        return output

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self.input_count:
            raise TypeError(
                f"Native layer region expects {self.input_count} tensor inputs, "
                f"got {len(inputs)}"
            )
        from torch._subclasses.fake_tensor import FakeTensor

        for value, expected in zip(inputs, self.input_values):
            meta = expected.meta
            if meta is None:
                continue
            if (
                type(value) not in (torch.Tensor, torch.nn.Parameter, FakeTensor)
                or value.dtype != meta.dtype
                or tuple(value.shape) != meta.shape
                or meta.stride is not None
                and tuple(value.stride()) != meta.stride
                or value.requires_grad != meta.requires_grad
            ):
                raise ValueError(
                    "Native AOT input metadata changed; recompile the graph"
                )
        if self.backward_fn is None or not any(t.requires_grad for t in inputs):
            return self._forward(*inputs)
        return _AOTRegionFunction.apply(self, *inputs)


@dataclass
class AutogradKernelRegion:
    """A Triton forward kernel and its AOT-compiled first-order backward."""

    forward_artifact: Any
    backward_fn: Any | None
    saved_input_indices: tuple[int, ...]
    gradient_target_input_indices: tuple[int, ...]
    input_count: int
    block_size: int
    kernel_policy: Any = None

    def _forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        output_meta = self.forward_artifact.output.meta
        if output_meta is None:
            raise ValueError(
                f"Output {self.forward_artifact.output.name!r} has no metadata"
            )
        if len(inputs) != self.input_count:
            raise TypeError(f"Triton region expects {self.input_count} tensor inputs")
        if not inputs:
            raise RuntimeError("A Triton region must have at least one tensor input")
        output = torch.empty(
            tuple(output_meta.shape),
            dtype=output_meta.dtype,
            device=inputs[0].device,
        )
        launch_pointwise(self.forward_artifact, inputs, output, self.block_size)
        return output

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.backward_fn is None or not any(t.requires_grad for t in inputs):
            return self._forward(*inputs)
        return _AOTRegionFunction.apply(self, *inputs)


class _AOTRegionFunction(torch.autograd.Function):
    """Attach a traced backward graph to one Triton or native layer region."""

    @staticmethod
    def forward(ctx, region: AutogradKernelRegion, *inputs: torch.Tensor):
        ctx.region = region
        # A normalization forward updates running statistics. Preserve their
        # pre-forward values for the explicit ATen backward while calling the
        # forward on the caller's original buffers so the update is visible.
        buffers = set(getattr(region, "buffer_input_indices", ()))
        ctx.save_for_backward(
            *(
                inputs[index].detach().clone(memory_format=torch.preserve_format)
                if index in buffers
                else inputs[index]
                for index in region.saved_input_indices
            )
        )
        return region._forward(*inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        region: AutogradKernelRegion = ctx.region
        saved = ctx.saved_tensors
        # Upstream autograd can supply an expanded, zero-stride gradient (for
        # example after sum()). The compiled backward ABI assumes the dense
        # contiguous layout described by its output metadata.
        from deepopt.kernel_policy import kernel_policy_scope

        with kernel_policy_scope(getattr(region, "kernel_policy", None)):
            raw_gradients = region.backward_fn(grad_output.contiguous(), *saved)
        if not isinstance(raw_gradients, tuple):
            raw_gradients = (raw_gradients,)

        input_gradients: list[torch.Tensor | None] = [None] * region.input_count
        for input_index, gradient in zip(
            region.gradient_target_input_indices, raw_gradients
        ):
            input_gradients[input_index] = gradient
        return (None, *input_gradients)


def build_autograd_region(
    partition: SinglePointwiseOp | FusionGroup,
    forward_artifact: Any,
    graph_module: torch.fx.GraphModule,
    *,
    registry: BackendRegistry,
    cache_dir,
    block_size: int,
    kernel_policy=None,
) -> AutogradKernelRegion:
    """Trace and compile a backward graph for one Triton partition."""
    forward_graph, forward_inputs = _build_partition_graph(partition)
    trace = AOTAutoDiff(
        forward_graph, activation_policy="recompute", registry=registry
    ).diff()

    saved_input_indices: list[int] = []
    # AOT traces the cloned region graph, so saved values have local identities
    # rather than the identities of the enclosing graph's external inputs.
    input_indices = {
        node.value: node.index
        for node in forward_graph.nodes
        if isinstance(node, InputNode)
    }
    for value in trace.saved_tensors:
        try:
            saved_input_indices.append(input_indices[value])
        except KeyError as exc:
            raise NotImplementedError(
                f"Activation policy retained internal value {value.name!r}; "
                "the base runtime saves region inputs and recomputes pointwise intermediates"
            ) from exc

    gradient_target_input_indices: list[int] = []
    for target in trace.grad_targets:
        if target.kind != "input":
            raise NotImplementedError(
                "A Triton partition backward target must be a region input"
            )
        gradient_target_input_indices.append(int(target.key))

    backward_fn = None
    if trace.grad_output_source and trace.grad_targets:
        backward_schedule = Scheduler(
            trace.graph,
            registry=registry,
            kernel_policy=kernel_policy,
            backward=True,
        ).run()
        # Backward partitions are ordinary graph operations. They must not be
        # wrapped in another custom autograd Function during this first-order
        # implementation.
        from .codegen.wrapper import WrapperCodegen

        backward_artifact = (
            WrapperCodegen(
                graph_module,
                backward_schedule,
                registry=registry,
                cache_dir=cache_dir,
                block_size=block_size,
                enable_autodiff=False,
                kernel_policy=kernel_policy,
            )
            .run()
        )
        backward_fn = backward_artifact.fn
        backward_fn.backend_artifact = backward_artifact

    return AutogradKernelRegion(
        forward_artifact=forward_artifact,
        backward_fn=backward_fn,
        saved_input_indices=tuple(saved_input_indices),
        gradient_target_input_indices=tuple(gradient_target_input_indices),
        input_count=len(forward_inputs),
        block_size=block_size,
        kernel_policy=kernel_policy,
    )


def is_native_layer_node(node: IRNode) -> bool:
    """Whether an external node has a registered direct ATen backward rule."""
    from .autodiff.native_layer_backward_rules import is_registered_native_layer_node

    return is_registered_native_layer_node(node)


def is_native_pointwise_node(node: IRNode, registry: BackendRegistry) -> bool:
    """Use this project's AOT rule for pointwise ops without a Triton kernel."""
    if not isinstance(node, PointwiseNode) or not callable(node.target):
        return False
    if node.value.meta is None or not node.value.meta.requires_grad:
        return False
    if node.value.meta.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return False
    if not node.inputs() or any(value.meta is None for value in node.inputs()):
        return False
    # The baseline rules do not sum broadcasted gradients or model promotion.
    # Leave those operations on PyTorch autograd until there is an exact rule.
    if any(
        value.meta.shape != node.value.meta.shape
        or value.meta.dtype != node.value.meta.dtype
        or value.meta.representation != "dense"
        for value in node.inputs()
    ):
        return False
    return registry.has_backward_rule(node.op)


def build_native_autograd_region(
    node: IRNode,
    graph_module: torch.fx.GraphModule,
    *,
    registry: BackendRegistry,
    cache_dir,
    block_size: int,
    kernel_policy=None,
) -> NativeAutogradRegion:
    """Trace and compile a native forward with this project's AOT backward."""
    if not isinstance(node, (ExternKernelNode, PointwiseNode)) or node.value is None:
        raise TypeError("Native AOT region requires one tensor-producing node")
    if node.value.meta is None:
        raise ValueError(f"Native layer output {node.value.name!r} has no TensorMeta")

    original_inputs = tuple(dict.fromkeys(node.inputs()))
    graph = IRGraph()
    value_map: dict[TensorValue, TensorValue] = {}
    local_inputs: list[TensorValue] = []
    for index, value in enumerate(original_inputs):
        if value.meta is None:
            raise ValueError(f"Native layer input {value.name!r} has no TensorMeta")
        local = TensorValue(name=value.name, meta=value.meta)
        graph.add(InputNode(index=index, value=local))
        value_map[value] = local
        local_inputs.append(local)

    cloned = copy.copy(node)
    cloned.args = _map_tree(node.args, value_map)
    cloned.kwargs = _map_tree(node.kwargs, value_map)
    if isinstance(node, ExternKernelNode):
        cloned.autograd_inputs = tuple(
            value_map[value] for value in node.autograd_inputs
        )
        cloned.autograd_buffers = tuple(
            value_map[value] for value in node.autograd_buffers
        )
    cloned.value = TensorValue(name=node.value.name, meta=node.value.meta)
    cloned.backward_rule = None
    cloned.saved_tensors_for_bwd = ()
    cloned.activation_save_policy = None
    graph.add(cloned)
    graph.add(OutputNode(result=cloned.value))
    graph.validate()

    trace = AOTAutoDiff(graph, activation_policy="recompute", registry=registry).diff()
    input_indices = {value: index for index, value in enumerate(local_inputs)}
    saved_input_indices = tuple(input_indices[value] for value in trace.saved_tensors)
    gradient_target_input_indices: list[int] = []
    for target in trace.grad_targets:
        if target.kind != "input":
            raise NotImplementedError(
                "A native layer backward target must map to an explicit region input"
            )
        gradient_target_input_indices.append(int(target.key))

    backward_fn = None
    backward_graph = None
    if trace.grad_output_source and trace.grad_targets:
        backward_schedule = Scheduler(trace.graph, registry=registry, kernel_policy=kernel_policy, backward=True).run()
        from .codegen.wrapper import WrapperCodegen

        backward_artifact = WrapperCodegen(
            graph_module,
            backward_schedule,
            registry=registry,
            cache_dir=cache_dir,
            block_size=block_size,
            enable_autodiff=False,
            kernel_policy=kernel_policy,
        ).run()
        backward_fn = backward_artifact.fn
        backward_fn.backend_artifact = backward_artifact
        backward_graph = trace.graph

    return NativeAutogradRegion(
        target=node.target,
        args_template=node.args,
        kwargs_template=node.kwargs,
        input_values=original_inputs,
        backward_fn=backward_fn,
        backward_graph=backward_graph,
        saved_input_indices=saved_input_indices,
        gradient_target_input_indices=tuple(gradient_target_input_indices),
        buffer_input_indices=tuple(
            input_indices[value_map[value]]
            for value in (
                node.autograd_buffers
                if isinstance(node, ExternKernelNode) and node.autograd_buffers
                else tuple(
                    value for value in node.args[1:3] if isinstance(value, TensorValue)
                )
                if isinstance(node, ExternKernelNode)
                and node.target
                in {
                    torch.nn.functional.batch_norm,
                    torch.nn.functional.instance_norm,
                }
                else ()
            )
        ),
        input_count=len(original_inputs),
        kernel_policy=kernel_policy,
    )
