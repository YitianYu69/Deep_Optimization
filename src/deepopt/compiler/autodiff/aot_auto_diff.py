from __future__ import annotations

from ..ir import (
    ExternKernelNode,
    CallNode,
    GetAttrNode,
    IRGraph,
    InputNode,
    OutputNode,
    TensorValue,
    TensorMeta,
    find_tensor_values,
)

from ..trace.builder import GraphBuilder
from ..trace.metadata_utils import clone_meta
from ..trace.tracer import SymbolicTensor
from .activation_saving import ActivationSavePolicy, SavedActivationPolicy, SavedUse, resolve_activation_policy
from .backward_rules import GradWrapper

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Literal, Tuple, Dict, List


@dataclass(frozen=True)
class GradientTarget:
    value: TensorValue
    kind: Literal["input", "get_attr"]
    key: int | str


@dataclass(frozen=True)
class BackwardTraceResult:
    graph: IRGraph
    grad_output_source: Tuple[TensorValue, ...]
    saved_tensors: Tuple[TensorValue, ...]
    grad_targets: Tuple[GradientTarget, ...]
    saved_value_specs: tuple = ()

    @property
    def backward_argument_sources(self) -> Tuple[TensorValue, ...]:
        return self.grad_output_source + self.saved_tensors


class BackwardTraceContext:
    def __init__(
        self,
        builder: GraphBuilder,
        *,
        first_saved_input_index: int,
        activation_policy: ActivationSavePolicy,
        forward_graph: IRGraph | None = None,
        saved_activation_policy: SavedActivationPolicy | None = None,
    ):
        self.builder = builder
        self.activation_policy = activation_policy
        self.forward_graph = forward_graph
        self.saved_activation_policy = saved_activation_policy or SavedActivationPolicy()
        self.grad_map: Dict[TensorValue, TensorValue] = {}

        self._next_input_index = first_saved_input_index
        self._saved_aliases: Dict[TensorValue, SymbolicTensor] = {}
        self._recomputed_values: Dict[TensorValue, SymbolicTensor] = {}
        self._recomputing: set[TensorValue] = set()
        self._saved_forward_values: List[TensorValue] = []
        self._current_node_saved: List[TensorValue] = []
        self._encoded_aliases = {}
        self._encoded_components = {}
        self._residual_records = {}
        self._current_node = None
        self.saved_value_specs = []

    @property
    def saved_forward_value(self) -> Tuple[TensorValue, ...]:
        return tuple(self._saved_forward_values)

    def begin_node(self, node=None) -> None:
        self._current_node = node
        self._current_node_saved = []

    def finish_node(self) -> Tuple[TensorValue, ...]:
        result = tuple(self._current_node_saved)
        self._current_node_saved = []
        return result

    def needs_grad(self, value: Any) -> bool:
        return (
            isinstance(value, TensorValue)
            and value.meta is not None
            and bool(value.meta.requires_grad)
        )

    def saved(self, value: Any, *, use: SavedUse | None = None) -> Any:
        if not isinstance(value, TensorValue):
            return value
        if value.meta is None:
            raise ValueError(f"Cannot save TensorValue {value.name!r} without metadata")

        use = use or SavedUse()
        if value.provenance == "stochastic_depth_mask" and getattr(value.producer, "op", None) == "stochastic_depth.snapshot":
            use = SavedUse(role="stochastic_depth.scaled_mask", consumer=use.consumer,
                           rounding_domain="native_mask_dtype")
        recipe, reason = self.saved_activation_policy.select(value, use)
        if recipe is not None:
            result = self._store_codec(value, recipe)
            self.record_residual(value, self._encoded_components[(value, recipe)],
                representation="packed_saved_operand", use=use,
                recipe=self._encoded_components[(value, recipe)][0].producer.specification,
                component_roles=("payload", "codec_step", "codec_minimum"),
                retention_reason="consumer_permits_affine")
            return result

        if self.activation_policy.should_recompute(value.producer, value):
            rematerialized = self._recompute(value)
            if rematerialized is not None:
                self.record_residual(value, (), representation="recomputed", use=use,
                                     retention_reason="pure_pointwise_replay")
                return rematerialized

        result = self._store(value)
        components = self._encoded_components.get((value, "fp8"), (value,))
        self.record_residual(value, components, representation=("fp8_scaled" if (value, "fp8") in self._encoded_components
                             else value.meta.representation), use=use,
                             retention_reason=reason or "preserve_fp8_storage",
                             recipe=getattr(value.producer, "residual_recipe", ()))
        return result

    def record_residual(self, source, components, *, representation, use=None, recipe=(),
                        component_roles=(), retention_reason):
        """Export metadata only; reports never own runtime activation payloads."""
        import torch

        def freeze_metadata(value):
            if value is None or type(value) in (str, int, float, bool):
                return value
            if isinstance(value, (tuple, list)):
                return tuple(freeze_metadata(item) for item in value)
            if isinstance(value, dict):
                return tuple(sorted((key, freeze_metadata(item)) for key, item in value.items()))
            if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
                return str(value)
            raise TypeError(f"Residual recipes accept metadata only, got {type(value).__name__}")

        use = use or SavedUse()
        from collections.abc import Mapping
        from ..ir import model_storage_source, model_state_origin

        recipe = tuple(sorted(recipe.items())) if isinstance(recipe, Mapping) else tuple(recipe)
        recipe = freeze_metadata(recipe)
        components = tuple(components)
        if component_roles and len(component_roles) != len(components):
            raise ValueError("Each physical residual component requires one role")
        names = tuple(item.name if isinstance(item, TensorValue) else item for item in components)
        key = (source, names, representation, recipe, tuple(component_roles))
        record = self._residual_records.get(key)
        if record is None:
            meta = source.meta
            origin = model_state_origin(source)
            record = {"source": source.source_name or source.name, "physical_source": source.name,
                "provenance": source.provenance, "storage_source": model_storage_source(source),
                "model_origin": ({"source": origin[0], "shape": origin[1].shape,
                    "dtype": str(origin[1].dtype), "device": str(origin[1].device)} if origin is not None else None),
                "model_transforms": source.model_state_transforms,
                "producer": getattr(source.producer, "op", type(source.producer).__name__),
                "bf16_policy": source.producer.bf16_policy.as_dict() if source.producer is not None else None,
                "components": names, "representation": representation,
                "logical_shape": meta.shape, "logical_dtype": str(meta.dtype),
                "logical_device": str(meta.device),
                "logical_stride": meta.stride, "logical_layout": getattr(meta, "layout", None),
                "presence": getattr(meta, "presence", "required"), "recipe": recipe,
                "retention_reason": retention_reason, "consumers": [],
                "physical_components": tuple({"name": item.name,
                    "role": component_roles[index] if component_roles else "value",
                    "shape": item.meta.shape, "dtype": str(item.meta.dtype) if item.meta.dtype is not None else None,
                    "device": str(item.meta.device) if item.meta.device is not None else None,
                    "presence": getattr(item.meta, "presence", "required"),
                    "constructor": getattr(item.producer, "op", type(item.producer).__name__)}
                    for index, item in enumerate(components) if isinstance(item, TensorValue))}
            self.saved_value_specs.append(record)
            self._residual_records[key] = record
        node = self._current_node
        consumer = {"operation": use.consumer or getattr(node, "op", "extension"),
            "node": tuple(value.name for value in node.outputs()) if node is not None else (),
            "role": use.role, "allow_approximation": use.allow_approximation,
            "rounding_domain": use.rounding_domain, "retention_reason": retention_reason}
        if consumer not in record["consumers"]:
            record["consumers"].append(consumer)
        # Composite bindings name the same physical leaves, rather than making
        # their logical tensor an additional allocation in residual ledgers.
        for other in self.saved_value_specs:
            for parent, child in ((record, other), (other, record)):
                if (child is not parent and child.get("physical_source") in parent["components"]
                        and parent["components"] != (parent.get("physical_source"),)):
                    child["component_of"] = tuple(sorted(set((*child.get("component_of", ()), parent["source"]))))

    def record_codec(self, source, components, *, representation, recipe, role):
        """Declare an existing constructor ABI without touching its tensors."""
        self.record_residual(source, components, representation=representation,
            component_roles=("payload", "codec_step", "codec_minimum"), recipe=recipe,
            use=SavedUse(role=role, allow_approximation=True), retention_reason="declared_affine_residual")

    def _mark_saved(self, values):
        for value in values:
            if value not in self._current_node_saved:
                self._current_node_saved.append(value)

    def _store(self, value: TensorValue) -> SymbolicTensor:
        import torch

        if getattr(value.meta, "storage_dtype", None) in (torch.float8_e4m3fn, torch.float8_e5m2):
            return self._store_fp8(value)
        self._mark_saved((value,))

        existing = self._saved_aliases.get(value)
        if existing is not None:
            return existing

        alias = self.builder.input(
            self._next_input_index,
            clone_meta(value.meta, requires_grad=False),
            name=f"saved_{value.name}",
        )

        self._next_input_index += 1
        self._saved_aliases[value] = alias
        self._saved_forward_values.append(value)
        return alias

    def _store_fp8(self, value):
        """Export physical storage/scales rather than an opaque tensor wrapper."""
        import torch
        from math import prod

        key = (value, "fp8")
        if key in self._encoded_aliases:
            self._mark_saved(self._encoded_components[key])
            return self._encoded_aliases[key]
        if self.forward_graph is None:
            raise ValueError("Saving an FP8 value requires its semantic forward graph")
        metadata = value.meta
        shape = ((metadata.shape[0] * prod(metadata.shape[2:]), metadata.shape[1])
                 if metadata.layout == "channels" else (prod(metadata.shape[:-1]), metadata.shape[-1]) if metadata.shape else (1, 1))
        grid = ((1, 1) if metadata.block_shape is None else tuple(
            (size + tile - 1) // tile for size, tile in zip(shape, metadata.block_shape)))
        data = TensorValue(f"{value.name}__saved_fp8_data", TensorMeta(shape, metadata.storage_dtype,
            metadata.device, requires_grad=False, representation="fp8_storage"))
        inverse = TensorValue(f"{value.name}__saved_fp8_scale", TensorMeta(grid, torch.float32,
            metadata.device, requires_grad=False, representation="fp8_storage"))
        extract = CallNode("fp8.saved_components", lambda x: (x._data, x._scale_inv), (value,), {}, (data, inverse))
        graph = self.forward_graph
        index = graph.nodes.index(value.producer) + 1
        graph.add(extract)
        graph.nodes.pop()
        graph.nodes.insert(index, extract)
        stored_data, stored_inverse = self._store(data), self._store(inverse)

        def restore(data, inverse):
            from deepopt.opt.quantization.activation_tensor import FP8Activation

            return FP8Activation(data, inverse, metadata.shape, metadata.block_shape,
                                 metadata.layout, metadata.dtype, metadata.stride)

        logical = self.builder.emit_native("fp8.restore_saved", restore, (stored_data, stored_inverse), {},
            replace(metadata, requires_grad=False))
        self._encoded_aliases[key] = logical
        self._encoded_components[key] = (data, inverse)
        return logical

    def _store_codec(self, value, recipe):
        """Resolve an explicitly requested approximate save into codec IR."""
        import torch
        from math import prod
        from deepopt.opt.quantization.activation import _dimensions

        key = (value, recipe)
        if key in self._encoded_aliases:
            self._mark_saved(self._encoded_components[key])
            return self._encoded_aliases[key]
        if self.forward_graph is None:
            raise ValueError("Saving an encoded coefficient requires its semantic forward graph")
        metadata, options = value.meta, dict(recipe)
        from ..trace.metadata_utils import contiguous_stride

        restore_stride = contiguous_stride(metadata.shape)
        if metadata.stride is not None:
            dimensions = sorted((stride, size) for stride, size in zip(metadata.stride, metadata.shape) if size > 1)
            expected, dense = 1, True
            for stride, size in dimensions:
                dense = dense and stride == expected
                expected *= size
            if dense:
                restore_stride = metadata.stride
        shape = (prod(metadata.shape[:-1]), metadata.shape[-1]) if metadata.shape else (1, 1)
        group, bits = options["group_size"], options["bits"]
        groups, _, words = _dimensions(shape, group, None, bits)
        components = tuple(TensorValue(f"{value.name}__saved_b{bits}_g{group}__{suffix}", TensorMeta(component_shape, dtype, metadata.device,
            requires_grad=False, representation="packed" if suffix == "payload" else "dense"))
            for suffix, component_shape, dtype in (("payload", (groups, words), torch.int32),
                                                  ("step", (groups,), torch.float32), ("minimum", (groups,), torch.float32)))
        def pack(x):
            from deepopt.opt.quantization.activation import _quantize_pack_strided_activation

            packed = _quantize_pack_strided_activation(x if x.ndim >= 2 else x.reshape(shape), group, bits=bits)
            return packed.data, packed.step, packed.minimum
        from types import MappingProxyType

        specification = MappingProxyType({**options, "logical_shape": metadata.shape,
            "matrix_shape": shape,
            "bf16_policy": (value.producer.bf16_policy.as_dict() if value.producer is not None else
                            {"name": "torch_amp", "version": 1}),
            "logical_dtype": metadata.dtype, "logical_stride": metadata.stride,
            "restored_stride": restore_stride, "traversal": "logical_row_major",
            "rounding_domain": "operand_dtype", "source_cast": None})
        extract = CallNode("codec.save_activation", pack, (value,), {}, components, specification)
        if value.producer is not None:
            extract.bf16_policy = value.producer.bf16_policy
        graph = self.forward_graph
        index = graph.nodes.index(value.producer) + 1
        graph.add(extract)
        graph.nodes.pop()
        graph.nodes.insert(index, extract)
        stored = tuple(self._store(component) for component in components)
        def restore(data, step, minimum):
            from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

            result = dequantize_unpack_activation(PackedActivation(data, step, minimum, shape, group, bits=bits),
                                                dtype=metadata.dtype, row_alignment=1).reshape(metadata.shape)
            if result.stride() != restore_stride:
                # Preserve dense permutations (including channels-last). Views
                # with holes/zero strides are restored compactly: expanding
                # them would cost more storage and overlapping copies are not legal.
                restored = torch.empty_strided(metadata.shape, restore_stride, dtype=result.dtype, device=result.device)
                restored.copy_(result)
                result = restored
            return result
        result = self.builder.new_value(replace(metadata, stride=restore_stride, requires_grad=False))
        self.builder.graph.add(CallNode("codec.restore_activation", restore,
            tuple(item.value for item in stored), {}, (result,), specification))
        logical = self.builder.wrap(result)
        self._encoded_aliases[key] = logical
        self._encoded_components[key] = components
        return logical

    def _map_tree(self, value: Any) -> Any:
        if isinstance(value, TensorValue):
            return self.saved(value)
        if isinstance(value, tuple):
            return tuple(self._map_tree(item) for item in value)
        if isinstance(value, list):
            return [self._map_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: self._map_tree(item) for key, item in value.items()}
        return value

    def _recompute(self, value: TensorValue) -> SymbolicTensor | None:
        cached = self._recomputed_values.get(value)
        if cached is not None:
            # Reuse execution, but register the requesting node's physical
            # leaves and consumer records on every logical replay use.
            self._map_tree(value.producer.args)
            self._map_tree(value.producer.kwargs)
            return cached
        producer = value.producer
        from .activation_saving import pure_pointwise_replay

        if not pure_pointwise_replay(producer) or value in self._recomputing:
            return None

        self._recomputing.add(value)
        try:
            args = self._map_tree(producer.args)
            kwargs = self._map_tree(producer.kwargs)
            # FX call_module nodes keep activation parameters on the module
            # target rather than in args/kwargs. Preserve them when rebuilding
            # a pointwise value for the backward graph.
            for name in (
                "approximate",
                "negative_slope",
                "min_val",
                "max_val",
                "beta",
                "threshold",
            ):
                if name not in kwargs and hasattr(producer.target, name):
                    kwargs[name] = getattr(producer.target, name)
            rematerialized = self.builder.emit(producer.op, *args, **kwargs)
            if (
                rematerialized.value.meta is None
                or rematerialized.value.meta.shape != value.meta.shape
                or rematerialized.value.meta.dtype != value.meta.dtype
                or rematerialized.value.meta.device != value.meta.device
                or rematerialized.value.meta.representation != value.meta.representation
                or any(getattr(rematerialized.value.meta, name, None) != getattr(value.meta, name, None)
                       for name in ("storage_dtype", "block_shape", "layout"))
            ):
                return None
            self._recomputed_values[value] = rematerialized
            return rematerialized
        except (NotImplementedError, TypeError, ValueError, KeyError):
            return None
        finally:
            self._recomputing.discard(value)

    def _as_symbolic(self, grad: SymbolicTensor | TensorValue) -> SymbolicTensor:
        return self.builder.wrap(grad)

    def _project_to_target(
        self, grad: SymbolicTensor, target: TensorValue
    ) -> SymbolicTensor:
        if target.meta is None or grad.value.meta is None:
            raise ValueError("Gradient projection requires complete TensorMeta")

        if grad.value.meta.shape != target.meta.shape:
            grad = self.builder.emit("sum_to_size", grad, target.meta.shape)

        if grad.value.meta.shape != target.meta.shape:
            raise ValueError(
                f"Gradient shape mismatch for {target.name}"
                f"got {grad.value.meta.shape}, expected {target.meta.shape}"
            )
        if grad.value.meta.dtype != target.meta.dtype:
            from dataclasses import replace
            import torch

            grad = self.builder.emit_native(
                "autodiff.cast", torch.ops.aten.to.dtype,
                (grad, target.meta.dtype), {},
                replace(grad.value.meta, dtype=target.meta.dtype),
            )
        return grad

    def accumulate(
        self, target: Any, grad: SymbolicTensor | TensorValue | None
    ) -> None:
        if grad is None or not self.needs_grad(target):
            return
        assert isinstance(target, TensorValue)

        projected = self._project_to_target(self._as_symbolic(grad), target)
        previous = self.grad_map.get(target)
        if previous is None:
            self.grad_map[target] = projected.value
        else:
            self.grad_map[target] = self.builder.add(previous, projected).value


class AOTAutoDiff:
    """
    Trace a first-order backward IR from a forward IRGraph

    Current contract:
        * multi-result rules receive an optional cotangent for each output;
        * every differential dependency is explicit in the node.args/node.kwargs;
        * forward TensorMeta is complete;
        * side-effecting/in-place ops are not accepted
    """

    def __init__(
        self,
        graph: IRGraph,
        *,
        activation_policy: str | ActivationSavePolicy | None = "save",
        saved_activation_policy: SavedActivationPolicy | None = None,
        registry=None,
    ):
        self.fwd_graph = graph
        self.activation_policy = resolve_activation_policy(activation_policy)
        self.saved_activation_policy = saved_activation_policy or SavedActivationPolicy()
        from ..registry import BackendRegistry

        self.registry = (registry if registry is not None else BackendRegistry()).copy()

    def _output_node(self) -> OutputNode:
        outputs = [n for n in self.fwd_graph.nodes if isinstance(n, OutputNode)]
        if len(outputs) != 1:
            raise ValueError(f"Expected exactly one OutputNode, found {len(outputs)}")
        return outputs[0]

    def _grad_targets(self) -> Tuple[GradientTarget, ...]:
        targets: List[GradientTarget] = []
        for node in self.fwd_graph.nodes:
            if isinstance(node, InputNode):
                if node.value.meta is not None and node.value.meta.requires_grad:
                    targets.append(GradientTarget(node.value, "input", node.index))
            elif isinstance(node, GetAttrNode):
                if node.value.meta is not None and node.value.meta.requires_grad:
                    targets.append(GradientTarget(node.value, "get_attr", node.target))
        return tuple(targets)

    def diff(self) -> BackwardTraceResult:
        self.fwd_graph.validate()
        bwd_graph = IRGraph()
        builder = GraphBuilder(bwd_graph, value_prefix="g")

        output_node = self._output_node()
        differentiable_outputs = tuple(
            value
            for value in find_tensor_values(output_node.result)
            if value.meta is not None and value.meta.requires_grad
        )

        ctx = BackwardTraceContext(
            builder,
            first_saved_input_index=len(differentiable_outputs),
            activation_policy=self.activation_policy,
            forward_graph=self.fwd_graph,
            saved_activation_policy=self.saved_activation_policy,
        )

        # Seed dL/d(output). Duplicaate forward outputs intentionally create
        # duplicate seeds and are accumulated correctly
        for index, fwd_output in enumerate(differentiable_outputs):
            from ..trace.metadata_utils import contiguous_stride

            seed = builder.input(
                index,
                replace(fwd_output.meta, requires_grad=True, stride=contiguous_stride(fwd_output.meta.shape)),
                name=f"grad_{fwd_output.name}_{index}",
            )

            ctx.accumulate(fwd_output, seed)

        for node in reversed(tuple(self.fwd_graph.nodes)):
            if isinstance(node, (OutputNode, InputNode, GetAttrNode)):
                continue

            outputs = node.outputs()
            if not outputs:
                continue

            grad_outputs = tuple(ctx.grad_map.get(output) for output in outputs)
            if all(gradient is None for gradient in grad_outputs):
                continue

            differentiable_inputs = [v for v in node.inputs() if ctx.needs_grad(v)]
            if not differentiable_inputs:
                continue

            op_name = getattr(node, "op", None)
            if not op_name:
                raise NotImplementedError(
                    f"Differentiable node {type(node).__name__} producing {outputs[0].name!r}"
                    f"has not stable op name. Canonicalize it during forward lowering"
                )

            rule = None
            if isinstance(node, ExternKernelNode):
                from .native_layer_backward_rules import get_native_layer_backward_rule

                rule = get_native_layer_backward_rule(node)
            if rule is None:
                try:
                    rule = self.registry.backward_rule(op_name)
                except NotImplementedError as ext:
                    raise NotImplementedError(
                        f"No backward rule for op {op_name!r}; node={node!r}"
                    ) from ext

            ctx.begin_node(node)
            first_derivative_node = len(bwd_graph.nodes)
            if self.registry.multi_output_backward_rule(op_name):
                cotangents = tuple(builder.wrap(value) if value is not None else None
                                   for value in grad_outputs)
                contributions = tuple(rule(ctx, node, cotangents))
            elif len(outputs) == 1:
                contributions = tuple(rule(ctx, node, builder.wrap(grad_outputs[0])))
            else:
                raise NotImplementedError(
                    f"Multi-output operator {op_name!r} requires a multi-output VJP rule"
                )
            node.backward_rule = op_name
            node.saved_tensors_for_bwd = ctx.finish_node()
            node.activation_save_policy = self.activation_policy.name

            valid_inputs = set(node.inputs())
            for contribution in contributions:
                if not isinstance(contribution, GradWrapper):
                    raise TypeError(
                        f"Backward rule {op_name!r} returned "
                        f"{type(contribution).__name__}, expected GradWrapper"
                    )
                target = contribution.target
                if isinstance(target, TensorValue) and target not in valid_inputs:
                    raise RuntimeError(
                        f"Backward rule {op_name!r} produced a gradient for "
                        f"non-input TensorValue {target.name!r}"
                    )
                ctx.accumulate(target, contribution.grad)
            for derivative_node in bwd_graph.nodes[first_derivative_node:]:
                derivative_node.bf16_policy = node.bf16_policy
                for attribute in ("module_path", "source_node"):
                    if getattr(derivative_node, attribute, None) is None:
                        setattr(derivative_node, attribute, getattr(node, attribute, None))

        grad_targets = self._grad_targets()
        grad_results = tuple(ctx.grad_map.get(target.value) for target in grad_targets)
        builder.output(grad_results)
        bwd_graph.validate()

        return BackwardTraceResult(
            graph=bwd_graph,
            grad_output_source=differentiable_outputs,
            saved_tensors=ctx.saved_forward_value,
            grad_targets=grad_targets,
            saved_value_specs=tuple(ctx.saved_value_specs),
        )


AOT_AUTO_Diff = AOTAutoDiff
