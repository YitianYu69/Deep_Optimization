from __future__ import annotations

import functools
import operator
from collections.abc import Mapping
from typing import Any

import torch
import torch.fx as fx

from .ir import (
    ExternKernelNode,
    GetAttrNode,
    InputNode,
    IRGraph,
    IRNode,
    OutputNode,
    PointwiseNode,
    TensorMeta,
    TensorValue,
)
from .registry import BackendRegistry, OperatorExtension


_FUNCTION_KINDS = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.truediv: "div",
    operator.neg: "neg",
    torch.add: "add",
    torch.sub: "sub",
    torch.mul: "mul",
    torch.multiply: "mul",
    torch.multiply: "mul",
    torch.div: "div",
    torch.neg: "neg",
    torch.relu: "relu",
    torch.nn.functional.relu: "relu",
    torch.sigmoid: "sigmoid",
}
_FUNCTIONAL_KINDS = {
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "gelu": "gelu",
    "silu": "silu",
    "leaky_relu": "leaky_relu",
    "hardtanh": "hardtanh",
    "hardswish": "hardswish",
    "hardsigmoid": "hardsigmoid",
    "softplus": "softplus",
    "mish": "mish",
}
for _name, _kind in _FUNCTIONAL_KINDS.items():
    for _namespace in (torch, torch.nn.functional):
        _target = getattr(_namespace, _name, None)
        if _target is not None:
            _FUNCTION_KINDS[_target] = _kind

_METHOD_KINDS = {
    name: name
    for name in (
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "relu",
        "sigmoid",
        "tanh",
        "gelu",
        "silu",
        "leaky_relu",
        "hardtanh",
        "hardswish",
        "hardsigmoid",
        "softplus",
        "mish",
    )
}
_METHOD_KINDS["multiply"] = "mul"
_FUNCTION_KINDS[torch.Tensor.mul] = "mul"
_FUNCTION_KINDS[torch.Tensor.multiply] = "mul"
_METHOD_KINDS["multiply"] = "mul"
_SUPPORTED_POINTWISE = frozenset(_METHOD_KINDS.values())
_MODULE_KINDS = {}
for _module_name, _kind in (
    ("ReLU", "relu"),
    ("Sigmoid", "sigmoid"),
    ("Tanh", "tanh"),
    ("GELU", "gelu"),
    ("SiLU", "silu"),
    ("LeakyReLU", "leaky_relu"),
    ("Hardtanh", "hardtanh"),
    ("Hardswish", "hardswish"),
    ("Hardsigmoid", "hardsigmoid"),
    ("Softplus", "softplus"),
    ("Mish", "mish"),
):
    _module_type = getattr(torch.nn, _module_name, None)
    if _module_type is not None:
        _MODULE_KINDS[_module_type] = _kind


def op_kind(target: Any) -> str | None:
    """Return a canonical name for supported pointwise functions and ATen ops."""
    try:
        if target in _FUNCTION_KINDS:
            return _FUNCTION_KINDS[target]
        if target in _METHOD_KINDS:
            return _METHOD_KINDS[target]
    except TypeError:
        pass

    text = str(target)
    for name in _SUPPORTED_POINTWISE:
        if text.startswith(f"aten.{name}.") or text == f"torch.{name}":
            return name
    return None


def _target_kind(node: fx.Node, gm: fx.GraphModule) -> str | None:
    if node.op in ("call_function", "call_method"):
        return op_kind(node.target)
    if node.op == "call_module":
        module = gm.get_submodule(node.target)
        return _MODULE_KINDS.get(type(module))
    return None


def is_pointwise_fx_node(node: fx.Node, gm: fx.GraphModule) -> bool:
    if node.op not in ("call_function", "call_method", "call_module"):
        return False
    kind = _target_kind(node, gm)
    if kind is None:
        return False

    if node.op == "call_module":
        module = gm.get_submodule(node.target)
        if bool(getattr(module, "inplace", False)):
            return False
    if kind in ("add", "sub") and node.kwargs.get("alpha", 1) != 1:
        return False
    # In-place operators mutate aliases and must stay on the native path.
    if node.kwargs.get("inplace", False):
        return False
    allowed = {
        "add": {"alpha"},
        "sub": {"alpha"},
        "gelu": {"approximate"},
        "leaky_relu": {"negative_slope"},
        "hardtanh": {"min_val", "max_val"},
        "softplus": {"beta", "threshold"},
    }.get(kind, set())
    allowed.update(("input", "self"))
    if kind in ("add", "sub", "mul", "div"):
        allowed.add("other")
    if "inplace" in node.kwargs:
        allowed.add("inplace")
    return not (set(node.kwargs) - allowed)


def _meta_from_tensor(
    value: Any, fallback_device: torch.device | str | None = None
) -> TensorMeta | None:
    if value is None or not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    try:
        device = getattr(value, "device", None) or fallback_device
        stride = value.stride() if callable(getattr(value, "stride", None)) else None
        # Symbolic dimensions must not acquire equality guards as a side effect
        # of metadata inspection. Unknown shapes remain native operations.
        if any(type(dim) is not int for dim in value.shape):
            return None
        if stride is not None and any(type(item) is not int for item in stride):
            return None
        from torch._subclasses.fake_tensor import FakeTensor

        dense = type(value) in (torch.Tensor, torch.nn.Parameter, FakeTensor)
        representation = "dense" if dense else "opaque"
        storage = getattr(value, "_data", None) if not dense else None
        return TensorMeta(
            shape=tuple(int(dim) for dim in value.shape),
            dtype=value.dtype,
            device=device,
            stride=tuple(int(item) for item in stride) if stride is not None else None,
            requires_grad=bool(getattr(value, "requires_grad", False)),
            representation=representation,
            storage_dtype=getattr(storage, "dtype", None),
            block_shape=getattr(value, "_block_shape", None) if not dense else None,
            layout=getattr(value, "_activation_layout", None) if not dense else None,
        )
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None


def metadata_from_fx(
    node: fx.Node,
    fallback_device: torch.device | str | None = None,
    fallback_tensor: Any = None,
) -> TensorMeta | None:
    if fallback_device is None and isinstance(fallback_tensor, torch.Tensor):
        fallback_device = fallback_tensor.device
    for key in ("example_value", "val", "value"):
        meta = _meta_from_tensor(node.meta.get(key), fallback_device)
        if meta is not None:
            return meta

    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is not None:
        try:
            if any(type(dim) is not int for dim in tensor_meta.shape):
                return None
            if any(type(item) is not int for item in tensor_meta.stride):
                return None
            return TensorMeta(
                shape=tuple(int(dim) for dim in tensor_meta.shape),
                dtype=tensor_meta.dtype,
                device=getattr(tensor_meta, "device", None) or fallback_device,
                stride=tuple(int(item) for item in tensor_meta.stride),
                requires_grad=bool(getattr(tensor_meta, "requires_grad", False)),
            )
        except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
            pass
    return _meta_from_tensor(fallback_tensor, fallback_device)


def map_fx_arg(value: Any, value_map: dict[fx.Node, Any]) -> Any:
    """Map FX references recursively while preserving Python container types."""
    if isinstance(value, fx.Node):
        try:
            return value_map[value]
        except KeyError as exc:
            raise ValueError(
                f"FX value {value.name!r} is used before it is lowered"
            ) from exc
    if isinstance(value, tuple):
        return tuple(map_fx_arg(item, value_map) for item in value)
    if isinstance(value, list):
        return [map_fx_arg(item, value_map) for item in value]
    if isinstance(value, dict):
        return {key: map_fx_arg(item, value_map) for key, item in value.items()}
    return value


def _source_module_path(node: fx.Node) -> str | None:
    """Retain FX module attribution without keeping modules or tensor data."""
    path = node.target if node.op == "call_module" else getattr(node.target, "path", None)
    if not isinstance(path, str):
        stack = node.meta.get("nn_module_stack", {})
        stack = stack if isinstance(stack, Mapping) else {}
        paths = [entry[0] for entry in stack.values()
                 if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str)]
        path = paths[-1] if paths else None
    for root in ("L['self']", "L['fn'].__self__", "function.__self__"):
        if path == root:
            return "<root>"
        prefix = root + "."
        if path is not None and path.startswith(prefix):
            path = path[len(prefix):]
    return path


def call_method_helper(method_name: str, obj: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(obj, method_name)(*args, **kwargs)


class LoweringContext:
    """Safe helpers exposed to custom FX-to-IR lowering callbacks."""

    def __init__(self, lowering: GraphLowering):
        self._lowering = lowering

    @property
    def graph(self) -> IRGraph:
        return self._lowering.graph

    @property
    def native_autodiff(self) -> bool:
        return self._lowering.native_autodiff

    def write_rewrite(self, node):
        return self._lowering.write_rewrites.get(node.name)

    @property
    def graph_module(self) -> fx.GraphModule:
        return self._lowering.gm

    def map_arg(self, value: Any) -> Any:
        return map_fx_arg(value, self._lowering.value_map)

    def bind_result(self, fx_node, result):
        """Bind a structured logical result after adding its tensor producers."""
        self._lowering.value_map[fx_node] = result

    def tensor_value(self, fx_node: fx.Node, *, name: str | None = None) -> TensorValue:
        meta = metadata_from_fx(
            fx_node,
            fallback_device=self._lowering._fallback_device(fx_node),
        )
        return TensorValue(name=name or fx_node.name, meta=meta)

    def tensor_values(self, fx_node: fx.Node):
        """Infer a flat tuple of explicit tensor results for custom lowerings."""
        values = next(
            (fx_node.meta[key] for key in ("example_value", "val", "value") if key in fx_node.meta),
            fx_node.meta.get("tensor_meta"),
        )
        if not isinstance(values, (tuple, list)) or hasattr(values, "shape"):
            raise TypeError("A multi-result lowering requires a tuple of tensor metadata")
        device = self._lowering._fallback_device(fx_node)
        results = []
        for index, value in enumerate(values):
            if not isinstance(value, torch.Tensor) and isinstance(getattr(value, "stride", None), tuple):
                meta = TensorMeta(
                    shape=tuple(value.shape), dtype=value.dtype, device=device,
                    stride=tuple(value.stride), requires_grad=value.requires_grad,
                )
            else:
                meta = _meta_from_tensor(value, device)
            if meta is None:
                raise TypeError(f"Custom output {index} has no supported tensor metadata")
            results.append(TensorValue(name=f"{fx_node.name}_{index}", meta=meta))
        return tuple(results)

    def add_node(self, fx_node: fx.Node, ir_node: IRNode):
        outputs = ir_node.outputs()
        if not outputs or not all(isinstance(value, TensorValue) for value in outputs):
            raise TypeError(
                f"Custom node {type(ir_node).__name__} must produce TensorValues"
            )
        self.graph.add(ir_node)
        output = outputs[0] if len(outputs) == 1 else outputs
        self._lowering.value_map[fx_node] = output
        return output

    def parameter(self, fx_node, name, parameter):
        target = f"{fx_node.target}.{name}"
        if target not in self._lowering.synthetic_attrs:
            value = TensorValue(name=f"{fx_node.name}_{name}", meta=_meta_from_tensor(parameter),
                                provenance="parameter" if isinstance(parameter, torch.nn.Parameter) else "persistent_buffer")
            self.graph.add(GetAttrNode(target=target, value=value))
            self._lowering.synthetic_attrs[target] = value
        return self._lowering.synthetic_attrs[target]

    def external(self, fx_node, *, reason):
        self._lowering.fallback_reasons[fx_node.name] = reason
        self._lowering.lower_external(fx_node)
        node = self.graph.nodes[-1]
        self._lowering.external_fallbacks.add(id(node))
        return node

    def resolve_target(self, fx_node: fx.Node) -> Any:
        if fx_node.op == "call_function":
            return fx_node.target
        if fx_node.op == "call_method":
            return functools.partial(call_method_helper, fx_node.target)
        if fx_node.op == "call_module":
            return self.graph_module.get_submodule(fx_node.target)
        raise ValueError(f"FX node {fx_node.name!r} has no callable target")


class GraphLowering:
    """Lower an FX GraphModule to the backend's single-output tensor IR."""

    def __init__(
        self,
        gm: fx.GraphModule,
        example_inputs,
        *,
        registry: BackendRegistry | None = None,
        functionalize_nodes: frozenset[str] = frozenset(),
        write_rewrites: tuple = (),
        native_autodiff: bool = True,
    ):
        self.gm = gm
        self.example_inputs = tuple(example_inputs)
        self.graph = IRGraph()
        self.value_map: dict[fx.Node, Any] = {}
        self.synthetic_attrs: dict[str, TensorValue] = {}
        self.placeholder_index = 0
        self.registry = registry if registry is not None else BackendRegistry()
        self.functionalize_nodes = functionalize_nodes
        self.write_rewrites = {rewrite.node: rewrite for rewrite in write_rewrites}
        self.native_autodiff = native_autodiff
        self.context = LoweringContext(self)
        self.fallback_reasons = {}
        self.external_fallbacks = set()

    def lower_placeholder(self, node: fx.Node) -> None:
        index = self.placeholder_index
        sample = (
            self.example_inputs[index] if index < len(self.example_inputs) else None
        )
        value = TensorValue(
            name=node.name,
            meta=metadata_from_fx(node, fallback_tensor=sample),
            provenance=("parameter" if isinstance(sample, torch.nn.Parameter) else
                        "persistent_buffer" if any(sample is buffer for buffer in self.gm.buffers()) else "user_input"),
        )
        ir_node = InputNode(index=index, value=value)
        self.graph.add(ir_node)
        self.value_map[node] = value
        self.placeholder_index += 1

    def lower_get_attr(self, node: fx.Node) -> None:
        value = self.gm
        for part in str(node.target).split("."):
            value = getattr(value, part)
        tensor_value = TensorValue(
            name=node.name,
            meta=metadata_from_fx(node, fallback_tensor=value),
            provenance="parameter" if isinstance(value, torch.nn.Parameter) else "persistent_buffer",
        )
        ir_node = GetAttrNode(target=str(node.target), value=tensor_value)
        self.graph.add(ir_node)
        self.value_map[node] = tensor_value

    def _fallback_device(self, node: fx.Node) -> torch.device | str | None:
        devices: list[torch.device] = []

        def visit(value: Any) -> None:
            if isinstance(value, fx.Node):
                lowered = self.value_map.get(value)
                from .ir import find_tensor_values

                for result in find_tensor_values(lowered):
                    meta = result.meta
                    if meta is not None and meta.device is not None:
                        devices.append(torch.device(meta.device))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)

        visit(node.args)
        visit(node.kwargs)
        unique = {device for device in devices}
        if len(unique) > 1:
            return None
        return next(iter(unique), None)

    def _make_ir_node(
        self,
        node: fx.Node,
        node_type: type,
        op: str,
    ) -> IRNode:
        args = map_fx_arg(node.args, self.value_map)
        kwargs = map_fx_arg(node.kwargs, self.value_map)
        value = TensorValue(
            name=node.name,
            meta=metadata_from_fx(node, fallback_device=self._fallback_device(node)),
        )
        return node_type(
            op=op,
            args=args,
            kwargs=kwargs,
            value=value,
            target=self.context.resolve_target(node),
        )

    def lower_pointwise(self, node: fx.Node) -> None:
        functionalized = node.name in self.functionalize_nodes
        if functionalized:
            kind = self.write_rewrites[node.name].operation
        else:
            kind = _target_kind(node, self.gm)
        if kind is None:
            raise ValueError(f"No canonical pointwise kind for FX node {node.name!r}")
        lowered = self._make_ir_node(node, PointwiseNode, kind)
        kwargs = dict(lowered.kwargs)
        args = list(lowered.args)
        if not args and "input" in kwargs:
            args.append(kwargs.pop("input"))
        elif not args and "self" in kwargs:
            args.append(kwargs.pop("self"))
        if kind in ("add", "sub", "mul", "div") and len(args) == 1 and "other" in kwargs:
            args.append(kwargs.pop("other"))
        lowered.args, lowered.kwargs = tuple(args), kwargs
        if node.op == "call_module" and not functionalized:
            module = self.gm.get_submodule(node.target)
            fields = {
                "gelu": ("approximate",),
                "leaky_relu": ("negative_slope",),
                "hardtanh": ("min_val", "max_val"),
                "softplus": ("beta", "threshold"),
            }.get(kind, ())
            lowered.kwargs = {
                **lowered.kwargs,
                **{name: getattr(module, name) for name in fields},
            }
            lowered.target = getattr(torch.nn.functional, kind)
        if functionalized:
            from deepopt.opt.nn.primitive_state import destination_add

            lowered.target = destination_add if kind == "add" else getattr(torch.nn.functional, kind)
            if kind != "add":
                lowered.kwargs = {}
                lowered.args = lowered.args[:1]
        self.graph.add(lowered)
        self.value_map[node] = self.graph.nodes[-1].output()

    def lower_external(self, node: fx.Node) -> None:
        """Preserve an unsupported callable as a native PyTorch graph operation.

        Common differentiable layers also keep their native forward callable.
        Their parameters are added as explicit IR dependencies so WrapperCodegen
        can put an AOT custom-autograd boundary around the call and use the
        registered ATen backward rule in the resulting AOT graph. Buffers are
        explicit dependencies when a layer's backward requires running stats.
        """
        args = map_fx_arg(node.args, self.value_map)
        kwargs = map_fx_arg(node.kwargs, self.value_map)
        target = self.context.resolve_target(node)
        rewrite = self.write_rewrites.get(node.name)
        if rewrite is not None and rewrite.operation == "dropout":
            from copy import copy

            if type(target) is torch.nn.Dropout:
                target = copy(target)
                target.inplace = False
            else:
                args, kwargs = list(args), dict(kwargs)
                if len(args) > 3:
                    args[3] = False
                else:
                    kwargs["inplace"] = False
                args = tuple(args)
        from .autodiff.primitives import multi_kind

        result_kind = multi_kind(target)
        if result_kind is not None:
            from .ir import CallNode

            outputs = self.context.tensor_values(node)
            lowered = CallNode(f"autodiff_multi_{result_kind}", target, args, kwargs, outputs)
            self.context.add_node(node, lowered)
            return
        if (
            target is torch.nn.functional.dropout
            and node.kwargs.get("inplace", node.args[3] if len(node.args) > 3 else False) is False
            and (
                node.kwargs.get("p", node.args[1] if len(node.args) > 1 else 0.5) == 0
                or node.kwargs.get("training", node.args[2] if len(node.args) > 2 else True) is False
            )
        ):
            self.value_map[node] = args[0] if args else kwargs["input"]
            return
        from .operations import is_deepopt_target

        deepopt = is_deepopt_target(target)
        if deepopt:
            self.fallback_reasons.setdefault(
                node.name, "no generated lowering for this DeepOpt operation or recipe",
            )
        autograd_inputs: list[TensorValue] = []
        autograd_input_names: list[str] = []
        autograd_buffers: list[TensorValue] = []
        autograd_buffer_names: list[str] = []
        if node.op == "call_module" and isinstance(target, torch.nn.Module):
            for parameter_name, parameter in target.named_parameters(recurse=True):
                attribute_name = f"{node.target}.{parameter_name}"
                value = self.synthetic_attrs.get(attribute_name)
                if value is None:
                    value = TensorValue(
                        name=f"{node.name}_{parameter_name.replace('.', '_')}",
                        meta=_meta_from_tensor(parameter),
                        provenance="parameter",
                    )
                    self.graph.add(GetAttrNode(target=attribute_name, value=value))
                    self.synthetic_attrs[attribute_name] = value
                autograd_inputs.append(value)
                autograd_input_names.append(parameter_name)
            for buffer_name, buffer in target.named_buffers(recurse=True):
                attribute_name = f"{node.target}.{buffer_name}"
                value = self.synthetic_attrs.get(attribute_name)
                if value is None:
                    value = TensorValue(
                        name=f"{node.name}_{buffer_name.replace('.', '_')}",
                        meta=_meta_from_tensor(buffer),
                        provenance="persistent_buffer",
                    )
                    self.graph.add(GetAttrNode(target=attribute_name, value=value))
                    self.synthetic_attrs[attribute_name] = value
                autograd_buffers.append(value)
                autograd_buffer_names.append(buffer_name)
        value = TensorValue(
            name=node.name,
            meta=metadata_from_fx(node, fallback_device=self._fallback_device(node)),
        )
        if deepopt and value.meta is not None:
            from dataclasses import replace

            value.meta = replace(value.meta, representation="opaque")
        ir_node = ExternKernelNode(
            op=str(node.target),
            target=target,
            args=args,
            kwargs=kwargs,
            value=value,
            autograd_inputs=tuple(autograd_inputs),
            autograd_input_names=tuple(autograd_input_names),
            autograd_buffers=tuple(autograd_buffers),
            autograd_buffer_names=tuple(autograd_buffer_names),
            autograd_owner="deepopt" if deepopt else "pytorch",
            stateful=deepopt,
        )
        if "deepopt_exact_mask" in node.meta:
            from .effects import stochastic_depth_snapshot

            if target is stochastic_depth_snapshot:
                ir_node.op = "stochastic_depth.snapshot"
                ir_node.residual_recipe = dict(node.meta["deepopt_exact_mask"])
                value.provenance = "stochastic_depth_mask"
        self.graph.add(ir_node)
        self.value_map[node] = value

    def lower_extension(self, node: fx.Node, extension: OperatorExtension) -> None:
        before = len(self.graph.nodes)
        lowered = extension.lowering(self.context, node)
        if not isinstance(lowered, extension.node_type) and id(lowered) not in self.external_fallbacks:
            raise TypeError(
                f"Lowering for {extension.name!r} returned {type(lowered).__name__}; "
                f"expected {extension.node_type.__name__}"
            )
        if (
            len(self.graph.nodes) <= before or self.graph.nodes[-1] is not lowered
            or not extension.decomposes and any(not isinstance(item, GetAttrNode) for item in self.graph.nodes[before:-1])
        ):
            raise RuntimeError(
                f"Lowering for {extension.name!r} must add exactly its returned node "
                "through context.add_node()"
            )

    def lower_output(self, node: fx.Node) -> None:
        result = map_fx_arg(node.args[0], self.value_map)
        self.graph.add(OutputNode(result=result))

    def run(self) -> IRGraph:
        self.gm.graph.lint()
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self.lower_placeholder(node)
            elif node.op == "get_attr":
                self.lower_get_attr(node)
            elif node.op in ("call_function", "call_method", "call_module"):
                first_new_node = len(self.graph.nodes)
                # Runtime/device dispatch can leave unused metadata predicates
                # such as getattr(tensor, "is_xpu") in a captured FX graph.
                # They have no tensor derivative or side effect. Do not lower
                # these dead reads as unknown differentiable tensor outputs.
                if (node.op == "call_function" and node.target is getattr and not node.users
                        and len(node.args) == 2 and node.args[1] in {
                            "dtype", "device", "shape", "ndim", "is_cpu", "is_cuda", "is_xpu",
                            "is_mps", "is_meta", "layout", "requires_grad"}):
                    continue
                if node.op == "call_function" and node.target is operator.getitem:
                    values = map_fx_arg(node.args[0], self.value_map)
                    index = node.args[1]
                    if isinstance(values, (tuple, list, dict)) and isinstance(index, (int, str)):
                        self.value_map[node] = values[index]
                        continue
                extension = self.registry.match_fx(node, self.gm)
                if extension is not None:
                    self.lower_extension(node, extension)
                elif node.name in self.write_rewrites and self.write_rewrites[node.name].operation == "dropout":
                    self.lower_external(node)
                elif node.name in self.functionalize_nodes or is_pointwise_fx_node(
                    node, self.gm
                ):
                    self.lower_pointwise(node)
                else:
                    self.lower_external(node)
                module_path = _source_module_path(node)
                for created in self.graph.nodes[first_new_node:]:
                    if getattr(created, "source_node", None) is None:
                        created.source_node = node.name
                    if getattr(created, "module_path", None) is None and module_path is not None:
                        created.module_path = module_path
                # Dynamo can refer to the original destination after a write.
                # Rebind those later reads to the proved fresh SSA result.
                rewrite = self.write_rewrites.get(node.name)
                if rewrite is not None:
                    destination = next(value for value in self.gm.graph.nodes if value.name == rewrite.destination)
                    self.value_map[destination] = self.value_map[node]
            elif node.op == "output":
                self.lower_output(node)
            else:
                raise NotImplementedError(
                    f"Unsupported FX opcode {node.op!r} at node {node.name!r}"
                )
        self.graph.refresh_model_ownership()
        self.graph.validate()
        return self.graph
