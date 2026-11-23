"""User operations with one eager/native backward and saved-state contract.

Composite registrations use ``decomposition=True`` and inherit the existing
primitive derivative/storage policies. Opaque, pure, single-result operations
provide ``fake``, ``saved_state(ctx, inputs, output)`` and ``vjp(state, grad)``.
The VJP returns a mapping from argument names to cotangents. It sees only
explicitly declared state, and is traced into ordinary native backward IR.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
from inspect import Parameter, signature
from types import MappingProxyType

import torch
from torch.autograd.function import once_differentiable

from .compiler.autodiff.activation_saving import SavedUse, capture_saved_activation_policy
from .compiler.ir import CallNode, TensorMeta, TensorValue
from .kernel_policy import kernel_forward, kernel_backward

_SAVED_TENSOR = object()


def _metadata(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format, TensorMeta)):
        return value
    if isinstance(value, tuple):
        return tuple(_metadata(item) for item in value)
    raise TypeError("Operation metadata must be immutable and contain no tensor payloads")


def _no_tensor_closure(callback):
    from torch.utils._pytree import tree_leaves

    cells = getattr(callback, "__closure__", None) or ()
    payload = (tuple(cell.cell_contents for cell in cells), getattr(callback, "__defaults__", ()),
               getattr(callback, "__kwdefaults__", {}))
    if any(isinstance(item, (torch.Tensor, torch.nn.Module)) for item in tree_leaves(payload)):
        raise ValueError("Operation callbacks must receive tensors as explicit operands, not closures")
    if isinstance(getattr(callback, "__self__", None), (torch.Tensor, torch.nn.Module)):
        raise ValueError("Operation callbacks cannot hide tensor/module owners in bound methods")
    code = getattr(callback, "__code__", None)
    globals_ = getattr(callback, "__globals__", {})
    if code is not None and any(isinstance(globals_.get(name), (torch.Tensor, torch.nn.Module)) for name in code.co_names):
        raise ValueError("Operation callbacks must receive global tensor/module state as explicit operands")


class OperationState(dict):
    """Declared residuals supplied to the VJP; tensor reads must use these names.

    ``call`` emits an explicit native backward call when given symbolic
    operands. ``meta`` is mandatory there and describes its single result.
    The callback must be pure and operate only on its explicit operands.
    """

    @staticmethod
    def call(target, *args, meta, **kwargs):
        from .compiler.trace.tracer import SymbolicTensor
        from torch.utils._pytree import tree_leaves

        symbolic = next((value for value in tree_leaves((args, kwargs)) if isinstance(value, SymbolicTensor)), None)
        if symbolic is None:
            return target(*args, **kwargs)
        _no_tensor_closure(target)
        return symbolic.builder.emit_native("registered.vjp_call", target, args, kwargs, meta)


class _SaveContext:
    def __init__(self, operation, inputs, *, native=None, policy=None):
        self.operation = operation
        self.native = native
        self.policy = native.saved_activation_policy if native is not None else policy
        self.state = OperationState()
        self.tensors, self.bindings = [], []
        self.owned = tuple(inputs[name] for name in operation.state_operands)
        self.metadata_items = {}

    @staticmethod
    def meta(value):
        if isinstance(value, TensorValue):
            return value.meta
        from .compiler.lowering import _meta_from_tensor
        return _meta_from_tensor(value)

    def needs_grad(self, value):
        meta = self.meta(value)
        return meta is not None and meta.requires_grad

    def _name(self, name):
        if not isinstance(name, str) or not name or name in self.state:
            raise ValueError("Saved residuals need unique nonempty names")

    def metadata(self, name, value):
        self._name(name)
        self.state[name] = _metadata(value)
        self.metadata_items[name] = value

    def save(self, name, value, *, use=None):
        self._name(name)
        if not isinstance(value, (torch.Tensor, TensorValue)):
            raise TypeError("save requires a tensor; use metadata for immutable geometry")
        use = use or SavedUse()
        use = replace(use, consumer=use.consumer or self.operation.name, role=f"{name}:{use.role}")
        model_owned = any(value is owned for owned in self.owned) or isinstance(value, torch.nn.Parameter)
        if model_owned:
            use = replace(use, allow_approximation=False, codec=None)
        if self.native is not None:
            if value.provenance == "persistent_buffer" and use.allow_approximation and not model_owned:
                raise ValueError("Declare persistent-buffer operands in state_operands before requesting approximation")
            self.state[name] = self.native.saved(value, use=use)
            return
        symbolic = TensorValue(name, self.meta(value), provenance="parameter" if isinstance(value, torch.nn.Parameter) else None)
        recipe, _ = self.policy.select(symbolic, use)
        start = len(self.tensors)
        if recipe is None:
            self.tensors.append(value)
            binding = (name, start, None, None)
        else:
            from .opt.quantization.activation import _quantize_pack_strided_activation

            shape = tuple(value.shape)
            matrix_shape = (value.numel() // shape[-1], shape[-1]) if shape else (1, 1)
            options = dict(recipe)
            packed = _quantize_pack_strided_activation(value if value.ndim >= 2 else value.reshape(matrix_shape),
                                                       options["group_size"], bits=options["bits"])
            self.tensors.extend((packed.data, packed.step, packed.minimum))
            binding = (name, start, (shape, matrix_shape, tuple(value.stride()), value.dtype), options)
        self.bindings.append(binding)
        # Placeholders prevent duplicate names without retaining originals in ctx.
        self.state[name] = _SAVED_TENSOR

    def compute(self, target, *args, meta, **kwargs):
        """Produce declared auxiliary state from explicit forward operands.

        ``meta`` must describe a nondifferentiable single tensor. This is useful
        for branch masks/statistics; stochastic replay and mutation are outside
        this pure-operation API.
        """
        if not isinstance(meta, TensorMeta) or meta.requires_grad:
            raise ValueError("Auxiliary state requires nondifferentiable TensorMeta")
        _no_tensor_closure(target)
        if self.native is None:
            return target(*args, **kwargs)
        from .compiler.trace.builder import _tree_map
        from .compiler.trace.tracer import SymbolicTensor

        def unwrap(value):
            return value.value if isinstance(value, SymbolicTensor) else value
        graph = self.native.forward_graph
        output = TensorValue(f"{self.native._current_node.output().name}__aux_{len(graph.nodes)}", meta)
        prepared = CallNode("registered.prepare", target, _tree_map(unwrap, args), _tree_map(unwrap, kwargs),
                            (output,), MappingProxyType({"operation": self.operation.name, "schema": self.operation.schema}))
        # Source and output auxiliary values are all available after this op;
        # append consecutive preparations after earlier dependencies.
        dependencies = [graph.nodes.index(value.producer) for value in prepared.inputs()]
        index = max([graph.nodes.index(self.native._current_node), *dependencies]) + 1
        graph.add(prepared)
        graph.nodes.pop()
        graph.nodes.insert(index, prepared)
        return output

    def mask(self, name, value):
        """Save a boolean branch decision losslessly using the one-bit codec."""
        from math import prod
        from .opt.quantization.boolean import PackedBooleanMask, pack_boolean_mask, unpack_boolean_mask

        self._name(name)
        metadata = self.meta(value)
        if metadata is None or metadata.dtype != torch.bool:
            raise TypeError("mask requires a boolean tensor")
        if self.native is None:
            packed = pack_boolean_mask(value)
            start = len(self.tensors)
            self.tensors.append(packed.data)
            self.bindings.append((name, start, metadata.shape, "boolean"))
            self.state[name] = _SAVED_TENSOR
            return
        packed = self.compute(lambda mask: pack_boolean_mask(mask).data, value,
            meta=TensorMeta(((prod(metadata.shape) + 31) // 32,), torch.int32, metadata.device,
                            representation="packed"))
        stored = self.native.saved(packed)
        self.state[name] = self.native.builder.emit_native("registered.restore_mask",
            lambda payload: unpack_boolean_mask(PackedBooleanMask(payload, metadata.shape)), (stored,), {}, metadata)
        self.native.record_residual(value, (packed,), representation="lossless_boolean_mask",
            use=SavedUse(role=name, consumer=self.operation.name), recipe=(("schema", self.operation.schema),),
            retention_reason="exact_branch_state")


class _OperationFunction(torch.autograd.Function):
    @staticmethod
    @kernel_forward
    def forward(ctx, operation, *values):
        inputs = dict(zip(operation.names, values))
        output = operation.invoke(operation.execute, inputs)
        devices = tuple(value.device.type for value in values if isinstance(value, torch.Tensor))
        device = next((device for device in devices if device != "cpu"), devices[0] if devices else "cpu")
        saves = _SaveContext(operation, inputs, policy=capture_saved_activation_policy(device))
        if ((output.dtype.is_floating_point or output.dtype.is_complex)
                and any(isinstance(value, torch.Tensor) and value.requires_grad for value in values)):
            operation.saved_state(saves, MappingProxyType(inputs), output)
        ctx.operation = operation
        ctx.state = {name: value for name, value in saves.state.items() if value is not _SAVED_TENSOR}
        ctx.bindings = tuple(saves.bindings)
        ctx.save_for_backward(*saves.tensors)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    @kernel_backward
    def backward(ctx, grad):
        from .opt._autocast import _suspend_fp8_interception

        device = grad.device.type if grad is not None else "cpu"
        with _suspend_fp8_interception(), torch.autocast(device, enabled=False):
            return _OperationFunction._backward(ctx, grad)

    @staticmethod
    def _backward(ctx, grad):
        if grad is None:
            return (None,) * (1 + len(ctx.operation.names))
        state = OperationState(ctx.state)
        tensors = ctx.saved_tensors
        for name, start, metadata, options in ctx.bindings:
            if options == "boolean":
                from .opt.quantization.boolean import PackedBooleanMask, unpack_boolean_mask

                state[name] = unpack_boolean_mask(PackedBooleanMask(tensors[start], metadata))
            elif options is None:
                state[name] = tensors[start]
            else:
                from .opt.quantization.activation import PackedActivation, dequantize_unpack_activation

                shape, matrix_shape, stride, dtype = metadata
                result = dequantize_unpack_activation(PackedActivation(*tensors[start:start + 3], matrix_shape,
                    options["group_size"], bits=options["bits"]), dtype=dtype, row_alignment=1).reshape(shape)
                # Match native restore for dense permutations; holey views stay compact.
                dimensions = sorted((step, size) for step, size in zip(stride, shape) if size > 1)
                expected, dense = 1, True
                for step, size in dimensions:
                    dense = dense and step == expected
                    expected *= size
                if dense and result.stride() != stride:
                    restored = torch.empty_strided(shape, stride, device=result.device, dtype=dtype)
                    restored.copy_(result)
                    result = restored
                state[name] = result
        contributions = ctx.operation.validate_gradients(ctx.operation.vjp(state, grad))
        return (None, *(contributions.get(name) if needed else None
                        for name, needed in zip(ctx.operation.names, ctx.needs_input_grad[1:])))


@dataclass(frozen=True)
class _RegisteredOperation:
    name: str
    forward: object
    fake: object
    saved_state: object
    vjp: object
    schema: str
    names: tuple
    state_operands: tuple
    _deepopt_registered_operation = True

    @property
    def __name__(self):
        return self.name

    def bind(self, args, kwargs):
        bound = signature(self.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments

    def invoke(self, function, inputs):
        parameters = signature(self.forward).parameters
        args = tuple(inputs[name] for name in self.names if parameters[name].kind != Parameter.KEYWORD_ONLY)
        kwargs = {name: inputs[name] for name in self.names if parameters[name].kind == Parameter.KEYWORD_ONLY}
        return function(*args, **kwargs)

    def validate_gradients(self, result):
        if not isinstance(result, dict) or set(result) - set(self.names):
            raise TypeError("The operation VJP must return a dict keyed by argument names")
        return result

    def execute(self, *args, **kwargs):
        tensors = tuple(value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor))
        versions = tuple(None if value.is_inference() else value._version for value in tensors)
        output = self.forward(*args, **kwargs)
        if tuple(None if value.is_inference() else value._version for value in tensors) != versions:
            raise ValueError("Registered operation mutated an operand despite declaring no effects")
        if not isinstance(output, torch.Tensor):
            raise TypeError("Explicit registered operations must return one tensor")
        if any(torch._C._is_alias_of(value, output) for value in tensors):
            raise ValueError("Registered operations must return a fresh tensor; aliases are unsupported")
        return output

    def __call__(self, *args, **kwargs):
        from torch._subclasses.fake_tensor import FakeTensor

        inputs = self.bind(args, kwargs)
        for value in inputs.values():
            if not isinstance(value, torch.Tensor):
                _metadata(value)
        if any(isinstance(value, FakeTensor) for value in inputs.values()):
            output = self.invoke(self.fake, inputs)
            if not isinstance(output, torch.Tensor):
                raise TypeError("Operation fake callback must return one tensor")
            output.requires_grad_(torch.is_grad_enabled() and (output.dtype.is_floating_point or output.dtype.is_complex)
                and any(isinstance(value, torch.Tensor) and value.requires_grad for value in inputs.values()))
            return output
        if not torch.is_grad_enabled():
            return self.invoke(self.execute, inputs)
        return _OperationFunction.apply(self, *inputs.values())


def register_operation(*, name, decomposition=None, fake=None, saved_state=None, vjp=None, schema="1", effects=(),
                       state_operands=()):
    """Register a pure single-result operation for eager and strict native use.

    ``decomposition=True`` uses the decorated function itself as the primitive
    recipe; a callable selects a separate decomposition for both paths.
    Otherwise all three callbacks are mandatory. Variadic operands, mutation,
    aliases and RNG are rejected; pass model state as explicit tensor operands.
    ``schema`` versions the immutable residual contract for reporting.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Operation names must be Python identifiers")
    if not isinstance(schema, str) or not schema or effects:
        raise ValueError("Operations require a schema version and must declare no effects")
    if not isinstance(state_operands, tuple) or any(not isinstance(name, str) for name in state_operands):
        raise TypeError("state_operands must be a tuple of named model-state operands")
    if decomposition is not None and decomposition is not True and not callable(decomposition):
        raise TypeError("decomposition must be True or a callable")
    if decomposition is not None and any(value is not None for value in (fake, saved_state, vjp)):
        raise ValueError("Choose a decomposition or explicit fake/saved_state/VJP callbacks")
    if decomposition is None and not all(callable(value) for value in (fake, saved_state, vjp)):
        raise ValueError("Explicit operations require fake, saved_state and vjp callbacks")

    def decorate(function):
        callbacks = (function, decomposition, fake, saved_state, vjp)
        for callback in callbacks:
            if callable(callback):
                _no_tensor_closure(callback)
        if decomposition is not None:
            recipe = function if decomposition is True else decomposition
            @wraps(function)
            def composite(*args, **kwargs):
                return recipe(*args, **kwargs)
            composite._deepopt_operation_schema = (name, schema, "decomposition")
            composite._deepopt_operation_state_operands = state_operands
            return composite
        parameters = signature(function).parameters
        if set(state_operands) - set(parameters):
            raise ValueError("state_operands must name operation arguments")
        if any(parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD) for parameter in parameters.values()):
            raise TypeError("Explicit registered operations require named nonvariadic operands")
        operation = _RegisteredOperation(name, function, fake, saved_state, vjp, schema, tuple(parameters), state_operands)
        return torch.compiler.allow_in_graph(operation)

    return decorate


@dataclass(eq=False)
class RegisteredOperationNode(CallNode):
    """Native forward with a user-declared, inspectable residual/VJP contract."""


def extension():
    from .compiler.lowering import metadata_from_fx
    from .compiler.registry import OperatorExtension
    from .compiler.autodiff.backward_rules import GradWrapper

    def lower(context, node):
        operation = node.target
        args, kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
        for value in operation.bind(args, kwargs).values():
            if not isinstance(value, TensorValue):
                _metadata(value)
        value = TensorValue(node.name, metadata_from_fx(node))
        result = RegisteredOperationNode("registered_operation", operation.execute, args, kwargs, (value,), operation)
        context.add_node(node, result)
        return result

    def backward(ctx, node, grad):
        operation = node.specification
        inputs = operation.bind(node.args, node.kwargs)
        saves = _SaveContext(operation, inputs, native=ctx)
        operation.saved_state(saves, MappingProxyType(inputs), node.output())
        if saves.state:
            source = next((value for value in inputs.values() if isinstance(value, TensorValue)), node.output())
            ctx.record_residual(source, (), representation="metadata_only",
                use=SavedUse(role="operation_contract", consumer=operation.name),
                recipe=(("operation", operation.name), ("state_schema", operation.schema),
                        ("metadata", tuple((name, repr(value)) for name, value in saves.metadata_items.items()))),
                retention_reason="registered_state_schema")
        contributions = operation.validate_gradients(operation.vjp(saves.state, grad))
        return tuple(GradWrapper(inputs[name], value) for name, value in contributions.items())

    return OperatorExtension("registered_operation", RegisteredOperationNode,
        lambda node, gm: node.op == "call_function" and isinstance(node.target, _RegisteredOperation), lower), backward


def operation_recipes(model):
    """Discover this model's referenced operation contracts before first use.

    Inspect numerical methods, their named helpers, callable attributes and
    ordinary containers without executing user code or consulting a global
    registry. Dynamic dispatch that static references cannot describe must
    declare its possible decorated callables as ``model.__deepopt_operations__
    = (my_operation, ...)``. The caller owns schema versioning when callbacks'
    numerical behavior changes; source hashing is a separate provenance check.
    """
    from inspect import getattr_static, isfunction, ismethod
    from types import ModuleType

    seen, records, pending = set(), set(), []

    def visit(value, owner=None, references=()):
        pending.append((value, owner, references))

    def discover(value, owner, references):
        if isinstance(value, torch.Tensor) or value is None:
            return
        # Attribute access creates fresh bound-method objects. Identify the
        # stable function/self pair before cycle detection, including when a
        # compiler facade points back to its owning module's forward method.
        if ismethod(value):
            owner = value.__self__
            value = value.__func__
        elif isinstance(value, torch.nn.Module):
            # A module has its own numerical methods; the facade through which
            # it was reached does not create a different operation contract.
            owner = None
        identity = (id(value), id(owner), tuple(references) if isinstance(value, ModuleType) else ())
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, _RegisteredOperation):
            records.add((value.name, value.schema, "explicit_vjp", value.state_operands))
            return
        schema = getattr(value, "_deepopt_operation_schema", None) if isfunction(value) else None
        if schema is not None:
            records.add((*schema, getattr(value, "_deepopt_operation_state_operands", ())))
        if isinstance(value, (tuple, list)):
            for child in value:
                visit(child, owner)
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child, owner)
            return
        if isinstance(value, ModuleType):
            # Only resolve bytecode-named members, without invoking lazy module
            # __getattr__ hooks or walking all imported operations in a package.
            namespace = vars(value)
            for name in references:
                if name in namespace:
                    visit(namespace[name], owner, references)
            return
        if isinstance(value, torch.nn.Module):
            for module in value.modules():
                visit(module.forward, module)
                for name, child in vars(module).items():
                    if name == "__deepopt_operations__" or callable(child):
                        visit(child, module)
                declared = getattr_static(module, "__deepopt_operations__", ())
                if not isinstance(declared, (tuple, list)):
                    raise TypeError("__deepopt_operations__ must list the model's decorated operation callables")
                if any(not isinstance(child, _RegisteredOperation) and not (
                        isfunction(child) and getattr(child, "_deepopt_operation_schema", None)) for child in declared):
                    raise TypeError("__deepopt_operations__ entries must be decorated operation callables")
                visit(declared, module)
            return
        if isfunction(value):
            for wrapped in (getattr(value, "__wrapped__", None), *(
                    cell.cell_contents for cell in (value.__closure__ or ())),
                    value.__defaults__, value.__kwdefaults__):
                visit(wrapped, owner)
            if (getattr(value, "__module__", None) or "").startswith("torch."):
                return
            names = value.__code__.co_names
            namespace = value.__globals__
            for name in names:
                if name in namespace:
                    visit(namespace[name], owner, names)
                if owner is not None:
                    child = getattr_static(owner, name, None)
                    if isinstance(child, (staticmethod, classmethod)):
                        child = child.__func__
                    if not isinstance(child, property):
                        visit(child, owner)
            return
        # Compiler/precision facades retain the original bound numerical body
        # through these fields. Avoid arbitrary descriptors and object graphs.
        for name in ("__wrapped__", "original", "function", "override", "operation"):
            child = getattr_static(value, name, None)
            if not isinstance(child, property):
                visit(child, owner)
        call = getattr_static(type(value), "__call__", None)
        if isfunction(call) and not (getattr(call, "__module__", None) or "").startswith(("torch.", "builtins")):
            visit(call, value)

    visit(model)
    # Compiler/capture wrappers can form deep chains as well as cycles. Static
    # discovery must not consume the Python call stack or execute these bodies.
    while pending:
        discover(*pending.pop())
    return tuple({"name": name, "schema": schema, "route": route, "state_operands": operands}
                 for name, schema, route, operands in sorted(records))
