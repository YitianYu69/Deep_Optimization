"""Native VJPs for tensor geometry, reductions and matrix primitives."""

from dataclasses import replace
from functools import partial
import operator

import torch

from ..ir import CallNode, ExternKernelNode, TensorValue
from ..lowering import call_method_helper
from .semantic import emit as emit_semantic
from .backward_rules import GradWrapper


def emit(ctx, name, function, args, output_meta):
    """Make each primitive contribution's runtime dtype match its IR contract.

    A promoted forward (cat/stack or dtype= reduction) can deliver a wider
    cotangent. Metadata alone must not hide the cast back to the operand's
    derivative domain: projection would otherwise incorrectly omit it.
    """
    def typed_contribution(*values):
        return function(*values).to(output_meta.dtype)

    return emit_semantic(ctx, name, typed_contribution, args, output_meta)


METHODS = frozenset({"reshape", "view", "flatten", "unflatten", "transpose", "permute", "t",
    "squeeze", "unsqueeze", "contiguous", "clone", "to", "float", "bfloat16", "half", "double",
    "type_as", "sum", "mean", "matmul", "bmm", "mm", "softmax", "expand", "expand_as",
    "repeat", "pow", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "abs", "square"})

PADDING_MODULES = {getattr(torch.nn, f"{prefix}Pad{rank}d"): mode
    for prefix, mode in (("Reflection", "reflect"), ("Replication", "replicate"),
                         ("Circular", "circular"), ("Constant", "constant"), ("Zero", "constant"))
    for rank in (1, 2, 3)}


def kind(node):
    if getattr(node, "op", None) in ("storage_add", "storage_cat"):
        return node.op
    if not isinstance(node, ExternKernelNode) or node.autograd_owner != "pytorch":
        return None
    from .residual_primitives import kind as residual_kind

    residual = residual_kind(node)
    if residual is not None:
        return residual
    target = node.target
    from deepopt.opt.nn.primitive_state import model_exp

    if getattr(target, "original", target) is model_exp:
        return "exp"
    if getattr(target, "original", target) in (torch.exp, torch.Tensor.exp):
        return "exp"
    if target in (torch.nn.functional.pad, torch._C._nn.pad) or type(target) in PADDING_MODULES:
        return "pad"
    if isinstance(target, partial) and target.func is call_method_helper:
        name = target.args[0]
        return name if name in METHODS else None
    for name in METHODS | {"cat", "concat", "stack"}:
        if target is getattr(torch, name, None):
            return "cat" if name == "concat" else name
    if target is operator.getitem:
        return "getitem"
    if target is operator.matmul:
        return "matmul"
    if target is operator.pow:
        return "pow"
    if target in (torch.nn.functional.softmax,):
        return "softmax"
    if target is torch.nn.functional.dropout or type(target) is torch.nn.Dropout:
        return "dropout"
    if isinstance(target, torch._ops.OpOverload):
        name = target._schema.name.removeprefix("aten::")
        if name in {"pad", "constant_pad_nd", "_pad_circular", "reflection_pad1d", "reflection_pad2d",
                    "reflection_pad3d", "replication_pad1d", "replication_pad2d", "replication_pad3d"}:
            return "pad"
        return {"_to_copy": "to", "_unsafe_view": "view", "_softmax": "softmax"}.get(
            name, name if name in METHODS | {"cat", "stack"} else None)
    return None


def argument(node, index, name, default=None):
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


def lower(node):
    name = kind(node)
    if name == "storage_add":
        from deepopt.opt.nn.primitive_state import StorageAddState

        dtypes = tuple(value.meta.dtype if isinstance(value, TensorValue) else None for value in node.args)
        arithmetic = torch.promote_types(*dtypes) if all(dtypes) else next(dtype for dtype in dtypes if dtype is not None)
        if all(isinstance(value, TensorValue) for value in node.args):
            a, b = node.args
            if not a.meta.shape and b.meta.shape:
                arithmetic = b.meta.dtype
            elif a.meta.shape and not b.meta.shape:
                arithmetic = a.meta.dtype
        spec = StorageAddState(tuple(value.meta.shape if isinstance(value, TensorValue) else None for value in node.args), dtypes, arithmetic)
        return CallNode("autodiff_storage_add", spec.forward, node.args, {}, (node.value,), spec)
    if name == "storage_cat":
        return CallNode("autodiff_storage_cat", node.target, node.args, {}, (node.value,))
    if name in ("clamp", "nearest2d"):
        from .residual_primitives import lower as lower_residual

        return lower_residual(node)
    args, kwargs = list(node.args), dict(node.kwargs)
    first_name = "tensors" if name in ("cat", "stack") else "input"
    if not args and first_name in kwargs:
        args.append(kwargs.pop(first_name))
    elif not args and "self" in kwargs:
        args.append(kwargs.pop("self"))
    if name in ("matmul", "bmm", "mm") and len(args) == 1:
        second = "other" if "other" in kwargs else "mat2"
        if second in kwargs:
            args.append(kwargs.pop(second))
    node.args, node.kwargs = tuple(args), kwargs
    if name == "exp" and node.bf16_policy.storage_enabled and node.args[0].meta.dtype == torch.bfloat16:
        from math import prod
        from deepopt.opt.nn.primitive_state import StorageExpState
        from deepopt.opt._autocast import _get_activation_compression
        from .semantic import packed_metas

        metadata = node.args[0].meta
        compression = _get_activation_compression(torch.device(metadata.device).type) or {}
        specification = StorageExpState(metadata.shape, node.bf16_policy,
            compression.get("bits", 2), compression.get("group_size", 1024),
            bool(compression.get("enabled", False)) and prod(metadata.shape) > 0)
        state = (packed_metas((1, prod(metadata.shape)), compression, metadata.device) if specification.packed else
                 (replace(metadata, requires_grad=False, stride=None),))
        values = (node.value,) + tuple(TensorValue(f"{node.value.name}__exp_state_{index}", value)
                                      for index, value in enumerate(state))
        return CallNode("autodiff_storage_exp", specification.forward, node.args[:1], {}, values, specification)
    if name == "pad":
        from deepopt.opt.nn.padding import PaddingGeometry

        target_name = (node.target._schema.name.removeprefix("aten::")
                       if isinstance(node.target, torch._ops.OpOverload) else "pad")
        mode = (PADDING_MODULES[type(node.target)] if type(node.target) in PADDING_MODULES else
                "reflect" if target_name.startswith("reflection") else
                "replicate" if target_name.startswith("replication") else
                "circular" if target_name == "_pad_circular" else
                argument(node, 2, "mode", "constant") if target_name == "pad" else "constant")
        pads = (node.target.padding if type(node.target) in PADDING_MODULES else
                argument(node, 1, "pad", node.kwargs.get("padding")))
        if isinstance(pads, int):
            # Module constructors normalize padding to a tuple on current
            # PyTorch; retain the explicit rank contract for older versions.
            pads = (pads,) * (2 * int(type(node.target).__name__[-2]))
        metadata = node.args[0].meta
        memory_format = (torch.channels_last if len(metadata.shape) == 4 else torch.channels_last_3d
                         ) if metadata.stride is not None and len(metadata.shape) in (4, 5) and metadata.stride[1] == 1 else torch.contiguous_format
        geometry = PaddingGeometry(metadata.shape, tuple(pads), mode, metadata.dtype, memory_format)
        return CallNode("autodiff_primitive_pad", node.target, node.args, node.kwargs, (node.value,), geometry)
    if name == "dropout":
        target = node.target
        p = target.p if type(target) is torch.nn.Dropout else argument(node, 1, "p", 0.5)
        training = target.training if type(target) is torch.nn.Dropout else argument(node, 2, "training", True)
        if not training or p == 0:
            return CallNode("autodiff_identity", lambda x: x.clone(), node.args[:1], {}, (node.value,))
        mask = TensorValue(f"{node.value.name}__dropout_mask", replace(node.value.meta, dtype=torch.bool, requires_grad=False))
        return CallNode("autodiff_dropout", partial(torch.ops.aten.native_dropout.default, p=p, train=True),
                        node.args[:1], {}, (node.value, mask), p)
    clone = node
    clone.op = f"autodiff_primitive_{name}"
    return clone


def _reshape(g, shape):
    return g.reshape(shape)


def _matmul_gradient(g, other, source_shape, left):
    vector_source = len(source_shape) == 1
    if left:
        if other.ndim == 1:
            value = g.unsqueeze(-1) * other
        else:
            value = torch.matmul(g.unsqueeze(-2) if vector_source else g, other.transpose(-2, -1))
            if vector_source:
                value = value.squeeze(-2)
    else:
        if other.ndim == 1:
            value = other.unsqueeze(-1) * g.unsqueeze(-2) if not vector_source else other * g
        else:
            value = torch.matmul(other.transpose(-2, -1), g.unsqueeze(-1) if vector_source else g)
            if vector_source:
                value = value.squeeze(-1)
    return value.sum_to_size(source_shape)


def _reduction(g, shape, dims, keepdim, mean):
    if dims is None or isinstance(dims, (tuple, list)) and not dims:
        dims = tuple(range(len(shape)))
    elif isinstance(dims, int):
        dims = (dims,)
    dims = tuple(dimension % len(shape) for dimension in dims) if shape else ()
    if not keepdim:
        for dimension in sorted(dims):
            g = g.unsqueeze(dimension)
    value = g.expand(shape)
    if mean:
        from math import prod

        value = value / prod(shape[dimension] for dimension in dims)
    return value


def _slice_gradient(g, shape, index):
    value = torch.zeros(shape, device=g.device, dtype=g.dtype)
    if isinstance(index, torch.Tensor) and not (index.dtype == torch.bool and index.ndim == 0):
        return value.index_put((index,), g, accumulate=True)
    if isinstance(index, tuple) and index and all(
            isinstance(item, torch.Tensor) and not (item.dtype == torch.bool and item.ndim == 0)
            for item in index):
        return value.index_put(index, g, accumulate=True)
    if isinstance(index, (torch.Tensor, list)) or isinstance(index, tuple) and any(
            isinstance(item, (torch.Tensor, list, tuple)) for item in index):
        from math import prod

        locations = torch.arange(prod(shape), device=g.device).reshape(shape)[index]
        return value.flatten().scatter_add_(0, locations.reshape(-1), g.reshape(-1)).reshape(shape)
    value[index] = g
    return value


def _repeat_gradient(g, shape, repeats):
    original_shape = shape
    shape = (1,) * (len(repeats) - len(shape)) + tuple(shape)
    expanded = tuple(item for pair in zip(repeats, shape) for item in pair)
    return g.reshape(expanded).sum(tuple(range(0, len(expanded), 2))).reshape(original_shape)


def multi_kind(target):
    names = ("split", "chunk", "unbind")
    if isinstance(target, partial) and target.func is call_method_helper:
        return target.args[0] if target.args[0] in names else None
    for name in names:
        if target is getattr(torch, name, None):
            return name
    if isinstance(target, torch._ops.OpOverload):
        name = target._schema.name.removeprefix("aten::")
        return name if name in names else None
    return None


def multi_vjp(ctx, node, gradients):
    name = node.op.removeprefix("autodiff_multi_")
    x = node.args[0]
    dim = argument(node, 1 if name == "unbind" else 2, "dim", 0)
    present = tuple(index for index, gradient in enumerate(gradients) if gradient is not None)
    if not present:
        return ()
    shapes = tuple(value.meta.shape for value in node.values)

    def concatenate(*values):
        first = values[0]
        mapping = dict(zip(present, values))
        components = [mapping[index] if index in mapping else torch.zeros(shape, device=first.device, dtype=first.dtype)
                      for index, shape in enumerate(shapes)]
        return torch.stack(components, dim) if name == "unbind" else torch.cat(components, dim)

    value = emit(ctx, "geometry.split_vjp", concatenate,
                 tuple(gradients[index] for index in present), x.meta)
    return (GradWrapper(x, value),)


def primitive_vjp(ctx, node, gradient):
    name = node.op.removeprefix("autodiff_primitive_")
    x = node.args[0]
    if name == "pow":
        exponent = argument(node, 1, "exponent")
        saved_x = ctx.saved(x) if isinstance(x, TensorValue) else x
        saved_exponent = ctx.saved(exponent) if isinstance(exponent, TensorValue) else exponent
        contributions = []
        if ctx.needs_grad(x):
            def base_gradient(g, a, b):
                return torch.where(b == 0, torch.zeros_like(g), g * b * a.pow(b - 1)).sum_to_size(x.meta.shape) if isinstance(b, torch.Tensor) else (torch.zeros_like(a) if b == 0 else (g * b * a.pow(b - 1)).sum_to_size(x.meta.shape))
            contributions.append(GradWrapper(x, emit(ctx, "pointwise.pow_base_vjp", base_gradient,
                (gradient, saved_x, saved_exponent), x.meta)))
        if ctx.needs_grad(exponent):
            def exponent_gradient(g, a, b):
                a = torch.as_tensor(a, device=g.device, dtype=g.dtype)
                value = g * a.pow(b) * a.log()
                value = torch.where((a == 0) & (b >= 0), torch.zeros_like(value), value)
                return value.sum_to_size(exponent.meta.shape)
            contributions.append(GradWrapper(exponent, emit(ctx, "pointwise.pow_exponent_vjp", exponent_gradient,
                (gradient, saved_x, saved_exponent), exponent.meta)))
        return contributions
    if name in ("cat", "stack"):
        tensors = x
        dim = argument(node, 1, "dim", 0)
        offset, contributions = 0, []
        for index, tensor in enumerate(tensors):
            width = 1 if name == "stack" else tensor.meta.shape[dim]
            if ctx.needs_grad(tensor):
                def select(g, start, count, shape, axis, dtype):
                    return g.narrow(axis, start, count).reshape(shape).to(dtype)
                value = emit(ctx, f"geometry.{name}_vjp", partial(select, start=index if name == "stack" else offset,
                    count=width, shape=tensor.meta.shape, axis=dim, dtype=tensor.meta.dtype), (gradient,), tensor.meta)
                contributions.append(GradWrapper(tensor, value))
            offset += width
        return contributions
    if name in ("matmul", "bmm", "mm"):
        y = node.args[1]
        return tuple(GradWrapper(source, emit(ctx, "matrix.matmul_vjp",
                    partial(_matmul_gradient, source_shape=source.meta.shape, left=left),
                    (gradient, ctx.saved(other, use=node.saved_operand_uses.get(index))), source.meta))
                    for source, other, left, index in ((x, y, True, 1), (y, x, False, 0)) if ctx.needs_grad(source))
    if not ctx.needs_grad(x):
        return ()
    if name == "pad":
        from deepopt.opt.nn.padding import padding_adjoint
        from .activation_saving import SavedUse

        geometry = node.specification
        ctx.record_residual(x, (), representation="metadata_only",
            use=SavedUse(consumer=node.op, role="input_gradient"),
            recipe={"shape": geometry.shape, "pads": geometry.pads, "mode": geometry.mode,
                    "dtype": str(geometry.dtype), "memory_format": str(geometry.memory_format)},
            retention_reason="Padding adjoint requires geometry only")
        value = emit(ctx, "padding.metadata_vjp", partial(padding_adjoint, geometry=geometry), (gradient,), x.meta)
    elif name in ("sum", "mean"):
        target = partial(_reduction, shape=x.meta.shape, dims=argument(node, 1, "dim"),
                         keepdim=argument(node, 2, "keepdim", False), mean=name == "mean")
        value = emit(ctx, "reduction.vjp", target, (gradient,), x.meta)
    elif name in ("reshape", "view", "flatten", "unflatten", "squeeze", "unsqueeze"):
        value = emit(ctx, "geometry.reshape_vjp", partial(_reshape, shape=x.meta.shape), (gradient,), x.meta)
    elif name in ("transpose", "t", "permute"):
        if name == "permute":
            dims = node.args[1] if len(node.args) == 2 and isinstance(node.args[1], (tuple, list)) else node.kwargs.get("dims", node.args[1:])
            dims = tuple(d % len(dims) for d in dims)
            permutation = tuple(dims.index(index) for index in range(len(dims)))
            target = lambda g: g.permute(permutation)
        else:
            axes = (0, 1) if name == "t" else (argument(node, 1, "dim0"), argument(node, 2, "dim1"))
            target = (lambda g: g) if name == "t" and len(x.meta.shape) < 2 else lambda g: g.transpose(*axes)
        value = emit(ctx, "geometry.transpose_vjp", target, (gradient,), replace(x.meta, stride=None))
    elif name == "softmax":
        dim = argument(node, 1, "dim", -1)
        value = emit(ctx, "reduction.softmax_vjp", lambda g, y: torch.ops.aten._softmax_backward_data.default(
            g, y, dim, x.meta.dtype), (gradient, ctx.saved(node.value)), x.meta)
    elif name == "getitem":
        index = argument(node, 1, "index")
        from .joint import map_tree
        mapping = {value: ctx.saved(value).value for value in node.inputs() if value is not x}
        saved_index = map_tree(index, mapping)
        value = emit(ctx, "geometry.slice_vjp", lambda g, index: _slice_gradient(g, x.meta.shape, index), (gradient, saved_index), x.meta)
    elif name in ("expand", "expand_as"):
        value = emit(ctx, "geometry.broadcast_vjp", lambda g: g.sum_to_size(x.meta.shape), (gradient,), x.meta)
    elif name == "repeat":
        repeats = node.args[1] if len(node.args) == 2 and isinstance(node.args[1], (tuple, list)) else node.kwargs.get("repeats", node.args[1:])
        value = emit(ctx, "geometry.repeat_vjp", partial(_repeat_gradient, shape=x.meta.shape, repeats=repeats), (gradient,), x.meta)
    elif name == "exp":
        # CUDA autocast can promote a BF16 exp operand to an FP32 result. The
        # derivative uses that actual forward result, rather than replaying exp
        # in the operand dtype after backward has disabled autocast.
        from .activation_saving import SavedUse

        value = gradient * ctx.saved(node.value, use=SavedUse(
            role="exp.forward_result_coefficient", rounding_domain="forward_result_dtype"))
    elif name in ("pow", "square", "log", "sqrt", "rsqrt", "sin", "cos", "abs"):
        exponent = argument(node, 1, "exponent", 2)
        functions = {"pow": lambda g, a: g * exponent * a.pow(exponent - 1), "square": lambda g, a: g * 2 * a,
            "log": lambda g, a: g / a,
            "sqrt": lambda g, a: g / (2 * a.sqrt()), "rsqrt": lambda g, a: -0.5 * g * a.pow(-1.5),
            "sin": lambda g, a: g * a.cos(), "cos": lambda g, a: -g * a.sin(), "abs": lambda g, a: g * a.sign()}
        value = emit(ctx, f"pointwise.{name}_vjp", functions[name], (gradient, ctx.saved(x)), x.meta)
    else:
        value = emit(ctx, "geometry.identity_vjp", lambda g: g.to(x.meta.dtype), (gradient,), x.meta)
    return (GradWrapper(x, value),)


def dropout_vjp(ctx, node, gradients):
    x = node.args[0]
    p = node.specification
    value = emit(ctx, "dropout.mask_vjp", partial(torch.ops.aten.native_dropout_backward.default,
        scale=0.0 if p == 1 else 1.0 / (1.0 - p)), (gradients[0], ctx.saved(node.values[1])), x.meta)
    return (GradWrapper(x, value),)


def register_rules(registry):
    from .residual_primitives import register_rules as register_residual_rules

    register_residual_rules(registry)
    if not registry.has_backward_rule("autodiff_storage_add"):
        def storage_add_vjp(ctx, node, gradients):
            contributions = []
            for index, target in enumerate(node.args):
                if ctx.needs_grad(target):
                    function = partial(node.specification.gradient, index=index)
                    value = emit(ctx, "storage.add_vjp", function, (gradients[0],), target.meta)
                    contributions.append(GradWrapper(target, value))
            return contributions
        registry.register_backward_rule("autodiff_storage_add", storage_add_vjp, multi_output=True)
    if not registry.has_backward_rule("autodiff_storage_cat"):
        def storage_cat_vjp(ctx, node, gradients):
            values, dim = node.args
            start, contributions = 0, []
            for target in values:
                length = target.meta.shape[dim]
                if ctx.needs_grad(target):
                    function = lambda gradient, begin=start, size=length: gradient.narrow(dim, begin, size)
                    contributions.append(GradWrapper(target, emit(ctx, "storage.cat_vjp", function, (gradients[0],), target.meta)))
                start += length
            return contributions
        registry.register_backward_rule("autodiff_storage_cat", storage_cat_vjp, multi_output=True)
    if not registry.has_backward_rule("autodiff_storage_exp"):
        def storage_exp_vjp(ctx, node, gradients):
            x = node.args[0]
            value = emit(ctx, "exp.exact_input_vjp", node.specification.backward,
                (gradients[0], *(ctx.saved(value) for value in node.values[1:])), x.meta)
            spec = node.specification
            if spec.packed:
                from math import prod

                ctx.record_codec(x, node.values[1:], representation="packed_storage_exp_coefficient",
                    recipe={"bits": spec.bits, "group_size": spec.group_size,
                            "source_shape": spec.shape, "matrix_shape": (1, prod(spec.shape))},
                    role="exp.pre_output_rounding_coefficient")
            return (GradWrapper(x, value),)
        registry.register_backward_rule("autodiff_storage_exp", storage_exp_vjp, multi_output=True)
    for name in ("split", "chunk", "unbind"):
        key = f"autodiff_multi_{name}"
        if not registry.has_backward_rule(key):
            registry.register_backward_rule(key, multi_vjp, multi_output=True)
    for name in METHODS | {"cat", "stack", "getitem", "pad"}:
        key = f"autodiff_primitive_{name}"
        if not registry.has_backward_rule(key):
            registry.register_backward_rule(key, primitive_vjp)
    if not registry.has_backward_rule("autodiff_dropout"):
        registry.register_backward_rule("autodiff_dropout", dropout_vjp, multi_output=True)
        registry.register_backward_rule("autodiff_identity", lambda ctx, node, g: (GradWrapper(node.args[0], g),))
