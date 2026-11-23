"""Decompose MHA into typed projections, tensor geometry and attention IR."""

from dataclasses import dataclass, replace
from math import sqrt
from types import SimpleNamespace
import operator

import torch

from .ir import CallNode, IRNode, TensorMeta, find_tensor_values
from .trace.builder import GraphBuilder


def native_module(target):
    from deepopt.opt.nn.attention import DOMultiheadAttention

    module = getattr(target, "module", target)
    return module if type(module) in (torch.nn.MultiheadAttention, DOMultiheadAttention) else None


@dataclass(eq=False)
class MHAResultNode(IRNode):
    args: tuple
    kwargs: dict
    values: tuple
    op: str = "autodiff_mha_result"

    def inputs(self):
        return list(find_tensor_values(self.args))

    def outputs(self):
        return self.values


def _result(*values):
    return values[0] if len(values) == 1 else values


def extension():
    from .registry import OperatorExtension

    def match(node, gm):
        target = gm.get_submodule(node.target) if node.op == "call_module" else node.target
        return node.op in ("call_module", "call_function") and native_module(target) is not None

    def lower(context, fx_node):
        from deepopt.opt.nn.attention import DOMultiheadAttention
        from deepopt.opt.nn.functional import _prepare_linear_plan
        from deepopt.opt._autocast import _get_fp8_plan, _get_activation_compression, _get_attention_backend
        from deepopt.opt.quantization.cache import WeightCache
        from .linear import DeepOptLinearNode
        from .native_linear import NativeLinearNode
        from .native_attention import AttentionNode

        if not context.native_autodiff:
            return context.external(fx_node, reason="MHA decomposition requires native AutoDiff")
        target = context.resolve_target(fx_node)
        module = native_module(target)
        custom = type(module) is DOMultiheadAttention
        args, kwargs = context.map_arg(fx_node.args), context.map_arg(fx_node.kwargs)
        if fx_node.op == "call_function":
            args, kwargs, parameters, _buffers = args
            owner = module if getattr(target, "local_operands", False) else getattr(target, "root", module)
            mapping = {id(real): symbolic for real, symbolic in zip(owner.parameters(), parameters)}
            params = {name: mapping[id(value)] for name, value in module.named_parameters()}
        else:
            params = {name: context.parameter(fx_node, name, value) for name, value in module.named_parameters()}
        def argument(index, name, default=None):
            return args[index] if len(args) > index else kwargs.get(name, default)
        query, key, value = (argument(index, name) for index, name in enumerate(("query", "key", "value")))
        if any(tensor.meta is None or len(tensor.meta.shape) not in (2, 3) for tensor in (query, key, value)):
            return context.external(fx_node, reason="Native AutoDiff MHA requires dense rank-2/3 operands")
        if getattr(module, "bias_k", None) is not None or getattr(module, "bias_v", None) is not None or module.add_zero_attn:
            return context.external(fx_node, reason="MHA extra key/value tokens require an explicit decomposition")
        if argument(7, "is_causal", False) and argument(5, "attn_mask") is None:
            raise ValueError("MHA is_causal=True requires an explicit attention mask")
        device = torch.device(query.meta.device).type
        active, compression = _get_fp8_plan(device), _get_activation_compression(device)
        owned_plan = getattr(module, "plan", None)
        if not custom:
            # TorchFunctionMode dispatches the native MHA functional as a
            # whole. Its internal projections/attention do not inherit the
            # DeepOpt interception policy of an explicit DO adapter.
            plan = _prepare_linear_plan({"fp8": False, "activation_compression": {}})
        elif owned_plan is not None and owned_plan.get("fp8") is False:
            plan = (_prepare_linear_plan({**owned_plan, "activation_compression": compression})
                    if compression is not None else owned_plan)
        elif active is False or active is None and owned_plan is None:
            effective_bf16 = query.meta.dtype == torch.bfloat16 or torch.is_autocast_enabled(device) and torch.get_autocast_dtype(device) == torch.bfloat16
            plan = _prepare_linear_plan({"fp8": False, "activation_compression": compression if effective_bf16 and compression else {}})
        else:
            selected = active if isinstance(active, dict) else owned_plan
            plan = _prepare_linear_plan(selected)
            selected_compression = compression if compression is not None else (owned_plan or {}).get("fp8_training", {}).get("activation_compression")
            if selected_compression is not None:
                plan = _prepare_linear_plan({**plan, "fp8_training": {**plan["fp8_training"], "activation_compression": selected_compression}})
        attention_compression = compression
        if attention_compression is None and owned_plan is not None:
            attention_compression = owned_plan.get("fp8_training", owned_plan)["activation_compression"]
        if not custom:
            attention_compression = None
        from deepopt._attention_policy import effective_attention_bits

        codec = ({"group_size": attention_compression["group_size"],
                  "bits": effective_attention_bits(attention_compression)}
                 if attention_compression and attention_compression["enabled"] else None)
        builder = GraphBuilder(context.graph, value_prefix=f"mha_{fx_node.name}_")
        path = fx_node.target if fx_node.op == "call_module" else getattr(target, "path", "")
        root = getattr(target, "root", context.graph_module)
        def metadata(shape, dtype, requires):
            return TensorMeta(tuple(shape), dtype, query.meta.device, requires_grad=requires)
        def emit(op, function, operands, shape, dtype=None, requires=None):
            inputs = tuple(find_tensor_values(operands))
            if dtype is None:
                dtype = inputs[0].meta.dtype
            if requires is None:
                requires = any(tensor.meta.requires_grad for tensor in inputs)
            return builder.emit_native(op, function, tuple(operands), {}, metadata(shape, dtype, requires)).value
        def reshape(tensor, shape):
            return emit("autodiff_primitive_reshape", torch.reshape, (tensor, tuple(shape)), shape)
        def transpose(tensor, a, b):
            shape = list(tensor.meta.shape)
            shape[a], shape[b] = shape[b], shape[a]
            return emit("autodiff_primitive_transpose", torch.transpose, (tensor, a, b), shape)
        def cast(tensor, dtype):
            return emit("autodiff_primitive_to", lambda value: value.to(dtype), (tensor,), tensor.meta.shape, dtype)
        def narrow(tensor, start, length):
            shape = (length, *tensor.meta.shape[1:])
            index = (slice(start, start + length),) + (slice(None),) * (len(shape) - 1)
            return emit("autodiff_primitive_getitem", operator.getitem, (tensor, index), shape)
        self_attention = query is key and key is value
        same_kv = key is value
        unbatched = len(query.meta.shape) == 2
        if unbatched:
            query, key, value = (reshape(tensor, (1, *tensor.meta.shape) if custom else
                (tensor.meta.shape[0], 1, tensor.meta.shape[1])) for tensor in (query, key, value))
        elif custom and not module.batch_first or not custom and module.batch_first:
            query, key, value = (transpose(tensor, 0, 1) for tensor in (query, key, value))
        batch = query.meta.shape[0 if custom else 1]
        length = query.meta.shape[1 if custom else 0]
        source_length = key.meta.shape[1 if custom else 0]
        cache = getattr(module, "_fp8_weight_cache", None)
        if cache is None:
            cache = WeightCache()
        def projection(tensor, weight, bias, role, selected_plan=plan):
            native_dtype = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) and tensor.meta.dtype != torch.float64 else tensor.meta.dtype
            dtype = selected_plan.get("fp8_training", {}).get("out_dtype") or native_dtype
            output = builder.new_value(metadata((*tensor.meta.shape[:-1], weight.meta.shape[0]), dtype,
                any(item.meta.requires_grad for item in (tensor, weight, bias) if item is not None)))
            if "fp8_training" in selected_plan:
                scaling = module._projection_scaling[role] if custom and module._projection_scaling else None
                facade = SimpleNamespace(plan=selected_plan, training=module.training, _fp8_weight_cache=cache,
                    fp8_scaling=scaling, in_features=weight.meta.shape[1], out_features=weight.meta.shape[0])
                operation = DeepOptLinearNode("deepopt_linear", (tensor, weight, bias), {}, output,
                    SimpleNamespace(module=facade, root=root), selected_plan, module_path=f"{path}.{role}", module=facade)
            else:
                operation = NativeLinearNode("bf16_linear", (tensor, weight, bias), {}, output, selected_plan,
                    f"{path}.{role}", output.name, module.training)
            context.graph.add(operation)
            return output
        def split(tensor, count):
            shape = (*tensor.meta.shape[:-1], tensor.meta.shape[-1] // count)
            outputs = tuple(builder.new_value(metadata(shape, tensor.meta.dtype, tensor.meta.requires_grad)) for _ in range(count))
            context.graph.add(CallNode("autodiff_multi_chunk", torch.chunk, (tensor, count, -1), {}, outputs))
            return outputs
        if module.in_proj_weight is not None:
            weight, bias = params["in_proj_weight"], params.get("in_proj_bias")
            if self_attention:
                q, k, v = split(projection(query, weight, bias, "qkv"), 3)
            elif same_kv and not custom:
                q = projection(query, narrow(weight, 0, module.embed_dim), narrow(bias, 0, module.embed_dim) if bias is not None else None, "q")
                k, v = split(projection(key, narrow(weight, module.embed_dim, 2 * module.embed_dim),
                    narrow(bias, module.embed_dim, 2 * module.embed_dim) if bias is not None else None, "kv"), 2)
            else:
                q, k, v = (projection(tensor, narrow(weight, index * module.embed_dim, module.embed_dim),
                    narrow(bias, index * module.embed_dim, module.embed_dim) if bias is not None else None, role)
                    for index, (tensor, role) in enumerate(((query, "q"), (key, "k"), (value, "v"))))
        else:
            bias = params.get("in_proj_bias")
            q, k, v = (projection(tensor, params[f"{role}_proj_weight"],
                narrow(bias, index * module.embed_dim, module.embed_dim) if bias is not None else None, role)
                for index, (tensor, role) in enumerate(((query, "q"), (key, "k"), (value, "v"))))
        if not custom:
            q, k, v = (transpose(tensor, 0, 1) for tensor in (q, k, v))
        q = transpose(reshape(q, (batch, length, module.num_heads, module.head_dim)), 1, 2)
        k = transpose(reshape(k, (batch, source_length, module.num_heads, module.head_dim)), 1, 2)
        v = transpose(reshape(v, (batch, source_length, module.num_heads, module.head_dim)), 1, 2)
        if q.meta.dtype == torch.float64:
            codec, attention_compression = None, None
        attn_mask, padding = argument(5, "attn_mask"), argument(3, "key_padding_mask")
        need_weights, average = argument(4, "need_weights", True), argument(6, "average_attn_weights", True)
        causal = argument(7, "is_causal", False) and not custom and padding is None and not need_weights
        mask = None
        if causal:
            attn_mask = None
        if attn_mask is not None or padding is not None:
            mask_shape = (batch if padding is not None or attn_mask is not None and len(attn_mask.meta.shape) == 3 else 1,
                module.num_heads if attn_mask is not None and len(attn_mask.meta.shape) == 3 else 1,
                length if attn_mask is not None else 1, source_length)
            options = dict(batch=batch, heads=module.num_heads, length=length, source=source_length,
                unbatched=unbatched, force_float32=custom, dtype=query.meta.dtype)
            mask_dtype = torch.float32 if custom else query.meta.dtype
            if attn_mask is not None and attn_mask.meta.dtype != torch.bool and not custom:
                mask_dtype = attn_mask.meta.dtype
            mask = emit("autodiff_mha_mask", lambda a, b, reference: make_mask(a, b, reference, options),
                (attn_mask, padding, q), mask_shape, mask_dtype,
                any(item is not None and item.meta.requires_grad for item in (attn_mask, padding)))
        dropout = module.dropout if module.training else 0.0
        backend = (_get_attention_backend(device) or getattr(module, "attention_backend", "native")) if custom else "native"
        weights = None
        if need_weights:
            if backend == "flash_attn_2":
                raise ValueError("flash_attn_2 does not return full attention weights; use need_weights=False")
            if custom:
                q, k, v = (cast(tensor, torch.float32) for tensor in (q, k, v))
                k_transpose = transpose(k, -2, -1)
                scores = emit("autodiff_primitive_matmul", matmul_full_precision, (q, k_transpose), (batch, module.num_heads, length, source_length))
                if codec is not None:
                    from .autodiff.activation_saving import SavedUse

                    scores.producer.saved_operand_uses = {index: SavedUse(role="mha.qk_coefficient",
                        allow_approximation=True, consumer="mha.matmul", codec=tuple(codec.items())) for index in (0, 1)}
                scores = emit("mul", torch.mul, (scores, module.head_dim ** -0.5), scores.meta.shape)
                if mask is not None:
                    scores = emit("add", torch.add, (scores, mask), scores.meta.shape)
            else:
                compute_dtype = q.meta.dtype
                q, k, v = (cast(tensor, compute_dtype) if tensor.meta.dtype != compute_dtype else tensor for tensor in (q, k, v))
                q = emit("mul", torch.mul, (q, sqrt(1.0 / module.head_dim)), q.meta.shape)
                if mask is None:
                    scores = emit("autodiff_primitive_matmul", torch.matmul, (q, transpose(k, -2, -1)),
                                  (batch, module.num_heads, length, source_length))
                else:
                    scores = emit("autodiff_mha_scores", scores_with_bias, (q, k, mask),
                                  (batch, module.num_heads, length, source_length))
            probabilities = emit("autodiff_primitive_softmax", torch.softmax, (scores, -1), scores.meta.shape)
            if dropout:
                from functools import partial

                dropped = builder.new_value(probabilities.meta)
                drop_mask = builder.new_value(replace(probabilities.meta, dtype=torch.bool, requires_grad=False))
                context.graph.add(CallNode("autodiff_dropout", partial(torch.ops.aten.native_dropout.default, p=dropout, train=True),
                    (probabilities,), {}, (dropped, drop_mask), dropout))
                probabilities = dropped
            attended = emit("autodiff_primitive_matmul", matmul_full_precision if custom else torch.matmul, (probabilities, v), q.meta.shape)
            if custom and codec is not None:
                from .autodiff.activation_saving import SavedUse

                attended.producer.saved_operand_uses = {1: SavedUse(role="mha.value_coefficient",
                    allow_approximation=True, consumer="mha.matmul", codec=tuple(codec.items()))}
            attended = cast(attended, plan.get("fp8_training", {}).get("out_dtype") or (torch.get_autocast_dtype(device)
                if torch.is_autocast_enabled(device) else query.meta.dtype)) if custom else attended
            weights = cast(probabilities, attended.meta.dtype) if probabilities.meta.dtype != attended.meta.dtype else probabilities
        else:
            attended = builder.new_value(metadata(q.meta.shape, q.meta.dtype, any(tensor.meta.requires_grad for tensor in (q, k, v))))
            context.graph.add(AttentionNode("deepopt_sdpa", (q, k, v, mask), {}, attended, dict(
                dropout_p=dropout, is_causal=causal, scale=None, enable_gqa=False,
                compression=attention_compression if attention_compression and attention_compression["enabled"] else None, backend=backend)))
        attended = reshape(transpose(attended, 1, 2), (batch, length, module.embed_dim))
        if not custom:
            attended = transpose(attended, 0, 1)
        output = projection(attended, params["out_proj.weight"], params.get("out_proj.bias"), "out")
        if weights is not None and average:
            weights = emit("autodiff_primitive_mean", torch.mean, (weights, 1), (batch, length, source_length))
        if unbatched:
            output = reshape(output, (length, module.embed_dim))
            if weights is not None:
                weights = reshape(weights, weights.meta.shape[1:])
        elif custom and not module.batch_first or not custom and module.batch_first:
            output = transpose(output, 0, 1)
        values = [builder.new_value(replace(output.meta, stride=None))]
        operands = [output]
        if weights is not None:
            operands.append(weights)
            values.append(builder.new_value(replace(weights.meta, stride=None)))
        lowered = MHAResultNode(tuple(operands), {}, tuple(values))
        context.add_node(fx_node, lowered)
        context.bind_result(fx_node, (values[0], values[1] if weights is not None else None))
        return lowered

    return OperatorExtension("mha", MHAResultNode, match, lower, pure=True,
                             kernel_codegen=lambda node, context: _result, decomposes=True)


def make_mask(attn, padding, reference, options):
    batch, heads, length, source = (options[name] for name in ("batch", "heads", "length", "source"))
    def additive(mask):
        if mask.dtype == torch.bool:
            return torch.zeros_like(mask, dtype=torch.float32 if options["force_float32"] else options["dtype"]).masked_fill(mask, float("-inf"))
        return mask.float() if options["force_float32"] else mask
    result = None
    if attn is not None:
        result = additive(attn)
        result = result.reshape(1, 1, length, source) if attn.ndim == 2 else result.reshape(batch, heads, length, source)
    if padding is not None:
        bias = additive(padding).reshape(batch, 1, 1, source)
        result = bias if result is None else result + bias
    return result


def scores_with_bias(q, k, bias):
    batch, heads, length, width = q.shape
    source = k.shape[-2]
    if bias.shape[:2] == (1, 1):
        matrix = bias.reshape(1, bias.shape[-2], source)
    else:
        matrix = bias.expand(batch, heads, bias.shape[-2], source).reshape(batch * heads, bias.shape[-2], source)
    return torch.baddbmm(matrix,
        q.reshape(batch * heads, length, width), k.transpose(-2, -1).reshape(batch * heads, width, source)).reshape(batch, heads, length, source)


def matmul_full_precision(a, b):
    with torch.autocast(a.device.type, enabled=False):
        return torch.matmul(a, b)


def register_rules(registry):
    from .autodiff.backward_rules import GradWrapper
    from .autodiff.semantic import emit

    def result_vjp(ctx, node, gradients):
        return tuple(GradWrapper(value, gradient) for value, gradient in zip(node.args, gradients) if gradient is not None)
    def mask_vjp(ctx, node, gradient):
        contributions = []
        for index, value in enumerate(node.args[:2]):
            if value is None or not ctx.needs_grad(value):
                continue
            shape = value.meta.shape
            def project(g, index=index, shape=shape):
                if index == 0:
                    return g.reshape(-1, *g.shape[-2:]).sum_to_size(shape) if len(shape) == 3 else g.sum_to_size(shape)
                return g.sum((1, 2)).reshape(shape) if len(shape) == 2 else g.sum((0, 1, 2)).reshape(shape)
            contributions.append(GradWrapper(value, emit(ctx, "mha.mask_vjp", project, (gradient,), value.meta)))
        return contributions
    def scores_vjp(ctx, node, gradient):
        q, k, bias = node.args
        contributions = []
        if ctx.needs_grad(q):
            contributions.append(GradWrapper(q, emit(ctx, "mha.scores_dq", torch.matmul, (gradient, ctx.saved(k)), q.meta)))
        if ctx.needs_grad(k):
            contributions.append(GradWrapper(k, emit(ctx, "mha.scores_dk", lambda g, a: g.transpose(-2, -1) @ a,
                (gradient, ctx.saved(q)), k.meta)))
        if ctx.needs_grad(bias):
            def bias_gradient(g):
                shape = bias.meta.shape
                if shape[:2] == (1, 1):
                    return g.sum((0, 1), keepdim=True).to(bias.meta.dtype).sum_to_size(shape)
                if shape[-2] == 1:
                    return g.sum(-2, keepdim=True).to(bias.meta.dtype).sum_to_size(shape)
                return g.to(bias.meta.dtype).sum_to_size(shape)
            contributions.append(GradWrapper(bias, emit(ctx, "mha.scores_dbias", bias_gradient, (gradient,), bias.meta)))
        return contributions
    if not registry.has_backward_rule("autodiff_mha_result"):
        registry.register_backward_rule("autodiff_mha_result", result_vjp, multi_output=True)
        registry.register_backward_rule("autodiff_mha_mask", mask_vjp)
        registry.register_backward_rule("autodiff_mha_scores", scores_vjp)
