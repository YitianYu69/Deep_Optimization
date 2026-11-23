"""Capture native SDPA semantics for DeepOpt's own differentiation engine."""

from dataclasses import dataclass

import torch

from .ir import IRNode, find_tensor_values


def original(target):
    from deepopt.opt.nn import functional as dof

    function = getattr(target, "original", target)
    return function if function in (torch.nn.functional.scaled_dot_product_attention,
        dof.do_scaled_dot_product_attention, dof.scaled_dot_product_attention) else None


@dataclass(eq=False)
class AttentionNode(IRNode):
    op: str
    args: tuple
    kwargs: dict
    value: object
    options: dict

    def inputs(self):
        return list(find_tensor_values((self.args, self.kwargs)))

    def outputs(self):
        return (self.value,)


def extension():
    from .registry import OperatorExtension

    def match(node, gm):
        return node.op == "call_function" and original(node.target) is not None

    def lower(context, node):
        from deepopt.opt._autocast import _get_activation_compression, _get_fp8_plan, _get_attention_backend
        from deepopt.opt.nn.functional import _prepare_linear_plan

        target = context.resolve_target(node)
        args, kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
        if getattr(target, "_deepopt_compiler_operation", False):
            args, kwargs, _, _ = args
        def argument(index, name, default=None):
            return args[index] if len(args) > index else kwargs.get(name, default)
        q, k, v = (argument(index, name) for index, name in enumerate(("query", "key", "value")))
        if any(getattr(value, "meta", None) is None or len(value.meta.shape) != 4 for value in (q, k, v)):
            return context.external(node, reason="Native AutoDiff attention currently requires rank-4 Q/K/V")
        device = torch.device(q.meta.device).type
        selected = kwargs.get("plan")
        compression = _get_activation_compression(device)
        if compression is None:
            selected = selected if selected is not None else _get_fp8_plan(device)
            if isinstance(selected, dict):
                checked = _prepare_linear_plan(selected)
                compression = checked.get("fp8_training", checked)["activation_compression"]
        options = dict(dropout_p=argument(4, "dropout_p", 0.0), is_causal=argument(5, "is_causal", False),
            scale=kwargs.get("scale"), enable_gqa=kwargs.get("enable_gqa", False),
            compression=compression if compression and compression["enabled"] else None,
            backend=kwargs.get("attention_backend") or _get_attention_backend(device) or "native")
        if any(value.meta.dtype == torch.float64 for value in (q, k, v)):
            options["compression"] = None
        value = context.tensor_value(node)
        lowered = AttentionNode("deepopt_sdpa", (q, k, v, argument(3, "attn_mask")), {}, value, options)
        context.add_node(node, lowered)
        return lowered

    return OperatorExtension("deepopt_sdpa", AttentionNode, match, lower, pure=True,
        native_fallback=lambda node, args, kwargs: torch.nn.functional.scaled_dot_product_attention(
            *args, dropout_p=node.options["dropout_p"], is_causal=node.options["is_causal"],
            scale=node.options["scale"], enable_gqa=node.options["enable_gqa"]),
        kernel_codegen=lambda node, context: __import__(
            "deepopt.compiler.autodiff.attention", fromlist=["SDPA"]).SDPA(node, context).inference)
