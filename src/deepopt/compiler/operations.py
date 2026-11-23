"""Identify DeepOpt operations whose current custom autograd owns saved state."""


def is_deepopt_target(target):
    from torch import nn

    if getattr(target, "_deepopt_compiler_operation", False):
        return True
    kind = type(target) if isinstance(target, nn.Module) else target
    return getattr(kind, "__module__", "").startswith("deepopt.opt.nn.")


def activation_call(target, args, kwargs):
    """Resolve a trusted activation's per-call input/write contract."""
    import torch
    from .native_stateful import native_module, functional_activation

    captured = getattr(target, "_deepopt_compiler_operation", False)
    if captured:
        args, kwargs = args[:2]
    module = native_module(target)
    if type(target) in (torch.nn.ReLU, torch.nn.SiLU):
        module = target
    if module is not None and hasattr(module, "inplace"):
        source = args[0] if args else kwargs.get("input")
        flag = kwargs.get("inplace", args[1] if len(args) > 1 else None)
        flag = module.inplace if flag is None else flag
        kind = getattr(module, "kind", "relu" if isinstance(module, torch.nn.ReLU) else "silu" if isinstance(module, torch.nn.SiLU) else None)
        return (kind, source, flag) if kind in {"relu", "silu", "gelu", "sigmoid", "tanh", "relu6"} else None
    original = getattr(target, "original", target)
    if original in (torch.relu, torch.relu_, torch.nn.functional.relu):
        source = args[0] if args else kwargs.get("input")
        return "relu", source, original is torch.relu_ or kwargs.get("inplace", args[1] if len(args) > 1 else False)
    if functional_activation(target) is not None:
        from deepopt.opt.nn.activation.spec import arguments

        result = arguments(original, args, kwargs)
        if result is not None:
            kind, source, options = result
            return kind, source, options["inplace"]
    return None


def has_semantic_fresh_output(target):
    """Trusted compiler recipes which do not write input tensor storage.

    Cache/report bookkeeping is ordered at the operation, independently of
    tensor aliasing. Unknown DeepOpt operations keep conservative effects.
    """
    from .native_linear import native_module
    from .native_stateful import native_module as stateful_module, functional_activation, functional_exp
    from .linear import linear_module
    from .native_attention import original
    from .native_operations import native_module as spatial_module, functional_convolution
    from .native_mha import native_module as mha_module

    if original(target) is not None:
        return True
    if functional_exp(target):
        return True
    from deepopt.opt.nn.primitive_state import NEAREST2D_FUNCTIONS

    if getattr(target, "original", target) in NEAREST2D_FUNCTIONS:
        return True
    import operator
    import torch
    if getattr(target, "original", target) in (torch.cat, torch.concat, torch.concatenate, torch.add,
                                               operator.add, torch.Tensor.add, torch.Tensor.__add__):
        return True
    if getattr(target, "_deepopt_compiler_operation", False) and functional_activation(target) is not None:
        return True
    if functional_convolution(target) is not None:
        return True
    from deepopt.opt.nn.functional import do_linear

    if getattr(target, "original", target) is do_linear:
        return True

    module = native_module(target) or stateful_module(target) or linear_module(target) or spatial_module(target) or mha_module(target)
    return module is not None and not getattr(module, "inplace", False)


def numerical_scope():
    """Compare numerical policy before execution without reading tensor data."""
    import torch
    from deepopt.opt._autocast import (
        _active_fp8,
        _active_compression,
        _active_attention,
    )
    from deepopt.kernel_policy import current_kernel_policy
    from deepopt._bf16_policy import current_bf16_policy
    from deepopt._activation_memory import current_activation_memory
    from deepopt.opt._recipe import bf16_recipe
    from deepopt.opt._execution import finite_checks_enabled, backward_finite_check_policy
    from deepopt.opt.backends._shared import activation_codec as codec
    from deepopt.opt.backends._shared.normalization import normalization_variant, normalization_tuning
    from deepopt.opt.backends._shared.pooling import pooling_kernel_variant
    from deepopt.opt.backends._shared.gemm_policy import linear_forward_backend, linear_backward_policy, gradient_reduction_policy

    return (
        _active_fp8.get(),
        _active_compression.get(),
        _active_attention.get(),
        tuple(current_bf16_policy(device) for device in ("cpu", "cuda")),
        tuple(current_activation_memory(device) for device in ("cpu", "cuda")),
        tuple(tuple(sorted(bf16_recipe(current_bf16_policy(device)).items())) for device in ("cpu", "cuda")),
        torch.is_grad_enabled(),
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cpu"),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
        torch.is_autocast_cache_enabled(),
        current_kernel_policy(),
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        finite_checks_enabled(), backward_finite_check_policy(),
        tuple(getattr(codec, name)() for name in ("activation_layout_policy", "activation_pointwise_tile",
            "activation_packing", "spatial_activation_layout", "activation_pack_warps", "producer_packing",
            "gelu_saved_state", "silu_saved_state", "strided_indexing", "convolution_codec",
            "attention_packing", "attention_gradient_layout")),
        normalization_variant(), normalization_tuning(), pooling_kernel_variant(),
        linear_forward_backend(), linear_backward_policy(), gradient_reduction_policy(),
        torch.are_deterministic_algorithms_enabled(),
    )


_MODULE_FIELDS = (
    "_deepopt_activation_memory_plan",
    "_deepopt_bf16_primitive_recipe",
    "_deepopt_attention_recipe_version",
    "attention_backend",
    "native_bf16_only",
    "training",
    "inplace",
    "approximate",
    "negative_slope",
    "min_val",
    "max_val",
    "beta",
    "threshold",
    "eps",
    "momentum",
    "track_running_stats",
    "padding",
    "padding_mode",
    "_reversed_padding_repeated_twice",
    "stride",
    "dilation",
    "groups",
    "kernel_size",
    "ceil_mode",
    "return_indices",
    "output_padding",
    "normalized_shape",
    "affine",
    "elementwise_affine",
    "num_groups",
    "num_heads",
    "batch_first",
    "embed_dim",
    "head_dim",
    "kdim",
    "vdim",
    "dropout",
    "add_zero_attn",
    "_fp8_out_projection",
    "output_size",
    "count_include_pad",
    "divisor_override",
    "start_dim",
    "end_dim",
    "unflattened_size",
)
_UNKNOWN = object()


def _host_value(value):
    from deepopt._activation_memory import ActivationMemoryPlan

    if isinstance(value, ActivationMemoryPlan):
        return value
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (tuple, list)):
        values = tuple(_host_value(item) for item in value)
        return _UNKNOWN if any(item is _UNKNOWN for item in values) else values
    return _UNKNOWN


def module_contracts(graph_module):
    """Snapshot host configuration used by static native and opaque calls."""
    from torch import nn

    modules = {}
    for node in graph_module.graph.nodes:
        targets = ()
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)
            modules[id(module)] = (module, graph_module, node.target)
        elif node.op == "call_function" and getattr(
            node.target, "_deepopt_compiler_operation", False
        ):
            targets = (
                getattr(node.target, "root", None),
                getattr(node.target, "module", None),
            )
            root = getattr(node.target, "root", None)
            path = getattr(node.target, "path", None)
            module = getattr(node.target, "module", None)
            if isinstance(root, nn.Module) and isinstance(module, nn.Module) and path is not None:
                modules[id(module)] = (module, root, path)
        for module in targets:
            if isinstance(module, nn.Module):
                modules.setdefault(id(module), (module, None, None))
    contracts = []
    for module, owner, path in modules.values():
        fields = tuple(
            (name, _host_value(getattr(module, name)))
            for name in _MODULE_FIELDS
            if hasattr(module, name)
        )
        contracts.append(
            (
                module,
                tuple((name, value) for name, value in fields if value is not _UNKNOWN),
                owner,
                path,
                getattr(module, "plan", None),
            )
        )
    return tuple(contracts)


def module_contracts_changed(contracts):
    return any(
        (owner is not None and owner.get_submodule(path) is not module)
        or getattr(module, "plan", None) is not plan
        or any(
            _host_value(getattr(module, name, _UNKNOWN)) != value
            for name, value in fields
        )
        for module, fields, owner, path, plan in contracts
    )
