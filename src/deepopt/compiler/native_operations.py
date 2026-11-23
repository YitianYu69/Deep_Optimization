"""Typed capture for prepared spatial and normalization operations."""

from copy import deepcopy
from dataclasses import dataclass

import torch

from .ir import IRNode, find_tensor_values


def materialize_cached_bf16_casts(graph, example_inputs, graph_module, unsafe_values=()):
    """Expose native AMP's shared leaf casts before deriving its adjoints.

    Native autocast caches eligible FP32 leaves, including caller inputs.
    Their BF16 adjoints accumulate at that cast before converting to the FP32
    leaf. An opaque convolution/linear must not hide that shared boundary.
    Compressed eager Functions instead own independent adjoint conversions.
    """
    from dataclasses import replace
    from functools import partial
    from .ir import ExternKernelNode, GetAttrNode, InputNode, TensorValue, register_uses
    from .lowering import call_method_helper
    from .native_linear import NativeLinearNode

    if not torch.is_grad_enabled() or not torch.is_autocast_cache_enabled():
        return ()
    cached, ordered, aliases = {}, [], {}

    def eligible(value):
        metadata = value.meta
        if value in unsafe_values or metadata is None or metadata.dtype != torch.float32 or not metadata.requires_grad:
            return None
        device = torch.device(metadata.device).type
        if not torch.is_autocast_enabled(device) or torch.get_autocast_dtype(device) != torch.bfloat16:
            return None
        producer = value.producer
        if isinstance(producer, InputNode):
            sample = example_inputs[producer.index]
        elif isinstance(producer, GetAttrNode):
            sample = graph_module
            for component in producer.target.split("."):
                sample = getattr(sample, component)
        else:
            return None
        if not isinstance(sample, torch.Tensor) or not sample.is_leaf or sample._base is not None:
            return None
        if isinstance(producer, GetAttrNode):
            key = ("parameter", id(sample))
            paths = aliases.setdefault(key, [])
            if producer.target not in paths:
                paths.append(producer.target)
            return key
        return value

    for node in graph.nodes:
        plan = getattr(node, "plan", None)
        native = (isinstance(node, NativeLinearNode) or isinstance(node, OperationNode)
                  and node.op in ("deepopt_convolution", "deepopt_conv_transpose"))
        if native and plan is not None and "fp8_training" not in plan and not plan["activation_compression"]["enabled"]:
            arguments = list(node.args)
            for index, value in enumerate(arguments[:3]):
                key = eligible(value) if value is not None else None
                if key is None:
                    continue
                cast = cached.get(key)
                if cast is None:
                    from .ir import model_state_origin

                    cast = TensorValue(f"{value.name}__autocast_bf16", replace(value.meta, dtype=torch.bfloat16),
                        provenance=value.provenance, source_name=value.source_name, model_state_copy=True,
                        model_state_origin=model_state_origin(value), model_state_transforms=(*value.model_state_transforms, "to"))
                    conversion = ExternKernelNode("to", partial(call_method_helper, "to"),
                        (value, torch.bfloat16), {}, cast)
                    conversion.bf16_policy = node.bf16_policy
                    ordered.append(conversion)
                    cast.producer = conversion
                    cached[key] = cast
                arguments[index] = cast
            node.args = tuple(arguments)
        ordered.append(node)
    graph.nodes = ordered
    values = {value for node in ordered for value in (*node.inputs(), *node.outputs())}
    for value in values:
        value.users.clear()
    for node in ordered:
        register_uses(node)
    graph.validate()
    return tuple(tuple(paths) for paths in aliases.values() if len(paths) > 1)


def native_module(target):
    from deepopt.opt.nn.convolution.modules import DOConv1d, DOConv2d, DOConv3d, DOFunctionalConvolution
    from deepopt.opt.nn.convolution_transpose.modules import TRANSPOSE_TYPES
    from deepopt.opt.nn.normalization.modules import DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization
    from deepopt.opt.nn.pooling.modules import POOL_TYPES

    module = getattr(target, "module", target)
    native_convolutions = tuple(getattr(torch.nn, f"Conv{prefix}{rank}d")
                                for prefix in ("", "Transpose") for rank in (1, 2, 3))
    return module if (type(module) in (*native_convolutions, DOConv1d, DOConv2d, DOConv3d, DOFunctionalConvolution, *TRANSPOSE_TYPES,
                                       torch.nn.GroupNorm, torch.nn.BatchNorm2d, DOBatchNorm2d, DOGroupNorm, *POOL_TYPES)
                      or type(module) is DOFunctionalNormalization and module.kind in ("batch", "group")) else None


def functional_convolution(target):
    from deepopt.opt.nn import functional as dof

    function = getattr(target, "original", target)
    for prefix, family in (("conv", "convolution"), ("conv_transpose", "conv_transpose")):
        for rank in (1, 2, 3):
            if function in (getattr(torch.nn.functional, f"{prefix}{rank}d"),
                            getattr(dof, f"do_{prefix}{rank}d"), getattr(dof, f"{prefix}{rank}d")):
                return family, rank, f"{prefix}{rank}d"
    return None


def fake_functional_convolution(function, args, kwargs):
    """Allocation-only inference for functional convolution capture."""
    from deepopt.opt.nn.convolution.geometry import _spatial_tuple
    from deepopt.opt._autocast import _get_fp8_plan

    family, rank, _ = functional_convolution(function)
    def argument(index, name, default=None):
        return args[index] if len(args) > index else kwargs.get(name, default)
    x, weight, bias = argument(0, "input"), argument(1, "weight"), argument(2, "bias")
    if weight.ndim != rank + 2 or x.ndim not in (rank + 1, rank + 2):
        raise ValueError("Convolution input/weight ranks do not match")
    stride = _spatial_tuple(argument(3, "stride", 1), rank, "stride", minimum=1)
    dilation = _spatial_tuple(argument(7 if family == "conv_transpose" else 5, "dilation", 1), rank, "dilation", minimum=1)
    padding = argument(4, "padding", 0)
    if isinstance(padding, str):
        if padding == "same" and any(value != 1 for value in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")
        if padding not in ("same", "valid"):
            raise ValueError("Invalid convolution padding")
        same = padding == "same"
        padding = (0,) * rank
    else:
        same = False
        padding = _spatial_tuple(padding, rank, "padding", minimum=0)
    groups = argument(6, "groups", 1)
    spatial = x.shape[-rank:]
    channels = weight.shape[0]
    if family == "conv_transpose":
        extra = _spatial_tuple(argument(5, "output_padding", 0), rank, "output_padding", minimum=0)
        output_spatial = tuple((size - 1) * step - 2 * pad + dil * (kernel - 1) + tail + 1
            for size, step, pad, dil, kernel, tail in zip(spatial, stride, padding, dilation, weight.shape[2:], extra))
        channels = weight.shape[1] * groups
    else:
        output_spatial = tuple(spatial) if same else tuple((size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
            for size, step, pad, dil, kernel in zip(spatial, stride, padding, dilation, weight.shape[2:]))
    if any(size <= 0 for size in output_spatial):
        raise ValueError("Convolution kernel exceeds its input")
    native_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) and x.dtype != torch.float64 else x.dtype
    selected, active = kwargs.get("plan"), _get_fp8_plan(x.device.type)
    if not (isinstance(selected, dict) and selected.get("fp8") is False):
        selected = active if active is not None else selected
    dtype = selected.get("fp8_training", {}).get("out_dtype") or native_dtype if isinstance(selected, dict) else native_dtype
    shape = ((x.shape[0], channels, *output_spatial) if x.ndim == rank + 2 else (channels, *output_spatial))
    output = torch.empty(shape, dtype=dtype, device=x.device)
    if len(shape) == 4 and x.is_contiguous(memory_format=torch.channels_last):
        output = output.contiguous(memory_format=torch.channels_last)
    elif len(shape) == 5 and x.is_contiguous(memory_format=torch.channels_last_3d):
        output = output.contiguous(memory_format=torch.channels_last_3d)
    output.requires_grad_(torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x, weight, bias) if tensor is not None))
    return output


def fake_output(module, args, kwargs, parameters, buffers):
    """Clone only the leaf operation with module-local fake operands."""
    from torch._subclasses.fake_tensor import FakeCopyMode
    from torch.utils._pytree import tree_leaves
    from deepopt.opt.precision.gradient import _active
    from deepopt.opt.precision.planner import _frame, _Frame

    fake = next(value for value in tree_leaves((args, kwargs)) if isinstance(value, torch.Tensor))
    mode = getattr(fake, "_data", fake).fake_mode
    memo = {id(real): symbolic for real, symbolic in zip(module.parameters(), parameters)}
    memo.update({id(real): symbolic for real, symbolic in zip(module.buffers(), buffers)})
    with mode, FakeCopyMode(mode):
        copied = deepcopy(module, memo)
        current = _frame.get()
        # Output representation decisions are per-call-site. Preserve their
        # position map while suppressing gradient-edge/cache bookkeeping on
        # this isolated leaf metadata probe.
        if hasattr(copied, "_gradient_slots"):
            copied._gradient_slots = {}
        local_frame = (_Frame(copied, current.training, dict(current.positions)) if current is not None else None)
        frame, gradient = _frame.set(local_frame), _active.set(None)
        try:
            return copied(*args, **kwargs)
        finally:
            _active.reset(gradient)
            _frame.reset(frame)


@dataclass(eq=False)
class OperationNode(IRNode):
    op: str
    args: tuple
    kwargs: dict
    values: tuple
    module: object
    module_path: str | None
    node_name: str
    options: dict
    plan: object = None
    root: object = None

    def inputs(self):
        return list(find_tensor_values((self.args, self.kwargs)))

    def outputs(self):
        return self.values


def extension():
    from .registry import OperatorExtension

    def match(node, gm):
        target = gm.get_submodule(node.target) if node.op == "call_module" else node.target
        return node.op in ("call_module", "call_function") and (native_module(target) is not None or functional_convolution(target) is not None)

    def lower(context, node):
        from deepopt.opt.nn.normalization.modules import DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization
        from deepopt.opt.nn.convolution.modules import DOFunctionalConvolution
        from deepopt.opt.nn.convolution_transpose.modules import DOFunctionalConvTranspose
        from deepopt.opt.nn.pooling.modules import POOL_TYPES
        from deepopt.opt.nn.convolution_transpose.modules import TRANSPOSE_TYPES

        target = context.resolve_target(node)
        module = native_module(target)
        free_convolution = functional_convolution(target) if module is None else None
        if not context.native_autodiff:
            return context.external(node, reason="Spatial compiler training requires native AutoDiff")
        args, kwargs = context.map_arg(node.args), context.map_arg(node.kwargs)
        if free_convolution is not None:
            from types import SimpleNamespace
            from deepopt.opt.quantization.cache import WeightCache

            if getattr(target, "_deepopt_compiler_operation", False):
                args, kwargs, parameters, buffers = args
                for value in parameters:
                    value.provenance = "parameter"
                for value in buffers:
                    value.provenance = "persistent_buffer"
            root = getattr(target, "root", context.graph_module)
            paths = getattr(root, "_deepopt_backward_functional", {})
            kind = free_convolution[2]
            module = SimpleNamespace(plan=kwargs.get("plan"), rank=free_convolution[1], training=root.training,
                _fp8_weight_cache=WeightCache(), fp8_scaling=kwargs.get("_scaling_state"),
                _gradient_slots={mode: tuple(paths.get(mode, {}).get(kind, ())) for mode in (True, False)},
                _gradient_key=f"functional:{kind}", _compiler_gradient_key=f"gradient_functional:{kind}")
            bound = {}
        elif node.op == "call_module":
            bound = {name: context.parameter(node, name, value) for name, value in
                     (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)) if value is not None}
        else:
            args, kwargs, parameters, buffers = args
            for value in parameters:
                value.provenance = "parameter"
            for value in buffers:
                value.provenance = "persistent_buffer"
            owner = module if getattr(target, "local_operands", False) else getattr(target, "root", module)
            mapped = {id(real): symbolic for real, symbolic in zip(owner.parameters(), parameters)}
            mapped.update({id(real): symbolic for real, symbolic in zip(owner.buffers(), buffers)})
            bound = {name: mapped[id(value)] for name, value in
                     (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)) if value is not None}
        x = args[0] if args else kwargs.get("input", kwargs.get("x"))
        def argument(index, name, default=None):
            return args[index] if len(args) > index else kwargs.get(name, default)
        wrapped = x.meta is not None and x.meta.storage_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        if x.meta is None or x.meta.representation != "dense" and not wrapped:
            return context.external(node, reason="Native AutoDiff spatial input representation is not supported")
        from deepopt.opt.nn.functional import _prepare_linear_plan
        from deepopt.opt._autocast import _get_fp8_plan, _get_activation_compression

        plan = getattr(module, "plan", None)
        if plan is None:
            device = torch.device(x.meta.device).type
            active = _get_fp8_plan(device)
            plan = (_prepare_linear_plan(active) if isinstance(active, dict) else
                    _prepare_linear_plan({"fp8": False, "activation_compression": _get_activation_compression(device) or {}}))
        elif "fp8_training" in plan:
            device = torch.device(x.meta.device).type
            active, compression = _get_fp8_plan(device), _get_activation_compression(device)
            convolution_family = type(module) not in (*POOL_TYPES, DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization)
            if active is False:
                plan = _prepare_linear_plan({"fp8": False, "activation_compression": compression or {}})
            elif isinstance(active, dict):
                if compression is None and convolution_family:
                    compression = plan["fp8_training"]["activation_compression"]
                plan = _prepare_linear_plan(active)
            if compression is not None and "fp8_training" in plan:
                plan = _prepare_linear_plan({**plan, "fp8_training": {**plan["fp8_training"], "activation_compression": compression}})
        options = {}
        if type(module) is DOFunctionalNormalization:
            family = "normalization"
            if module.kind == "batch":
                options = {"kind": "batch", "training": argument(5, "training", False),
                           "momentum": argument(6, "momentum", 0.1), "eps": argument(7, "eps", 1e-5)}
                operands = (x, argument(3, "weight"), argument(4, "bias"), argument(1, "running_mean"),
                            argument(2, "running_var"), None)
            else:
                options = {"kind": "group", "groups": argument(1, "num_groups"), "eps": argument(4, "eps", 1e-5)}
                operands = (x, argument(2, "weight"), argument(3, "bias"), None, None, None)
        elif type(module) in (DOBatchNorm2d, DOGroupNorm, torch.nn.GroupNorm, torch.nn.BatchNorm2d):
            family = "normalization"
            operands = (x, bound.get("weight"), bound.get("bias"), bound.get("running_mean"),
                        bound.get("running_var"), bound.get("num_batches_tracked"))
        elif type(module) in POOL_TYPES:
            from deepopt.opt.nn.pooling.spec import SPECS

            family = "pooling"
            options = {name: getattr(module, name) for name in module._pool_fields}
            options.update(kwargs)
            options = SPECS[module.kind].arguments(args[1:], options)
            if getattr(module, "force_indices", False):
                options["return_indices"] = True
            operands = (x,)
        else:
            native_transposed = tuple(getattr(torch.nn, f"ConvTranspose{rank}d") for rank in (1, 2, 3))
            family = free_convolution[0] if free_convolution is not None else "conv_transpose" if type(module) in (*TRANSPOSE_TYPES, *native_transposed) else "convolution"
            functional = free_convolution is not None or type(module) in (DOFunctionalConvolution, DOFunctionalConvTranspose)
            if functional:
                options = dict(stride=argument(3, "stride", 1), padding=argument(4, "padding", 0),
                    dilation=argument(7 if family == "conv_transpose" else 5, "dilation", 1),
                    groups=argument(6, "groups", 1))
                if family == "conv_transpose":
                    options["output_padding"] = argument(5, "output_padding", 0)
                operands = (x, argument(1, "weight"), argument(2, "bias"))
            else:
                operands = (x, bound.get("weight"), bound.get("bias"))
                options["padding_mode"] = module.padding_mode
                if module.padding_mode != "zeros":
                    options["explicit_padding"] = tuple(module._reversed_padding_repeated_twice)
                if family == "conv_transpose" and (len(args) > 1 or kwargs.get("output_size") is not None):
                    output_size = argument(1, "output_size")
                    if not isinstance(output_size, (tuple, list)) or not all(type(value) is int for value in output_size):
                        return context.external(node, reason="Convolution output_size must have static integer dimensions")
                    rank = len(operands[1].meta.shape) - 2
                    desired = tuple(output_size[-rank:])
                    base = tuple((size - 1) * step - 2 * pad + dil * (kernel - 1) + 1 for size, step, pad, dil, kernel in
                                 zip(x.meta.shape[-rank:], module.stride, module.padding, module.dilation, module.kernel_size))
                    extra = tuple(size - minimum for size, minimum in zip(desired, base))
                    if any(value < 0 or value >= step for value, step in zip(extra, module.stride)):
                        raise ValueError("Requested transposed convolution output_size is outside its valid range")
                    options["output_padding"] = extra
        tensor_meta = node.meta.get("tensor_meta")
        value = node.meta.get("example_value", node.meta.get("val"))
        multiple = ((isinstance(tensor_meta, (tuple, list)) and not hasattr(tensor_meta, "shape"))
                    or isinstance(value, (tuple, list)))
        outputs = context.tensor_values(node) if multiple else (context.tensor_value(node),)
        lowered = OperationNode(f"deepopt_{family}", operands, options if family == "pooling" else {}, outputs, module,
            node.target if node.op == "call_module" else getattr(target, "path", None), node.name, options, plan,
            getattr(target, "root", context.graph_module))
        context.add_node(node, lowered)
        return lowered

    def codegen(node, context):
        from .autodiff.spatial import specification

        spec = specification(node, context)
        return spec.inference

    return OperatorExtension("native_spatial", OperationNode, match, lower, pure=False, kernel_codegen=codegen,
        native_fallback=lambda node, args, kwargs: node.module(args[0]))
