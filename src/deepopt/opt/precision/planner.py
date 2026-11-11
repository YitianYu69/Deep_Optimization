"""Plan scale-aware activation edges once, before wrapping or compilation."""

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch import fx, nn
from torch.nn import functional as F

from .pooling import annotate_pool_nodes, value_projection, index_projection, global_pool_flatten


_frame = ContextVar("deepopt_precision_call_sites", default=None)


@dataclass
class _Frame:
    root: nn.Module
    training: bool
    positions: dict


class _PrecisionForward:
    _deepopt_precision_forward = True

    def __init__(self, owner):
        self.owner = owner
        self.override = owner.__dict__.get("forward")

    def __call__(self, *args, **kwargs):
        if getattr(self.owner, "_deepopt_gradient_dynamic_device", False):
            from torch.utils._pytree import tree_leaves

            sample = next((value for value in tree_leaves((args, kwargs)) if isinstance(value, torch.Tensor)), None)
            if sample is not None:
                from ..scaling.delayed import DelayedScalingState

                for slot in getattr(self.owner, "_deepopt_gradient_edges", {}).values():
                    if slot.scaling is not None and slot.scaling.scales.device != sample.device:
                        slot.scaling.to(device=sample.device)
                for module in self.owner.modules():
                    if isinstance(module, DelayedScalingState) and module.scales.device != sample.device:
                        module.to(device=sample.device)
                self.owner._deepopt_gradient_dynamic_device = False
        token = _frame.set(_Frame(self.owner, self.owner.training, {}))
        try:
            function = self.override or type(self.owner).forward.__get__(self.owner)
            from .._autocast import _linear_mode

            with (_linear_mode() if self.owner._deepopt_precision_functional.get(self.owner.training) or
                  getattr(self.owner, "_deepopt_backward_functional", {}).get(self.owner.training) else nullcontext()):
                return function(*args, **kwargs)
        finally:
            _frame.reset(token)


def normalization_output(module):
    from .gradient import native_boundary_active

    if native_boundary_active():
        return None
    current = _frame.get()
    mode = current.training if current is not None else module.training
    policies = module._precision_outputs.get(mode, (None,))
    if current is None:
        return None
    key = module._precision_key
    index = current.positions.get(key, 0)
    current.positions[key] = index + 1
    return policies[index] if index < len(policies) else None


def functional_normalization(kind):
    from .gradient import native_boundary_active

    if native_boundary_active():
        return None
    current = _frame.get()
    if current is None:
        return None
    key = "functional:" + kind
    position = current.positions.get(key, 0)
    current.positions[key] = position + 1
    paths = current.root._deepopt_precision_functional.get(current.training, {}).get(kind, ())
    return current.root.get_submodule(paths[position]) if position < len(paths) else None


def _relay(node, graph, training):
    if value_projection(node) or global_pool_flatten(node, graph):
        return True
    if node.op == "call_module":
        module = graph.get_submodule(node.target)
        return (isinstance(module, nn.Identity) or isinstance(module, nn.ReLU) and not module.inplace
                or isinstance(module, nn.Dropout) and (module.p == 0 or not training))
    return (node.op == "call_function" and node.target in (F.relu, torch.relu)
            and not node.kwargs.get("inplace", False)
            or node.op == "call_method" and node.target == "relu")


def _metadata(node):
    if index_projection(node):
        return True
    return (node.op == "call_function" and node.target is getattr
            and len(node.args) > 1 and node.args[1] in ("shape", "dtype", "device", "ndim")
            or node.op == "call_method" and node.target in ("size", "stride", "dim", "numel"))


def _consumer(node, graph, producer, settings):
    from ..nn.linear import DOLinear
    from ..nn.hf_conv1d import DOHFConv1D
    from ..nn.functional_linear import DOFunctionalLinear
    from ..nn.convolution.modules import DOConv1d, DOConv2d, DOConv3d, DOFunctionalConvolution
    from ..nn.normalization.modules import DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization, DOLayerNorm
    from ..nn.activation.modules import DOReLU
    from ..nn.activation.elementwise_modules import ACTIVATION_TYPES
    from ..nn.activation.spec import FUNCTIONS, METHODS
    from ..nn.functional import do_linear, do_conv1d, do_conv2d, do_conv3d, do_batch_norm, do_group_norm

    first = node.args[0] if node.args else node.kwargs.get("input", node.kwargs.get("x"))
    if first is not producer:
        return None
    if node.op == "call_module":
        module = graph.get_submodule(node.target)
        from ..nn.pooling.modules import POOL_TYPES
        from ..nn.convolution_transpose.modules import TRANSPOSE_TYPES

        if isinstance(module, POOL_TYPES) and "fp8_training" in module.plan:
            return "either"
        if not hasattr(module, "plan") or "fp8_training" not in module.plan:
            return None
        if isinstance(module, TRANSPOSE_TYPES):
            return "channels"
        if isinstance(module, (DOLinear, DOHFConv1D, DOFunctionalLinear)):
            return "last"
        if isinstance(module, DOFunctionalConvolution):
            padding = node.kwargs.get("padding", node.args[4] if len(node.args) > 4 else 0)
            return None if isinstance(padding, str) and padding == "same" else "channels"
        if isinstance(module, (DOConv1d, DOConv2d, DOConv3d)):
            if module.padding_mode != "zeros" or module.padding == "same" and any(
                d * (k - 1) % 2 for d, k in zip(module.dilation, module.kernel_size)
            ):
                return None
            return "channels"
        if isinstance(module, DOLayerNorm) or isinstance(module, DOFunctionalNormalization) and module.kind == "layer":
            return "last"
        if isinstance(module, (DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization)):
            return "either"
        if isinstance(module, (DOReLU, *ACTIVATION_TYPES, *POOL_TYPES)):
            if getattr(module, "inplace", False) and getattr(module, "kind", "relu") not in ("relu", "relu6"):
                return None
            return "either"
    elif node.op == "call_function":
        from ..nn import functional as doF
        from ..nn.pooling.spec import call_kind, FUNCTIONS as POOL_FUNCTIONS

        if call_kind(node.target) is not None and (node.target not in POOL_FUNCTIONS or (
            settings.intercept_linear if settings.intercept_pooling is None else settings.intercept_pooling
        )):
            return "either"

        if node.target is doF.do_layer_norm or node.target is F.layer_norm and (
            settings.intercept_linear if settings.intercept_normalization is None else settings.intercept_normalization
        ):
            return "last"

        if node.target in FUNCTIONS and settings.intercept_linear or node.target in tuple(
            getattr(doF, "do_" + kind) for kind in ("sigmoid", "silu", "gelu", "tanh", "relu6")
        ):
            from ..nn.activation.spec import arguments

            activation = arguments(node.target, node.args, node.kwargs)
            if activation is not None and activation[2]["inplace"] and activation[0] != "relu6":
                return None
            if node.kwargs.get("inplace", False) and node.target not in (doF.do_relu6,):
                return None
            return "either"
        if node.target in (do_linear, F.linear) and (node.target is do_linear or settings.intercept_linear):
            return "last"
        transposed = tuple(getattr(doF, f"do_conv_transpose{rank}d") for rank in (1, 2, 3))
        native_transposed = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
        if node.target in transposed or node.target in native_transposed and (
            settings.intercept_linear if settings.intercept_convolution is None else settings.intercept_convolution
        ):
            return "channels"
        conv = (do_conv1d, do_conv2d, do_conv3d, F.conv1d, F.conv2d, F.conv3d)
        if node.target in conv and (node.target in conv[:3] or (
            settings.intercept_linear if settings.intercept_convolution is None else settings.intercept_convolution
        )):
            padding = node.kwargs.get("padding", node.args[4] if len(node.args) > 4 else 0)
            return None if isinstance(padding, str) and padding == "same" else "channels"
        if node.target in (do_batch_norm, do_group_norm):
            return "either"
        if node.target in (F.batch_norm, F.group_norm) and (
            settings.intercept_linear if settings.intercept_normalization is None else settings.intercept_normalization
        ):
            return "either"
    elif node.op == "call_method" and node.target in METHODS and settings.intercept_linear:
        return None if node.target.endswith("_") else "either"
    return None


def _requirements(node, graph, training, settings):
    result = set()
    for user in node.users:
        if _metadata(user):
            continue
        if _relay(user, graph, training):
            wanted = _requirements(user, graph, training, settings)
            result.update(({"channels"} if wanted else set()) if global_pool_flatten(user, graph) else wanted)
        else:
            layout = _consumer(user, graph, node, settings)
            if layout is not None:
                result.add(layout)
    return result


def _boundaries(graph, node, layout, training, settings, edges):
    from ..quantization.activation_tensor import dequantize_activation

    for user in tuple(node.users):
        if _metadata(user):
            continue
        if _relay(user, graph, training) and _requirements(user, graph, training, settings):
            edges.append({"producer": node.name, "consumer": user.name, "precision": "fp8", "reason": "FP8 relay",
                          "mode": "train" if training else "eval"})
            _boundaries(graph, user, "last" if global_pool_flatten(user, graph) else layout, training, settings, edges)
            continue
        wanted = _consumer(user, graph, node, settings)
        supported = wanted in (layout, "either")
        edges.append({"producer": node.name, "consumer": user.name, "precision": "fp8" if supported else "native",
                      "reason": "scale-aware consumer" if supported else "unsupported consumer or layout; native boundary",
                      "mode": "train" if training else "eval"})
        if not supported and isinstance(graph, fx.GraphModule) and getattr(graph, "_precision_rewrite_edges", False):
            with graph.graph.inserting_before(user):
                dense = graph.graph.call_function(dequantize_activation, (node,))
            user.replace_input_with(node, dense)


def prepare_precision_plan(model, settings):
    """Analyze both modes in one preparation phase; never trace in forward."""
    if hasattr(model, "_deepopt_precision_plan"):
        return model
    if not hasattr(model, "_deepopt_gradient_dynamic_device"):
        model._deepopt_gradient_dynamic_device = next(iter((*model.parameters(), *model.buffers())), None) is None
    from ..fx.registry import default_registry
    from ..fx.tracer import DOTracer
    from ..nn.normalization.modules import DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization, DOLayerNorm
    from ..nn.activation.modules import DOReLU
    from ..nn.activation.elementwise_modules import ACTIVATION_TYPES, adapter
    from ..nn.activation.spec import arguments, KINDS
    from ..nn import functional as doF
    from ..policy import fp8_plan

    from ..nn.pooling.modules import POOL_TYPES, DOFunctionalPool
    from ..nn.pooling.spec import SPECS, call_kind, FUNCTIONS as POOL_FUNCTIONS, FORCE_INDICES

    from ..nn.convolution_transpose.modules import TRANSPOSE_TYPES, DOFunctionalConvTranspose

    norm_types = (*TRANSPOSE_TYPES, DOBatchNorm2d, DOGroupNorm, DOLayerNorm, DOFunctionalNormalization, DOReLU, *ACTIVATION_TYPES, *POOL_TYPES)
    norms = [module for module in model.modules() if isinstance(module, norm_types)]
    # Avoid introducing tracing/forward overhead into models with no norms.
    # Functional roots still need inspection when interception is enabled.
    if not norms and not settings.enabled:
        return model
    backward_auto = settings.enabled and settings.backward_precision == "auto"
    if not norms and not backward_auto and not any(
        {"batch_norm", "group_norm", "layer_norm", "do_layer_norm", "relu", "relu_", *KINDS, *("do_" + kind for kind in KINDS), "sigmoid_", "tanh_", *SPECS, *("do_" + name for name in SPECS)}.intersection(getattr(getattr(type(module).forward, "__code__", None), "co_names", ()))
        for module in model.modules()
    ):
        return model
    registry = default_registry()
    modes = [(module, module.training) for module in model.modules()]
    edges, failures, functionals = [], [], {True: {}, False: {}}
    backward_edges = []
    plan = fp8_plan(settings, activation_compression=getattr(model, "_deepopt_activation_compression_config", {}))
    if backward_auto:
        from ..nn.functional import _prepare_fp8_plan

        model._deepopt_backward_plan = _prepare_fp8_plan(plan)
    for index, module in enumerate(norms):
        module._precision_key = f"normalization:{index}"
    try:
        for mode in (True, False):
            for module, _ in modes:
                module.training = mode
            try:
                if isinstance(model, fx.GraphModule):
                    graph = model
                else:
                    tracer = DOTracer(extra_leaf_types=(*registry.leaf_types, DOFunctionalNormalization, DOFunctionalPool, DOFunctionalConvTranspose))
                    # Registered root transpose operations are leaves too;
                    # wrapping only for analysis preserves their output_size
                    # API and parameter/state paths on the actual model.
                    trace_root = nn.Sequential(model) if isinstance(model, TRANSPOSE_TYPES) else model
                    graph = fx.GraphModule(trace_root, tracer.trace(trace_root))
                annotate_pool_nodes(graph)
                policies = {id(module): [] for module in norms}
                for node in tuple(graph.graph.nodes):
                    module = graph.get_submodule(node.target) if node.op == "call_module" else None
                    pool_kind = call_kind(node.target) if node.op == "call_function" else None
                    pool_active = settings.intercept_linear if settings.intercept_pooling is None else settings.intercept_pooling
                    if pool_kind is not None and (node.target not in POOL_FUNCTIONS or pool_active):
                        name = f"_deepopt_pool_call_{int(mode)}_{node.name}"
                        if hasattr(model, name):
                            raise ValueError(f"Reserved pooling call-site name exists: {name}")
                        module = DOFunctionalPool(plan=node.kwargs.get("plan") or plan, kind=pool_kind)
                        module.force_indices = node.target in FORCE_INDICES or getattr(node.target, "_deepopt_pool_indices", False)
                        module.train(mode)
                        module._precision_key = name
                        model.add_module(name, module)
                        norms.append(module)
                        policies[id(module)] = []
                        functionals[mode].setdefault(pool_kind, []).append(name)
                    transposed = tuple(getattr(doF, f"do_conv_transpose{rank}d") for rank in (1, 2, 3))
                    native_transposed = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
                    conv_active = settings.intercept_linear if settings.intercept_convolution is None else settings.intercept_convolution
                    if node.op == "call_function" and (node.target in transposed or node.target in native_transposed and conv_active):
                        rank = ((transposed if node.target in transposed else native_transposed).index(node.target) + 1)
                        name = f"_deepopt_transpose_call_{int(mode)}_{node.name}"
                        if hasattr(model, name):
                            raise ValueError(f"Reserved transposed call-site name exists: {name}")
                        parameter = next(model.parameters(), None)
                        module = DOFunctionalConvTranspose(node.kwargs.get("plan") or plan,
                                                           parameter.device if parameter is not None else "cpu", rank)
                        module.train(mode)
                        module._precision_key = name
                        model.add_module(name, module)
                        norms.append(module)
                        policies[id(module)] = []
                        functionals[mode].setdefault(f"conv_transpose{rank}d", []).append(name)
                    activation = arguments(node.target, node.args, node.kwargs) if node.op in ("call_function", "call_method") else None
                    direct_kind = next((kind for kind in KINDS if node.op == "call_function" and
                                        node.target is getattr(doF, "do_" + kind)), None)
                    if direct_kind is not None:
                        x = node.args[0] if node.args else node.kwargs.get("input")
                        options = {"inplace": node.kwargs.get("inplace", node.args[1] if len(node.args) > 1 and direct_kind in ("silu", "relu6") else False),
                                   "approximate": node.kwargs.get("approximate", node.args[1] if len(node.args) > 1 and direct_kind == "gelu" else "none")}
                        activation = direct_kind, x, options
                    if activation is not None and (settings.intercept_linear or direct_kind is not None):
                        kind, _, options = activation
                        name = f"_deepopt_{kind}_call_{int(mode)}_{node.name}"
                        if hasattr(model, name):
                            raise ValueError(f"Reserved activation call-site name exists: {name}")
                        module = adapter(kind, plan=node.kwargs.get("plan") or plan, **options)
                        module.train(mode)
                        module._precision_key = name
                        model.add_module(name, module)
                        norms.append(module)
                        policies[id(module)] = []
                        functionals[mode].setdefault(kind, []).append(name)
                    relu_call = (node.op == "call_function" and node.target in (F.relu, torch.relu, torch.relu_)
                                 or node.op == "call_method" and node.target in ("relu", "relu_"))
                    if relu_call and settings.intercept_linear:
                        name = f"_deepopt_relu_call_{int(mode)}_{node.name}"
                        if hasattr(model, name):
                            raise ValueError(f"Reserved ReLU call-site name exists: {name}")
                        inplace = (node.target in ("relu_", torch.relu_) or
                                   node.kwargs.get("inplace", node.args[1] if len(node.args) > 1 else False))
                        module = DOReLU(plan=plan, inplace=inplace)
                        module.train(mode)
                        module._precision_key = name
                        model.add_module(name, module)
                        norms.append(module)
                        policies[id(module)] = []
                        functionals[mode].setdefault("relu", []).append(name)
                    layer_call = node.op == "call_function" and node.target is doF.do_layer_norm
                    if node.op == "call_function" and (layer_call or node.target in (F.batch_norm, F.group_norm, F.layer_norm) and (
                        settings.intercept_linear if settings.intercept_normalization is None else settings.intercept_normalization
                    )):
                        kind = {F.batch_norm: "batch", F.group_norm: "group", F.layer_norm: "layer", doF.do_layer_norm: "layer"}[node.target]
                        name = f"_deepopt_norm_call_{int(mode)}_{node.name}"
                        if hasattr(model, name):
                            raise ValueError(f"Reserved normalization call-site name exists: {name}")
                        module = DOFunctionalNormalization(node.kwargs.get("plan") or plan, kind)
                        module.train(mode)
                        module._precision_key = name
                        model.add_module(name, module)
                        norms.append(module)
                        policies[id(module)] = []
                        functionals[mode].setdefault(kind, []).append(name)
                    if not isinstance(module, norm_types):
                        continue
                    native_override = (getattr(module, "inplace", False) or (direct_kind is not None or layer_call or pool_kind is not None) and
                                       node.kwargs.get("output_dtype") in (torch.bfloat16, torch.float32, torch.float16, torch.float64))
                    requirements = (_requirements(node, graph, mode, settings)
                                    if settings.enabled and "fp8_training" in module.plan and not native_override else set())
                    concrete = requirements - {"either"}
                    layer = isinstance(module, DOLayerNorm) or isinstance(module, DOFunctionalNormalization) and module.kind == "layer"
                    layout = next(iter(concrete)) if len(concrete) == 1 else ("last" if layer else "channels") if requirements else None
                    policies.setdefault(id(module), []).append(layout)
                    if layout is not None:
                        _boundaries(graph, node, layout, mode, settings, edges)
                    else:
                        edges.extend({"producer": node.name, "consumer": user.name, "precision": "native",
                                      "mode": "train" if mode else "eval", "reason": "no eligible FP8 consumer"}
                                     for user in node.users if not _metadata(user))
                for module in norms:
                    module._precision_outputs[mode] = tuple(policies.get(id(module)) or (None,))
                if backward_auto:
                    from .backward import analyze_backward

                    backward_edges.extend(analyze_backward(model, graph, settings, plan, mode))
            except (RuntimeError, TypeError, ValueError, fx.proxy.TraceError) as error:
                failures.append(f"{'train' if mode else 'eval'}: {type(error).__name__}: {error}")
    finally:
        for module, training in modes:
            module.training = training
    if not norms and not backward_auto:
        return model
    # GraphModules receive explicit native boundaries. Ordinary Python roots
    # retain their type/methods and use the same cached decisions in a scope;
    # the activation tensor handles native boundaries without retracing.
    if isinstance(model, fx.GraphModule):
        model._precision_rewrite_edges = True
        # Both modes have already been analyzed. Insert boundaries using the
        # existing graph and cached producer policies, without tracing again.
        for node in tuple(model.graph.nodes):
            if node.op == "call_module":
                module = model.get_submodule(node.target)
                if isinstance(module, norm_types):
                    choices = module._precision_outputs[model.training]
                    layout = choices[0] if len(choices) == 1 else None
                    if layout is not None:
                        _boundaries(model, node, layout, model.training, settings, [])
        model.graph.lint()
        model.recompile()
        del model._precision_rewrite_edges
    device_tensor = next(iter((*model.parameters(), *model.buffers())), None)
    device = device_tensor.device if device_tensor is not None else torch.device("cpu")
    for module in norms:
        module.prepare_output_scaling(device)
    model._deepopt_precision_functional = functionals
    model._deepopt_precision_plan = {"analyses": 1, "edges": tuple(edges), "fallbacks": tuple(failures)}
    if backward_auto:
        model._deepopt_precision_plan["backward_edges"] = tuple(backward_edges)
    model.forward = _PrecisionForward(model)
    return model


def precision_report(model):
    result = deepcopy(getattr(model, "_deepopt_precision_plan", None))
    if result is not None and hasattr(model, "_deepopt_gradient_edges"):
        result["backward_execution"] = {name: deepcopy(slot.last_execution)
                                         for name, slot in model._deepopt_gradient_edges.items()}
        result["backward_input_execution"] = {name: deepcopy(slot.last_input_execution)
                                               for name, slot in model._deepopt_gradient_edges.items()}
    return result
