"""Plan activation-gradient edges in the existing forward analysis phase."""

from dataclasses import asdict
import inspect

import torch
from torch import nn, fx
from torch.nn import functional as F

from .capabilities import BackwardCapability, module_capability
from .gradient import GradientEdgePlan, NativeGradientBoundary
from .pooling import value_projection, global_pool_flatten
from ..quantization.gradient import GradientSlot, GradientSlots


def node_capability(root, node, settings, default_plan):
    from ..nn import functional as doF

    if node.op == "call_module":
        module = root.get_submodule(node.target)
        cap = module_capability(module) or BackwardCapability("native", "last", False, False, "native module boundary")
        return cap, getattr(module, "plan", default_plan), None, module
    norm_active = settings.intercept_linear if settings.intercept_normalization is None else settings.intercept_normalization
    conv_active = settings.intercept_linear if settings.intercept_convolution is None else settings.intercept_convolution
    functions = {F.linear: ("linear", "linear", "last", settings.intercept_linear),
                 doF.do_linear: ("linear", "linear", "last", True),
                 F.relu: ("relu", "relu", "last", settings.intercept_linear),
                 torch.relu: ("relu", "relu", "last", settings.intercept_linear),
                 doF.do_relu: ("relu", "relu", "last", True),
                 F.batch_norm: ("normalization", "batch_norm", "channels", norm_active),
                 F.group_norm: ("normalization", "group_norm", "channels", norm_active),
                 doF.do_batch_norm: ("normalization", "batch_norm", "channels", True),
                 doF.do_group_norm: ("normalization", "group_norm", "channels", True)}
    functions[F.layer_norm] = ("normalization", "layer_norm", "last", norm_active)
    functions[doF.do_layer_norm] = ("normalization", "layer_norm", "last", True)
    for rank in (1, 2, 3):
        functions[getattr(F, f"conv{rank}d")] = ("convolution", f"conv{rank}d", "channels", conv_active)
        functions[getattr(doF, f"do_conv{rank}d")] = ("convolution", f"conv{rank}d", "channels", True)
        functions[getattr(F, f"conv_transpose{rank}d")] = ("convolution", f"conv_transpose{rank}d", "channels", conv_active)
        functions[getattr(doF, f"do_conv_transpose{rank}d")] = ("convolution", f"conv_transpose{rank}d", "channels", True)
    from ..nn.activation.spec import FUNCTIONS, METHODS, KINDS

    for function, kind in FUNCTIONS.items():
        functions[function] = (kind, kind, "last", settings.intercept_linear)
    for kind in KINDS:
        functions[getattr(doF, "do_" + kind)] = (kind, kind, "last", True)
    from ..nn.pooling.spec import SPECS, FUNCTIONS as POOL_FUNCTIONS

    pool_active = settings.intercept_linear if settings.intercept_pooling is None else settings.intercept_pooling
    for function, kind in POOL_FUNCTIONS.items():
        functions[function] = ("pooling", kind, "channels", pool_active)
    for kind in SPECS:
        functions[getattr(doF, "do_" + kind)] = ("pooling", kind, "channels", True)
        alias = getattr(doF, "do_" + kind + "_with_indices", None)
        if alias is not None:
            functions[alias] = ("pooling", kind, "channels", True)
    entry = functions.get(node.target) if node.op == "call_function" else None
    if node.op == "call_method" and node.target in ("relu", "relu_"):
        entry = ("relu", "relu", "last", settings.intercept_linear)
    elif node.op == "call_method" and node.target in METHODS:
        kind = METHODS[node.target]
        entry = (kind, kind, "last", settings.intercept_linear)
    if node.op == "call_function" and node.target in (F.scaled_dot_product_attention, doF.do_scaled_dot_product_attention):
        return BackwardCapability("attention", "last", False, False, "native SDPA backward boundary"), default_plan, None, None
    if entry is None:
        return None, default_plan, None, None
    kind, call, layout, active = entry
    plan = node.kwargs.get("plan", default_plan)
    if not isinstance(plan, dict) or "fp8_training" not in plan:
        return BackwardCapability(kind, layout, False, False, "native explicit plan"), default_plan, call, None
    if kind == "convolution" and isinstance(node.kwargs.get("padding", node.args[4] if len(node.args) > 4 else 0), str):
        active = False
    training = plan["fp8_training"]
    emits = active and (kind in ("relu", "normalization", "pooling", *KINDS) or call.startswith("conv_transpose") or training["backward_backend"] == "inherit")
    return BackwardCapability(kind, layout, active, emits,
                              "" if emits else "native backend or unintercepted functional boundary"), plan, call if active else None, None


def _data_users(node):
    from .planner import _metadata

    return [user for user in node.users if not _metadata(user)]


def analyze_backward(model, graph, settings, default_plan, mode):
    """Attach only decisions/state; no hooks, input tensors or backward tracing."""
    entries = {node: node_capability(graph, node, settings, default_plan) for node in graph.graph.nodes}
    opaque = any(node.op == "call_function" and entries[node][0] is None and inspect.isfunction(node.target)
                 and not str(getattr(node.target, "__module__", "")).startswith(("torch.", "operator", "builtins"))
                 for node in graph.graph.nodes)
    options = default_plan["fp8_training"]["delayed_scaling"]
    if not hasattr(model, "_deepopt_gradient_edges"):
        # Sequential executes every registered child. Keep numerical state on
        # a real operation, with a non-registering root alias for cached paths.
        candidates = [child for child in model.modules()
                      if not isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict))]
        anchor = next((child for child in candidates if module_capability(child) is not None), None)
        if anchor is None:
            anchor = next((child for child in candidates if type(child).__module__.startswith("torch.nn")),
                          candidates[0] if candidates else None)
        if anchor is None:
            return []
        slots = GradientSlots()
        anchor.add_module("_deepopt_gradient_edges", slots)
        if anchor is not model:
            object.__setattr__(model, "_deepopt_gradient_edges", slots)
        model._deepopt_backward_functional = {True: {}, False: {}}
    device_tensor = next(iter((*model.parameters(), *model.buffers())), None)
    if not hasattr(model, "_deepopt_gradient_dynamic_device"):
        model._deepopt_gradient_dynamic_device = device_tensor is None
    device = device_tensor.device if device_tensor is not None else torch.device("cpu")
    records = []
    for node, (cap, plan, call, module) in entries.items():
        if cap is None:
            continue
        parent = node.args[0] if node.args else node.kwargs.get("input", node.kwargs.get("x"))
        original_parent = parent
        reason, selected = "unsupported or leaf backward boundary", None
        if isinstance(parent, fx.Node):
            relay_fanout = len(_data_users(parent)) > 1
            while value_projection(parent) or global_pool_flatten(parent, graph):
                relay_fanout |= len(_data_users(parent)) > 1
                parent = parent.args[0]
            # Only exact identity relays are transparent to the backward ABI.
            while parent.op == "call_module" and isinstance(graph.get_submodule(parent.target), nn.Identity):
                if len(_data_users(parent)) != 1:
                    break
                parent = parent.args[0]
            target_cap, target_plan, _, _ = entries.get(parent, (None, None, None, None))
            references = []
            fx.map_arg((node.args, node.kwargs), lambda item: references.append(item) or item)
            if relay_fanout or len(_data_users(parent)) > 1 or references.count(original_parent) > 1:
                reason = "gradient accumulation/fan-out boundary"
            elif not cap.emits_fp8:
                reason = cap.reason or "producer cannot emit FP8 with selected backend"
            elif target_cap is not None and target_cap.accepts_fp8:
                target_training = target_plan["fp8_training"]
                layout = target_cap.layout if target_cap.kind in ("linear", "convolution") else cap.layout
                selected = GradientEdgePlan("fp8", str(target_training["fp8_dtypes"]["grad"]).removeprefix("torch.float8_"),
                                            layout, target_plan["block_shapes"]["grad"], target_training["scaling_recipe"],
                                            "compatible backward producer and consumer",
                                            "independent_transpose" if target_cap.kind == "linear" and
                                            target_plan["block_shapes"]["grad"] is not None and
                                            target_plan["block_shapes"]["grad"][0] == 1 else "normal")
                options = target_training["delayed_scaling"]
                reason = selected.reason
                if cap.kind in ("linear", "convolution") and layout != cap.layout:
                    selected = None
                    reason = "gradient output layout requires an unsupported fused permutation"
        edge = selected or GradientEdgePlan(reason=reason)
        if opaque:
            edge = GradientEdgePlan(reason="opaque Python call prevents reliable backward call-site analysis")
        name = f"{'train' if mode else 'eval'}_{node.name}"
        slot = GradientSlot(edge, options, device=device)
        slot.train(mode)
        model._deepopt_gradient_edges[name] = slot
        path = "_deepopt_gradient_edges." + name
        if module is not None:
            if not hasattr(module, "_gradient_slots"):
                module._gradient_slots = {True: (), False: ()}
                module._gradient_key = str(node.target)
            module._gradient_slots[mode] += (path,)
            if not getattr(module.forward, "_deepopt_gradient_operation", False) and not isinstance(module.forward, NativeGradientBoundary):
                module.forward = NativeGradientBoundary(module)
        elif call is not None:
            model._deepopt_backward_functional[mode].setdefault(call, []).append(path)
        records.append({"producer": node.name, "consumer": parent.name if isinstance(parent, fx.Node) else "leaf",
                        "mode": "train" if mode else "eval", **asdict(edge)})
    return records
