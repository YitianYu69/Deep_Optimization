"""Typed pooling projections and provable global-pool flatten relays."""

import operator

import torch
from torch import nn, fx

from ..nn.pooling.spec import SPECS, MODULE_SPECS, call_kind, FORCE_INDICES


def pool_node(graph, node):
    if node.op == "call_module":
        module = graph.get_submodule(node.target)
        kind = getattr(module, "kind", None)
        if kind in SPECS:
            fields = {name for name, _ in SPECS[kind].fields}
            if not getattr(module, "_pool_fields", ()):
                options = SPECS[kind].arguments(node.args[1:], {k: v for k, v in node.kwargs.items() if k in fields})
            else:
                options = {name: getattr(module, name) for name in fields if hasattr(module, name)}
                if SPECS[kind].family == "unpool":
                    if len(node.args) > 1:
                        options["indices"] = node.args[1]
                    if len(node.args) > 2:
                        options["output_size"] = node.args[2]
                options.update({k: v for k, v in node.kwargs.items() if k in fields})
            if getattr(module, "force_indices", False):
                options["return_indices"] = True
            return kind, options
        spec = MODULE_SPECS.get(type(module))
        return (spec.name, {name: getattr(module, name) for name, _ in spec.fields if hasattr(module, name)}) if spec else (None, {})
    if node.op == "call_function":
        kind = call_kind(node.target)
        if kind is not None:
            fields = {name for name, _ in SPECS[kind].fields}
            options = SPECS[kind].arguments(node.args[1:], {key: value for key, value in node.kwargs.items() if key in fields})
            if node.target in FORCE_INDICES or getattr(node.target, "_deepopt_pool_indices", False):
                options["return_indices"] = True
            return kind, options
    return None, {}


def annotate_pool_nodes(graph):
    for node in graph.graph.nodes:
        kind, options = pool_node(graph, node)
        if kind is not None:
            node.meta["_deepopt_pool_tuple"] = options.get("return_indices") is True


def value_projection(node):
    return (node.op == "call_function" and node.target is operator.getitem and len(node.args) == 2
            and isinstance(node.args[0], fx.Node) and node.args[0].meta.get("_deepopt_pool_tuple", False)
            and node.args[1] == 0)


def index_projection(node):
    return (node.op == "call_function" and node.target is operator.getitem and len(node.args) == 2
            and isinstance(node.args[0], fx.Node) and node.args[0].meta.get("_deepopt_pool_tuple", False)
            and node.args[1] == 1)


def global_pool_flatten(node, graph):
    parent = node.args[0] if node.args else node.kwargs.get("input")
    if not isinstance(parent, fx.Node):
        return False
    if node.op == "call_module":
        module = graph.get_submodule(node.target)
        flatten = isinstance(module, nn.Flatten) and module.start_dim == 1 and module.end_dim == -1
    elif node.op in ("call_function", "call_method"):
        flatten = node.target in (torch.flatten, "flatten") and node.kwargs.get("start_dim", node.args[1] if len(node.args) > 1 else 0) == 1 and node.kwargs.get("end_dim", node.args[2] if len(node.args) > 2 else -1) == -1
    else:
        flatten = False
    if not flatten:
        return False
    if value_projection(parent):
        parent = parent.args[0]
    kind, options = pool_node(graph, parent)
    size = options.get("output_size")
    return kind is not None and SPECS[kind].family.startswith("adaptive") and (
        size == 1 if isinstance(size, int) else isinstance(size, (tuple, list)) and bool(size) and all(v == 1 for v in size))
