import torch
from torch import fx
from torch.nn import functional as F 

from .registry import ConversionRegistry
from ..nn.activation.spec import FUNCTIONS, METHODS
from ..nn.pooling.spec import call_kind


def _excluded_function(graph_module, node, excluded_ids):
    # Existing GraphModules may already inline functional projections, so they
    # cannot rely on the tracer's excluded-subtree leaf rule.
    paths = [value[0] for value in node.meta.get("nn_module_stack", {}).values()]
    position = 2 if node.target in (F.group_norm, F.layer_norm) else 3 if node.target is F.batch_norm else 1
    weight = node.args[position] if len(node.args) > position else node.kwargs.get("weight")
    if isinstance(weight, fx.Node) and weight.op == "get_attr":
        paths.append(str(weight.target).rpartition(".")[0])
    for path in paths:
        try:
            module = graph_module.get_submodule(path)
        except AttributeError:
            continue
        if id(module) in excluded_ids:
            return True
    return False


def match_graph(graph_module: fx.GraphModule, registry: ConversionRegistry, *, excluded_ids=()) -> list[fx.Node]:
    matches: list[fx.Node] = []

    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)

            if id(module) not in excluded_ids and registry.lookup(module) is not None:
                matches.append(node)

        elif node.op == "call_function":
            if (call_kind(node.target) is not None or node.target in (F.linear, F.conv1d, F.conv2d, F.conv3d, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d, F.batch_norm, F.group_norm, F.layer_norm, F.relu, torch.relu, torch.relu_, *FUNCTIONS)) and not (
                excluded_ids and _excluded_function(graph_module, node, excluded_ids)
            ):
                matches.append(node)

        elif node.op == "call_method" and node.target in ("relu", "relu_", *METHODS) and not (
            excluded_ids and _excluded_function(graph_module, node, excluded_ids)
        ):
            matches.append(node)

    return matches
