from torch import fx
from torch.nn import functional as F 

from .registry import ConversionRegistry


def _excluded_function(graph_module, node, excluded_ids):
    # Existing GraphModules may already inline functional projections, so they
    # cannot rely on the tracer's excluded-subtree leaf rule.
    paths = [value[0] for value in node.meta.get("nn_module_stack", {}).values()]
    weight = node.args[1] if len(node.args) > 1 else node.kwargs.get("weight")
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
            if node.target is F.linear and not (excluded_ids and _excluded_function(graph_module, node, excluded_ids)):
                matches.append(node)

    return matches
