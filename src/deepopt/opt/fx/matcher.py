from torch import fx
from torch.nn import functional as F 

from .registry import ConversionRegistry


def match_graph(graph_module: fx.GraphModule, registry: ConversionRegistry) -> list[fx.Node]:
    matches: list[fx.Node] = []

    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)

            if registry.lookup(module) is not None:
                matches.append(node)

        elif node.op == "call_function":
            if node.target is F.linear:
                matches.append(node)

    return matches