"""
Replace matched modules while reserving parameter and alias ownership
"""

from torch import fx, nn

from .registry import ConversionRegistry
from ..do_nn.functional import do_linear



def rewrite_graph(
    graph_module: fx.GraphModule,
    matches: list[fx.Node], 
    registry: ConversionRegistry,
    *, plan: dict,
    replacements: dict[int, nn.Module] | None = None
) -> fx.GraphModule:

    if replacements is None:
        replacements = {}

    for node in matches:
        if node.op == "call_module":
            original = graph_module.get_submodule(node.target)
            module_id = id(original)

            if module_id not in replacements:
                rule = registry.lookup(original)
                replacements[module_id] = rule.converter(
                    original, plan=plan
                )
    
    for parent in list(graph_module.modules()):
        for name, child in list(parent._modules.items()):
            replacement = replacements.get(id(child))

            if replacement is not None:
                parent._modules[name] = replacement
    
    for node in matches:
        if node.op == "call_function":
            kwargs = dict(node.kwargs)

            if "input" in kwargs:
                kwargs["x"] = kwargs.pop("input")
            
            kwargs["plan"] = plan

            node.target = do_linear
            node.kwargs = kwargs

    graph_module.recompile()
    return graph_module