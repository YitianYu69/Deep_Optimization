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
    replacements: dict[int, nn.Module] | None = None,
    excluded_ids=(),
) -> fx.GraphModule:

    if replacements is None:
        replacements = {}

    for node in matches:
        if node.op == "call_module":
            original = graph_module.get_submodule(node.target)
            module_id = id(original)

            if module_id not in replacements:
                replacements[module_id] = registry.convert(
                    original, plan=plan, excluded_ids=excluded_ids
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
            if plan["fp8_training"].get("scaling_recipe", "current") == "delayed":
                from ..do_nn.functional_module import DOFunctionalLinear

                parameter = next(graph_module.parameters(), None)
                device = parameter.device if parameter is not None else "cpu"
                name = f"_deepopt_fp8_call_{node.name}"
                if hasattr(graph_module, name):
                    raise ValueError(f"Reserved FP8 call-site name already exists: {name}")
                adapter = DOFunctionalLinear(plan, device)
                adapter.train(graph_module.training)
                graph_module.add_submodule(name, adapter)
                kwargs.pop("plan")
                node.op, node.target = "call_module", name
            else:
                node.target = do_linear
            node.kwargs = kwargs

    graph_module.recompile()
    return graph_module
