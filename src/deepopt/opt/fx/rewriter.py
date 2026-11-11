"""
Replace matched modules while reserving parameter and alias ownership
"""

import torch
from torch import fx, nn
from torch.nn import functional as F

from .registry import ConversionRegistry
from deepopt.opt.nn.functional import do_linear, do_conv1d, do_conv2d, do_conv3d



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
        if node.op in ("call_function", "call_method"):
            from ..nn.activation.spec import arguments
            from ..nn.activation.elementwise_modules import adapter

            activation = arguments(node.target, node.args, node.kwargs)
            if activation is not None:
                kind, x, options = activation
                name = f"_deepopt_{kind}_{node.name}"
                if hasattr(graph_module, name):
                    raise ValueError(f"Reserved activation call-site name exists: {name}")
                replacement = adapter(kind, plan=plan, **options)
                replacement.train(graph_module.training)
                graph_module.add_submodule(name, replacement)
                node.op, node.target, node.args, node.kwargs = "call_module", name, (x,), {}
                continue
        if (node.op == "call_function" and node.target in (F.relu, torch.relu, torch.relu_)
                or node.op == "call_method" and node.target in ("relu", "relu_")):
            from deepopt.opt.nn.activation.modules import DOReLU

            name = f"_deepopt_relu_{node.name}"
            if hasattr(graph_module, name):
                raise ValueError(f"Reserved ReLU call-site name exists: {name}")
            inplace = (node.target in ("relu_", torch.relu_) or
                       node.kwargs.get("inplace", node.args[1] if len(node.args) > 1 else False))
            adapter = DOReLU(plan=plan, inplace=inplace)
            adapter.train(graph_module.training)
            graph_module.add_submodule(name, adapter)
            node.op, node.target = "call_module", name
            continue
        if node.op == "call_function":
            from ..nn.pooling.spec import call_kind, FORCE_INDICES
            from ..nn.pooling.modules import DOFunctionalPool

            pool_kind = call_kind(node.target)
            if pool_kind is not None:
                name = f"_deepopt_pool_{node.name}"
                if hasattr(graph_module, name):
                    raise ValueError(f"Reserved pooling call-site name exists: {name}")
                adapter = DOFunctionalPool(plan=plan, kind=pool_kind)
                adapter.train(graph_module.training)
                graph_module.add_submodule(name, adapter)
                if node.target in FORCE_INDICES or getattr(node.target, "_deepopt_pool_indices", False):
                    # Bind before rewriting, preserving positional return_indices.
                    from ..nn.pooling.spec import SPECS

                    fields = {field for field, _ in SPECS[pool_kind].fields}
                    bound = SPECS[pool_kind].arguments(node.args[1:], {k: v for k, v in node.kwargs.items() if k in fields})
                    bound["return_indices"] = True
                    controls = {k: v for k, v in node.kwargs.items() if k not in fields and k != "input"}
                    node.args, node.kwargs = (node.args[0] if node.args else node.kwargs["input"],), {**bound, **controls}
                node.op, node.target = "call_module", name
                continue
            transpose_rank = {F.conv_transpose1d: 1, F.conv_transpose2d: 2, F.conv_transpose3d: 3}.get(node.target)
            if transpose_rank is not None:
                from ..nn.convolution_transpose.modules import DOFunctionalConvTranspose

                name = f"_deepopt_transpose_{node.name}"
                if hasattr(graph_module, name):
                    raise ValueError(f"Reserved transposed call-site name exists: {name}")
                parameter = next(graph_module.parameters(), None)
                adapter = DOFunctionalConvTranspose(plan, parameter.device if parameter is not None else "cpu", transpose_rank)
                adapter.train(graph_module.training)
                graph_module.add_submodule(name, adapter)
                node.op, node.target = "call_module", name
                continue
            kwargs = dict(node.kwargs)
            convolution_rank = {F.conv1d: 1, F.conv2d: 2, F.conv3d: 3}.get(node.target)
            if node.target in (F.batch_norm, F.group_norm, F.layer_norm):
                from deepopt.opt.nn.normalization.modules import DOFunctionalNormalization

                name = f"_deepopt_norm_{node.name}"
                if hasattr(graph_module, name):
                    raise ValueError(f"Reserved normalization call-site name exists: {name}")
                adapter = DOFunctionalNormalization(plan, {F.batch_norm: "batch", F.group_norm: "group", F.layer_norm: "layer"}[node.target])
                adapter.train(graph_module.training)
                graph_module.add_submodule(name, adapter)
                node.op, node.target = "call_module", name
                continue

            if "input" in kwargs and convolution_rank is None:
                kwargs["x"] = kwargs.pop("input")
            
            kwargs["plan"] = plan
            if plan["fp8_training"].get("scaling_recipe", "current") == "delayed":
                from deepopt.opt.nn.functional_linear import DOFunctionalLinear

                parameter = next(graph_module.parameters(), None)
                device = parameter.device if parameter is not None else "cpu"
                name = f"_deepopt_fp8_call_{node.name}"
                if hasattr(graph_module, name):
                    raise ValueError(f"Reserved FP8 call-site name already exists: {name}")
                if convolution_rank is None:
                    adapter = DOFunctionalLinear(plan, device)
                else:
                    from deepopt.opt.nn.convolution.modules import DOFunctionalConvolution

                    adapter = DOFunctionalConvolution(plan, device, convolution_rank)
                adapter.train(graph_module.training)
                graph_module.add_submodule(name, adapter)
                kwargs.pop("plan")
                node.op, node.target = "call_module", name
            else:
                node.target = do_linear if convolution_rank is None else (do_conv1d, do_conv2d, do_conv3d)[convolution_rank - 1]
            node.kwargs = kwargs

    graph_module.recompile()
    return graph_module
