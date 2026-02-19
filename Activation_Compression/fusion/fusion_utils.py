import torch
import torch.nn as nn
import torch.fx as fx

from .fused_layers import DOBatchNormReLU2d
from modules.normalization.normalization_layers import DOBatchNorm2d
from modules.activations.act_layers import DOReLU_Variance

import copy
from typing import Tuple, Iterable, Type, Any, Dict

# Borrwoed from PT2's FX experimental REPO


def _parent_name(target: str) -> Tuple[str, str]:
    *parent, child = target.rsplit('.', 1)
    return parent[0] if parent else '', child


def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]) -> bool:
    if len(node.args) == 0:
        return False
    
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True
    

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: nn.Module, new_child_name: str):
    assert(isinstance(node.target, str))
    old_qual = node.target
    # parent_name, child_name = _parent_name(old_qual)
    parent_name, _, old_child_name = old_qual.rpartition(".")
    child_name = new_child_name or old_child_name
    new_qual = f"{parent_name}.{child_name}" if parent_name else child_name
    setattr(modules[parent_name], child_name, new_module)

        # ---- delete old module ----
    if hasattr(modules[parent_name], old_child_name):
        delattr(modules[parent_name], old_child_name)

    modules[new_qual] = new_module
    modules.pop(old_qual, None)
    node.target = new_qual


def fuse_bn_act(model: nn.Module, rename_fmt="{}_bnact") -> nn.Module:
    
    patterns = [
        (DOBatchNorm2d, DOReLU_Variance)
    ]

    fx_model = model
    modules = dict(fx_model.named_modules())
    graph = fx_model.graph

    for pattern in patterns:
        for node in graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:
                    continue
                bn = modules[node.args[0].target]
                relu = modules[node.target]

                old_qual = node.args[0].target
                parent_name, _, old_child = old_qual.rpartition(".")

                new_name = rename_fmt.format(old_child)
                target_name = rename_fmt.format(old_qual)
                bn.meta[f'{target_name}'] = bn.meta[f'{bn.target_name}']
                bn.meta.pop(f'{old_qual}', None)
                fused_bn_act_module = DOBatchNormReLU2d(
                    num_features=bn.num_features, eps=bn.eps, momentum=bn.momentum,
                    relu=True, relu6=False,
                    affine=bn.affine, track_running_stats=bn.track_running_stats,
                    target_name=target_name, meta=bn.meta
                )

                with torch.no_grad():
                    if bn.weight is not None:
                        fused_bn_act_module.weight.copy_(bn.weight)
                    if bn.bias is not None:
                        fused_bn_act_module.bias.copy_(bn.bias)
                    if bn.running_mean is not None and bn.running_var is not None:
                        fused_bn_act_module.running_mean.copy_(bn.running_mean)
                        fused_bn_act_module.running_var.copy_(bn.running_var)

                replace_node_module(node.args[0], modules, fused_bn_act_module, new_name)
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
                # fx_model.add_module(new_name, fused_bn_act_module)
                # node.args[0].target = new_name
                # node.replace_all_uses_with(node.args[0])
                # graph.erase_node(node)

    graph.lint()
    fx_model.recompile()
    return fx_model


