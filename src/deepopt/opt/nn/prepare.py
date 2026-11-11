"""Prepare compression-only operation adapters without enabling FP8 compute."""

from dataclasses import asdict

from torch import nn

from .activation.modules import DOReLU
from .activation.elementwise_modules import CONVERSIONS
from .normalization.modules import DOBatchNorm2d, DOGroupNorm, DOLayerNorm
from .pooling.modules import CONVERSIONS as POOL_CONVERSIONS
from .convolution_transpose.modules import CONVERSIONS as TRANSPOSE_CONVERSIONS


def prepare_native_operations(model, compression, exclusions=()):
    from ..transform import TransformReport, excluded_module_ids

    excluded, names = excluded_module_ids(model, exclusions)
    replacements, visited, converted = {}, set(), []
    plan = {"fp8": False, "activation_compression": asdict(compression)}

    def convert(module, path):
        if id(module) in excluded:
            return module
        if id(module) in replacements:
            return replacements[id(module)]
        target = {nn.BatchNorm2d: DOBatchNorm2d, nn.GroupNorm: DOGroupNorm, nn.LayerNorm: DOLayerNorm, nn.ReLU: DOReLU,
                  **CONVERSIONS, **POOL_CONVERSIONS, **TRANSPOSE_CONVERSIONS}.get(type(module))
        if target is not None:
            replacement = target(module, plan=plan)
            replacements[id(module)] = replacement
            converted.append(path or "<root>")
            return replacement
        if id(module) not in visited:
            visited.add(id(module))
            for name, child in tuple(module._modules.items()):
                if child is not None:
                    module._modules[name] = convert(child, f"{path}.{name}" if path else name)
        return module

    prepared = convert(model, "")
    report = (TransformReport("modules", type(model).__qualname__, tuple(converted), (), excluded_modules=names)
              if converted else None)
    return prepared, report
