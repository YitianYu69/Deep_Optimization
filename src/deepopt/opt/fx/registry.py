"""
Explicit registration of supported module conversion.
"""

from dataclasses import dataclass
from typing import Protocol

from torch import nn

from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.nn.hf_conv1d import DOHFConv1D, get_hf_conv1d_type
from deepopt.opt.nn.attention import DOMultiheadAttention
from deepopt.opt.nn.convolution.modules import DOConv1d, DOConv2d, DOConv3d
from deepopt.opt.nn.normalization.modules import DOBatchNorm2d, DOGroupNorm, DOLayerNorm
from deepopt.opt.nn.activation.modules import DOReLU
from deepopt.opt.nn.activation.elementwise_modules import CONVERSIONS
from deepopt.opt.nn.pooling.modules import CONVERSIONS as POOL_CONVERSIONS


class ModuleConverter(Protocol):
    def __call__(self, module: nn.Module, *, plan: dict) -> nn.Module:
        """
        Construct a replacement using the supplied transformation plan.
        """
        ...



@dataclass(frozen=True)
class ModuleConversion:
    source_type: type[nn.Module]
    target_type: type[nn.Module]
    converter: ModuleConverter



class ConversionRegistry:
    """Map exact source module types to explicit conversion rules."""

    def __init__(self) -> None:
        self._rules: dict[type[nn.Module], ModuleConversion] = {}

    def register(self, source_type: type[nn.Module], target_type: type[nn.Module], converter: ModuleConverter) -> None:
        for name, module_type in (("source_type", source_type), ("target_type", target_type)):
            if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
                raise TypeError(
                    f"{name} must be an nn.Module subclass"
                )
        
        if source_type is target_type:
            raise ValueError("Source and target module must differ")
        if not callable(converter):
            raise TypeError("converter must be callable")
        if source_type in self._rules:
            raise ValueError(
                f"A converter is already registered for "
                f"{source_type.__module__}.{source_type.__qualname__}"
            )
        self._rules[source_type] = ModuleConversion(
            source_type=source_type, target_type=target_type,
            converter=converter
        )

    def lookup(self, module: nn.Module) -> ModuleConversion | None:
        return self._rules.get(type(module), None)

    def convert(self, module, *, plan, excluded_ids=()):
        replacement = self.lookup(module).converter(module, plan=plan)
        if isinstance(replacement, DOMultiheadAttention) and id(module.out_proj) in excluded_ids:
            replacement._fp8_out_projection = False
            if "out" in replacement._projection_scaling:
                del replacement._projection_scaling["out"]
        return replacement

    def is_converted(self, module: nn.Module) -> bool:
        module_type = type(module)
        return any(
            module_type is rule.target_type
            for rule in self._rules.values()
        )
    
    @property
    def leaf_types(self) -> tuple[type[nn.Module], ...]:
        result: list[type[nn.Module]] = []

        for rule in self._rules.values():
            for module_type in (rule.source_type, rule.target_type):
                if module_type not in result:
                    result.append(module_type)
        return tuple(result)






def _convert_linear(module: nn.Module, *, plan: dict) -> nn.Module:
    return DOLinear.from_linear(module, plan=plan)

def _convert_hf_conv1d(module: nn.Module, *, plan: dict) -> nn.Module:
    return DOHFConv1D.from_hf_conv1d(module, plan=plan)

def _convert_mha(module: nn.Module, *, plan: dict) -> nn.Module:
    return DOMultiheadAttention.from_mha(module, plan=plan)


def _convert_conv1d(module, *, plan):
    return DOConv1d.from_conv(module, plan=plan)


def _convert_conv2d(module, *, plan):
    return DOConv2d.from_conv(module, plan=plan)


def _convert_conv3d(module, *, plan):
    return DOConv3d.from_conv(module, plan=plan)


def _convert_batch_norm(module, *, plan):
    return DOBatchNorm2d(module, plan=plan)


def _convert_group_norm(module, *, plan):
    return DOGroupNorm(module, plan=plan)


def _convert_layer_norm(module, *, plan):
    return DOLayerNorm(module, plan=plan)


def _convert_relu(module, *, plan):
    return DOReLU(module, plan=plan)


def default_registry() -> ConversionRegistry:
    """
    Create an independent registry of implemented conversions.
    """

    registry = ConversionRegistry()
    registry.register(
        source_type=nn.Linear,
        target_type=DOLinear,
        converter=_convert_linear,
    )

    registry.register(
        source_type=nn.MultiheadAttention,
        target_type=DOMultiheadAttention,
        converter=_convert_mha,
    )

    for source, target, converter in ((nn.Conv1d, DOConv1d, _convert_conv1d),
                                      (nn.Conv2d, DOConv2d, _convert_conv2d),
                                      (nn.Conv3d, DOConv3d, _convert_conv3d)):
        registry.register(source, target, converter)
    from ..nn.convolution_transpose.modules import CONVERSIONS as TRANSPOSE_CONVERSIONS

    for source, target in TRANSPOSE_CONVERSIONS.items():
        registry.register(source, target, lambda module, *, plan, target=target: target(module, plan=plan))
    registry.register(nn.BatchNorm2d, DOBatchNorm2d, _convert_batch_norm)
    registry.register(nn.GroupNorm, DOGroupNorm, _convert_group_norm)
    registry.register(nn.LayerNorm, DOLayerNorm, _convert_layer_norm)
    registry.register(nn.ReLU, DOReLU, _convert_relu)
    for source, target in {**CONVERSIONS, **POOL_CONVERSIONS}.items():
        registry.register(source, target, lambda module, *, plan, target=target: target(module, plan=plan))

    hf_conv1d_type = get_hf_conv1d_type()
    if hf_conv1d_type is not None:
        registry.register(
            source_type=hf_conv1d_type,
            target_type=DOHFConv1D,
            converter=_convert_hf_conv1d,
        )
    return registry

