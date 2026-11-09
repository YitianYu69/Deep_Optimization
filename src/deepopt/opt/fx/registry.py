"""
Explicit registration of supported module conversion.
"""

from dataclasses import dataclass
from typing import Protocol

from torch import nn

from ..do_nn.linear_module import DOLinear
from ..do_nn.hf_conv1d_module import DOHFConv1D, get_hf_conv1d_type
from ..do_nn.mha_module import DOMultiheadAttention


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

    hf_conv1d_type = get_hf_conv1d_type()
    if hf_conv1d_type is not None:
        registry.register(
            source_type=hf_conv1d_type,
            target_type=DOHFConv1D,
            converter=_convert_hf_conv1d,
        )
    return registry

