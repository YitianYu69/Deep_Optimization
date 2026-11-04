"""
FX tracing that preserves module as identifiable graph nodes
"""

import sys

from torch import fx, nn

from ..do_nn.linear_module import DOLinear


def _is_hf_conv1d(module: nn.Module) -> bool:
    """
    Recognize Transformers Conv1d without requiring Trnasformers.
    """

    modeling_utils = sys.modules.get("transformers.pytorch_utils")
    if modeling_utils is None:
        return False
    
    conv1d_type = getattr(modeling_utils, "Conv1D", None)
    return (
        isinstance(conv1d_type, type)
        and issubclass(conv1d_type, nn.Module)
        and isinstance(module, conv1d_type)
    )



class DOTracer(fx.Tracer):
    def __init__(self, *, extra_leaf_types: tuple[type[nn.Module], ...] = ()) -> None:
        super().__init__()

        if not isinstance(extra_leaf_types, tuple):
            raise TypeError("extra_leaf_types must be a tuple")

        for module_type in extra_leaf_types:
            if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
                raise TypeError("extra_leaf_types must contain nn.Module subclass")
        
        self._projection_leaf_types = (
            nn.Linear,
            nn.MultiheadAttention,
            DOLinear,
            *extra_leaf_types,
        )

    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(module, self._projection_leaf_types):
            return True
        
        if _is_hf_conv1d(module):
            return True
        
        return super().is_leaf_module(module, module_qualified_name)