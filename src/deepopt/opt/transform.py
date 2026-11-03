"""
Fx based conversion of registered DO moudle
"""

from collection.abc import Callable
from copy import deepcopy

from torch import fx, nn

from .do_nn.linear_module import DOLinear



class _DOTracer(fx.Tracer):
    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(module, (nn.Linear, DOLinear)):
            return True
        
        return super().is_leaf_module(module, module_qualified_name)

