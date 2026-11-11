"""
FX tracing that preserves module as identifiable graph nodes
"""

import sys

from torch import fx, nn

from deepopt.opt.nn.linear import DOLinear


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
    def __init__(self, *, extra_leaf_types: tuple[type[nn.Module], ...] = (), excluded_ids=()) -> None:
        from deepopt.opt.nn.functional import (do_linear, do_conv1d, do_conv2d, do_conv3d, do_batch_norm,
                                              do_group_norm, do_layer_norm, do_relu, do_sigmoid, do_silu, do_gelu, do_tanh, do_relu6)

        from ..nn import functional as doF
        from ..nn.pooling.spec import SPECS

        transposed = tuple(getattr(doF, f"do_conv_transpose{rank}d") for rank in (1, 2, 3))
        pools = tuple(getattr(doF, "do_" + name) for name in SPECS)
        pools += tuple(getattr(doF, "do_" + name + "_with_indices") for name, spec in SPECS.items() if "max" in spec.family)
        super().__init__(autowrap_functions=(do_linear, do_conv1d, do_conv2d, do_conv3d, do_batch_norm,
                                            do_group_norm, do_layer_norm, do_relu, do_sigmoid, do_silu, do_gelu, do_tanh, do_relu6, *pools, *transposed))
        # Match both imported aliases and doF.activation(...) attribute calls;
        # keep the allowlist above rather than wrapping every frontend helper.
        from deepopt.opt.nn import functional

        self._autowrap_search.append(functional)
        self._excluded_ids = frozenset(excluded_ids)

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
        if id(module) in self._excluded_ids:
            return True
        if isinstance(module, self._projection_leaf_types):
            return True
        
        if _is_hf_conv1d(module):
            return True
        
        return super().is_leaf_module(module, module_qualified_name)
