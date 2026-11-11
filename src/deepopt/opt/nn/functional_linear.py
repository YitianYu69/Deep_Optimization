"""FX call-site state without a second registration of the original parameters."""

from ._cached_weights import CachedWeightModule
from .functional import _prepare_fp8_plan, do_linear


from ..precision.gradient import gradient_operation

class DOFunctionalLinear(CachedWeightModule):
    def __init__(self, plan, device):
        super().__init__()
        self.plan = _prepare_fp8_plan(plan)
        self.configure_delayed_scaling(self.plan, device)

    @gradient_operation("linear")
    def forward(self, x, weight, bias=None):
        return do_linear(x, weight, bias, plan=self.plan, _weight_cache=self._fp8_weight_cache,
                         _scaling_state=self.fp8_scaling, _state_training=self.training)
