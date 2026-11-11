"""Lifecycle hooks for adapter-owned, nonpersistent quantized weights."""

from torch import nn

from ..quantization.cache import WeightCache


class CachedWeightModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._fp8_weight_cache = WeightCache()

    def clear_fp8_cache(self):
        self._fp8_weight_cache.clear()

    def configure_delayed_scaling(self, plan, device):
        from deepopt.opt.scaling.delayed import DelayedScalingState

        training = plan["fp8_training"]
        self.fp8_scaling = (DelayedScalingState(training["delayed_scaling"], training["fp8_dtypes"], device=device)
                            if training["scaling_recipe"] == "delayed" else None)

    def _apply(self, fn, recurse=True):
        self.clear_fp8_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_fp8_cache()
        return super()._load_from_state_dict(*args, **kwargs)
