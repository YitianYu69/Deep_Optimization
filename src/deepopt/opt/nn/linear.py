"""Precision-aware Linear with independent compute and saved-input policies."""

import torch
from torch import nn

from .functional import do_linear, _prepare_linear_plan
from ._cached_weights import CachedWeightModule





from ..precision.gradient import gradient_operation

class DOLinear(CachedWeightModule):
    def __init__(self, original: nn.Linear, *, plan: dict, native_bf16_only=False):
        super().__init__()

        if not isinstance(original, nn.Linear):
            raise ValueError("original must be an nn.Linear")

        self.plan = _prepare_linear_plan(plan)
        self.native_bf16_only = native_bf16_only
        self._native_uncompressed_plan = (_prepare_linear_plan({"fp8": False,
            "activation_compression": {"enabled": False},
            **({"kernel_policy": self.plan["kernel_policy"]} if "kernel_policy" in self.plan else {})})
            if native_bf16_only else None)
        if "fp8_training" in self.plan:
            self.configure_delayed_scaling(self.plan, original.weight.device)
        else:
            self.fp8_scaling = None

        self.in_features = original.in_features
        self.out_features = original.out_features

        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)

        self.train(original.training)

    @classmethod
    def from_linear(cls, original: nn.Linear, *, plan: dict) -> "DOLinear":
        return cls(original, plan=plan)

    @property
    def fp8_dtypes(self) -> dict[str, torch.dtype]:
        return dict(self.plan.get("fp8_training", {}).get("fp8_dtypes", {}))

    @property
    def block_shapes(self) -> dict[str, tuple[int, int] | None]:
        return dict(self.plan.get("block_shapes", {}))

    @gradient_operation("linear")
    def forward(self, x: torch.Tensor = None, *, input: torch.Tensor = None) -> torch.Tensor:
        if input is not None:
            if x is not None:
                raise TypeError("Linear received both x and input")
            x = input
        # Keep parameter ownership here; functional/autograd/backend layers own
        # computation and any future activation codecs or sparse kernels.
        plan = self.plan
        if self.native_bf16_only:
            dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
            if dtype != torch.bfloat16:
                plan = self._native_uncompressed_plan
        return do_linear(
            x, self.weight, self.bias,
            plan=plan, _weight_cache=self._fp8_weight_cache,
            _scaling_state=self.fp8_scaling, _state_training=self.training,
        )

    def extra_repr(self) -> str:
        backend = self.plan["fp8_training"]["backend"] if "fp8_training" in self.plan else "native"
        return (
            f"in_features={self.in_features}"
            f"out_features={self.out_features}"
            f"bias={self.bias is not None}"
            f"backend={backend!r}"
        )



