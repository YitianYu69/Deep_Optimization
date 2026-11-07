"""
FP8 linear with explicit forward and backward computations.
"""

import torch
from torch import nn

from .functional import do_linear, validate_fp8_plan





class DOLinear(nn.Module):
    def __init__(self, original: nn.Linear, *, plan: dict):
        super().__init__()

        if not isinstance(original, nn.Linear):
            raise ValueError("original must be an nn.Linear")
        
        self.plan = validate_fp8_plan(plan)

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
        return dict(self.plan["fp8_training"]["fp8_dtypes"])
    
    @property
    def block_shapes(self) -> dict[str, tuple[int, int] | None]:
        return dict(self.plan["block_shapes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep parameter ownership here; functional/autograd/backend layers own
        # computation and any future activation codecs or sparse kernels.
        return do_linear(
            x, self.weight, self.bias,
            plan=self.plan
        )

    def extra_repr(self) -> str:
        backend = self.plan["fp8_training"]["backend"]
        return (
            f"in_features={self.in_features}"
            f"out_features={self.out_features}"
            f"bias={self.bias is not None}"
            f"backend={backend!r}"
        )



