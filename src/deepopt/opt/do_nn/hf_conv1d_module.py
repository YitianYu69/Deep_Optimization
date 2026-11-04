import sys

import torch
from torch import nn

from .functional import do_linear, validate_fp8_plan


def get_hf_conv1d_type() -> type[nn.Module] | None:
    module = sys.modules.get("transformers.pytorch_utils")
    conv1d_type = getattr(module, "Conv1D", None)

    if isinstance(conv1d_type, type) and issubclass(conv1d_type, nn.Module):
        return conv1d_type
    return None



class DOHFConv1D(nn.Module):
    def __init__(self, original: nn.Module, *, plan: dict) -> None:
        super().__init__()

        source_type = get_hf_conv1d_type()
        if source_type is None or type(original) is not source_type:
            raise TypeError(
                "original must be a standard "
                "transformers.pytorch_utils.Conv1D"
            )

        weight = original.weight
        bias = original.bias
        if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
            raise ValueError(
                "Conv1D weight must be a 2D Parameter "
                "with shape [in_features, out_features]" 
            )
        if not isinstance(bias, nn.Parameter):
            raise ValueError("Conv1D bias must be a Parameter")
        if tuple(bias.shape) != (weight.shape[1],):
            raise ValueError("Conv1D bias must have shape [out_features]")
        if original.nf != weight.shape[1]:
            raise ValueError("Conv1D nf must match the weight's output dimension")

        self.plan = validate_fp8_plan(plan)

        self.nx = weight.shape[0]
        self.nf = original.nf

        self.in_features = self.nx
        self.out_features = self.nf

        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)

        self.train(original.training)

    @classmethod
    def from_hf_conv1d(cls, original: nn.Module, *, plan: dict) -> "DOHFConv1D":
        return cls(original, plan=plan)

    @property
    def fp8_dtypes(self) -> dict[str, torch.dtype]:
        return dict(self.plan["fp8_training"]["fp8_dtypes"])
    
    @property
    def block_shapes(self) -> dict[str, tuple[int, int] | None]:
        return dict(self.plan["block_shapes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve HF's parameter layout; execution policies belong below this
        # adapter, including future activation compression and sparse kernels.
        return do_linear(
            x, self.weight.transpose(0, 1), self.bias,
            plan=self.plan
        )

    def extra_repr(self) -> str:
        backend = self.plan["fp8_training"]["backend"]
        return (
            f"nf={self.nf}, "
            f"nx={self.nx}, "
            f"backend={backend!r}"
        )
