"""Process normalized gradients before an optimizer update."""

from abc import abstractmethod
from collections.abc import Iterable

from torch import Tensor, nn

from .base import Strategy


class GradientStrategy(Strategy):
    @abstractmethod
    def process(self, parameters: Iterable[nn.Parameter]) -> Tensor | None:
        """Modify gradients in place; do not update parameters or clear grads.

        Optionally return a tensor diagnostic, such as the norm before clipping.
        Runtime observers inspect it without changing the transformation. A
        direct process() call transforms only; the caller owns any checks.
        """


class ClipGradNorm(GradientStrategy):
    def __init__(self, max_norm: float):
        if not 0 < max_norm < float("inf"):
            raise ValueError("max_norm must be positive and finite")
        self.max_norm = max_norm

    def get_config(self) -> dict:
        return {"max_norm": self.max_norm}

    def process(self, parameters) -> Tensor:
        return nn.utils.clip_grad_norm_(parameters, self.max_norm, error_if_nonfinite=False)


class ClipGradValue(GradientStrategy):
    def __init__(self, clip_value: float):
        if not 0 < clip_value < float("inf"):
            raise ValueError("clip_value must be positive and finite")
        self.clip_value = clip_value

    def get_config(self) -> dict:
        return {"clip_value": self.clip_value}

    def process(self, parameters) -> None:
        parameters = [parameter for parameter in parameters if parameter.grad is not None]
        if parameters:
            nn.utils.clip_grad_value_(parameters, self.clip_value)
