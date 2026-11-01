"""Choose the backward objective independently of task computation and logging."""

from abc import abstractmethod
from dataclasses import dataclass

from torch import Tensor

from ...tasks.base import BatchOutput
from .base import Strategy


LOSS_SCALING_VERSION = "scalar_objective_v3"


@dataclass(frozen=True)
class BackwardLoss:
    """A scalar objective and its contribution to a window's gradient divisor.

    None means no division. A positive finite real number weights the objective
    according to the caller's strategy. Weights are added across a window,
    including its final partial window. Do not mix weighted/unweighted losses.
    """

    loss: Tensor
    normalization_weight: float | None = None
    distributed_reduction: str = "legacy"


class LossStrategy(Strategy):
    @abstractmethod
    def prepare(self, output: BatchOutput, *, accumulation_steps: int) -> BackwardLoss:
        """Choose the scalar used for backward; leave output.loss unchanged."""


class ExplicitObjective(LossStrategy):
    """Use only the caller's explicit objective and normalization weight."""

    # Existing subclasses may own __init__ without calling super().__init__.
    distributed_reduction = "legacy"

    def __init__(self, *, distributed_reduction="legacy"):
        if distributed_reduction not in ("legacy", "rank_mean", "global_mean", "global_sum"):
            raise ValueError("distributed_reduction must be legacy, rank_mean, global_mean, or global_sum")
        self.distributed_reduction = distributed_reduction

    def get_config(self) -> dict:
        return {"version": LOSS_SCALING_VERSION, **({"distributed_reduction": self.distributed_reduction}
                if self.distributed_reduction != "legacy" else {})}

    def prepare(self, output: BatchOutput, *, accumulation_steps: int) -> BackwardLoss:
        output.validate(check_finite=False)
        value, weight = output._objective_parts()
        if value.ndim != 0:
            raise ValueError("Automatic backward requires a scalar objective; reduce it in your loss_fn or step")
        if self.distributed_reduction in ("rank_mean", "global_sum"):
            return BackwardLoss(value, distributed_reduction=self.distributed_reduction)
        if self.distributed_reduction == "global_mean":
            if weight is None:
                raise ValueError("global_mean requires an explicit normalization weight")
            return BackwardLoss(value * float(weight), weight, "global_mean")
        if accumulation_steps > 1 and weight is not None:
            # Runtime observers check the prepared tensor before backward.
            # This strategy only chooses the objective and its divisor.
            if weight != 1:
                value = value * float(weight)
            return BackwardLoss(value, weight)
        return BackwardLoss(value)


class MicrobatchMean(LossStrategy):
    """Average caller-provided scalar objectives equally across microbatches.

    Never convert a sum to a mean or infer its denominator. Reporting remains
    independent of this explicitly chosen optimization strategy.
    """

    def get_config(self) -> dict:
        return {"version": 2}

    def prepare(self, output: BatchOutput, *, accumulation_steps: int) -> BackwardLoss:
        output.validate(check_finite=False)
        loss, _ = output._objective_parts()
        if loss.ndim != 0:
            raise ValueError("MicrobatchMean requires explicitly reduced scalar objectives")
        return BackwardLoss(loss, 1 if accumulation_steps > 1 else None)
