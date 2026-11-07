"""Batch results and the task interface; no persistent metric state lives here."""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Callable, Literal, Protocol

import torch
from torch import Tensor, nn
from torchmetrics import Metric
from ..observers import FiniteCheck, TensorEvent
from .._diagnostics import finite_checks_enabled
from .loss import LossReport, LossResult, validate_loss, validate_weight


LossFunction = Callable[[Tensor, Tensor], Tensor | LossResult]
MetricsFactory = Callable[[], dict[str, Metric]]


@dataclass(frozen=True)
class BatchOutput:
    """A scalar loss plus optional sample count and metric inputs.

    A plain scalar is used unchanged for one backward pass. Automatic
    accumulation and reporting average the supplied scalars across batches;
    sample_count counts examples only and never supplies an implicit weight.
    Leave it None when the step does not know the number of examples; the
    trainer still tracks batches and does not invent sample throughput.
    Use LossResult only for a different weighting/reporting policy. The optional
    legacy reduction argument is an explicit per-example shorthand: mean and
    sum describe existing scalars; none selects a mean of per-example values.
    """
    loss: Tensor | LossResult
    sample_count: int | None = None
    reduction: Literal["mean", "sum", "none"] | None = None
    metric_data: Any = None

    def __post_init__(self):
        object.__setattr__(self, "_result", self.loss if isinstance(self.loss, LossResult) else None)
        if self._result is not None:
            if self.reduction is not None:
                raise ValueError("LossResult cannot be combined with a legacy reduction declaration")
            object.__setattr__(self, "loss", self._result.value)

    @property
    def objective(self) -> LossResult:
        if self._result is not None:
            return self._result
        if self.reduction == "mean":
            return LossResult.from_mean(self.loss, weight=self.sample_count)
        if self.reduction == "sum":
            return LossResult.from_sum(self.loss, report_weight=self.sample_count)
        if self.reduction == "none":
            return LossResult(self.loss.mean(), self.sample_count,
                              LossReport(self.loss.detach().double().sum(), self.sample_count))
        return LossResult(self.loss, 1, LossReport(self.loss.detach().double(), 1))

    def _objective_parts(self) -> tuple[Tensor, float | None]:
        """Read the backward values without allocating detached reporting data.

        Do not cache tensors here: callers may modify their contents in place
        even though the BatchOutput container itself is frozen.
        """
        if type(self).objective is not BatchOutput.objective:
            # Respect an existing subclass's explicit objective. The ordinary
            # result can skip report creation without changing this extension.
            objective = self.objective
            return objective.value, objective.normalization_weight
        if self._result is not None:
            return self._result.value, self._result.normalization_weight
        if self.reduction == "none":
            return self.loss.mean(), self.sample_count
        if self.reduction == "mean":
            validate_weight(self.sample_count)
            return self.loss, float(self.sample_count)
        if self.reduction == "sum":
            validate_weight(self.sample_count, "reporting weight")
            return self.loss, None
        # A documented mean of batch objectives, not an inferred sample mean.
        return self.loss, 1

    @property
    def loss_report(self) -> LossReport | None:
        # Reporting must not create an autograd graph, including legacy inputs.
        with torch.no_grad():
            return self.objective.report

    def validate(self, *, check_finite: bool | None = None) -> None:
        """Check the declared shape and values without reducing or scaling loss."""
        if self.sample_count is not None and (type(self.sample_count) is not int or self.sample_count < 1):
            raise ValueError("sample_count must be a positive integer or None")
        if self.sample_count is None and self.reduction in ("mean", "sum", "none"):
            raise ValueError("A legacy per-example reduction requires an explicit sample_count")
        if self._result is not None:
            self._result.validate(check_finite=check_finite)
        else:
            validate_loss(self.loss, check_finite=check_finite)
        if self.reduction == "none":
            if self.loss.shape != (self.sample_count,):
                raise ValueError("Unreduced loss must contain one value per example with shape [sample_count]")
        elif self.reduction in ("mean", "sum"):
            if self.loss.ndim != 0:
                raise ValueError("Mean/sum loss must be a scalar; declare reduction='none' for per-example losses")
        elif self.reduction is not None:
            raise ValueError("Loss reduction must be mean, sum, or none")
        elif self._result is None and self.loss.ndim != 0:
            raise ValueError("loss_fn or a custom step must return a scalar loss; "
                             "reduce it with .mean() or .sum() in your computation")

    def loss_sum(self, *, check_finite: bool | None = None) -> Tensor:
        """Explicitly convert to a differentiable sum when a caller needs one."""
        if check_finite is None:
            check_finite = finite_checks_enabled("loss")
        self.validate(check_finite=check_finite)
        value, weight = self._objective_parts()
        total = value if weight is None else value * float(weight)
        if check_finite:
            FiniteCheck()(TensorEvent("loss", "accumulated loss", (total,)))
        return total


class Task(Protocol):
    def compute(self, model: nn.Module, batch: Any) -> Tensor | LossResult | BatchOutput | Mapping[str, Any]:
        """Return the same scalar/mapping forms as a step function, or BatchOutput."""
        ...

    def create_metrics(self) -> dict[str, Metric]:
        """Return NEW metric instances for each phase; all must inherit Metric."""
        ...

    def update_metrics(self, metrics: dict[str, Metric], output: BatchOutput) -> None:
        """Map task-specific output data to metric.update calls; do not compute."""
        ...
