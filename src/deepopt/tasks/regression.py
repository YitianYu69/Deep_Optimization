"""Regression loss and metric updates, independent of metric lifecycle."""

from torch import nn
from torchmetrics import MeanSquaredError

from .base import BatchOutput, LossFunction, MetricsFactory, Task
from .loss import LossReport, LossResult


class RegressionTask(Task):
    def __init__(self, loss_fn: LossFunction | None = None, *, reduction_type: str | None = None,
                 metrics_factory: MetricsFactory | None = None):
        if reduction_type not in (None, "mean", "sum"):
            raise ValueError("Built-in reduction_type must be mean or sum; reduce custom losses inside loss_fn")
        if loss_fn is not None and reduction_type is not None:
            raise ValueError("reduction_type configures the built-in loss; reduce custom losses inside loss_fn")
        self.loss_fn = loss_fn
        self.reduction_type = reduction_type or "mean"
        self.metrics_factory = metrics_factory

    def create_metrics(self):
        return (self.metrics_factory() if self.metrics_factory is not None else
                {"mse": MeanSquaredError(sync_on_compute=True)})

    def update_metrics(self, metrics, output):
        predictions, targets = output.metric_data
        for metric in metrics.values():
            metric.update(predictions, targets)

    def compute(self, model: nn.Module, batch) -> BatchOutput:
        inputs, targets = batch
        predictions = model(inputs)
        if predictions.shape != targets.shape or targets.ndim < 1 or targets.shape[0] == 0:
            raise ValueError("Regression predictions and targets need matching nonempty batch shapes")
        if self.loss_fn is None:
            per_example = (predictions - targets).square().reshape(targets.shape[0], -1).mean(dim=1)
            if self.reduction_type not in ("mean", "sum"):
                raise ValueError("Built-in reduction_type must be mean or sum")
            loss = per_example.mean() if self.reduction_type == "mean" else per_example.sum()
            result = LossResult(loss, targets.shape[0] if self.reduction_type == "mean" else None,
                                LossReport(per_example.detach().double().sum(), targets.shape[0]))
        else:
            result = self.loss_fn(predictions, targets)
        output = BatchOutput(result, targets.shape[0], metric_data=(predictions.detach(), targets.detach()))
        output.validate(check_finite=False)
        return output
