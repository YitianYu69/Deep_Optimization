"""Multiclass classification with caller-supplied class count and metrics."""

from torch import nn
from torch._subclasses.fake_tensor import FakeTensor
from torchmetrics.classification import MulticlassAccuracy

from .base import BatchOutput, LossFunction, MetricsFactory, Task
from .loss import LossReport, LossResult


class ClassificationTask(Task):
    def __init__(self, loss_fn: LossFunction | None = None, *, num_classes: int | None = None,
                 reduction_type: str | None = None, metrics_factory: MetricsFactory | None = None):
        if reduction_type not in (None, "mean", "sum"):
            raise ValueError("Built-in reduction_type must be mean or sum; reduce custom losses inside loss_fn")
        if loss_fn is not None and reduction_type is not None:
            raise ValueError("reduction_type configures the built-in loss; reduce custom losses inside loss_fn")
        if num_classes is not None and (type(num_classes) is not int or num_classes < 2):
            raise ValueError("num_classes must be an integer >= 2")
        self.num_classes = num_classes
        self.reduction_type = reduction_type or "mean"
        self.loss_fn = loss_fn
        self._default_loss = nn.CrossEntropyLoss(reduction="none")
        self.metrics_factory = metrics_factory

    def create_metrics(self):
        if self.metrics_factory is not None:
            return self.metrics_factory()
        if self.num_classes is None:
            return {}  # Loss-only is valid; do not assume a user's class count.
        return {"accuracy": MulticlassAccuracy(num_classes=self.num_classes, average="micro",
                                               sync_on_compute=True)}

    def update_metrics(self, metrics, output):
        logits, targets = output.metric_data
        for metric in metrics.values():
            metric.update(logits, targets)

    def compute(self, model: nn.Module, batch) -> BatchOutput:
        inputs, targets = batch
        if targets.ndim != 1 or targets.numel() == 0:
            raise ValueError("Expected nonempty class-index targets with shape [batch]")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Input and target batch sizes differ")
        logits = model(inputs)
        if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
            raise ValueError("Model must return logits with shape [batch, classes]")
        if self.num_classes is not None and logits.shape[1] != self.num_classes:
            raise ValueError("Logit class count differs from num_classes")
        if self.loss_fn is None:
            # Fake tensors describe shapes, not label values. Memory planning
            # assumes valid labels; real training still checks their range.
            if not isinstance(targets, FakeTensor) and ((targets < 0).any() or (targets >= logits.shape[1]).any()):
                raise ValueError("Built-in classification loss requires valid class indices; "
                                 "provide a custom loss_fn for masked targets")
            if self.reduction_type not in ("mean", "sum"):
                raise ValueError("Built-in reduction_type must be mean or sum")
            values = self._default_loss(logits, targets)
            loss = values.mean() if self.reduction_type == "mean" else values.sum()
            result = LossResult(loss, targets.shape[0] if self.reduction_type == "mean" else None,
                                LossReport(values.detach().double().sum(), targets.shape[0]))
        else:
            result = self.loss_fn(logits, targets)
        output = BatchOutput(result, targets.shape[0], metric_data=(logits.detach(), targets.detach()))
        output.validate(check_finite=False)
        return output
