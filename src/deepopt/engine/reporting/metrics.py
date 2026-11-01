"""Phase-local TorchMetrics state; tasks supply the metric update semantics."""

from collections.abc import Callable, Mapping

import torch
from torch import nn
from torchmetrics import Metric

from deepopt.observers import CheckObservers
from deepopt.tasks.base import BatchOutput
from deepopt.engine.reporting.metric_state import snapshot_metrics, restore_metrics


class MetricAccumulator(Metric):
    """Aggregate explicit loss reports and compute each task metric once per phase.

    Task metrics can implement non-additive statistics (F1, AUROC, etc.); their
    own TorchMetrics compute/sync behavior is preserved. No batch metric values
    are averaged. Every accumulator must receive fresh Metric instances.
    Standalone numerical checks default on. The engine injects its runtime's
    observe callback and disables the loss check already performed by its loop.
    """
    is_differentiable = False
    full_state_update = False

    def __init__(self, metrics: Mapping[str, Metric] | None = None, *,
                 update_metrics: Callable | None = None, sync_on_compute: bool = True,
                 checks=True, check_loss: bool = True, observe: Callable | None = None, **kwargs):
        super().__init__(sync_on_compute=sync_on_compute, **kwargs)
        if type(check_loss) is not bool:
            raise TypeError("check_loss must be a boolean")
        if observe is not None and not callable(observe):
            raise TypeError("observe must be callable")
        if observe is not None and checks is not True:
            raise ValueError("Configure checks or supply observe, not both")
        self._observe = CheckObservers(checks).observe if observe is None else observe
        self._check_loss = check_loss
        self._report_presence = None
        metrics = {} if metrics is None else metrics
        if not isinstance(metrics, Mapping):
            raise ValueError("create_metrics() must return a mapping of names to torchmetrics.Metric instances")
        if any(not isinstance(name, str) or not name for name in metrics):
            raise ValueError("Metric names must be nonempty strings")
        if {"loss", "samples", "batches", "seconds", "samples_per_second"}.intersection(metrics):
            raise ValueError("Task metric names collide with reserved engine metrics")
        if any(not isinstance(metric, Metric) for metric in metrics.values()):
            raise ValueError("Custom metrics must inherit torchmetrics.Metric")
        if len({id(metric) for metric in metrics.values()}) != len(metrics):
            raise ValueError("Each metric name must have its own Metric instance")
        if metrics and not callable(update_metrics):
            raise ValueError("Task metrics require an update_metrics function")
        self.metric_names = tuple(sorted(metrics))
        self.task_metrics = nn.ModuleList([metrics[name] for name in self.metric_names])
        self._update_metrics = update_metrics
        self.add_state("samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("loss_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("loss_weight", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("batches", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("counted_batches", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("loss_batches", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @property
    def metrics(self) -> dict[str, Metric]:
        return dict(zip(self.metric_names, self.task_metrics))

    def get_extra_state(self):
        return {"metric_names": self.metric_names, "report_presence": self._report_presence}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("metric_names") != self.metric_names:
            raise ValueError("Step resume requires the same metric names")
        presence = state.get("report_presence")
        if presence is not None and type(presence) is not bool:
            raise ValueError("Invalid saved reporting presence")
        self._report_presence = presence

    def reset(self):
        super().reset()
        self._report_presence = None
        for metric in getattr(self, "task_metrics", ()):
            metric.reset()

    def forward(self, output: BatchOutput):
        """Report this batch while retaining the complete phase state.

        TorchMetrics' base forward restores only the parent's registered
        states. Snapshot the children too. The trainer uses update/compute to
        avoid these copies (especially costly for metrics storing predictions).
        """
        self.update(output)
        saved = snapshot_metrics(self)
        try:
            self.reset()
            self.update(output)
            return self.compute()
        finally:
            self.reset()
            restore_metrics(self, saved, self.device)

    def update(self, output: BatchOutput) -> None:
        output.validate(check_finite=False)
        report = output.loss_report
        if output.loss.device != self.device:
            raise ValueError("Move MetricAccumulator to the batch device before updating")
        if self._check_loss:
            self._observe("loss", output.loss, name="loss")
        has_report = report is not None
        if self._report_presence is not None and self._report_presence != has_report:
            raise ValueError("Loss reporting must be supplied for every batch in a phase or omitted for all batches")
        if report is not None:
            report.validate(check_finite=False)
            if report.total.device != self.device:
                raise ValueError("Reporting total must be on the metric device")
            total = report.total.detach().to(dtype=self.loss_sum.dtype)
            new_total, new_weight = self.loss_sum + total, self.loss_weight + float(report.weight)
            self._observe("report", (new_total, new_weight), name="accumulated reporting loss or weight")
        # Metric updates must not retain a computation graph. Differentiable
        # objectives belong in output.loss, not in reporting metrics.
        with torch.no_grad():
            if self.task_metrics:
                self._update_metrics(self.metrics, output)
            if output.sample_count is not None:
                self.samples += output.sample_count
                self.counted_batches += 1
            self.batches += 1
            if report is not None:
                self.loss_sum.copy_(new_total)
                self.loss_weight.copy_(new_weight)
                self.loss_batches += 1
            self._report_presence = has_report

    def compute(self) -> dict[str, torch.Tensor]:
        if self.batches == 0:
            raise ValueError("Data loader is empty")
        if self.loss_batches != 0 and self.loss_batches != self.batches:
            raise ValueError("Loss reporting must be consistent across all batches and ranks")
        loss = {"loss": self.loss_sum / self.loss_weight} if self.loss_batches > 0 else {}
        counts = ({"samples": self.samples.clone()} if self.counted_batches == self.batches else
                  {"batches": self.batches.clone()})
        return {**loss, **counts,
                **{name: metric.compute() for name, metric in self.metrics.items()}}
