"""An optional model wrapper with overridable batch computation."""

from typing import Any

from torch import nn

from .tasks.base import Task


class TrainingModule(nn.Module):
    """Use as an ordinary nn.Module, in a manual loop, or with Trainer.fit.

    Override training_step/validation_step to customize batch computation, or
    supply an existing Task. These hooks compute outputs; automatic optimization
    in Trainer owns backward and updates. Manual loops can use arbitrary logic.
    Hooks can return a scalar loss or a mapping containing loss, just like a
    step function. LossResult and BatchOutput remain optional for explicit
    weighting, reporting, and sample counts.
    Register persistent tensor state as parameters or buffers as usual.
    """

    def __init__(self, model: nn.Module, *, task: Task | None = None):
        super().__init__()
        self.model = model
        self.task = task

    def create_metrics(self):
        """Create fresh reporting metrics, independently for training/validation."""
        return self.task.create_metrics() if self.task is not None else {}

    def update_metrics(self, metrics, output):
        if self.task is not None:
            self.task.update_metrics(metrics, output)
        elif metrics:
            raise NotImplementedError("Override update_metrics for custom module metrics")

    def forward(self, *args, **kwargs):
        """Preserve the wrapped model's positional and keyword input interface."""
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any) -> Any:
        """Return a scalar loss, a loss mapping, LossResult, or BatchOutput.

        A mapping may include predictions and targets for metric updates;
        explicit metric_data takes precedence. Sample counts are optional.
        """
        if self.task is None:
            raise NotImplementedError("Override training_step or provide a Task")
        return self.task.compute(self, batch)

    def validation_step(self, batch: Any) -> Any:
        """Default to the same computation; override for different validation."""
        return self.training_step(batch)
