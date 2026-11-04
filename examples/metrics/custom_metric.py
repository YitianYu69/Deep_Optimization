"""A custom TorchMetrics metric with registered, synchronizable states.

Run: python examples/metrics/custom_metric.py
"""

import torch
from torchmetrics import Metric

from deepopt.logger import log_output


class AbsoluteError(Metric):
    is_differentiable = False
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # sync_on_compute=True by default.
        self.add_state("total", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        errors = (predictions.detach() - targets).abs().reshape(targets.shape[0], -1).mean(1)
        self.total += errors.double().sum()
        self.count += errors.numel()

    def compute(self):
        if self.count == 0:
            raise ValueError("Cannot compute error without samples")
        return self.total / self.count


def create_metrics():
    return {"mae": AbsoluteError(sync_on_compute=True)}


def main():
    metric = AbsoluteError(sync_on_compute=True)
    metric.update(torch.tensor([[1.0], [4.0]]), torch.tensor([[2.0], [2.0]]))
    metric.update(torch.tensor([[5.0]]), torch.tensor([[2.0]]))
    log_output("Mean absolute error:", metric.compute().item())  # (1 + 2 + 3) / 3 = 2.
    metric.reset()
    log_output("Samples after reset:", metric.count.item())
    # In a recipe: Trainer(model, optimizer, loss_fn=..., metrics=create_metrics()).
    # The trainer gives training/validation independent metric instances and
    # owns update/compute/reset. See examples/metrics/metric_routing.py for a full run.


if __name__ == "__main__":
    main()
