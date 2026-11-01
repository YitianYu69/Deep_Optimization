"""Custom TorchMetrics, nested scalar reports, and different metric signatures.

Run from the repository root: python examples/metrics/metric_routing.py
"""

import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric, Metric

from deepopt.logger import log_output
from deepopt import Trainer


class ErrorSummary(Metric):
    is_differentiable = False
    full_state_update = False

    def __init__(self):
        super().__init__(sync_on_compute=True)
        self.add_state("absolute_total", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("squared_total", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        errors = predictions - targets
        self.absolute_total += errors.abs().double().sum()
        self.squared_total += errors.square().double().sum()
        self.count += errors.numel()

    def compute(self):
        return {"absolute": {"mean": self.absolute_total / self.count},
                "squared": {"mean": self.squared_total / self.count}}


def step(model, batch):
    inputs, targets = batch
    predictions = model(inputs)
    return {"loss": F.mse_loss(predictions, targets),
            "predictions": predictions,
            "metric_data": (predictions, targets),
            "sample_count": targets.shape[0]}


def main():
    torch.manual_seed(17)
    inputs = torch.randn(40, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    training = DataLoader(TensorDataset(inputs[:32], targets[:32]), batch_size=8)
    validation = DataLoader(TensorDataset(inputs[32:], targets[32:]), batch_size=8)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    trainer = Trainer(
        model, optimizer, step_fn=step,
        metrics={"quality": ErrorSummary(), "prediction_mean": MeanMetric(sync_on_compute=True)},
        # quality uses the step's metric_data. MeanMetric receives one argument.
        metric_inputs={"prediction_mean": lambda output, batch: (output["predictions"],)},
    )
    trainer.fit(training, validation, epochs=3, monitor="quality/absolute/mean",
                progress_bar=False, callbacks=[lambda _, row: log_output(json.dumps(row))])
    log_output(json.dumps(trainer.evaluate(validation)))


if __name__ == "__main__":
    main()
