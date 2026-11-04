"""Warm up a loss coefficient once per optimizer update, including partial windows.

Run: python examples/scheduling/coefficients.py
Two epochs of five microbatches with accumulation=2 produce six updates.
Learning-rate scheduling is demonstrated separately in learning_rate.py.
"""

import json

import torch
from torch import nn

from deepopt import Trainer
from deepopt.logger import log_output
from deepopt.scheduling import LinearWarmup, ScheduleController


def main():
    torch.manual_seed(7)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    schedules = ScheduleController({"regularization": LinearWarmup(0., .01, duration=4)})

    def step(model, batch):
        inputs, targets = batch
        prediction = model(inputs)
        regularization = sum(parameter.square().sum() for parameter in model.parameters())
        return {"loss": (prediction - targets).square().mean() + schedules["regularization"] * regularization,
                "sample_count": len(targets)}

    trainer = Trainer(model, optimizer, step_fn=step, schedules=schedules, accumulation_steps=2)
    features = torch.randn(28, 3)
    targets = features.sum(dim=1, keepdim=True)
    training = [(features[start:start + 4], targets[start:start + 4]) for start in range(0, 20, 4)]
    validation = [(features[20:], targets[20:])]

    def report_update(_, event):
        log_output(json.dumps({"optimizer_step": event["optimizer_step"],
                               "applied_coefficients": event["scheduled_values"]}))

    def report_epoch(_, row):
        log_output(json.dumps({"epoch": row["epoch"],
                               "first_coefficients": row["scheduled_values_start"],
                               "last_coefficients": row["scheduled_values"],
                               "validation_loss": row["validation"]["loss"]}))

    trainer.fit(training, validation, epochs=2, progress_bar=False,
                callbacks=[report_epoch], step_callbacks=[report_update])


if __name__ == "__main__":
    main()
