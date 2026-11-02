"""Offline coefficient/LR warmup example: python examples/scheduling/coefficients.py."""

import torch
from torch import nn

from deepopt import Trainer
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
        return {"loss": (prediction - targets).square().mean() + schedules["regularization"] * regularization}

    trainer = Trainer(model, optimizer, step_fn=step, schedules=schedules, accumulation_steps=2)
    batches = [(torch.randn(4, 3), torch.zeros(4, 1)) for _ in range(5)]
    trainer.fit(batches, batches, epochs=2, progress_bar=False,
                callbacks=[lambda trainer, row: print(row["epoch"], row["scheduled_values_start"], row["scheduled_values"])])


if __name__ == "__main__":
    main()
