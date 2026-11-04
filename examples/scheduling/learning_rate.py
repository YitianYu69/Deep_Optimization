"""Apply a configured warmup/cosine LR policy at optimizer or epoch boundaries.

Run: python examples/scheduling/learning_rate.py --interval step
     python examples/scheduling/learning_rate.py --interval epoch
Small generated CPU data; no downloads or output files.
"""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt import Trainer
from deepopt.config import TrainingConfig
from deepopt.logger import log_output
from deepopt.scheduling.lr import build_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", choices=("step", "epoch"), default="step")
    args = parser.parse_args()
    torch.manual_seed(7)
    features = torch.randn(28, 3)
    targets = features.sum(dim=1, keepdim=True)
    training = DataLoader(TensorDataset(features[:20], targets[:20]), batch_size=4)
    validation = DataLoader(TensorDataset(features[20:], targets[20:]), batch_size=4)
    # Five microbatches form three updates per epoch, including the partial window.
    epochs, updates_per_epoch = 3, 3
    horizon = epochs * updates_per_epoch if args.interval == "step" else epochs
    settings = TrainingConfig(
        epochs=epochs, batch_size=4, learning_rate=.05,
        accumulation_steps=2, scheduler_interval=args.interval,
        lr_schedule={"type": "warmup_cosine", "warmup_steps": 2,
                     "total_steps": horizon, "start_factor": .2, "eta_min": .001},
    )
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    scheduler = build_scheduler(optimizer, settings)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss,
                      accumulation_steps=settings.accumulation_steps,
                      scheduler=scheduler, scheduler_interval=settings.scheduler_interval)

    def report_update(_, event):
        # Step callbacks run after scheduler.step(): this is the NEXT update's LR.
        log_output(json.dumps({"completed_update": event["optimizer_step"],
                               "next_learning_rate": optimizer.param_groups[0]["lr"]}))

    def report_epoch(_, row):
        log_output(json.dumps({"epoch": row["epoch"],
                               "learning_rate_at_start": row["learning_rate"],
                               "learning_rate_after_epoch": optimizer.param_groups[0]["lr"],
                               "validation_loss": row["validation"]["loss"]}))

    log_output(json.dumps({"interval": args.interval, "horizon": horizon,
                           "initial_learning_rate": optimizer.param_groups[0]["lr"]}))
    trainer.fit(training, validation, epochs=epochs, progress_bar=False,
                callbacks=[report_epoch],
                step_callbacks=[report_update] if args.interval == "step" else [])


if __name__ == "__main__":
    main()
