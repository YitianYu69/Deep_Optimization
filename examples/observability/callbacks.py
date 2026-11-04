"""Report updates, stop at an epoch boundary, then continue the same trainer.

Run: python examples/observability/callbacks.py
This CPU demonstration writes no files.
"""

import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt.logger import log_output
from deepopt import EveryNSteps, Trainer


def main():
    torch.manual_seed(7)
    inputs = torch.randn(24, 2)
    targets = inputs.sum(dim=1, keepdim=True)
    train = DataLoader(TensorDataset(inputs[:17], targets[:17]), batch_size=4)
    validation = DataLoader(TensorDataset(inputs[17:], targets[17:]), batch_size=4)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss,
                      scheduler=scheduler, scheduler_interval="step")

    def report_epoch(current, row):
        log_output(json.dumps({"epoch": row["epoch"], "updates": row["optimizer_steps"],
                          "validation_loss": row["validation"]["loss"]}))

    def stop_after_two(current, row):
        if row["epoch"] == 2:
            current.request_stop()  # This is a successful completed epoch.

    def report_update(current, event):
        log_output(json.dumps({"update": event["optimizer_step"],
                          "batches_this_epoch": event["batches_processed"]}))

    # Declare cadence before entering callbacks, avoiding callback-related
    # device synchronization on updates that do not produce a report.
    state = trainer.fit(train, validation, epochs=5, progress_bar=False,
                        callbacks=[report_epoch, stop_after_two],
                        step_callbacks=[EveryNSteps(report_update, 5)])
    log_output("Stopped after completed epoch:", state.completed_epochs)
    # epochs is a TOTAL target; this call trains one more epoch, not three.
    trainer.fit(train, validation, epochs=3, progress_bar=False, callbacks=[report_epoch])
    log_output("Completed epochs after continuing:", trainer.state.completed_epochs)
    log_output("Evaluation:", trainer.evaluate(validation))


if __name__ == "__main__":
    main()
