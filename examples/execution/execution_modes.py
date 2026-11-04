"""Schedule diagnostic modes by optimizer update, including a partial window.

Run: python examples/execution/execution_modes.py
     python examples/execution/execution_modes.py --device cuda
"""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Trainer
from deepopt.logger import log_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(19)
    model = nn.Linear(4, 1).to(args.device)
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), device=args.device,
        loss_fn=F.mse_loss, accumulation_steps=2, execution_mode="max_speed",
        mode_schedule={2: "analyze", 4: "balanced"},
    )

    def report_mode(trainer, event):
        log_output(json.dumps({"update": event["optimizer_step"],
                               "mode": trainer.execution_mode}))

    batches = [(torch.randn(8, 4), torch.zeros(8, 1)) for _ in range(7)]
    metrics = trainer.train_epoch(batches, step_callbacks=[report_mode])
    # Seven microbatches produce four updates: max_speed, analyze, analyze, balanced.
    # Switching explicitly is safe now that the last partial window has completed.
    trainer.set_execution_mode("max_speed")
    log_output(json.dumps({"updates": trainer.state.optimizer_steps,
                           "mode_after_switch": trainer.execution_mode, "loss": metrics["loss"]}))


if __name__ == "__main__":
    main()
