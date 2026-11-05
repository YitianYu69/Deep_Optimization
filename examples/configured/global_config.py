"""Global defaults, local overrides, and stable trainer/loader snapshots.

Run: python examples/configured/global_config.py
Generated CPU data only; no downloads or output files.
"""

import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import Trainer, get_config, load_config, reset_config
from deepopt.data import build_loaders
from deepopt.logger import log_output


def main():
    torch.manual_seed(7)
    inputs = torch.randn(24, 3)
    targets = inputs.sum(dim=1, keepdim=True)
    training = TensorDataset(inputs[:16], targets[:16])
    validation = TensorDataset(inputs[16:], targets[16:])
    defaults = {
        "name": "first-defaults", "seed": 7,
        "training": {"epochs": 2, "batch_size": 4, "learning_rate": .03,
                     "optimizer": "sgd", "accumulation_steps": 4},
        "runtime": {"device": "cpu", "checks": True},
    }
    try:
        load_config(defaults)
        train_a, valid_a, _ = build_loaders(training, validation)
        # Explicit accumulation=1 wins over local 2 and global 4. Local epochs=1
        # wins over global 2. The optimizer is built from the inherited policy.
        first = Trainer(nn.Linear(3, 1), loss_fn=F.mse_loss, accumulation_steps=1,
                        checks=False, config={"training": {"epochs": 1, "accumulation_steps": 2}})

        load_config({**defaults, "name": "later-defaults",
                     "training": {**defaults["training"], "batch_size": 8,
                                  "learning_rate": .005, "accumulation_steps": 2}})
        # The existing trainer and loader keep their original snapshots.
        first.fit(train_a, valid_a, monitor=None, progress_bar=False)
        train_b, valid_b, _ = build_loaders(training, validation)
        second = Trainer(nn.Linear(3, 1), loss_fn=F.mse_loss)
        second.fit(train_b, valid_b, progress_bar=False)
        result = {
            "first": {"epochs": first.state.completed_epochs, "updates": first.state.optimizer_steps,
                      "batch_size": train_a.batch_size, "accumulation": first.accumulation_steps,
                      "learning_rate": first.optimizer.param_groups[0]["lr"],
                      "checks": first.config["runtime"]["checks"], "monitor": first.state.monitor},
            "second": {"epochs": second.state.completed_epochs, "updates": second.state.optimizer_steps,
                       "batch_size": train_b.batch_size, "accumulation": second.accumulation_steps,
                       "learning_rate": second.optimizer.param_groups[0]["lr"]},
            "first_snapshot": first.config["name"], "current_defaults": get_config().name,
        }
    finally:
        reset_config()
    log_output(json.dumps({**result, "global_reset": get_config() is None}))


if __name__ == "__main__":
    main()
