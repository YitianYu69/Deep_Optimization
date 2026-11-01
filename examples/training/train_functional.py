"""Train ordinary PyTorch models using a loss function or a custom step.

Run: python examples/training/train_functional.py --mode supervised
     python examples/training/train_functional.py --mode custom
"""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from deepopt.logger import log_output
from deepopt import Trainer


def custom_step(model, batch):
    """Combine any loss terms and choose metric inputs in ordinary Python."""
    predictions = model(batch["features"])
    alternate_predictions = model(batch["alternate_features"])
    targets = batch["targets"]
    data_loss = F.smooth_l1_loss(predictions, targets)
    consistency_loss = F.mse_loss(predictions, alternate_predictions)
    regularization = sum(parameter.square().mean() for parameter in model.parameters())
    return {
        "loss": data_loss + 0.1 * consistency_loss + 0.001 * regularization,
        "sample_count": targets.shape[0],  # Optional; enables sample throughput.
        "metric_data": {"preds": predictions, "target": targets},
    }


def make_data(count, generator, *, custom):
    inputs = torch.randn(count, 4, generator=generator)
    targets = inputs.sum(dim=1, keepdim=True)
    if not custom:
        return TensorDataset(inputs, targets)
    alternate = inputs + 0.05 * torch.randn(count, 4, generator=generator)
    return [{"features": features, "alternate_features": other, "targets": target}
            for features, other, target in zip(inputs, alternate, targets)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("supervised", "custom"), default="supervised")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    custom = args.mode == "custom"
    train_loader = DataLoader(make_data(64, generator, custom=custom), batch_size=8, shuffle=True)
    validation_loader = DataLoader(make_data(24, generator, custom=custom), batch_size=8)
    model = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    metrics = {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}

    if custom:
        trainer = Trainer(model, optimizer, step_fn=custom_step, metrics=metrics)
    else:
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, metrics=metrics)

    def report_epoch(_, row):
        log_output(json.dumps({"epoch": row["epoch"], "train": row["train"],
                          "validation": row["validation"]}))

    trainer.fit(train_loader, validation_loader, epochs=args.epochs, callbacks=[report_epoch])
    log_output(f"Completed {trainer.state.completed_epochs} epochs with {trainer.state.optimizer_steps} updates")


if __name__ == "__main__":
    main()
