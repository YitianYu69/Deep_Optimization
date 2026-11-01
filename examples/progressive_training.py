"""One ordinary PyTorch recipe, progressively adding DeepOpt features.

python examples/progressive_training.py --stage basic
python examples/progressive_training.py --stage logging --output runs/progressive
python examples/progressive_training.py --stage checkpoint --output runs/progressive --epochs 2
python examples/progressive_training.py --stage resume --output runs/progressive --epochs 3
python examples/progressive_training.py --stage auto --output runs/progressive-auto
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanSquaredError

from deepopt import Checkpoint, Trainer
from deepopt.data import ResumableDataLoader
from deepopt.logger import init_logger, logging_session
from deepopt.memory import prepare_batch_size

logger = init_logger(__name__)


def sample_batch(size):
    # Describe this recipe's actual model/loss inputs. Auto planning uses fakes.
    return torch.empty(size, 3), torch.empty(size, 1)


def make_components(batch_size=8, *, seed=7, learning_rate=0.03,
                    with_metrics=True, resumable=False):
    """Fresh ordinary PyTorch components for this example, including notebooks.

    This is recipe setup, not a required DeepOpt factory or model/data schema.
    """
    torch.manual_seed(seed)
    inputs = torch.randn(33, 3)
    targets = inputs.sum(dim=1, keepdim=True)
    data = TensorDataset(inputs[:25], targets[:25])
    generator = torch.Generator().manual_seed(17)
    train_loader = (ResumableDataLoader(data, batch_size=batch_size, generator=generator)
                    if resumable else DataLoader(data, batch_size=batch_size, shuffle=True,
                                                 generator=generator))
    validation_loader = DataLoader(TensorDataset(inputs[25:], targets[25:]), batch_size=batch_size,
                                   generator=torch.Generator().manual_seed(19))
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    metrics = {"mse": MeanSquaredError()} if with_metrics else {}
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, metrics=metrics)
    return trainer, train_loader, validation_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("basic", "logging", "checkpoint", "resume", "auto"), default="basic")
    parser.add_argument("--output", type=Path, default=Path("runs/progressive"))
    parser.add_argument("--epochs", type=int, default=2, help="Total target epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--every-steps", type=int, default=0, help="Optional step saving in optimizer updates")
    parser.add_argument("--memory-budget-bytes", type=int, default=None)
    args = parser.parse_args()
    if args.every_steps < 0:
        parser.error("--every-steps must be nonnegative")
    settings = {"name": "progressive-regression", "seed": 7,
                "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": 0.03}}
    session = (nullcontext() if args.stage == "basic" else logging_session(
        settings, directory=None if args.stage == "auto" else args.output / "logs",
        failure_directory=args.output / "logs"))
    with session as logs:
        # Ordinary DataLoader works for epoch checkpoints. Opt into a stateful
        # loader only for step saving; no silent wrapping of the user's loader.
        trainer, train_loader, validation_loader = make_components(
            args.batch_size, seed=settings["seed"], learning_rate=settings["training"]["learning_rate"],
            with_metrics=args.stage != "basic", resumable=args.every_steps > 0)
        callbacks = []
        if args.stage in ("checkpoint", "resume", "auto"):
            checkpoint = Checkpoint(args.output / "checkpoints", every_steps=args.every_steps, config=settings)
            callbacks.append(checkpoint)
            if args.stage == "resume":
                checkpoint.restore(trainer, train_loader, validation_loader)
        if args.stage == "auto":
            plan = prepare_batch_size(trainer, train_loader, validation_loader, sample_batch_fn=sample_batch,
                                      memory_budget_bytes=args.memory_budget_bytes)
            settings["training"]["batch_size"] = plan["new_batch_size"]
            checkpoint.config = settings
            logs.bind(args.output / "logs", config=settings)
        trainer.fit(train_loader, validation_loader, epochs=args.epochs, callbacks=callbacks,
                    monitor="loss" if args.stage == "basic" else "mse", progress_bar=False)
        logger.info("Final evaluation: %s", trainer.evaluate(validation_loader))
        # 'loss' is a mean of scalar batch objectives; 'mse' is dataset-wide MSE.


if __name__ == "__main__":
    main()
