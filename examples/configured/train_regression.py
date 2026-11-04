"""Replace the task, dataset, and model without modifying the training engine.

Run: python examples/configured/train_regression.py --output runs/regression
Resume: python examples/configured/train_regression.py --resume runs/regression/last.pt --epochs 12
"""

import argparse
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.config import config_from_dict
from deepopt.experiment import build_experiment
from deepopt.run import train
from deepopt.tasks.regression import RegressionTask


def regression_data(*, train_samples, validation_samples, input_dim, seed):
    """Generate this example's data; these choices do not belong to core config."""
    generator = torch.Generator().manual_seed(seed)

    def split(count):
        inputs = torch.randn(count, input_dim, generator=generator)
        targets = inputs.sum(dim=1, keepdim=True)
        return TensorDataset(inputs, targets)

    return split(train_samples), split(validation_samples)


def regression_experiment(config):
    # This recipe owns the schema; the core only saves application metadata.
    options = config.application["generated_regression"]
    return build_experiment(
        config,
        model_factory=partial(nn.Linear, options["input_dim"], 1),
        dataset_factory=partial(regression_data, **options, seed=config.seed),
        task_factory=RegressionTask,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path, help="New run directory")
    location.add_argument("--resume", type=Path, help="Continue last.pt or last_step.pt")
    parser.add_argument("--epochs", type=int, help="Total target epochs; new runs default to 8")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.resume is not None:
        # Keep saved settings unless the user explicitly overrides them.
        train(resume=args.resume, epochs=args.epochs, experiment_factory=regression_experiment,
              progress_bar=not args.no_progress)
        return
    config = config_from_dict({
        "name": "regression", "seed": 42,
        "training": {"epochs": args.epochs if args.epochs is not None else 8,
                     "batch_size": 16, "learning_rate": 0.05,
                     "monitor": "loss", "monitor_mode": "min"},
        "application": {"generated_regression": {
            "train_samples": 256, "validation_samples": 64, "input_dim": 4,
        }},
    })
    train(config, output=args.output, experiment_factory=regression_experiment,
          progress_bar=not args.no_progress)


if __name__ == "__main__":
    main()
