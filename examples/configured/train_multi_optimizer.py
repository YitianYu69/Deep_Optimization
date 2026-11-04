"""Choose optimizer/layer assignments outside the training core.

Run: python examples/configured/train_multi_optimizer.py --output runs/multi
Resume: python examples/configured/train_multi_optimizer.py --resume runs/multi/last.pt --epochs 8
"""

import argparse
from functools import partial
from pathlib import Path

import torch
from torch import nn

from deepopt.config import config_from_dict
from deepopt.data.synthetic import SyntheticDataSettings, build_datasets
from deepopt.experiment import build_experiment
from deepopt.optimizers import MultiOptimizer
from deepopt.run import train
from deepopt.tasks.classification import ClassificationTask


class Classifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, num_classes=4):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        return self.head(self.backbone(inputs))


def apply_optimizer(model, settings):
    # THIS function owns which optimizer updates which layers.
    return MultiOptimizer({
        "backbone": torch.optim.AdamW(model.backbone.parameters(), lr=settings.learning_rate,
                                       weight_decay=settings.weight_decay),
        "head": torch.optim.SGD(model.head.parameters(), lr=settings.learning_rate * 5,
                                 momentum=settings.momentum),
    })


def experiment(config):
    # Synthetic settings belong to this demonstration, not the training engine.
    settings = SyntheticDataSettings.from_dict(config.application.get("synthetic", {}))
    return build_experiment(
        config,
        model_factory=partial(Classifier, input_dim=settings.input_dim,
                              num_classes=settings.num_classes),
        dataset_factory=partial(build_datasets, settings, config.seed),
        optimizer_factory=apply_optimizer,
        task_factory=partial(ClassificationTask, num_classes=settings.num_classes),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path, help="New run directory")
    location.add_argument("--resume", type=Path, help="Continue last.pt or last_step.pt")
    parser.add_argument("--epochs", type=int, help="Total target epochs; new runs default to 5")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.resume is not None:
        train(resume=args.resume, epochs=args.epochs, experiment_factory=experiment,
              progress_bar=not args.no_progress)
        return
    config = config_from_dict({
        "name": "multiple-optimizers", "seed": 42,
        "training": {"epochs": args.epochs if args.epochs is not None else 5,
                     "batch_size": 32, "learning_rate": 0.01,
                     "optimizer": "custom", "lr_gamma": 0.9},
    })
    train(config, output=args.output, experiment_factory=experiment,
          progress_bar=not args.no_progress)


if __name__ == "__main__":
    main()
