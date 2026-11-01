"""Plan or choose batch sizes on both fresh starts and checkpoint resumes.

Run: python examples/memory/batch_size_modes.py
Keep artifacts: python examples/memory/batch_size_modes.py --output runs/batch-modes

This offline CPU demonstration uses an explicit tensor-memory budget, derived
from the same recipe at size four. It does not measure available CPU RAM.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt.logger import log_output
from deepopt.config import config_from_dict
from deepopt.experiment import build_experiment
from deepopt.memory import estimate_training_memory
from deepopt.run import train


def datasets():
    generator = torch.Generator().manual_seed(23)
    features = torch.randn(36, 3, generator=generator)
    targets = features.sum(dim=1, keepdim=True)
    return (TensorDataset(features[:24], targets[:24]),
            TensorDataset(features[24:], targets[24:]))


def recipe(config):
    return build_experiment(config, model_factory=lambda: nn.Linear(3, 1),
                            dataset_factory=datasets, loss_fn=F.mse_loss)


def sample_batch(size):
    # The planner calls this in fake mode. These constructors describe the
    # complete batch's shapes without allocating corresponding data payloads.
    return torch.empty(size, 3), torch.empty(size, 1)


def demonstrate(directory):
    config = config_from_dict({
        "name": "batch-size-modes", "seed": 7,
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.02,
                     "checkpoint_every_steps": 0},
    })
    reference, _, _, _ = recipe(config)
    budget = estimate_training_memory(reference, sample_batch(4)).peak_bytes_by_device["cpu"]
    del reference

    automatic = train(config, output=directory / "auto", experiment_factory=recipe,
                      sample_batch_fn=sample_batch, memory_budget_bytes=budget,
                      progress_bar=False)  # batch_size_mode="auto" is the default.
    train(resume=automatic / "last.pt", epochs=2, batch_size=8,
          experiment_factory=recipe, sample_batch_fn=sample_batch,
          memory_budget_bytes=budget, progress_bar=False)

    manual = train(config, output=directory / "manual", experiment_factory=recipe,
                   batch_size_mode="manual", batch_size=6, progress_bar=False)
    train(resume=manual / "last.pt", epochs=2, experiment_factory=recipe,
          batch_size_mode="manual", batch_size=4, progress_bar=False)

    for run in (automatic, manual):
        decisions = [json.loads(line) for line in (run / "batch_size.jsonl").read_text().splitlines()]
        log_output(json.dumps({"run": run.name, "decisions": [
            {key: row[key] for key in ("event", "mode", "requested_batch_size", "new_batch_size", "status")}
            for row in decisions
        ]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New parent directory in which to keep both runs")
    args = parser.parse_args()
    if args.output is not None:
        directory = args.output.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        demonstrate(directory)
        log_output(f"Artifacts: {directory}")
    else:
        with TemporaryDirectory(prefix="deepopt-batch-modes-") as directory:
            demonstrate(Path(directory))


if __name__ == "__main__":
    main()
