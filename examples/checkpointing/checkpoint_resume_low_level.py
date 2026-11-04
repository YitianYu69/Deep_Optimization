"""Advanced integration: manually own checkpoint history, generators, and saves.

For ordinary recipes use checkpoint_resume.py and deepopt.Checkpoint instead.
Run: python examples/checkpointing/checkpoint_resume_low_level.py --kind epoch
     python examples/checkpointing/checkpoint_resume_low_level.py --kind step

The initial batch size is 4. Resume defaults to 6 to show cursor-preserving
rebatching. Files live in a temporary directory unless --output is supplied.
"""

import argparse
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt.logger import init_logger, log_output
from deepopt import Trainer
from deepopt.data import ResumableDataLoader
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.runtime import seed_everything

logger = init_logger(__name__)


class CheckpointSaved(Exception):
    """Simulate interruption after a successful within-epoch save."""


def make_components(batch_size):
    seed_everything(17, deterministic=True)
    data_rng = torch.Generator().manual_seed(23)
    inputs = torch.randn(17, 3, generator=data_rng)
    targets = inputs.sum(dim=1, keepdim=True)
    dataset = TensorDataset(inputs, targets)
    validation_inputs = torch.randn(9, 3, generator=data_rng)
    validation_data = TensorDataset(validation_inputs, validation_inputs.sum(dim=1, keepdim=True))
    generators = {"train": torch.Generator().manual_seed(31),
                  "validation": torch.Generator().manual_seed(37)}
    train_loader = ResumableDataLoader(dataset, batch_size=batch_size, generator=generators["train"])
    validation_loader = DataLoader(validation_data, batch_size=batch_size, generator=generators["validation"])
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss,
                      scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9))
    return trainer, train_loader, validation_loader, generators


def demonstrate(directory, kind, resume_batch_size):
    trainer, train_loader, validation_loader, generators = make_components(4)
    history = []
    config = {"recipe": "checkpoint-regression-v1"}

    def record_epoch(current, row):
        history.append(row)
        save_checkpoint(directory / "last.pt", current, config,
                        generators=generators, history=history, train_loader=train_loader)

    def save_then_interrupt(current, event):
        save_checkpoint(directory / "last_step.pt", current, config,
                        generators=generators, history=history, kind="step", train_loader=train_loader)
        raise CheckpointSaved()

    try:
        trainer.fit(train_loader, validation_loader, epochs=1, callbacks=[record_epoch],
                    step_callbacks=[save_then_interrupt] if kind == "step" else [], progress_bar=False)
    except CheckpointSaved:
        pass  # A real interrupted process would exit here.

    path = directory / ("last_step.pt" if kind == "step" else "last.pt")
    payload = read_checkpoint(path)
    saved_updates = payload["state"]["optimizer_steps"]
    # Recreate every component; restore does not reconstruct Python recipes.
    trainer, train_loader, validation_loader, generators = make_components(resume_batch_size)
    history = restore_checkpoint(payload, trainer, config, generators=generators,
                                 train_loader=train_loader, validation_loader=validation_loader,
                                 batch_size_mode="manual")
    del payload  # The restored components now own their state.
    # The callback closes over the replacement loader, generator map, and history.
    trainer.fit(train_loader, validation_loader, epochs=2, callbacks=[record_epoch], progress_bar=False)
    log_output(json.dumps({"restored_kind": kind, "saved_optimizer_steps": saved_updates,
                      "resume_batch_size": resume_batch_size,
                      "completed_epochs": trainer.state.completed_epochs,
                      "optimizer_steps": trainer.state.optimizer_steps,
                      "samples_per_epoch": [row["train"]["samples"] for row in history],
                      "validation_loss": history[-1]["validation"]["loss"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("epoch", "step"), default="step")
    parser.add_argument("--resume-batch-size", type=int, default=6)
    parser.add_argument("--output", type=Path, help="New directory in which to retain checkpoints")
    args = parser.parse_args()
    if args.resume_batch_size < 1:
        parser.error("--resume-batch-size must be positive")
    if args.output is None:
        with tempfile.TemporaryDirectory(prefix="deepopt-checkpoint-") as directory:
            demonstrate(Path(directory), args.kind, args.resume_batch_size)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        demonstrate(args.output, args.kind, args.resume_batch_size)
        logger.info("Checkpoint artifacts retained in %s", args.output.resolve())


if __name__ == "__main__":
    main()
