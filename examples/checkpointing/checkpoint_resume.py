"""Checkpoint, resume, and evaluate the best model using the ready-made callback.

Run: python examples/checkpointing/checkpoint_resume.py --kind epoch
     python examples/checkpointing/checkpoint_resume.py --kind step
     python examples/checkpointing/checkpoint_resume.py --kind step --resume-batch-size 4

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
from torchmetrics import MeanSquaredError

from deepopt.logger import init_logger, log_output
from deepopt import Checkpoint, Trainer
from deepopt.data import ResumableDataLoader
from deepopt.runtime import seed_everything

logger = init_logger(__name__)


class CheckpointSaved(Exception):
    """Simulate interruption after a successful within-epoch save."""


def make_components(batch_size, kind):
    seed_everything(17, deterministic=True)
    data_rng = torch.Generator().manual_seed(23)
    inputs = torch.randn(17, 3, generator=data_rng)
    targets = inputs.sum(dim=1, keepdim=True)
    dataset = TensorDataset(inputs, targets)
    validation_inputs = torch.randn(9, 3, generator=data_rng)
    validation_data = TensorDataset(validation_inputs, validation_inputs.sum(dim=1, keepdim=True))
    generator = torch.Generator().manual_seed(31)
    train_loader = (ResumableDataLoader(dataset, batch_size=batch_size, generator=generator)
                    if kind == "step" else DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                      generator=generator))
    validation_loader = DataLoader(validation_data, batch_size=batch_size,
                                   generator=torch.Generator().manual_seed(37))
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss,
                      metrics={"mse": MeanSquaredError()},
                      scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9))
    return trainer, train_loader, validation_loader


def demonstrate(directory, kind, resume_batch_size):
    trainer, train_loader, validation_loader = make_components(4, kind)
    checkpoint = Checkpoint(directory, every_steps=1 if kind == "step" else 0)

    def interrupt_after_save(current, event):
        # Checkpoint's optimizer hook runs before explicit step_callbacks.
        raise CheckpointSaved()

    try:
        trainer.fit(train_loader, validation_loader, epochs=1, callbacks=[checkpoint],
                    step_callbacks=[interrupt_after_save] if kind == "step" else [],
                    monitor="mse", progress_bar=False)
    except CheckpointSaved:
        pass  # A real interrupted process would exit here.

    # Recreate every component; restore does not reconstruct Python recipes.
    trainer, train_loader, validation_loader = make_components(resume_batch_size, kind)
    checkpoint = Checkpoint(directory, every_steps=1 if kind == "step" else 0)
    checkpoint.restore(trainer, train_loader, validation_loader)
    saved_updates = trainer.state.optimizer_steps
    trainer.fit(train_loader, validation_loader, epochs=2, callbacks=[checkpoint],
                monitor="mse", progress_bar=False)

    # Evaluate best weights separately, keeping training weights/optimizer intact.
    evaluator, _, evaluation_loader = make_components(resume_batch_size, kind)
    payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    evaluator.model.load_state_dict(payload["model"])
    best_epoch = payload["state"]["best_epoch"]
    del payload
    best_results = evaluator.evaluate(evaluation_loader)
    log_output(json.dumps({"restored_kind": kind, "saved_optimizer_steps": saved_updates,
                      "resume_batch_size": resume_batch_size,
                      "completed_epochs": trainer.state.completed_epochs,
                      "optimizer_steps": trainer.state.optimizer_steps,
                      "samples_per_epoch": [row["train"]["samples"] for row in checkpoint.history],
                      "validation_loss": checkpoint.history[-1]["validation"]["loss"],
                      "best_epoch": best_epoch, "best_validation_loss": best_results["loss"],
                      "best_validation_mse": best_results["mse"]}))


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
