"""Train an offline regression model with automatic rank-aware data loading.

Run: torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py
     torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py --device cuda
     torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py --output runs/ddp-example
     torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py --output runs/ddp-example --resume --epochs 3
"""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import traceback

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import Checkpoint, Trainer, initialize_distributed, shutdown_distributed
from deepopt.data import build_loaders
from deepopt.logger import log_output


def train(args, context):
    # Identical source data and seeds on every rank; build_loaders owns sharding.
    torch.manual_seed(7)
    inputs = torch.randn(80, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    train_loader, valid_loader, _ = build_loaders(
        TensorDataset(inputs[:64], targets[:64]), TensorDataset(inputs[64:], targets[64:]),
        batch_size=args.batch_size, seed=7, resumable=True,
    )
    model = nn.Linear(4, 1).to(context.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    trainer = Trainer(model, optimizer, device=context.device, loss_fn=F.mse_loss,
                      accumulation_steps=2, experiment_identity={"example": "ddp-regression-v1"})

    callbacks = []
    if args.output is not None:
        checkpoint = Checkpoint(args.output, every_steps=2)
        if args.resume:
            checkpoint.restore(trainer, train_loader, valid_loader)
        # Every rank calls checkpoint operations; the callback owns rank-zero I/O.
        callbacks.append(checkpoint)

    trainer.fit(train_loader, valid_loader, epochs=args.epochs, callbacks=callbacks, progress_bar=False)
    log_output(json.dumps({"distributed": context.as_dict(), "state": asdict(trainer.state),
                           "validation": trainer.evaluate(valid_loader)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--epochs", type=int, default=2, help="Total completed epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Microbatch size on each rank")
    parser.add_argument("--output", type=Path, help="Optional shared checkpoint directory; omitted means no files")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.output is None:
        parser.error("--resume requires --output")

    context = initialize_distributed(args.device)
    try:
        train(args, context)
    except BaseException as exc:
        # Keep the original traceback locations without retaining training objects.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        # The training scope must end before its native process group is released.
        gc.collect()
        shutdown_distributed()


if __name__ == "__main__":
    main()
