"""Three ways to use the same module, without experiment configuration.

Run: python examples/training/train_module.py --mode fit
     python examples/training/train_module.py --mode manual
     python examples/training/train_module.py --mode wrapper
"""

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt.logger import log_output
from deepopt.engine import Trainer, TrainingRuntime
from deepopt.module import TrainingModule
from deepopt.tasks.base import BatchOutput


class RegressionModule(TrainingModule):
    def training_step(self, batch):
        inputs, targets = batch
        prediction = self(inputs)
        # Multiple forward passes or additional loss terms can go here.
        loss = F.mse_loss(prediction, targets)
        return BatchOutput(loss, targets.shape[0])

    def validation_step(self, batch):
        inputs, targets = batch
        loss = F.l1_loss(self(inputs), targets)
        return BatchOutput(loss, targets.shape[0])


def manual_train(module, optimizer, loader):
    runtime = TrainingRuntime(module, optimizer)

    def compute_loss(batch):
        inputs, targets = batch
        prediction = module(inputs)
        # The manual API accepts a scalar; no BatchOutput or task is required.
        loss = F.mse_loss(prediction, targets)
        penalty = 0.001 * sum(p.square().sum() for p in module.parameters())
        return loss + penalty

    module.train()
    for _ in range(3):
        for batch in loader:
            runtime.zero_grad()
            runtime.backward(runtime.compute(batch, compute_loss))
            runtime.step()  # Already a mean loss: no extra sample division.
            # Arbitrary update algorithms can call optimizer.step() directly.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit", "manual", "wrapper"), default="fit")
    args = parser.parse_args()
    torch.manual_seed(42)
    module = RegressionModule(nn.Linear(4, 1))
    inputs = torch.randn(32, 4)
    targets = inputs.sum(dim=1, keepdim=True)

    if args.mode == "wrapper":
        module.eval()
        with torch.no_grad():
            log_output("Prediction shape:", tuple(module(inputs).shape))
        return

    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.05)
    if args.mode == "fit":
        trainer = Trainer(module, optimizer)
        trainer.fit(loader, loader, epochs=3)  # Demo only; use a separate validation split in real runs.
        log_output("Completed epochs:", trainer.state.completed_epochs)
    else:
        manual_train(module, optimizer, loader)
        log_output("Completed manual training")


if __name__ == "__main__":
    main()
