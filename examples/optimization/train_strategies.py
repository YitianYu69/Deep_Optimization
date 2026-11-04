"""The same accumulation policies in a manual loop and in fit().

Run: python examples/optimization/train_strategies.py --api fit --weighting samples
     python examples/optimization/train_strategies.py --api manual --weighting microbatches
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepopt.logger import log_output
from deepopt.engine import OptimizationWindow, Trainer, TrainingRuntime
from deepopt.engine.strategies import ClipGradValue, ExplicitObjective, MicrobatchMean
from deepopt.module import TrainingModule
from deepopt.tasks.regression import RegressionTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", choices=("fit", "manual"), default="fit")
    parser.add_argument("--weighting", choices=("samples", "microbatches"), default="samples")
    args = parser.parse_args()
    torch.manual_seed(42)
    inputs = torch.randn(39, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    # Five microbatches (6, 6, 6, 6, 3) exercise unequal sizes and a final
    # partial window of unequal microbatches when accumulation_steps=3.
    train = DataLoader(TensorDataset(inputs[:27], targets[:27]), batch_size=6)
    validation = DataLoader(TensorDataset(inputs[27:], targets[27:]), batch_size=6)
    module = TrainingModule(nn.Linear(4, 1), task=RegressionTask())
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.05)
    runtime = TrainingRuntime(module, optimizer, gradient_strategy=ClipGradValue(1.0))
    loss_strategy = ExplicitObjective() if args.weighting == "samples" else MicrobatchMean()

    if args.api == "fit":
        trainer = Trainer(runtime, accumulation_steps=3, loss_strategy=loss_strategy)
        trainer.fit(train, validation, epochs=3, progress_bar=False)
        log_output("Validation loss:", trainer.evaluate(validation)["loss"])
    else:
        for _ in range(3):
            module.train()
            runtime.zero_grad()
            # Optional: reuse accumulation while owning the outer loop.
            # For custom backward/update algorithms, omit OptimizationWindow
            # and use runtime.backward(), runtime.step(), or the optimizer.
            window = OptimizationWindow(runtime, accumulation_steps=3, loss_strategy=loss_strategy)
            for batch in train:
                window.backward(runtime.compute(batch, module.training_step))
            window.flush()  # Apply a final partial accumulation window.
        module.eval()
        with torch.no_grad():
            loss = (module(inputs[27:]) - targets[27:]).square().mean()
        log_output("Validation loss:", loss.item())


if __name__ == "__main__":
    main()
