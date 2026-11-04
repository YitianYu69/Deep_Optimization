"""Console + configuration-named file logs without custom Trainer wrappers.

Run: python examples/observability/logging_recipe.py --log-dir logs
This offline CPU example retains append-only logs. Identical settings append
to the same file; use a different --log-dir for a separate experiment.
"""

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt import Trainer
from deepopt.logger import init_logger, logging_session

logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-level", default=None, help="Console level; default DEEPOPT_LOGGING_LEVEL or INFO")
    args = parser.parse_args()
    config = {"name": "logged-regression", "seed": 7,
              "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": 0.03},
              "runtime": {"device": "cpu", "precision": "fp32"}}
    with logging_session(config, args.log_dir, level=args.log_level) as session:
        torch.manual_seed(config["seed"])
        inputs = torch.randn(24, 3)
        targets = inputs.sum(dim=1, keepdim=True)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=config["training"]["batch_size"])
        model = nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
        logger.info("Recipe ready; dataset has %s samples", len(loader.dataset))
        trainer.fit(loader, epochs=config["training"]["epochs"], progress_bar=False)
        logger.info("Evaluation: %s", trainer.evaluate(loader))
        logger.debug("A debug message is saved even when the console uses INFO")
        logger.info("Saved logs: %s", session.path)


if __name__ == "__main__":
    main()
