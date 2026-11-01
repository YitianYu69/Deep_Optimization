"""Show batch-average vs explicitly element-weighted loss on uneven batches."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanSquaredError

from deepopt import LossResult, Trainer
from deepopt.logger import init_logger

logger = init_logger(__name__)


def element_mean_loss(predictions, targets):
    # This recipe explicitly knows F.mse_loss averages all target elements.
    return LossResult.from_mean(F.mse_loss(predictions, targets), weight=targets.numel())


def main():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    loader = DataLoader(TensorDataset(torch.zeros(3, 1), torch.tensor([[0.], [0.], [10.]])), batch_size=2)
    for name, loss_fn in (("batch average", F.mse_loss), ("element weighted", element_mean_loss)):
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), loss_fn=loss_fn,
                          metrics={"mse": MeanSquaredError()})
        logger.info("%s: %s", name, trainer.evaluate(loader))
    # Batch average: loss=50, mse≈33.3333. Element weighted: both≈33.3333.


if __name__ == "__main__":
    main()
