"""A regular PyTorch model and loaders, trained with one loss function.

Run: python examples/training/regression.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanAbsoluteError

from deepopt import Trainer


def main():
    # Your data, architecture, and optimizer.
    torch.manual_seed(42)
    features = torch.randn(80, 4)
    targets = features.sum(dim=1, keepdim=True)
    train = DataLoader(TensorDataset(features[:64], targets[:64]), batch_size=8,
                       shuffle=True, generator=torch.Generator().manual_seed(43))
    validation = DataLoader(TensorDataset(features[64:], targets[64:]), batch_size=8)
    model = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    # DeepOpt owns the training loop; no task, config, or factory is needed.
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, metrics={"mae": MeanAbsoluteError()})
    trainer.fit(train, validation, epochs=3)  # The shared logger already reports epochs.


if __name__ == "__main__":
    main()
