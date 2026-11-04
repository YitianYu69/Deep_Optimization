"""Assign different optimizers to disjoint layers, then use the normal fit API.

Run: python examples/optimization/multiple_optimizers.py
"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt.logger import log_output
from deepopt import MultiOptimizer, Trainer


def main():
    torch.manual_seed(42)
    features = torch.randn(48, 4)
    targets = features.square().sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(features, targets), batch_size=8,
                        shuffle=True, generator=torch.Generator().manual_seed(43))
    model = nn.Sequential(OrderedDict([
        ("backbone", nn.Sequential(nn.Linear(4, 16), nn.SiLU())),
        ("head", nn.Linear(16, 1)),
    ]))

    # You choose the assignments. Each update steps both children once.
    optimizer = MultiOptimizer({
        "backbone": torch.optim.AdamW(model.backbone.parameters(), lr=0.01),
        "head": torch.optim.SGD(model.head.parameters(), lr=0.02, momentum=0.9),
    })
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
    trainer.fit(loader, epochs=3, progress_bar=False, callbacks=[lambda _, row: log_output(row)])


if __name__ == "__main__":
    main()
