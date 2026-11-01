"""Count parameters and inspect allocated model/optimizer storage.

Run: python examples/memory/model_memory.py
Initialization logs the summary; an explicit refresh shows Adam's lazy state.
"""

import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt import Trainer
from deepopt.logger import log_output
from deepopt.memory import log_model_memory


def main():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    # Frozen parameters still occupy model storage and appear in the percentage.
    model[0].requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)  # Logs initial counts/GB.
    inputs = torch.randn(8, 3)
    loader = DataLoader(TensorDataset(inputs, inputs.sum(dim=1, keepdim=True)), batch_size=4)
    trainer.fit(loader, epochs=1, progress_bar=False)
    # Call again after training or checkpoint restore to inspect current state.
    report = log_model_memory(model, optimizer)
    log_output(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
