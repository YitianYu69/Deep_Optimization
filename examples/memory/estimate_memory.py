"""Estimate a training step from a representative batch using fake tensors.

Run: python examples/memory/estimate_memory.py
"""

import json

import torch
from torch import nn
from torch.nn import functional as F

from deepopt.logger import log_output
from deepopt import Trainer
from deepopt.memory import estimate_training_memory


def main():
    model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
    sample_batch = (torch.empty(8, 6), torch.empty(8, 2))
    estimate = estimate_training_memory(trainer, sample_batch)
    log_output(json.dumps(estimate.as_dict(), indent=2))


if __name__ == "__main__":
    main()
