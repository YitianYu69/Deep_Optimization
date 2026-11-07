"""Shared deterministic recipes for global training configuration tests."""

import torch


def make_config():
    return {"name": "global-training", "seed": 17,
            "training": {"epochs": 2, "batch_size": 3, "learning_rate": 0.05,
                         "optimizer": "sgd", "momentum": 0.0, "monitor": None},
            "runtime": {"device": "cpu", "precision": "fp32", "checks": True}}


def make_batches():
    generator = torch.Generator().manual_seed(615)
    return [(torch.randn(size, 2, generator=generator), torch.randn(size, 1, generator=generator))
            for size in (2, 1, 3)]
