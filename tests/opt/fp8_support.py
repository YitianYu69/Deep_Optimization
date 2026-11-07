"""Small FP8 training fixtures and an independent quantization oracle."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import Trainer
from deepopt.data import ResumableDataLoader
from deepopt.opt import transform_model


def fp8_plan(backend="reference"):
    return {
        "fp8_training": {"fp8_dtypes": {
            "activation": torch.float8_e4m3fn,
            "weight": torch.float8_e4m3fn,
            "grad": torch.float8_e4m3fn if backend == "triton" else torch.float8_e5m2,
        }, "backend": backend},
        "block_shapes": {name: None for name in ("activation", "weight", "grad")},
    }


def make_training(mode="full", *, device="cpu", backend="reference"):
    torch.manual_seed(81)
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4)).to(device)
    model, _ = transform_model(model, plan=fp8_plan(backend), trace_mode=mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, accumulation_steps=2, device=device,
                      scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9))
    rng = torch.Generator().manual_seed(12)
    dataset = TensorDataset(torch.randn(13, 8, generator=rng), torch.randn(13, 4, generator=rng))
    loader = ResumableDataLoader(dataset, batch_size=3,
                                 generator=torch.Generator().manual_seed(27))
    return trainer, loader


class InterruptedTraining(Exception):
    pass


def round_fp8(matrix, dtype, block_shape):
    """Independent, deliberately slow scalar-block oracle; no deepopt helpers."""
    matrix = matrix.detach().float()
    reconstructed = torch.empty_like(matrix)
    block_rows, block_cols = block_shape or tuple(matrix.shape)
    fp8_max = torch.finfo(dtype).max
    for row in range(0, matrix.shape[0], block_rows):
        for col in range(0, matrix.shape[1], block_cols):
            block = matrix[row:row + block_rows, col:col + block_cols]
            amax = block.abs().max()
            scale = torch.ones_like(amax) if amax == 0 else fp8_max / amax.clamp_min(1e-12)
            values = (block * scale).clamp(-fp8_max, fp8_max).to(dtype)
            reconstructed[row:row + block_rows, col:col + block_cols] = (
                values.float() * scale.reciprocal()
            )
    return reconstructed
