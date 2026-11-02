"""Optimizer construction, separate from shipped algorithms and scheduling."""

from collections.abc import Callable, Iterable

import torch
from torch import nn

from .config import TrainingConfig


OptimizerApplyFunction = Callable[[Iterable[nn.Parameter], TrainingConfig], torch.optim.Optimizer]


def build_optimizer(model: nn.Module, config: TrainingConfig,
                    apply_func: OptimizerApplyFunction | None = None) -> torch.optim.Optimizer:
    """Create an optimizer, optionally using apply_func(parameters, settings).

    apply_func runs once during construction, after the model is on its device.
    Return a torch.optim.Optimizer so state saving and scheduling work.
    Without a function, use the configured built-in optimizer.
    """
    if apply_func is not None:
        optimizer = apply_func(model.parameters(), config)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("apply_func must return a torch.optim.Optimizer")
        return optimizer
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                               momentum=config.momentum, weight_decay=config.weight_decay)
    if config.optimizer == "custom":
        raise ValueError("optimizer='custom' requires an apply_func or a custom optimizer_factory")
    raise ValueError(f"Unknown optimizer: {config.optimizer}")
