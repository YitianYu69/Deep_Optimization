"""Optimizer construction, separate from shipped algorithms and scheduling."""

from collections.abc import Callable, Iterable, Mapping

import torch
from torch import nn

from .config import TrainingConfig
from ._config_resolution import resolve_training_config


OptimizerApplyFunction = Callable[[Iterable[nn.Parameter], TrainingConfig], torch.optim.Optimizer]


def build_optimizer(model: nn.Module, config: TrainingConfig | Mapping | None = None,
                    apply_func: OptimizerApplyFunction | None = None) -> torch.optim.Optimizer:
    """Create an optimizer, optionally using apply_func(parameters, settings).

    apply_func runs once during construction, after the model is on its device.
    Return a torch.optim.Optimizer so state saving and scheduling work.
    Without a function, use the configured built-in optimizer.
    Omitted configuration inherits current defaults. Explicit training-table
    dictionaries retain their required fields; apply_func always receives a
    TrainingConfig. Automatic partial snapshots need a learning rate and use
    1 for the callback settings' unused epoch/batch fields, without adding
    those values to the effective experiment snapshot.
    """
    config = resolve_training_config(config, defaults={"epochs": 1, "batch_size": 1})
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
