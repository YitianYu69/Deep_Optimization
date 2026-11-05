"""The CLI's synthetic demo recipe; these choices are not core requirements."""

from functools import partial

import torch

from .data.synthetic import SyntheticDataSettings, build_datasets
from .config import config_from_dict
from ._config_resolution import resolve_config
from .experiment import build_experiment
from .models.mlp import build_mlp
from .tasks.classification import ClassificationTask


def baseline_sample_batch(config=None, size=None):
    """Describe the synthetic recipe's batch for fake-tensor memory planning."""
    if type(size) is not int or size < 1:
        raise ValueError("Sample batch size must be a positive integer")
    config = config_from_dict(resolve_config(config, require_complete=True))
    settings = SyntheticDataSettings.from_dict(config.application.get("synthetic", {}))
    return (torch.empty(size, settings.input_dim, dtype=torch.float32),
            torch.empty(size, dtype=torch.long))


def build_baseline_experiment(config=None, **overrides):
    config = config_from_dict(resolve_config(config, require_complete=True))
    settings = SyntheticDataSettings.from_dict(config.application.get("synthetic", {}))
    defaults = {
        "dataset_factory": partial(build_datasets, settings, config.seed),
        "model_factory": partial(build_mlp, input_dim=settings.input_dim,
                                 num_classes=settings.num_classes, hidden_dim=32, dropout=0.1),
        "task_factory": partial(ClassificationTask, num_classes=settings.num_classes),
    }
    return build_experiment(config, **(defaults | overrides))
