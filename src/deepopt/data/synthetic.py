"""A small offline classification dataset for checking the training pipeline."""

from dataclasses import dataclass, fields

import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class SyntheticDataSettings:
    """Options for this artificial dataset only, never a framework schema."""

    train_samples: int = 256
    validation_samples: int = 64
    input_dim: int = 16
    num_classes: int = 4

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 2 if field.name == "num_classes" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"Synthetic data {field.name} must be an integer >= {minimum}")

    @classmethod
    def from_dict(cls, options):
        if not isinstance(options, dict):
            raise ValueError("application.synthetic must be a table of synthetic-demo options")
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown synthetic-demo options: {', '.join(sorted(unknown))}")
        return cls(**options)


def build_datasets(config: SyntheticDataSettings | dict, seed: int) -> tuple[TensorDataset, TensorDataset]:
    """Build this demo's datasets from typed settings or the same settings dict."""
    if not isinstance(config, SyntheticDataSettings):
        config = SyntheticDataSettings.from_dict(config)
    # This generator does not consume model/dropout randomness. The splits
    # share a teacher, but contain independently drawn input examples.
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(config.input_dim, config.num_classes, generator=generator)

    def make_split(count: int) -> TensorDataset:
        inputs = torch.randn(count, config.input_dim, generator=generator)
        targets = (inputs @ teacher).argmax(dim=1)
        return TensorDataset(inputs, targets)

    return make_split(config.train_samples), make_split(config.validation_samples)
