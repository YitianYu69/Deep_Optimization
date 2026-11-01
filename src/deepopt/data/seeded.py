"""Stateless CPU transforms keyed by epoch, sample index, and augmentation view."""

import hashlib

import torch
from torch.utils.data import Dataset


def sample_seed(base_seed: int, epoch: int, sample_index: int, view: int = 0) -> int:
    """A stable 64-bit seed; independent of Python hash randomization or workers."""
    values = (base_seed, epoch, sample_index, view)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Seed components must be nonnegative integers")
    encoded = ":".join(map(str, values)).encode("ascii")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")


class SeededTransformDataset(Dataset):
    """Apply transform(sample, *, generator) using a per-sample CPU generator.

    Use with ResumableDataLoader, which sends (epoch, index) keys to workers.
    Persistent workers therefore see the correct epoch without mutable shared
    state. The wrapped dataset must be deterministic, and the transform must use
    the supplied generator for all randomness. Global Python/NumPy RNGs are not
    intercepted. Return the sample index yourself when it is useful to the task.
    """

    def __init__(self, dataset, transform, *, seed: int, view: int = 0, transform_id: str | None = None):
        sample_seed(seed, 0, 0, view)
        if not callable(transform):
            raise TypeError("transform must be callable")
        if transform_id is not None and (not isinstance(transform_id, str) or not transform_id):
            raise ValueError("transform_id must be a nonempty string")
        self.dataset, self.transform, self.seed, self.view = dataset, transform, seed, view
        self.transform_id = transform_id or f"{transform.__module__}.{getattr(transform, '__qualname__', type(transform).__qualname__)}"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        generator = torch.Generator().manual_seed(sample_seed(self.seed, epoch, index, self.view))
        return self.transform(self.dataset[index], generator=generator)

    def checkpoint_config(self):
        return {"size": len(self), "seed": self.seed, "view": self.view, "transform_id": self.transform_id}
