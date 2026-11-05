"""Build ordinary or explicitly resumable loaders from caller-owned datasets."""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, BatchSampler
from deepopt.distributed import _process as distributed
from deepopt._config_resolution import _UNSET, resolve_config

from .resumable import ResumableDataLoader


class _EpochDataLoader(DataLoader):
    """Keep epoch generator draws consistent when persistent workers are reused."""

    def __iter__(self):
        if self.persistent_workers and self._iterator is not None:
            # A freshly reconstructed iterator draws a base seed before sampling.
            # Match that draw on reuse so epoch resume preserves shuffle order
            # and named generator states. This does not restore worker-local RNGs.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        return super().__iter__()

    def set_batch_size(self, size):
        if type(size) is not int or size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self._iterator is not None or type(self.batch_sampler) is not BatchSampler:
            raise ValueError("Resize ordinary loaders only before iteration with a standard BatchSampler")
        self.batch_sampler.batch_size = size
        object.__setattr__(self, "batch_size", size)


class DistributedEvalSampler(Sampler):
    """Validation shards without padding or duplication, including empty ranks."""

    def __init__(self, dataset):
        self.size, self.rank, self.world_size = len(dataset), distributed.rank(), distributed.world_size()

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))


def build_loaders(train_data: Dataset, validation_data: Dataset | None, *, batch_size=_UNSET, seed=_UNSET,
                  num_workers=_UNSET, resumable=_UNSET, config=None, **loader_options):
    """Use resumable=True for step and epoch resume with any worker count.

    Restores sample order/cursor, including with prefetch and rebatching. For
    worker-local dataset/RNG recovery, supply a StatefulDataLoader factory.

    Extra DataLoader options (collate_fn, pin_memory, prefetch_factor, etc.) apply
    to both phases. The helper owns batch_size, generator and shuffle; supply a
    custom loader_factory for other sampler/batching contracts.

    Omitted batch size, seed, and loader options inherit the selected config.
    Explicit keywords override it. A positive configured step-save interval
    selects resumable loading unless resumable is explicitly provided.
    """
    overrides = {}
    if batch_size is not _UNSET:
        overrides["training"] = {"batch_size": batch_size}
    if seed is not _UNSET:
        overrides["seed"] = seed
    explicit_loader = {key: loader_options[key] for key in
                       ("pin_memory", "persistent_workers", "prefetch_factor", "drop_last")
                       if key in loader_options}
    if num_workers is not _UNSET:
        explicit_loader["num_workers"] = num_workers
    if explicit_loader:
        overrides["loader"] = explicit_loader
    snapshot = resolve_config(config, overrides=overrides)
    training = snapshot.get("training", {})
    options = dict(snapshot.get("loader", {}))
    if batch_size is _UNSET:
        batch_size = training.get("batch_size", _UNSET)
    if seed is _UNSET:
        seed = snapshot.get("seed", _UNSET)
    if batch_size is _UNSET or seed is _UNSET:
        raise ValueError("build_loaders requires batch_size and seed, explicitly or in configuration")
    configured_workers = options.pop("num_workers", 0)
    if num_workers is _UNSET:
        num_workers = configured_workers
    if resumable is _UNSET:
        resumable = training.get("checkpoint_every_steps", 0) > 0
    loader_options = {**options, **loader_options}
    if type(resumable) is not bool:
        raise ValueError("resumable must be a boolean")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if distributed.world_size() > 1:
        distributed.require_equal({"train_size": len(train_data),
                                    "validation_size": len(validation_data) if validation_data is not None else None,
                                    "batch_size": batch_size, "seed": seed, "resumable": resumable,
                                    "drop_last": loader_options.get("drop_last", False)}, "loader settings")
    worker_offset = distributed.rank() * 2**32
    # Ordinary DistributedSampler owns its shuffle seed independently. The
    # resumable sampler instead shares a shuffle generator across ranks and
    # derives distinct worker seeds inside its iterator.
    generators = {"train": torch.Generator().manual_seed(seed + 1 + (0 if resumable else worker_offset)),
                  "validation": torch.Generator().manual_seed(seed + 2 + worker_offset)}
    epoch_loader = _EpochDataLoader if loader_options.get("persistent_workers", False) else DataLoader
    distributed_training = distributed.world_size() > 1
    if resumable:
        train = ResumableDataLoader(train_data, batch_size=batch_size,
                                    generator=generators["train"], num_workers=num_workers, **loader_options)
    else:
        sampler = DistributedSampler(train_data, shuffle=True, seed=seed + 1) if distributed_training else None
        train = epoch_loader(train_data, batch_size=batch_size, generator=generators["train"],
                             shuffle=sampler is None, sampler=sampler, num_workers=num_workers, **loader_options)
    validation = (epoch_loader(validation_data, batch_size=batch_size, generator=generators["validation"],
                               sampler=DistributedEvalSampler(validation_data) if distributed_training else None,
                               num_workers=num_workers, **loader_options)
                  if validation_data is not None else None)
    return train, validation, generators
