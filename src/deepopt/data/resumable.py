"""Shuffled batches with a consumed cursor independent of worker prefetching."""

from collections import deque
from collections.abc import Mapping
import weakref

import torch
from torch.utils.data import DataLoader, IterableDataset, Sampler
from deepopt.distributed import _process as distributed
from .seeded import SeededTransformDataset


class ResumableBatchSampler(Sampler):
    def __init__(self, size: int, batch_size: int, generator: torch.Generator, *, drop_last: bool = False):
        if type(size) is not int or size < 0:
            raise ValueError("Dataset size must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("ResumableBatchSampler requires a CPU torch.Generator")
        if type(drop_last) is not bool:
            raise ValueError("drop_last must be a boolean")
        self.size, self.batch_size, self.generator = size, batch_size, generator
        self.drop_last = drop_last
        self.order = None
        self.position = 0
        self.batch_origin = 0

    def __len__(self):
        if self.drop_last:
            return self.size // self.batch_size
        return (self.size + self.batch_size - 1) // self.batch_size

    def _ensure_order(self):
        if self.order is None:
            self.order = torch.randperm(self.size, generator=self.generator)
            self.position = 0
            self.batch_origin = 0

    def __iter__(self):
        self._ensure_order()
        while self.position < self.size:
            if self.drop_last and self.size - self.position < self.batch_size:
                self.position = self.size
                break
            end = min(self.position + self.batch_size, self.size)
            indices = self.order[self.position:end].tolist()
            # Commit a discarded tail with the last full batch, before yielding.
            # A checkpoint at that update must remain exhausted after rebatching.
            self.position = (self.size if self.drop_last and self.size - end < self.batch_size else end)
            yield indices
        # Keep the exhausted order until a NEW epoch iterator is requested.
        # A final partial accumulation window may checkpoint after exhaustion.

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "drop_last": self.drop_last,
                "order": self.order.clone() if self.order is not None else None,
                "position": self.position, "batch_origin": self.batch_origin}

    def _valid_order(self, order):
        return (isinstance(order, torch.Tensor) and order.device.type == "cpu" and order.dtype == torch.long
                and order.shape == (self.size,)
                and torch.equal(order.sort().values, torch.arange(self.size)))

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or not {"size", "batch_size", "order", "position"}.issubset(state):
            raise ValueError("Invalid saved sampler state")
        if type(state["size"]) is not int or state["size"] != self.size:
            raise ValueError("Resumed dataset size must match the checkpoint")
        saved_drop_last = state.get("drop_last", False)
        if type(saved_drop_last) is not bool or saved_drop_last != self.drop_last:
            raise ValueError("Resumed drop_last must match the checkpoint")
        saved_batch_size = state["batch_size"]
        if type(saved_batch_size) is not int or saved_batch_size < 1:
            raise ValueError("Saved batch_size must be a positive integer")
        order, position = state["order"], state["position"]
        if type(position) is not int or not 0 <= position <= self.size:
            raise ValueError("Invalid saved batch position")
        # Legacy snapshots were always batched from the start of the epoch.
        origin = state.get("batch_origin", 0)
        if type(origin) is not int or not 0 <= origin <= position:
            raise ValueError("Invalid saved batch_origin")
        if order is not None:
            if not self._valid_order(order):
                raise ValueError("Saved sampler order must be a permutation of dataset indices")
            if position != self.size and (position - origin) % saved_batch_size:
                raise ValueError("Saved sampler position must be a batch boundary")
        elif position != 0 or origin != 0:
            raise ValueError("A sampler without an active order must have position and batch_origin zero")
        restored_order = order.clone() if order is not None else None
        # Keep the requested batch size. New groups begin at the exact consumed
        # sample cursor, which may not be divisible by either batch size.
        restored_origin = position if saved_batch_size != self.batch_size else origin
        self.order = restored_order
        self.position = position
        self.batch_origin = restored_origin


class _DeliveryBatchSampler(ResumableBatchSampler):
    """Remember each dispatched boundary until its batch reaches the caller."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_positions = deque()
        self.augmentation_epoch = None

    def __iter__(self):
        for indices in super().__iter__():
            self.pending_positions.append(self.position)
            yield ([(self.augmentation_epoch, index) for index in indices]
                   if self.augmentation_epoch is not None else indices)


class _DistributedDeliveryBatchSampler(_DeliveryBatchSampler):
    """Equal rank shards of a common permutation, with PyTorch-style padding."""

    def __init__(self, size, batch_size, generator, *, drop_last, rank, world_size):
        self.dataset_size, self.rank, self.world_size = size, rank, world_size
        super().__init__((size + world_size - 1) // world_size, batch_size, generator, drop_last=drop_last)

    def _ensure_order(self):
        if self.order is None:
            order = torch.randperm(self.dataset_size, generator=self.generator)
            total = self.size * self.world_size
            if total:
                order = order.repeat((total + self.dataset_size - 1) // self.dataset_size)[:total]
            self.order = order[self.rank:total:self.world_size].clone()
            self.position = self.batch_origin = 0

    def _valid_order(self, order):
        return (isinstance(order, torch.Tensor) and order.device.type == "cpu" and order.dtype == torch.long
                and order.shape == (self.size,) and bool(((order >= 0) & (order < self.dataset_size)).all()))

    def state_dict(self):
        return {**super().state_dict(), "distributed": {"dataset_size": self.dataset_size,
                                                        "rank": self.rank, "world_size": self.world_size}}

    def load_state_dict(self, state):
        expected = {"dataset_size": self.dataset_size, "rank": self.rank, "world_size": self.world_size}
        if not isinstance(state, Mapping) or state.get("distributed") != expected:
            raise ValueError("Distributed sampler resume requires the same dataset size, rank, and world size")
        super().load_state_dict(state)


class _DeliveryIterator:
    def __init__(self, loader, iterator):
        self.loader = loader
        self.iterator = iterator
        self.valid = True
        self.finished = False

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader.batch_sampler)

    def __next__(self):
        if not self.valid:
            raise RuntimeError("This loader iterator is no longer active; request a new iterator")
        if self.finished:
            raise StopIteration
        if self.iterator is None:
            # A live resize discarded prefetched batches. Continue the same
            # public iterator, rebuilding workers' queues at the consumed cursor.
            self.iterator = self.loader._make_iterator(resuming=True)
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.loader._consumed_position = self.loader.batch_sampler.position
            self.finished = True
            self.iterator = None
            raise
        except Exception:
            # A failed batch must not be skipped by continuing this iterator.
            # The last delivered cursor remains available for checkpoint restore.
            self.invalidate()
            raise
        self.loader._consumed_position = self.loader.batch_sampler.pending_positions.popleft()
        return batch

    def invalidate(self):
        self.valid = False
        self.iterator = None


class ResumableDataLoader(DataLoader):
    """Restore shuffled sample order and the last delivered batch boundary.

    Supports workers, prefetch_factor and persistent_workers with ordered
    delivery. Snapshots exclude prefetched batches that have not been returned
    to the caller. Restoring or resizing regroups only the remaining samples.
    Worker count and prefetch settings may change on restore.

    Worker-local dataset/transform RNGs are not checkpointed. For exact random
    transform recovery, use StatefulDataLoader with dataset state methods.
    """
    supports_batch_size_change = True

    def __init__(self, dataset, *, batch_size: int, generator: torch.Generator,
                 drop_last: bool = False, num_workers: int = 0, **loader_options):
        if isinstance(dataset, IterableDataset):
            raise ValueError("ResumableDataLoader requires a map-style dataset")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        # Older supported PyTorch versions always deliver in order and do not
        # accept this keyword. Keep their default while rejecting unordered use.
        if loader_options.pop("in_order", True) is not True:
            raise ValueError("ResumableDataLoader requires in_order=True")
        if distributed.world_size() > 1:
            distributed.require_equal({"size": len(dataset), "batch_size": batch_size, "drop_last": drop_last,
                                        "generator": generator.get_state().tolist()}, "resumable sampler settings")
            sampler = _DistributedDeliveryBatchSampler(len(dataset), batch_size, generator, drop_last=drop_last,
                                                        rank=distributed.rank(), world_size=distributed.world_size())
        else:
            sampler = _DeliveryBatchSampler(len(dataset), batch_size, generator, drop_last=drop_last)
        super().__init__(dataset, batch_sampler=sampler, generator=generator, num_workers=num_workers, **loader_options)
        # The custom batch sampler owns this policy; expose the same value on
        # the loader without forwarding an incompatible constructor argument.
        object.__setattr__(self, "drop_last", drop_last)
        self._resume_iterator = False
        self._consumed_position = 0
        self._active_iterator = None
        self._augmentation_epoch = None
        self._next_augmentation_epoch = 0

    def __iter__(self):
        self._invalidate_iterator()
        resuming = self._resume_iterator
        if not resuming:
            self.batch_sampler.order, self.batch_sampler.position, self.batch_sampler.batch_origin = None, 0, 0
            self._consumed_position = 0
            if isinstance(self.dataset, SeededTransformDataset):
                self._augmentation_epoch = self._next_augmentation_epoch
                self._next_augmentation_epoch += 1
        self.batch_sampler.augmentation_epoch = self._augmentation_epoch
        self.batch_sampler.pending_positions.clear()
        iterator = _DeliveryIterator(self, self._make_iterator(resuming=resuming))
        # Do not keep abandoned nonpersistent worker iterators alive.
        self._active_iterator = weakref.ref(iterator)
        self._resume_iterator = False
        return iterator

    def _make_iterator(self, *, resuming):
        # DataLoader draws an iterator base seed. Recreating an existing epoch
        # must not consume this generator for a second time.
        generator = self.generator
        state = generator.get_state() if resuming else None
        try:
            if isinstance(self.batch_sampler, _DistributedDeliveryBatchSampler):
                # All ranks share the shuffle generator, but workers must have
                # independent RNG streams. Consume the normal epoch base-seed
                # draw, then give DataLoader a temporary rank-specific generator.
                # The sampler still references the original shared generator.
                base_seed = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
                self.generator = torch.Generator().manual_seed((base_seed + self.batch_sampler.rank) % 2**64)
            elif not resuming and self.persistent_workers and self._iterator is not None:
                # PyTorch skips the base-seed draw when reusing workers. Keep
                # shuffle RNG progression independent of worker reuse/count.
                torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            iterator = super().__iter__()
            # Also initialize zero-worker orders before a possible checkpoint
            # between iter(loader) and the first next(iterator).
            self.batch_sampler._ensure_order()
            return iterator
        finally:
            self.generator = generator
            if state is not None:
                generator.set_state(state)

    def _invalidate_iterator(self):
        iterator = self._active_iterator() if self._active_iterator is not None else None
        if iterator is not None:
            iterator.invalidate()
        self._active_iterator = None

    def state_dict(self):
        sampler_state = self.batch_sampler.state_dict()
        sampler_state["position"] = self._consumed_position
        augmentation = ({"augmentation": {"config": self.dataset.checkpoint_config(),
                                           "epoch": self._augmentation_epoch, "next_epoch": self._next_augmentation_epoch}}
                        if isinstance(self.dataset, SeededTransformDataset) else {})
        return {"format_version": 3, "sampler": sampler_state,
                "generator": self.generator.get_state(), **augmentation}

    def set_epoch(self, epoch):
        """Set the next fresh augmentation epoch; active/restored cursors stay intact."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._next_augmentation_epoch = epoch

    def remaining_batches(self):
        """Number still deliverable in the active/restored epoch, not prefetched."""
        sampler = self.batch_sampler
        remaining = sampler.size - self._consumed_position if sampler.order is not None else sampler.size
        if sampler.drop_last:
            return remaining // sampler.batch_size
        return (remaining + sampler.batch_size - 1) // sampler.batch_size

    def set_batch_size(self, batch_size: int) -> None:
        """Regroup future examples without resetting the current data cursor.

        Call at an optimizer boundary when training. Loading a saved snapshot
        afterwards keeps this requested size while restoring its exact cursor.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        sampler = self.batch_sampler
        if batch_size != sampler.batch_size:
            iterator = self._active_iterator() if self._active_iterator is not None else None
            if iterator is not None:
                iterator.iterator = None
            sampler.position = self._consumed_position
            sampler.pending_positions.clear()
            sampler.batch_size = batch_size
            sampler.batch_origin = sampler.position if sampler.order is not None else 0

    def reset_epoch(self):
        """Discard an active cursor when restoring a completed epoch instead."""
        self._invalidate_iterator()
        self.batch_sampler.order, self.batch_sampler.position, self.batch_sampler.batch_origin = None, 0, 0
        self.batch_sampler.pending_positions.clear()
        self._consumed_position = 0
        self._resume_iterator = False
        self._augmentation_epoch = None
        self.batch_sampler.augmentation_epoch = None

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or type(state.get("format_version")) is not int or state["format_version"] not in (1, 2, 3):
            raise ValueError("Unsupported loader checkpoint format")
        sampler_state = state.get("sampler")
        if not isinstance(sampler_state, Mapping):
            raise ValueError("Invalid saved sampler state")
        if state["format_version"] >= 2 and "batch_origin" not in sampler_state:
            raise ValueError("Loader checkpoint format 2+ requires sampler batch_origin")
        if state["format_version"] == 3 and "drop_last" not in sampler_state:
            raise ValueError("Loader checkpoint format 3 requires sampler drop_last")
        augmentation = state.get("augmentation")
        if isinstance(self.dataset, SeededTransformDataset):
            if not isinstance(augmentation, Mapping) or augmentation.get("config") != self.dataset.checkpoint_config():
                raise ValueError("Resume requires the same seeded transform configuration")
            epoch, next_epoch = augmentation.get("epoch"), augmentation.get("next_epoch")
            if ((epoch is not None and (type(epoch) is not int or epoch < 0))
                    or type(next_epoch) is not int or next_epoch < 0):
                raise ValueError("Invalid saved augmentation epoch")
            if (sampler_state.get("order") is None) != (epoch is None):
                raise ValueError("Augmentation epoch must match the saved sampler activity")
        elif augmentation is not None:
            raise ValueError("Seeded augmentation checkpoints require SeededTransformDataset")
        # Validate the RNG state before changing either sampler or generator.
        # A temporary generator does not draw from this loader or global RNGs.
        generator_state = state.get("generator")
        try:
            torch.Generator().set_state(generator_state)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError("Invalid saved loader generator state") from exc
        self.batch_sampler.load_state_dict(sampler_state)
        self._invalidate_iterator()
        self.batch_sampler.pending_positions.clear()
        self._consumed_position = self.batch_sampler.position
        self.generator.set_state(generator_state)
        self._resume_iterator = self.batch_sampler.order is not None
        if augmentation is not None:
            self._augmentation_epoch = augmentation["epoch"]
            self._next_augmentation_epoch = augmentation["next_epoch"]
            self.batch_sampler.augmentation_epoch = self._augmentation_epoch
