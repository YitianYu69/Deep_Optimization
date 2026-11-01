"""Skip a known batch prefix before fetching data; not worker-state recovery."""

from itertools import islice

from torch.utils.data import DataLoader, IterableDataset, Sampler


def _validate_skip(value):
    if type(value) is not int or value < 0:
        raise ValueError("skip_batches must be a nonnegative integer")


class SkipBatchSampler(Sampler):
    """Wrap a batch sampler and skip its first N batches on the next iterator.

    Only indices are traversed; skipped samples are not fetched or collated.
    Reconstruct the SAME sampler order first (including its epoch-start RNG).
    This utility does not restore model, sampler, dataset, or worker RNG state.
    Later iterators start normally; call skip(count) to request another skip.
    """

    def __init__(self, batch_sampler, skip_batches: int = 0):
        self.batch_sampler = batch_sampler
        self.skip(skip_batches)

    def skip(self, count: int):
        _validate_skip(count)
        self.skip_batches = count

    def __iter__(self):
        count = self.skip_batches
        self.skip_batches = 0
        # DataLoader may construct and discard an iterator during worker setup.
        # Consume the skip only when this generator actually starts yielding.
        yield from islice(iter(self.batch_sampler), count, None)

    def __len__(self):
        return max(0, len(self.batch_sampler) - self.skip_batches)

    @property
    def batch_size(self):
        return getattr(self.batch_sampler, "batch_size", None)


class SkipDataLoader(DataLoader):
    """Ordinary PyTorch DataLoader with skip_batches=N for its next iterator.

    Accepts DataLoader options, including num_workers and collate_fn, or an
    explicit batch_sampler. Auto-batching is required. This is index skipping,
    not a stateful loader: step-checkpoint validation intentionally rejects it.
    For exact worker-state recovery use deepopt.data.StatefulDataLoader instead.
    """

    def __init__(self, *args, skip_batches: int = 0, **kwargs):
        _validate_skip(skip_batches)
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            raise ValueError("SkipDataLoader requires a map-style dataset; index skipping cannot skip an iterable data stream")
        if self.batch_sampler is None:
            raise ValueError("SkipDataLoader requires batch_size or batch_sampler")
        # Install before any iterator/worker exists. Keep the rest of DataLoader
        # initialization and worker behavior unchanged.
        object.__setattr__(self, "batch_sampler", SkipBatchSampler(self.batch_sampler, skip_batches))

    def skip(self, count: int):
        """Set the prefix for the next iterator; do not call during iteration."""
        self.batch_sampler.skip(count)
