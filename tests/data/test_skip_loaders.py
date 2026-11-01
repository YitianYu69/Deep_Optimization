import unittest

import torch
from torch.utils.data import BatchSampler, Dataset, IterableDataset, SequentialSampler

from deepopt.data import SkipBatchSampler, SkipDataLoader
from deepopt.engine.checkpointing.checkpoint import require_step_loader


class GuardedDataset(Dataset):
    def __init__(self, size=10, minimum_index=0):
        self.size, self.minimum_index = size, minimum_index

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index < self.minimum_index:
            raise AssertionError("Skipped samples must not be fetched")
        return index


class Stream(IterableDataset):
    def __iter__(self):
        return iter(range(10))


class SkipLoaderTests(unittest.TestCase):
    def test_batch_sampler_skips_once_and_can_be_rearmed(self):
        sampler = SkipBatchSampler(BatchSampler(SequentialSampler(range(9)), batch_size=2, drop_last=False), 3)
        self.assertEqual(len(sampler), 2)
        self.assertEqual(list(sampler), [[6, 7], [8]])
        self.assertEqual(list(sampler), [[0, 1], [2, 3], [4, 5], [6, 7], [8]])
        sampler.skip(99)
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])

    def test_loader_never_fetches_skipped_samples(self):
        loader = SkipDataLoader(GuardedDataset(10, minimum_index=6), batch_size=2, skip_batches=3)
        self.assertEqual(torch.cat(list(loader)).tolist(), [6, 7, 8, 9])
        loader.dataset.minimum_index = 0
        self.assertEqual(torch.cat(list(loader)).tolist(), list(range(10)))
        loader.skip(4)
        self.assertEqual(torch.cat(list(loader)).tolist(), [8, 9])

    def test_worker_prefetch_does_not_fetch_skipped_indices(self):
        loader = SkipDataLoader(GuardedDataset(10, minimum_index=6), batch_size=2, skip_batches=3,
                                num_workers=2, multiprocessing_context="spawn", collate_fn=list, timeout=15)
        # Lists test real workers/prefetch without requiring tensor shared-memory
        # services, which are unavailable in some restricted test environments.
        self.assertEqual([item for batch in loader for item in batch], [6, 7, 8, 9])

    def test_custom_batch_sampler_and_drop_last(self):
        sampler = BatchSampler(SequentialSampler(range(9)), batch_size=2, drop_last=True)
        loader = SkipDataLoader(GuardedDataset(9, minimum_index=6), batch_sampler=sampler, skip_batches=3)
        self.assertEqual(torch.cat(list(loader)).tolist(), [6, 7])

    def test_skip_does_not_claim_checkpoint_or_iterable_recovery(self):
        loader = SkipDataLoader(GuardedDataset(), batch_size=2)
        with self.assertRaisesRegex(ValueError, "resumable loader"):
            require_step_loader(loader)
        with self.assertRaisesRegex(ValueError, "map-style"):
            SkipDataLoader(Stream(), batch_size=2)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            SkipDataLoader(GuardedDataset(), batch_size=None)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                SkipDataLoader(GuardedDataset(), skip_batches=value)

    def test_shuffle_replay_requires_same_epoch_start_generator(self):
        first = SkipDataLoader(GuardedDataset(), batch_size=2, shuffle=True,
                               generator=torch.Generator().manual_seed(17))
        expected = list(first)
        resumed = SkipDataLoader(GuardedDataset(), batch_size=2, shuffle=True, skip_batches=2,
                                 generator=torch.Generator().manual_seed(17))
        self.assertEqual(torch.cat(list(resumed)).tolist(), torch.cat(expected[2:]).tolist())


if __name__ == "__main__":
    unittest.main()
