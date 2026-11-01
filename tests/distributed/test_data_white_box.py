"""White-box checks for rank metadata and generator ownership."""

import unittest
from unittest.mock import patch

import torch

from deepopt.data import ResumableDataLoader
from deepopt.data.resumable import _DistributedDeliveryBatchSampler


class DistributedDataWhiteBoxTests(unittest.TestCase):
    def test_worker_generator_is_separate_and_restored_even_on_iterator_failure(self):
        generator = torch.Generator().manual_seed(7)
        loader = ResumableDataLoader(list(range(8)), batch_size=2, generator=generator)
        sampler = _DistributedDeliveryBatchSampler(8, 2, generator, drop_last=False, rank=1, world_size=2)
        sampler._ensure_order()
        object.__setattr__(loader, "batch_sampler", sampler)
        before = generator.get_state().clone()

        def fail():
            self.assertIsNot(loader.generator, generator)
            self.assertIs(sampler.generator, generator)
            raise RuntimeError("iterator construction failed")

        with patch("torch.utils.data.DataLoader.__iter__", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "construction failed"):
                loader._make_iterator(resuming=True)
        self.assertIs(loader.generator, generator)
        torch.testing.assert_close(generator.get_state(), before, rtol=0, atol=0)

    def test_rank_state_cannot_restore_into_another_rank(self):
        source = _DistributedDeliveryBatchSampler(8, 2, torch.Generator().manual_seed(7),
                                                  drop_last=False, rank=0, world_size=2)
        next(iter(source))
        target = _DistributedDeliveryBatchSampler(8, 2, torch.Generator().manual_seed(7),
                                                  drop_last=False, rank=1, world_size=2)
        with self.assertRaisesRegex(ValueError, "rank"):
            target.load_state_dict(source.state_dict())
        self.assertIsNone(target.order)
        self.assertEqual(target.position, 0)


if __name__ == "__main__":
    unittest.main()
