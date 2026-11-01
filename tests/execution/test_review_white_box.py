import unittest
from unittest.mock import patch

import torch

from deepopt.data import ResumableDataLoader, SeededTransformDataset, sample_seed
from deepopt.distributed.ddp import DDPCore
from deepopt.observers import CheckObservers


class ReviewWhiteBoxTests(unittest.TestCase):
    def test_remaining_batches_uses_delivery_cursor_and_new_batch_size(self):
        loader = ResumableDataLoader(list(range(11)), batch_size=4, generator=torch.Generator().manual_seed(7))
        self.assertEqual(loader.remaining_batches(), 3)
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        loader.batch_sampler.position = 11  # Simulate dispatch running ahead.
        self.assertEqual(loader.remaining_batches(), 2)
        loader.load_state_dict(state)
        loader.set_batch_size(3)
        self.assertEqual(loader.remaining_batches(), 3)
        list(loader)
        self.assertEqual(loader.remaining_batches(), 0)

    def test_unobserved_events_and_unweighted_divisor_need_no_collective(self):
        core = object.__new__(DDPCore)
        core._observer_interests = set()
        checks = CheckObservers(False)

        def unavailable():
            raise AssertionError("Disabled payload was inspected")
            yield

        with patch("torch.distributed.all_reduce", side_effect=AssertionError("redundant collective")), \
                patch("torch.distributed.all_gather_object", side_effect=AssertionError("redundant object collective")):
            self.assertFalse(core.interested(checks, "weights"))
            core.observe(checks, "loss", unavailable())
            self.assertEqual(core.gradient_divisor(1, normalize=False), 1)

    def test_seed_components_are_stable_and_do_not_touch_global_rng(self):
        before = torch.get_rng_state().clone()
        seed = sample_seed(17, 2, 8, 0)
        self.assertEqual(seed, sample_seed(17, 2, 8, 0))
        self.assertEqual(len({sample_seed(17, epoch, index, view) for epoch in range(2)
                             for index in range(3) for view in range(2)}), 12)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        for arguments in ((True, 0, 0), (0, -1, 0), (0, 0, 1.5)):
            with self.assertRaises(ValueError):
                sample_seed(*arguments)

    def test_seeded_snapshot_rejects_changed_transform_before_mutation(self):
        def transform(sample, *, generator):
            return sample + torch.rand((), generator=generator).item()

        original = ResumableDataLoader(SeededTransformDataset(list(range(11)), transform, seed=7, transform_id="v1"),
                                       batch_size=4, generator=torch.Generator().manual_seed(3))
        next(iter(original))
        state = original.state_dict()
        restored = ResumableDataLoader(SeededTransformDataset(list(range(11)), transform, seed=7, transform_id="v2"),
                                       batch_size=4, generator=torch.Generator().manual_seed(3))
        before = restored.generator.get_state().clone()
        with self.assertRaisesRegex(ValueError, "transform"):
            restored.load_state_dict(state)
        self.assertIsNone(restored.batch_sampler.order)
        torch.testing.assert_close(restored.generator.get_state(), before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
