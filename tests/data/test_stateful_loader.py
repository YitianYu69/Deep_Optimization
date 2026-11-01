import copy
import importlib.util
import unittest

import torch
from torch.utils.data import Dataset


class WorkerData(Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, index):
        return index, torch.rand(())

    def state_dict(self):
        return {"rng": torch.get_rng_state()}

    def load_state_dict(self, state):
        torch.set_rng_state(state["rng"])


@unittest.skipUnless(importlib.util.find_spec("torchdata"), "Install optional [workers] dependencies")
class StatefulLoaderTests(unittest.TestCase):
    def loader(self, **kwargs):
        from deepopt.data import StatefulDataLoader
        return StatefulDataLoader(WorkerData(), batch_size=2, num_workers=2, multiprocessing_context="spawn", timeout=15,
                                   generator=torch.Generator().manual_seed(17), **kwargs)

    def test_mid_epoch_restores_worker_rng_and_consumed_batches(self):
        first = self.loader()
        iterator = iter(first)
        next(iterator)
        next(iterator)
        saved = copy.deepcopy(first.state_dict())
        expected = list(iterator)
        resumed = self.loader()
        resumed.load_state_dict(saved)
        actual = list(resumed)
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_exhausted_resume_is_empty_once_then_new_epoch_starts(self):
        first = self.loader()
        list(first)
        saved = copy.deepcopy(first.state_dict())
        resumed = self.loader()
        resumed.load_state_dict(saved)
        self.assertEqual(list(resumed), [])
        self.assertEqual(len(list(resumed)), 5)

    def test_snapshot_between_restore_and_iteration_preserves_position(self):
        first = self.loader()
        iterator = iter(first)
        next(iterator)
        saved = copy.deepcopy(first.state_dict())
        expected = list(iterator)
        resumed = self.loader()
        resumed.load_state_dict(saved)
        resumed.state_dict()
        actual = list(resumed)
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
