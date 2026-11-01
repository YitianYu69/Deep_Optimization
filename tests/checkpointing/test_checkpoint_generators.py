"""Epoch continuation through ordinary and wrapped PyTorch batch samplers."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from deepopt import Checkpoint, Trainer
from deepopt.checkpoint import _loader_generators
from deepopt.data import SkipBatchSampler


def make_components(wrapped):
    torch.manual_seed(11)
    inputs = torch.arange(12, dtype=torch.float32).reshape(-1, 1) / 12
    dataset = TensorDataset(inputs, inputs * 2)
    def loader(seed):
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
        batches = BatchSampler(sampler, 3, drop_last=False)
        if wrapped:
            batches = SkipBatchSampler(SkipBatchSampler(batches))
        return DataLoader(dataset, batch_sampler=batches, generator=torch.Generator().manual_seed(seed + 1))
    model = nn.Linear(1, 1)
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=0.03), loss_fn=F.mse_loss)
    return trainer, loader(23), loader(37)


class CheckpointGeneratorTests(unittest.TestCase):
    def test_nested_sampler_resume_matches_order_weights_and_metrics(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped), tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
                reference, reference_loader, reference_validation = make_components(wrapped)
                rows = []
                reference.fit(reference_loader, reference_validation, epochs=2,
                              callbacks=[lambda _, row: rows.append(row)], progress_bar=False)
                trainer, loader, validation = make_components(wrapped)
                callback = Checkpoint(directory)
                trainer.fit(loader, validation, epochs=1, callbacks=[callback], progress_bar=False)
                resumed, resumed_loader, resumed_validation = make_components(wrapped)
                restored = Checkpoint(directory)
                restored.restore(resumed, resumed_loader, resumed_validation)
                self.assertEqual(len(restored.generators), 4)  # Worker + sampling RNGs for both phases.
                resumed.fit(resumed_loader, resumed_validation, epochs=2,
                            callbacks=[restored], progress_bar=False)
                for actual, expected in zip(resumed.model.parameters(), reference.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(restored.history[-1]["train"]["loss"], rows[-1]["train"]["loss"])
                self.assertEqual(restored.history[-1]["validation"], rows[-1]["validation"])
                for actual, expected in zip(resumed_loader, reference_loader):
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)

    def test_discovery_deduplicates_cycles_and_preserves_existing_names_without_rng_draws(self):
        shared = torch.Generator().manual_seed(5)
        nested = torch.Generator().manual_seed(7)
        before = [generator.get_state().clone() for generator in (shared, nested)]
        leaf = SimpleNamespace(generator=nested)
        leaf.sampler = leaf
        batch_sampler = SimpleNamespace(sampler=leaf, generator=shared)
        train = SimpleNamespace(generator=shared, sampler=None, batch_sampler=batch_sampler)
        validation = SimpleNamespace(generator=nested, sampler=None, batch_sampler=batch_sampler)
        result = _loader_generators(train, validation)
        # Previously registered validation RNGs retain their key even if newly
        # discovered through the training sampler, keeping old snapshots usable.
        self.assertEqual(result, {"train": shared, "validation": nested})
        for generator, state in zip((shared, nested), before):
            torch.testing.assert_close(generator.get_state(), state, rtol=0, atol=0)

    def test_discovery_does_not_fetch_data_or_iterate_sampler(self):
        class Sampler:
            generator = torch.Generator().manual_seed(9)
            def __iter__(self):
                raise AssertionError("Discovery must not iterate")
        class Dataset:
            @property
            def generator(self):
                raise AssertionError("Discovery must not inspect dataset internals")
        loader = SimpleNamespace(batch_sampler=SimpleNamespace(sampler=Sampler()), dataset=Dataset())
        self.assertEqual(_loader_generators(loader, None), {"train.batch_sampler.sampler": Sampler.generator})

    def test_old_missing_nested_rng_is_rejected_instead_of_silently_replaying(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            trainer, loader, validation = make_components(False)
            callback = Checkpoint(directory)
            trainer.fit(loader, validation, epochs=1, callbacks=[callback], progress_bar=False)
            path = Path(directory) / "last.pt"
            payload = torch.load(path, weights_only=True)
            payload["generators"].pop("train.batch_sampler.sampler")
            torch.save(payload, path)
            resumed, resumed_loader, resumed_validation = make_components(False)
            before = deepcopy(resumed.model.state_dict())
            with self.assertRaisesRegex(ValueError, "generators"):
                Checkpoint(directory).restore(resumed, resumed_loader, resumed_validation)
            for name, value in resumed.model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
