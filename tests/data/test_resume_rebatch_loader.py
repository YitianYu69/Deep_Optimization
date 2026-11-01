import copy
import unittest

import torch
from torch.utils.data import TensorDataset

from deepopt.data.resumable import ResumableBatchSampler, ResumableDataLoader


def make_loader(size, batch_size, seed=17, *, drop_last=False):
    return ResumableDataLoader(TensorDataset(torch.arange(size)), batch_size=batch_size,
                               generator=torch.Generator().manual_seed(seed), drop_last=drop_last)


def ids(batches):
    return [item for batch in batches for item in batch[0].tolist()]


class ResumeRebatchLoaderTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in expected.items():
            if isinstance(value, dict):
                self.assert_state_equal(actual[name], value)
            elif isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
            else:
                self.assertEqual(actual[name], value)

    def test_repeated_four_to_six_to_three_preserves_every_remaining_example(self):
        original = make_loader(23, 4)
        iterator = iter(original)
        consumed = ids([next(iterator), next(iterator)])
        first_state = original.state_dict()
        expected_order = first_state["sampler"]["order"].tolist()
        self.assertEqual(first_state["sampler"]["position"], 8)

        resized = make_loader(23, 6, seed=99)
        global_rng = torch.get_rng_state().clone()
        resized.load_state_dict(first_state)
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
        self.assertEqual(resized.batch_sampler.batch_size, 6)
        self.assertEqual(resized.batch_sampler.batch_origin, 8)
        consumed += ids([next(iter(resized))])
        second_state = resized.state_dict()
        self.assertEqual(second_state["sampler"]["position"], 14)
        self.assertEqual(second_state["sampler"]["batch_origin"], 8)
        torch.testing.assert_close(second_state["generator"], first_state["generator"], rtol=0, atol=0)

        # A same-size restore after resizing must retain the shifted grid.
        same_size = make_loader(23, 6)
        same_size.load_state_dict(second_state)
        self.assertEqual(same_size.batch_sampler.batch_origin, 8)
        self.assertEqual(same_size.batch_sampler.position, 14)

        resized_again = make_loader(23, 3)
        resized_again.load_state_dict(second_state)
        self.assertEqual(resized_again.batch_sampler.batch_origin, 14)
        consumed += ids(list(resized_again))
        self.assertEqual(consumed, expected_order)
        self.assertEqual(len(set(consumed)), 23)
        self.assertEqual(resized_again.batch_sampler.position, 23)
        torch.testing.assert_close(first_state["sampler"]["order"], torch.tensor(expected_order), rtol=0, atol=0)

    def test_legacy_v1_same_size_resume_preserves_remaining_data_and_next_epoch_rng(self):
        original = make_loader(19, 4, seed=31)
        iterator = iter(original)
        next(iterator)
        next(iterator)
        legacy = original.state_dict()
        legacy["format_version"] = 1
        legacy["sampler"].pop("batch_origin")
        expected_remaining = ids(list(iterator))

        restored = make_loader(19, 4, seed=93)
        restored.load_state_dict(legacy)
        self.assertEqual(restored.batch_sampler.batch_origin, 0)
        self.assertEqual(ids(list(restored)), expected_remaining)
        torch.testing.assert_close(restored.generator.get_state(), original.generator.get_state(), rtol=0, atol=0)
        self.assertEqual(ids(list(restored)), ids(list(original)))
        torch.testing.assert_close(restored.generator.get_state(), original.generator.get_state(), rtol=0, atol=0)

    def test_standalone_legacy_sampler_uses_saved_size_for_boundary_validation(self):
        original = ResumableBatchSampler(17, 4, torch.Generator().manual_seed(7))
        iterator = iter(original)
        next(iterator)
        next(iterator)
        legacy = original.state_dict()
        legacy.pop("batch_origin")
        restored = ResumableBatchSampler(17, 6, torch.Generator().manual_seed(8))
        restored.load_state_dict(legacy)
        self.assertEqual(restored.position, 8)
        self.assertEqual(restored.batch_origin, 8)
        self.assertEqual(next(iter(restored)), legacy["order"][8:14].tolist())

    def test_exhausted_resized_loader_yields_nothing_once_then_starts_a_new_epoch(self):
        original = make_loader(11, 4)
        self.assertEqual(len(ids(list(original))), 11)
        state = original.state_dict()
        restored = make_loader(11, 6)
        restored.load_state_dict(state)
        self.assertEqual(restored.batch_sampler.position, 11)
        self.assertEqual(restored.batch_sampler.batch_origin, 11)
        self.assertEqual(list(restored), [])
        torch.testing.assert_close(restored.generator.get_state(), state["generator"], rtol=0, atol=0)
        batches = list(restored)
        self.assertEqual([len(batch[0]) for batch in batches], [6, 5])
        self.assertEqual(sorted(ids(batches)), list(range(11)))
        self.assertEqual(restored.batch_sampler.batch_origin, 0)

    def test_batch_larger_than_remaining_data_does_not_drop_the_partial_batch(self):
        original = make_loader(13, 4)
        iterator = iter(original)
        consumed = ids([next(iterator), next(iterator)])
        state = original.state_dict()
        restored = make_loader(13, 50)
        restored.load_state_dict(state)
        remaining = list(restored)
        self.assertEqual([len(batch[0]) for batch in remaining], [5])
        self.assertEqual(consumed + ids(remaining), state["sampler"]["order"].tolist())
        last = make_loader(13, 3)
        last.load_state_dict(restored.state_dict())
        self.assertEqual(list(last), [])

    def test_inactive_state_and_epoch_reset_start_at_origin_zero(self):
        original = make_loader(13, 4)
        state = original.state_dict()
        restored = make_loader(13, 6, seed=88)
        restored.load_state_dict(state)
        self.assertEqual(restored.batch_sampler.position, 0)
        self.assertEqual(restored.batch_sampler.batch_origin, 0)
        self.assertFalse(restored._resume_iterator)
        self.assertEqual(ids(list(restored)), ids(list(original)))

        resumed = make_loader(13, 3)
        resumed.load_state_dict(restored.state_dict())
        self.assertEqual(resumed.batch_sampler.batch_origin, 13)
        resumed.reset_epoch()
        self.assertIsNone(resumed.batch_sampler.order)
        self.assertEqual(resumed.batch_sampler.position, 0)
        self.assertEqual(resumed.batch_sampler.batch_origin, 0)
        self.assertFalse(resumed._resume_iterator)
        self.assertEqual(sorted(ids(list(resumed))), list(range(13)))

    def test_malformed_sampler_metadata_is_rejected_before_any_mutation(self):
        original = make_loader(20, 4)
        iterator = iter(original)
        next(iterator)
        next(iterator)
        state = original.batch_sampler.state_dict()
        invalid_fields = (
            {"size": 21}, {"size": True}, {"batch_size": 0}, {"batch_size": True},
            {"batch_origin": -1}, {"batch_origin": 9}, {"batch_origin": True},
            {"position": 7}, {"position": 21}, {"position": True},
            {"order": torch.zeros(20, dtype=torch.long)}, {"order": torch.arange(20).float()},
            {"order": torch.empty(20, dtype=torch.long, device="meta")},
            {"order": None}, {"order": None, "position": 0, "batch_origin": 1},
        )
        target = make_loader(20, 6)
        next(iter(target))
        before = target.batch_sampler.state_dict()
        for changed in invalid_fields:
            with self.subTest(changed=tuple(changed)):
                malformed = {**state, **changed}
                with self.assertRaises(ValueError):
                    target.batch_sampler.load_state_dict(malformed)
                self.assert_state_equal(target.batch_sampler.state_dict(), before)
        for malformed in (None, {}, {"size": 20}):
            with self.assertRaises(ValueError):
                target.batch_sampler.load_state_dict(malformed)
            self.assert_state_equal(target.batch_sampler.state_dict(), before)

    def test_malformed_loader_schema_or_rng_does_not_mutate_sampler_or_generator(self):
        original = make_loader(20, 4)
        iterator = iter(original)
        next(iterator)
        next(iterator)
        state = original.state_dict()
        missing_origin = copy.deepcopy(state)
        missing_origin["sampler"].pop("batch_origin")
        invalid = (missing_origin, {**state, "format_version": True}, {**state, "format_version": 4},
                   {**state, "sampler": None}, {**state, "generator": None},
                   {**state, "generator": torch.zeros(3, dtype=torch.uint8)},
                   {**state, "generator": torch.zeros(3, dtype=torch.long)})
        target = make_loader(20, 6, seed=72)
        next(iter(target))
        before, resume_iterator = target.state_dict(), target._resume_iterator
        for malformed in invalid:
            with self.subTest(version=malformed["format_version"]):
                with self.assertRaises(ValueError):
                    target.load_state_dict(malformed)
                self.assert_state_equal(target.state_dict(), before)
                self.assertEqual(target._resume_iterator, resume_iterator)

    def test_capability_and_new_snapshot_schema_are_explicit(self):
        loader = make_loader(2, 1)
        self.assertIs(loader.supports_batch_size_change, True)
        state = loader.state_dict()
        self.assertEqual(state["format_version"], 3)
        self.assertEqual(state["sampler"]["batch_origin"], 0)
        self.assertEqual(loader.batch_sampler.batch_size, 1)

    def test_drop_last_resume_and_rebatch_keep_remaining_full_batches(self):
        original = make_loader(15, 4, drop_last=True)
        iterator = iter(original)
        consumed = ids([next(iterator)])
        saved = original.state_dict()
        self.assertEqual(len(original), 3)
        self.assertTrue(original.drop_last)
        resized = make_loader(15, 3, drop_last=True)
        resized.load_state_dict(saved)
        batches = list(resized)
        self.assertEqual([len(batch[0]) for batch in batches], [3, 3, 3])
        self.assertEqual(consumed + ids(batches), saved["sampler"]["order"][:13].tolist())
        self.assertEqual(resized.batch_sampler.position, 15)

    def test_drop_last_final_yield_commits_tail_before_iterator_exhaustion(self):
        original = make_loader(11, 4, drop_last=True)
        iterator = iter(original)
        next(iterator)
        next(iterator)  # Checkpoint here, without asking the iterator to stop.
        saved = original.state_dict()
        self.assertEqual(saved["sampler"]["position"], 11)
        restored = make_loader(11, 1, drop_last=True)
        restored.load_state_dict(saved)
        self.assertEqual(list(restored), [])  # Do not resurrect the discarded 3.
        self.assertEqual(sorted(ids(list(restored))), list(range(11)))

    def test_drop_last_empty_and_exactly_divisible_epochs(self):
        for size, expected in ((0, 0), (3, 0), (8, 2)):
            with self.subTest(size=size):
                loader = make_loader(size, 4, drop_last=True)
                self.assertEqual(len(loader), expected)
                self.assertEqual(len(list(loader)), expected)
                self.assertEqual(loader.batch_sampler.position, size)

    def test_drop_last_snapshot_validation_and_legacy_defaults(self):
        loader = make_loader(11, 4, drop_last=True)
        next(iter(loader))
        saved = loader.state_dict()
        target = make_loader(11, 3, drop_last=True)
        before = target.state_dict()
        for value in (False, 1, None):
            malformed = copy.deepcopy(saved)
            malformed["sampler"]["drop_last"] = value
            with self.assertRaisesRegex(ValueError, "drop_last"):
                target.load_state_dict(malformed)
            self.assert_state_equal(target.state_dict(), before)
        missing = copy.deepcopy(saved)
        missing["sampler"].pop("drop_last")
        with self.assertRaisesRegex(ValueError, "drop_last"):
            target.load_state_dict(missing)
        for version in (1, 2):
            legacy = make_loader(11, 4).state_dict()
            legacy["format_version"] = version
            legacy["sampler"].pop("drop_last")
            if version == 1:
                legacy["sampler"].pop("batch_origin")
            make_loader(11, 4).load_state_dict(legacy)
            with self.assertRaisesRegex(ValueError, "drop_last"):
                target.load_state_dict(legacy)
        for invalid in (1, None, "yes"):
            with self.assertRaisesRegex(ValueError, "drop_last"):
                make_loader(11, 4, drop_last=invalid)

    def test_set_batch_size_regroups_active_iterator_and_snapshot_restore(self):
        loader = make_loader(23, 4)
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        saved = loader.state_dict()
        loader.set_batch_size(6)
        self.assertEqual(loader.batch_sampler.position, 8)
        self.assertEqual(loader.batch_sampler.batch_origin, 8)
        torch.testing.assert_close(loader.generator.get_state(), saved["generator"], rtol=0, atol=0)
        self.assertEqual(ids([next(iterator)]), saved["sampler"]["order"][8:14].tolist())
        loader.set_batch_size(6)
        self.assertEqual(loader.batch_sampler.batch_origin, 8)

        # Restore consumed data/RNG while retaining the selected candidate size.
        loader.load_state_dict(saved)
        self.assertEqual(loader.batch_sampler.batch_size, 6)
        self.assertEqual(loader.batch_sampler.position, 8)
        self.assertEqual(loader.batch_sampler.batch_origin, 8)
        self.assertEqual(ids([next(iter(loader))]), saved["sampler"]["order"][8:14].tolist())
        self.assertIsNone(loader.batch_size)  # DataLoader's fixed constructor field is not changed.

    def test_set_batch_size_between_epochs_and_invalid_sizes_preserves_state(self):
        loader = make_loader(11, 4)
        initial_rng = loader.generator.get_state().clone()
        loader.set_batch_size(6)
        self.assertIsNone(loader.batch_sampler.order)
        self.assertEqual(loader.batch_sampler.batch_origin, 0)
        torch.testing.assert_close(loader.generator.get_state(), initial_rng, rtol=0, atol=0)
        self.assertEqual([len(batch[0]) for batch in loader], [6, 5])

        exhausted = loader.state_dict()
        loader.set_batch_size(3)
        self.assertEqual(loader.batch_sampler.position, 11)
        self.assertEqual(loader.batch_sampler.batch_origin, 11)
        torch.testing.assert_close(loader.batch_sampler.order, exhausted["sampler"]["order"], rtol=0, atol=0)
        torch.testing.assert_close(loader.generator.get_state(), exhausted["generator"], rtol=0, atol=0)
        before = loader.state_dict()
        for size in (0, -1, True, 1.5, None):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "positive integer"):
                loader.set_batch_size(size)
            self.assert_state_equal(loader.state_dict(), before)
        self.assertEqual([len(batch[0]) for batch in loader], [3, 3, 3, 2])
        self.assertEqual(loader.batch_sampler.batch_origin, 0)


if __name__ == "__main__":
    unittest.main()
