import unittest

import torch
from torch.utils.data import Dataset

from deepopt.data import ResumableDataLoader, build_loaders


class IndexData(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def make_loader(size=23, batch_size=4, workers=2, **options):
    worker_options = dict(multiprocessing_context="spawn", timeout=20, prefetch_factor=4) if workers else {}
    worker_options.update(options)
    return ResumableDataLoader(IndexData(size), batch_size=batch_size, num_workers=workers,
                               generator=torch.Generator().manual_seed(17), collate_fn=list,
                               **worker_options)


def flatten(batches):
    return [index for batch in batches for index in batch]


def compact_collate(batch):
    return {"first": batch[0]}


def failing_collate(batch):
    raise ValueError("intentional collation failure")


class ResumableWorkerTests(unittest.TestCase):
    def test_factory_epoch_resume_preserves_persistent_loader_generators(self):
        for resumable in (False, True):
            with self.subTest(resumable=resumable):
                def create():
                    return build_loaders(IndexData(9), IndexData(5), batch_size=4, seed=17,
                                         num_workers=2, resumable=resumable, collate_fn=list,
                                         multiprocessing_context="spawn", persistent_workers=True, timeout=20)

                train, valid, generators = create()
                list(train)
                list(valid)
                saved = {name: generator.get_state() for name, generator in generators.items()}
                expected_train, expected_valid = list(train), list(valid)
                resumed_train, resumed_valid, resumed_generators = create()
                for name, generator in resumed_generators.items():
                    generator.set_state(saved[name])
                self.assertEqual(list(resumed_train), expected_train)
                self.assertEqual(list(resumed_valid), expected_valid)
                for name in generators:
                    torch.testing.assert_close(resumed_generators[name].get_state(),
                                               generators[name].get_state(), rtol=0, atol=0)

    def test_factory_supports_step_and_epoch_resume_with_and_without_workers(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                options = dict(multiprocessing_context="spawn", prefetch_factor=3,
                               persistent_workers=True, timeout=20) if workers else {}
                loader, validation, generators = build_loaders(
                    IndexData(23), IndexData(5), batch_size=4, seed=17,
                    num_workers=workers, resumable=True, collate_fn=list, **options)
                self.assertIsInstance(loader, ResumableDataLoader)
                self.assertIs(loader.generator, generators["train"])
                self.assertIs(validation.generator, generators["validation"])
                self.assertEqual(list(validation), [[0, 1, 2, 3], [4]])
                iterator = iter(loader)
                next(iterator)
                saved = loader.state_dict()
                self.assertEqual(saved["sampler"]["position"], 4)
                expected = list(iterator)

                restored, validation, _ = build_loaders(
                    IndexData(23), None, batch_size=4, seed=99,
                    num_workers=workers, resumable=True, collate_fn=list, **options)
                self.assertIsNone(validation)
                restored.load_state_dict(saved)
                self.assertEqual(list(restored), expected)

                # Epoch checkpoints restore generator state and reset the cursor.
                epoch_rng = generators["train"].get_state()
                restored.generator.set_state(epoch_rng)
                restored.reset_epoch()
                self.assertEqual(list(restored), list(loader))

    def test_cursor_does_not_depend_on_collated_batch_shape(self):
        loader = ResumableDataLoader(IndexData(23), batch_size=4, num_workers=2,
                                     generator=torch.Generator().manual_seed(17), collate_fn=compact_collate,
                                     multiprocessing_context="spawn", timeout=20)
        iterator = iter(loader)
        next(iterator)
        saved = loader.state_dict()
        self.assertEqual(saved["sampler"]["position"], 4)
        list(iterator)
        restored = make_loader(workers=0)
        restored.load_state_dict(saved)
        self.assertEqual(flatten(restored), saved["sampler"]["order"][4:].tolist())

    def test_worker_error_does_not_commit_failed_or_prefetched_batches(self):
        loader = ResumableDataLoader(IndexData(23), batch_size=4, num_workers=2,
                                     generator=torch.Generator().manual_seed(17), collate_fn=failing_collate,
                                     multiprocessing_context="spawn", timeout=20)
        iterator = iter(loader)
        with self.assertRaisesRegex(ValueError, "intentional collation failure"):
            next(iterator)
        saved = loader.state_dict()
        self.assertEqual(saved["sampler"]["position"], 0)
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            next(iterator)
        restored = make_loader(workers=0)
        restored.load_state_dict(saved)
        self.assertEqual(flatten(restored), saved["sampler"]["order"].tolist())

    def test_prefetch_snapshot_commits_only_delivered_batches(self):
        original = make_loader()
        iterator = iter(original)
        first = next(iterator)
        saved = original.state_dict()
        self.assertEqual(saved["sampler"]["position"], 4)
        self.assertGreater(original.batch_sampler.position, 4)
        expected = list(iterator)
        self.assertEqual(first + flatten(expected), saved["sampler"]["order"].tolist())

        restored = make_loader()
        restored.load_state_dict(saved)
        self.assertEqual(list(restored), expected)
        torch.testing.assert_close(restored.generator.get_state(), original.generator.get_state(), rtol=0, atol=0)
        self.assertEqual(list(restored), list(original))

    def test_snapshot_before_first_delivery_and_changed_worker_count(self):
        original = make_loader()
        iterator = iter(original)
        saved = original.state_dict()
        self.assertEqual(saved["sampler"]["position"], 0)
        restored = make_loader(workers=0)
        restored.load_state_dict(saved)
        self.assertEqual(list(restored), list(iterator))
        self.assertEqual(list(restored), list(original))

    def test_repeated_rebatch_and_worker_changes_preserve_remaining_samples(self):
        original = make_loader()
        iterator = iter(original)
        consumed = next(iterator)
        saved = original.state_dict()
        list(iterator)
        resized = make_loader(batch_size=6, workers=1, prefetch_factor=1)
        resized.load_state_dict(saved)
        iterator = iter(resized)
        consumed += next(iterator)
        second = resized.state_dict()
        self.assertEqual(second["sampler"]["position"], 10)
        self.assertEqual(second["sampler"]["batch_origin"], 4)
        list(iterator)
        final = make_loader(batch_size=3, workers=0)
        final.load_state_dict(second)
        consumed += flatten(final)
        self.assertEqual(consumed, saved["sampler"]["order"].tolist())

    def test_live_resize_discards_prefetched_batches(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                loader = make_loader(persistent_workers=persistent)
                iterator = iter(loader)
                first = next(iterator)
                saved = loader.state_dict()
                loader.set_batch_size(6)
                resized = loader.state_dict()
                self.assertEqual(resized["sampler"]["position"], 4)
                self.assertEqual(resized["sampler"]["batch_origin"], 4)
                rest = list(iterator)
                self.assertEqual([len(batch) for batch in rest], [6, 6, 6, 1])
                self.assertEqual(first + flatten(rest), saved["sampler"]["order"].tolist())
                torch.testing.assert_close(loader.generator.get_state(), saved["generator"], rtol=0, atol=0)

    def test_restore_on_active_persistent_loader_discards_old_prefetch(self):
        loader = make_loader(persistent_workers=True)
        iterator = iter(loader)
        next(iterator)
        saved = loader.state_dict()
        next(iterator)
        loader.load_state_dict(saved)
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            next(iterator)
        self.assertEqual(flatten(loader), saved["sampler"]["order"][4:].tolist())

    def test_invalid_restore_leaves_active_prefetch_usable(self):
        loader = make_loader()
        iterator = iter(loader)
        next(iterator)
        saved = loader.state_dict()
        malformed = {**saved, "sampler": {**saved["sampler"], "position": 3}}
        with self.assertRaises(ValueError):
            loader.load_state_dict(malformed)
        self.assertEqual(flatten(iterator), saved["sampler"]["order"][4:].tolist())

    def test_persistent_resume_preserves_next_epoch_order_across_worker_settings(self):
        original = make_loader(persistent_workers=True)
        list(original)  # Exercise reuse of workers before checkpointing.
        iterator = iter(original)
        next(iterator)
        saved = original.state_dict()
        expected = list(iterator)
        restored = make_loader(workers=0)
        restored.load_state_dict(saved)
        self.assertEqual(list(restored), expected)
        self.assertEqual(list(restored), list(original))

    def test_drop_last_tail_commits_on_delivery_not_prefetch(self):
        loader = make_loader(size=11, drop_last=True)
        iterator = iter(loader)
        next(iterator)
        saved = loader.state_dict()
        self.assertEqual(saved["sampler"]["position"], 4)
        resized = make_loader(size=11, batch_size=1, workers=0, drop_last=True)
        resized.load_state_dict(saved)
        self.assertEqual(flatten(resized), saved["sampler"]["order"][4:].tolist())
        next(iterator)
        exhausted = loader.state_dict()
        self.assertEqual(exhausted["sampler"]["position"], 11)
        list(iterator)
        resized.load_state_dict(exhausted)
        self.assertEqual(list(resized), [])
        self.assertEqual(sorted(flatten(resized)), list(range(11)))

    def test_final_partial_batch_checkpoint_is_exhausted_once(self):
        loader = make_loader(size=7, persistent_workers=True)
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        saved = loader.state_dict()
        list(iterator)
        loader.load_state_dict(saved)
        self.assertEqual(list(loader), [])
        self.assertEqual(sorted(flatten(loader)), list(range(7)))

    def test_empty_and_all_dropped_epochs(self):
        for size in (0, 3):
            with self.subTest(size=size):
                loader = make_loader(size=size, drop_last=True)
                self.assertEqual(list(loader), [])
                self.assertEqual(loader.state_dict()["sampler"]["position"], size)

    def test_reset_epoch_invalidates_old_iterator(self):
        loader = make_loader(persistent_workers=True)
        iterator = iter(loader)
        next(iterator)
        loader.reset_epoch()
        self.assertIsNone(loader.state_dict()["sampler"]["order"])
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            next(iterator)
        self.assertEqual(sorted(flatten(loader)), list(range(23)))

    def test_out_of_order_delivery_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "in_order=True"):
            make_loader(in_order=False)


if __name__ == "__main__":
    unittest.main()
