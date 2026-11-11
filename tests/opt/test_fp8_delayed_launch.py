"""Valid-chunk launch planning, metadata ownership, and retained scratch accounting."""

from bisect import bisect_right
import unittest
from types import SimpleNamespace

import torch

from deepopt.opt.backends.triton.quantization.launch import (observation_rows, scratch_tile, MetadataCache,
                                                  conversion_cache_key, retained_bytes)
from deepopt.opt.scaling.delayed import DelayedScalingState, _UpdateBatch


def state():
    return DelayedScalingState({"history_length": 3},
                                dict.fromkeys(("activation", "weight", "grad"), torch.float8_e4m3fn), device="cpu")


class DelayedLaunchTests(unittest.TestCase):
    def test_flat_observation_programs_cover_only_valid_chunks(self):
        sizes = [1, 1024, 1025, 290000, 64]
        values = [(group, torch.empty(size)) for group, size in enumerate(sizes)]
        rows, programs = observation_rows(values)
        expected = sum((size + 1023) // 1024 for size in sizes)
        self.assertEqual(programs, expected)
        ends = [row[3] for row in rows]
        covered = [0] * len(sizes)
        for pid in range(programs):
            observation = bisect_right(ends, pid)
            first = ends[observation - 1] if observation else 0
            offset = (pid - first) * 1024
            self.assertLess(offset, sizes[observation])
            covered[observation] += 1
        self.assertEqual(covered, [(size + 1023) // 1024 for size in sizes])
        self.assertLess(programs, max(covered) * len(sizes))

    def test_scratch_uses_cached_or_pinned_tile_with_conservative_cold_capacity(self):
        configs = [SimpleNamespace(kwargs={"BM": bm, "BN": bn}) for bm, bn in ((32, 32), (64, 64), (32, 128))]
        kernel = SimpleNamespace(configs=configs, cache={})
        self.assertEqual(scratch_tile(kernel, "key"), (32, 32))
        kernel.cache["key"] = configs[1]
        self.assertEqual(scratch_tile(kernel, "key"), (64, 64))
        kernel.configs = [configs[2]]
        self.assertEqual(scratch_tile(kernel, "key"), (32, 128))

    def test_conversion_key_matches_explicit_keys_then_pointer_dtype_order(self):
        x, scale = torch.empty(65, 129), torch.ones(1)
        key = conversion_cache_key(x, scale, torch.float8_e4m3fn, normal=True, transposed=True,
                                   bias=True, observe=True, check_finite=False)
        self.assertEqual(key[:10], (65, 129, 129, 1, True, True, True, True, False, None))
        self.assertEqual(key[10:], tuple(map(str, (torch.float32, torch.float32, torch.float8_e4m3fn,
                                                  torch.float8_e4m3fn, *([torch.float32] * 4)))))

    def test_state_pointer_metadata_reuses_tables_and_invalidates_on_move_or_load(self):
        owner, cache, device = state(), MetadataCache(), torch.device("cpu")
        first = cache.state_table([owner], device)
        self.assertIs(cache.state_table([owner], device), first)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        owner.bfloat16()
        second = cache.state_table([owner], device)
        self.assertIsNot(second, first)
        owner.load_state_dict(owner.state_dict())
        self.assertIsNot(cache.state_table([owner], device), second)
        self.assertEqual(cache.misses, 3)

    def test_dynamic_tables_reuse_equal_pointer_layouts_and_are_bounded(self):
        cache, device = MetadataCache(capacity=2), torch.device("cpu")
        first = cache.table(("observation", 1), lambda: [[10, 20]], device)
        self.assertIs(cache.table(("observation", 1), lambda: self.fail("rebuilt metadata"), device), first)
        for key in (2, 3):
            cache.table(("observation", key), lambda: [[key]], device)
        self.assertEqual(len(cache._tables), 2)

    def test_retained_view_storage_is_charged_once_and_triggers_compaction(self):
        owner = state()
        base = torch.zeros(9 * 1024 * 1024)
        views = [base[:1], base[1:2]]
        self.assertEqual(retained_bytes(views), base.untyped_storage().nbytes())
        batch = _UpdateBatch([owner])
        batch.record(owner, 0, views[0])
        self.assertEqual(batch.bytes, batch.observations[0][2].untyped_storage().nbytes())
        self.assertLess(batch.bytes, 1024)
        batch.finish()
        self.assertEqual(owner.updates.tolist(), [1, 0, 0])
