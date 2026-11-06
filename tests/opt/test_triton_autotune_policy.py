"""Exercise the real host-side tuning policy without importing a GPU backend."""

from itertools import product
from types import SimpleNamespace
import unittest

from deepopt.opt.backends._triton_autotune import (
    AUTOTUNE_KEY, AUTOTUNE_TILES, MAX_BLOCK_M, MAX_BLOCK_N, MAX_BLOCK_K,
    prune_gemm_configs, set_descriptor_block_shapes,
)


class TritonAutotunePolicyTests(unittest.TestCase):
    def setUp(self):
        self.configs = [
            SimpleNamespace(kwargs={"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=warps, num_stages=stages)
            for bm, bn, bk, warps in AUTOTUNE_TILES
            for stages in (1, 2, 3, 4)
        ]

    def test_96_wide_scales_reject_reductions_that_cross_scale_boundaries(self):
        for a_width, b_width in ((96, 128), (128, 96), (96, 96)):
            with self.subTest(widths=(a_width, b_width)):
                args = {"A_BLOCK_COLS": a_width, "B_BLOCK_COLS": b_width,
                        "NATIVE_FP8": True, "PIPELINE_STAGES": 3}
                original = dict(args)
                selected = prune_gemm_configs(self.configs, args)
                self.assertTrue(selected)
                self.assertEqual({config.kwargs["BLOCK_K"] for config in selected}, {32})
                self.assertEqual(args, original)

    def test_explicit_pipeline_stages_are_preserved(self):
        for stages in (1, 2, 3, 4):
            with self.subTest(stages=stages):
                selected = prune_gemm_configs(
                    self.configs, {"PIPELINE_STAGES": stages, "NATIVE_FP8": True},
                    A_BLOCK_COLS=128, B_BLOCK_COLS=128,
                )
                self.assertEqual(len(selected), len(AUTOTUNE_TILES))
                self.assertEqual({config.num_stages for config in selected}, {stages})

    def test_tensorwise_padding_keeps_all_compatible_tile_choices(self):
        for k in (1, 31, 97, 128, 129, 257):
            padded_k = ((k + MAX_BLOCK_K - 1) // MAX_BLOCK_K) * MAX_BLOCK_K
            with self.subTest(k=k, padded_k=padded_k):
                selected = prune_gemm_configs(
                    self.configs, {"PIPELINE_STAGES": 2, "NATIVE_FP8": True,
                                   "TENSORWISE": True, "A_BLOCK_COLS": padded_k,
                                   "B_BLOCK_COLS": padded_k},
                )
                self.assertEqual(len(selected), len(AUTOTUNE_TILES))
                mixed = prune_gemm_configs(
                    self.configs, {"PIPELINE_STAGES": 2, "NATIVE_FP8": True,
                                   "TENSORWISE": False, "A_BLOCK_COLS": padded_k,
                                   "B_BLOCK_COLS": 96},
                )
                self.assertEqual({config.kwargs["BLOCK_K"] for config in mixed}, {32})

    def test_unaligned_scale_fallback_keeps_small_tiles_and_requested_stages(self):
        for stages in (1, 2, 3, 4):
            selected = prune_gemm_configs(
                self.configs, {"PIPELINE_STAGES": stages, "NATIVE_FP8": False,
                               "A_BLOCK_COLS": 1, "B_BLOCK_COLS": 9},
            )
            self.assertGreaterEqual(len(selected), 2)
            for config in selected:
                self.assertLessEqual(config.kwargs["BLOCK_M"], 64)
                self.assertLessEqual(config.kwargs["BLOCK_N"], 64)
                self.assertEqual(config.kwargs["BLOCK_K"], 32)
                self.assertEqual(config.num_stages, stages)

    def test_descriptor_hook_updates_each_candidate_and_fresh_cached_call(self):
        descriptors = {name: SimpleNamespace(block_shape=[16, 16], base=object())
                       for name in ("a_desc", "b_desc", "c_desc")}
        bases = {name: descriptor.base for name, descriptor in descriptors.items()}
        for bm, bn, bk in ((64, 128, 32), (128, 64, 64), (128, 128, 128), (64, 128, 32)):
            set_descriptor_block_shapes({**descriptors, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
            self.assertEqual(descriptors["a_desc"].block_shape, [bm, bk])
            self.assertEqual(descriptors["b_desc"].block_shape, [bn, bk])
            self.assertEqual(descriptors["c_desc"].block_shape, [bm, bn])
            for name, descriptor in descriptors.items():
                self.assertIs(descriptor.base, bases[name])

        fresh = {name: SimpleNamespace(block_shape=[16, 16]) for name in descriptors}
        set_descriptor_block_shapes({**fresh, "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64})
        self.assertEqual(fresh["a_desc"].block_shape, [128, 64])
        self.assertEqual(fresh["b_desc"].block_shape, [64, 64])
        self.assertEqual(fresh["c_desc"].block_shape, [128, 64])
        self.assertEqual(descriptors["a_desc"].block_shape, [64, 32])

    def test_cache_key_covers_format_device_and_pruning_policy(self):
        required = {
            "M", "N", "K", "A_FP8_FORMAT", "B_FP8_FORMAT", "OUTPUT_DTYPE",
            "A_BLOCK_ROWS", "A_BLOCK_COLS", "B_BLOCK_ROWS", "B_BLOCK_COLS",
            "A_SCALE_COLS", "B_SCALE_COLS", "TENSORWISE", "NATIVE_FP8", "HAS_BIAS",
            "PIPELINE_STAGES", "DEVICE_INDEX", "DEVICE_CAPABILITY",
        }
        self.assertTrue(required.issubset(AUTOTUNE_KEY))
        self.assertEqual(len(AUTOTUNE_KEY), len(set(AUTOTUNE_KEY)))
        self.assertTrue({"BLOCK_M", "BLOCK_N", "BLOCK_K"}.isdisjoint(AUTOTUNE_KEY))

    def test_all_tiles_fit_common_padding_and_descriptor_alignment(self):
        self.assertEqual((MAX_BLOCK_M, MAX_BLOCK_N, MAX_BLOCK_K),
                         tuple(max(tile[axis] for tile in AUTOTUNE_TILES) for axis in range(3)))
        for bm, bn, bk, warps in AUTOTUNE_TILES:
            with self.subTest(tile=(bm, bn, bk, warps)):
                for size, largest in zip((bm, bn, bk), (MAX_BLOCK_M, MAX_BLOCK_N, MAX_BLOCK_K)):
                    self.assertGreater(size, 0)
                    self.assertEqual(size & (size - 1), 0)
                    self.assertLessEqual(size, 256)
                    self.assertEqual(largest % size, 0)
                self.assertGreaterEqual(bm, 16)
                self.assertGreaterEqual(bn, 16)
                self.assertGreaterEqual(bk, 32)
                self.assertEqual(bk % 16, 0)  # FP8 operand byte alignment.
                for output_bytes in (2, 4):
                    self.assertEqual((bn * output_bytes) % 16, 0)
                self.assertIn(warps, (4, 8))

    def test_accepted_scale_layouts_always_leave_a_configuration(self):
        widths = (32, 64, 96, 128, 160, 224, 256, 288, 512)
        for a_width, b_width, stages in product(widths, widths, (1, 2, 3, 4)):
            selected = prune_gemm_configs(
                self.configs, {"A_BLOCK_COLS": a_width, "B_BLOCK_COLS": b_width,
                               "NATIVE_FP8": True, "PIPELINE_STAGES": stages},
            )
            self.assertTrue(selected, (a_width, b_width, stages))
            for config in selected:
                self.assertEqual(a_width % config.kwargs["BLOCK_K"], 0)
                self.assertEqual(b_width % config.kwargs["BLOCK_K"], 0)
        for a_width, b_width in product((1, 5, 17, 31, 48, 63, 127), repeat=2):
            self.assertTrue(prune_gemm_configs(
                self.configs, {"A_BLOCK_COLS": a_width, "B_BLOCK_COLS": b_width,
                               "NATIVE_FP8": False, "PIPELINE_STAGES": 3},
            ))


if __name__ == "__main__":
    unittest.main()
