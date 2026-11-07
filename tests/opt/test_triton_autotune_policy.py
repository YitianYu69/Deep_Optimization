"""Exercise the real host-side tuning policy without importing a GPU backend."""

from itertools import product
from types import SimpleNamespace
import unittest

from deepopt.opt.backends._triton_autotune import (
    AUTOTUNE_KEY, AUTOTUNE_TILES, MAX_BLOCK_M, MAX_BLOCK_N, MAX_BLOCK_K,
    MAX_TUNING_CONFIGS, SPLIT_K_CHOICES, baseline_gemm_tile, prune_gemm_configs,
    autotune_cache_key, set_descriptor_block_shapes, split_k_enabled,
)


def arguments(**updates):
    return {"M": 1024, "N": 1024, "K": 512,
            "A_BLOCK_ROWS": 32, "A_BLOCK_COLS": 128,
            "B_BLOCK_ROWS": 32, "B_BLOCK_COLS": 128,
            "NATIVE_FP8": True, "TENSORWISE": False, "PIPELINE_STAGES": 3, "NUM_SMS": 132,
            **updates}


def tile(config):
    return (*(config.kwargs[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K")),
            config.num_warps)


class TritonAutotunePolicyTests(unittest.TestCase):
    def setUp(self):
        self.configs = [
            SimpleNamespace(kwargs={"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                                    "NUM_STAGES": stages, "SPLIT_K": split},
                            num_warps=warps, num_stages=stages)
            for bm, bn, bk, warps in AUTOTUNE_TILES
            for stages in (1, 2, 3, 4)
            for split in SPLIT_K_CHOICES
        ]

    def test_classifier_search_measures_narrow_tiles_and_original_baseline(self):
        for batch in (2048, 848, 1808):
            cases = (
                (arguments(M=batch, N=10, K=512, A_BLOCK_ROWS=1,
                           A_BLOCK_COLS=32, B_BLOCK_COLS=32), 1, (128, 128, 32, 8)),
                (arguments(M=batch, N=512, K=10, A_BLOCK_COLS=32,
                           B_BLOCK_COLS=32), None, (128, 128, 32, 8)),
                (arguments(M=10, N=512, K=batch, A_BLOCK_COLS=32,
                           B_BLOCK_COLS=1, NATIVE_FP8=False), 0, (64, 64, 32, 4)),
            )
            for args, skinny_axis, baseline in cases:
                with self.subTest(shape=(args["M"], args["N"], args["K"])):
                    selected = prune_gemm_configs(self.configs, args)
                    tiles = [tile(config) for config in selected]
                    self.assertIn(baseline, tiles)
                    self.assertGreater(len(tiles), 1)
                    self.assertLessEqual(len(tiles), MAX_TUNING_CONFIGS)
                    if skinny_axis is not None:
                        alternatives = [candidate for candidate in tiles if candidate != baseline]
                        self.assertTrue({16, 32}.issubset(
                            {candidate[skinny_axis] for candidate in alternatives}))
                        self.assertTrue(any(candidate[skinny_axis] == 64 for candidate in tiles))
                    else:
                        self.assertEqual({candidate[2] for candidate in tiles}, {32})

    def test_large_shapes_keep_reduction_and_output_tile_diversity(self):
        selected = prune_gemm_configs(self.configs, arguments())
        tiles = [tile(config) for config in selected]
        self.assertLessEqual(len(tiles), MAX_TUNING_CONFIGS)
        self.assertIn((128, 128, 64, 8), tiles)
        self.assertEqual({candidate[2] for candidate in tiles}, {32, 64, 128})
        self.assertTrue({(64, 64), (64, 128), (128, 64), (128, 128)}
                        .issubset({candidate[:2] for candidate in tiles}))

    def test_96_wide_scales_never_cross_scale_boundaries(self):
        for a_width, b_width in ((96, 128), (128, 96), (96, 96)):
            args = arguments(A_BLOCK_COLS=a_width, B_BLOCK_COLS=b_width)
            original = dict(args)
            selected = prune_gemm_configs(self.configs, args)
            self.assertEqual({config.kwargs["BLOCK_K"] for config in selected}, {32})
            self.assertIn((128, 128, 32, 8), [tile(config) for config in selected])
            self.assertEqual(args, original)

    def test_stage_constraint_and_named_keyword_merging_are_preserved(self):
        for stages in (1, 2, 3, 4):
            selected = prune_gemm_configs(self.configs, arguments(), PIPELINE_STAGES=stages)
            self.assertGreater(len(selected), 1)
            self.assertLessEqual(len(selected), MAX_TUNING_CONFIGS)
            self.assertEqual({config.num_stages for config in selected}, {stages})

    def test_short_reductions_drop_oversized_tiles_but_measure_baseline(self):
        for k in (1, 10, 31):
            selected = prune_gemm_configs(self.configs, arguments(K=k, TENSORWISE=True))
            tiles = [tile(config) for config in selected]
            self.assertIn((128, 128, 64, 8), tiles)
            self.assertTrue(all(candidate[2] == 32 for candidate in tiles
                                if candidate != (128, 128, 64, 8)))

    def test_tensorwise_scale_extent_keeps_compatible_reduction_choices(self):
        for k in (1, 31, 97, 128, 129, 257):
            scale_extent = ((k + MAX_BLOCK_K - 1) // MAX_BLOCK_K) * MAX_BLOCK_K
            args = arguments(K=k, TENSORWISE=True, A_BLOCK_COLS=scale_extent,
                             B_BLOCK_COLS=scale_extent)
            selected = prune_gemm_configs(self.configs, args)
            self.assertGreater(len(selected), 1)
            for config in selected:
                self.assertEqual(scale_extent % config.kwargs["BLOCK_K"], 0)
            mixed = prune_gemm_configs(self.configs, {**args, "TENSORWISE": False,
                                                      "B_BLOCK_COLS": 96})
            self.assertEqual({config.kwargs["BLOCK_K"] for config in mixed}, {32})

    def test_fallback_search_preserves_resource_limits_and_baseline(self):
        for m, n, stages in product((10, 48, 128), (10, 48, 128), (1, 2, 3, 4)):
            selected = prune_gemm_configs(self.configs, arguments(
                M=m, N=n, PIPELINE_STAGES=stages, NATIVE_FP8=False,
                A_BLOCK_COLS=1, B_BLOCK_COLS=9,
            ))
            self.assertGreater(len(selected), 1)
            self.assertLessEqual(len(selected), MAX_TUNING_CONFIGS)
            self.assertIn((64, 64, 32, 4), [tile(config) for config in selected])
            for config in selected:
                bm, bn, bk, _ = tile(config)
                self.assertLessEqual(bm, 64)
                self.assertLessEqual(bn, 64)
                self.assertEqual(bk, 32)
                self.assertEqual(config.num_stages, stages)

    def test_descriptor_hook_updates_candidates_and_fresh_cached_call(self):
        descriptors = {name: SimpleNamespace(block_shape=[16, 16], base=object())
                       for name in ("a_desc", "b_desc", "c_desc")}
        bases = {name: descriptor.base for name, descriptor in descriptors.items()}
        for bm, bn, bk in ((64, 16, 32), (16, 64, 64), (128, 128, 128), (32, 16, 32)):
            set_descriptor_block_shapes({**descriptors, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
            self.assertEqual(descriptors["a_desc"].block_shape, [bm, bk])
            self.assertEqual(descriptors["b_desc"].block_shape, [bn, bk])
            self.assertEqual(descriptors["c_desc"].block_shape, [bm, bn])
            for name, descriptor in descriptors.items():
                self.assertIs(descriptor.base, bases[name])
        fresh = {name: SimpleNamespace(block_shape=[16, 16]) for name in descriptors}
        set_descriptor_block_shapes({**fresh, "BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64})
        self.assertEqual(fresh["a_desc"].block_shape, [128, 64])
        self.assertEqual(fresh["b_desc"].block_shape, [16, 64])
        self.assertEqual(fresh["c_desc"].block_shape, [128, 16])
        self.assertEqual(descriptors["a_desc"].block_shape, [32, 32])

    def test_cache_key_covers_format_device_and_pruning_policy(self):
        required = {
            "M", "N", "K", "A_FP8_FORMAT", "B_FP8_FORMAT", "OUTPUT_DTYPE",
            "A_BLOCK_ROWS", "A_BLOCK_COLS", "B_BLOCK_ROWS", "B_BLOCK_COLS",
            "A_SCALE_COLS", "B_SCALE_COLS", "TENSORWISE", "NATIVE_FP8", "HAS_BIAS",
            "PIPELINE_STAGES", "DEVICE_INDEX", "DEVICE_CAPABILITY", "NUM_SMS",
        }
        self.assertTrue(required.issubset(AUTOTUNE_KEY))
        self.assertEqual(len(AUTOTUNE_KEY), len(set(AUTOTUNE_KEY)))
        self.assertTrue({"BLOCK_M", "BLOCK_N", "BLOCK_K"}.isdisjoint(AUTOTUNE_KEY))

    def test_tiles_satisfy_descriptor_alignment_and_include_original_baselines(self):
        self.assertEqual(len(AUTOTUNE_TILES), len(set(AUTOTUNE_TILES)))
        self.assertEqual((MAX_BLOCK_M, MAX_BLOCK_N, MAX_BLOCK_K), (256, 256, 128))
        for bm, bn, bk, warps in AUTOTUNE_TILES:
            for size in (bm, bn, bk):
                self.assertGreaterEqual(size, 16)
                self.assertEqual(size & (size - 1), 0)
                self.assertLessEqual(size, 256)
            self.assertGreaterEqual(bk, 32)
            self.assertEqual(bk % 16, 0)
            for output_bytes in (2, 4):
                self.assertEqual((bn * output_bytes) % 16, 0)
            self.assertIn(warps, (2, 4, 8))
        self.assertEqual(baseline_gemm_tile(None, None), (128, 128, 64, 8))
        for a_width, b_width in product((1, 32, 64, 96, 128, 160, 256), repeat=2):
            self.assertIn(baseline_gemm_tile((1, a_width), (32, b_width)), AUTOTUNE_TILES)

    def test_supported_layouts_and_shapes_have_bounded_compatible_searches(self):
        widths = (32, 64, 96, 128, 160, 224, 256, 288, 512)
        for a_width, b_width, stages, shape in product(
            widths, widths, (1, 2, 3, 4), ((10, 512, 848), (512, 10, 512), (1024, 1024, 512)),
        ):
            m, n, k = shape
            selected = prune_gemm_configs(self.configs, arguments(
                M=m, N=n, K=k, A_BLOCK_COLS=a_width, B_BLOCK_COLS=b_width,
                PIPELINE_STAGES=stages,
            ))
            self.assertGreater(len(selected), 1)
            self.assertLessEqual(len(selected), MAX_TUNING_CONFIGS)
            self.assertIn(baseline_gemm_tile((32, a_width), (32, b_width)),
                          [tile(config) for config in selected])
            for config in selected:
                self.assertEqual(a_width % config.kwargs["BLOCK_K"], 0)
                self.assertEqual(b_width % config.kwargs["BLOCK_K"], 0)

    def test_invalid_stage_or_layout_never_silently_selects_a_config(self):
        for updates in ({"PIPELINE_STAGES": 5}, {"A_BLOCK_COLS": 17, "NATIVE_FP8": True}):
            with self.assertRaisesRegex(ValueError, "No Triton autotune configuration"):
                prune_gemm_configs(self.configs, arguments(**updates))

    def test_auto_stages_and_warps_are_searched_with_historical_launch(self):
        selected = prune_gemm_configs(self.configs, arguments(PIPELINE_STAGES=0))
        self.assertGreater(len({config.num_stages for config in selected}), 1)
        self.assertTrue({2, 4, 8}.issubset({config.num_warps for config in selected}))
        self.assertTrue(any(tile(config) == (128, 128, 64, 8) and config.num_stages == 3
                            and config.kwargs["SPLIT_K"] == 1 for config in selected))
        self.assertLessEqual(len(selected), MAX_TUNING_CONFIGS)

    def test_search_is_independent_of_candidate_order(self):
        def signatures(configs):
            return [(tile(config), config.num_stages, config.kwargs["SPLIT_K"]) for config in configs]

        for m, n, k in ((1024, 1024, 512), (10, 10, 512), (2048, 10, 512), (10, 512, 2048)):
            args = arguments(M=m, N=n, K=k, PIPELINE_STAGES=0)
            expected = signatures(prune_gemm_configs(self.configs, args))
            self.assertEqual(expected, signatures(prune_gemm_configs(list(reversed(self.configs)), args)))
            self.assertEqual(len(expected), len(set(expected)))

    def test_skinny_search_includes_tiles_with_more_parallelism(self):
        for m, n, k in ((2048, 10, 512), (10, 512, 2048)):
            selected = prune_gemm_configs(self.configs, arguments(
                M=m, N=n, K=k, A_BLOCK_COLS=32, B_BLOCK_COLS=32, PIPELINE_STAGES=0))
            blocks = [((m + config.kwargs["BLOCK_M"] - 1) // config.kwargs["BLOCK_M"])
                      * ((n + config.kwargs["BLOCK_N"] - 1) // config.kwargs["BLOCK_N"])
                      * config.kwargs["SPLIT_K"] for config in selected]
            self.assertGreaterEqual(max(blocks), 128)
            self.assertTrue(any(config.kwargs["SPLIT_K"] > 1 for config in selected))
            long_axis = "BLOCK_M" if m > n else "BLOCK_N"
            self.assertTrue(any(config.kwargs[long_axis] <= 32 for config in selected))

    def test_split_buffers_and_search_are_bounded(self):
        for m, n, k in ((1024, 1024, 512), (10, 512, 10), (32, 10000, 1024)):
            self.assertFalse(split_k_enabled(m, n, k))
            selected = prune_gemm_configs(self.configs, arguments(M=m, N=n, K=k, PIPELINE_STAGES=0))
            self.assertEqual({config.kwargs["SPLIT_K"] for config in selected}, {1})

    def test_cache_lookup_matches_explicit_and_ordered_implicit_dtype_keys(self):
        metadata = dict.fromkeys(AUTOTUNE_KEY, 0)
        named = {"a_desc": object(), "a_scale_inv_ptr": SimpleNamespace(dtype="fp32"),
                 "b_scale_inv_ptr": SimpleNamespace(dtype="fp32"),
                 "bias_ptr": SimpleNamespace(dtype="bf16"),
                 "partials_ptr": SimpleNamespace(dtype="fp32"),
                 "output_ptr": SimpleNamespace(dtype="bf16"), **metadata}
        expected = tuple(metadata[name] for name in AUTOTUNE_KEY) + ("fp32", "fp32", "bf16", "fp32", "bf16")
        self.assertEqual(autotune_cache_key(named), expected)
        named["partials_ptr"] = SimpleNamespace(dtype="fp32")
        self.assertEqual(autotune_cache_key(named), expected)
        named["PIPELINE_STAGES"] = 3
        self.assertNotEqual(autotune_cache_key(named), expected)

    def test_saturated_output_grid_does_not_search_split_k(self):
        selected = prune_gemm_configs(self.configs, arguments(
            M=10, N=512, K=2048, PIPELINE_STAGES=0, NUM_SMS=1))
        self.assertEqual({config.kwargs["SPLIT_K"] for config in selected}, {1})


if __name__ == "__main__":
    unittest.main()
