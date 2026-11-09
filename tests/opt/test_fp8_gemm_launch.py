"""Pointer-free layout reuse and warm launch behavior without a GPU backend."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from deepopt.opt.backends._gemm_launch import gemm_launch_plan, launch_selected, selected_config
from deepopt.opt.backends._triton_autotune import autotune_cache_key, DescriptorKernel


def plan(**updates):
    values = dict(m=33, n=65, k=128, a_block=None, b_block=None,
                  a_format="torch.float8_e4m3fn", b_format="torch.float8_e5m2", output_format="torch.float32",
                  device=0, capability=90, sms=132, a_stride=128, b_stride=128, stages="auto", bias_format=None)
    return gemm_launch_plan(**{**values, **updates})


class GemmLaunchTests(unittest.TestCase):
    def test_plan_key_matches_triton_order_and_storage_alignment(self):
        for output in ("torch.float32", "torch.bfloat16", "torch.float16"):
            for blocks in ((None, None), ((1, 32), (32, 32)), ((32, 1), None)):
                for bias in (None, "torch.float16"):
                    value = plan(output_format=output, a_block=blocks[0], b_block=blocks[1], bias_format=bias)
                    tensors = {name: SimpleNamespace(dtype=dtype) for name, dtype in zip(
                        ("a_scale", "b_scale", "bias", "partials", "output"),
                        ("torch.float32", "torch.float32", bias or output, "torch.float32", output))}
                    named = {"a_desc": object(), "b_desc": object(), "c_desc": object(), **tensors,
                             "M": 33, "N": 65, "K": 128, **value.arguments}
                    self.assertEqual(value.cache_key, autotune_cache_key(named))
                    self.assertEqual(value.output_stride * (4 if output == "torch.float32" else 2) % 16, 0)

    def test_layout_cache_is_bounded_immutable_and_separates_specializations(self):
        gemm_launch_plan.cache_clear()
        first = plan()
        self.assertIs(plan(), first)
        with self.assertRaises(TypeError):
            first.arguments["DEVICE_INDEX"] = 5
        for changes in ({"device": 1}, {"stages": 4}, {"a_stride": 144}, {"bias_format": "torch.float32"}):
            self.assertNotEqual(plan(**changes).cache_key, first.cache_key)
        for rows in range(300):
            plan(m=rows + 1)
        self.assertEqual(gemm_launch_plan.cache_info().currsize, 256)
        self.assertTrue(all(isinstance(value, (int, str, bool)) for value in first.arguments.values()))

    def test_selected_launch_uses_fresh_arguments_config_hooks_and_options(self):
        calls = []

        def run(*args, grid, warmup, **kwargs):
            self.assertFalse(warmup)
            calls.append((args, {"grid": grid, **kwargs}))

        fn = SimpleNamespace(arg_names=("a_desc", "b_desc", "c_desc"), run=run)
        hook = Mock()
        config = SimpleNamespace(kwargs={"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, pre_hook=hook,
                                 all_kwargs=lambda: {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3})
        kernel = SimpleNamespace(fn=DescriptorKernel(fn), arg_names=fn.arg_names, configs=[config], cache={})
        self.assertIs(selected_config(kernel, ()), config)
        for _ in range(2):
            descriptors = tuple(SimpleNamespace(block_shape=None) for _ in range(3))
            launch_selected(kernel, config, descriptors, {"OUTPUT_FORMAT": "torch.float32"}, (2, 1))
            self.assertEqual([value.block_shape for value in descriptors], [[32, 32], [64, 32], [32, 64]])
        self.assertIsNot(calls[0][0][0], calls[1][0][0])
        self.assertEqual(calls[1][1]["grid"], (2, 1))
        self.assertEqual(calls[1][1]["num_stages"], 3)
        self.assertEqual(hook.call_count, 2)
        self.assertFalse(hook.call_args.args[0]["warmup"])
