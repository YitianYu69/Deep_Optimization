"""Emitted pointwise geometry: baked block, index width and mask elision."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from deepopt.compiler.codegen.runtime import launch_pointwise
from deepopt.compiler.codegen.triton.pointwise import (
    MAX_INT32_INDEX,
    PointwiseConfig,
    generate_entire_kernel_source,
    generate_single_kernel_source,
    select_pointwise_config,
)
from deepopt.compiler.ir import PointwiseNode, TensorMeta, TensorValue
from deepopt.compiler.scheduler import FusionGroup


def chain(numel, dtype=torch.bfloat16, ops=("neg", "relu")):
    meta = TensorMeta((numel,), dtype, "cuda", (1,))
    values = [TensorValue(f"v{index}", meta) for index in range(len(ops) + 1)]
    nodes = []
    for index, (op, target) in enumerate(zip(ops, (torch.neg, torch.relu))):
        nodes.append(PointwiseNode(op, (values[index],), {}, values[index + 1], target))
    return FusionGroup(nodes)


class SelectionTests(unittest.TestCase):
    def test_block_follows_element_count_like_inductor(self):
        self.assertEqual(select_pointwise_config(1 << 20).block_size, 1024)
        self.assertEqual(select_pointwise_config(100003).block_size, 512)
        self.assertEqual(select_pointwise_config(1000).block_size, 256)
        self.assertEqual(select_pointwise_config(8).block_size, 256)

    def test_mask_is_elided_only_for_an_exact_grid(self):
        self.assertFalse(select_pointwise_config(1 << 20).masked)
        self.assertTrue(select_pointwise_config((1 << 20) + 1).masked)
        self.assertTrue(select_pointwise_config(100003).masked)

    def test_index_width_switches_at_int32_limit(self):
        self.assertEqual(select_pointwise_config(MAX_INT32_INDEX).index_dtype, "tl.int32")
        self.assertEqual(select_pointwise_config(MAX_INT32_INDEX + 1).index_dtype, "tl.int64")

    def test_warps_stay_in_range(self):
        for numel in (0, 8, 1000, 1 << 20, 1 << 40):
            with self.subTest(numel=numel):
                self.assertIn(select_pointwise_config(numel).num_warps, (1, 2, 4, 8))

    def test_user_block_size_acts_as_an_upper_bound(self):
        _, infos = generate_entire_kernel_source([chain(1 << 20)], max_block=256)
        self.assertEqual(infos[0][3].block_size, 256)


class EmittedSourceTests(unittest.TestCase):
    def test_exact_grid_emits_no_mask_and_no_widening(self):
        source, _ = generate_single_kernel_source(chain(1 << 20), "k")
        self.assertNotIn("mask", source.replace("'masked': False", ""))
        self.assertIn("pid = tl.program_id(0)", source)
        self.assertIn("offs = pid * 1024 + tl.arange(0, 1024)", source)
        self.assertIn("tl.load(x0_ptr + offs)", source)
        self.assertIn("tl.store(out_ptr + offs, v1)", source)
        self.assertNotIn("BLOCK_SIZE", source)
        compile(source, "<generated>", "exec")

    def test_ragged_grid_keeps_a_bounds_mask(self):
        source, _ = generate_single_kernel_source(chain(100003), "k")
        self.assertIn("mask = offs < N", source)
        self.assertIn("tl.load(x0_ptr + offs, mask=mask, other=0.0)", source)
        self.assertIn("tl.store(out_ptr + offs, v1, mask=mask)", source)
        compile(source, "<generated>", "exec")

    def test_large_tensor_widens_the_program_id(self):
        source, _ = generate_single_kernel_source(chain(MAX_INT32_INDEX + 1), "k")
        self.assertIn("pid = tl.program_id(0).to(tl.int64)", source)

    def test_rounding_cast_survives_every_node(self):
        source, _ = generate_single_kernel_source(chain(1 << 20, ops=("neg", "neg")), "k")
        self.assertIn("(v0).to(tl.bfloat16)", source.replace("((-x0)).to(tl.bfloat16)",
                                                            "(v0).to(tl.bfloat16)"))

    def test_n_remains_in_the_signature_for_abi_stability(self):
        source, _ = generate_single_kernel_source(chain(1 << 20), "k")
        self.assertIn("out_ptr, N):", source)


class LauncherTests(unittest.TestCase):
    def fake_triton(self):
        recorded = {}

        def cdiv(value, divisor):
            return -(-value // divisor)

        module = types.SimpleNamespace(cdiv=cdiv, recorded=recorded)
        return module

    def artifact(self, **overrides):
        base = types.SimpleNamespace(
            kernel=lambda *args, **kwargs: None,
            baked_block_size=True,
            block_size=256,
            num_warps=4,
            config=PointwiseConfig(256, 4, "tl.int32", False, 1024),
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def test_baked_artifact_uses_its_own_geometry_and_no_constexpr(self):
        module = self.fake_triton()
        seen = {}

        class Kernel:
            def __getitem__(self, grid):
                seen["grid"] = grid
                return self

            def __call__(self, *args, **kwargs):
                seen["args"] = args
                seen["kwargs"] = kwargs

        artifact = self.artifact()
        artifact.kernel = Kernel()
        output = torch.empty(1024)
        with patch.dict(sys.modules, {"triton": module}):
            launch_pointwise(artifact, (torch.empty(1024),), output, 4096)
        self.assertEqual(seen["grid"], (4,))
        self.assertNotIn("BLOCK_SIZE", seen["kwargs"])
        self.assertEqual(seen["kwargs"]["num_warps"], 4)
        self.assertFalse(seen["kwargs"]["enable_fp_fusion"])
        self.assertEqual(seen["args"][-1], 1024)

    def test_legacy_artifact_still_receives_block_size(self):
        module = self.fake_triton()
        seen = {}

        class Kernel:
            def __getitem__(self, grid):
                seen["grid"] = grid
                return self

            def __call__(self, *args, **kwargs):
                seen["kwargs"] = kwargs

        artifact = types.SimpleNamespace(kernel=Kernel())
        with patch.dict(sys.modules, {"triton": module}):
            launch_pointwise(artifact, (torch.empty(64),), torch.empty(64), 64)
        self.assertEqual(seen["kwargs"]["BLOCK_SIZE"], 64)
        self.assertEqual(seen["grid"], (1,))

    def test_baked_artifact_must_declare_its_geometry(self):
        module = self.fake_triton()
        artifact = types.SimpleNamespace(
            kernel=lambda *a, **k: self.fail("must not launch"),
            baked_block_size=True, block_size=256, num_warps=4)
        with patch.dict(sys.modules, {"triton": module}):
            with self.assertRaisesRegex(ValueError, "recorded launch geometry"):
                launch_pointwise(artifact, (torch.empty(1024),), torch.empty(1024), 1024)

    def test_unmasked_artifact_rejects_a_grid_that_does_not_divide(self):
        module = self.fake_triton()
        artifact = self.artifact()
        artifact.kernel = lambda *a, **k: self.fail("must not launch")
        with patch.dict(sys.modules, {"triton": module}):
            with self.assertRaisesRegex(ValueError, "does not divide"):
                launch_pointwise(artifact, (torch.empty(1000),), torch.empty(1000), 256)

    def test_masked_artifact_accepts_a_ragged_grid(self):
        module = self.fake_triton()
        seen = {}

        class Kernel:
            def __getitem__(self, grid):
                seen["grid"] = grid
                return self

            def __call__(self, *args, **kwargs):
                seen["kwargs"] = kwargs

        artifact = self.artifact(
            kernel=Kernel(), block_size=256,
            config=PointwiseConfig(256, 4, "tl.int32", True, 1000))
        with patch.dict(sys.modules, {"triton": module}):
            launch_pointwise(artifact, (torch.empty(1000),), torch.empty(1000), 256)
        self.assertEqual(seen["grid"], (4,))

    def test_empty_output_launches_nothing(self):
        module = self.fake_triton()
        artifact = types.SimpleNamespace(kernel=lambda *a, **k: self.fail("must not launch"))
        with patch.dict(sys.modules, {"triton": module}):
            launch_pointwise(artifact, (), torch.empty(0), 1024)


if __name__ == "__main__":
    unittest.main()
