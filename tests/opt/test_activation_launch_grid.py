"""Large activation domains must not overflow CUDA's secondary grid axes."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments
from tests.opt import test_activation_kernels as activation_cases


class Recorder:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            self.calls.append((self.name, grid))
        return call


class ActivationLaunchGridTests(unittest.TestCase):
    def test_production_launches_keep_large_statistics_on_the_primary_axis(self):
        path = files("deepopt.opt.backends.triton").joinpath("activation/elementwise.py")
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_forward")
        calls = []
        namespace = dict(torch=torch, prod=prod, tensor_arguments=tensor_arguments,
                         allocate_saved=lambda *a, **k: None,
                         triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                                next_power_of_2=lambda n: 1 << (n - 1).bit_length()))
        for name in ("_statistics", "_reduce_ranges", "_reduce_amax", "_block_scales", "_activation_forward"):
            namespace[name] = Recorder(name, calls)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        original_empty = torch.empty
        def empty(*args, **kwargs):
            return original_empty(*args, **{**kwargs, "device": "meta"})
        for parts in (65535, 65536, 75648):
            cols = parts * 4096 - 1
            x = torch.zeros(1, dtype=torch.bfloat16).expand(1, cols)
            calls.clear()
            with patch.object(torch, "empty", empty):
                namespace["_forward"](x, ReLUGeometry(tuple(x.shape)), kind="gelu", approximate="none",
                    dtype=torch.float8_e4m3fn, block_shape=None, packed=False, bits=2,
                    group_size=512, save_mask=False, inplace=False)
            self.assertEqual(calls[0], ("_statistics", (parts,)))
            for name, grid in calls:
                self.assertTrue(all(d <= 65535 for d in grid[1:]), (name, grid))

    def test_flat_statistics_preserve_partition_addresses_and_amax_at_large_tails(self):
        case = activation_cases.ActivationKernelTests()
        case.setUp()
        for parts in (65535, 65536, 75648):
            cols = parts * 4 - 1
            x = torch.linspace(-3, 3, cols).reshape(1, cols)
            data, inverse, meta = tensor_arguments(x)
            old, flat = torch.full((1, parts), float("nan")), torch.full((1, parts), float("nan"))
            from tests.opt.test_relu_kernels import TypedPointer
            for part in (0, parts // 2, parts - 1):
                common = (TypedPointer(data), TypedPointer(inverse), None, None, 1, cols, tuple(x.shape),
                          False, meta, "gelu", "none", cols, 1, cols, parts, False, 4)
                args = list(common)
                args[2:4] = [TypedPointer(old), TypedPointer(old)]
                case.tl.programs = (0, part, 0)
                case.k["_statistics"](*args)
                args[2:4] = [TypedPointer(flat), TypedPointer(flat)]
                case.tl.programs = (part, 0, 0)
                case.k["_statistics"](*args, FLAT_GRID=True)
                torch.testing.assert_close(flat[:, part], old[:, part], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
