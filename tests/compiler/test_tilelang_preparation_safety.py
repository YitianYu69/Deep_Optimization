"""Reproduce TileLang-only generated preparation indexing/nonfinite hazards."""

import importlib.util
import os
import unittest
from unittest.mock import patch

import torch

from deepopt.compiler.codegen.tilelang.activation_quantize import generate_preparation_source
from deepopt.opt.backends.tilelang import _runtime
from tests.opt.test_tilelang_operations import _Language


def kernels(rows, cols, language, **options):
    namespace = {}
    source = generate_preparation_source(rows, cols, "float32", "float8_e4m3fn", ("silu", "none"), **options)
    exec(compile(source, "<tilelang-preparation-safety>", "exec"), namespace)
    with patch.object(_runtime, "runtime", return_value=(None, language)), \
            patch.object(_runtime, "compile_kernel", side_effect=lambda prim, device: prim):
        return namespace["build"]("cpu")


class TileLangPreparationSafetyTests(unittest.TestCase):
    def test_cast_widens_before_signed_int32_program_offset_overflow(self):
        class VirtualStorage:
            def __init__(self):
                self.writes = {}

            def __getitem__(self, index):
                self.assert_nonnegative(index)
                return torch.tensor(0., dtype=torch.float32)

            @staticmethod
            def assert_nonnegative(index):
                if int(index) < 0:
                    raise AssertionError("overflowed address before int64 cast")

            def __setitem__(self, index, value):
                self.assert_nonnegative(index)
                self.writes[int(index)] = value

        language = _Language()
        language.program = (torch.tensor(2**24, dtype=torch.int32),)
        _, cast = kernels(2**31 + 512, 1, language)
        normal = VirtualStorage()
        cast(VirtualStorage(), torch.ones(1), normal)
        self.assertEqual(min(normal.writes), 2**31)
        self.assertEqual(len(normal.writes), 128)

    def test_amax_preserves_an_earlier_nan_with_nonpropagating_max_lowering(self):
        language = _Language()
        language.max = lambda a, b: torch.fmax(torch.as_tensor(a), torch.as_tensor(b))
        language.program = (0,)
        statistics, _ = kernels(1, 1025, language)
        for position in (0, 511, 1023, 1024):
            values = torch.ones(1025)
            values[position] = float("nan")
            partial = torch.empty(2)
            with self.subTest(position=position):
                statistics(values, partial)
                self.assertTrue(torch.isnan(partial[position // 1024]))


class TileLangPreparationFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tilelang") is None:
            if os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1":
                raise RuntimeError("TileLang frontend is required")
            raise unittest.SkipTest("optional TileLang frontend is absent")

    def test_fixed_generated_preparation_constructs_real_ir(self):
        from deepopt.compiler.codegen.source import load_source, write_source

        for rows, cols, options in ((3, 19, {}), (3, 19, {"paired": True}),
                                     (3, 19, {"group": 17, "bits": 2}),
                                     (2**31 + 512, 1, {})):
            source = generate_preparation_source(rows, cols, "float32", "float8_e4m3fn", ("silu", "none"), **options)
            # The real frontend reads the defining source via inspect.
            namespace = load_source(write_source(source)).__dict__
            with self.subTest(rows=rows, options=options), \
                    patch.object(_runtime, "compile_kernel", side_effect=lambda prim, device: prim):
                pair = namespace["build"]("cuda:0")
                self.assertTrue(all(type(prim).__name__ == "PrimFunc" for prim in pair))


if __name__ == "__main__":
    unittest.main()
