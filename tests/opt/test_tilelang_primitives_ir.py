"""Construct real TileLang IR without requiring CUDA hardware or NVCC."""

import importlib
import importlib.util
import os
import unittest
from unittest.mock import patch


class TileLangPrimitiveIRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tilelang") is None:
            if os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1":
                raise RuntimeError("Mandatory TileLang IR validation requires TileLang")
            raise unittest.SkipTest("optional TileLang dependency is absent")

    def test_primitive_factories_construct_real_ir(self):
        cases = [
            ("quantization.fp8", "_statistics", (5, 7, 7, 1, 35, "float32", 3, 4, 448., "cuda:0")),
            ("quantization.fp8", "_cast", (5, 7, 7, 1, 35, "float32", "float8_e4m3fn", 3, 4, 2, 1, 4, 8, False, 448., "cuda:0", True)),
            ("quantization.fp8", "_cast", (5, 7, 7, 1, 35, "float8_e5m2", "float32", 3, 4, 2, 1, 4, 16, True, 0., "cuda:0", False)),
            ("scaling.reduction", "_amax", (5, 7, 7, 1, 35, "float32", 3, 4, 0., "cuda:0")),
            ("scaling.reduction", "_partial_amax", (37, 65, 130, 2, 4809, "bfloat16", 17, 65, 2, "cuda:0")),
            ("scaling.reduction", "_merge_amax", (6, 1025, 2, "cuda:0")),
            ("scaling.reduction", "_merge_amax", (6, 2, 1, "cuda:0")),
            ("scaling.reduction", "_finish_amax", (6, 1e-12, "cuda:0")),
            ("quantization.fp8", "_scales_from_amax", (6, 448., "cuda:0")),
            ("gradient.reduction", "_columns", (5, 7, 7, 1, 35, "float32", 4, 2, 1, 3, 4, True, "cuda:0")),
            ("quantization.delayed", "_snapshot", ("cuda:0",)),
            ("quantization.delayed", "_observe", (13, 13, 3, 1, "cuda:0")),
            ("quantization.delayed", "_update", (16, 3, 1, 0, 0, False, 448., "cuda:0")),
            ("quantization.delayed", "_update", (16, 3, 1, 0, 1, True, 57344., "cuda:0")),
            ("quantization.activation.affine", "_pack", ((5, 7), (7, 1), 35, 5, 7, 23, 0, 0, 2, "float32", "bfloat16", "cuda:0")),
            ("quantization.activation.affine", "_pack", ((5, 7), (1, 5), 35, 5, 7, 12, 3, 4, 1, "float32", "float32", "cuda:0")),
            ("quantization.activation.affine", "_pack", ((2, 3, 4), (12, 1, 3), 24, 6, 4, 7, 0, 0, 8, "float32", "float32", "cuda:0")),
            ("quantization.activation.affine", "_unpack", (5, 7, 23, 0, 0, 2, (5, 7), (16, 1), False, 0, "float32", "cuda:0")),
            ("quantization.activation.affine", "_unpack", (5, 7, 12, 3, 4, 1, (7, 5), (16, 1), True, 0, "float8_e5m2", "cuda:0")),
            ("quantization.activation.affine", "_unpack", (10, 3, 7, 0, 0, 4, (2, 3, 5), (15, 5, 1), False, 5, "bfloat16", "cuda:0")),
            ("quantization.activation.indices", "_pack_kernel", ((79,), (1,), 79, "int64", 79, 5, 13, "int32", "cuda:0")),
            ("quantization.activation.indices", "_pack_kernel", ((7, 9), (1, 7), 63, "int64", 63, 64, 63, "int64", "cuda:0")),
            ("quantization.activation.indices", "_unpack_kernel", (79, 5, 13, "int32", "int64", "cuda:0")),
            ("quantization.activation.indices", "_unpack_kernel", (79, 0, 0, "int32", "int64", "cuda:0")),
            ("quantization.activation.indices", "_unpack_kernel", (79, 64, 79, "int64", "int64", "cuda:0")),
            ("gemm.fp8", "_copy_rows", (5, 7, 7, 1, 35, 16, "float8_e4m3fn", "cuda:0")),
            ("gemm.epilogue", "_bias", (5, 7, 7, 1, 35, "float32", "float32", 1, 7, "bfloat16", "cuda:0")),
            ("quantization.activation.amax", "_encode", (64, 32, 64, 32, 1, "float32", 2, "cuda:0")),
            ("quantization.activation.amax", "_decode", (64, 32, 2, "bfloat16", "cuda:0")),
        ]
        gemm = (5, 9, 7, 7, 1, 7, 1, 35, 63, "float8_e4m3fn", "float8_e5m2",
                1, 1, 1, 1, 1, 1, 5, 7, 9, 7, True, "float32", 1, 0, "float32", 12, "cuda:0")
        generic = (*gemm[:11], 3, 1, 3, 1, 9, 15, 2, 3, 2, 3, False, *gemm[22:])
        cases.extend(("gemm.fp8", name, args) for name in ("_gemm", "_gemm_fp8") for args in (gemm, generic))
        for module_name, factory_name, args in cases:
            with self.subTest(module=module_name, factory=factory_name, args=args):
                module = importlib.import_module("deepopt.opt.backends.tilelang." + module_name)
                factory = getattr(module, factory_name)
                factory.cache_clear()
                try:
                    with patch.object(module, "compile_kernel", side_effect=lambda prim, device: prim):
                        prim = factory(*args)
                    self.assertEqual(type(prim).__name__, "PrimFunc")
                    self.assertGreaterEqual(len(prim.params), 2)
                finally:
                    factory.cache_clear()


if __name__ == "__main__":
    unittest.main()
