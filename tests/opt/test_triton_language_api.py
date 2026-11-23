"""Check deferred kernel API lookups without requiring a GPU or launching kernels."""

import ast
import importlib.util
from pathlib import Path
import unittest

import deepopt


class TritonSyntaxRegressionTests(unittest.TestCase):
    def test_pooling_jit_avoids_membership_and_loop_local_constexpr_annotations(self):
        root = Path(deepopt.__file__).parent / "opt/backends/triton/pooling"
        for path in (root / "window/forward.py", root / "window/backward.py", root / "common/windows.py"):
            tree = ast.parse(path.read_text())
            for function in (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.decorator_list):
                self.assertFalse(any(isinstance(node, (ast.In, ast.NotIn)) for node in ast.walk(function)), str(path))
                if path.name == "windows.py":
                    for loop in (node for node in ast.walk(function) if isinstance(node, ast.For)):
                        self.assertFalse(any(isinstance(node, ast.AnnAssign) for node in ast.walk(loop)), str(path))


@unittest.skipUnless(importlib.util.find_spec("triton"), "Triton is not installed")
class TritonLanguageAPITests(unittest.TestCase):
    def test_kernel_language_attributes_exist(self):
        import triton.language as tl

        root = Path(deepopt.__file__).parent / "opt" / "backends"
        references = 0
        for name in ("triton/gemm/fp8.py", "triton/quantization/fp8.py", "triton/quantization/delayed.py",
                     "triton/scaling/amax.py", "triton/quantization/activation/affine.py",
                     "triton/quantization/activation/fp8.py", "triton/quantization/activation/boolean.py",
                     "triton/activation/common.py", "triton/activation/forward.py", "triton/activation/backward.py",
                     "triton/activation/elementwise_forward.py", "triton/activation/elementwise_backward.py",
                     "triton/gradient/epilogue.py", "triton/gradient/reduction.py", "triton/gradient/transpose.py",
                     "triton/activation/gradient.py", "triton/normalization/gradient.py", "triton/normalization/recenter.py",
                     "triton/normalization/common.py", "triton/normalization/forward.py", "triton/normalization/backward.py",
                     "triton/normalization/small.py", "triton/normalization/layer/common.py",
                     "triton/normalization/layer/forward.py", "triton/normalization/layer/backward.py",
                     "triton/quantization/activation/indices.py", "triton/pooling/common/addressing.py",
                     "triton/pooling/common/windows.py", "triton/pooling/common/output.py",
                     "triton/pooling/common/global_reduce.py", "triton/pooling/window/forward.py",
                     "triton/pooling/window/backward.py", "triton/pooling/power/factors.py", "triton/pooling/unpool/forward.py"):
            path = root / name
            tree = ast.parse(path.read_text(), filename=str(path))
            aliases = {alias.asname for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names if alias.name == "triton.language" and alias.asname}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id in aliases):
                    references += 1
                    with self.subTest(path=str(path.relative_to(root)), line=node.lineno, attribute=node.attr):
                        self.assertTrue(hasattr(tl, node.attr),
                                        f"{path}:{node.lineno}: triton.language has no attribute {node.attr!r}")
        self.assertGreater(references, 0, "No Triton language references were checked")
