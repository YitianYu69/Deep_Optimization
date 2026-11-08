"""Check deferred kernel API lookups without requiring a GPU or launching kernels."""

import ast
import importlib.util
from pathlib import Path
import unittest

import deepopt


@unittest.skipUnless(importlib.util.find_spec("triton"), "Triton is not installed")
class TritonLanguageAPITests(unittest.TestCase):
    def test_kernel_language_attributes_exist(self):
        import triton.language as tl

        root = Path(deepopt.__file__).parent / "opt" / "backends"
        references = 0
        for name in ("triton_fp8.py", "triton_quantization.py", "triton_delayed.py", "scaling_utils/fused_amax.py"):
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
