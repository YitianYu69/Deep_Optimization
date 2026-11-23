"""Byte-exact FP8 layout conversion, including transpose tails and padding."""

import ast
from importlib.resources import files
from types import SimpleNamespace
import unittest

import torch

from tests.opt.test_activation_codec_kernels import CodecLanguage
from tests.opt.test_relu_kernels import TypedPointer


class FP8LayoutPackingTests(unittest.TestCase):
    def test_flat_and_tiled_packing_preserve_every_byte_for_strides_tails_and_formats(self):
        tl = CodecLanguage()
        namespace = dict(tl=tl, torch=torch, triton=SimpleNamespace(cdiv=tl.cdiv))
        tree = ast.parse(files("deepopt.opt.backends.triton").joinpath("gemm/fp8.py").read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_pack_rows", "_pack_transposed_rows")]
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "layout.py", "exec"), namespace)
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            source = torch.randn(65, 37).to(dtype).T
            source[0, 0] = -0.0
            rows, cols = source.shape
            padded = (cols + 15) // 16 * 16
            expected = torch.zeros(rows, padded, dtype=dtype)
            expected[:, :cols].copy_(source)
            for name in ("_pack_rows", "_pack_transposed_rows"):
                output = torch.empty_like(expected)
                args = (TypedPointer(source), TypedPointer(output), rows, cols, padded, *source.stride())
                if name == "_pack_rows":
                    for pid in range((rows * padded + 127) // 128):
                        tl.programs = (pid, 0, 0)
                        namespace[name](*args, 128)
                else:
                    for r in range((rows + 31) // 32):
                        for c in range((padded + 31) // 32):
                            tl.programs = (r, c, 0)
                            namespace[name](*args, 32)
                self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
