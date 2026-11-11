"""Interpret installed ReLU JIT bodies on CPU; this does not compile CUDA."""

import ast
from importlib.resources import files
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments
from deepopt.opt.quantization.activation_tensor import to_fp8_activation
from deepopt.opt.quantization.boolean import pack_boolean_mask
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.opt.convolution_kernel_support import KernelLanguage, Pointer


class ReLULanguage(KernelLanguage):
    static_range = staticmethod(range)
    reshape = staticmethod(torch.reshape)
    broadcast_to = staticmethod(torch.broadcast_to)
    abs = staticmethod(torch.abs)
    div_rn = staticmethod(torch.true_divide)

    @staticmethod
    def max(value, axis=0):
        return value.amax(dim=axis)

    @staticmethod
    def where(predicate, a, b):
        return torch.where(torch.as_tensor(predicate), a, b)

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis, dtype=torch.int32)


class TypedPointer(Pointer):
    def __init__(self, tensor):
        super().__init__(tensor)
        self.dtype = SimpleNamespace(element_ty=tensor.dtype)


class CPUCast(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        # torch.to uses the same RNE FP8 conversion; the Triton-specific keyword
        # has no spelling in torch.Tensor.to. GPU code still compiles unchanged.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "fp_downcast_rounding"]
        return node


class ReLUKernelTests(unittest.TestCase):
    def setUp(self):
        self.tl = ReLULanguage()
        self.k = {"tl": self.tl, "triton": SimpleNamespace(cdiv=self.tl.cdiv)}
        package = files("deepopt.opt.backends.triton")
        for path in ("quantization/fp8.py", "activation/common.py", "activation/forward.py", "activation/backward.py"):
            tree = ast.parse(package.joinpath(path).read_text())
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.decorator_list
                         and (path != "quantization/fp8.py" or node.name in ("_scale", "_block_scales"))]
            for node in functions:
                node.decorator_list = []
            module = CPUCast().visit(ast.Module(body=functions, type_ignores=[]))
            exec(compile(ast.fix_missing_locations(module), path, "exec"), self.k)

    def launch(self, name, programs, *args):
        args = tuple(TypedPointer(value) if isinstance(value, torch.Tensor) else value for value in args)
        for pid in range(programs):
            self.tl.programs = (pid, 0, 0)
            self.k[name](*args)

    def check(self, shape, *, fp8_input=False, fp8_output=False, packed=True, layout="last", blocks=None,
              current="none", inplace=False):
        torch.manual_seed(314)
        source = torch.randn(shape, dtype=torch.bfloat16)
        if len(shape) == 4:
            source = source.transpose(2, 3)
        geometry = ReLUGeometry(tuple(source.shape), layout)
        x = to_fp8_activation(source, layout="channels" if len(shape) == 4 else "last", block_shape=blocks) if fp8_input else source
        dense = geometry.matrix(x.dequantize(torch.float32) if fp8_input else source.float())
        expected = dense.relu()
        data, inverse, meta = tensor_arguments(x)
        rows, cols = geometry.matrix_shape
        br, bc = blocks or (rows, cols)
        sc = (cols + bc - 1) // bc
        sr = (rows + br - 1) // br
        dtype = torch.float8_e4m3fn if fp8_output else torch.bfloat16
        output = data if inplace else torch.empty(rows, cols, dtype=dtype)
        mask = torch.full(((source.numel() + 31) // 32,) if packed else (source.numel(),),
                          123, dtype=torch.int32 if packed else torch.bool)
        scale = torch.tensor([[2.0]]) if blocks is None else torch.full((sr, sc), 2.0)
        out_inverse = torch.empty(sr, sc)
        errors, observations, snapshot = torch.empty(1000, dtype=torch.int32), torch.empty(1000), torch.empty(2)
        reuse = fp8_input and fp8_output and current == "none"
        maximum = torch.finfo(dtype).max if fp8_output else 0
        if current == "blocks":
            width = max(32, bc) if cols % 32 == 0 and br * max(32, bc) <= 16384 else bc
            programs = sr * ((cols + width - 1) // width)
            mask_elements = max(32, 1 << (((source.numel() + programs - 1) // programs) - 1).bit_length())
            self.launch("_forward_blocks", programs, data, inverse, output, mask,
                        out_inverse, errors, rows, cols, geometry.shape, layout == "channels", meta,
                        br, bc, width, width // bc, sc, maximum, True, packed, True, mask_elements)
        else:
            if current == "reduced":
                parts = (br * bc + 127) // 128
                partials = torch.empty(sr * sc, parts)
                self.launch("_partial_relu_amax", sr * sc * parts, data, inverse, partials, rows, cols,
                            geometry.shape, layout == "channels", meta, br, bc, sc, parts, 128)
                self.launch("_block_scales", sr * sc, partials, scale, out_inverse, parts,
                            1 << (parts - 1).bit_length(), maximum)
            block = max(32, 1 << (source.numel() - 1).bit_length()) if current == "small" else 128
            self.launch("_forward", (source.numel() + block - 1) // block, data, inverse, output, mask, scale,
                        out_inverse, errors, observations, snapshot, rows, cols, geometry.shape,
                        layout == "channels", meta, fp8_output, reuse, current == "small", br, bc, sc,
                        maximum, True, packed, inplace, fp8_output and not reuse, False, False, block)
        if fp8_output:
            if reuse:
                from deepopt.opt.quantization.activation_tensor import restore_activation

                raw = geometry.matrix(restore_activation(data, geometry.shape, x._activation_layout))
                oracle = torch.where(dense < 0, 0, raw.float()).to(dtype)
            elif current != "none":
                q = quantize_fp8(expected, fp8_dtype=dtype, block_shape=blocks)
                oracle = q.data
                # The CPU quantizer uses reciprocal; Triton uses correctly
                # rounded division. Permit the resulting FP32 ULP difference.
                torch.testing.assert_close(out_inverse, q.scale_inv, rtol=2e-7, atol=0)
            else:
                expanded = scale if blocks is None else scale.repeat_interleave(br, 0).repeat_interleave(bc, 1)[:rows, :cols]
                oracle = (expected * expanded).clamp(-maximum, maximum).to(dtype)
            actual = geometry.matrix(output) if inplace and not fp8_input else output
            torch.testing.assert_close(actual.float(), oracle.float(), rtol=0, atol=0)
        else:
            actual = geometry.matrix(output) if inplace else output
            torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
        predicate = ~(dense <= 0)
        torch.testing.assert_close(mask, pack_boolean_mask(predicate).data if packed else predicate.reshape(-1))
        dy = torch.randn(geometry.shape, dtype=torch.bfloat16)
        gradient = to_fp8_activation(dy, layout="last") if fp8_input else dy
        gd, gs, gm = tensor_arguments(gradient)
        dx = torch.empty(rows, cols, dtype=torch.bfloat16)
        self.launch("_backward", (source.numel() + 127) // 128, gd, gs, mask, dx, rows, cols,
                    geometry.shape, layout == "channels", gm, packed, False, False, 128)
        logical_dy = gradient.dequantize(torch.float32) if fp8_input else dy.float()
        torch.testing.assert_close(dx, torch.where(predicate, geometry.matrix(logical_dy), 0).bfloat16(), rtol=0, atol=0)

    def test_dense_fp8_strides_layouts_provided_scales_and_both_mask_formats(self):
        for fp8_input in (False, True):
            for fp8_output in (False, True):
                for packed in (False, True):
                    for layout in ("last", "channels"):
                        with self.subTest(fp8_input=fp8_input, fp8_output=fp8_output, packed=packed, layout=layout):
                            self.check((2, 8, 3, 5), fp8_input=fp8_input, fp8_output=fp8_output, packed=packed, layout=layout)

    def test_current_scaling_one_program_block_groups_and_hierarchical_tails(self):
        for packed in (False, True):
            for mode, shape, blocks in (("small", (3, 33), None),
                                        ("blocks", (5, 96), (4, 8)),
                                        ("blocks", (5, 96), (4, 64)),
                                        ("blocks", (5, 33), (4, 16)),
                                        ("blocks", (131, 8), (4, 8)),
                                        ("blocks", (16384, 1), (16384, 1)),
                                        ("reduced", (5, 33), (4, 16)),
                                        ("reduced", (131, 33), None)):
                with self.subTest(mode=mode, shape=shape, blocks=blocks, packed=packed):
                    self.check(shape, fp8_output=True, packed=packed, blocks=blocks, current=mode)

    def test_inplace_writes_original_dense_strides_and_fp8_storage(self):
        self.check((2, 8, 3, 5), inplace=True)
        self.check((5, 33), fp8_input=True, fp8_output=True, inplace=True)


if __name__ == "__main__":
    unittest.main()
