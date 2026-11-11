"""Interpret actual installed activation JIT bodies; CUDA compilation is separate."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest

import torch

from tests.opt.test_relu_kernels import ReLULanguage, TypedPointer, CPUCast
from tests.opt.test_activations import VARIANTS
from deepopt.opt.backends.reference.activation.math import evaluate
from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation, _dimensions
from deepopt.opt.quantization.activation_tensor import to_fp8_activation


class Language(ReLULanguage):
    floor = staticmethod(torch.floor)
    exp = staticmethod(torch.exp)
    int1 = torch.bool

    @staticmethod
    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype)

    @staticmethod
    def min(value, axis=0):
        return value.amin(dim=axis)


class ActivationKernelTests(unittest.TestCase):
    def setUp(self):
        self.tl = Language()
        self.k = {"tl": self.tl, "triton": SimpleNamespace(cdiv=self.tl.cdiv),
                  "libdevice": SimpleNamespace(erf=torch.erf, tanh=torch.tanh)}
        package = files("deepopt.opt.backends.triton")
        for path in ("quantization/fp8.py", "quantization/activation/affine.py", "activation/common.py",
                     "activation/math.py", "activation/elementwise_forward.py", "activation/elementwise_backward.py"):
            tree = ast.parse(package.joinpath(path).read_text())
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.decorator_list
                         and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in n.decorator_list)]
            for node in functions:
                node.decorator_list = []
            module = CPUCast().visit(ast.Module(body=functions, type_ignores=[]))
            exec(compile(ast.fix_missing_locations(module), path, "exec"), self.k)

    def launch(self, name, grid, *args):
        args = tuple(TypedPointer(a) if isinstance(a, torch.Tensor) else a for a in args)
        grid = (grid, 1) if isinstance(grid, int) else grid
        for p in range(grid[0]):
            for q in range(grid[1]):
                self.tl.programs = (p, q, 0)
                self.k[name](*args)

    def check(self, kind, approximate, *, fp8_input=False, fp8_output=False, bits=2,
              group=17, block_shape=None, shape=(3, 33), layout="last", packed=True,
              gradient_layout=None, fresh_gradient=False, current=False):
        torch.manual_seed(719)
        source = torch.randn(shape, dtype=torch.bfloat16)
        if len(shape) == 4:
            source = source.transpose(2, 3)
        geometry = ReLUGeometry(tuple(source.shape), layout)
        x = to_fp8_activation(source, layout="channels" if len(shape) == 4 else "last",
                              block_shape=block_shape) if fp8_input else source
        dense = geometry.matrix(x.dequantize(torch.float32) if fp8_input else source.float())
        expected, derivative = evaluate(dense, kind, approximate)
        data, inverse, meta = tensor_arguments(x)
        rows, cols = geometry.matrix_shape
        mask = kind == "relu6"
        codec = packed and not mask
        br, bc = block_shape if codec and block_shape else (0, 0)
        group = br * bc if br else group if codec else 512
        groups, _, words = _dimensions((rows, cols), group, (br, bc) if br else None, bits)
        chunk = min(4096, max(32, 1 << (group - 1).bit_length()))
        parts = (group + chunk - 1) // chunk
        if codec:
            saved = torch.zeros(groups, words, dtype=torch.int32)
            steps, minima = torch.empty(groups), torch.empty(groups)
        else:
            saved = torch.empty(((source.numel() + 31) // 32,) if mask and packed else (rows, cols),
                                dtype=torch.int32 if packed else torch.bool if mask else torch.float32)
            steps = minima = torch.empty(1)
        if codec and parts > 1:
            lo, hi = torch.empty(groups, parts), torch.empty(groups, parts)
            self.launch("_statistics", (groups, parts), data, inverse, lo, hi, rows, cols, geometry.shape,
                        layout == "channels", meta, kind, approximate, group, br, bc, parts, True, chunk)
            self.launch("_finish_min_max", groups, lo, hi, steps, minima, parts, bits, 1 << (parts - 1).bit_length())
        dtype = torch.float8_e4m3fn if fp8_output else torch.bfloat16
        output = torch.empty(rows, cols, dtype=dtype)
        obr, obc = block_shape or (rows, cols)
        sr, sc = (rows + obr - 1) // obr, (cols + obc - 1) // obc
        scale = torch.full((sr, sc), 2.0)
        out_inverse = torch.empty_like(scale)
        observations = torch.empty(groups * parts)
        snapshot, errors = torch.empty(2), torch.empty(groups * parts, dtype=torch.int32)
        local = current == "local"
        if current == "reduced":
            n = (obr * obc + 127) // 128
            partial = torch.empty(sr * sc, n)
            self.launch("_statistics", (sr * sc, n), data, inverse, partial, partial,
                        rows, cols, geometry.shape, layout == "channels", meta, kind, approximate,
                        obr * obc, obr, obc, n, False, 128)
            self.launch("_block_scales", sr * sc, partial, scale, out_inverse, n, 1 << (n - 1).bit_length(), 448.0)
        self.launch("_activation_forward", (groups, parts), data, inverse, output, saved, steps, minima,
                    scale, out_inverse, observations, snapshot, errors, rows, cols, geometry.shape,
                    layout == "channels", meta, kind, approximate, group, br, bc, words, True, packed, bits,
                    codec and parts > 1, fp8_output, False, local, obr, obc, sc,
                    448.0 if fp8_output else 0.0, False, True, not current, True, parts, chunk)
        expected_output = (expected * 2).clamp(-448, 448).to(dtype) if fp8_output else expected.to(dtype)
        if current:
            from deepopt.opt.quantization.quantizer import quantize_fp8

            q = quantize_fp8(expected, fp8_dtype=dtype, block_shape=block_shape)
            expected_output = q.data
            torch.testing.assert_close(out_inverse, q.scale_inv, rtol=2e-6, atol=1e-10)
        torch.testing.assert_close(output.float(), expected_output.float(), rtol=0.02, atol=0.008)
        self.assertFalse(errors.any())
        if codec:
            q = quantize_pack_activation(derivative, group, (br, bc) if br else None, bits=bits)
            # Tiny transcendental differences may move a quantization tie by one
            # code; check represented coefficients against the native oracle.
            from deepopt.opt.quantization.activation import PackedActivation

            actual = dequantize_unpack_activation(PackedActivation(saved, steps, minima, (rows, cols),
                                                 group, (br, bc) if br else None, bits), dtype=torch.float32)
            reference = dequantize_unpack_activation(q, dtype=torch.float32)
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
            derivative = actual
        elif mask:
            from deepopt.opt.quantization.boolean import pack_boolean_mask

            torch.testing.assert_close(saved, pack_boolean_mask(derivative).data if packed else derivative)
        else:
            torch.testing.assert_close(saved, derivative, rtol=2e-5, atol=2e-6)
        target = ReLUGeometry(geometry.shape, gradient_layout or layout)
        dy_native = torch.randn(geometry.shape, dtype=torch.bfloat16)
        dy = to_fp8_activation(dy_native, layout="last", fp8_dtype=torch.float8_e5m2) if fp8_input else dy_native
        gd, gi, gm = tensor_arguments(dy)
        gr, gc = target.matrix_shape
        gbr, gbc = (2, 8) if fresh_gradient else (1, 1)
        programs = ((gr + gbr - 1) // gbr * ((gc + gbc - 1) // gbc)) if fresh_gradient else (gr * gc + 127) // 128
        out = torch.empty(gr, gc, dtype=torch.float8_e5m2 if fresh_gradient else torch.bfloat16)
        ginverse = torch.empty((gr + gbr - 1) // gbr, (gc + gbc - 1) // gbc)
        c, s = (geometry.shape[1], prod(geometry.shape[2:])) if len(shape) >= 2 else (1, 1)
        self.launch("_activation_backward", programs, gd, gi, saved, steps, minima, out, scale, ginverse,
                    observations, gr, gc, geometry.shape, gm, target.layout == "channels", layout == "channels",
                    c, s, cols, group, br, bc, words, bits, mask, packed, fresh_gradient, fresh_gradient,
                    gbr, gbc, False, 57344.0, 32 if fresh_gradient else 128)
        expected_dx = geometry.matrix(dy.dequantize(torch.float32) if fp8_input else dy_native.float())
        expected_dx = torch.where(derivative, expected_dx, 0) if mask else expected_dx * derivative
        expected_dx = target.matrix(geometry.restore(expected_dx))
        if fresh_gradient:
            from deepopt.opt.quantization.quantizer import quantize_fp8

            q = quantize_fp8(expected_dx, fp8_dtype=torch.float8_e5m2, block_shape=(gbr, gbc))
            torch.testing.assert_close(out.float(), q.data.float(), rtol=0.25, atol=0.0)
            torch.testing.assert_close(ginverse, q.scale_inv, rtol=2e-5, atol=1e-10)
        else:
            torch.testing.assert_close(out, expected_dx.bfloat16(), rtol=0.01, atol=0.002)

    def test_fused_forward_backward_math_and_all_precision_pairs(self):
        for kind, approximate in VARIANTS:
            for source in (False, True):
                for output in (False, True):
                    for packed in (False, True):
                        with self.subTest(kind=kind, approx=approximate, input=source, output=output, packed=packed):
                            self.check(kind, approximate, fp8_input=source, fp8_output=output, packed=packed)

    def test_rectangular_strided_groups_and_retiled_fp8_gradients(self):
        for kind, approximate in VARIANTS:
            for layout in ("last", "channels"):
                for bits in (1, 2, 4, 8):
                    with self.subTest(kind=kind, layout=layout, bits=bits):
                        self.check(kind, approximate, fp8_input=True, fp8_output=True, bits=bits,
                                   block_shape=(2, 8), shape=(2, 3, 5, 7), layout=layout,
                                   gradient_layout="channels" if layout == "last" else "last", fresh_gradient=True)

    def test_bounded_multistage_derivative_statistics_and_mask_word_tails(self):
        for kind, approximate in VARIANTS:
            with self.subTest(kind=kind):
                self.check(kind, approximate, group=17003, shape=(3, 5701), bits=4)

    def test_local_and_partial_output_amax_measure_post_activation_values(self):
        for kind, approximate in VARIANTS:
            for mode in ("local", "reduced"):
                with self.subTest(kind=kind, mode=mode):
                    self.check(kind, approximate, group=512, shape=(3, 33), fp8_output=True, current=mode)


if __name__ == "__main__":
    unittest.main()
