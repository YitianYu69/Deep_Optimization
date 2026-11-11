"""Execute installed FP8 gradient kernel bodies on CPU; CUDA compilation is separate."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments
from deepopt.opt.nn.normalization.geometry import NormGeometry
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation
from deepopt.opt.quantization.boolean import pack_boolean_mask
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.opt.test_relu_kernels import ReLULanguage, TypedPointer, CPUCast


class GradientLanguage(ReLULanguage):
    @staticmethod
    def min(value, axis=0):
        return value.amin(dim=axis)

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis, dtype=value.dtype)


class GradientKernelTests(unittest.TestCase):
    def setUp(self):
        self.tl = GradientLanguage()
        self.k = {"tl": self.tl, "triton": SimpleNamespace(cdiv=self.tl.cdiv), "FP8Activation": FP8Activation, "prod": prod}
        package = files("deepopt.opt.backends.triton")
        for path in ("gradient/epilogue.py", "gradient/transpose.py", "gradient/reduction.py", "gemm/fp8.py",
                     "activation/common.py", "activation/gradient.py", "normalization/recenter.py", "normalization/common.py", "normalization/gradient.py"):
            tree = ast.parse(package.joinpath(path).read_text())
            functions = []
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                jit = any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)
                if (jit or node.name == "input_arguments") and (path != "gemm/fp8.py" or node.name == "_reduce_split_k"):
                    node.decorator_list = []
                    functions.append(node)
            module = CPUCast().visit(ast.Module(body=functions, type_ignores=[]))
            exec(compile(ast.fix_missing_locations(module), path, "exec"), self.k)

    def call(self, name, *args, **kwargs):
        return self.k[name](*(TypedPointer(arg) if isinstance(arg, torch.Tensor) else arg for arg in args),
                            **{k: TypedPointer(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})

    def test_blockwise_gradient_epilogue_preserves_group_scales_and_edge_tiles(self):
        torch.manual_seed(47)
        values = torch.randn(5, 11)
        output = torch.empty_like(values, dtype=torch.float8_e5m2)
        inverse = torch.empty(5, 3)
        for pm in range(2):
            for pn in range(2):
                rows, cols = torch.arange(4) + pm * 4, torch.arange(8) + pn * 8
                valid = (rows[:, None] < 5) & (cols[None, :] < 11)
                tile = torch.zeros(4, 8)
                tile[valid] = values[rows[:, None].expand_as(valid)[valid], cols[None, :].expand_as(valid)[valid]]
                self.tl.programs = (pm * 2 + pn, 0, 0)
                result = self.k["_gradient_tile"](tile, rows, cols, valid, TypedPointer(inverse), TypedPointer(inverse),
                                                  TypedPointer(inverse), 5, 11, 4, 8, 1, 4, 57344.0, True, False)
                output[rows[:, None].expand_as(valid)[valid], cols[None, :].expand_as(valid)[valid]] = result[valid].to(output.dtype)
        expected = quantize_fp8(values, fp8_dtype=output.dtype, block_shape=(1, 4))
        torch.testing.assert_close(output.float(), expected.data.float(), rtol=0, atol=0)
        torch.testing.assert_close(inverse, expected.scale_inv, rtol=2e-7, atol=0)

    def test_split_k_quantizes_only_completed_sum_and_observes_completed_gradient(self):
        partials = torch.randint(-4, 5, (4, 5, 16)).float()
        output = torch.empty(5, 16, dtype=torch.float8_e5m2)
        scale, inverse, observed = torch.tensor([2.0]), torch.empty(1), torch.empty(1)
        self.tl.programs = (0, 0, 0)
        self.call("_reduce_split_k", partials, output, output, 5, 11, 16, 4, False, 128,
                  TypedPointer(scale), TypedPointer(inverse), TypedPointer(observed), True, 57344.0, True)
        expected = partials.sum(0)[:, :11]
        torch.testing.assert_close(output[:, :11].float(), (expected * 2).to(output.dtype).float(), rtol=0, atol=0)
        self.assertEqual(inverse.item(), 0.5)
        self.assertEqual(observed.item(), expected.abs().max().item())

    def test_transpose_requantizes_represented_values_and_bias_reduction_decodes_scales(self):
        torch.manual_seed(41)
        q = quantize_fp8(torch.randn(9, 7), fp8_dtype=torch.float8_e5m2, block_shape=(2, 4))
        output, inverse = torch.empty(7, 9, dtype=torch.float8_e5m2), torch.empty(7, 3)
        for pid in range(21):
            self.tl.programs = (pid, 0, 0)
            self.call("_transpose_quantize", q.data, q.scale_inv, output, inverse, 9, 7, *q.data.stride(),
                      2, 4, *q.scale_inv.stride(), 1, 4, 57344.0, 4)
        expected = quantize_fp8(q.dequantize().T, fp8_dtype=output.dtype, block_shape=(1, 4))
        torch.testing.assert_close(output.float(), expected.data.float(), rtol=0, atol=0)
        torch.testing.assert_close(inverse, expected.scale_inv, rtol=2e-7, atol=0)
        partial, bias = torch.empty(7, 3), torch.empty(7)
        for col in range(7):
            for part in range(3):
                self.tl.programs = (col * 3 + part, 0, 0)
                self.call("_bias_partial", q.data, q.scale_inv, partial, 9, 7, *q.data.stride(), 2, 4,
                          *q.scale_inv.stride(), 3, 4)
            self.tl.programs = (col, 0, 0)
            self.call("_bias_finish", partial, bias, 3, 4)
        torch.testing.assert_close(bias, q.dequantize().sum(0), rtol=1e-6, atol=1e-6)

    def test_bias_partial_reduction_is_bounded_for_large_row_counts(self):
        partials = (torch.arange(3 * 1025).reshape(3, 1025) % 7).float()
        reduced, output = torch.empty(3, 5), torch.empty(3)
        for pid in range(15):
            self.tl.programs = (pid, 0, 0)
            self.call("_bias_sum_chunks", partials, reduced, 1025, 5, 256)
        for col in range(3):
            self.tl.programs = (col, 0, 0)
            self.call("_bias_finish", reduced, output, 5, 8)
        torch.testing.assert_close(output, partials.sum(1), rtol=0, atol=0)

    def test_relu_gradient_layout_conversion_mask_extraction_and_fresh_scales(self):
        torch.manual_seed(51)
        shape = (2, 4, 3, 5)
        source = torch.randn(shape)
        dy = to_fp8_activation(torch.randn(shape), layout="channels", fp8_dtype=torch.float8_e5m2)
        data, inv, meta = tensor_arguments(dy)
        for packed in (False, True):
            mask_geom = ReLUGeometry(shape, "channels")
            predicate = mask_geom.matrix(source > 0)
            saved = pack_boolean_mask(predicate).data if packed else predicate.flatten()
            geom = ReLUGeometry(shape, "last")
            rows, cols = geom.matrix_shape
            result, inverse = torch.empty(rows, cols, dtype=torch.float8_e5m2), torch.empty(rows, 2)
            for pid in range(rows * 2):
                self.tl.programs = (pid, 0, 0)
                self.call("_relu_gradient", data, inv, saved, result, inverse, inverse, inverse,
                          rows, cols, shape, meta, False, True, 4, 15, packed, 1, 4, True, False, 57344.0, 4)
            expected = quantize_fp8(geom.matrix(torch.where(source > 0, dy.dequantize(torch.float32), 0)),
                                    fp8_dtype=result.dtype, block_shape=(1, 4))
            torch.testing.assert_close(result.float(), expected.data.float(), rtol=0, atol=0)
            torch.testing.assert_close(inverse, expected.scale_inv, rtol=2e-7, atol=0)

    def test_normalization_gradient_scatter_cast_and_xhat_compression(self):
        torch.manual_seed(55)
        shape = (2, 4, 3, 5)
        x = torch.randn(shape).bfloat16()
        dy = to_fp8_activation(torch.randn(shape), layout="last", fp8_dtype=torch.float8_e5m2)
        gamma = torch.randn(4)
        for kind in ("batch", "group"):
            geo = NormGeometry(kind, shape, 2)
            matrix = geo.matrix(x.float())
            mean = matrix.mean(1)
            inverse = (matrix.var(1, correction=0) + 1e-5).rsqrt()
            h = (matrix - mean[:, None]) * inverse[:, None]
            for compressed, recenter in ((False, False), (True, False), (True, True)):
                codec = quantize_pack_activation(h, 17, (1, 17))
                used_h = dequantize_unpack_activation(codec, torch.float32) if compressed else h
                shift = (used_h - h).mean(1) if recenter else None
                if recenter:
                    used_h = used_h - shift[:, None]
                u = geo.matrix(dy.dequantize(torch.float32) * gamma[None, :, None, None])
                sums = torch.stack((u.sum(1), (u * used_h).sum(1)))
                dense_dx = geo.restore(((u - sums[0, :, None] / geo.elements) -
                                       used_h * sums[1, :, None] / geo.elements) * inverse[:, None])
                source, xs, xm = self.k["input_arguments"](x)
                gd, gs, gm = self.k["input_arguments"](dy)
                for layout in ("last", "channels"):
                    target = ReLUGeometry(shape, layout)
                    rows, cols = target.matrix_shape
                    output = torch.empty(rows, cols, dtype=torch.float8_e5m2)
                    scale, out_inverse, observed = torch.tensor([2.0]), torch.empty(1), torch.empty(4)
                    for pid in range((rows * cols + 31) // 32):
                        self.tl.programs = (pid, 0, 0)
                        self.call("_input_gradient_fp8", gd, gs, codec.data, codec.step, codec.minimum,
                                  source, xs, mean, inverse, gamma, sums, output, scale, out_inverse, observed,
                                  4, 15, 2, kind == "batch", geo.domains, geo.elements, True,
                                  compressed, 17, codec.data.shape[1], 2, True, gm, xm,
                                  rows, cols, 5, layout == "channels", 1, 4, False, True, 57344.0, 32, center_shift=shift, RECENTER=recenter)
                    expected = (target.matrix(dense_dx) * 2).to(output.dtype)
                    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)
                    self.assertEqual(out_inverse.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
