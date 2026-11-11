"""Execute installed LayerNorm JIT bodies on CPU; real CUDA tests are separate."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest

import torch

from tests.opt.test_activation_kernels import Language
from tests.opt.test_relu_kernels import TypedPointer, CPUCast
from deepopt.opt.nn.activation.geometry import tensor_arguments, ReLUGeometry
from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
from deepopt.opt.quantization.activation import PackedActivation, quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import to_fp8_activation
from deepopt.opt.quantization.quantizer import quantize_fp8


class LayerLanguage(Language):
    rsqrt = staticmethod(torch.rsqrt)

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis, dtype=value.dtype)


class LayerNormKernelTests(unittest.TestCase):
    def setUp(self):
        self.tl = LayerLanguage()
        self.k = {"tl": self.tl, "triton": SimpleNamespace(cdiv=self.tl.cdiv)}
        root = files("deepopt.opt.backends.triton")
        for path in ("quantization/fp8.py", "quantization/activation/affine.py", "activation/common.py",
                     "normalization/recenter.py", "normalization/statistics.py", "normalization/layer/common.py", "normalization/layer/statistics.py",
                     "normalization/layer/forward.py", "normalization/layer/backward.py"):
            tree = ast.parse(root.joinpath(path).read_text())
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
                isinstance(d, ast.Attribute) and d.attr == "jit" for d in n.decorator_list)]
            for node in nodes:
                node.decorator_list = []
            module = CPUCast().visit(ast.Module(body=nodes, type_ignores=[]))
            exec(compile(ast.fix_missing_locations(module), path, "exec"), self.k)

    def launch(self, name, grid, *args, **kwargs):
        grid = (grid, 1) if isinstance(grid, int) else grid
        values = tuple(TypedPointer(a) if isinstance(a, torch.Tensor) else a for a in args)
        for p in range(grid[0]):
            for q in range(grid[1]):
                self.tl.programs = (p, q, 0)
                self.k[name](*values, **{k: TypedPointer(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})

    def check(self, shape=(3, 33), norm=(33,), *, bits=2, group=17, compressed=True,
              source_fp8=False, output_fp8=False, layout="last", staged=False, scaling="provided",
              gradient_fp8=False, affine=True, gradient_mode="retile", recenter=False):
        torch.manual_seed(624)
        source = torch.randn((*shape[:-1], shape[-1] * 2), dtype=torch.bfloat16)[..., ::2]
        source = to_fp8_activation(source, layout="channels" if len(shape) > 1 else "last", block_shape=(2, 8)) if source_fp8 else source
        dense = source.dequantize(torch.float32) if source_fp8 else source.float()
        geometry = LayerNormGeometry(shape, norm, layout=layout)
        m, k = geometry.matrix_shape
        matrix = dense.reshape(m, k)
        variance, expected_mean = torch.var_mean(matrix, 1, correction=0)
        expected_inverse = torch.rsqrt(variance + 1e-5)
        dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
        output = torch.empty(geometry.transport.matrix_shape, dtype=dtype)
        mean, inverse = torch.empty(m), torch.empty(m)
        data, xinv, meta = tensor_arguments(source)
        weight = torch.randn((*norm[:-1], norm[-1] * 2))[..., ::2] if affine else output
        bias = torch.randn((*norm[:-1], norm[-1] * 2))[..., ::2] if affine else output
        ws, bs = weight.stride() if affine else (), bias.stride() if affine else ()
        group = min(group, k)
        groups, pad = (k + group - 1) // group, max(32 // bits, 1 << (group - 1).bit_length())
        pgroups = 1 << (groups - 1).bit_length()
        words = (group + 32 // bits - 1) // (32 // bits)
        packed, steps, minima = torch.zeros(m * groups, words, dtype=torch.int32), torch.empty(m * groups), torch.empty(m * groups)
        rows, cols = geometry.transport.matrix_shape
        block_shape = (1, 8) if scaling == "local" and m > 1 else None
        br, bc = block_shape or (rows, cols)
        sr, sc = (rows + br - 1) // br, (cols + bc - 1) // bc
        scales, oinv = torch.full((sr, sc), 2.0), torch.empty(sr, sc)
        c, s = (shape[1], prod(shape[2:])) if len(shape) > 1 else (1, 1)
        capacity = m * groups * max(20, (group + 31) // 32)
        obs, snap, errors = torch.empty(capacity), torch.empty(2), torch.empty(capacity, dtype=torch.int32)
        center = torch.empty(m) if recenter else None
        correction = dict(center_shift=center, RECENTER=recenter)
        if not staged:
            self.launch("_row_forward", m, data, xinv, weight, bias, output, mean, inverse, packed, steps, minima,
                        scales, oinv, obs, snap, errors, m, k, shape, norm, meta, ws, bs, affine, affine, 1e-5,
                        group, groups, pgroups, pad, words, bits, compressed, output_fp8, False,
                        scaling == "local", block_shape is None, cols, layout == "channels", c, s, br, bc, sc, 448.,
                        False, scaling == "provided" and output_fp8, True, 1 << (k - 1).bit_length(), **correction)
        else:
            if k <= 8192:
                self.launch("_complete_statistics", m, data, xinv, mean, inverse, k, shape, meta, 1e-5, 1 << (k - 1).bit_length())
            else:
                parts = (k + 31) // 32
                partial = torch.empty(3, m, parts)
                self.launch("_partial_statistics", (m, parts), data, xinv, partial, m, k, shape, meta, parts, 32)
                while parts > 1:
                    nxt = (parts + 3) // 4
                    reduced = torch.empty(3, m, nxt)
                    self.launch("_merge_stats", (m, nxt), partial, reduced, m, parts, nxt, 4)
                    partial, parts = reduced, nxt
                self.launch("_finish_stats", (m + 31) // 32, partial, mean, inverse, output, output, output,
                            m, 1e-5, True, False, False, 0.0, 32)
            if output_fp8 and scaling == "reduced":
                parts = (br * bc + 31) // 32
                partial = torch.empty(sr * sc, parts)
                self.launch("_output_amax", (sr * sc, parts), data, xinv, weight, bias, mean, inverse, partial,
                            k, shape, norm, meta, ws, bs, affine, affine, rows, cols, layout == "channels", c, s,
                            br, bc, sc, parts, 32)
                self.launch("_block_scales", sr * sc, partial, scales, oinv, parts, 1 << (parts - 1).bit_length(), 448.)
            block = min(128, max(32 // bits, 1 << (group - 1).bit_length()))
            parts = (group + block - 1) // block
            if parts > 1 and compressed:
                lows, highs = torch.empty(m * groups, parts), torch.empty(m * groups, parts)
                self.launch("_xhat_ranges", (m * groups, parts), data, xinv, mean, inverse, lows, highs,
                            k, shape, meta, group, groups, parts, block)
                self.launch("_finish_min_max", m * groups, lows, highs, steps, minima, parts, bits, 1 << (parts - 1).bit_length())
            center_errors = torch.empty(2, m, groups * parts)
            self.launch("_epilogue", (m * groups, parts), data, xinv, weight, bias, mean, inverse, output, packed, steps, minima,
                        scales, oinv, obs, snap, errors, k, shape, norm, meta, ws, bs, affine, affine,
                        group, groups, words, bits, compressed, parts > 1, output_fp8, cols, layout == "channels", c, s,
                        br, bc, sc, 448., parts, False, False, True, block, center_errors=center_errors, M=m, **correction)
            if recenter and groups * parts > 1:
                self.launch("_merge_center", m, center_errors, center, center, m, groups * parts, 1, True,
                            1 << (groups * parts - 1).bit_length())
        torch.testing.assert_close(mean, expected_mean, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(inverse, expected_inverse, rtol=2e-5, atol=2e-6)
        h = (matrix - mean[:, None]) * inverse[:, None]
        expected_y = h * weight.reshape(1, -1) + bias.reshape(1, -1) if affine else h
        expected_y = geometry.transport.matrix(expected_y.reshape(shape))
        if output_fp8:
            if scaling in ("local", "reduced"):
                q = quantize_fp8(expected_y, fp8_dtype=dtype, block_shape=block_shape)
                torch.testing.assert_close(oinv, q.scale_inv, rtol=2e-5, atol=1e-9)
                expected_y = q.data.float()
            else:
                expected_y = (expected_y * 2).clamp(-448, 448).to(dtype).float()
            torch.testing.assert_close(output.float(), expected_y.float(), rtol=0.13, atol=0.015)
        else:
            torch.testing.assert_close(output, expected_y.bfloat16(), rtol=0.01, atol=0.004)
        if compressed:
            q = PackedActivation(packed, steps, minima, (m, k), group, (1, group), bits)
            oracle = quantize_pack_activation(h, group, (1, group), bits=bits)
            original_h = h
            h = dequantize_unpack_activation(q, dtype=torch.float32)
            torch.testing.assert_close(h, dequantize_unpack_activation(oracle, dtype=torch.float32), rtol=2e-5, atol=2e-6)
            if recenter:
                torch.testing.assert_close(center, (h - original_h).mean(1), rtol=3e-5, atol=2e-6)
                h = h - center[:, None]
                torch.testing.assert_close(h.mean(1), original_h.mean(1), rtol=0, atol=2e-6)
        dy = torch.randn(shape, dtype=torch.bfloat16)
        dy = to_fp8_activation(dy, layout="last", fp8_dtype=torch.float8_e5m2) if source_fp8 else dy
        gd, gi, gm = tensor_arguments(dy)
        dense_dy = dy.dequantize(torch.float32) if source_fp8 else dy.float()
        u = dense_dy.reshape(m, k) * weight.reshape(1, -1) if affine else dense_dy.reshape(m, k)
        dx_expected = (u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inverse[:, None]
        base = (gd, gi, packed, steps, minima, data, xinv, mean, inverse, weight)
        metadata = (m, k, shape, norm, gm, meta, ws, affine, compressed, group, words, bits)
        if gradient_fp8:
            target = ReLUGeometry(shape, "channels" if gradient_mode == "retile" and len(shape) > 1 else "last")
            gr, gc = target.matrix_shape
            gbr, gbc = ((gr, gc) if gradient_mode in ("provided", "tensorwise") else
                         (1, 8) if gradient_mode == "row" else (2, 8))
            inv = torch.empty((gr + gbr - 1) // gbr, (gc + gbc - 1) // gbc)
            dx = torch.empty(gr, gc, dtype=torch.float8_e5m2)
        else:
            target, dx = ReLUGeometry(shape, "last"), torch.empty(m, k, dtype=torch.bfloat16)
            gr, gc = target.matrix_shape
            gbr, gbc, inv = 1, 1, oinv
        if not staged and (not gradient_fp8 or gradient_mode != "retile"):
            if gradient_fp8:
                self.launch("_row_backward", m, *base, dx, scales, inv, obs, *metadata, True,
                            gradient_mode != "provided", gradient_mode in ("provided", "tensorwise"),
                            gc, target.layout == "channels", c, s, gbc, (gc + gbc - 1) // gbc,
                            gradient_mode == "provided", 57344., 1 << (k - 1).bit_length(), **correction)
            else:
                self.launch("_row_backward", m, *base, dx, scales, oinv, obs, *metadata, False, False, False,
                            gc, False, c, s, 1, gc, False, 0., 1 << (k - 1).bit_length(), **correction)
        else:
            parts = (k + 31) // 32
            sums = torch.empty(2, m, parts)
            self.launch("_row_sums", (m, parts), *base, sums, *metadata, parts, 32, **correction)
            while parts > 1:
                nxt = (parts + 3) // 4
                reduced = torch.empty(2, m, nxt)
                self.launch("_merge_sums", (m, nxt), sums, reduced, m, parts, nxt, 4)
                sums, parts = reduced, nxt
            programs = ((gr + 1) // 2 * ((gc + 7) // 8)) if gradient_fp8 else (gr * gc + 127) // 128
            self.launch("_input_gradient", programs, *base, sums, dx, scales, inv, obs, *metadata,
                        gr, gc, target.layout == "channels", c, s, gradient_fp8, gradient_fp8,
                        gbr, gbc, False, 57344., 32 if gradient_fp8 else 128, **correction)
        if gradient_fp8:
            expected = target.matrix(dx_expected.reshape(shape))
            if gradient_mode == "provided":
                torch.testing.assert_close(dx.float(), (expected * 2).clamp(-57344, 57344).to(torch.float8_e5m2).float(), rtol=0.25, atol=0.001)
                torch.testing.assert_close(inv, torch.full_like(inv, 0.5))
                torch.testing.assert_close(obs[:m], dx_expected.abs().amax(1))
            else:
                q = quantize_fp8(expected, fp8_dtype=torch.float8_e5m2, block_shape=None if gradient_mode == "tensorwise" else (gbr, gbc))
                torch.testing.assert_close(dx.float(), q.data.float(), rtol=0.25, atol=0.001)
                torch.testing.assert_close(inv, q.scale_inv, rtol=3e-5, atol=1e-9)
        else:
            torch.testing.assert_close(dx.reshape(shape), dx_expected.reshape(shape).bfloat16(), rtol=0.02, atol=0.004)
        parts = (m + 3) // 4
        affine_grad = torch.empty(2, parts, k)
        self.launch("_affine_partial", ((k + 7) // 8, parts), gd, gi, packed, steps, minima, data, xinv, mean, inverse, affine_grad,
                    m, k, shape, gm, meta, compressed, group, words, bits, True, True, parts, 4, 8, **correction)
        while parts > 1:
            nxt = (parts + 3) // 4
            reduced = torch.empty(2, nxt, k)
            self.launch("_affine_merge", ((k + 7) // 8, nxt), affine_grad, reduced, k, parts, nxt, True, True, 4, 8)
            affine_grad, parts = reduced, nxt
        torch.testing.assert_close(affine_grad[0, 0], (dense_dy.reshape(m, k) * h).sum(0), rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(affine_grad[1, 0], dense_dy.reshape(m, k).sum(0), rtol=3e-5, atol=3e-6)
        return output.clone(), mean.clone(), inverse.clone()

    def test_recenter_fused_rows_chunked_encode_and_fp8_dx(self):
        for bits in (1, 2, 4, 8):
            for staged in (False, True):
                for mode in (("retile",) if staged else ("row", "provided", "retile")):
                    self.check(bits=bits, group=17, recenter=True, staged=staged, source_fp8=True,
                               output_fp8=True, gradient_fp8=True, gradient_mode=mode)
        self.check((3, 17003), (17003,), staged=True, group=17003, bits=1, recenter=True)

    def test_row_fast_paths_precision_pairs_and_all_codec_widths(self):
        for bits in (1, 2, 4, 8):
            for src in (False, True):
                for output in (False, True):
                    for compressed in (False, True):
                        with self.subTest(bits=bits, src=src, output=output, compressed=compressed):
                            self.check(bits=bits, source_fp8=src, output_fp8=output, compressed=compressed)

    def test_suffix_and_channel_transport_mapping_with_retiled_fp8_dx(self):
        for shape, norm in (((2, 3, 7), (3, 7)), ((2, 3, 5, 7), (5, 7))):
            for layout in ("last", "channels"):
                for staged in (False, True):
                    with self.subTest(shape=shape, layout=layout, staged=staged):
                        self.check(shape, norm, source_fp8=True, output_fp8=True, layout=layout, staged=staged, gradient_fp8=True)

    def test_partial_statistics_large_codec_groups_and_bounded_affine_reductions(self):
        self.check((129, 33), (33,), staged=True, group=17)
        self.check((3, 17003), (17003,), staged=True, group=17003, bits=4)

    def test_current_scales_measure_post_affine_values_for_local_and_global_domains(self):
        self.check((1, 33), (33,), group=17, output_fp8=True, scaling="local")
        self.check((3, 33), (33,), group=32, output_fp8=True, scaling="local")
        self.check((3, 33), (33,), group=17, output_fp8=True, staged=True, scaling="reduced")

    def test_row_backward_fuses_current_block_tensorwise_and_delayed_scale_outputs(self):
        self.check((3, 33), (33,), source_fp8=True, gradient_fp8=True, gradient_mode="row")
        self.check((1, 33), (33,), source_fp8=True, gradient_fp8=True, gradient_mode="tensorwise")
        self.check((2, 3, 7), (3, 7), source_fp8=True, gradient_fp8=True, gradient_mode="provided")

    def test_codec_partition_and_fallback_do_not_change_forward_values(self):
        baseline = self.check((2, 257), (257,), compressed=False, group=512)
        for group in (1, 17, 512):
            for bits in (1, 8):
                actual = self.check((2, 257), (257,), compressed=True, group=group, bits=bits, staged=group == 1)
                for value, expected in zip(actual, baseline):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
