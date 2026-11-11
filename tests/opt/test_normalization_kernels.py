"""Execute installed normalization JIT bodies on CPU; CUDA compilation is separate."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.nn.normalization import NormGeometry
from deepopt.opt.quantization.activation import PackedActivation, quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation
from tests.opt.convolution_kernel_support import KernelLanguage, Pointer


class NormLanguage(KernelLanguage):
    static_range = staticmethod(range)
    reshape = staticmethod(torch.reshape)
    broadcast_to = staticmethod(torch.broadcast_to)
    floor = staticmethod(torch.floor)
    abs = staticmethod(torch.abs)
    rsqrt = staticmethod(torch.rsqrt)
    div_rn = staticmethod(torch.true_divide)

    @staticmethod
    def where(condition, a, b):
        return torch.where(torch.as_tensor(condition), a, b)

    @staticmethod
    def min(value, axis=0):
        return value.amin(axis)

    @staticmethod
    def max(value, axis=0):
        return value.amax(axis)


def kernels():
    language = NormLanguage()
    namespace = {"tl": language, "triton": SimpleNamespace(cdiv=language.cdiv), "FP8Activation": FP8Activation, "prod": prod}
    package = files("deepopt.opt.backends.triton")
    for path in ("quantization/activation/affine.py", "normalization/common.py", "normalization/recenter.py", "normalization/statistics.py",
                 "normalization/forward.py", "normalization/backward.py"):
        tree = ast.parse(package.joinpath(path).read_text())
        functions = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and (path != "quantization/activation/affine.py" or node.name in (
                "_affine_step", "_affine_quantize_pack_with_metadata", "_affine_quantize_pack", "_finish_min_max",
            )):
                if node.name == "input_arguments" or any(
                    isinstance(item, ast.Attribute) and item.attr == "jit" for item in node.decorator_list
                ):
                    node.decorator_list = []
                    functions.append(node)
        exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    return language, namespace


class NormalizationKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1973)
        self.tl, self.k = kernels()

    def launch(self, name, grid, *args, **kwargs):
        for x in range(grid[0]):
            for y in range(grid[1] if len(grid) > 1 else 1):
                self.tl.programs = (x, y, 0)
                self.k[name](*(Pointer(value) if isinstance(value, torch.Tensor) else value for value in args),
                             **{k: Pointer(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})

    def reduce(self, value, domains, parts, planes, name):
        while parts > 1:
            block = min(4, 1 << (parts - 1).bit_length())
            next_parts = (parts + block - 1) // block
            output = torch.empty(planes, domains, next_parts)
            self.launch(name, (domains, next_parts), value, output, domains, parts, next_parts, block)
            value, parts = output, next_parts
        return value

    def _check(self, kind, shape, *, storage="bf16", bits=2, compressed=True, group=17, eval_mode=False, recenter=False):
        original = (torch.randn(shape) * 3 + 20).bfloat16()
        if len(shape) > 3:
            original = original.transpose(-1, -2)
        geometry = NormGeometry(kind, tuple(original.shape), 2, batch_stats=not eval_mode)
        x = to_fp8_activation(original, block_shape=(4, 4)) if storage == "fp8" else original
        source, scale, meta = self.k["input_arguments"](x)
        c, s, g, d, e = geometry.channels, geometry.spatial, geometry.groups, geometry.domains, geometry.elements
        b = kind == "batch"
        count = (e + 31) // 32
        partial = torch.empty(3, d, count)
        mu, inv = torch.empty(d), torch.empty(d)
        running_mu, running_var = torch.randn(c), torch.rand(c) + 1
        expected_mu, expected_var = running_mu.clone(), running_var.clone()
        if not eval_mode:
            self.launch("_partial_stats", (d, count), source, scale, partial, c, s, g, b, d, e, count, meta, 32)
            partial = self.reduce(partial, d, count, 3, "_merge_stats")
        update = b and not eval_mode
        self.launch("_finish_stats", ((d + 31) // 32,), partial, mu, inv, running_mu, running_var, mu,
                    d, 1e-5, not eval_mode, update, False, 0.1, 32)
        values = x.dequantize(torch.float32) if storage == "fp8" else original.float()
        matrix = geometry.matrix(values)
        var_ref, mu_ref = torch.var_mean(matrix, 1, correction=0)
        torch.testing.assert_close(mu, expected_mu if eval_mode else mu_ref, rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(inv, torch.rsqrt((expected_var if eval_mode else var_ref) + 1e-5), rtol=3e-6, atol=3e-6)
        if update:
            torch.testing.assert_close(running_mu, expected_mu * 0.9 + mu_ref * 0.1)
            torch.testing.assert_close(running_var, expected_var * 0.9 + var_ref * (e / (e - 1)) * 0.1)
        group = min(group, e)
        groups, words = (e + group - 1) // group, (group + 32 // bits - 1) // (32 // bits)
        data = torch.empty((d * groups, words), dtype=torch.int32)
        step, low = torch.empty(d * groups), torch.empty(d * groups)
        weight, bias = torch.randn(c), torch.randn(c)
        output = torch.empty(geometry.shape)
        block = max(32 // bits, 1 << (group - 1).bit_length())
        center = torch.empty(d) if recenter else None
        parts = (group + 1023) // 1024 if compressed and group > 4096 else 1
        center_parts = groups * parts
        center_errors = torch.empty(2, d, center_parts) if recenter else None
        correction = dict(center_shift=center, errors=center_errors, DOMAINS=d, RECENTER=recenter)
        if compressed and group > 4096:
            block = 1024
            parts = (group + block - 1) // block
            partial_low, partial_high = torch.empty(d * groups, parts), torch.empty(d * groups, parts)
            args = (source, scale, weight, bias, mu, inv, output, data, step, low, partial_low, partial_high,
                    c, s, g, b, e, groups, group, words, bits)
            self.launch("_apply", (d * groups, parts), *args, True, True, False, parts, True, True,
                        meta, output.stride(), block, **correction)
            self.launch("_finish_min_max", (d * groups,), partial_low, partial_high, step, low, parts, bits,
                        1 << (parts - 1).bit_length())
            self.launch("_apply", (d * groups, parts), *args, True, False, True, parts, True, True,
                        meta, output.stride(), block, **correction)
        else:
            self.launch("_apply", (d * groups, 1), source, scale, weight, bias, mu, inv, output, data, step, low, mu, mu,
                        c, s, g, b, e, groups, group, words, bits, compressed, False, False, 1, True, True,
                        meta, output.stride(), block, **correction)
        if recenter and center_parts > 1:
            self.launch("_merge_center", (d, 1), center_errors, center, center, d, center_parts, 1, True,
                        1 << (center_parts - 1).bit_length())
        h = (matrix - mu[:, None]) * inv[:, None]
        affine = geometry.restore(h) * weight.reshape((1, c) + (1,) * (len(shape) - 2))
        affine = affine + bias.reshape((1, c) + (1,) * (len(shape) - 2))
        torch.testing.assert_close(output, affine, rtol=0, atol=0)
        if compressed:
            packed = PackedActivation(data, step, low, (d, e), group, (1, group), bits)
            expected = quantize_pack_activation(h, group, (1, group), bits=bits)
            torch.testing.assert_close(data, expected.data, rtol=0, atol=0)
            decoded = dequantize_unpack_activation(packed, torch.float32)
            if recenter:
                torch.testing.assert_close(center, (decoded - h).mean(1), rtol=3e-5, atol=2e-6)
                torch.testing.assert_close((decoded - center[:, None]).mean(1), h.mean(1), rtol=3e-5, atol=2e-6)
                decoded = decoded - center[:, None]
            h = decoded
        gradient = torch.randn(geometry.shape)
        dy, dy_scale, dy_meta = self.k["input_arguments"](gradient)
        count = (e + 31) // 32
        partial = torch.empty(2, d, count)
        args = (dy, dy_scale, data, step, low, source, scale, mu, inv, weight)
        if not eval_mode:
            self.launch("_partial_backward", (d, count), *args, partial, c, s, g, b, d, e, count,
                        compressed, group, words, bits, True, dy_meta, meta, 32, center_shift=center, RECENTER=recenter)
            partial = self.reduce(partial, d, count, 2, "_merge_sums")
        dx = torch.empty(geometry.shape)
        self.launch("_input_gradient", (d, count), *args, partial, dx, c, s, g, b, d, e, not eval_mode,
                    compressed, group, words, bits, True, dy_meta, meta, 32, center_shift=center, RECENTER=recenter)
        u = geometry.matrix(gradient * weight.reshape((1, c) + (1,) * (len(shape) - 2)))
        if not eval_mode:
            u = u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)
        torch.testing.assert_close(dx, geometry.restore(u * inv[:, None]), rtol=3e-5, atol=3e-5)
        count = (shape[0] * s + 31) // 32
        partial = torch.empty(2, c, count)
        self.launch("_partial_affine", (c, count), dy, dy_scale, data, step, low, source, scale, mu, inv, partial,
                    shape[0], c, s, g, b, e, count, compressed, group, words, bits, True, True, dy_meta, meta, 32, center_shift=center, RECENTER=recenter)
        reduced = self.reduce(partial, c, count, 2, "_merge_sums")
        dw, db = torch.empty(c), torch.empty(c)
        self.launch("_finish_affine", ((c + 31) // 32,), reduced, dw, db, c, True, True, 32)
        dims = (0, *range(2, len(shape)))
        torch.testing.assert_close(dw, (gradient * geometry.restore(h)).sum(dims), rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(db, gradient.sum(dims), rtol=3e-5, atol=3e-5)

    def test_recenter_register_codec_error_and_all_backward_consumers(self):
        for kind in ("batch", "group"):
            for bits in (1, 2, 4, 8):
                for group in (1, 17, 512):
                    self._check(kind, (2, 4, 3, 5), bits=bits, group=group, recenter=True, storage="fp8")
        self._check("batch", (2, 4, 3, 5), bits=2, recenter=True, eval_mode=True)
        self._check("group", (1, 4, 1, 2500), group=8192, recenter=True)

    def test_center_finish_is_weighted_hierarchical_and_rstd_does_not_scale_centering(self):
        domains, parts = 3, 1033
        partial = torch.stack((torch.randn(domains, parts), torch.randint(1, 1025, (domains, parts)).float()))
        expected = partial[0].sum(1) / partial[1].sum(1)
        shift = torch.empty(domains)
        while parts > 1:
            block = min(128, 1 << (parts - 1).bit_length())
            nxt = (parts + block - 1) // block
            output = torch.empty(2, domains, nxt)
            self.launch("_merge_center", (domains, nxt), partial, output, shift, domains, parts, nxt, nxt == 1, block)
            partial, parts = output, nxt
        torch.testing.assert_close(shift, expected)
        h = torch.randn(5, 7)
        indices = torch.arange(5)[:, None]
        valid = torch.ones_like(h, dtype=torch.bool)
        shift = torch.arange(5).float() + 1
        inverse = torch.tensor([1., 2., 0., float("inf"), float("nan")])
        corrected, rstd = self.k["_correct_xhat"](h, Pointer(inverse), Pointer(shift), indices, valid, True)
        torch.testing.assert_close(corrected[:2], h[:2] - shift[:2, None])
        torch.testing.assert_close(corrected[2:], h[2:])
        torch.testing.assert_close(rstd, inverse[:, None].expand_as(h), equal_nan=True)
        doubled, _ = self.k["_correct_xhat"](h, Pointer(inverse * 2), Pointer(shift), indices, valid, True)
        torch.testing.assert_close(doubled, corrected)

    def test_batch_and_group_bf16_fp8_normalized_packing_and_gradients(self):
        for kind in ("batch", "group"):
            for storage in ("bf16", "fp8"):
                for bits in (1, 2, 4, 8):
                    with self.subTest(kind=kind, storage=storage, bits=bits):
                        self._check(kind, (2, 4, 3, 5), storage=storage, bits=bits)

    def test_uncompressed_and_eval_backward(self):
        for kind, shape in (("group", (2, 4)), ("group", (1, 4, 3, 5, 7)), ("batch", (2, 4, 3, 5))):
            with self.subTest(kind=kind, shape=shape):
                self._check(kind, shape, compressed=False)
        self._check("batch", (2, 4, 3, 5), storage="fp8", eval_mode=True)

    def test_large_compression_groups_use_partial_minmax_and_chunked_encode(self):
        self._check("group", (1, 4, 1, 2500), storage="fp8", group=8192)


if __name__ == "__main__":
    unittest.main()
