"""Independent CPU oracles for bounded TileLang amax kernel bodies.

The interpreter checks arithmetic/indexing only; real frontend construction and
CUDA execution are covered separately in the primitive IR and CUDA suites.
"""

from contextlib import contextmanager, ExitStack, nullcontext
import importlib
import unittest
from unittest.mock import patch

import torch

from tests.opt.test_tilelang_operations import _Language


class ReductionLanguage(_Language):
    def __init__(self):
        super().__init__()
        self.reduction_widths = []

    def reduce_max(self, source, output, dim=-1, clear=True):
        self.reduction_widths.append(source.shape[dim])
        value = source.amax(dim=dim).reshape(output.shape)
        output.copy_(value if clear else torch.maximum(output, value))


@contextmanager
def interpreted_amax():
    language = ReductionLanguage()
    modules = [importlib.import_module("deepopt.opt.backends.tilelang." + name)
               for name in ("quantization.fp8", "scaling.reduction")]
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.device", lambda *args: nullcontext()))
        for module in modules:
            for value in vars(module).values():
                if hasattr(value, "cache_clear"):
                    value.cache_clear()
            stack.enter_context(patch.object(module, "runtime", lambda: (None, language)))
            stack.enter_context(patch.object(module, "compile_kernel", language.compile))
            stack.enter_context(patch.object(module, "require_cuda", lambda value, *args: value.device))
        try:
            yield language
        finally:
            for module in modules:
                for value in vars(module).values():
                    if hasattr(value, "cache_clear"):
                        value.cache_clear()


def reference_amax(value, blocks):
    m, n = value.shape
    br, bc = blocks or (m, n)
    return torch.stack([torch.stack([value[r:r + br, c:c + bc].float().abs().amax()
        for c in range(0, n, bc)]) for r in range(0, m, br)])


def reference_scales(value, maximum, blocks):
    magnitude = reference_amax(value, blocks)
    # Tensor/Tensor division expresses the contract's IEEE division directly;
    # Python-scalar/Tensor uses a reciprocal-and-multiply path in Torch.
    multiplier = torch.true_divide(torch.full_like(magnitude, maximum), magnitude.clamp_min(1e-12))
    multiplier = torch.where(magnitude == 0, torch.ones_like(multiplier), multiplier)
    return multiplier, torch.true_divide(torch.ones_like(multiplier), multiplier)


def reference_quantize(value, dtype, blocks):
    maximum = torch.finfo(dtype).max
    scale, inverse = reference_scales(value, maximum, blocks)
    expanded = (scale if blocks is None else scale.repeat_interleave(blocks[0], 0).repeat_interleave(blocks[1], 1))
    values = value.float() * expanded[:value.shape[0], :value.shape[1]]
    return values.clamp(-maximum, maximum).to(dtype), inverse


class TileLangAmaxTests(unittest.TestCase):
    def test_large_groups_do_not_enter_single_lane_serial_scanners(self):
        from deepopt.opt.backends.tilelang.quantization import fp8
        from deepopt.opt.backends.tilelang.scaling import reduction
        values = torch.arange(1025.).reshape(1, -1)
        with interpreted_amax(), patch.object(fp8, "_statistics", side_effect=AssertionError("serial statistics")), \
                patch.object(reduction, "_amax", side_effect=AssertionError("serial amax")):
            scale, _ = fp8.amax_scale(values, 448., None, True)
            actual = reduction.block_amax(values, 1, 1025, minimum=0)
            torch.testing.assert_close(scale, torch.tensor([[448. / 1024.]]), rtol=0, atol=0)
            torch.testing.assert_close(actual, torch.tensor([[1024.]]), rtol=0, atol=0)

    def test_parallel_tensorwise_strides_grid_tails_and_small_boundary(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import amax_scale
        from deepopt.opt.backends.tilelang.scaling.reduction import block_amax, AMAX_CHUNK
        dense = torch.linspace(-7, 11, 33 * 129).reshape(33, 129)
        storage = torch.linspace(-17, 19, 78 * 70).reshape(78, 70)
        strided = storage[1::2, 2::2].T
        cases = ((dense, None), (dense, (17, 65)), (strided, None), (strided, (9, 33)),
                 (torch.arange(256.).reshape(16, 16), None),
                 (torch.arange(257.).reshape(1, 257), None),
                 (torch.zeros(3, 1025), None), (torch.full((1, 1025), 1e-15), None))
        with interpreted_amax() as language:
            for value, blocks in cases:
                for maximum in (448., 57344.):
                    with self.subTest(shape=value.shape, strides=value.stride(), blocks=blocks, maximum=maximum):
                        actual = amax_scale(value, maximum, blocks, True)
                        expected = reference_scales(value, maximum, blocks)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                br, bc = blocks or tuple(value.shape)
                for minimum in (0., 1e-12):
                    actual = block_amax(value, br, bc, minimum=minimum)
                    expected = reference_amax(value, blocks).clamp_min(minimum)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(language.reduction_widths)
            self.assertLessEqual(max(language.reduction_widths), AMAX_CHUNK)

    def test_nonfinite_priority_at_first_last_and_different_chunks(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import amax_scale
        from deepopt.opt.backends.tilelang.scaling.reduction import block_amax
        with interpreted_amax():
            for position, other in ((0, float("inf")), (2050, -float("inf")), (1024, 31.)):
                values = torch.full((1, 2051), 0.25)
                values[0, 1023] = other
                values[0, position] = float("nan")
                with self.subTest(nan_position=position, other=other):
                    scale, inverse = amax_scale(values, 448., None, False)
                    self.assertTrue(torch.isnan(scale).all())
                    self.assertTrue(torch.isnan(inverse).all())
                    self.assertTrue(torch.isnan(block_amax(values, 1, 2051)).all())
                    with self.assertRaisesRegex(ValueError, "finite inputs"):
                        amax_scale(values, 448., None, True)
            values = torch.zeros(1, 1025)
            values[0, -1] = float("inf")
            scale, inverse = amax_scale(values, 448., None, False)
            torch.testing.assert_close(scale, torch.zeros(1, 1), rtol=0, atol=0)
            self.assertTrue(torch.isposinf(inverse).all())
            self.assertTrue(torch.isposinf(block_amax(values, 1, 1025)).all())

    def test_multilevel_merge_retains_nan_infinity_zero_and_maximum(self):
        from deepopt.opt.backends.tilelang.scaling.reduction import _merge_amax, AMAX_CHUNK
        from deepopt.opt.backends.tilelang.quantization.fp8 import _scales_from_amax
        with interpreted_amax() as language:
            maxima = torch.zeros(4, AMAX_CHUNK + 1)
            kinds = torch.zeros_like(maxima, dtype=torch.int32)
            maxima[0, -1] = 23.
            maxima[1, 0] = float("inf")
            kinds[1, 0], kinds[1, -1] = 1, 2
            maxima[2, -1], kinds[2, -1] = float("inf"), 1
            count = maxima.shape[1]
            while count > 1:
                following = (count + AMAX_CHUNK - 1) // AMAX_CHUNK
                a = torch.empty((4, following))
                k = torch.empty((4, following), dtype=torch.int32)
                _merge_amax(4, count, following, "cpu")(maxima, kinds, a, k)
                maxima, kinds, count = a, k, following
            scale, inverse, errors = torch.empty(4), torch.empty(4), torch.empty(4, dtype=torch.int32)
            _scales_from_amax(4, 448., "cpu")(maxima.view(-1), kinds.view(-1), scale, inverse, errors)
            torch.testing.assert_close(scale, torch.tensor([448. / 23., float("nan"), 0., 1.]), equal_nan=True)
            torch.testing.assert_close(errors, torch.tensor([0, 1, 1, 0], dtype=torch.int32))
            self.assertEqual(max(language.reduction_widths), AMAX_CHUNK)

    def test_quantized_bytes_preserve_format_and_rectangular_scale_grid(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import quantize
        with interpreted_amax():
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32):
                values = torch.arange(-769, 770).reshape(1, -1).to(dtype)
                for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
                    for blocks in (None, (1, 513)):
                        with self.subTest(dtype=dtype, fp8=fp8, blocks=blocks):
                            data, inverse = quantize(values, fp8, blocks, True)
                            expected_data, expected_inverse = reference_quantize(values, fp8, blocks)
                            torch.testing.assert_close(inverse, expected_inverse, rtol=0, atol=0)
                            self.assertTrue(torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8)))

    def test_partial_source_offset_widens_before_two_billion_elements(self):
        from deepopt.opt.backends.tilelang.scaling import reduction
        language = ReductionLanguage()
        language.program = (torch.tensor(2**21, dtype=torch.int32),)

        class VirtualStorage:
            def __init__(self):
                self.reads, self.writes = [], {}

            def __getitem__(self, index):
                self.reads.append(int(index))
                return torch.tensor(7.)

            def __setitem__(self, index, value):
                self.writes[tuple(int(v) for v in index)] = value

        x, a, k = VirtualStorage(), VirtualStorage(), VirtualStorage()
        reduction._partial_amax.cache_clear()
        try:
            with patch.object(reduction, "runtime", lambda: (None, language)), \
                    patch.object(reduction, "compile_kernel", lambda function, device: function):
                kernel = reduction._partial_amax(1, 2**31 + 3, 2**31 + 3, 1, 2**31 + 3, "float32",
                    1, 2**31 + 3, 2**21 + 1, "cuda:0")
                kernel(x, a, k)
        finally:
            reduction._partial_amax.cache_clear()
        self.assertEqual(x.reads, [2**31, 2**31 + 1, 2**31 + 2])
        self.assertEqual(float(a.writes[(0, 2**21)]), 7.)
        self.assertEqual(int(k.writes[(0, 2**21)]), 0)


if __name__ == "__main__":
    unittest.main()
