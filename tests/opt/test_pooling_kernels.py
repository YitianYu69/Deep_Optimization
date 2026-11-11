"""Real pooling JIT/launch bodies on CPU, independently checked against native math."""

import unittest

import torch

from deepopt.opt.nn.pooling.spec import SPECS
from deepopt.opt.nn.pooling.geometry import geometry
from deepopt.opt.quantization.activation_tensor import to_fp8_activation, FP8Activation
from deepopt.opt.quantization.indices import unpack_indices, empty_indices, pack_indices
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.gradient import GradientSlot, GradientOutput
from deepopt.opt.precision.gradient import GradientEdgePlan
from deepopt.opt.nn.activation.geometry import ReLUGeometry
from deepopt.opt.backends.reference import pooling as reference
from tests.opt.pooling_kernel_support import KernelHarness
from tests.opt.test_pooling import case


class PoolingKernelTests(unittest.TestCase):
    def setUp(self):
        self.h = KernelHarness()
        torch.manual_seed(4923)
        # Value comparison uses native unpool, whose deterministic mode is
        # unsupported. Restore the caller's mode after each case.
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        torch.use_deterministic_algorithms(False)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    @torch.no_grad()
    def check(self, kind, *, encoded=False, compressed=True, layout="last", dtype=torch.float32,
              bits=2, group=17, blocks=None, scale=None, source=None, options=None):
        if source is None:
            source, options = case(kind, dtype=torch.bfloat16)
        source = source.detach()
        if encoded and not isinstance(source, FP8Activation):
            source = to_fp8_activation(source, layout="channels", block_shape=(2, 4))
        g = geometry(kind, source, options, layout)
        args = dict(options=options, dtype=dtype, block_shape=blocks,
                    compression=dict(enabled=compressed, bits=bits, group_size=group), check_finite=False,
                    need_state=True, samples=options.get("_random_samples"), scale=scale)
        expected = reference.forward(source, g, **args)
        actual = self.h.host["_forward"](source, g, **args)
        torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0.13 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 3e-5, atol=2e-5, equal_nan=True)
        if expected[1] is not None:
            torch.testing.assert_close(actual[1], expected[1], rtol=3e-5, atol=1e-8)
        if actual[2].public_indices is not None:
            torch.testing.assert_close(actual[2].public_indices, expected[2].public_indices, rtol=0, atol=0)
        if g.routes and compressed:
            torch.testing.assert_close(unpack_indices(actual[2].routing), unpack_indices(expected[2].routing), rtol=0, atol=0)
        dy = torch.randn(g.output_shape, dtype=torch.bfloat16)
        if encoded:
            dy = to_fp8_activation(dy, layout="channels", block_shape=(2, 4), fp8_dtype=torch.float8_e5m2)
        results = []
        for state, compute in ((expected[2], reference.backward), (actual[2], self.h.host["_backward"])):
            results.append(compute(dy, g, routing=state.routing, coefficient=state.coefficient, factor=state.factor,
                                   samples=state.samples, coefficient_layout=state.coefficient_layout, input_dtype=torch.float32))
        torch.testing.assert_close(*results, rtol=4e-5, atol=3e-5, equal_nan=True)
        return actual

    def test_all_twenty_families_and_routing_modes(self):
        for kind in SPECS:
            for compressed in (False, True):
                with self.subTest(kind=kind, compressed=compressed):
                    self.check(kind, compressed=compressed)

    def test_fp8_operands_and_outputs_both_transports(self):
        for kind in SPECS:
            for layout in ("last", "channels"):
                with self.subTest(kind=kind, layout=layout):
                    self.check(kind, encoded=True, layout=layout, dtype=torch.float8_e5m2, scale=torch.ones(1, 1))

    def test_lp_every_codec_bit_count_and_group_shape(self):
        for rank in (1, 2, 3):
            for bits in (1, 2, 4, 8):
                self.check(f"lp_pool{rank}d", bits=bits)
                self.check(f"lp_pool{rank}d", bits=bits, encoded=True)
        self.check("lp_pool1d", group=5003, source=torch.rand(1, 2, 5005) + 1,
                   options=dict(norm_type=2, kernel_size=2, stride=2, ceil_mode=False))

    def test_large_global_and_overlapping_adaptive_bins(self):
        for family in ("adaptive_avg_pool", "adaptive_max_pool"):
            for size in ((1, 1), (9, 11)):
                self.check(family + "2d", source=torch.randn(1, 2, 17, 19), options=dict(output_size=size))
        self.check("max_pool2d", source=torch.randn(1, 2, 7, 9),
                   options=dict(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True, return_indices=True))
        self.check("avg_pool2d", source=torch.randn(1, 2, 7, 9),
                   options=dict(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False, divisor_override=7))
        self.check("max_pool2d", source=torch.randn(1, 2, 7, 9),
                   options=dict(kernel_size=(2,), stride=(), padding=(0,), dilation=(1,), return_indices=True))
        self.check("avg_pool3d", source=torch.randn(1, 2, 5, 7, 9),
                   options=dict(kernel_size=(2,), stride=(), padding=(0,)))
        for kind in ("adaptive_avg_pool1d", "adaptive_max_pool1d"):
            result = self.check(kind, source=torch.randn(1, 2, 17003), options=dict(output_size=1))
            self.assertTrue(result[-1]["global_reduction"])
            self.assertEqual(result[-1]["forward_launches"], 2)
            self.assertGreater(result[-1]["scratch_bytes"], 0)
        self.check("lp_pool1d", source=torch.randn(1, 2, 17003), options=dict(norm_type=2, kernel_size=17003))

    def test_exact_index_bitstreams_arbitrary_widths(self):
        for bits in (0, 1, 2, 3, 5, 9, 16, 31, 32, 40, 63):
            limit = 1 << bits
            values = torch.randint(min(limit, 2**62), (5, 74))[:, ::2]
            values[-1, -1] = limit - 1
            result = empty_indices(tuple(values.shape), limit, "cpu")
            if bits:
                self.h.launch("_pack", ((values.numel() + 31) // 32,), values, result.data, values.numel(),
                              tuple(values.shape), values.stride(), bits, 1 << (min(bits, 32) - 1).bit_length())
            torch.testing.assert_close(result.data, pack_indices(values, limit).data, rtol=0, atol=0)
            decoded = torch.empty_like(values, memory_format=torch.contiguous_format)
            self.h.launch("_unpack", ((values.numel() + 255) // 256,), result.data, decoded, values.numel(), bits, 256)
            torch.testing.assert_close(decoded, values, rtol=0, atol=0)

    def test_fused_lp_packing_and_invariant_forward_tiling(self):
        source = torch.randn(2, 3, 8, 16)
        options = dict(norm_type=2, kernel_size=2, stride=2, ceil_mode=False)
        baseline = self.check("lp_pool2d", source=source, options=options, compressed=False)
        for bits in (1, 2, 4, 8):
            self.h.events.clear()
            actual = self.check("lp_pool2d", source=source, options=options, bits=bits, group=512)
            self.assertTrue(actual[-1]["coefficient_fused"])
            self.assertEqual(actual[-1]["packing_launches"], 0)
            torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
        for group in (17, 512, 1024):
            actual = self.check("lp_pool2d", source=source, options=options, group=group)
            torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
        encoded = to_fp8_activation(torch.randn(2, 32, 8, 16), layout="channels", block_shape=(1, 32))
        actual = self.check("lp_pool2d", source=encoded, options=options, layout="channels", bits=2, group=512)
        self.assertTrue(actual[-1]["coefficient_fused"])
        self.assertEqual(actual[-1]["packing_launches"], 0)

    @torch.no_grad()
    def test_completed_fp8_dx_current_and_provided_scales_for_all_families(self):
        for kind in SPECS:
            x, options = case(kind, dtype=torch.bfloat16)
            x = x.detach()
            g = geometry(kind, x, options, "channels")
            args = dict(options=options, dtype=torch.bfloat16, block_shape=None,
                        compression=dict(enabled=True, bits=2, group_size=17), check_finite=False,
                        need_state=True, samples=options.get("_random_samples"))
            state = self.h.host["_forward"](x, g, **args)[2]
            dy = to_fp8_activation(torch.randn(g.output_shape), fp8_dtype=torch.float8_e5m2)
            backward = dict(routing=state.routing, coefficient=state.coefficient, factor=state.factor,
                            samples=state.samples, coefficient_layout=state.coefficient_layout, input_dtype=torch.float32)
            expected = reference.backward(dy, g, **backward)
            for layout, blocks, provided in (("last", (1, 1), False), ("channels", (2, 4), False), ("channels", None, True)):
                edge = GradientEdgePlan("fp8", "e5m2", layout, blocks)
                slot = GradientSlot(edge, {}, device="cpu")
                out = GradientOutput(slot, g.shape, torch.bfloat16, layout, check_finite=False)
                if provided:
                    out.scale = torch.tensor([2.])
                actual = self.h.host["_backward"](dy, g, gradient_output=out, **backward)
                matrix = ReLUGeometry(g.shape, layout).matrix(expected)
                wanted = (matrix * 2).clamp(-57344, 57344).to(torch.float8_e5m2) if provided else quantize_fp8(matrix, fp8_dtype=torch.float8_e5m2, block_shape=blocks).data
                torch.testing.assert_close(actual._data.float(), wanted.float(), rtol=0.25, atol=0.001)

    def test_infinite_powers_zero_values_and_tie_routing(self):
        for rank in (1, 2, 3):
            source = torch.tensor([0., -1., 1., -2., 2.]).repeat(5**(rank - 1)).reshape((1, 1) + (5,) * rank)
            for power in (float("inf"), -float("inf")):
                self.check(f"lp_pool{rank}d", source=source, options=dict(norm_type=power, kernel_size=2, stride=1, ceil_mode=True))
        self.check("lp_pool2d", source=torch.zeros(1, 2, 4, 4), options=dict(norm_type=2, kernel_size=2))


if __name__ == "__main__":
    unittest.main()
