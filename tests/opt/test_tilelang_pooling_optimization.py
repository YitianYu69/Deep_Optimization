"""Analytic pooling keeps the generic TileLang ownership and codec contracts."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends.tilelang import pooling as tile
from deepopt.opt.nn.pooling.geometry import geometry
from tests.opt.test_tilelang_operations import interpreted_kernels


class TileLangPoolingOptimizationTests(unittest.TestCase):
    def run_case(self, kind, x, options, *, compressed=False, layout="last", generic=False, gradient=None):
        g = geometry(kind, x, options, layout)
        context = patch.object(tile, "_regular_windows", return_value=False) if generic else patch.object(
            tile, "_windows", side_effect=AssertionError("Regular pooling allocated a window table"))
        with context:
            result = tile.forward(x, g, options=options, dtype=torch.float32, block_shape=None,
                compression=dict(enabled=compressed, bits=2, group_size=7), check_finite=False, need_state=True)
            state = result[2]
            gradient = torch.randn(g.output_shape) if gradient is None else gradient
            info = {}
            dx = tile.backward(gradient, g, routing=state.routing, coefficient=state.coefficient,
                factor=state.factor, samples=state.samples, coefficient_layout=state.coefficient_layout,
                input_dtype=torch.float32, execution_info=info)
        return g, result, dx, gradient, info

    def check_native(self, kind, shape, options, *, layout="last", compressed=False):
        x = torch.randn(shape)
        source = x.clone().requires_grad_()
        g, result, dx, gradient, info = self.run_case(kind, x, options, layout=layout, compressed=compressed)
        expected = g.spec.native(source, **options)
        expected, indices = expected if isinstance(expected, tuple) else (expected, None)
        expected.backward(gradient)
        torch.testing.assert_close(g.transport.restore(result[0]), expected, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(dx, source.grad, rtol=3e-6, atol=3e-6)
        if indices is not None:
            torch.testing.assert_close(result[2].public_indices, indices)
        self.assertEqual(result[-1]["forward_variant"], "regular_analytic")
        self.assertEqual(result[-1]["window_metadata_bytes"], 0)
        self.assertEqual(result[-1]["window_launches"], 0)
        self.assertEqual(info["backward_window_launches"], 0)
        self.assertEqual(info["backward_variant"], tile._backward_variant(g))

    def test_max_ranks_dilation_padding_ceil_and_stride_gaps(self):
        cases = (
            ("max_pool1d", (1, 2, 11), dict(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True, return_indices=True)),
            ("max_pool1d", (2, 13), dict(kernel_size=2, stride=4, ceil_mode=True, return_indices=True)),
            ("max_pool2d", (1, 2, 5, 7), dict(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=(2, 1), ceil_mode=True, return_indices=True)),
            ("max_pool2d", (1, 1, 5, 7), dict(kernel_size=2, stride=2, return_indices=True)),
            ("max_pool3d", (1, 1, 4, 5, 6), dict(kernel_size=(2, 3, 2), stride=(2, 1, 3), padding=(0, 1, 0), dilation=(1, 1, 2), ceil_mode=True, return_indices=True)),
        )
        with interpreted_kernels():
            for kind, shape, options in cases:
                for layout in ("last", "channels"):
                    with self.subTest(kind=kind, options=options, layout=layout):
                        self.check_native(kind, shape, options, layout=layout)

    def test_average_ranks_include_pad_divisor_and_ceil_tails(self):
        cases = (
            ("avg_pool1d", (2, 9), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)),
            ("avg_pool2d", (1, 1, 5, 7), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=True)),
            ("avg_pool2d", (1, 2, 5, 6), dict(kernel_size=3, stride=1, padding=1, count_include_pad=False, divisor_override=7)),
            ("avg_pool3d", (1, 1, 3, 4, 5), dict(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False, divisor_override=7)),
        )
        with interpreted_kernels():
            for kind, shape, options in cases:
                with self.subTest(kind=kind, options=options):
                    self.check_native(kind, shape, options)

    def test_ties_nan_and_infinity_preserve_public_and_packed_routes(self):
        from deepopt.opt.quantization.indices import unpack_indices
        x = torch.tensor([[[[2., 2., float("nan"), float("nan"), -float("inf")],
                            [2., 1., 3., 3., -float("inf")], [0., 0., 0., 0., 0.]]]])
        options = dict(kernel_size=(2, 3), stride=1, padding=(0, 1), return_indices=True)
        with interpreted_kernels():
            for compressed in (False, True):
                g = geometry("max_pool2d", x, options, "channels")
                dy = torch.arange(1, 1 + g.output_elements, dtype=torch.float32).reshape(g.output_shape)
                actual = self.run_case("max_pool2d", x, options, compressed=compressed, layout="channels", gradient=dy)
                generic = self.run_case("max_pool2d", x, options, compressed=compressed, layout="channels", generic=True, gradient=dy)
                torch.testing.assert_close(actual[1][0], generic[1][0], rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(actual[1][2].public_indices, generic[1][2].public_indices)
                torch.testing.assert_close(actual[2], generic[2], rtol=0, atol=0, equal_nan=True)
                if compressed:
                    self.assertTrue(torch.equal(actual[1][2].routing.data, generic[1][2].routing.data))
                    self.assertTrue(torch.equal(unpack_indices(actual[1][2].routing), unpack_indices(generic[1][2].routing)))
                source = x.clone().requires_grad_()
                expected, indices = torch.nn.functional.max_pool2d(source, **options)
                expected.backward(dy)
                torch.testing.assert_close(g.transport.restore(actual[1][0]), expected, equal_nan=True)
                torch.testing.assert_close(actual[1][2].public_indices, indices)
                torch.testing.assert_close(actual[2], source.grad)

    def test_bounded_overlap_keeps_generic_accumulation_order(self):
        # Different summation orders lose different low-order contributions.
        # Bounds may skip unrelated windows, but must not reorder contributors.
        x = torch.zeros(1, 1, 11)
        options = dict(kernel_size=5, stride=1, padding=2, count_include_pad=False)
        dy = torch.tensor([[[1e20, 1., -1e20, 3., 1e20, -1e20, 7., 1., -2., 4., 9.]]])
        with interpreted_kernels():
            actual = self.run_case("avg_pool1d", x, options, gradient=dy)
            generic = self.run_case("avg_pool1d", x, options, generic=True, gradient=dy)
        self.assertTrue(torch.equal(actual[2], generic[2]))
        self.assertEqual(actual[-1]["backward_variant"], "regular_bounded")

    def test_nonoverlap_candidate_work_does_not_grow_with_output_volume(self):
        # Count real W reads in generic mode and forbid any analytic W read.
        class Unreadable:
            def __getitem__(self, index):
                raise AssertionError("Analytic kernel read a window table")
        with interpreted_kernels() as language:
            lengths = []
            serial = language.serial
            def counted_serial(*bounds):
                values = serial(*bounds)
                lengths.append(len(values))
                return values
            language.serial = counted_serial
            inputs, outputs = (1, 1, 17), (1, 1, 8)
            kernel = tile._backward_kernel(1, inputs, outputs, (1, 1, 1), (0, 0, 0),
                "max", 1., True, None, "cpu", (1, 1, 2), (1, 1, 2))
            gradient = torch.arange(8, dtype=torch.float32)
            result = torch.full((17,), float("nan"))
            kernel(gradient, Unreadable(), torch.ones(8, dtype=torch.int64), torch.empty(17), torch.empty(8), result)
        expected = torch.zeros(17)
        expected[1:16:2] = gradient
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(lengths, [1] * 16 + [0])

    def test_fp8_transport_matches_generic_payload_scales_and_gradient(self):
        from deepopt.opt.quantization.activation_tensor import FP8Activation
        shape = (1, 2, 5, 7)
        represented = torch.arange(70, dtype=torch.float32).reshape(shape) % 9 - 4
        matrix = represented.movedim(1, -1).reshape(-1, 2).to(torch.float8_e4m3fn)
        x = FP8Activation(matrix, torch.ones(1, 1), shape, layout="channels", logical_dtype=torch.float32)
        options = dict(kernel_size=3, stride=2, padding=1, ceil_mode=True, return_indices=True)
        g = geometry("max_pool2d", x, options, "channels")
        dy = torch.randn(g.output_shape)
        arguments = dict(options=options, dtype=torch.float8_e5m2, block_shape=None,
            compression=dict(enabled=True, bits=2, group_size=7), check_finite=False, need_state=True)
        with interpreted_kernels():
            actual = tile.forward(x, g, **arguments)
            with patch.object(tile, "_regular_windows", return_value=False):
                generic = tile.forward(x, g, **arguments)
            self.assertTrue(torch.equal(actual[0].view(torch.uint8), generic[0].view(torch.uint8)))
            self.assertTrue(torch.equal(actual[1], generic[1]))
            self.assertTrue(torch.equal(actual[2].routing.data, generic[2].routing.data))
            torch.testing.assert_close(actual[2].public_indices, generic[2].public_indices)
            gradients = []
            for result, force_generic in ((actual, False), (generic, True)):
                state = result[2]
                with patch.object(tile, "_regular_windows", return_value=not force_generic):
                    gradients.append(tile.backward(dy, g, routing=state.routing, coefficient=state.coefficient,
                        factor=state.factor, samples=state.samples, coefficient_layout=state.coefficient_layout,
                        input_dtype=torch.float32))
            self.assertTrue(torch.equal(*gradients))

    def test_mixed_codec_dispatch_is_preserved(self):
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt.backends.tilelang.quantization.activation import indices
        codec = SimpleNamespace(pack=indices.pack, unpack=indices.unpack)
        policy = dict(backend="mixed", preferred="tilelang", families={"activation_codec": "triton"})
        with interpreted_kernels(), kernel_policy_scope(policy), patch(
                "deepopt.opt.backends.dispatch.backend_module", return_value=codec) as dispatch:
            self.check_native("max_pool2d", (1, 1, 5, 7), dict(kernel_size=2, stride=2), compressed=True)
        self.assertEqual(dispatch.call_count, 2)
        self.assertTrue(all(call.args == ("activation_codec", "quantization.activation.indices")
                            for call in dispatch.call_args_list))

    def test_unused_buffers_have_scalar_abis_and_reuse_existing_storage(self):
        forward_factory, backward_factory = tile._forward_kernel, tile._backward_kernel
        launches = {}
        def recorded(factory, key):
            def make(*arguments):
                launch = factory(*arguments)
                def call(*buffers):
                    launches[key] = [(tuple(value.shape), value.untyped_storage().data_ptr()) for value in buffers]
                    return launch(*buffers)
                return call
            return make
        with interpreted_kernels(), patch.object(tile, "_forward_kernel", side_effect=recorded(forward_factory, "forward")), \
                patch.object(tile, "_backward_kernel", side_effect=recorded(backward_factory, "backward")):
            for kind in ("max_pool2d", "avg_pool2d"):
                with self.subTest(kind=kind):
                    x = torch.randn(1, 2, 5, 7)
                    g, result, _, _, _ = self.run_case(kind, x, dict(kernel_size=2, stride=2))
                    fw, bw = launches["forward"], launches["backward"]
                    self.assertEqual(fw[3][0], (1,))  # Unreturned public indices.
                    self.assertEqual(fw[5][0], (1,))  # Unused power factor.
                    self.assertEqual(fw[5][1], fw[2][1])  # Scalar view of Y.
                    self.assertEqual(bw[3][0], (1,))  # Unused input coefficient.
                    self.assertEqual(bw[4][0], (1,))  # Unused output factor.
                    self.assertEqual(bw[3][1], bw[5][1])
                    self.assertEqual(bw[4][1], bw[5][1])
                    self.assertEqual(result[-1]["public_indices_buffer_bytes"], 0)
                    self.assertEqual(result[-1]["factor_buffer_bytes"], 0)
                    if kind == "avg_pool2d":
                        self.assertEqual(fw[4][0], (1,))
                        self.assertEqual(bw[2][0], (1,))
                        self.assertEqual(result[-1]["routing_buffer_bytes"], 0)
                    else:
                        self.assertEqual(fw[4][0], (g.batch * g.channels * g.output_elements,))
            x = torch.randn(1, 1, 5, 7)
            options = dict(kernel_size=2, stride=2, return_indices=True)
            g = geometry("max_pool2d", x, options, "last")
            result = tile.forward(x, g, options=options, dtype=torch.float32, block_shape=None,
                compression=dict(enabled=False), check_finite=False, need_state=False)
            expected, indices = torch.nn.functional.max_pool2d(x, **options)
            torch.testing.assert_close(g.transport.restore(result[0]), expected)
            torch.testing.assert_close(result[2].public_indices, indices)
            self.assertIsNone(result[2].routing)
            self.assertEqual(launches["forward"][4][0], (1,))
            self.assertEqual(result[-1]["routing_buffer_bytes"], 0)

    def test_irregular_families_keep_generic_tilelang_windows(self):
        from tests.opt.test_pooling import case
        from deepopt.opt.backends.reference import pooling as reference
        with interpreted_kernels(), patch.object(tile, "_windows", wraps=tile._windows) as windows:
            for kind in ("adaptive_max_pool2d", "adaptive_avg_pool2d", "fractional_max_pool2d", "lp_pool2d"):
                x, options = case(kind)
                g = geometry(kind, x, options, "last")
                arguments = dict(options=options, dtype=torch.float32, block_shape=None,
                    compression=dict(enabled=False), check_finite=False, need_state=True,
                    samples=options.get("_random_samples"))
                actual, expected = tile.forward(x, g, **arguments), reference.forward(x, g, **arguments)
                torch.testing.assert_close(actual[0], expected[0], rtol=3e-6, atol=3e-6)
                self.assertEqual(actual[-1]["forward_variant"], "generic")
                self.assertFalse(tile._regular_windows(g))
                self.assertEqual(tile._backward_variant(g), "generic")
            self.assertEqual(windows.call_count, 4)

    def test_pooling_source_does_not_import_triton(self):
        for path in Path(tile.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(any("triton" in name.split(".") for name in names), (path, names))


class TileLangPoolingOptimizationIRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tilelang") is None:
            if os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1":
                raise RuntimeError("Mandatory TileLang IR validation requires TileLang")
            raise unittest.SkipTest("optional TileLang dependency is absent")

    def test_analytic_and_generic_factories_construct_real_ir(self):
        for rank, inputs, outputs, kernel_size, strides, dilation, padding in (
                (1, (1, 1, 13), (1, 1, 4), (1, 1, 2), (1, 1, 4), (1, 1, 1), (0, 0, 0)),
                (2, (1, 5, 7), (1, 3, 6), (1, 3, 2), (1, 2, 1), (1, 2, 1), (0, 1, 0)),
                (3, (4, 5, 6), (2, 5, 2), (2, 3, 2), (2, 1, 3), (1, 1, 2), (0, 1, 0))):
            for family in ("max", "average", "adaptive_max"):
                regular = family != "adaptive_max"
                for factory in (tile._forward_kernel, tile._backward_kernel):
                    for compact in (False, True):
                        with self.subTest(rank=rank, family=family, factory=factory.__name__, compact=compact):
                            factory.cache_clear()
                            arguments = (1, inputs, outputs, dilation, padding, family, 1., True, None)
                            if factory is tile._forward_kernel:
                                arguments += (kernel_size[0] * kernel_size[1] * kernel_size[2],)
                            arguments += ("cuda:0", kernel_size if regular else None, strides if regular else None)
                            if compact:
                                arguments += (False, family != "average", False) if factory is tile._forward_kernel else (True,)
                            try:
                                with patch.object(tile, "compile_kernel", side_effect=lambda prim, device: prim):
                                    prim = factory(*arguments)
                                self.assertEqual(type(prim).__name__, "PrimFunc")
                                if regular:
                                    self.assertNotIn("W[", prim.script())
                                if compact:
                                    scalar_params = (3, 5) if factory is tile._forward_kernel else (3, 4)
                                    for index in scalar_params:
                                        self.assertEqual(int(prim.buffer_map[prim.params[index]].shape[0]), 1)
                            finally:
                                factory.cache_clear()

    def test_compact_power_and_irregular_buffer_shapes_construct_real_ir(self):
        for family, power in (("power", 2.5), ("power", float("inf")), ("power", -float("inf")),
                              ("adaptive_average", 1.), ("fractional_max", 1.)):
            finite_power = family == "power" and abs(power) != float("inf")
            routes = family == "fractional_max" or family == "power" and not finite_power
            for factory in (tile._forward_kernel, tile._backward_kernel):
                with self.subTest(family=family, power=power, factory=factory.__name__):
                    factory.cache_clear()
                    arguments = (1, (1, 5, 7), (1, 3, 4), (1, 1, 1), (0, 0, 0), family, power, True, None)
                    if factory is tile._forward_kernel:
                        arguments += (4, "cuda:0", None, None, False, routes, finite_power)
                    else:
                        arguments += ("cuda:0", None, None, True)
                    try:
                        with patch.object(tile, "compile_kernel", side_effect=lambda prim, device: prim):
                            prim = factory(*arguments)
                        self.assertEqual(type(prim).__name__, "PrimFunc")
                        if factory is tile._forward_kernel:
                            expected = {3: 1, 4: 12 if routes else 1, 5: 12 if finite_power else 1}
                        else:
                            expected = {2: 12 if routes else 1, 3: 35 if finite_power else 1,
                                        4: 12 if finite_power else 1}
                        for index, count in expected.items():
                            self.assertEqual(int(prim.buffer_map[prim.params[index]].shape[0]), count)
                    finally:
                        factory.cache_clear()


if __name__ == "__main__":
    unittest.main()
