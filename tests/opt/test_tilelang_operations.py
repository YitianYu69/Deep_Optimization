"""Execute TileLang semantic kernel bodies on CPU against independent oracles.

This harness checks indexing and arithmetic without pretending to validate
TileLang's CUDA compiler, synchronization lowering, or GPU execution.
"""

from contextlib import contextmanager, nullcontext
import importlib
from itertools import product
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class _Probe(Exception):
    pass


class _Language:
    """Scalar interpreter for the deliberately simple correctness schedules."""

    def __init__(self):
        self.program = None
        self.grid = None

    @staticmethod
    def Tensor(*args):
        return torch.Tensor

    @staticmethod
    def prim_func(function):
        return function

    def Kernel(self, *grid, threads=128):
        language = self

        class Frame:
            def __enter__(self):
                if language.program is None:
                    language.grid = tuple(int(v) for v in grid)
                    raise _Probe()
                return language.program[0] if len(grid) == 1 else language.program

            def __exit__(self, *args):
                return False

        return Frame()

    @staticmethod
    def ceildiv(a, b):
        return (int(a) + int(b) - 1) // int(b)

    @staticmethod
    def Parallel(*shape):
        return range(int(shape[0])) if len(shape) == 1 else product(*(range(int(v)) for v in shape))

    serial = staticmethod(lambda *args: range(*(int(v) for v in args)))
    unroll = serial
    get_thread_binding = staticmethod(lambda: 0)
    alloc_var = staticmethod(lambda dtype: 0.0 if "float" in dtype else 0)
    alloc_local = staticmethod(lambda shape, dtype: torch.zeros(shape, dtype=getattr(torch, "int64" if dtype in ("uint32", "uint64") else dtype)))
    alloc_shared = alloc_local
    alloc_fragment = alloc_local
    infinity = staticmethod(lambda dtype: float("inf"))
    @staticmethod
    def cast(value, dtype):
        if dtype in ("uint32", "uint64"):
            return torch.as_tensor(value).to(torch.int64) & ((1 << 32) - 1) if dtype == "uint32" else torch.as_tensor(value).to(torch.int64)
        return torch.as_tensor(value).to(getattr(torch, dtype))
    if_then_else = staticmethod(lambda condition, yes, no: yes if bool(condition) else no)
    exp = staticmethod(lambda value: torch.exp(torch.as_tensor(value)))
    tanh = staticmethod(lambda value: torch.tanh(torch.as_tensor(value)))
    erf = staticmethod(lambda value: torch.erf(torch.as_tensor(value)))
    sqrt = staticmethod(lambda value: torch.sqrt(torch.as_tensor(value)))
    rsqrt = staticmethod(lambda value: torch.rsqrt(torch.as_tensor(value)))
    abs = staticmethod(lambda value: torch.abs(torch.as_tensor(value)))
    pow = staticmethod(lambda value, exponent: torch.pow(torch.as_tensor(value), exponent))
    max = staticmethod(lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)))
    min = staticmethod(lambda a, b: torch.minimum(torch.as_tensor(a), torch.as_tensor(b)))
    round = staticmethod(lambda value: torch.round(torch.as_tensor(value)))
    floor = staticmethod(lambda value: torch.floor(torch.as_tensor(value)))
    ieee_fdiv = staticmethod(lambda a, b: torch.true_divide(torch.as_tensor(a), torch.as_tensor(b)))
    clear = staticmethod(lambda value: value.zero_())
    float32 = staticmethod(lambda value: torch.tensor(value, dtype=torch.float32))
    uint32 = staticmethod(lambda value: int(value) & 0xffffffff)
    uint64 = staticmethod(lambda value: int(value) & 0xffffffffffffffff)

    @staticmethod
    def gemm(a, b, output, *, transpose_B=False, clear_accum=False):
        value = a.float() @ (b.float().T if transpose_B else b.float())
        output.copy_(value if clear_accum else output + value)

    def compile(self, function, device, **kwargs):
        def launch(*args):
            self.program = None
            try:
                function(*args)
            except _Probe:
                pass
            for coordinates in product(*(range(v) for v in self.grid)):
                self.program = coordinates
                function(*args)
            self.program = None
        return launch


@contextmanager
def interpreted_kernels():
    language = _Language()
    names = (
        "activation.common", "activation.elementwise", "normalization.common", "pooling", "convolution",
        "quantization.fp8", "quantization.delayed", "quantization.activation.affine",
        "quantization.activation.indices", "scaling.reduction", "gradient.reduction",
    )
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.device", lambda *args: nullcontext()))
        stack.enter_context(patch("torch.cuda.get_device_capability", lambda *args: (9, 0)))
        for name in names:
            module = importlib.import_module("deepopt.opt.backends.tilelang." + name)
            for value in vars(module).values():
                if hasattr(value, "cache_clear"):
                    value.cache_clear()
            stack.enter_context(patch.object(module, "runtime", lambda: (None, language)))
            stack.enter_context(patch.object(module, "compile_kernel", language.compile))
            if hasattr(module, "require_cuda"):
                stack.enter_context(patch.object(module, "require_cuda", lambda *args: next(x.device for x in args if isinstance(x, torch.Tensor))))
        try:
            yield language
        finally:
            for name in names:
                module = importlib.import_module("deepopt.opt.backends.tilelang." + name)
                for value in vars(module).values():
                    if hasattr(value, "cache_clear"):
                        value.cache_clear()


@contextmanager
def pooling_value_oracle_scope(family):
    """Native max-unpool rejects deterministic mode even for unique indices."""
    if family != "unpool":
        yield
        return
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class TileLangOperationTests(unittest.TestCase):
    def kernel_context(self):
        return interpreted_kernels()

    def test_unpool_value_oracle_restores_determinism_even_after_failure(self):
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        torch.use_deterministic_algorithms(True, warn_only=False)
        with self.assertRaisesRegex(RuntimeError, "oracle failure"):
            with pooling_value_oracle_scope("unpool"):
                self.assertFalse(torch.are_deterministic_algorithms_enabled())
                raise RuntimeError("oracle failure")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_element_addressing_widens_before_crossing_two_billion_elements(self):
        from deepopt.opt.backends.tilelang.activation import common
        language = _Language()
        language.program = (torch.tensor(2**23, dtype=torch.int32),)

        class VirtualStorage:
            def __init__(self):
                self.writes = {}

            def __getitem__(self, index):
                return torch.tensor(int(index) % 17, dtype=torch.float32)

            def __setitem__(self, index, value):
                self.writes[int(index)] = value

        source, result = VirtualStorage(), VirtualStorage()
        common._copy_kernel.cache_clear()
        with patch.object(common, "runtime", lambda: (None, language)), patch.object(common, "compile_kernel", lambda function, device: function):
            kernel = common._copy_kernel(2**31 + 3, "float32", "float32", "cuda:0")
            kernel(source, result)
        common._copy_kernel.cache_clear()
        self.assertEqual(set(result.writes), {2**31, 2**31 + 1, 2**31 + 2})

    def test_mixed_auxiliary_routes_and_rectangular_provided_scales(self):
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt.backends.tilelang.activation import common
        calls = []

        def converter(values, scale, dtype, **options):
            calls.append(("convert", tuple(values.shape)))
            metadata = torch.stack((scale.reshape(-1)[0], scale.reshape(-1)[0].reciprocal()))
            return ((values * scale.reshape(-1)[0]).to(dtype), None, metadata,
                    values.abs().amax().reshape(1) if options["observe"] else None, None)

        def module(family, path):
            calls.append((family, path))
            if path == "quantization.delayed":
                return SimpleNamespace(__name__="deepopt.opt.backends.triton.quantization.delayed", convert=converter)
            return SimpleNamespace(__name__="deepopt.opt.backends.triton." + path,
                amax_scale=lambda *args: (torch.ones(1, 1), torch.ones(1, 1)))

        policy = dict(backend="mixed", preferred="tilelang", families={"quantization": "triton", "scaling": "triton"})
        with kernel_policy_scope(policy), patch("deepopt.opt.backends.dispatch.backend_module", module):
            values = torch.randn(3, 7)
            data, inverse, snapshot, observed = common.convert(values, torch.float8_e5m2, (2, 4),
                torch.full((2, 2), 2.0), observe=True, capture_scale=True)
            torch.testing.assert_close(data.float(), (values * 2).to(torch.float8_e5m2).float())
            torch.testing.assert_close(inverse, torch.full((2, 2), 0.5))
            torch.testing.assert_close(snapshot, torch.tensor([2.0, 0.5]))
            torch.testing.assert_close(observed, values.abs().amax().reshape(1))
            common.check_finite(values)
        self.assertIn(("quantization", "quantization.fp8"), calls)
        self.assertIn(("quantization", "quantization.delayed"), calls)
        self.assertIn(("scaling", "quantization.delayed"), calls)
        self.assertIn(("scaling", "quantization.fp8"), calls)

    def test_all_six_activation_values_derivatives_and_masks(self):
        from deepopt.opt.backends.tilelang.activation import elementwise
        from deepopt.opt.backends.reference.activation.math import evaluate
        from deepopt.opt.nn.activation.geometry import ReLUGeometry
        with self.kernel_context():
            x = torch.linspace(-7, 7, 65).reshape(5, 13)
            g = ReLUGeometry(tuple(x.shape))
            for kind in ("relu", "relu6", "sigmoid", "silu", "gelu", "tanh"):
                for approximation in (("none", "tanh") if kind == "gelu" else ("none",)):
                    expected, derivative = (torch.relu(x), ~(x <= 0)) if kind == "relu" else evaluate(x, kind, approximation)
                    actual = elementwise.forward(x, g, kind=kind, approximate=approximation, dtype=torch.float32,
                        block_shape=None, packed=False, bits=2, group_size=17, save_mask=True, inplace=False)
                    torch.testing.assert_close(actual[0], expected)
                    dy = torch.randn_like(x)
                    dx = elementwise.backward(dy, actual[2], g, dtype=torch.float32)
                    torch.testing.assert_close(dx, dy * derivative)

    def test_normalization_values_affine_gradients_and_dx(self):
        from deepopt.opt.backends.tilelang import normalization as tile
        from deepopt.opt.backends.reference import normalization as reference
        from deepopt.opt.nn.normalization.geometry import NormGeometry
        with self.kernel_context():
            for kind in ("batch", "group"):
                x = torch.randn(2, 4, 2, 3)
                weight, bias = torch.randn(4), torch.randn(4)
                g = NormGeometry(kind, tuple(x.shape), groups=2)
                expected = reference.forward(x, weight, bias, g, dict(enabled=False))
                actual = tile.forward(x, weight, bias, g, dict(enabled=False))
                for a, b in zip(actual[:3], expected[:3]):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
                dy = torch.randn_like(x)
                args = dict(x=x, mean=actual[1], inverse=actual[2], input_dtype=torch.float32)
                for a, b in zip(tile.backward(dy, weight, g, **args), reference.backward(dy, weight, g, **args)):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_layernorm_multidimensional_suffix_and_moment_correction(self):
        from deepopt.opt.backends.tilelang.normalization import layer as tile
        from deepopt.opt.backends.reference.normalization import layer as reference
        from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
        from deepopt.opt.backends.tilelang.normalization.common import _moments_kernel, xhat
        with self.kernel_context():
            x = torch.randn(2, 3, 4)
            weight, bias = torch.randn(3, 4), torch.randn(3, 4)
            g = LayerNormGeometry(tuple(x.shape), (3, 4), layout="channels")
            options = dict(weight=weight, bias=bias, dtype=torch.float32, block_shape=None,
                compression=dict(enabled=False, group_size=17, bits=2), check_finite=False)
            actual, expected = tile.forward(x, g, **options), reference.forward(x, g, **options)
            torch.testing.assert_close(actual[0], expected[0])
            dy = torch.randn_like(x)
            args = dict(packed=None, x=x, mean=actual[2].mean, inverse=actual[2].inverse,
                need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32)
            for a, b in zip(tile.backward(dy, weight, g, **args), reference.backward(dy, weight, g, **args)):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
            original, decoded = torch.randn(2, 12), torch.randn(2, 12)
            shift, gain = torch.empty(2), torch.empty(2)
            _moments_kernel(2, 12, True, str(x.device))(original, decoded, shift, gain)
            corrected = xhat(decoded, None, torch.ones(2), reconstruct=False, center_shift=shift, moment_gain=gain)
            torch.testing.assert_close(corrected.mean(1), torch.zeros(2), atol=1e-6, rtol=0)
            torch.testing.assert_close(corrected.square().mean(1), original.square().mean(1))

    def test_activation_packed_state_all_bits_and_fp8_scale_grids(self):
        from deepopt.opt.backends.tilelang.activation import elementwise
        from deepopt.opt.backends.reference.activation import elementwise as reference
        from deepopt.opt.nn.activation.geometry import ReLUGeometry
        with self.kernel_context():
            x = torch.randn(3, 7)
            dy = torch.randn_like(x)
            g = ReLUGeometry(tuple(x.shape))
            for bits in (1, 2, 4, 8):
                args = dict(kind="silu", approximate="none", dtype=torch.float32, block_shape=None,
                    packed=True, bits=bits, group_size=17, save_mask=True, inplace=False)
                actual, expected = elementwise.forward(x, g, **args), reference.forward(x, g, **args)
                torch.testing.assert_close(actual[2].data, expected[2].data, atol=0, rtol=0)
                torch.testing.assert_close(actual[2].step, expected[2].step)
                torch.testing.assert_close(elementwise.backward(dy, actual[2], g, dtype=torch.float32),
                    reference.backward(dy, expected[2], g, dtype=torch.float32))
            args.update(dtype=torch.float8_e5m2, block_shape=(2, 4), packed=False)
            for scale in (None, torch.full((2, 2), 2.0)):
                actual, expected = elementwise.forward(x, g, **args, scale=scale), reference.forward(x, g, **args, scale=scale)
                torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=0, rtol=0)
                torch.testing.assert_close(actual[1], expected[1])

    def test_compressed_norm_backward_without_saved_mean(self):
        from deepopt.opt.backends.tilelang.normalization import layer as tile
        from deepopt.opt.backends.reference.normalization import layer as reference
        from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
        with self.kernel_context():
            x = torch.randn(3, 7)
            dy = torch.randn_like(x)
            weight, bias = torch.randn(7), torch.randn(7)
            g = LayerNormGeometry(tuple(x.shape), (7,))
            shift, gain = torch.empty(3), torch.empty(3)
            args = dict(weight=weight, bias=bias, dtype=torch.float32, block_shape=None,
                compression=dict(enabled=True, group_size=5, bits=2), check_finite=False,
                center_shift=shift, moment_gain=gain)
            actual = tile.forward(x, g, **args)
            opts = dict(packed=actual[2].packed, x=None, mean=None, inverse=actual[2].inverse,
                center_shift=shift, moment_gain=gain, need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32)
            for a, b in zip(tile.backward(dy, weight, g, **opts), reference.backward(dy, weight, g, **opts)):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_twenty_pooling_families_values_routing_and_backward(self):
        from deepopt.opt.backends.tilelang import pooling as tile
        from deepopt.opt.backends.reference import pooling as reference
        from deepopt.opt.nn.pooling.spec import SPECS
        from deepopt.opt.nn.pooling.geometry import geometry
        from tests.opt.test_pooling import case
        with self.kernel_context():
            for kind in SPECS:
                with self.subTest(kind=kind):
                    with pooling_value_oracle_scope(SPECS[kind].family):
                        x, options = case(kind)
                        g = geometry(kind, x, options, "channels")
                    args = dict(options=options, dtype=torch.float32, block_shape=None,
                        compression=dict(enabled=False), check_finite=False, need_state=True,
                        samples=options.get("_random_samples"))
                    with pooling_value_oracle_scope(g.family):
                        actual, expected = tile.forward(x, g, **args), reference.forward(x, g, **args)
                    torch.testing.assert_close(actual[0], expected[0], equal_nan=True, atol=3e-6, rtol=3e-6)
                    if g.routes:
                        torch.testing.assert_close(actual[2].routing, expected[2].routing)
                    if actual[2].public_indices is not None:
                        torch.testing.assert_close(actual[2].public_indices, expected[2].public_indices)
                    dy = torch.randn(g.output_shape)
                    values = []
                    for state, backend in ((actual[2], tile), (expected[2], reference)):
                        values.append(backend.backward(dy, g, routing=state.routing, coefficient=state.coefficient,
                            factor=state.factor, samples=state.samples, coefficient_layout=state.coefficient_layout,
                            input_dtype=torch.float32))
                    torch.testing.assert_close(*values, equal_nan=True, atol=3e-6, rtol=3e-6)

    def test_packed_pool_routes_lp_state_nan_ties_and_infinite_power(self):
        from deepopt.opt.backends.tilelang import pooling as tile
        from deepopt.opt.backends.reference import pooling as reference
        from deepopt.opt.nn.pooling.geometry import geometry
        from deepopt.opt.quantization.indices import unpack_indices
        cases = (
            ("max_pool1d", torch.full((1, 1, 5), -float("inf")), dict(kernel_size=2, stride=2, return_indices=True)),
            ("max_pool1d", torch.tensor([[[1., float("nan"), float("nan"), 2., 2.]]]), dict(kernel_size=3, stride=1, return_indices=True)),
            ("lp_pool1d", torch.tensor([[[-3., -2., 0., 2., 3.]]]), dict(norm_type=float("inf"), kernel_size=3, stride=1)),
            ("lp_pool1d", torch.tensor([[[-3., -2., 0., 2., 3.]]]), dict(norm_type=-float("inf"), kernel_size=3, stride=1)),
            ("lp_pool2d", torch.rand(1, 2, 3, 3) + 0.5, dict(norm_type=2.5, kernel_size=2, stride=1)),
        )
        with self.kernel_context():
            for kind, x, options in cases:
                with self.subTest(kind=kind, options=options):
                    g = geometry(kind, x, options, "channels")
                    args = dict(options=options, dtype=torch.float32, block_shape=None,
                        compression=dict(enabled=True, bits=2, group_size=5), check_finite=False, need_state=True)
                    actual, expected = tile.forward(x, g, **args), reference.forward(x, g, **args)
                    torch.testing.assert_close(actual[0], expected[0], equal_nan=True)
                    if actual[2].public_indices is not None:
                        torch.testing.assert_close(actual[2].public_indices, expected[2].public_indices)
                    if actual[2].routing is not None:
                        torch.testing.assert_close(unpack_indices(actual[2].routing), unpack_indices(expected[2].routing))
                    dy = torch.randn(g.output_shape)
                    gradients = []
                    for state, backend in ((actual[2], tile), (expected[2], reference)):
                        gradients.append(backend.backward(dy, g, routing=state.routing, coefficient=state.coefficient,
                            factor=state.factor, samples=state.samples, coefficient_layout=state.coefficient_layout,
                            input_dtype=torch.float32))
                    torch.testing.assert_close(*gradients, equal_nan=True, atol=3e-6, rtol=3e-6)

    def test_convolution_all_spatial_ranks_groups_and_gradients(self):
        from deepopt.opt.backends.tilelang import convolution as tile
        from deepopt.opt.nn.convolution.geometry import ConvGeometry
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from deepopt.opt.backends.reference import convolution as reference
        with self.kernel_context():
            for rank in (1, 2, 3):
                for groups in (1, 2):
                    with self.subTest(rank=rank, groups=groups):
                        x = torch.randn((1, 2) + (3,) * rank)
                        w = torch.randn((2, 2 // groups) + (2,) * rank)
                        g = ConvGeometry.from_tensors(x, w, groups=groups)
                        qx = quantize_fp8(x.movedim(1, -1).reshape(-1, 2))
                        qw = quantize_fp8(w.reshape(2, -1))
                        qg = quantize_fp8(torch.randn(g.grad_matrix_shape))
                        for operation, args in (("forward", (qx, qw)), ("input_gradient", (qg, qw)), ("weight_gradient", (qx, qg))):
                            actual = getattr(tile, operation)(*args, g)
                            expected = getattr(reference, operation)(*args, g)
                            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_convolution_block_scales_mixed_formats_stride_and_dilation(self):
        from deepopt.opt.backends.tilelang import convolution as tile
        from deepopt.opt.backends.reference import convolution as reference
        from deepopt.opt.nn.convolution.geometry import ConvGeometry
        from deepopt.opt.quantization.quantizer import quantize_fp8
        with self.kernel_context():
            x = torch.randn(1, 2, 5)
            w = torch.randn(2, 1, 2)
            g = ConvGeometry.from_tensors(x, w, groups=2, stride=2, padding=1, dilation=2)
            qx = quantize_fp8(x.movedim(1, -1).reshape(-1, 2), block_shape=(2, 1), fp8_dtype=torch.float8_e5m2)
            qw = quantize_fp8(w.reshape(2, -1), block_shape=(1, 2))
            qg = quantize_fp8(torch.randn(g.grad_matrix_shape), block_shape=(2, 1))
            for name, operands in (("forward", (qx, qw)), ("input_gradient", (qg, qw)), ("weight_gradient", (qx, qg))):
                torch.testing.assert_close(getattr(tile, name)(*operands, g), getattr(reference, name)(*operands, g), atol=2e-5, rtol=2e-5)

    def test_transposed_convolution_all_ranks_and_adjoint_gradients(self):
        from deepopt.opt.backends.tilelang import convolution_transpose as tile
        from deepopt.opt.backends.reference import convolution_transpose as reference
        from deepopt.opt.nn.convolution_transpose.geometry import ConvTransposeGeometry
        from deepopt.opt.quantization.quantizer import quantize_fp8
        with self.kernel_context():
            for rank in (1, 2, 3):
                with self.subTest(rank=rank):
                    x = torch.randn((1, 2) + (2,) * rank)
                    w = torch.randn((2, 1) + (2,) * rank)
                    g = ConvTransposeGeometry.from_tensors(x, w, groups=2, stride=2 if rank == 1 else 1,
                        output_padding=1 if rank == 1 else 0)
                    qx = quantize_fp8(x.movedim(1, -1).reshape(-1, 2))
                    qw = quantize_fp8(w.reshape(2, -1))
                    qg = quantize_fp8(torch.randn(g.grad_matrix_shape))
                    for name, operands in (("forward", (qx, qw)), ("input_gradient", (qg, qw)), ("weight_gradient", (qx, qg))):
                        torch.testing.assert_close(getattr(tile, name)(*operands, g, out_dtype=torch.float32),
                            getattr(reference, name)(*operands, g, out_dtype=torch.float32), atol=2e-5, rtol=2e-5)


class TileLangCudaOperationTests(TileLangOperationTests):
    """The same semantic cases using the real compiler and GPU launches."""

    @classmethod
    def setUpClass(cls):
        required = os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1"
        if not torch.cuda.is_available() or importlib.util.find_spec("tilelang") is None:
            error = RuntimeError if required else unittest.SkipTest
            raise error("TileLang CUDA execution requires TileLang and an NVIDIA GPU")
        if torch.cuda.get_device_capability() < (9, 0):
            error = RuntimeError if required else unittest.SkipTest
            raise error("FP8 operation coverage requires compute capability >= 9")

    @contextmanager
    def kernel_context(self):
        from deepopt.kernel_policy import kernel_policy_scope
        with torch.device("cuda"), kernel_policy_scope("tilelang"):
            yield


if __name__ == "__main__":
    unittest.main()
