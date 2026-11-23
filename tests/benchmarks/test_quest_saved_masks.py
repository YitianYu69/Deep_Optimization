"""Exact saved-mask derivatives must never inherit the floating-value codec's error."""

from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from activation_compression import toy_quest_saved_linear as toy


class QuestExactMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def compare(self, factory, source, method, bits, dtype, upstream=None):
        model = factory()
        reference = deepcopy(model)
        x, native = source.clone().requires_grad_(), source.clone().requires_grad_()
        codec = toy.SavedActivations(model, method, bits, 32, 8)
        with codec.capture(), toy.amp(source.device, dtype):
            actual = model(x + 0.)
        with toy.amp(source.device, dtype):
            expected = reference(native + 0.)
        if upstream is None:
            upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(x.grad, native.grad, rtol=0, atol=0, equal_nan=True)
        return codec

    def test_mask_only_modules_match_native_at_every_value_bit_width(self):
        factories = (nn.ReLU, lambda: nn.ReLU(inplace=True), nn.ReLU6, lambda: nn.ReLU6(inplace=True),
                     lambda: nn.LeakyReLU(.1), lambda: nn.LeakyReLU(.1, inplace=True),
                     lambda: nn.Hardtanh(-.5, .75))
        for dtype in (torch.float32, torch.bfloat16):
            source = torch.randn(7, 65, dtype=dtype).T
            for factory in factories:
                for method in toy.METHODS:
                    for bits in (1, 2, 4, 8):
                        with self.subTest(dtype=dtype, module=type(factory()).__name__, method=method, bits=bits):
                            codec = self.compare(factory, source, method, bits, dtype)
                            summary = codec.summary()
                            self.assertEqual(summary["exact_mask_tensors"], 1)
                            self.assertEqual(summary["lossy_value_tensors"], 0)
                            self.assertEqual(summary["activation_bytes_after"], 4 * math.ceil(source.numel() / 32))

    def test_relu_functionals_are_scoped_to_their_own_backward_save(self):
        class FunctionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(16, 32))

            def forward(self, x):
                return F.linear(F.relu(x), self.weight)

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = FunctionalModel()
                reference = deepcopy(model)
                source = torch.randn(3, 32)
                x, native = source.clone().requires_grad_(), source.clone().requires_grad_()
                dy = torch.randn(3, 16).to(dtype)
                codec = toy.SavedActivations(model, "minmax", 2, 16, 8)
                with codec.capture(), toy.amp(torch.device("cpu"), dtype):
                    actual = model(x)
                with toy.amp(torch.device("cpu"), dtype):
                    expected = reference(native)
                actual.backward(dy)
                expected.backward(dy)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, native.grad, rtol=0, atol=0)
                relu_output = source.relu().to(dtype)
                reconstructed = toy.decode(toy.encode(relu_output, "minmax", 2, 16, 8), dtype)
                reconstructed = reconstructed.clamp(min=relu_output.min(), max=relu_output.max())
                torch.testing.assert_close(model.weight.grad, (dy.T @ reconstructed).float(), rtol=0, atol=0)
                self.assertEqual(codec.summary()["exact_mask_tensors"], 1)
                self.assertEqual(codec.summary()["lossy_value_tensors"], 1)

    def test_thresholds_special_values_and_nonfinite_upstream_match_native(self):
        factories = (nn.ReLU, lambda: nn.ReLU(inplace=True), nn.ReLU6, lambda: nn.ReLU6(inplace=True),
                     lambda: nn.LeakyReLU(.1), lambda: nn.LeakyReLU(.1, inplace=True), nn.Hardtanh)
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            source = torch.tensor([-float("inf"), -7, -1, -0., 0., 1e-6, 1., 5.99, 6., 6.01,
                                   float("inf"), float("nan")], dtype=dtype)
            upstream = torch.tensor([float("nan"), -float("inf"), 3., float("nan"), 1., 0., -3., 4.,
                                     float("nan"), float("inf"), -float("inf"), 7.], dtype=dtype)
            for factory in factories:
                with self.subTest(dtype=dtype, module=type(factory()).__name__):
                    self.compare(factory, source, "quest", 1, dtype, upstream)

    def test_relu_tensor_apis_and_shapes_use_exact_mask_payloads(self):
        class Call(nn.Module):
            def __init__(self, function):
                super().__init__()
                self.function = function

            def forward(self, x):
                return self.function(x)

        functions = (F.relu, torch.relu, torch.relu_, lambda x: x.relu(), lambda x: x.relu_(), F.relu6)
        for shape in ((), (0,), (2, 0, 5), (33,), (3, 65)):
            source = torch.randn(shape)
            for function in functions:
                with self.subTest(shape=shape, function=function):
                    codec = self.compare(lambda: Call(function), source, "quest-centered", 1, torch.float32)
                    if source.numel():
                        self.assertEqual(codec.summary()["activation_bytes_after"], 4 * math.ceil(source.numel() / 32))

    def test_dropout_masks_and_native_scaling_are_lossless(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            for cls, shape in ((nn.Dropout, (3, 65)), (nn.Dropout2d, (2, 8, 7, 5)),
                               (nn.Dropout3d, (2, 8, 3, 5, 7))):
                for probability in (0., .25, .7, 1.):
                    with self.subTest(dtype=dtype, module=cls.__name__, p=probability):
                        model = cls(probability)
                        source = torch.randn(shape, dtype=dtype)
                        x, native = source.clone().requires_grad_(), source.clone().requires_grad_()
                        codec = toy.SavedActivations(model, "quest", 1, 32, 8)
                        torch.manual_seed(19)
                        with codec.capture():
                            actual = model(x)
                        torch.manual_seed(19)
                        baseline = toy.SavedActivations(model, "native", 32, 32, 8)
                        with baseline.capture():
                            expected = model(native)
                        upstream = torch.randn_like(actual)
                        actual.backward(upstream)
                        expected.backward(upstream)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(x.grad, native.grad, rtol=0, atol=0)
                        raw, packed = baseline.summary(), codec.summary()
                        self.assertEqual(raw["activation_bytes_before"] + raw["exact_saved_bytes"],
                                         packed["activation_bytes_before"] + packed["exact_saved_bytes"])
                        if probability not in (0., 1.):
                            self.assertEqual(codec.summary()["exact_mask_tensors"], 1)
                            self.assertEqual(codec.summary()["lossy_value_tensors"], 0)
                            self.assertLess(codec.summary()["activation_bytes_after"], codec.summary()["activation_bytes_before"])

    def test_boolean_saved_tensors_roundtrip_without_float_conversion(self):
        source = torch.randint(2, (3, 65), dtype=torch.bool).T
        codec = toy.SavedActivations(nn.Identity(), "quest", 1, 32, 8)
        saved = codec.pack(source)
        self.assertIsInstance(saved, toy.SavedMask)
        self.assertEqual(saved.data.dtype, torch.int32)
        self.assertEqual(saved.data.numel(), math.ceil(source.numel() / 32))
        restored = codec.unpack(saved)
        self.assertEqual(restored.dtype, torch.bool)
        torch.testing.assert_close(restored, source, rtol=0, atol=0)

    def test_hardtanh_narrow_double_interval_preserves_branch_proxy_precision(self):
        source = torch.tensor([1., 1.000000004, 1.000000007, 1.1], dtype=torch.float64)
        self.compare(lambda: nn.Hardtanh(1.000000001, 1.00000001), source, "quest", 1, torch.float64)

    def test_mask_only_catalog_ops_are_exact_across_noisy_distributions_and_low_bits(self):
        catalog = toy.module_cases()
        for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            args = SimpleNamespace(seed=14, dtype=label, methods=list(toy.METHODS), bits=[1, 2, 4],
                                   group_size=256, hadamard_size=128, classes=3)
            for name in ("relu", "relu-inplace", "relu6", "leaky-relu"):
                factory, shape = catalog[name]
                for distribution in toy.DISTRIBUTIONS:
                    with self.subTest(dtype=label, module=name, distribution=distribution):
                        source = toy.samples(2, math.prod(shape), distribution, torch.Generator().manual_seed(14), .25)
                        rows = toy.graph_probe(factory(), source.reshape(2, *shape), args, torch.device("cpu"), dtype)
                        for row in rows:
                            self.assertEqual(row["status"], "ok", row.get("error"))
                            self.assertEqual(row["errors_vs_native"]["dx"]["max_abs"], 0.)
                            self.assertAlmostEqual(row["errors_vs_native"]["dx"]["cosine"], 1.)
                            if row["method"] != "native":
                                self.assertEqual(row["saved_activations"]["exact_mask_tensors"], 1)
                                self.assertEqual(row["saved_activations"]["lossy_value_tensors"], 0)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    def test_cuda_bf16_mask_only_ops_match_native(self):
        source = torch.randn(3, 65, dtype=torch.bfloat16, device="cuda")
        for factory in (nn.ReLU, lambda: nn.ReLU(inplace=True), nn.ReLU6, lambda: nn.LeakyReLU(.1)):
            for method in toy.METHODS:
                with self.subTest(module=type(factory()).__name__, method=method):
                    self.compare(factory, source, method, 1, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
