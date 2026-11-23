"""Operator-state regressions for derivative and coupled normalization backward."""

from copy import deepcopy
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from activation_compression import toy_quest_saved_linear as toy


class RecordingCodec(toy.SavedActivations):
    """Retain oracle copies only in tests, never in the experiment's codec."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def pack(self, value):
        state = self.pending_states.get(id(value))
        result = super().pack(value)
        if state is not None:
            self.states.append((state["kind"], value.detach().clone(), result))
        return result


class LosslessCoefficientCodec(RecordingCodec):
    """Exercise the new backward formulas separately from quantizer accuracy."""
    def pack_preprocessed(self, value, record, state):
        record.update(kind="native_activation", encoding="native_" + state["kind"])
        self.records.append(record)
        return value.detach()


class QuestDerivativeStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_smooth_activations_save_derivatives_and_use_them_directly(self):
        factories = (nn.Sigmoid, nn.Tanh, nn.GELU, lambda: nn.GELU(approximate="tanh"),
                     nn.SiLU, lambda: nn.SiLU(inplace=True), lambda: nn.ELU(.7),
                     lambda: nn.ELU(.7, inplace=True), lambda: nn.Softplus(beta=1.7, threshold=4.))
        for dtype in (torch.float32, torch.bfloat16):
            source = (torch.randn(7, 65) * 2 + 1).to(dtype).T
            for factory in factories:
                native = source.float().clone().requires_grad_()
                reference = factory()(native + 0.)
                reference.sum().backward()
                derivative = native.grad
                for method in toy.METHODS:
                    for bits in (1, 2, 4, 8):
                        with self.subTest(dtype=dtype, module=type(factory()).__name__, method=method, bits=bits):
                            model = factory()
                            x = source.clone().requires_grad_()
                            codec = RecordingCodec(model, method, bits, 32, 8)
                            with codec.capture():
                                actual = model(x + 0.)
                            expected = factory()(source.clone())
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            self.assertEqual(len(codec.states), 1)
                            kind, coefficients, saved = codec.states[0]
                            self.assertEqual(kind, "derivative")
                            torch.testing.assert_close(coefficients, derivative, rtol=2e-6, atol=2e-7)
                            dy = torch.randn_like(actual)
                            actual.backward(dy)
                            oracle = (dy.float() * codec.unpack(saved)).to(dtype)
                            torch.testing.assert_close(x.grad, oracle, rtol=0, atol=0)
                            self.assertEqual(codec.summary()["derivative_tensors"], 1)
                            self.assertEqual(codec.summary()["lossy_value_tensors"], 0)
                            self.assertFalse(codec.pending_states)

    def test_functional_and_tensor_activations_keep_consumer_magnitudes(self):
        class Model(nn.Module):
            def __init__(self, activation):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(16, 32))
                self.activation = activation

            def forward(self, x):
                return F.linear(self.activation(x), self.weight)

        functions = (lambda x: F.gelu(input=x, approximate="tanh"), F.silu,
                     torch.sigmoid, lambda x: x.sigmoid(), lambda x: x.sigmoid_(),
                     torch.tanh, lambda x: x.tanh(), lambda x: x.tanh_())
        for activation in functions:
            with self.subTest(activation=activation):
                model = Model(activation)
                reference = deepcopy(model)
                source = torch.randn(3, 32)
                x = source.clone().requires_grad_()
                codec = RecordingCodec(model, "minmax", 2, 16, 8)
                with codec.capture():
                    actual = model(x + 0.)
                dy = torch.randn_like(actual)
                actual.backward(dy)
                torch.testing.assert_close(actual, reference(source.clone()), rtol=0, atol=0)
                _, _, saved = codec.states[0]
                torch.testing.assert_close(x.grad, (dy @ model.weight.detach()) * codec.unpack(saved), rtol=0, atol=0)
                magnitudes = activation(source.clone())
                packed = toy.encode(magnitudes, "minmax", 2, 16, 8)
                restored = toy.decode(packed, torch.float32)
                if bool(magnitudes.min() >= 0 or magnitudes.max() <= 0):
                    restored = restored.clamp(min=magnitudes.min(), max=magnitudes.max())
                torch.testing.assert_close(model.weight.grad, dy.T @ restored, rtol=0, atol=0)
                self.assertEqual(codec.summary()["derivative_tensors"], 1)
                self.assertEqual(codec.summary()["lossy_value_tensors"], 1)

    def test_inplace_leaf_errors_do_not_modify_the_input(self):
        for factory in (lambda: nn.SiLU(inplace=True), lambda: nn.ELU(inplace=True)):
            for view in (False, True):
                with self.subTest(module=type(factory()).__name__, view=view):
                    model = factory()
                    leaf = torch.randn(3, 32, requires_grad=True)
                    before = leaf.detach().clone()
                    codec = toy.SavedActivations(model, "quest", 2, 32, 8)
                    with codec.capture(), self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                        model(leaf[:, :] if view else leaf)
                    torch.testing.assert_close(leaf, before, rtol=0, atol=0)
                    self.assertFalse(codec.pending_states)

    def test_softmax_and_logsoftmax_use_probabilities_with_coupled_reductions(self):
        for shape, dim in (((3, 7, 65), -1), ((3, 7, 65), 1), ((), 0), ((2, 0, 7), -1)):
            source = torch.randn(shape) * 2 + 3
            for factory in (lambda: nn.Softmax(dim), lambda: nn.LogSoftmax(dim)):
                for method in toy.METHODS:
                    for bits in (1, 2, 4, 8):
                        with self.subTest(shape=shape, dim=dim, kind=type(factory()).__name__, method=method, bits=bits):
                            model = factory()
                            x = source.clone().requires_grad_()
                            codec = RecordingCodec(model, method, bits, 32, 8)
                            with codec.capture():
                                actual = model(x)
                            torch.testing.assert_close(actual, model(source), rtol=0, atol=0)
                            _, _, saved = codec.states[0]
                            matrix = codec.unpack(saved)
                            if isinstance(saved, toy.SavedActivation):
                                self.assertTrue(bool((matrix >= 0).all()))
                                torch.testing.assert_close(matrix.sum(1), torch.ones(matrix.shape[0]), rtol=2e-6, atol=2e-7)
                            dy = torch.randn_like(actual)
                            actual.backward(dy)
                            axis = dim % len(shape) if shape else 0
                            tail = list(shape)
                            width = tail.pop(axis) if tail else 1
                            p = matrix.reshape(*tail, width).movedim(-1, axis).reshape(shape)
                            expected = (dy - p * dy.sum(dim, keepdim=True) if isinstance(model, nn.LogSoftmax) else
                                        p * (dy - (dy * p).sum(dim, keepdim=True)))
                            torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
                            torch.testing.assert_close(x.grad.sum(dim), torch.zeros_like(x.grad.sum(dim)), rtol=0, atol=5e-6)

    def test_normalization_restores_domain_moments_and_keeps_epsilon_and_stats(self):
        factories = ((lambda: nn.LayerNorm((8, 16), eps=.7), (2, 3, 8, 16)),
                     (lambda: nn.GroupNorm(4, 8, eps=.7), (2, 8, 8, 16)),
                     (lambda: nn.BatchNorm2d(8, eps=.7), (2, 8, 8, 16)),
                     (lambda: nn.InstanceNorm2d(8, eps=.7, affine=True), (2, 8, 8, 16)))
        for factory, shape in factories:
            for method in toy.METHODS:
                for bits in (1, 2, 4, 8):
                    with self.subTest(module=type(factory()).__name__, method=method, bits=bits):
                        model = factory()
                        source = torch.randn(shape) * .4 + 80
                        x = source.clone().requires_grad_()
                        codec = RecordingCodec(model, method, bits, 32, 8)
                        with codec.capture():
                            y = model(x)
                        self.assertEqual([kind for kind, _, _ in codec.states],
                                         ["normalized_state", "normalization_statistics"])
                        _, original, saved = codec.states[0]
                        self.assertIsInstance(saved, toy.SavedActivation)
                        restored = codec.unpack(saved)
                        torch.testing.assert_close(restored.mean(1), original.mean(1), rtol=0, atol=1e-6)
                        torch.testing.assert_close(restored.square().mean(1), original.square().mean(1), rtol=2e-5, atol=2e-6)
                        self.assertTrue(bool((original.square().mean(1) < .5).all()))  # real epsilon, not unit variance
                        _, statistics, exact = codec.states[1]
                        torch.testing.assert_close(exact, statistics, rtol=0, atol=0)
                        self.assertEqual(exact.dtype, torch.float32)
                        self.assertEqual(codec.summary()["normalized_state_tensors"], 1)
                        self.assertEqual(codec.summary()["lossy_value_tensors"], 0)
                        self.assertEqual(codec.summary()["exact_reasons"]["normalization_statistics"], 1)
                        expected_bytes = saved.packed.saved_bytes + saved.correction.numel() * 4
                        self.assertEqual(codec.summary()["normalized_state_bytes"], expected_bytes)
                        y.backward(torch.randn_like(y))
                        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_normalization_backward_without_quantization_matches_native(self):
        factories = ((lambda: nn.LayerNorm((3, 7)), (2, 3, 7)),
                     (lambda: nn.GroupNorm(2, 4), (2, 4, 7)),
                     (lambda: nn.BatchNorm1d(4), (2, 4, 7)),
                     (lambda: nn.InstanceNorm1d(4, affine=True, track_running_stats=True), (2, 4, 7)))
        for factory, shape in factories:
            for training in (True, False):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(module=type(factory()).__name__, training=training, dtype=dtype):
                        model = factory().train(training)
                        with torch.no_grad():
                            model.weight.uniform_(-1.5, 2.)
                            model.bias.normal_()
                        reference = deepcopy(model)
                        source = (torch.randn(shape) + 3).to(dtype)
                        x, native = source.clone().requires_grad_(), source.clone().requires_grad_()
                        codec = LosslessCoefficientCodec(model, "minmax", 2, 32, 8)
                        with codec.capture():
                            actual = model(x)
                        expected = reference(native)
                        dy = torch.randn_like(actual)
                        actual.backward(dy)
                        expected.backward(dy)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(x.grad, native.grad, rtol=.02 if dtype == torch.bfloat16 else 2e-4,
                                                   atol=.04 if dtype == torch.bfloat16 else 2e-5)
                        for value, ref in zip(model.parameters(), reference.parameters()):
                            torch.testing.assert_close(value.grad, ref.grad, rtol=.02 if dtype == torch.bfloat16 else 2e-4,
                                                       atol=.06 if dtype == torch.bfloat16 else 3e-5)
                        for value, ref in zip(model.buffers(), reference.buffers()):
                            torch.testing.assert_close(value, ref, rtol=0, atol=0)
                        self.assertEqual(codec.summary()["compressed_tensors"], 0)

    def test_functional_norm_kwargs_without_affine_parameters(self):
        class Functional(nn.Module):
            def __init__(self, function):
                super().__init__()
                self.function = function

            def forward(self, x):
                return self.function(x)

        functions = (lambda x: F.layer_norm(input=x, normalized_shape=(4, 32), eps=.2),
                     lambda x: F.group_norm(input=x, num_groups=2, eps=.2),
                     lambda x: F.batch_norm(input=x, running_mean=None, running_var=None, training=True, eps=.2),
                     lambda x: F.instance_norm(input=x, running_mean=None, running_var=None, use_input_stats=True, eps=.2))
        for function in functions:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(function=function, dtype=dtype):
                    model = Functional(function)
                    source = (torch.randn(2, 4, 32) + 3).to(dtype)
                    x, native = source.clone().requires_grad_(), source.clone().requires_grad_()
                    codec = LosslessCoefficientCodec(model, "quest-centered", 2, 32, 8)
                    with codec.capture():
                        actual = model(x)
                    expected = model(native)
                    dy = torch.randn_like(actual)
                    actual.backward(dy)
                    expected.backward(dy)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, native.grad, rtol=.02 if dtype == torch.bfloat16 else 2e-4,
                                               atol=.04 if dtype == torch.bfloat16 else 2e-5)
                    self.assertEqual(codec.summary()["saved_state_tensors"]["normalized_state"], 1)

    def test_normalization_domain_padding_and_constant_inputs(self):
        for factory, shape in ((lambda: nn.LayerNorm(65), (3, 65)),
                               (lambda: nn.GroupNorm(3, 6), (2, 6, 7))):
            for method in toy.METHODS:
                for bits in (1, 2, 4, 8):
                    for constant in (False, True):
                        with self.subTest(module=type(factory()).__name__, method=method, bits=bits, constant=constant):
                            model = factory()
                            source = torch.full(shape, 3.) if constant else torch.randn(shape) + 3
                            x = source.requires_grad_()
                            codec = RecordingCodec(model, method, bits, 32, 8)
                            with codec.capture():
                                y = model(x)
                            _, original, saved = codec.states[0]
                            restored = codec.unpack(saved)
                            self.assertEqual(restored.shape, original.shape)
                            torch.testing.assert_close(restored.mean(1), original.mean(1), rtol=0, atol=1e-6)
                            torch.testing.assert_close(restored.square().mean(1), original.square().mean(1), rtol=3e-5, atol=3e-6)
                            self.assertGreater(codec.summary()["normalization_correction_bytes"], 0)
                            self.assertEqual(codec.summary()["degenerate_normalization_domains"], 0)
                            y.backward(torch.randn_like(y))
                            self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_short_structured_domains_use_smaller_legal_codec_blocks(self):
        for factory in (lambda: nn.Softmax(-1), lambda: nn.LayerNorm(32)):
            with self.subTest(module=type(factory()).__name__):
                model = factory()
                codec = RecordingCodec(model, "quest-centered", 2, 256, 128)
                with codec.capture():
                    result = model(torch.randn(3, 32, requires_grad=True))
                _, _, saved = codec.states[0]
                self.assertEqual(saved.packed.spec.group_size, 32)
                self.assertEqual(saved.packed.spec.hadamard_size, 32)
                self.assertEqual(saved.packed.spec.shape, (3, 32))
                result.sum().backward()

    def test_norm_bias_only_weight_only_and_fixed_statistics_backward(self):
        for kind in ("layer", "batch", "group", "instance"):
            for needs_x, needs_w, needs_b in ((True, False, False), (False, True, False),
                                            (False, False, True), (True, True, True)):
                with self.subTest(kind=kind, needs=(needs_x, needs_w, needs_b)):
                    factories = {"layer": lambda: nn.LayerNorm((4, 32)), "batch": lambda: nn.BatchNorm1d(4),
                                 "group": lambda: nn.GroupNorm(2, 4),
                                 "instance": lambda: nn.InstanceNorm1d(4, affine=True, track_running_stats=True)}
                    model = factories[kind]().eval()
                    model.weight.requires_grad_(needs_w)
                    model.bias.requires_grad_(needs_b)
                    source = torch.randn(2, 4, 32, requires_grad=needs_x)
                    codec = RecordingCodec(model, "quest", 2, 32, 8)
                    with codec.capture():
                        y = model(source)
                    y.sum().backward()
                    self.assertEqual(source.grad is not None, needs_x)
                    self.assertEqual(model.weight.grad is not None, needs_w)
                    self.assertEqual(model.bias.grad is not None, needs_b)
                    if needs_b and not needs_x and not needs_w:
                        self.assertEqual(codec.states, [])
                    elif needs_x and not needs_w and kind in ("batch", "instance"):
                        self.assertEqual([name for name, _, _ in codec.states], ["normalization_statistics"])

    def test_values_comparison_mode_retains_masks_and_bypasses_operator_states(self):
        model = nn.Sequential(nn.LayerNorm(32), nn.GELU(), nn.ReLU(), nn.Linear(32, 8))
        for state in ("operator", "values"):
            with self.subTest(state=state):
                codec = toy.SavedActivations(model, "quest-centered", 2, 32, 8, state)
                with codec.capture():
                    y = model(torch.randn(3, 32, requires_grad=True))
                y.sum().backward()
                summary = codec.summary()
                self.assertEqual(summary["exact_mask_tensors"], 1)
                self.assertEqual(summary["saved_state"], state)
                self.assertEqual(summary["derivative_tensors"], int(state == "operator"))
                self.assertEqual(summary["normalized_state_tensors"], int(state == "operator"))

    def test_normalized_state_removes_constant_direction_gradient_leakage_at_low_bits(self):
        for method in toy.METHODS:
            for bits in (1, 2):
                with self.subTest(method=method, bits=bits):
                    model = nn.GroupNorm(4, 8)
                    x = (torch.randn(3, 8, 8, 8) + 3).requires_grad_()
                    codec = toy.SavedActivations(model, method, bits, 32, 8)
                    with codec.capture():
                        y = model(x)
                    y.sum().backward()
                    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=3e-6)

    def test_error_reports_signed_mean_error_scaled_by_reference_rms(self):
        reference = torch.tensor([1., 2., 3., 4.])
        result = toy.error(reference + .25, reference)
        self.assertEqual(result["mean_error"], .25)
        self.assertAlmostEqual(result["mean_error_over_reference_rms"], .25 / math.sqrt(7.5))

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    def test_cuda_bf16_derivative_and_normalized_states(self):
        for factory in (nn.GELU, nn.SiLU, lambda: nn.LayerNorm(256), lambda: nn.BatchNorm1d(256)):
            with self.subTest(module=type(factory()).__name__):
                model = factory().cuda()
                x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
                codec = RecordingCodec(model, "quest-centered", 2, 256, 128)
                with codec.capture(), toy.amp(x.device, torch.bfloat16):
                    actual = model(x)
                actual.sum().backward()
                self.assertTrue(bool(torch.isfinite(x.grad).all()))
                self.assertTrue(codec.states)


if __name__ == "__main__":
    unittest.main()
