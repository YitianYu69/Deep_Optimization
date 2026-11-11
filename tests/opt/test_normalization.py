"""Normalization storage, gradients and once-only precision planning."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.opt.nn import functional as DF
from deepopt.opt.nn.normalization import NormGeometry
from deepopt.opt.policy import normalization_plan, prepare_fp8_model
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1893)

    def test_fp8_activation_autograd_relu_and_native_boundaries(self):
        x = torch.randn(2, 4, 3, 5, requires_grad=True)
        q = to_fp8_activation(x, block_shape=(4, 4))
        self.assertIsInstance(q, FP8Activation)
        self.assertTrue(q.requires_grad)
        expected = q.dequantize().detach()
        actual = torch.relu(q)
        self.assertIsInstance(actual, FP8Activation)
        torch.testing.assert_close(dequantize_activation(actual), expected.relu(), rtol=0, atol=0)
        (actual + 2).sum().backward()
        torch.testing.assert_close(x.grad, (expected > 0).float(), rtol=0, atol=0)

    def test_compression_saves_normalized_values_and_matches_decode_backward(self):
        for kind in ("batch", "group"):
            for bits in (1, 2, 4, 8):
                with self.subTest(kind=kind, bits=bits):
                    x = (torch.randn(2, 4, 3, 5) * 3 + 100).requires_grad_()
                    weight, bias = torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)
                    geometry = NormGeometry(kind, tuple(x.shape), 2)
                    plan = normalization_plan(False, activation_compression={"enabled": True, "group_size": 17, "bits": bits})
                    saved = []
                    with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
                        actual = (DF.batch_norm(x, None, None, weight, bias, True, plan=plan) if kind == "batch" else
                                  DF.group_norm(x, 2, weight, bias, plan=plan))
                    native = (F.batch_norm(x, None, None, weight, bias, True) if kind == "batch" else
                              F.group_norm(x, 2, weight, bias))
                    torch.testing.assert_close(actual, native, rtol=2e-4, atol=2e-4)
                    matrix = geometry.matrix(x.detach())
                    variance, mean = torch.var_mean(matrix, 1, correction=0)
                    inverse = torch.rsqrt(variance + 1e-5)
                    h = (matrix - mean[:, None]) * inverse[:, None]
                    packed = quantize_pack_activation(h, 17, (1, 17), bits=bits)
                    words = [value for value in saved if value.dtype == torch.int32]
                    self.assertEqual(len(words), 1)
                    torch.testing.assert_close(words[0], packed.data, rtol=0, atol=0)
                    self.assertFalse(any(value is x or tuple(value.shape) in (tuple(x.shape), geometry.matrix_shape)
                                         for value in saved))
                    h = dequantize_unpack_activation(packed, torch.float32)
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    u = geometry.matrix(gradient * weight.detach().reshape(1, 4, 1, 1))
                    dx = geometry.restore((u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inverse[:, None])
                    dw = (gradient * geometry.restore(h)).sum((0, 2, 3))
                    torch.testing.assert_close(x.grad, dx, rtol=0, atol=0)
                    torch.testing.assert_close(weight.grad, dw, rtol=0, atol=0)
                    torch.testing.assert_close(bias.grad, gradient.sum((0, 2, 3)), rtol=0, atol=0)

    def test_bf16_and_fp8_inputs_with_and_without_compression(self):
        settings = deepopt.FP8Config(enabled=True, backend="reference")
        for storage in ("bf16", "fp8", "raw_fp8"):
            for compressed in (False, True):
                with self.subTest(storage=storage, compressed=compressed):
                    source = torch.randn(2, 4, 3, 5, dtype=torch.bfloat16, requires_grad=True)
                    x = (to_fp8_activation(source, block_shape=(4, 4)) if storage == "fp8" else
                         source.detach().to(torch.float8_e4m3fn).requires_grad_() if storage == "raw_fp8" else source)
                    weight = torch.randn(4, requires_grad=True)
                    plan = normalization_plan(settings, activation_compression=compressed)
                    actual = DF.group_norm(x, 2, weight, plan=plan)
                    dense = x.dequantize(torch.float32) if storage == "fp8" else x.float()
                    expected = F.group_norm(dense, 2, weight).to(torch.bfloat16)
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
                    actual.float().square().sum().backward()
                    self.assertTrue(torch.isfinite(weight.grad).all())

    def test_batch_norm_running_state_and_momentum_none_match_native(self):
        for momentum in (0.1, None):
            with self.subTest(momentum=momentum):
                original = nn.BatchNorm2d(4, momentum=momentum)
                expected = deepcopy(original)
                model, _ = prepare_fp8_model(original, False, activation_compression=True)
                for _ in range(3):
                    x = torch.randn(2, 4, 3, 5)
                    torch.testing.assert_close(model(x), expected(x), rtol=2e-5, atol=2e-5)
                self.assertIs(model.running_mean, original.running_mean)
                self.assertIs(model.weight, original.weight)
                self.assertEqual(model.num_batches_tracked.item(), 3)
                torch.testing.assert_close(model.running_mean, expected.running_mean)
                torch.testing.assert_close(model.running_var, expected.running_var)
                model.eval()
                expected.eval()
                x = torch.randn(2, 4, 3, 5, requires_grad=True)
                y = model(x)
                torch.testing.assert_close(y, expected(x), rtol=2e-5, atol=2e-5)
                y.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_precision_plan_hands_fp8_to_conv_and_preserves_native_output(self):
        from deepopt.opt.nn.autograd.convolution import fp8 as convolution

        original = nn.Sequential(nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 8, 3, padding=1))
        model, _ = prepare_fp8_model(original, deepopt.FP8Config(enabled=True, backend="reference", intercept_linear=False),
                                      activation_compression=True)
        self.assertIs(model, original)
        self.assertEqual(model._deepopt_precision_plan["analyses"], 1)
        with patch.object(convolution, "quantize_fp8", wraps=convolution.quantize_fp8) as quantize:
            x = torch.randn(2, 4, 3, 5, requires_grad=True)
            y = model(x)
            self.assertNotIsInstance(y, FP8Activation)
            self.assertEqual(model[0].execution_info["output"], "fp8")
            # Only weights are cast inside convolution forward: its input is
            # the norm's existing FP8 payload, passed through ReLU unchanged.
            self.assertEqual(len(quantize.call_args_list), 1)
            y.sum().backward()
        self.assertIs(prepare_fp8_model(model, None)[0], model)
        model.eval()
        model(torch.randn(2, 4, 3, 5))
        self.assertEqual(model._deepopt_precision_plan["analyses"], 1)
        copied = deepcopy(model)
        with torch.no_grad():
            copied[2].bias.add_(1)
        value = torch.randn(2, 4, 3, 5)
        self.assertFalse(torch.equal(copied(value), model(value)))

    def test_shared_norm_call_sites_have_independent_cached_output_choices(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.GroupNorm(2, 4)
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                return self.fc(self.norm(x)) + self.norm(x).sum(1, keepdim=True)

        model, _ = prepare_fp8_model(Model(), deepopt.FP8Config(enabled=True, backend="reference", intercept_linear=False))
        self.assertEqual(model.norm._precision_outputs[True], ("last", None))
        x = torch.randn(2, 4, requires_grad=True)
        model(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(model._deepopt_precision_plan["analyses"], 1)

    def test_functional_roots_full_and_module_modes_use_cached_call_sites(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4))
                self.bias = nn.Parameter(torch.zeros(4))
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                return self.fc(F.group_norm(x, 2, self.weight, self.bias))

        for mode in ("full", "modules"):
            with self.subTest(mode=mode):
                settings = deepopt.FP8Config(enabled=True, backend="reference", trace_mode=mode,
                                            intercept_linear=False, intercept_normalization=True)
                model, _ = prepare_fp8_model(Model(), settings, activation_compression=True)
                x = torch.randn(3, 4, requires_grad=True)
                model(x).sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(any(row["precision"] == "fp8" for row in model._deepopt_precision_plan["edges"]))

    def test_train_eval_policies_and_explicit_graph_native_boundary(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.GroupNorm(2, 4)
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                x = self.norm(x)
                return self.fc(x) if self.training else x.sin()

        model, _ = prepare_fp8_model(Model(), deepopt.FP8Config(enabled=True, backend="reference", intercept_linear=False))
        self.assertEqual(model.norm._precision_outputs[True], ("last",))
        self.assertEqual(model.norm._precision_outputs[False], (None,))
        model(torch.randn(3, 4)).sum().backward()
        model.eval()
        self.assertNotIsInstance(model(torch.randn(3, 4)), FP8Activation)
        class Fanout(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm, self.fc = nn.GroupNorm(2, 4), nn.Linear(4, 3)

            def forward(self, x):
                y = self.norm(x)
                return self.fc(y), y.sin()

        graph, _ = prepare_fp8_model(Fanout(), deepopt.FP8Config(enabled=True, backend="reference", trace_mode="full"))
        self.assertTrue(any(node.op == "call_function" and node.target is dequantize_activation
                            for node in graph.graph.nodes))
        values = graph(torch.randn(3, 4, requires_grad=True))
        sum(value.sum() for value in values).backward()

    def test_checkpoint_replay_restores_cached_shared_callsite_positions(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm, self.fc = nn.GroupNorm(2, 4), nn.Linear(4, 3)

            def forward(self, x):
                a = self.fc(self.norm(x))
                b = deepopt.fp8_checkpoint(lambda value: self.norm(value), x)
                return a.sum() + b.sum()

        model, _ = prepare_fp8_model(Model(), deepopt.FP8Config(enabled=True, backend="reference", intercept_linear=False))
        model(torch.randn(3, 4, requires_grad=True)).backward()
        self.assertEqual(model._deepopt_precision_plan["analyses"], 1)

    def test_fake_metadata_and_channels_last(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        plan = normalization_plan(False, activation_compression=True)
        x = torch.randn(2, 4, 3, 5).to(memory_format=torch.channels_last).requires_grad_()
        result = DF.group_norm(x, 2, plan=plan)
        self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))
        with FakeTensorMode():
            x = torch.empty(2, 4, 3, 5, requires_grad=True)
            output = DF.group_norm(x, 2, plan=plan)
            self.assertEqual(output.shape, x.shape)
            output.sum().backward()

    def test_trainer_analyzes_once_across_training_and_validation(self):
        from torch.utils.data import DataLoader, TensorDataset
        from deepopt.opt.fx.tracer import DOTracer

        model = nn.Sequential(nn.GroupNorm(2, 4), nn.Linear(4, 3))
        loader = DataLoader(TensorDataset(torch.randn(6, 4), torch.randint(0, 3, (6,))), batch_size=3)
        settings = deepopt.FP8Config(enabled=True, backend="reference", intercept_linear=False)
        with patch.object(DOTracer, "trace", autospec=True, side_effect=DOTracer.trace) as trace:
            trainer = deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), device="cpu",
                                       loss_fn=nn.CrossEntropyLoss(), fp8=settings,
                                       config={"runtime": {"activation_compression": {"enabled": True}}})
            prepared = trace.call_count
            trainer.fit(loader, loader, epochs=2, progress_bar=False)
            self.assertEqual(trace.call_count, prepared)
            self.assertEqual(trainer.model._deepopt_precision_plan["analyses"], 1)

    def test_delayed_output_scale_snapshots_and_checkpoint_replay(self):
        settings = deepopt.FP8Config(enabled=True, backend="reference", intercept_linear=False, scaling_recipe="delayed")
        model, _ = prepare_fp8_model(nn.Sequential(nn.GroupNorm(2, 4), nn.Linear(4, 3)), settings,
                                      activation_compression=True)
        self.assertIsNotNone(model[0].fp8_scaling)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            x = torch.randn(3, 4, requires_grad=True)
            with deepopt.fp8_scaling_scope():
                deepopt.fp8_checkpoint(model, x).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(model[0].fp8_scaling.updates[0].item(), 2)

    def test_memory_probe_understands_physical_fp8_payload(self):
        from deepopt.memory import estimate_training_memory

        model = nn.Sequential(nn.GroupNorm(2, 4), nn.Linear(4, 3))
        trainer = deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), device="cpu",
                                   loss_fn=nn.MSELoss(), fp8=deepopt.FP8Config(enabled=True, backend="reference"),
                                   config={"runtime": {"activation_compression": {"enabled": True}}})
        report = estimate_training_memory(trainer, (torch.randn(3, 4), torch.randn(3, 3)))
        self.assertGreater(report.peak_bytes_by_device["cpu"], 0)
        self.assertEqual(trainer.model._deepopt_precision_plan["analyses"], 1)

    def test_fake_cuda_fp8_storage_never_selects_a_real_kernel(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from deepopt.opt.backends.normalization import executor

        with FakeTensorMode(), torch.no_grad():
            data = torch.empty(30, 4, device="cuda", dtype=torch.float8_e4m3fn)
            scale = torch.empty(8, 1, device="cuda", dtype=torch.float32)
            x = FP8Activation(data, scale, (2, 4, 3, 5), (4, 4))
            self.assertEqual(executor(x, "triton").__name__, "deepopt.opt.backends.reference.normalization")
            output = torch.relu(x)
            self.assertIsInstance(output._data, FakeTensor)

    def test_other_batch_norm_ranks_remain_native_in_fp8_scope(self):
        for kind, shape in ((nn.BatchNorm1d, (2, 4, 5)), (nn.BatchNorm3d, (2, 4, 3, 5, 7))):
            with self.subTest(kind=kind):
                original = kind(4)
                baseline = deepcopy(original)
                x = torch.randn(shape)
                with deepopt.autocast("cpu", enabled=False, fp8=True):
                    actual = original(x)
                torch.testing.assert_close(actual, baseline(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
