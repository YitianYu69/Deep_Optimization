"""Vision operator-pattern contracts; these fixtures are not full models."""

from copy import deepcopy
import unittest

import torch
from torch import nn

from deepopt import autocast
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model


class ResidualPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = (nn.Conv2d(8, 8, 3, padding=1, bias=False) for _ in range(2))
        self.bn1, self.bn2 = nn.BatchNorm2d(8), nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return self.relu(out)


class StochasticDepthPattern(nn.Module):
    def forward(self, x):
        noise = torch.empty([x.shape[0]] + [1] * (x.ndim - 1), dtype=x.dtype, device=x.device)
        noise = noise.bernoulli_(.8)
        noise.div_(.8)
        return x * noise


class AttentionPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)

    def forward(self, x):
        return self.attention(x, x, x, need_weights=False)[0]


class TokenPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(1, 1, 768) * .02)
        self.position = nn.Parameter(torch.randn(1, 197, 768) * .02)
        self.norm = nn.LayerNorm(768, eps=1e-6)

    def forward(self, x):
        tokens = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        added = tokens + self.position
        return self.norm(added), tokens, added


class VisionPatternTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(913)

    def pair(self, module, shape, policy, compressed=True, dtype=torch.bfloat16, attention_bits=None):
        compression = dict(enabled=compressed, bits=2, group_size=1024, norm_xhat_recenter=True,
                           attention_bits=attention_bits)
        config = dict(runtime=dict(bf16_policy=policy, fp8=dict(enabled=False), activation_compression=compression))
        eager, _ = prepare_fp8_model(module, False, config=config)
        native = deepcopy(eager)
        compile_model(native, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options=dict(require_full_aot=True, use_native_autodiff=True, collect_saved_state_metadata=True)), config=config)
        x = torch.randn(shape, dtype=dtype, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        rng = torch.get_rng_state()
        def execute(model, value):
            torch.set_rng_state(rng)
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy=policy,
                          activation_compression=compression, config=config):
                output = model(value)
            return output, torch.get_rng_state()
        expected, expected_rng = execute(eager, reference)
        actual, actual_rng = execute(native, x)
        self.assertTrue(torch.equal(actual_rng, expected_rng))
        actuals = actual if isinstance(actual, tuple) else (actual,)
        controls = expected if isinstance(expected, tuple) else (expected,)
        for a, b in zip(actuals, controls):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        gradients = tuple(torch.linspace(-1, 1, a.numel(), dtype=a.dtype).reshape(a.shape) for a in actuals)
        torch.autograd.backward(actuals, gradients)
        torch.autograd.backward(controls, gradients)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        for (name, parameter), (other, control) in zip(native.named_parameters(), eager.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(parameter.grad, control.grad, rtol=0, atol=0, msg=name)
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.dtype, parameter.dtype)
        for (name, buffer), (other, control) in zip(native.named_buffers(), eager.named_buffers()):
            self.assertEqual(name, other)
            torch.testing.assert_close(buffer, control, rtol=0, atol=0)
        report = compilation_report(native).as_dict()
        from examples.vae.qualification import saved_state_evidence

        self.assertTrue(saved_state_evidence(report, compressed)["passed"])
        self.assertEqual(report["runtime_fallbacks"], 0)
        self.assertGreater(report["native_autodiff_operations"], 0)
        for graph in report["graphs"]:
            self.assertFalse(graph["eager_autograd_operations"])
            self.assertFalse(graph["legacy_backward_operations"])
        return eager, native, actual, report

    def test_prepared_shared_inplace_relu_and_mixed_residual(self):
        for policy in ("torch_amp", "bf16_storage"):
            for dtype in (torch.bfloat16, torch.float32):
                for compressed in (False, True):
                    with self.subTest(policy=policy, dtype=dtype, compressed=compressed):
                        torch.compiler.reset()
                        self.pair(ResidualPattern(), (2, 8, 8, 8), policy, compressed, dtype)

    def test_prepared_inplace_silu_dense_and_compressed(self):
        for policy in ("torch_amp", "bf16_storage"):
            for compressed in (False, True):
                with self.subTest(policy=policy, compressed=compressed):
                    torch.compiler.reset()
                    self.pair(nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.SiLU(inplace=True)),
                              (2, 8, 8, 8), policy, compressed)

    def test_dense_max_pool_overlapping_windows_keep_native_bf16_accumulation(self):
        for policy in ("torch_amp", "bf16_storage"):
            with self.subTest(policy=policy):
                torch.compiler.reset()
                self.pair(nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1)), (2, 3, 17, 19), policy, False)

    def test_original_stochastic_depth_expression_has_exact_native_derivative(self):
        for policy in ("torch_amp", "bf16_storage"):
            with self.subTest(policy=policy):
                torch.compiler.reset()
                self.pair(StochasticDepthPattern(), (8, 8, 8, 8), policy)

    def test_default_mha_is_adapted_with_preserved_parameters(self):
        from deepopt.opt.nn.attention import DOMultiheadAttention

        model = AttentionPattern()
        parameters = [(name, id(value)) for name, value in model.named_parameters()]
        projection = model.attention.out_proj
        eager, _, _, _ = self.pair(model, (1, 197, 768), "bf16_storage", attention_bits=2)
        self.assertIsInstance(eager.attention, DOMultiheadAttention)
        self.assertIs(eager.attention.out_proj, projection)
        self.assertEqual([(name, id(value)) for name, value in eager.named_parameters()], parameters)
        repr(eager)

    def test_storage_class_position_edges_and_master_gradients(self):
        eager, native, outputs, report = self.pair(TokenPattern(), (2, 196, 768), "bf16_storage")
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in outputs))
        for model in (eager, native):
            self.assertEqual(model.class_token.grad.dtype, torch.float32)
            self.assertEqual(model.position.grad.dtype, torch.float32)
            self.assertGreater(model.class_token.grad.abs().max().item(), 0)
            self.assertGreater(model.position.grad.abs().max().item(), 0)
        edges = [edge for graph in report["graphs"] for edge in graph["precision_contracts"]
                 if edge["operation"] in ("cat", "add", "storage_cat", "storage_add")]
        self.assertTrue(edges)
        self.assertTrue(all(value["dtype"] == "torch.bfloat16" for edge in edges for value in edge["outputs"]))


if __name__ == "__main__":
    unittest.main()
