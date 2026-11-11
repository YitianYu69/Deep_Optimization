"""One-time backward planning, encoded-gradient reuse and numerical edge oracles."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.config import fp8_config_from_dict, ConfigError
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation
from deepopt.opt.quantization.quantizer import quantize_fp8


def settings(**kwargs):
    return deepopt.FP8Config(enabled=True, backend="reference", backward_precision="auto", grad_dtype="e5m2", **kwargs)


class BackwardPrecisionTests(unittest.TestCase):
    def test_configuration_is_explicit_and_native_is_backwards_compatible(self):
        self.assertEqual(fp8_config_from_dict({}).backward_precision, "native")
        self.assertEqual(fp8_config_from_dict({"backward_precision": "auto"}).backward_precision, "auto")
        for invalid in (True, "fp8", None, 7):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                fp8_config_from_dict({"backward_precision": invalid})

    def test_incoming_gradient_layout_conversion_uses_represented_fp32_values(self):
        layer, _ = prepare_fp8_model(nn.Linear(8, 4), settings())
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
        dy = to_fp8_activation(torch.randn(2, 3, 4).bfloat16(), layout="channels", fp8_dtype=torch.float8_e5m2,
                               block_shape=(2, 2))
        qg = quantize_fp8(dy.dequantize(torch.float32).reshape(-1, 4), fp8_dtype=torch.float8_e5m2)
        qw = quantize_fp8(layer.weight)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(x)
        output.backward(dy)
        torch.testing.assert_close(x.grad, (qg.dequantize() @ qw.dequantize()).reshape(x.shape).bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(layer.bias.grad, dy.dequantize(torch.float32).reshape(-1, 4).sum(0), rtol=0, atol=0)

    def test_linear_relu_chain_matches_encoded_gradient_oracle_and_keeps_parameter_grads_native(self):
        torch.manual_seed(19)
        for block in (None, (1, 4)):
            model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
            original = deepcopy(model)
            model, _ = prepare_fp8_model(model, settings(block_shapes={"activation": None, "weight": None, "grad": block}))
            self.assertEqual(len(model), 3, "gradient state must not become a Sequential operation")
            x = torch.randn(5, 8, requires_grad=True)
            dy = torch.randn(5, 4)
            qx = quantize_fp8(x.detach())
            w1, w2 = quantize_fp8(original[0].weight), quantize_fp8(original[2].weight)
            h = qx.dequantize() @ w1.dequantize().T + original[0].bias
            qh = quantize_fp8(h.relu())
            qdy = quantize_fp8(dy, fp8_dtype=torch.float8_e5m2, block_shape=block)
            raw_dx2 = qdy.dequantize() @ w2.dequantize()
            qdx2 = quantize_fp8(raw_dx2, fp8_dtype=torch.float8_e5m2, block_shape=block)
            masked_data = torch.where(h > 0, qdx2.data.float(), 0).to(torch.float8_e5m2)
            from deepopt.opt.quantization.tensor import QuantizedTensor

            qmasked = QuantizedTensor(masked_data, qdx2.scale_inv, block)
            expected_dx = qmasked.dequantize() @ w1.dequantize()
            dw_gradient = (quantize_fp8(qmasked.dequantize().T, fp8_dtype=torch.float8_e5m2, block_shape=block).dequantize()
                           if block else qmasked.dequantize().T)
            expected_dw1 = dw_gradient @ qx.dequantize()
            expected_g2 = (quantize_fp8(dy.T, fp8_dtype=torch.float8_e5m2, block_shape=block).dequantize()
                           if block else qdy.dequantize().T)
            output = model(x)
            # Compatible input gradients must reach preceding backward kernels
            # as wrappers, rather than reconstructed BF16/FP32 tensors.
            received = []
            import deepopt.opt.nn.autograd.linear as linear

            actual_operand = linear.gradient_operand
            with patch.object(linear, "gradient_operand", side_effect=lambda value, layout:
                              received.append(type(value)) or actual_operand(value, layout)):
                output.backward(dy)
            self.assertIn(FP8Activation, received)
            torch.testing.assert_close(x.grad, expected_dx, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(model[0].weight.grad, expected_dw1, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(model[2].weight.grad, expected_g2 @ qh.dequantize(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(model[0].bias.grad, qmasked.dequantize().sum(0), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(model[2].bias.grad, dy.sum(0), rtol=1e-5, atol=1e-6)
            for parameter in model.parameters():
                self.assertNotIsInstance(parameter.grad, FP8Activation)
                self.assertEqual(parameter.grad.dtype, parameter.dtype)

    def test_convolutions_norms_and_both_compression_policies_execute_planned_edges(self):
        for rank in (1, 2, 3):
            for norm in ("batch", "group"):
                if norm == "batch" and rank != 2:
                    continue
                for compressed in (False, True):
                    conv = getattr(nn, f"Conv{rank}d")
                    model = nn.Sequential(conv(4, 8, 1), nn.BatchNorm2d(8) if norm == "batch" else nn.GroupNorm(2, 8),
                                          nn.ReLU(), conv(8, 4, 1))
                    model, _ = prepare_fp8_model(model, settings(), activation_compression=compressed)
                    x = torch.randn(2, 4, *((3,) * rank), requires_grad=True)
                    model(x).sum().backward()
                    report = precision_report(model)
                    self.assertEqual(report["analyses"], 1)
                    self.assertEqual(report["fallbacks"], ())
                    self.assertEqual(sum(e["precision"] == "fp8" and e["mode"] == "train" for e in report["backward_edges"]), 3)
                    self.assertEqual(sum(info is not None for info in report["backward_execution"].values()), 3)
                    self.assertNotIsInstance(x.grad, FP8Activation)
                    self.assertTrue(torch.isfinite(x.grad).all())
                    for parameter in model.parameters():
                        self.assertNotIsInstance(parameter.grad, FP8Activation)

    def test_shared_calls_branches_identity_native_backend_and_attention_boundaries(self):
        class Branch(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.shared, self.b = nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)

            def forward(self, x):
                first = self.a(x)
                return self.b(self.shared(first)) + first + self.shared(x)

        model, _ = prepare_fp8_model(Branch(), settings())
        train = [e for e in precision_report(model)["backward_edges"] if e["mode"] == "train"]
        self.assertTrue(any("fan-out" in e["reason"] for e in train))
        self.assertEqual(sum(e["producer"].startswith("shared") for e in train), 2)
        with patch("deepopt.opt.fx.tracer.DOTracer.trace", side_effect=AssertionError("retrace")):
            model(torch.randn(3, 8, requires_grad=True)).sum().backward()
        native_settings = deepopt.FP8Config(enabled=True, backend="triton", backward_backend="native_fast", backward_precision="auto")
        native, _ = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)), native_settings)
        self.assertFalse(any(e["precision"] == "fp8" for e in precision_report(native)["backward_edges"]))
        identity, _ = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8), nn.Identity(), nn.Linear(8, 8)), settings())
        identity(torch.randn(3, 8, requires_grad=True)).sum().backward()
        mha, _ = prepare_fp8_model(nn.MultiheadAttention(8, 2, batch_first=True), settings())
        # Root attention is traced only if safe; native core boundaries are
        # never silently declared FP8-capable.
        report = precision_report(mha)
        self.assertIsNotNone(report)
        self.assertFalse(any(e["precision"] == "fp8" for e in report["backward_edges"]))

    def test_functional_roots_and_tensor_methods_use_cached_decisions(self):
        class Functional(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1, self.w2 = nn.Parameter(torch.randn(8, 8)), nn.Parameter(torch.randn(4, 8))

            def forward(self, x):
                return F.linear(F.linear(x, self.w1).relu(), self.w2)

        for trace in ("modules", "full"):
            model, _ = prepare_fp8_model(Functional(), settings(trace_mode=trace))
            with patch("deepopt.opt.fx.tracer.DOTracer.trace", side_effect=AssertionError("retrace")):
                model(torch.randn(3, 8, requires_grad=True)).sum().backward()
            report = precision_report(model)
            self.assertEqual(report["fallbacks"], ())
            self.assertEqual(sum(info is not None for info in report["backward_execution"].values()), 2)

    def test_delayed_gradient_state_checkpoint_replay_and_state_dict_resume(self):
        torch.manual_seed(23)
        original = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        cfg = settings(scaling_recipe="delayed")
        model, _ = prepare_fp8_model(deepcopy(original), cfg, activation_compression=True)
        x = torch.randn(3, 8, requires_grad=True)
        deepopt.fp8_checkpoint(model, x, use_reentrant=False).sum().backward()
        for slot in model._deepopt_gradient_edges.values():
            if slot.edge.precision == "fp8" and slot.training:
                self.assertEqual(slot.scaling.updates[2].item(), 1)
        resumed, _ = prepare_fp8_model(deepcopy(original), cfg, activation_compression=True)
        resumed.load_state_dict(deepcopy(model.state_dict()))
        model.zero_grad(set_to_none=True)
        resumed.zero_grad(set_to_none=True)
        xx = torch.randn(3, 8, requires_grad=True)
        yy = xx.detach().clone().requires_grad_()
        a, b = model(xx), resumed(yy)
        a.sum().backward()
        b.sum().backward()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(xx.grad, yy.grad, rtol=0, atol=0)

    def test_dense_override_and_compiled_model_keep_saved_forward_policy(self):
        model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)), settings())
        compiled = deepcopy(model)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        compiled.compile(backend="eager", fullgraph=False)
        for target in (model, compiled):
            x = torch.randn(3, 8, requires_grad=True)
            with deepopt.autocast("cpu", enabled=False, fp8=False):
                output = target(x)
            output.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertFalse(any(info is not None for info in precision_report(target)["backward_execution"].values()))
        x = torch.randn(3, 8, requires_grad=True)
        compiled(x).sum().backward()
        self.assertEqual(sum(info is not None for info in precision_report(compiled)["backward_execution"].values()), 2)

    def test_attention_internal_projections_do_not_consume_outer_functional_call_sites(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
                self.weight = nn.Parameter(torch.randn(8, 8))

            def forward(self, x):
                attended = self.attn(x, x, x, need_weights=False)[0]
                return F.linear(F.relu(attended), self.weight)

        model, _ = prepare_fp8_model(Mixed(), settings())
        model(torch.randn(2, 3, 8, requires_grad=True)).sum().backward()
        report = precision_report(model)
        self.assertEqual(report["fallbacks"], ())
        outputs = [info for info in report["backward_execution"].values() if info is not None]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["precision"], "fp8")

    def test_trainer_prepares_backward_policy_once_across_train_and_validation(self):
        from torch.utils.data import DataLoader, TensorDataset

        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = deepopt.Trainer(model, optimizer, device="cpu", loss_fn=nn.MSELoss(), fp8=settings(),
                                  execution_mode="max_speed")
        loader = DataLoader(TensorDataset(torch.randn(8, 8), torch.randn(8, 4)), batch_size=4)
        with patch("deepopt.opt.fx.tracer.DOTracer.trace", side_effect=AssertionError("retrace")):
            trainer.fit(loader, loader, epochs=2)
        self.assertEqual(precision_report(trainer.model)["analyses"], 1)


if __name__ == "__main__":
    unittest.main()
