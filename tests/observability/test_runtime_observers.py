import copy
import unittest
from unittest.mock import patch
import warnings
import weakref

import torch
from torch import nn

from deepopt.engine import TrainingRuntime
from deepopt.engine.strategies import ClipGradNorm, GradientStrategy
from deepopt.observers import FiniteCheck


class RuntimeObserverTests(unittest.TestCase):
    def test_default_loss_check_clears_pending_gradients_and_blocks_retries(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        before = model.weight.detach().clone()
        runtime.backward(model(torch.ones(1, 1)).sum())
        with self.assertRaises(FloatingPointError):
            runtime.backward(model(torch.ones(1, 1)).sum() * float("nan"))
        self.assertIsNone(model.weight.grad)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()

    def test_checks_false_removes_tensor_scans_but_keeps_divisor_validation(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1), checks=False)
        with patch("torch.isfinite", side_effect=AssertionError("disabled scan")), \
                patch("torch.aminmax", side_effect=AssertionError("disabled scan")):
            runtime.backward(model(torch.ones(1, 1)).sum())
            model.weight.grad.fill_(3e38)
            with self.assertRaisesRegex(ValueError, "positive finite real"):
                runtime.step(gradient_divisor=float("nan"))
            runtime.step(gradient_divisor=0.5)
        self.assertTrue(torch.isinf(model.weight).all())
        self.assertIsNone(model.weight.grad)

    def test_observers_see_detached_values_at_each_gradient_stage(self):
        class DoubleGradients(GradientStrategy):
            def get_config(self):
                return {}

            def process(self, parameters):
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(2)

        recorded = []
        references = []

        def record(event):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(not tensor.requires_grad and tensor.grad_fn is None
                                for tensor in event.tensors))
            recorded.append((event.kind, event.name, tuple(tensor.item() for tensor in event.tensors)))
            references.extend(weakref.ref(tensor) for tensor in event.tensors)

        model = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.weight)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1),
                                  checks=[FiniteCheck(), record], gradient_strategy=DoubleGradients())
        runtime.backward(model(torch.tensor([[3.]])).sum())
        runtime.step(gradient_divisor=2)
        gradients = [(name, values) for kind, name, values in recorded if kind == "gradients"]
        self.assertEqual(gradients, [("gradient", (3.0,)), ("normalized gradient", (1.5,)),
                                     ("gradient after processing", (3.0,))])
        self.assertEqual(recorded[0], ("loss", "backward loss", (3.0,)))
        self.assertAlmostEqual(model.weight.item(), 0.7, places=6)
        self.assertTrue(all(reference() is None for reference in references))

    def test_checks_do_not_detach_original_higher_order_gradients(self):
        observed = []

        def record(event):
            observed.append((event.kind, tuple(tensor.requires_grad for tensor in event.tensors)))

        model = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.weight)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1), checks=[record])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            runtime.backward(model.weight.square().sum(), create_graph=True)
        try:
            self.assertTrue(model.weight.grad.requires_grad)
            self.assertIsNotNone(model.weight.grad.grad_fn)
            runtime.observe("gradients", model.weight.grad, name="manual higher-order gradient")
            self.assertTrue(model.weight.grad.requires_grad)
            self.assertEqual(observed, [("loss", (False,)), ("gradients", (False,))])
        finally:
            runtime.zero_grad()

    def test_filtered_weight_check_catches_post_update_failure_and_can_restore(self):
        model = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        runtime = TrainingRuntime(model, optimizer, checks=[FiniteCheck(kinds={"weights"})])
        model_state, optimizer_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        # A weights-only observer must not reject this at the gradient stage.
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "weights after update"):
            runtime.step()
        self.assertTrue(torch.isinf(model.weight).all())
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        runtime.reset_after_restore()
        runtime.backward(model(torch.ones(1, 1)).sum())
        runtime.step()
        torch.testing.assert_close(model.weight, torch.zeros_like(model.weight), rtol=0, atol=0)

    def test_weight_observation_uses_only_optimizer_owned_parameters(self):
        model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model[0].weight.fill_(float("nan"))
            model[1].weight.fill_(1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model[1].parameters(), lr=0.1),
                                  checks=[FiniteCheck(kinds={"weights"})])
        model[1].weight.grad = torch.ones_like(model[1].weight)
        runtime.step()
        self.assertAlmostEqual(model[1].weight.item(), 0.9, places=6)

    def test_manual_reporting_observer_failure_invalidates_runtime(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        runtime.backward(model(torch.ones(1, 1)).sum())
        with self.assertRaises(FloatingPointError):
            runtime.observe("report", torch.tensor(float("nan")), name="custom report")
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.backward(model(torch.ones(1, 1)).sum())

    def test_custom_observer_exception_propagates_and_blocks_updates(self):
        def fail(event):
            raise LookupError("observer failed")

        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1), checks=[fail])
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(LookupError, "observer failed"):
            runtime.observe("activations", model.weight.square(), name="hidden")
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()

    def test_disabled_checks_keep_sparse_coalescing_and_normalization(self):
        model = nn.Embedding(3, 1, sparse=True)
        nn.init.zeros_(model.weight)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=1), checks=False)
        runtime.backward(model(torch.tensor([1, 1])).sum())
        runtime.step(gradient_divisor=2)
        torch.testing.assert_close(model.weight, torch.tensor([[0.], [-1.], [0.]]), rtol=0, atol=0)

    def test_norm_overflow_aborts_update_even_when_clipping_produces_finite_gradients(self):
        model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        runtime = TrainingRuntime(model, optimizer, grad_clip_norm=1.0)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, 1e20)
        with self.assertRaisesRegex(FloatingPointError, "gradient processing result"):
            runtime.step()
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(optimizer.state, {})
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()

    def test_disabled_norm_checks_match_native_clipping_without_diagnostic_scans(self):
        model = nn.Linear(2, 1, bias=False)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        runtime = TrainingRuntime(model, optimizer, grad_clip_norm=1.0, checks=False)
        model.weight.grad = torch.full_like(model.weight, 1e20)
        reference.weight.grad = model.weight.grad.clone()
        norm = nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=False)
        self.assertTrue(torch.isinf(norm))
        expected_optimizer.step()
        with patch("torch.isfinite", side_effect=AssertionError("disabled finite scan")), \
                patch("torch.aminmax", side_effect=AssertionError("disabled finite scan")), \
                patch("torch.Tensor.isnan", side_effect=AssertionError("disabled norm check")), \
                patch("torch.Tensor.isinf", side_effect=AssertionError("disabled norm check")):
            runtime.step()
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)

    def test_finite_clipping_matches_pytorch_and_reports_pre_clip_norm(self):
        model = nn.Linear(2, 1, bias=False).double()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        norms = []

        def record(event):
            if event.name == "gradient processing result":
                self.assertEqual(event.kind, "gradients")
                self.assertFalse(event.tensors[0].requires_grad)
                norms.append(event.tensors[0].item())

        runtime = TrainingRuntime(model, optimizer, grad_clip_norm=1.0, checks=[FiniteCheck(), record])
        model.weight.grad = torch.tensor([[6., 8.]], dtype=torch.double)
        reference.weight.grad = model.weight.grad.clone().div_(2)
        expected_norm = nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
        expected_optimizer.step()
        runtime.step(gradient_divisor=2)
        self.assertEqual(norms, [expected_norm.item()])
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)

    def test_direct_norm_strategy_transforms_without_runtime_diagnostics(self):
        parameter = nn.Parameter(torch.ones(2))
        parameter.grad = torch.full_like(parameter, 1e20)
        norm = ClipGradNorm(1.0).process([parameter])
        self.assertTrue(torch.isinf(norm))
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)

    def test_custom_strategy_can_return_a_diagnostic_without_changing_gradients(self):
        class Diagnostic(GradientStrategy):
            def get_config(self):
                return {}

            def process(self, parameters):
                return torch.tensor(float("nan"))

        model = nn.Linear(1, 1, bias=False)
        before = model.weight.detach().clone()
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1),
                                  gradient_strategy=Diagnostic())
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(FloatingPointError, "gradient processing result"):
            runtime.step()
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)


if __name__ == "__main__":
    unittest.main()
