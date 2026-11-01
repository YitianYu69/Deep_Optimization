import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt.engine import OptimizationWindow, TrainingRuntime
from deepopt.engine.strategies import BackwardLoss, ClipGradNorm, ClipGradValue, GradientStrategy, LossStrategy
from deepopt.tasks.base import BatchOutput


class FailOnce(GradientStrategy):
    def __init__(self):
        self.calls = 0

    def get_config(self):
        return {}

    def process(self, parameters):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated gradient processing failure")


class ExplicitWeights(LossStrategy):
    def get_config(self):
        return {}

    def prepare(self, output, *, accumulation_steps):
        return BackwardLoss(output.loss, output.metric_data)


class PartialFailureSGD(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("simulated failure after parameter update")


class RuntimeSafetyTests(unittest.TestCase):
    def test_sparse_gradients_match_pytorch_with_duplicate_indices_and_fractional_divisor(self):
        torch.manual_seed(3)
        model = nn.Embedding(5, 2, sparse=True)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        runtime = TrainingRuntime(model, optimizer)
        indices = torch.tensor([1, 1, 3])
        runtime.backward(model(indices).square().sum())
        reference(indices).square().sum().backward()
        reference.weight.grad = reference.weight.grad.coalesce()
        reference.weight.grad.values().div_(2.5)
        expected_optimizer.step()
        runtime.step(gradient_divisor=2.5)
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)

    def test_sparse_duplicate_overflow_is_detected_before_update(self):
        model = nn.Embedding(3, 2, sparse=True)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        before = model.weight.detach().clone()
        model.weight.grad = torch.sparse_coo_tensor(
            torch.tensor([[0, 0]]), torch.full((2, 2), 2e38), model.weight.shape,
            check_invariants=True)
        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient"):
            runtime.step()
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)

    def test_builtin_clipping_rejects_sparse_gradients_with_readable_error(self):
        for strategy in (ClipGradNorm(1.0), ClipGradValue(1.0)):
            with self.subTest(strategy=type(strategy).__name__):
                model = nn.Embedding(3, 2, sparse=True)
                runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1),
                                          gradient_strategy=strategy)
                before = model.weight.detach().clone()
                runtime.backward(model(torch.tensor([1])).sum())
                with self.assertRaisesRegex(ValueError, "sparse-aware"):
                    runtime.step()
                self.assertIsNone(model.weight.grad)
                torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def test_subset_optimizer_leaves_other_parameters_and_gradients_untouched(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model[1].parameters(), lr=0.1)
        runtime = TrainingRuntime(model, optimizer)
        runtime.backward(model(torch.ones(3, 2)).sum())
        before = [(p.detach().clone(), p.grad.clone()) for p in model[0].parameters()]
        head_before = model[1].weight.detach().clone()
        runtime.step(gradient_divisor=3.0)
        for parameter, (value, gradient) in zip(model[0].parameters(), before):
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        self.assertFalse(torch.equal(model[1].weight, head_before))
        with self.assertRaisesRegex(ValueError, "Every trainable"):
            runtime.require_complete_optimizer()
        model[0].requires_grad_(False)
        runtime.require_complete_optimizer()

    def test_fractional_normalization_weights_are_accumulated(self):
        model = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(model.weight)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=1.0))
        window = OptimizationWindow(runtime, accumulation_steps=2, loss_strategy=ExplicitWeights())
        self.assertFalse(window.backward(BatchOutput(model(torch.tensor([[0.25]])).sum(), 1,
                                                    metric_data=0.25)))
        self.assertTrue(window.backward(BatchOutput(model(torch.tensor([[2.25]])).sum(), 1,
                                                   metric_data=0.75)))
        self.assertEqual(model.weight.item(), -2.5)
        self.assertFalse(window.flush())

    def test_failed_window_cannot_repeat_gradient_normalization(self):
        model = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(model.weight)
        strategy = FailOnce()
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=1.0),
                                  gradient_strategy=strategy)
        window = OptimizationWindow(runtime, accumulation_steps=2)
        window.backward(BatchOutput(model(torch.ones(1, 1)).mean(), 1, "mean"))
        with self.assertRaisesRegex(RuntimeError, "simulated gradient"):
            window.backward(BatchOutput(model(torch.ones(1, 1)).mean(), 1, "mean"))
        self.assertEqual(model.weight.item(), 0.0)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(window.pending_batches, 0)
        with self.assertRaisesRegex(RuntimeError, "window failed"):
            window.flush()
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()
        runtime.zero_grad()  # Clearing gradients does not clear the failed-update guard.
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.backward(model(torch.ones(1, 1)).mean())
        self.assertEqual(strategy.calls, 1)

    def test_partial_optimizer_failure_requires_state_restore(self):
        model = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(model.weight)
        optimizer = PartialFailureSGD(model.parameters(), lr=1.0, momentum=0.9)
        runtime = TrainingRuntime(model, optimizer)
        model_state, optimizer_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        runtime.backward(model(torch.ones(1, 1)).sum())
        with self.assertRaisesRegex(RuntimeError, "after parameter update"):
            runtime.step()
        self.assertEqual(model.weight.item(), -1.0)
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        runtime.reset_after_restore()
        runtime.backward(model(torch.ones(1, 1)).sum())
        self.assertEqual(model.weight.grad.item(), 1.0)
        self.assertEqual(model.weight.item(), 0.0)
        self.assertEqual(len(optimizer.state), 0)

    def test_invalid_divisor_is_rejected_without_mutating_or_invalidating_gradients(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        runtime.backward(model(torch.ones(1, 1)).sum())
        before = model.weight.detach().clone()
        for divisor in (0, -1, True, float("nan"), float("inf"), "2", 1j, 10**1000):
            with self.subTest(divisor=divisor):
                with self.assertRaisesRegex(ValueError, "positive finite real"):
                    runtime.step(gradient_divisor=divisor)
                self.assertEqual(model.weight.grad.item(), 1.0)
        with self.assertRaisesRegex(ValueError, "Choose samples or gradient_divisor"):
            runtime.step(samples=2, gradient_divisor=2)
        runtime.step(gradient_divisor=0.5)
        torch.testing.assert_close(model.weight, before - 0.2)

    def test_invalid_normalization_weight_clears_pending_window(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=1.0))
        window = OptimizationWindow(runtime, accumulation_steps=2, loss_strategy=ExplicitWeights())
        window.backward(BatchOutput(model(torch.ones(1, 1)).sum(), 1, metric_data=0.5))
        with self.assertRaisesRegex(ValueError, "normalization_weight"):
            window.backward(BatchOutput(model(torch.ones(1, 1)).sum(), 1, metric_data=True))
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "window failed"):
            window.flush()

    def test_failed_backward_clears_gradients_and_blocks_update(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        runtime.backward(model(torch.ones(1, 1)).sum())
        with self.assertRaises(RuntimeError):
            runtime.backward(torch.tensor(1.0))  # No autograd graph.
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()

    def test_window_rejects_overflowed_total_weight_before_update(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        window = OptimizationWindow(runtime, accumulation_steps=2, loss_strategy=ExplicitWeights())
        window.backward(BatchOutput(model(torch.ones(1, 1)).sum(), 1, metric_data=1e308))
        with self.assertRaisesRegex(ValueError, "Accumulated normalization weight"):
            window.backward(BatchOutput(model(torch.ones(1, 1)).sum(), 1, metric_data=1e308))
        self.assertIsNone(model.weight.grad)

    def test_unsupported_precision_is_rejected_before_cuda_initialization(self):
        with patch("torch.cuda.current_device", side_effect=AssertionError("must not initialize CUDA")) as current:
            with self.assertRaisesRegex(ValueError, "fp16 is unsupported"):
                TrainingRuntime(nn.Linear(1, 1), device="cuda", precision="fp16")
            current.assert_not_called()


if __name__ == "__main__":
    unittest.main()
