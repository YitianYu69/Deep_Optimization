"""Execution-mode checks propagate through FP8 and direct/runtime entry points."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import DiagnosticPolicy, Trainer, TrainingRuntime
from deepopt.engine.strategies import ExplicitObjective, GradientStrategy
from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.engine.execution.policy import ModeController
from deepopt.observers import CheckObservers
from deepopt.opt._execution import finite_checks_enabled
from deepopt.opt.scaling_strategy.amax import AmaxScaling
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.tasks.base import BatchOutput


class ExecutionModeCoverageTests(unittest.TestCase):
    def test_fp8_forward_and_backward_use_independent_update_cadences(self):
        model = nn.Linear(7, 4)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01), fp8=True,
                                  diagnostics=DiagnosticPolicy(loss_finite_every=2, gradient_finite_every=3))
        observed = []
        phase = None
        original = AmaxScaling.compute

        def compute(self, *args, **kwargs):
            observed.append((runtime.mode_controller.completed_updates + 1, phase, kwargs["check_finite"]))
            return original(self, *args, **kwargs)

        with patch.object(AmaxScaling, "compute", new=compute):
            for _ in range(6):
                phase = "forward"
                loss = runtime.compute(torch.randn(2, 7, requires_grad=True),
                                       lambda value: runtime.model(value).square().mean())
                phase = "backward"
                runtime.backward(loss)
                runtime.step()
        self.assertTrue(observed)
        for update, operation, enabled in observed:
            self.assertEqual(enabled, update % (2 if operation == "forward" else 3) == 0,
                             (update, operation, enabled))

    def test_direct_quantization_and_scaling_in_runtime_follow_max_speed(self):
        runtime = TrainingRuntime(nn.Identity(), execution_mode="max_speed")
        with torch.no_grad(), patch("torch.isfinite", side_effect=AssertionError("passive finite scan")):
            runtime.compute(torch.ones(2, 7), lambda value: quantize_fp8(value))
            runtime.compute(torch.ones(2, 7), lambda value: AmaxScaling().compute(value))
        with self.assertRaisesRegex(ValueError, "finite"):
            AmaxScaling().compute(torch.full((2, 7), float("nan")))

    def test_saved_manual_autocast_context_applies_schedule_before_forward(self):
        model = nn.Linear(7, 4)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01), fp8=True,
                                  execution_mode="max_speed", mode_schedule={2: "analyze", 3: "max_speed"})
        context = runtime.autocast()
        for expected in (False, True, False):
            with context:
                self.assertEqual(finite_checks_enabled(), expected)
                loss = runtime.model(torch.ones(2, 7)).sum()
            runtime.backward(loss)
            runtime.step()

    def test_default_loss_validators_inherit_mode_but_explicit_checks_and_shapes_remain(self):
        runtime = TrainingRuntime(nn.Identity(), execution_mode="max_speed")
        invalid = BatchOutput(torch.tensor(float("nan")))
        with runtime.autocast():
            invalid.validate()
            invalid.loss_sum()
            invalid.objective.validate()
            invalid.loss_report.validate()
            with self.assertRaises(FloatingPointError):
                invalid.validate(check_finite=True)
            with self.assertRaises(ValueError):
                BatchOutput(torch.ones(2)).validate()
        with self.assertRaises(FloatingPointError):
            invalid.validate()

    def test_max_speed_clipping_skips_diagnostic_gradient_traversals(self):
        model = nn.Linear(7, 4)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01),
                                  execution_mode="max_speed", grad_clip_norm=.5)
        loss = runtime.compute(torch.ones(2, 7), lambda value: runtime.model(value).sum())
        runtime.backward(loss)
        with patch.object(runtime._stepper, "_values", side_effect=AssertionError("diagnostic traversal")):
            runtime.step()

    def test_custom_loss_gradient_and_metric_paths_inherit_max_speed(self):
        seen = []

        class Objective(ExplicitObjective):
            def prepare(self, output, **kwargs):
                output.validate()
                seen.append(("loss", finite_checks_enabled()))
                return super().prepare(output, **kwargs)

        class Gradients(GradientStrategy):
            def get_config(self):
                return {}

            def process(self, parameters):
                seen.append(("gradient", finite_checks_enabled()))
                for parameter in parameters:
                    if parameter.grad is not None:
                        AmaxScaling().compute(parameter.grad.reshape(1, -1))

        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                          loss_fn=nn.functional.mse_loss, loss_strategy=Objective(),
                          gradient_strategy=Gradients(), execution_mode="max_speed")
        original = trainer._steps.update_metrics

        def metrics(*args, **kwargs):
            seen.append(("metrics", finite_checks_enabled()))
            AmaxScaling().compute(torch.ones(1, 2))
            return original(*args, **kwargs)

        # An empty task-metric set does not invoke an update callback, so cover
        # the same reporting context at its public phase-computation boundary.
        original_compute = trainer._epochs.metrics.compute

        def report(*args, **kwargs):
            metrics({}, BatchOutput(torch.ones(())))
            return original_compute(*args, **kwargs)

        with patch.object(trainer._epochs.metrics, "compute", new=report), \
             patch("torch.isfinite", side_effect=AssertionError("passive finite scan")):
            trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        self.assertEqual({kind for kind, _ in seen}, {"loss", "gradient", "metrics"})
        self.assertTrue(all(not enabled for _, enabled in seen))

    def test_custom_metric_factory_default_checks_are_gated_by_runtime_mode(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                          loss_fn=nn.functional.mse_loss, metric_factory=MetricAccumulator,
                          execution_mode="max_speed")
        with patch("torch.isfinite", side_effect=AssertionError("metric factory bypassed mode")):
            result = trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        self.assertEqual(result["samples"], 2)

    def test_local_opt_out_preserves_collective_observer_slots(self):
        checks = CheckObservers(False)
        with ModeController().scope("legacy", legacy_checks=False):
            self.assertFalse(checks.allows("weights"))
            self.assertTrue(checks.allows("weights", collective=True))
        with ModeController("max_speed").scope("fast"):
            self.assertFalse(checks.allows("weights"))
            self.assertFalse(checks.allows("weights", collective=True))

    def test_manual_step_prepares_mode_and_gradient_cadence_without_runtime_forward(self):
        seen = []

        class Gradients(GradientStrategy):
            def get_config(self):
                return {}

            def process(self, parameters):
                seen.append(finite_checks_enabled())

        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01),
                                  gradient_strategy=Gradients(), execution_mode="max_speed",
                                  mode_schedule={2: "analyze", 3: "max_speed"})
        for _ in range(3):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            runtime.step()
        self.assertEqual(seen, [False, True, False])

    def test_raw_custom_autograd_backward_inherits_gradient_cadence(self):
        seen = []

        class Operation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, grad):
                seen.append(finite_checks_enabled())
                AmaxScaling().compute(grad.reshape(1, -1))
                return grad

        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01),
                                  diagnostics=DiagnosticPolicy(loss_finite_every=None, gradient_finite_every=2))
        for _ in range(3):
            loss = runtime.compute(torch.ones(2, 2), lambda value: Operation.apply(runtime.model(value)).sum())
            runtime.backward(loss)
            runtime.step()
        self.assertEqual(seen, [False, True, False])

    def test_runtime_backward_controls_fp8_checks_after_an_unscoped_forward(self):
        model = nn.Linear(7, 4)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01),
                                  fp8=True, execution_mode="max_speed")
        loss = runtime.model(torch.ones(2, 7, requires_grad=True)).sum()
        with patch("torch.isfinite", side_effect=AssertionError("backward bypassed runtime mode")):
            runtime.backward(loss)
            runtime.step()

    def test_fp8_schedules_keep_one_policy_per_accumulation_window(self):
        seen = []
        model = nn.Linear(2, 1)
        trainer = None

        def step(current, value):
            seen.append((trainer.execution_mode, finite_checks_enabled()))
            return current(value).square().mean()

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), fp8=True, step_fn=step,
                          accumulation_steps=2, execution_mode="max_speed",
                          mode_schedule={2: "analyze", 3: "max_speed"})
        trainer.train_epoch([torch.ones(2, 2)] * 5)
        self.assertEqual(seen, [("max_speed", False)] * 2 + [("analyze", True)] * 2 + [("max_speed", False)])

    def test_nested_runtime_contexts_restore_flags_after_errors(self):
        fast = TrainingRuntime(nn.Identity(), execution_mode="max_speed")
        strict = TrainingRuntime(nn.Identity(), execution_mode="analyze")
        with fast.autocast():
            self.assertFalse(finite_checks_enabled())
            with self.assertRaisesRegex(RuntimeError, "boundary"):
                fast.set_execution_mode("analyze")
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with strict.autocast():
                    self.assertTrue(finite_checks_enabled())
                    raise RuntimeError("intentional")
            self.assertFalse(finite_checks_enabled())
        self.assertTrue(finite_checks_enabled())

    def test_sparse_gradient_processing_has_identical_inputs_in_all_modes(self):
        class Square(GradientStrategy):
            def get_config(self):
                return {}

            def process(self, parameters):
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad._values().square_()

        initial = torch.randn(3, 2)
        expected = initial.clone()
        expected[0] -= .4  # Two duplicate unit gradients sum to 2, then square to 4.
        for mode in (None, "max_speed", "balanced", "analyze"):
            model = nn.Embedding(3, 2, sparse=True)
            with torch.no_grad():
                model.weight.copy_(initial)
            runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.1),
                                      gradient_strategy=Square(), execution_mode=mode)
            loss = runtime.compute(torch.tensor([0, 0]), lambda value: runtime.model(value).sum())
            runtime.backward(loss)
            runtime.step()
            torch.testing.assert_close(model.weight, expected, rtol=1e-6, atol=1e-6)
