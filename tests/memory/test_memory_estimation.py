import copy
from dataclasses import dataclass
import json
import random
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from deepopt import MultiOptimizer, Trainer, TrainingRuntime
from deepopt.engine.strategies import ExplicitObjective, GradientStrategy
from deepopt.memory import MemoryEstimationError, estimate_training_memory
from deepopt.memory.estimation import _StorageTrace
from deepopt.tasks.classification import ClassificationTask


class CountingLoss(ExplicitObjective):
    def __init__(self):
        self.calls = 0

    def prepare(self, output, *, accumulation_steps):
        self.calls += 1
        return super().prepare(output, accumulation_steps=accumulation_steps)


class CountingGradients(GradientStrategy):
    def __init__(self):
        self.calls = 0

    def get_config(self):
        return {}

    def process(self, parameters):
        self.calls += 1
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(0.5)


class MemoryEstimationTests(unittest.TestCase):
    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_fresh_adam_lazy_states_and_second_window_are_counted_without_real_mutation(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, accumulation_steps=2)
        before = copy.deepcopy(model.state_dict())
        report = estimate_training_memory(trainer, (torch.zeros(5, 3), torch.zeros(5, 2)))
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        self.assertEqual(report.initial_bytes_by_device, {"cpu": parameter_bytes})
        self.assertEqual(report.persistent_bytes_by_device["cpu"], parameter_bytes * 3 + 4 * 2)
        self.assertGreater(report.peak_bytes_by_device["cpu"], report.persistent_bytes_by_device["cpu"])
        self.assert_nested_equal(model.state_dict(), before)
        self.assertEqual(optimizer.state, {})
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(trainer.state.optimizer_steps, 0)
        json.dumps(report.as_dict())

    def test_resumed_optimizer_scheduler_strategies_modes_and_rng_are_unchanged(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(0.2), nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        loss_strategy, gradient_strategy = CountingLoss(), CountingGradients()

        def loss(predictions, targets):
            return F.mse_loss(predictions, targets) * (1 + random.random())

        trainer = Trainer(model, optimizer, loss_fn=loss, scheduler=scheduler,
                          loss_strategy=loss_strategy, gradient_strategy=gradient_strategy, accumulation_steps=2)
        batch = (torch.randn(5, 3), torch.randn(5, 2))
        trainer.train_epoch([batch, batch])
        model.eval()
        model[2].train()
        model_before, optimizer_before = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        scheduler_before = copy.deepcopy(scheduler.state_dict())
        state_before, modes = copy.deepcopy(trainer.state), [module.training for module in model.modules()]
        python_rng, torch_rng = random.getstate(), torch.get_rng_state()
        calls = (loss_strategy.calls, gradient_strategy.calls)
        trainer.runtime.invalidate()  # Checkpoint planning can run before re-enabling the real runtime.
        report = estimate_training_memory(trainer, batch)
        self.assertGreater(report.peak_bytes_by_device["cpu"], 0)
        self.assert_nested_equal(model.state_dict(), model_before)
        self.assert_nested_equal(optimizer.state_dict(), optimizer_before)
        self.assert_nested_equal(scheduler.state_dict(), scheduler_before)
        self.assertEqual(trainer.state, state_before)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertEqual((loss_strategy.calls, gradient_strategy.calls), calls)
        self.assertEqual(random.getstate(), python_rng)
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            trainer.runtime.step()

    def test_sample_bytes_are_excluded_from_initial_credit_and_larger_inputs_increase_peak(self):
        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
        with FakeTensorMode(allow_fallback_kernels=False):
            small = estimate_training_memory(trainer, (torch.empty(2, 3), torch.empty(2, 2)))
            large = estimate_training_memory(trainer, (torch.empty(128, 3), torch.empty(128, 2)))
        self.assertEqual(small.initial_bytes_by_device, large.initial_bytes_by_device)
        self.assertGreater(large.peak_bytes_by_device["cpu"], small.peak_bytes_by_device["cpu"])

    def test_unused_already_placed_batch_storage_is_counted_without_initial_credit(self):
        class AlreadyPlacedRuntime(TrainingRuntime):
            def prepare_batch(self, batch):
                return batch

        model = nn.Linear(3, 2)
        runtime = AlreadyPlacedRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        trainer = Trainer(runtime, step_fn=lambda current, batch: current(batch["features"]).square().mean())
        with FakeTensorMode(allow_fallback_kernels=False):
            features = torch.empty(4, 3)
            ordinary = estimate_training_memory(trainer, {"features": features})
            extra = estimate_training_memory(trainer, {"features": features, "unused": torch.empty(1_000_000)})
        self.assertGreaterEqual(extra.peak_bytes_by_device["cpu"],
                                ordinary.peak_bytes_by_device["cpu"] + 4_000_000)
        self.assertEqual(extra.initial_bytes_by_device, ordinary.initial_bytes_by_device)
        self.assertEqual(extra.persistent_bytes_by_device, ordinary.persistent_bytes_by_device)

    def test_real_tensor_propagation_is_rejected_before_cloning_or_execution(self):
        from torch._functorch import config

        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
        real_batch = (torch.empty(4, 3), torch.empty(4, 2))
        with config.patch(fake_tensor_propagate_real_tensors=True):
            with FakeTensorMode(allow_fallback_kernels=False):
                fake_batch = (torch.empty(4, 3), torch.empty(4, 2))
            for sample in (real_batch, fake_batch):
                with self.subTest(fake=isinstance(sample[0], FakeTensor)), \
                        patch("deepopt.memory.estimation._clone_training", side_effect=AssertionError("must not clone")) as clone, \
                        self.assertRaisesRegex(MemoryEstimationError, "real tensor propagation disabled"):
                    estimate_training_memory(trainer, sample)
                clone.assert_not_called()

    def test_aliases_share_storage_and_discarded_temporaries_stop_counting(self):
        with FakeTensorMode(allow_fallback_kernels=False), _StorageTrace() as trace:
            first = torch.empty(100)
            alias = first.view(10, 10)
            self.assertEqual(trace.peak["cpu"], 400)
            del first, alias
            second = torch.empty(150)
            self.assertEqual(trace.peak["cpu"], 600)
            self.assertIsInstance(second, FakeTensor)

    def test_model_payload_deepcopy_never_clones_real_tensors(self):
        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
        batch = (torch.ones(4, 3), torch.zeros(4, 2))
        original_clone = torch.Tensor.clone

        def only_fake_clone(tensor, *args, **kwargs):
            if not isinstance(tensor, FakeTensor):
                raise AssertionError("Real payload clone")
            return original_clone(tensor, *args, **kwargs)

        with patch("torch.Tensor.clone", only_fake_clone):
            report = estimate_training_memory(trainer, batch)
        self.assertGreater(report.peak_bytes_by_device["cpu"], 0)

    def test_metrics_and_observers_are_not_invoked(self):
        def unexpected(*args, **kwargs):
            raise AssertionError("A diagnostic callback was executed")

        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss,
                          checks=[unexpected], metric_inputs=unexpected)
        trainer.metric_factory = unexpected
        report = estimate_training_memory(trainer, (torch.ones(4, 3), torch.zeros(4, 2)))
        self.assertGreater(report.peak_bytes_by_device["cpu"], 0)

    def test_builtin_classification_and_multi_optimizer_are_traceable(self):
        model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
        optimizer = MultiOptimizer({"first": torch.optim.AdamW(model[0].parameters(), lr=0.01),
                                    "last": torch.optim.SGD(model[2].parameters(), lr=0.1, momentum=0.9)})
        trainer = Trainer(model, optimizer, task=ClassificationTask(num_classes=2), grad_clip_norm=1.0)
        report = estimate_training_memory(trainer, (torch.empty(4, 3, device="meta"),
                                                     torch.empty(4, dtype=torch.long, device="meta")))
        self.assertGreater(report.persistent_bytes_by_device["cpu"], report.initial_bytes_by_device["cpu"])
        self.assertTrue(all(not child.state for child in optimizer.optimizers.values()))

    def test_dataclass_batch_does_not_require_a_known_batch_axis(self):
        @dataclass(frozen=True)
        class Batch:
            features: torch.Tensor
            targets: torch.Tensor

        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                          step_fn=lambda current, batch: F.mse_loss(current(batch.features), batch.targets))
        batch = Batch(torch.empty(2, 4, 3, device="meta"), torch.empty(2, 4, 2, device="meta"))
        self.assertGreater(estimate_training_memory(trainer, batch).peak_bytes_by_device["cpu"], 0)

    def test_bound_step_owner_is_cloned_with_shared_model_and_owned_tensors(self):
        class Computation:
            def __init__(self, model):
                self.model = model
                self.scale = torch.ones(())
                self.calls = 0

            def step(self, current, batch):
                if current is not self.model:
                    raise AssertionError("Model alias was broken")
                self.calls += 1
                return (current(batch) * self.scale).square().mean()

        model = nn.Linear(3, 2)
        computation = Computation(model)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), step_fn=computation.step)
        report = estimate_training_memory(trainer, torch.empty(4, 3, device="meta"))
        self.assertEqual(report.initial_bytes_by_device["cpu"], 36)
        self.assertEqual(computation.calls, 0)
        self.assertIs(computation.model, model)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_data_dependent_computation_fails_without_mutation_or_rng_drift(self):
        model = nn.Linear(3, 2)

        def step(current, batch):
            random.random()
            output = current(batch)
            return output.sum() if output.sum().item() > 0 else output.square().mean()

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), step_fn=step)
        before, rng = copy.deepcopy(model.state_dict()), random.getstate()
        with self.assertRaisesRegex(MemoryEstimationError, "unsupported.*DataDependentOutput"):
            estimate_training_memory(trainer, torch.ones(2, 3))
        self.assert_nested_equal(model.state_dict(), before)
        self.assertEqual(random.getstate(), rng)

    def test_real_scalar_sample_values_are_not_specialized_into_trace_branches(self):
        model = nn.Linear(1, 1)

        def step(current, batch):
            return current(torch.ones(2, 1)).sum() * batch.item()

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), step_fn=step)
        with self.assertRaisesRegex(MemoryEstimationError, "DataDependentOutput"):
            estimate_training_memory(trainer, torch.tensor(2.0, dtype=torch.float64))

    def test_no_grad_callers_still_trace_training_and_restore_their_grad_mode(self):
        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
        with torch.no_grad():
            report = estimate_training_memory(trainer, (torch.ones(4, 3), torch.zeros(4, 2)))
            self.assertFalse(torch.is_grad_enabled())
        self.assertGreater(report.peak_bytes_by_device["cpu"], 0)
        with torch.inference_mode():
            with self.assertRaisesRegex(MemoryEstimationError, "inference_mode"):
                estimate_training_memory(trainer, None)

    def test_distributed_contexts_and_wrappers_are_rejected_before_tracing(self):
        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
        with patch("torch.distributed.is_available", return_value=True), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2):
            with self.assertRaisesRegex(MemoryEstimationError, "multi-rank"):
                estimate_training_memory(trainer, None)
        wrapped = nn.DataParallel(model)
        trainer = Trainer(wrapped, torch.optim.SGD(wrapped.parameters(), lr=0.1), loss_fn=F.mse_loss)
        with self.assertRaisesRegex(MemoryEstimationError, "wrappers"):
            estimate_training_memory(trainer, None)

    def test_permissive_or_multiple_fake_modes_are_rejected(self):
        model = nn.Linear(3, 2)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
        with FakeTensorMode() as permissive:
            batch = (torch.empty(2, 3), torch.empty(2, 2))
        with self.assertRaisesRegex(MemoryEstimationError, "strict"):
            estimate_training_memory(trainer, batch)
        with FakeTensorMode(allow_fallback_kernels=False):
            inputs = torch.empty(2, 3)
        with FakeTensorMode(allow_fallback_kernels=False):
            targets = torch.empty(2, 2)
        with self.assertRaisesRegex(MemoryEstimationError, "one FakeTensorMode"):
            estimate_training_memory(trainer, (inputs, targets))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is required to verify zero device allocation")
    def test_cuda_estimation_does_not_allocate_real_device_payloads(self):
        device = torch.device("cuda", torch.cuda.current_device())
        model = nn.Linear(3, 2).to(device)
        trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=0.01),
                          device=device, loss_fn=F.mse_loss, accumulation_steps=2)
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        report = estimate_training_memory(trainer, (torch.empty(5, 3, device="meta"),
                                                     torch.empty(5, 2, device="meta")))
        torch.cuda.synchronize(device)
        self.assertEqual(torch.cuda.memory_allocated(device), allocated)
        self.assertEqual(torch.cuda.max_memory_allocated(device), allocated)
        self.assertGreater(report.peak_bytes_by_device[str(device)], report.initial_bytes_by_device[str(device)])


if __name__ == "__main__":
    unittest.main()
