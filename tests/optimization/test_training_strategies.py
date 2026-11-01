import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepopt.data.resumable import ResumableDataLoader
from deepopt.engine import OptimizationWindow, Trainer, TrainingRuntime
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.engine.strategies import (
    BackwardLoss, ClipGradNorm, ClipGradValue, ExplicitObjective,
    GradientStrategy, LossStrategy, MicrobatchMean,
)
from deepopt.module import TrainingModule
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.regression import RegressionTask


class ScheduledLoss(ExplicitObjective):
    """A small stateful policy makes missing checkpoint state observable."""

    def __init__(self, increment=0.1):
        self.increment, self.calls = increment, 0

    def get_config(self):
        return {"increment": self.increment}

    def prepare(self, output, *, accumulation_steps):
        result = super().prepare(output, accumulation_steps=accumulation_steps)
        self.calls += 1
        return BackwardLoss(result.loss * (1 + self.increment * self.calls), result.normalization_weight)

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


class ScheduledGradients(GradientStrategy):
    def __init__(self):
        self.updates = 0

    def get_config(self):
        return {"version": 1}

    def process(self, parameters):
        self.updates += 1
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(self.updates)

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state["updates"]


class TrainingStrategiesTests(unittest.TestCase):
    def test_runtime_accepts_arbitrary_computation_without_task_or_optimizer(self):
        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model)
        x = torch.ones(3, 2)
        result = runtime.compute({"input": x}, lambda batch: {"prediction": model(batch["input"])})
        self.assertTrue(torch.equal(result["prediction"], model(x)))
        runtime.backward(result["prediction"].sum())
        self.assertIsNotNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "requires an optimizer"):
            runtime.step()
        with self.assertRaisesRegex(RuntimeError, "requires an optimizer"):
            runtime.zero_grad()
        with self.assertRaisesRegex(ValueError, "with an optimizer"):
            Trainer(runtime, task=RegressionTask())

    def test_fit_uses_the_supplied_runtime_in_both_phases(self):
        class TupleRuntime(TrainingRuntime):
            def prepare_batch(self, batch):
                return super().prepare_batch((batch["x"], batch["y"]))

        model = nn.Linear(2, 1)
        runtime = TupleRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        trainer = Trainer(runtime, task=RegressionTask())
        rows = []
        batches = [{"x": torch.ones(3, 2), "y": torch.zeros(3, 1)}]
        trainer.fit(batches, batches, epochs=1, callbacks=[lambda t, row: rows.append(row)])
        self.assertIs(trainer.runtime, runtime)
        self.assertIs(trainer.optimizer, runtime.optimizer)
        self.assertEqual(rows[0]["train"]["samples"], 3)
        self.assertEqual(rows[0]["validation"]["samples"], 3)
        with self.assertRaisesRegex(ValueError, "supplied runtime"):
            Trainer(runtime, device="cpu", task=RegressionTask())

    def test_manual_window_matches_fit_for_both_loss_strategies(self):
        for strategy_type in (ExplicitObjective, MicrobatchMean):
            with self.subTest(strategy=strategy_type):
                torch.manual_seed(7)
                module = TrainingModule(nn.Linear(2, 1).double(), task=RegressionTask())
                reference = copy.deepcopy(module)
                batches = [(torch.randn(n, 2, dtype=torch.double), torch.randn(n, 1, dtype=torch.double))
                           for n in (3, 1, 4, 2, 1)]
                trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.02),
                                  accumulation_steps=2, loss_strategy=strategy_type())
                trainer.fit(batches, batches, epochs=2)
                runtime = TrainingRuntime(reference, torch.optim.SGD(reference.parameters(), lr=0.02))
                updates = 0
                for _ in range(2):
                    runtime.zero_grad()
                    window = OptimizationWindow(runtime, accumulation_steps=2, loss_strategy=strategy_type())
                    for batch in batches:
                        updates += window.backward(runtime.compute(batch, reference.training_step))
                    updates += window.flush()
                    self.assertFalse(window.flush())  # No duplicate update for an empty window.
                self.assertEqual(updates, trainer.state.optimizer_steps)
                for actual, expected in zip(reference.parameters(), module.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_microbatch_mean_matches_explicit_pytorch_objective_and_partial_window(self):
        for reduction in ("mean", "sum"):
            for custom_loss in (False, True):
                with self.subTest(reduction=reduction, custom_loss=custom_loss):
                    torch.manual_seed(6)
                    model = nn.Linear(2, 1).double()
                    reference = copy.deepcopy(model)
                    batches = [(torch.randn(n, 2, dtype=torch.double), torch.randn(n, 1, dtype=torch.double))
                               for n in (4, 1, 2)]
                    if custom_loss:
                        task = RegressionTask(
                            loss_fn=lambda predictions, targets: getattr((predictions - targets).square(), reduction)())
                    else:
                        task = RegressionTask(reduction_type=reduction)
                    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.03),
                                      task=task, accumulation_steps=2, loss_strategy=MicrobatchMean())
                    trainer.train_epoch(batches)
                    optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
                    for window in (batches[:2], batches[2:]):
                        optimizer.zero_grad()
                        # Average the selected scalar objectives as supplied;
                        # a summed microbatch loss remains a sum.
                        objective = sum(getattr((reference(x) - y).square(), reduction)()
                                        for x, y in window) / len(window)
                        objective.backward()
                        optimizer.step()
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_gradient_strategies_run_after_accumulation_normalization(self):
        for strategy in (ClipGradNorm(0.2), ClipGradValue(0.2)):
            with self.subTest(strategy=type(strategy)):
                torch.manual_seed(2)
                model = nn.Linear(2, 1)
                reference = copy.deepcopy(model)
                runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1),
                                          gradient_strategy=strategy)
                x = torch.tensor([[2., 4.], [3., 6.], [1., 3.]])
                runtime.backward(model(x).square().sum())
                runtime.step(samples=len(x))
                optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
                reference(x).square().mean().backward()
                if isinstance(strategy, ClipGradNorm):
                    nn.utils.clip_grad_norm_(reference.parameters(), 0.2)
                else:
                    nn.utils.clip_grad_value_(reference.parameters(), 0.2)
                optimizer.step()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected)
                with self.assertRaisesRegex(ValueError, "not both"):
                    TrainingRuntime(model, runtime.optimizer, grad_clip_norm=1, gradient_strategy=strategy)

    def test_bad_custom_gradient_processing_aborts_updates_and_clears_gradients(self):
        class NonfiniteGradients(GradientStrategy):
            def get_config(self):
                return {}

            def process(self, parameters):
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.fill_(float("nan"))

        model = nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), task=RegressionTask(),
                          gradient_strategy=NonfiniteGradients())
        with self.assertRaisesRegex(FloatingPointError, "after processing"):
            trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        self.assertEqual(trainer.state.optimizer_steps, 0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)

    def test_bad_custom_loss_is_rejected_before_optimizer_update(self):
        class BadLoss(LossStrategy):
            def __init__(self, result):
                self.result = result

            def get_config(self):
                return {}

            def prepare(self, output, *, accumulation_steps):
                return self.result

        for result in (torch.tensor(2.), BackwardLoss(torch.ones(2)), BackwardLoss(torch.tensor(float("nan"))),
                       BackwardLoss(torch.tensor(2.), 0), BackwardLoss(torch.tensor(2.), True)):
            model = nn.Linear(2, 1)
            runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
            before = model.weight.detach().clone()
            window = OptimizationWindow(runtime, loss_strategy=BadLoss(result))
            with self.assertRaises((ValueError, FloatingPointError)):
                window.backward(BatchOutput(model(torch.ones(2, 2)).mean(), 2, "mean"))
            self.assertEqual(window.pending_batches, 0)
            self.assertIsNone(model.weight.grad)
            torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def setup_stateful(self):
        torch.manual_seed(14)
        dataset = TensorDataset(torch.randn(11, 2), torch.randn(11, 1))
        model = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.2), nn.Linear(4, 1))
        runtime = TrainingRuntime(model, torch.optim.AdamW(model.parameters(), lr=0.01),
                                  gradient_strategy=ScheduledGradients())
        trainer = Trainer(runtime, task=RegressionTask(), accumulation_steps=2, loss_strategy=ScheduledLoss(),
                          scheduler=torch.optim.lr_scheduler.StepLR(runtime.optimizer, 1, gamma=0.9))
        generator = torch.Generator().manual_seed(9)
        loader = ResumableDataLoader(dataset, batch_size=3, generator=generator)
        return trainer, loader, DataLoader(dataset, batch_size=4), {"train": generator}

    def test_stateful_strategies_resume_exactly_at_step_and_epoch_boundaries(self):
        full, train, validation, _ = self.setup_stateful()
        full.fit(train, validation, epochs=3)
        for kind in ("step", "epoch"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                interrupted, train, validation, generators = self.setup_stateful()
                path = Path(directory) / "checkpoint.pt"

                def stop(trainer, event):
                    save_checkpoint(path, trainer, {}, generators=generators,
                                    history=[] if kind == "step" else [event], kind=kind, train_loader=train)
                    raise InterruptedError()

                callback_key = "step_callbacks" if kind == "step" else "callbacks"
                with self.assertRaises(InterruptedError):
                    interrupted.fit(train, validation, epochs=3, **{callback_key: [stop]})
                resumed, train, validation, generators = self.setup_stateful()
                restore_checkpoint(read_checkpoint(path), resumed, {}, generators=generators, train_loader=train)
                resumed.fit(train, validation, epochs=3)
                self.assertEqual(resumed.loss_strategy.calls, full.loss_strategy.calls)
                self.assertEqual(resumed.runtime.gradient_strategy.updates, full.runtime.gradient_strategy.updates)
                self.assertEqual(resumed.state, full.state)
                self.assertEqual(resumed.scheduler.state_dict(), full.scheduler.state_dict())
                for actual, expected in zip(resumed.model.parameters(), full.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_resume_checks_strategy_configuration_before_loading_model(self):
        trainer, _, _, generators = self.setup_stateful()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, trainer, {}, generators=generators, history=[])
            payload = read_checkpoint(path)
        trainer.loss_strategy.increment = 0.3
        with torch.no_grad():
            trainer.model[0].weight.add_(2)
        before = trainer.model[0].weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "same loss strategy"):
            restore_checkpoint(payload, trainer, {}, generators=generators)
        torch.testing.assert_close(trainer.model[0].weight, before, rtol=0, atol=0)

    def test_legacy_format_four_is_accepted_only_for_original_strategies(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), task=RegressionTask(), grad_clip_norm=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, trainer, {}, generators={}, history=[])
            payload = read_checkpoint(path)
        payload.pop("strategies")
        restore_checkpoint(payload, trainer, {}, generators={})
        other = Trainer(model, trainer.optimizer, task=RegressionTask(), grad_clip_norm=1, loss_strategy=MicrobatchMean())
        with self.assertRaisesRegex(ValueError, "no strategy state"):
            restore_checkpoint(payload, other, {}, generators={})


if __name__ == "__main__":
    unittest.main()
