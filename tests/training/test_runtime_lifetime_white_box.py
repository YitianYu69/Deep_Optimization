"""Internal diagnostic callbacks must not extend runtime/native-core lifetime."""

from contextlib import contextmanager
import gc
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from deepopt import TrainingRuntime
from deepopt.engine.strategies.loss import ExplicitObjective
from deepopt.engine.training.loops import EpochRunner
from deepopt.engine.training.steps import StepAdapter


@contextmanager
def without_cyclic_collection():
    """Observe reference ownership without an opportunistic GC hiding cycles."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class RuntimeLifetimeWhiteBoxTests(unittest.TestCase):
    def test_optimizer_stepper_does_not_own_its_runtime_or_core(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with without_cyclic_collection():
            runtime = TrainingRuntime(model, optimizer)
            runtime_ref = weakref.ref(runtime)
            core_ref = weakref.ref(runtime.core)
            del runtime

            self.assertIsNone(runtime_ref(), "The optimizer stepper retained its owning runtime")
            self.assertIsNone(core_ref(), "The execution core outlived its runtime")

    def test_retained_metric_does_not_own_its_observer_runtime_or_core(self):
        model = nn.Linear(2, 1)
        with without_cyclic_collection():
            # Omit the optimizer to isolate the metric observer retention path
            # independently of the optimizer stepper's back-reference.
            runtime = TrainingRuntime(model)
            runner = EpochRunner(
                runtime, StepAdapter(model, loss_fn=nn.functional.mse_loss),
                accumulation_steps=1, loss_strategy=ExplicitObjective(),
            )
            metric = runner.new_metrics()
            runtime_ref = weakref.ref(runtime)
            core_ref = weakref.ref(runtime.core)
            del runner, runtime

            self.assertIsNotNone(metric)
            self.assertIsNone(runtime_ref(), "A retained metric kept the runtime observer alive")
            self.assertIsNone(core_ref(), "The metric retained the execution core")

    def test_runner_owns_runtime_and_metric_observer_forwards_while_alive(self):
        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model)
        runner = EpochRunner(
            runtime, StepAdapter(model, loss_fn=nn.functional.mse_loss),
            accumulation_steps=1, loss_strategy=ExplicitObjective(),
        )
        runtime_ref = weakref.ref(runtime)
        del runtime
        self.assertIsNotNone(runtime_ref())
        self.assertIs(runtime_ref(), runner.runtime)

        metric = runner.new_metrics()
        tensors = (torch.ones(1),)
        with patch.object(runner.runtime, "observe") as observe:
            metric._observe("metrics", tensors, name="retained metric")
            observe.assert_called_once_with("metrics", tensors, name="retained metric")


if __name__ == "__main__":
    unittest.main()
