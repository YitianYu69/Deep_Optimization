import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from deepopt import Checkpoint, LossResult, MultiOptimizer, Trainer
from deepopt.engine.strategies import ExplicitObjective
from deepopt.engine.checkpointing.checkpoint import read_checkpoint


def weighted_step(model, batch):
    x, y = batch
    loss = nn.functional.mse_loss(model(x), y)
    return {"loss": LossResult.from_mean(loss, weight=y.numel()), "sample_count": len(y)}


class PolicyBlackBoxTests(unittest.TestCase):
    def assert_nested_close(self, actual, expected, *, exact=False):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0 if exact else 1e-12, atol=0 if exact else 1e-12)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_close(actual[key], expected[key], exact=exact)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_close(left, right, exact=exact)
        else:
            self.assertEqual(actual, expected)

    def test_all_modes_match_torch_reference_for_weighted_partial_windows(self):
        torch.manual_seed(17)
        initial = nn.Linear(2, 1).double()
        batches = [(torch.randn(size, 2).double(), torch.randn(size, 1).double()) for size in (3, 1, 4, 2, 1)]
        for optimizer_type in (torch.optim.SGD, torch.optim.AdamW, MultiOptimizer):
            def make_optimizer(model):
                if optimizer_type is MultiOptimizer:
                    return MultiOptimizer({"weight": torch.optim.SGD([model.weight], lr=.01),
                                           "bias": torch.optim.AdamW([model.bias], lr=.01)})
                return optimizer_type(model.parameters(), lr=.01)

            reference = copy.deepcopy(initial)
            optimizer = make_optimizer(reference)
            for start in range(0, len(batches), 2):
                window = batches[start:start + 2]
                x, y = (torch.cat([batch[index] for batch in window]) for index in (0, 1))
                nn.functional.mse_loss(reference(x), y).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            reports = []
            mode_optimizer_state = None
            for mode in (None, "max_speed", "balanced", "analyze"):
                with self.subTest(optimizer=optimizer_type.__name__, mode=mode):
                    model = copy.deepcopy(initial)
                    trainer = Trainer(model, make_optimizer(model), step_fn=weighted_step,
                                      accumulation_steps=2, execution_mode=mode)
                    reports.append(trainer.train_epoch(batches))
                    for expected, actual in zip(reference.parameters(), model.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    self.assert_nested_close(trainer.optimizer.state_dict(), optimizer.state_dict())
                    if mode_optimizer_state is None:
                        mode_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
                    else:
                        self.assert_nested_close(trainer.optimizer.state_dict(), mode_optimizer_state, exact=True)
                    self.assertEqual(trainer.state.optimizer_steps, 3)
            self.assertTrue(all(report["loss"] == reports[0]["loss"] for report in reports))

    def test_switching_is_boundary_safe_and_schedule_is_update_based(self):
        model = nn.Linear(1, 1)
        seen = []
        trainer = None

        def step(model, batch):
            seen.append(trainer.execution_mode)
            with self.assertRaisesRegex(RuntimeError, "boundary"):
                trainer.set_execution_mode("analyze")
            return model(batch).square().mean()

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), step_fn=step,
                          execution_mode="max_speed", mode_schedule={2: "analyze", 3: "max_speed"})
        trainer.train_epoch([torch.ones(2, 1)] * 3)
        self.assertEqual(seen, ["max_speed", "analyze", "max_speed"])
        trainer.set_execution_mode("balanced")
        self.assertEqual(trainer.execution_mode, "balanced")

    def test_legacy_checks_remain_strict_while_max_speed_is_explicit_opt_out(self):
        for mode in (None, "analyze", "max_speed"):
            model = nn.Linear(1, 1)
            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), execution_mode=mode,
                              step_fn=lambda model, batch: model(batch).sum() * float("nan"))
            if mode == "max_speed":
                result = trainer.train_epoch([torch.ones(1, 1)])
                self.assertTrue(math.isnan(result["loss"]))
            else:
                with self.assertRaises(FloatingPointError):
                    trainer.train_epoch([torch.ones(1, 1)])

    def test_diagnostic_changes_resume_but_identity_and_math_changes_do_not(self):
        batches = [(torch.ones(2, 1), torch.zeros(2, 1))]

        def make(mode, identity, strategy=None):
            model = nn.Linear(1, 1)
            return Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), loss_fn=nn.functional.mse_loss,
                           execution_mode=mode, experiment_identity=identity, loss_strategy=strategy)

        with tempfile.TemporaryDirectory() as directory:
            identity = {"recipe_version": "v1", "dataset_version": "fixed"}
            original = make("max_speed", identity)
            original.fit(batches, epochs=1, callbacks=[Checkpoint(directory)], progress_bar=False)
            payload = read_checkpoint(Path(directory, "last.pt"))
            self.assertEqual(payload["experiment_identity"]["required"], identity)
            self.assertEqual(payload["diagnostics"]["active_mode"], "max_speed")
            self.assertEqual(len(payload["experiment_identity"]["observed"]["config_fingerprint"]), 64)
            restored = make("analyze", identity)
            Checkpoint(directory).restore(restored, batches)
            self.assertEqual(restored.execution_mode, "analyze")
            self.assertEqual(restored.runtime.mode_controller.completed_updates, 1)
            with self.assertRaisesRegex(ValueError, "experiment_identity"):
                Checkpoint(directory).restore(make("analyze", {"recipe_version": "v2"}), batches)
            with self.assertRaisesRegex(ValueError, "loss strategy"):
                Checkpoint(directory).restore(make("analyze", identity, ExplicitObjective(distributed_reduction="global_sum")), batches)

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile is required")
    def test_mode_switches_do_not_recompile_the_model(self):
        torch._dynamo.reset()
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        try:
            model = torch.compile(nn.Linear(2, 1), backend=backend, fullgraph=True)
            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                              loss_fn=nn.functional.mse_loss, execution_mode="max_speed",
                              mode_schedule={2: "analyze", 3: "max_speed"})
            trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))] * 3)
            self.assertEqual(len(graphs), 1)
        finally:
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
