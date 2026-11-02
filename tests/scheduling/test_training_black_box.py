from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import MultiOptimizer, Trainer, TrainingRuntime
from deepopt.scheduling import ScheduleController, LinearWarmup, Constant


class SchedulingTrainingTests(unittest.TestCase):
    def test_accumulation_partial_windows_modes_and_validation(self):
        for accumulation in (1, 2, 4, 8):
            for mode in ("max_speed", "balanced", "analyze"):
                with self.subTest(accumulation=accumulation, mode=mode):
                    schedules = ScheduleController({"beta": LinearWarmup(.1, 1., 10)})
                    model = nn.Linear(1, 1)
                    seen = []
                    def step(model, batch):
                        seen.append(schedules["beta"])
                        return model(batch).square().mean() * schedules["beta"]
                    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), step_fn=step,
                                      schedules=schedules, accumulation_steps=accumulation, execution_mode=mode)
                    trainer.train_epoch([torch.ones(2, 1)] * 9)
                    expected = [LinearWarmup(.1, 1., 10).value_at(i // accumulation + 1) for i in range(9)]
                    self.assertEqual(seen, expected)
                    self.assertEqual(schedules.completed_update, (9 + accumulation - 1) // accumulation)
                    before = schedules.state_dict()
                    for _ in range(2):
                        trainer.evaluate([torch.ones(2, 1)])
                    self.assertEqual(before, schedules.state_dict())
                    self.assertEqual(seen[-2:], [expected[-1]] * 2)

    def test_failure_does_not_commit_prepared_coefficients(self):
        schedules = ScheduleController({"x": LinearWarmup(0, 1, 3)})
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        trainer = Trainer(model, optimizer, loss_fn=nn.functional.mse_loss, schedules=schedules)
        initial = schedules.state_dict()
        with patch.object(optimizer, "step", side_effect=RuntimeError("failed update")):
            with self.assertRaisesRegex(RuntimeError, "failed update"):
                trainer.train_epoch([(torch.ones(2, 1), torch.zeros(2, 1))])
        self.assertEqual(schedules.state_dict(), initial)
        self.assertEqual(trainer.state.optimizer_steps, 0)

    def test_validation_cannot_prepare_an_update(self):
        schedules = ScheduleController({"x": Constant(1)})
        model = nn.Linear(1, 1)
        def invalid_validation(model, batch):
            schedules.prepare_update(1)
            return model(batch).sum()
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                          step_fn=lambda model, batch: model(batch).sum(),
                          validation_step_fn=invalid_validation, schedules=schedules)
        initial = schedules.state_dict()
        with self.assertRaisesRegex(RuntimeError, "Validation"):
            trainer.evaluate([torch.ones(1, 1)])
        self.assertEqual(initial, schedules.state_dict())

    def test_adamw_sgd_and_multi_optimizer_match_eager_reference(self):
        for kind in ("sgd", "adamw", "multi"):
            torch.manual_seed(3)
            model = nn.Linear(1, 1)
            reference = deepcopy(model)
            def optimizer(model):
                if kind == "multi":
                    return MultiOptimizer({"weight": torch.optim.SGD([model.weight], lr=.01),
                                           "bias": torch.optim.AdamW([model.bias], lr=.01)})
                cls = torch.optim.SGD if kind == "sgd" else torch.optim.AdamW
                return cls(model.parameters(), lr=.01)
            actual_optimizer, reference_optimizer = optimizer(model), optimizer(reference)
            schedules = ScheduleController({"weight": LinearWarmup(.2, .8, 3)})
            trainer = Trainer(model, actual_optimizer, schedules=schedules, execution_mode="max_speed",
                              step_fn=lambda model, batch: model(batch).square().mean() * schedules["weight"])
            trainer.train_epoch([torch.ones(2, 1)] * 3)
            for update in range(1, 4):
                (reference(torch.ones(2, 1)).square().mean() * LinearWarmup(.2, .8, 3).value_at(update)).backward()
                reference_optimizer.step()
                reference_optimizer.zero_grad(set_to_none=True)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_manual_runtime_and_epoch_reporting(self):
        schedules = ScheduleController({"x": LinearWarmup(1, 3, 3)})
        model = nn.Linear(1, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01), checks=False)
        for update in range(1, 4):
            schedules.prepare_update(update)
            runtime.backward(model(torch.ones(1, 1)).sum() * schedules["x"])
            runtime.step()
            schedules.complete_update(update)
        self.assertEqual(schedules.completed_update, 3)
        schedules = ScheduleController({"epoch_weight": LinearWarmup(0, 1, 2, interval="epoch")})
        trainer = Trainer(model, runtime.optimizer, schedules=schedules,
                          step_fn=lambda model, batch: model(batch).square().mean())
        rows = []
        trainer.fit([torch.ones(1, 1)] * 2, epochs=2, callbacks=[lambda trainer, row: rows.append(row)], progress_bar=False)
        self.assertEqual([row["scheduled_values"]["epoch_weight"] for row in rows], [0., 1.])
        self.assertEqual([row["scheduled_values_start"]["epoch_weight"] for row in rows], [0., 1.])


if __name__ == "__main__":
    unittest.main()
