import math
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from deepopt import FiniteCheck, TensorEvent, Trainer, TrainingRuntime
from deepopt.observers import CheckObservers, all_finite


class ObserverTests(unittest.TestCase):
    def test_default_disabled_and_invalid_configuration(self):
        for disabled in (False, []):
            dispatcher = CheckObservers(disabled)
            self.assertFalse(dispatcher.interested("loss"))
            dispatcher.observe("loss", torch.tensor(float("nan")))
        with self.assertRaises(FloatingPointError):
            CheckObservers().observe("loss", torch.tensor(float("nan")))
        for invalid in (None, 1, "loss", [None], [True]):
            with self.subTest(checks=invalid), self.assertRaises(TypeError):
                CheckObservers(invalid)

    def test_disabled_and_unsubscribed_observers_do_not_materialize_payloads(self):
        def unavailable():
            raise AssertionError("Unobserved tensors should not be materialized")
            yield

        for dispatcher in (CheckObservers(False), CheckObservers([FiniteCheck(kinds={"loss"})])):
            dispatcher.observe("activation", unavailable())
        checker = FiniteCheck()
        dispatcher = CheckObservers([checker])
        checker.enabled = False
        dispatcher.observe("loss", unavailable())
        checker.enabled = True
        with self.assertRaises(FloatingPointError):
            dispatcher.observe("loss", torch.tensor(float("nan")))

    def test_named_future_events_are_filtered_and_do_not_retain_a_graph(self):
        captured = []
        references = []

        def record(event):
            self.assertIsInstance(event, TensorEvent)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(not tensor.requires_grad for tensor in event.tensors))
            captured.append((event.kind, event.name))
            references.extend((weakref.ref(event), weakref.ref(event.tensors[0])))

        record.kinds = {"activation", "custom_tensor"}
        runtime = TrainingRuntime(nn.Identity(), checks=[record])
        source = torch.tensor(2., requires_grad=True)
        value = source.square()
        runtime.observe("activation", value, name="encoder")
        runtime.observe("custom_tensor", value, name="attention")
        runtime.backward(value)
        self.assertEqual(captured, [("activation", "encoder"), ("custom_tensor", "attention")])
        self.assertEqual(source.grad.item(), 4.)
        self.assertTrue(all(reference() is None for reference in references))

    def test_finite_check_accepts_integral_masks_and_sparse_tensor_events(self):
        self.assertTrue(all_finite([torch.ones(65536, dtype=torch.bool), torch.arange(65536)]))
        sparse = torch.sparse_coo_tensor(torch.tensor([[0, 0]]), torch.tensor([3e38, 3e38]), (1,))
        checker = CheckObservers([FiniteCheck(kinds={"activation"})])
        with self.assertRaisesRegex(FloatingPointError, "activation"):
            checker.observe("activation", sparse)

    def test_disabled_checks_cover_the_whole_default_training_and_validation_path(self):
        for task in (None, "regression"):
            with self.subTest(task=task):
                torch.manual_seed(7)
                model = nn.Linear(2, 1)
                options = {"loss_fn": nn.functional.mse_loss}
                if task is not None:
                    from deepopt.tasks.regression import RegressionTask
                    options = {"task": RegressionTask(metrics_factory=lambda: {})}
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                                  checks=False, accumulation_steps=2, grad_clip_norm=1., **options)
                batches = [(torch.ones(2, 2), torch.zeros(2, 1))] * 3
                with patch("torch.isfinite", side_effect=AssertionError("Hidden finite scan")):
                    trainer.fit(batches, batches, epochs=1)
                self.assertEqual(trainer.state.optimizer_steps, 2)

    def test_nonfinite_reports_require_disabling_best_metric_selection(self):
        def make_trainer():
            model = nn.Linear(2, 1)
            return Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), checks=False,
                           step_fn=lambda model, batch: model(batch).sum() * float("nan"))

        trainer = make_trainer()
        trainer.fit([torch.ones(1, 2)], [torch.ones(1, 2)], epochs=1, monitor=None)
        self.assertTrue(any(not math.isfinite(p.flatten()[0].item()) for p in trainer.model.parameters()))
        trainer = make_trainer()
        with self.assertRaisesRegex(ValueError, "monitored metric must be finite"):
            trainer.fit([torch.ones(1, 2)], [torch.ones(1, 2)], epochs=1)
        self.assertIsNone(trainer.state.best_score)

    def test_trainer_inherits_supplied_runtime_checks_and_rejects_conflicts(self):
        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01), checks=False)
        self.assertIs(Trainer(runtime, loss_fn=nn.functional.mse_loss).runtime, runtime)
        for checks in (True, False, []):
            with self.subTest(checks=checks), self.assertRaisesRegex(ValueError, "supplied runtime"):
                Trainer(runtime, loss_fn=nn.functional.mse_loss, checks=checks)


if __name__ == "__main__":
    unittest.main()
