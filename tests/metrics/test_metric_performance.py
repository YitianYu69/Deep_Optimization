import copy
import unittest
from unittest.mock import PropertyMock, patch

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torchmetrics import Metric

from deepopt import Trainer
from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.engine.training.steps import StepAdapter
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossReport, LossResult


class StoredPredictions(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, predictions, targets=None):
        self.values.append(predictions)

    def compute(self):
        return torch.cat([value.reshape(-1) for value in self.values]).mean()


class DetachCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function == torch.ops.aten.detach.default:
            self.count += 1
        return function(*args, **(kwargs or {}))


class MetricPerformanceTests(unittest.TestCase):
    def test_shared_metric_inputs_are_detached_once_without_copying_storage(self):
        predictions = torch.tensor([1., 2.], requires_grad=True)
        output = BatchOutput(predictions.square().mean(), metric_data=(predictions,))
        counts = []
        for size in (1, 4):
            metrics = {str(index): StoredPredictions() for index in range(size)}
            adapter = StepAdapter(nn.Identity(), step_fn=lambda model, batch: output, metrics=metrics)
            with DetachCounter() as counter:
                adapter.update_metrics(metrics, output)
            counts.append(counter.count)
            for metric in metrics.values():
                stored = metric.values[0]
                self.assertIsNone(stored.grad_fn)
                self.assertFalse(stored.requires_grad)
                self.assertEqual(stored.data_ptr(), predictions.data_ptr())
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[0], counts[1])
        output.loss.backward()
        torch.testing.assert_close(predictions.grad, predictions.detach())

    def test_metric_callbacks_do_not_build_or_retain_graphs_and_preserve_loss_gradients(self):
        for per_metric in (False, True):
            with self.subTest(per_metric=per_metric):
                model = nn.Linear(2, 1).double()
                inputs, targets = torch.randn(3, 2, dtype=torch.double), torch.randn(3, 1, dtype=torch.double)
                expected = torch.autograd.grad((model(inputs) - targets).square().mean(), tuple(model.parameters()))
                grad_modes = []

                def inputs_for_metrics(raw, batch):
                    grad_modes.append(torch.is_grad_enabled())
                    return raw["predictions"].square(), batch[1]

                adapter = StepAdapter(
                    model, loss_fn=nn.functional.mse_loss,
                    metrics={"stored": StoredPredictions()},
                    metric_inputs={"stored": inputs_for_metrics} if per_metric else inputs_for_metrics,
                )
                output = adapter.training_step((inputs, targets))
                data = output.metric_data["stored"] if per_metric else output.metric_data
                self.assertEqual(grad_modes, [False])
                self.assertFalse(data[0].requires_grad)
                self.assertIsNone(data[0].grad_fn)
                output.loss.backward()
                for parameter, gradient in zip(model.parameters(), expected):
                    torch.testing.assert_close(parameter.grad, gradient)

    def test_callback_returning_existing_tensor_is_detached_before_storage(self):
        inputs = torch.tensor([1., 2.], requires_grad=True)
        adapter = StepAdapter(
            nn.Identity(), loss_fn=nn.functional.mse_loss,
            metric_inputs=lambda raw, batch: (raw["predictions"], batch[1]),
        )
        output = adapter.training_step((inputs, torch.zeros_like(inputs)))
        self.assertFalse(output.metric_data[0].requires_grad)
        self.assertEqual(output.metric_data[0].data_ptr(), inputs.data_ptr())
        self.assertTrue(output.loss.requires_grad)

    def test_metric_mapping_preserves_legacy_loss_without_materializing_objective(self):
        values = torch.tensor([1., 3.], requires_grad=True)
        original_data = (values,)
        original = BatchOutput(values, 2, "none", metric_data=original_data)
        adapter = StepAdapter(nn.Identity(), step_fn=lambda model, batch: original,
                              metric_inputs=lambda raw, batch: (raw.loss,))
        with patch.object(BatchOutput, "objective", new_callable=PropertyMock,
                          side_effect=AssertionError("Metric mapping must not construct an objective")):
            output = adapter.training_step(None)
        self.assertIsNot(output, original)
        self.assertIs(original.metric_data, original_data)
        self.assertIs(output.loss, values)
        self.assertEqual(output.reduction, "none")
        self.assertEqual(output.sample_count, 2)
        output.loss_sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))

    def test_metric_mapping_preserves_explicit_objective_and_independent_report(self):
        loss = torch.tensor(3., requires_grad=True)
        objective = LossResult(loss, .5, LossReport(torch.tensor(7.), 2.))
        original = BatchOutput(objective, 4)
        adapter = StepAdapter(nn.Identity(), step_fn=lambda model, batch: original,
                              metric_inputs=lambda raw, batch: (raw.loss,))
        output = adapter.training_step(None)
        self.assertIs(output.objective, objective)
        self.assertIs(output.loss, loss)
        self.assertIsNone(original.metric_data)
        self.assertFalse(output.metric_data[0].requires_grad)
        output.loss_sum().backward()
        torch.testing.assert_close(loss.grad, torch.tensor(.5))

    def test_metric_callback_cannot_bypass_finite_loss_checks(self):
        model = nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())

        def mutate_loss(raw, batch):
            raw["loss"].fill_(float("nan"))
            return raw["predictions"], batch[1]

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1),
                          loss_fn=nn.functional.mse_loss, metric_inputs=mutate_loss)
        with self.assertRaises(FloatingPointError):
            trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        self.assertEqual(trainer.state.optimizer_steps, 0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)

    def test_reporting_failures_do_not_update_parent_or_child_metrics(self):
        largest = torch.finfo(torch.float64).max
        for total, weight, reported in ((largest, 1., True), (1., largest, True), (1., 1., False)):
            with self.subTest(total=total, weight=weight, reported=reported):
                child = StoredPredictions()
                metric = MetricAccumulator(
                    {"stored": child}, update_metrics=lambda metrics, output: metrics["stored"].update(output.loss)
                )
                report = LossReport(torch.tensor(total, dtype=torch.float64), weight)
                metric.update(BatchOutput(LossResult(torch.tensor(1.), report=report), 2))
                before = {name: getattr(metric, name).clone() for name in metric._defaults}
                next_report = report if reported else None
                with self.assertRaises((FloatingPointError, ValueError)):
                    metric.update(BatchOutput(LossResult(torch.tensor(2.), report=next_report), 3))
                for name, value in before.items():
                    torch.testing.assert_close(getattr(metric, name), value, rtol=0, atol=0)
                self.assertEqual(child.update_count, 1)
                self.assertEqual(len(child.values), 1)


if __name__ == "__main__":
    unittest.main()
