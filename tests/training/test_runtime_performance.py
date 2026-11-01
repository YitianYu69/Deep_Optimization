import copy
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from deepopt.engine import Trainer, TrainingRuntime
from deepopt.observers import all_finite
from deepopt.engine.strategies import GradientStrategy
from deepopt.tasks.base import BatchOutput


class RuntimePerformanceTests(unittest.TestCase):
    def test_no_step_callbacks_only_synchronize_epoch_timing(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          loss_fn=nn.MSELoss(), accumulation_steps=2)
        batches = [(torch.ones(2, 2), torch.zeros(2, 1))] * 5
        observed_steps = []
        with patch.object(trainer.runtime, "synchronize",
                          side_effect=lambda: observed_steps.append(trainer.state.optimizer_steps)):
            result = trainer.train_epoch(batches)
        self.assertEqual(observed_steps, [0, 3])
        self.assertEqual(trainer.state.optimizer_steps, 3)
        self.assertEqual(trainer.state.completed_epochs, 1)
        self.assertEqual(result["samples"], 10)
        self.assertGreater(result["seconds"], 0)

    def test_step_callbacks_receive_synchronized_timing_excluding_callback_work(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, loss_fn=nn.MSELoss())
        clock, timings, events = [0.0], [], []
        original_step = optimizer.step

        def synchronize():
            clock[0] += 2.0

        def optimizer_step(*args, **kwargs):
            clock[0] += 5.0
            return original_step(*args, **kwargs)

        def callback(current, event):
            self.assertTrue(current.at_step_boundary)
            self.assertTrue(all(parameter.grad is None for parameter in current.model.parameters()))
            timings.append(current.epoch_progress.seconds)
            events.append(event)
            clock[0] += 100.0

        with patch.object(trainer.runtime, "synchronize", side_effect=synchronize) as sync, \
                patch.object(optimizer, "step", side_effect=optimizer_step), \
                patch("deepopt.engine.training.loops.perf_counter", side_effect=lambda: clock[0]):
            result = trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))] * 2,
                                         step_callbacks=[callback])
        self.assertEqual(sync.call_count, 4)
        self.assertEqual(timings, [7.0, 14.0])
        self.assertEqual(result["seconds"], 16.0)
        self.assertEqual([event["optimizer_step"] for event in events], [1, 2])
        self.assertEqual([event["batches_processed"] for event in events], [1, 2])
        self.assertEqual([event["samples_processed"] for event in events], [2, 4])

    def test_completed_batch_output_is_released_before_callback_and_next_forward(self):
        for use_callbacks in (False, True):
            with self.subTest(step_callbacks=use_callbacks):
                model = nn.Linear(2, 1)
                references = []
                callbacks_seen = []

                def assert_released():
                    self.assertTrue(all(reference() is None for reference in references))

                def step(current, batch):
                    assert_released()
                    predictions = current(batch)
                    output = BatchOutput(predictions.square().mean(), sample_count=len(batch),
                                         metric_data=predictions)
                    references.extend((weakref.ref(output), weakref.ref(predictions)))
                    return output

                def callback(current, event):
                    assert_released()
                    callbacks_seen.append(event["optimizer_step"])

                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                                  step_fn=step, accumulation_steps=2)
                trainer.train_epoch([torch.ones(2, 2)] * 3,
                                    step_callbacks=[callback] if use_callbacks else [])
                assert_released()
                self.assertEqual(callbacks_seen, [1, 2] if use_callbacks else [])

    def test_large_real_and_complex_gradients_detect_all_nonfinite_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                      torch.complex64, torch.complex128):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    backing = torch.ones(65536, dtype=dtype)
                    gradient = backing[::2] if strided else backing
                    self.assertTrue(all_finite([gradient, torch.empty(0, dtype=dtype)]))
                    for bad in (float("nan"), float("inf"), float("-inf")):
                        gradient[-1] = bad
                        self.assertFalse(all_finite([gradient]))
                        gradient[-1] = 1
                    if gradient.is_complex():
                        gradient[-1] = complex(1, float("inf"))
                        self.assertFalse(all_finite([gradient]))
        self.assertTrue(all_finite([]))

    def test_normalization_overflow_aborts_without_updating_parameters(self):
        model = nn.Linear(1, 1, bias=False)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.01))
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, 3e38)
        with self.assertRaisesRegex(FloatingPointError, "normalized gradient"):
            runtime.step(gradient_divisor=0.5)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()

    def test_finite_check_preserves_higher_order_gradient_without_building_a_graph(self):
        parameter = torch.ones(32768, requires_grad=True)
        gradient, = torch.autograd.grad(parameter.square().sum(), parameter, create_graph=True)
        with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: self.fail("Finite checking retained an autograd tensor"),
                lambda tensor: tensor):
            self.assertTrue(all_finite([gradient]))
        self.assertTrue(gradient.requires_grad)
        gradient.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2.))

    def test_custom_strategy_replacement_gradients_are_checked_again(self):
        class ReplaceGradient(GradientStrategy):
            def get_config(self):
                return {}

            def process(self, parameters):
                parameters = list(parameters)
                parameters[-1].grad = torch.full_like(parameters[-1], float("inf"))

        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.01),
                                  gradient_strategy=ReplaceGradient())
        before = copy.deepcopy(model.state_dict())
        model.weight.grad = torch.ones_like(model.weight)
        self.assertIsNone(model.bias.grad)
        with self.assertRaisesRegex(FloatingPointError, "after processing"):
            runtime.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to inspect device synchronization")
    def test_cuda_finite_reduction_reads_only_one_scalar(self):
        gradients = [torch.ones(64, device="cuda", dtype=dtype)
                     for dtype in (torch.float32, torch.float64, torch.float16)]
        gradients.append(torch.empty(0, device="cuda"))
        # Warm up dispatch before counting device-to-host scalar reads.
        self.assertTrue(all_finite(gradients))
        for bad in (None, float("nan"), float("inf"), float("-inf")):
            gradients[-2][-1] = 1 if bad is None else bad
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                finite = all_finite(gradients)
            self.assertEqual(finite, bad is None)
            reads = sum(event.count for event in profile.key_averages()
                        if event.key == "aten::_local_scalar_dense")
            self.assertEqual(reads, 1)


if __name__ == "__main__":
    unittest.main()
