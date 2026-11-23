"""Real CUDA capture/replay with process defaults and immutable handle snapshots."""

from copy import deepcopy
import unittest

import torch
from torch import nn

from deepopt import CUDAGraphCore, cuda_graph_core
from deepopt.config import load_config, reset_config
from tests.cuda.config_support import graph_config


class CUDAGraphDisabledBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        load_config(graph_config(enabled=False))

    def test_disabled_model_handle_matches_eager_training_and_preserves_forward(self):
        model = nn.Linear(4, 2)
        reference = deepcopy(model)
        original = model.forward
        parameters = tuple(model.parameters())
        sample = torch.randn(3, 4)
        handle = cuda_graph_core.capture_model(model, (sample,))
        self.addCleanup(handle.close)
        self.assertEqual(model.forward, original)
        self.assertEqual(tuple(model.parameters()), parameters)
        self.assertFalse(handle.enabled)
        self.assertFalse(handle.captured)
        self.assertIsNone(handle.graph)
        self.assertIsNone(handle.capture_policy)
        actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for batch in (sample, torch.randn(5, 4)):
            actual_optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            actual = handle.replay(input=batch)
            expected = reference(batch)
            torch.testing.assert_close(actual, expected)
            actual.square().mean().backward()
            expected.square().mean().backward()
            for parameter, baseline in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, baseline.grad)
            actual_optimizer.step()
            expected_optimizer.step()
        for parameter, baseline in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, baseline)
        self.assertEqual(handle.statistics["forward_replays"], 0)
        handle.close()
        handle.close()
        self.assertEqual(model.forward, original)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            handle(sample)

    def test_disabled_callable_preserves_autograd_and_inference_modes(self):
        value = torch.tensor([2., 3.], requires_grad=True)
        handle = cuda_graph_core.capture_callable(lambda value: value.square(), (value,))
        self.addCleanup(handle.close)
        handle(value).sum().backward()
        torch.testing.assert_close(value.grad, 2 * value)
        inference = cuda_graph_core.capture_callable(lambda value: value.square(), (value,), autograd=False)
        self.addCleanup(inference.close)
        self.assertFalse(inference(value).requires_grad)
        self.assertFalse(inference.captured)

    def test_disabled_decorator_runs_python_each_time_without_warmup_and_keeps_snapshot(self):
        calls = []
        @cuda_graph_core.capture(warmup=5)
        def transform(value, *, offset=0):
            calls.append(value.shape)
            return value.square() + offset
        self.addCleanup(transform.close)
        load_config(graph_config(enabled=True))
        for size in (2, 5):
            value = torch.arange(float(size), requires_grad=True)
            result = transform(value, offset=3)
            torch.testing.assert_close(result, value.square() + 3)
            self.assertFalse(result.requires_grad)
        self.assertEqual(calls, [(2,), (5,)])
        self.assertFalse(transform.enabled)
        self.assertFalse(transform.captured)
        self.assertIsNone(transform.graph)
        transform.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            transform(torch.ones(2))


@unittest.skipUnless(torch.cuda.is_available(), "Real CUDA hardware is required")
class CUDAGlobalConfigBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        torch.manual_seed(281)

    def test_singleton_new_captures_follow_global_policy_but_existing_inference_handle_does_not(self):
        device = torch.device("cuda", torch.cuda.current_device())
        load_config(graph_config(enabled=True, device=str(device), warmup=2, autograd=False))
        sample = torch.randn(8, device=device)
        captured = cuda_graph_core.capture_callable(lambda value: value.sin() * 2, (sample,))
        try:
            self.assertFalse(captured.autograd)
            self.assertIsInstance(captured.graph, torch.cuda.CUDAGraph)
            load_config(graph_config(device=str(device), warmup=1, autograd=True))
            fresh = torch.randn_like(sample)
            torch.testing.assert_close(captured(fresh), fresh.sin() * 2)
            self.assertEqual(captured.config["runtime"]["cuda_graph"]["warmup"], 2)
            differentiable = fresh.clone().requires_grad_()
            training = cuda_graph_core.capture_callable(lambda value: value.square(), (differentiable,))
            try:
                self.assertTrue(training.autograd)
                training(differentiable).sum().backward()
                torch.testing.assert_close(differentiable.grad, 2 * differentiable)
            finally:
                training.close()
        finally:
            captured.close()

    def test_core_snapshot_keeps_autograd_model_capture_after_global_change(self):
        device = torch.device("cuda", torch.cuda.current_device())
        load_config(graph_config(enabled=True, device=str(device), warmup=2, autograd=True))
        core = CUDAGraphCore()
        load_config(graph_config(device=str(device), warmup=5, autograd=False))
        model = nn.Linear(4, 2).to(device)
        reference = deepcopy(model)
        sample = torch.randn(3, 4, device=device)
        capture = core.capture_model(model, (sample,))
        try:
            self.assertTrue(capture.autograd)
            actual = model(sample)
            expected = reference(sample)
            torch.testing.assert_close(actual, expected)
            actual.square().mean().backward()
            expected.square().mean().backward()
            for parameter, baseline in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, baseline.grad)
            torch.optim.SGD(model.parameters(), lr=0.01).step()
            torch.optim.SGD(reference.parameters(), lr=0.01).step()
            for parameter, baseline in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, baseline)
        finally:
            capture.close()


if __name__ == "__main__":
    unittest.main()
