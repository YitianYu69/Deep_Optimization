"""Real CUDA capture/replay with process defaults and immutable handle snapshots."""

from copy import deepcopy
import unittest

import torch
from torch import nn

from deepopt import CUDAGraphCore, cuda_graph_core
from deepopt.config import load_config, reset_config
from tests.cuda.test_global_config_white_box import graph_config


@unittest.skipUnless(torch.cuda.is_available(), "Real CUDA hardware is required")
class CUDAGlobalConfigBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        torch.manual_seed(281)

    def test_singleton_new_captures_follow_global_policy_but_existing_inference_handle_does_not(self):
        device = torch.device("cuda", torch.cuda.current_device())
        load_config(graph_config(device=str(device), warmup=2, autograd=False))
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
        load_config(graph_config(device=str(device), warmup=2, autograd=True))
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
