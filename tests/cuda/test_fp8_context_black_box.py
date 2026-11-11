"""Real Triton FP8 under CUDA BF16 AMP through the public combined context."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import TrainingRuntime, autocast
from deepopt.opt.nn.functional import do_linear
from deepopt.opt.policy import fp8_plan
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class FP8ContextCUDABlackBoxTests(unittest.TestCase):
    def test_runtime_modes_control_real_triton_forward_and_backward_checks(self):
        model = nn.Linear(65, 37, device="cuda")
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01), device="cuda",
                                  fp8={"enabled": True, "backend": "triton"},
                                  execution_mode="max_speed", mode_schedule={2: "analyze", 3: "max_speed"})
        for update in range(1, 4):
            inputs = torch.randn(9, 65, device="cuda", requires_grad=True) * .1
            if update == 2:
                loss = runtime.compute(inputs, lambda value: runtime.model(value).square().mean())
                runtime.backward(loss)
                runtime.step()
            else:
                with patch("torch.isfinite", side_effect=AssertionError("max_speed passive scan")):
                    loss = runtime.compute(inputs, lambda value: runtime.model(value).square().mean())
                    runtime.backward(loss)
                    runtime.step()
        self.assertEqual(runtime.mode_controller.completed_updates, 3)
        self.assertTrue(torch.isfinite(runtime.model.weight).all().item())

    def test_native_linear_in_combined_bf16_amp_fp8_context_matches_reference_forward_backward(self):
        torch.manual_seed(739)
        model = nn.Linear(65, 37, device="cuda")
        baseline = deepcopy(model)
        inputs = torch.randn(2, 17, 65, device="cuda", requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        with autocast("cuda", dtype=torch.bfloat16, fp8={"enabled": True, "backend": "triton"}):
            actual = model(inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = do_linear(reference_inputs, baseline.weight, baseline.bias, plan=fp8_plan(True))
            dense = F.linear(inputs.detach(), model.weight.detach(), model.bias.detach())
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertFalse(torch.equal(actual, dense))
        upstream = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(actual, (inputs, model.weight, model.bias), upstream)
        expected_gradients = torch.autograd.grad(expected, (reference_inputs, baseline.weight, baseline.bias), upstream)
        for value, reference in zip((actual, *actual_gradients), (expected, *expected_gradients)):
            self.assertTrue(torch.isfinite(value).all().item())
            torch.testing.assert_close(value, reference, rtol=2e-2, atol=5e-3)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
