"""Real CUDA graph replay contracts for current-scaled FP8 spatial convolution."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
import unittest

import torch
from torch import nn

import deepopt
from deepopt.opt.policy import prepare_fp8_model
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class ConvolutionGraphTests(unittest.TestCase):
    def test_current_scaling_forward_backward_replays_new_inputs_and_weights(self):
        torch.manual_seed(958)
        device = torch.device("cuda", torch.cuda.current_device())
        for compressed, blocked in product((False, True), (False, True)):
            with self.subTest(compressed=compressed, blocked=blocked):
                settings = deepopt.FP8Config(enabled=True, backend="triton", trace_mode="modules",
                                            scaling_recipe="current", out_dtype="fp32",
                                            block_shapes={role: (4, 8) if blocked else None
                                                          for role in ("activation", "weight", "grad")})
                compression = {"enabled": compressed, "group_size": 32, "bits": 2}
                original = nn.Conv2d(16, 16, kernel_size=3, padding=1)
                reference, _ = prepare_fp8_model(
                    deepcopy(original), replace(settings, backend="reference"), activation_compression=compression)
                model, _ = prepare_fp8_model(original.to(device), settings, activation_compression=compression)
                sample = torch.randn(2, 16, 9, 11, device=device, requires_grad=True)
                captured = None
                try:
                    # Current scale calculation, FP8 casts, codec work, and
                    # backward are all inside the captured model graph pair.
                    with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings,
                                          activation_compression=compression,
                                          cache_enabled=False, check_finite=False):
                        captured = deepopt.CUDAGraphCore(device).capture_model(
                            model, (sample,), warmup=3, autograd=True)
                        for replay in range(2):
                            value = torch.randn_like(sample, requires_grad=True)
                            reference_value = value.detach().cpu().requires_grad_()
                            actual, expected = model(value), reference(reference_value)
                            gradient = torch.randn_like(actual)
                            actual_grads = torch.autograd.grad(actual, (value, *model.parameters()), gradient)
                            expected_grads = torch.autograd.grad(
                                expected, (reference_value, *reference.parameters()), gradient.cpu())
                            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)
                            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                                torch.testing.assert_close(actual_grad.cpu(), expected_grad, rtol=2e-3, atol=2e-3)
                            if replay == 0:
                                # Replays must cast the current parameter
                                # values rather than reuse setup's FP8 weights.
                                with torch.no_grad():
                                    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
                                        parameter.add_(0.03125)
                                        reference_parameter.add_(0.03125)
                        self.assertEqual(captured.statistics, {
                            "forward_replays": 2, "eager_mode_fallbacks": 0, "eager_autocast_fallbacks": 0,
                        })
                finally:
                    if captured is not None:
                        captured.close()


if __name__ == "__main__":
    unittest.main()
