"""Hardware FP8 and CUDA Graph acceptance for transposed convolutions."""

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
class ConvTransposeCUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(309)
        self.enterContext(torch.backends.cudnn.flags(deterministic=True, allow_tf32=False))

    def test_fp8_all_ranks_groups_scaling_and_compression(self):
        cases = (
            (1, (2, 16, 7), dict(kernel_size=3, stride=2, padding=1, output_padding=1, groups=2)),
            (2, (2, 16, 5, 7), dict(kernel_size=4, stride=2, padding=1)),
            (3, (1, 16, 3, 4, 5), dict(kernel_size=2, stride=2, groups=4)),
            (1, (2, 16, 6), dict(kernel_size=3, stride=1, dilation=3, output_padding=2)),
            (2, (2, 16, 4, 5), dict(kernel_size=1)),
            (2, (2, 16, 4, 5), dict(kernel_size=3, padding=1, groups=16)),
            (2, (1, 3, 3, 4), dict(kernel_size=4, stride=2, padding=1)),
        )
        for rank, shape, options in cases:
            for compressed, blocked in product((False, True), repeat=2):
                with self.subTest(rank=rank, options=options, compressed=compressed, blocked=blocked):
                    original = getattr(nn, f"ConvTranspose{rank}d")(shape[1], 16, **options)
                    settings = deepopt.FP8Config(enabled=True, backend="triton", out_dtype="fp32", trace_mode="modules",
                                                block_shapes={role: (4, 8) if blocked else None for role in ("activation", "weight", "grad")})
                    compression = {"enabled": compressed, "group_size": 17, "bits": 2}
                    reference, _ = prepare_fp8_model(deepcopy(original), replace(settings, backend="reference"), activation_compression=compression)
                    model, _ = prepare_fp8_model(original.cuda(), settings, activation_compression=compression)
                    x = torch.randn(shape, device="cuda", requires_grad=True)
                    rx = x.detach().cpu().requires_grad_()
                    with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings, check_finite=False):
                        actual = model(x)
                        expected = reference(rx)
                        g = torch.randn_like(actual)
                        gradients = torch.autograd.grad(actual, (x, *model.parameters()), g)
                        expected_gradients = torch.autograd.grad(expected, (rx, *reference.parameters()), g.cpu())
                    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)
                    for a, b in zip(gradients, expected_gradients):
                        torch.testing.assert_close(a.cpu(), b, rtol=3e-3, atol=3e-3)

    def test_graph_replays_current_training_with_optimizer(self):
        for compressed, blocked in product((False, True), repeat=2):
            with self.subTest(compressed=compressed, blocked=blocked):
                settings = deepopt.FP8Config(enabled=True, backend="triton", trace_mode="modules",
                                            block_shapes={role: (4, 8) if blocked else None for role in ("activation", "weight", "grad")})
                original = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
                compression = {"enabled": compressed, "group_size": 32, "bits": 2}
                model, _ = prepare_fp8_model(original.cuda(), settings, activation_compression=compression)
                sample = torch.randn(2, 16, 5, 7, device="cuda", requires_grad=True)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                captured = None
                try:
                    with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings, activation_compression=compression,
                                          cache_enabled=False, check_finite=False):
                        captured = deepopt.CUDAGraphCore(sample.device).capture_model(model, (sample,), warmup=3, autograd=True)
                        initial = model.weight.detach().clone()
                        for _ in range(3):
                            optimizer.zero_grad(set_to_none=True)
                            x = torch.randn_like(sample, requires_grad=True)
                            y = model(x)
                            y.float().square().mean().backward()
                            self.assertTrue(torch.isfinite(model.weight.grad).all())
                            self.assertIsNotNone(x.grad)
                            optimizer.step()
                        self.assertFalse(torch.equal(initial, model.weight))
                        self.assertEqual(captured.statistics["forward_replays"], 3)
                finally:
                    if captured is not None:
                        captured.close()

    def test_planned_fp8_forward_and_backward_epilogues(self):
        from deepopt.opt.quantization.activation_tensor import FP8Activation

        for recipe, blocked in (("current", True), ("delayed", False)):
            settings = deepopt.FP8Config(enabled=True, backend="triton", trace_mode="modules", backward_precision="auto",
                                        scaling_recipe=recipe,
                                        block_shapes={role: (1, 32) if blocked else None for role in ("activation", "weight", "grad")})
            model = nn.Sequential(nn.Conv2d(32, 32, 1), nn.GroupNorm(8, 32),
                                  nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.SiLU(), nn.Conv2d(32, 16, 1)).cuda()
            model, _ = prepare_fp8_model(model, settings, activation_compression={"enabled": True})
            observed = []
            model[2].register_forward_hook(lambda m, args, y: observed.append(isinstance(y, FP8Activation)))
            with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings, cache_enabled=False, check_finite=False):
                model(torch.randn(2, 32, 5, 7, device="cuda")).float().square().mean().backward()
            self.assertEqual(observed, [True])
            self.assertTrue(model[2].output_scaling.last_execution["fused"])


if __name__ == "__main__":
    unittest.main()
