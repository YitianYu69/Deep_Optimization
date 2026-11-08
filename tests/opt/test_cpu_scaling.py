"""CPU reference scaling must work without importing the CUDA dependency."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class CPUScalingTests(unittest.TestCase):
    def test_fake_cuda_scaling_does_not_materialize_fp32_or_contiguous_inputs(self):
        import torch
        from unittest.mock import patch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from deepopt.opt.scaling_strategy.amax import AmaxScaling

        with FakeTensorMode(), patch.dict(sys.modules, {"deepopt.opt.backends.triton_quantization": None}):
            x = torch.empty(257, 259, device="cuda", dtype=torch.bfloat16).T
            with patch.object(torch.Tensor, "float", side_effect=AssertionError("full FP32 copy")), \
                 patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("dense input copy")):
                for blocks in (None, (1, 32), (32, 32), (256, 256), (1024, 1024)):
                    result = AmaxScaling(blocks).compute(x, check_finite=False)
                    expected = ((1, 1) if blocks is None else
                                tuple((size + block - 1) // block for size, block in zip(x.shape, blocks)))
                    self.assertEqual(result.scale.shape, expected)
                    self.assertEqual(result.scale_inv.shape, expected)
                    self.assertEqual(result.scale.dtype, torch.float32)
                    self.assertEqual(result.scale.device, x.device)

    def test_cpu_fp8_scaling_and_training_do_not_require_triton(self):
        root = Path(__file__).resolve().parents[2]
        code = '''
import sys
import torch

# PyTorch may probe the optional package and catch ImportError. Force it to
# be unavailable even on hosts with Triton installed, without blocking probes.
sys.modules["triton"] = None
from deepopt.opt.scaling_strategy.amax import AmaxScaling
from deepopt.opt.do_nn.functional import do_linear

x = torch.tensor([[0., 0., 4., -8., 2.], [0., 0., 1., -2., -3.],
                  [16., -2., -6., 1., .5]])
expected = torch.tensor([[1., 448. / 8, 448. / 3], [448. / 16, 448. / 6, 448. / .5]])
result = AmaxScaling((2, 2)).compute(x)
torch.testing.assert_close(result.scale, expected)
torch.testing.assert_close(result.scale_inv, expected.reciprocal())
tensorwise = AmaxScaling().compute(x, torch.float8_e5m2)
torch.testing.assert_close(tensorwise.scale, torch.tensor([[57344. / 16]]))

view = torch.tensor([[1., 100.], [2., 3.]]).T
rowwise = AmaxScaling((1, 2)).compute(view)
torch.testing.assert_close(rowwise.scale, torch.tensor([[448. / 2], [448. / 100]]))
edge = AmaxScaling((3, 5)).compute(x)
torch.testing.assert_close(edge.scale, torch.tensor([[448. / 16]]))
zeros = AmaxScaling().compute(torch.zeros(2, 2))
torch.testing.assert_close(zeros.scale, torch.ones(1, 1))
tiny = AmaxScaling((1, 2)).compute(torch.tensor([[0., 1e-37]]))
assert torch.isfinite(tiny.scale).all()
assert torch.isfinite(tiny.scale_inv).all()
pointwise = AmaxScaling((1, 1)).compute(torch.ones(2, 2))
torch.testing.assert_close(pointwise.scale, torch.full((2, 2), 448.))

plan = {"fp8_training": {"backend": "reference", "fp8_dtypes": {
    "activation": torch.float8_e4m3fn, "weight": torch.float8_e4m3fn,
    "grad": torch.float8_e5m2}},
    "block_shapes": {"activation": (1, 32), "weight": (32, 32), "grad": (1, 32)}}
inputs = torch.randn(5, 7, requires_grad=True)
weight = torch.randn(4, 7, requires_grad=True)
do_linear(inputs, weight, plan=plan).sum().backward()
assert torch.isfinite(inputs.grad).all() and torch.isfinite(weight.grad).all()
assert sys.modules["triton"] is None
assert "deepopt.opt.backends.scaling_utils.fused_amax" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                                env={**os.environ, "PYTHONPATH": str(root / "src")},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cuda_block_restrictions_remain_enforced_before_kernel_execution(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from deepopt.opt.scaling_strategy.amax import AmaxScaling

        # Fake tensors exercise device-specific validation, not GPU execution.
        with FakeTensorMode():
            x = torch.empty(2, 2, device="cuda")
            for blocks, message in (((1, 1), "must not"), ((3, 5), "powers of two")):
                with self.subTest(blocks=blocks), self.assertRaisesRegex(ValueError, message):
                    AmaxScaling(blocks).compute(x)


if __name__ == "__main__":
    unittest.main()
