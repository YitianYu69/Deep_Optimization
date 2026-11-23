"""Prepared CUDA projections can be captured without executing custom kernels."""

from copy import deepcopy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from deepopt import FP8Config
from deepopt.compiler.capture import prepare_capture
from deepopt.opt.nn.autograd.linear import _DOLinearFunction
from deepopt.opt.policy import fp8_plan, prepare_fp8_model


BLOCKED_EXECUTORS = {
    name: None
    for name in (
        "triton",
        "deepopt.opt.backends.triton.gemm.fp8",
        "deepopt.opt.backends.triton.quantization.fp8",
        "deepopt.opt.backends.triton.quantization.delayed",
        "deepopt.opt.backends.triton.quantization.activation.fp8",
        "deepopt.opt.backends.triton.quantization.activation.affine",
    )
}


class CaptureMetadataTests(unittest.TestCase):
    def test_cuda_projection_capture_preserves_layout_and_real_module_state(self):
        for recipe in ("current", "delayed"):
            for compressed in (False, True):
                with self.subTest(recipe=recipe, compressed=compressed):
                    model, _ = prepare_fp8_model(
                        nn.Sequential(nn.Linear(128, 65)),
                        FP8Config(enabled=True, backend="triton", scaling_recipe=recipe),
                        activation_compression={"enabled": compressed, "bits": 2, "group_size": 1024},
                    )
                    state = deepcopy(model.state_dict())
                    prepare_capture(model)
                    with (
                        FakeTensorMode(),
                        torch.no_grad(),
                        patch.dict(sys.modules, BLOCKED_EXECUTORS),
                        patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("GPU query")),
                        patch.object(torch.Tensor, "data_ptr", side_effect=AssertionError("data pointer")),
                    ):
                        parameters = tuple(
                            torch.empty(value.shape, dtype=value.dtype, device="cuda:0", requires_grad=value.requires_grad)
                            for value in model.parameters()
                        )
                        buffers = tuple(
                            torch.empty(value.shape, dtype=value.dtype, device="cuda:0")
                            for value in model.buffers()
                        )
                        x = torch.empty(2, 17, 128, device="cuda:0", dtype=torch.bfloat16)
                        actual = model[0].forward.operation((x,), {}, parameters, buffers)
                        self.assertIsInstance(actual, FakeTensor)
                        self.assertEqual(actual.shape, (2, 17, 65))
                        self.assertEqual(actual.stride(), (17 * 72, 72, 1))
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        self.assertEqual(actual.device, x.device)
                    for name, value in model.state_dict().items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
                        else:
                            self.assertEqual(value, state[name])
                    self.assertEqual(len(model[0]._fp8_weight_cache._entries), 0)

    def test_cuda_linear_forward_and_backward_metadata_need_no_executor(self):
        for compressed in (False, True):
            for blocks in (None, {"activation": (1, 32), "weight": (32, 32), "grad": (1, 32)}):
                with (
                    self.subTest(compressed=compressed, blocks=blocks),
                    FakeTensorMode(),
                    torch.no_grad(),
                    patch.dict(sys.modules, BLOCKED_EXECUTORS),
                ):
                    settings = FP8Config(enabled=True, backend="triton", **({"block_shapes": blocks} if blocks else {}))
                    plan = fp8_plan(settings, activation_compression={"enabled": compressed, "bits": 2, "group_size": 1024})
                    x = torch.empty(2, 17, 128, device="cuda:0", dtype=torch.bfloat16)
                    weight = torch.empty(65, 128, device="cuda:0")
                    bias = torch.empty(65, device="cuda:0")
                    # CPU-only PyTorch cannot attach real CUDA autograd device
                    # guards to fake leaves. Exercise both Function contracts
                    # directly, with the same input-gradient requirements.
                    ctx = SimpleNamespace(needs_input_grad=(True, True, True))
                    ctx.save_for_backward = lambda *values: setattr(ctx, "saved_tensors", values)
                    actual = _DOLinearFunction.forward(ctx, x, weight, bias, plan, torch.bfloat16, True, None)
                    gradients = _DOLinearFunction.backward(ctx, torch.empty_like(actual))
                    self.assertEqual(actual.shape, (2, 17, 65))
                    self.assertEqual(actual.stride(), (17 * 72, 72, 1))
                    for gradient, reference in zip(gradients[:3], (x, weight, bias)):
                        self.assertIsInstance(gradient, FakeTensor)
                        self.assertEqual(gradient.shape, reference.shape)
                        self.assertEqual(gradient.device, reference.device)
                    self.assertEqual(gradients[0].dtype, x.dtype)
                    self.assertEqual(gradients[1].dtype, torch.bfloat16)
                    self.assertEqual(gradients[2].dtype, bias.dtype)
                    self.assertTrue(all(value is None for value in gradients[3:]))


if __name__ == "__main__":
    unittest.main()
