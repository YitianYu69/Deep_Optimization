"""Real CUDA ReLU casts, exact packed masks, launch fusion and graph replay."""

import importlib.util
import os
import unittest

import torch
from torch import nn

import deepopt
from deepopt.opt.nn.functional import do_relu
from deepopt.opt.policy import prepare_fp8_model, relu_plan
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation
from deepopt.opt.quantization.quantizer import quantize_fp8


REASON = ("NVIDIA CUDA hardware is required" if not torch.cuda.is_available() or torch.version.hip else
          "Triton is not installed" if importlib.util.find_spec("triton") is None else "")
if REASON and os.environ.get("DEEPOPT_REQUIRE_CUDA_RELU") == "1":
    raise RuntimeError(REASON)


@unittest.skipIf(REASON, REASON)
class ReLUCUDATests(unittest.TestCase):
    def settings(self, packed, block=None):
        return relu_plan(fp8={"enabled": True, "backend": "triton", "block_shapes": {"activation": block}},
                         activation_compression=packed)

    def test_all_precision_pairs_mask_formats_strides_and_scale_groups(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 casts in this suite require NVIDIA compute capability >= 9")
        for block in (None, (4, 8), (4, 64), (4, 16)):
            for source_fp8 in (False, True):
                for output_fp8 in (False, True):
                    for packed in (False, True):
                        with self.subTest(block=block, input=source_fp8, output=output_fp8, packed=packed):
                            x = torch.randn(2, 8, 3, 11, device="cuda", dtype=torch.bfloat16).transpose(2, 3).requires_grad_()
                            source = to_fp8_activation(x, layout="channels", block_shape=block) if source_fp8 else x
                            dense = source.dequantize(torch.float32) if source_fp8 else x.detach().float()
                            dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
                            with deepopt.autocast("cuda", enabled=False, check_finite=False):
                                output = do_relu(source, plan=self.settings(packed, block), output_dtype=dtype, output_layout="channels")
                            if output_fp8:
                                if source_fp8:
                                    torch.testing.assert_close(output._scale_inv, source._scale_inv, rtol=0, atol=0)
                                    expected_data = torch.where(source._data.float() < 0, 0, source._data.float())
                                else:
                                    q = quantize_fp8(dense.relu().movedim(1, -1).reshape(-1, 8).cpu(),
                                                     fp8_dtype=dtype, block_shape=block)
                                    expected_data = q.data.float().cuda()
                                    torch.testing.assert_close(output._scale_inv.cpu(), q.scale_inv, rtol=2e-7, atol=0)
                                torch.testing.assert_close(output._data.float(), expected_data, rtol=0, atol=0)
                            else:
                                torch.testing.assert_close(output, dense.relu().bfloat16(), rtol=0, atol=0)
                            gradient = torch.randn_like(x)
                            output.backward(gradient)
                            torch.testing.assert_close(x.grad, torch.where(dense <= 0, 0, gradient).bfloat16(), rtol=0, atol=0)

    def test_bf16_special_values_and_packed_threshold_backward(self):
        values = torch.tensor([-float("inf"), -0.0, 0.0, 2.0, float("inf"), float("nan")],
                              device="cuda", dtype=torch.bfloat16)
        x, native = values.clone().requires_grad_(), values.clone().requires_grad_()
        actual = do_relu(x, plan=relu_plan(fp8=False, activation_compression=True))
        expected = torch.relu(native)
        dy = torch.tensor([float("nan"), float("inf"), 3, 4, 5, 6], device="cuda", dtype=torch.bfloat16)
        actual.backward(dy)
        expected.backward(dy)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(x.grad, native.grad, rtol=0, atol=0, equal_nan=True)

    def test_profile_one_fused_forward_and_one_backward_without_mask_unpack(self):
        x = torch.randn(513, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        plan = relu_plan(fp8=False, activation_compression=True)
        gradient = torch.ones_like(x)

        def step():
            x.grad = None
            do_relu(x, plan=plan).backward(gradient)

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            step()
        names = [event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertEqual(sum("_forward" in name for name in names), 1, names)
        self.assertEqual(sum("_backward" in name for name in names), 1, names)
        self.assertFalse(any("unpack_boolean" in name or "pack_boolean" in name for name in names), names)

    def test_scale_launch_counts_small_blocks_tensorwise_and_provided(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 casts require NVIDIA compute capability >= 9")
        for shape, block, scale, expected in (((3, 33), None, None, 0),
                                               ((131, 96), (4, 8), None, 0),
                                               ((131, 33), None, None, 2),
                                               ((131, 33), None, torch.ones(1, 1, device="cuda"), 0)):
            info = {}
            with deepopt.autocast("cuda", enabled=False, check_finite=False):
                do_relu(torch.randn(shape, dtype=torch.bfloat16, device="cuda"), plan=self.settings(True, block),
                        output_dtype=torch.float8_e4m3fn, output_scale=scale, _execution_info=info)
            self.assertEqual(info["forward_launches"], 1)
            self.assertEqual(info["scale_launches"], expected)

    def test_capture_replay_and_inplace_masks(self):
        x = torch.randn(513, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.ones_like(x)
        policy = relu_plan(fp8=False, activation_compression=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                x.grad = None
                do_relu(x, plan=policy).backward(dy)
        torch.cuda.current_stream().wait_stream(stream)
        x.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = do_relu(x, plan=policy)
            output.backward(dy)
        for sign in (-1, 1, -1):
            with torch.no_grad():
                x.fill_(sign)
                x.grad.zero_()
            graph.replay()
            torch.testing.assert_close(output, x.detach().relu(), rtol=0, atol=0)
            torch.testing.assert_close(x.grad, (x > 0).bfloat16(), rtol=0, atol=0)
        leaf = torch.randn(7, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        value = leaf + 0
        expected = value.detach().clone()
        output = do_relu(value, inplace=True, plan=policy)
        self.assertIs(output, value)
        output.backward(torch.ones_like(value))
        torch.testing.assert_close(leaf.grad, (expected > 0).bfloat16(), rtol=0, atol=0)

    def test_prepared_normalization_relu_conv_pipeline(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 convolution requires NVIDIA compute capability >= 9")
        model = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 1)).cuda()
        model, _ = prepare_fp8_model(model, deepopt.FP8Config(enabled=True, backend="triton", trace_mode="modules"),
                                     activation_compression=True)
        with deepopt.autocast("cuda", dtype=torch.bfloat16, check_finite=False):
            model(torch.randn(2, 32, 5, 7, device="cuda")).sum().backward()
        info = model[1].execution_info
        self.assertEqual(info["scale_mode"], "reuse")
        self.assertEqual((info["forward_launches"], info["scale_launches"]), (1, 0))
        self.assertEqual(info["mask"], "packed")
        self.assertTrue(any(isinstance(m, nn.ReLU) for m in model.modules()))

    def test_fp8_gradient_relay_underflow_mask_and_inplace_storage(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 casts require NVIDIA compute capability >= 9")
        x = torch.randn(5, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q = to_fp8_activation(x, layout="last")
        expected_input = q.dequantize(torch.float32)
        gradient = to_fp8_activation(torch.randn_like(x), layout="last", fp8_dtype=torch.float8_e5m2)
        y = do_relu(q, inplace=True, plan=self.settings(True))
        self.assertIs(y, q)
        y.backward(gradient)
        self.assertIsInstance(x.grad, FP8Activation)
        expected = torch.where(expected_input <= 0, 0, gradient.dequantize(torch.float32)).bfloat16()
        torch.testing.assert_close(dequantize_activation(x.grad), expected, rtol=0, atol=0)
        tiny = torch.tensor([[1e-8, -1e-8, 0.0, 1.0]], device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with deepopt.autocast("cuda", enabled=False, check_finite=False):
            output = do_relu(tiny, plan=self.settings(True), output_dtype=torch.float8_e4m3fn,
                             output_scale=torch.ones(1, 1, device="cuda"))
        self.assertEqual(output._data.float()[0, 0].item(), 0)
        output.backward(torch.ones_like(tiny))
        torch.testing.assert_close(tiny.grad, torch.tensor([[1, 0, 0, 1]], device="cuda", dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
