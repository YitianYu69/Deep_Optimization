"""Real CUDA activation fusion, FP8 transport, state and graph replay acceptance."""

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import activation_plan, prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation
from deepopt.opt.backends.reference.activation.math import evaluate
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


VARIANTS = (("sigmoid", "none", nn.Sigmoid), ("silu", "none", nn.SiLU), ("gelu", "none", nn.GELU),
            ("gelu", "tanh", lambda: nn.GELU(approximate="tanh")), ("tanh", "none", nn.Tanh), ("relu6", "none", nn.ReLU6))
if FP8_CUDA_SKIP_REASON and os.getenv("DEEPOPT_REQUIRE_CUDA_ACTIVATIONS") == "1":
    raise RuntimeError(FP8_CUDA_SKIP_REASON)


def call(kind, x, approximate, **options):
    return getattr(doF, kind)(x, **({"approximate": approximate} if kind == "gelu" else {}), **options)


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class ActivationCUDATests(unittest.TestCase):
    def policy(self, packed, block=None, bits=2, group=17):
        return activation_plan(fp8={"enabled": True, "backend": "triton", "block_shapes": {"activation": block}},
                               activation_compression={"enabled": packed, "bits": bits, "group_size": group})

    def test_precision_combinations_strides_and_derivative_compression(self):
        for kind, approximate, _ in VARIANTS:
            for block in (None, (2, 8)):
                for source_fp8 in (False, True):
                    for output_fp8 in (False, True):
                        for packed in (False, True):
                            with self.subTest(kind=kind, approx=approximate, block=block, input=source_fp8, output=output_fp8, packed=packed):
                                x = torch.randn(2, 8, 3, 11, device="cuda", dtype=torch.bfloat16).transpose(2, 3).requires_grad_()
                                source = to_fp8_activation(x, layout="channels", block_shape=block) if source_fp8 else x
                                represented = source.dequantize(torch.float32) if source_fp8 else x.detach().float()
                                matrix = represented.movedim(1, -1).reshape(-1, 8)
                                y, d = evaluate(matrix.cpu(), kind, approximate)
                                if packed and kind != "relu6":
                                    q = quantize_pack_activation(d, 17, block, bits=2)
                                    d = dequantize_unpack_activation(q, dtype=torch.float32)
                                d = d.reshape(x.shape[0], *x.shape[2:], x.shape[1]).movedim(-1, 1).cuda()
                                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                                    output = call(kind, source, approximate, plan=self.policy(packed, block),
                                                  output_dtype=torch.float8_e4m3fn if output_fp8 else torch.bfloat16,
                                                  output_layout="channels")
                                expected_y = y.reshape(x.shape[0], *x.shape[2:], x.shape[1]).movedim(-1, 1).cuda()
                                torch.testing.assert_close(dequantize_activation(output).float(), expected_y,
                                                           rtol=0.15 if output_fp8 else 0.01, atol=0.02 if output_fp8 else 0.004)
                                dy = torch.randn_like(x)
                                output.backward(dy)
                                expected_dx = torch.where(d, dy.float(), 0) if kind == "relu6" else dy.float() * d
                                torch.testing.assert_close(x.grad.float(), expected_dx, rtol=0.04, atol=0.02)

    def test_all_bits_and_large_groups_match_saved_derivative_oracle(self):
        for kind, approximate, _ in VARIANTS:
            for group in (17, 512, 17003):
                for bits in (1, 2, 4, 8):
                    x = torch.randn(3, 5701, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                    with deepopt.autocast("cuda", enabled=False, check_finite=False):
                        output = call(kind, x, approximate, plan=self.policy(True, bits=bits, group=group))
                    _, d = evaluate(x.detach().float().cpu(), kind, approximate)
                    if kind != "relu6":
                        d = dequantize_unpack_activation(quantize_pack_activation(d, group, bits=bits), dtype=torch.float32)
                    output.backward(torch.ones_like(x))
                    torch.testing.assert_close(x.grad.float().cpu(), d.float(), rtol=0.03, atol=0.02)

    def test_bf16_compression_has_one_forward_and_one_backward_kernel(self):
        for kind, approximate, _ in VARIANTS:
            x = torch.randn(513, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            dy = torch.ones_like(x)
            info = {}
            def step():
                x.grad = None
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    call(kind, x, approximate, plan=self.policy(True, group=512), _execution_info=info).backward(dy)
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                step()
            names = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            self.assertEqual(sum("_activation_forward" in n for n in names), 1, names)
            self.assertEqual(sum("_activation_backward" in n for n in names), 1, names)
            self.assertFalse(any("_dequantize_unpack" in n or "_quantize_pack_groups" in n for n in names), names)
            self.assertEqual((info["forward_launches"], info["coefficient_launches"]), (1, 0))

    def test_prepared_linear_and_normalization_conv_pipelines_use_fused_fp8_dx(self):
        for _, _, module in VARIANTS:
            for spatial in (False, True):
                source = (nn.Sequential(nn.Conv2d(32, 32, 1), nn.GroupNorm(4, 32), module(), nn.Conv2d(32, 32, 1))
                          if spatial else nn.Sequential(nn.Linear(32, 32), module(), nn.Linear(32, 32)))
                cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", grad_dtype="e5m2",
                                       block_shapes={"activation": None, "weight": None, "grad": (1, 32)})
                model, _ = prepare_fp8_model(source.cuda(), cfg, activation_compression=True)
                x = torch.randn((2, 32, 3, 5) if spatial else (16, 32), device="cuda", requires_grad=True)
                with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=cfg, activation_compression=True, check_finite=False):
                    output = model(x)
                with patch.object(FP8Activation, "dequantize", side_effect=AssertionError("dense FP8 reconstruction")):
                    output.sum().backward()
                reports = precision_report(model)["backward_execution"]
                self.assertTrue(all(info["fused"] for info in reports.values() if info is not None), reports)
                self.assertEqual(precision_report(model)["analyses"], 1)

    def test_current_and_fixed_scale_forward_backward_cuda_graph_replay(self):
        for kind, approximate, _ in VARIANTS:
            for provided in (False, True):
                x = torch.randn(3, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                dy = torch.ones_like(x)
                scale = torch.ones(1, 1, device="cuda") if provided else None
                policy = self.policy(True, group=512)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                def step():
                    with deepopt.autocast("cuda", enabled=False, check_finite=False):
                        out = call(kind, x, approximate, plan=policy, output_dtype=torch.float8_e4m3fn, output_scale=scale)
                    out.backward(dy)
                    return out
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        x.grad = None
                        step()
                torch.cuda.current_stream().wait_stream(stream)
                x.grad = None
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = step()
                for value in (-2.0, 0.5, 8.0):
                    with torch.no_grad():
                        x.fill_(value)
                        x.grad.zero_()
                    graph.replay()
                    expected_y, expected_d = evaluate(x.detach().float(), kind, approximate)
                    torch.testing.assert_close(dequantize_activation(captured).float(), expected_y, rtol=0.15, atol=0.03)
                    torch.testing.assert_close(x.grad.float(), expected_d.float(), rtol=0.03, atol=0.01)

    def test_delayed_updates_and_recomputation_freeze_scales_until_step(self):
        for _, _, module in VARIANTS:
            source = nn.Sequential(nn.Linear(32, 32), module(), nn.Linear(32, 32)).cuda()
            cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", scaling_recipe="delayed", grad_dtype="e5m2")
            optimizer = torch.optim.SGD(source.parameters(), lr=0.001)
            runtime = deepopt.TrainingRuntime(source, optimizer, device="cuda", precision="bf16", fp8=cfg, checks=False)
            runtime.zero_grad()
            for _ in range(2):
                loss = runtime.compute(torch.randn(16, 32, device="cuda", requires_grad=True),
                                       lambda x: deepopt.fp8_checkpoint(runtime.model, x).square().mean())
                runtime.backward(loss)
                self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 0)
            runtime.step()
            self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 1)


if __name__ == "__main__":
    unittest.main()
