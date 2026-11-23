"""CUDA compilation and replay gates for opt-in activation speed candidates."""

import unittest

import torch

import deepopt
from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
from deepopt.opt.nn import functional as doF
from deepopt.opt.nn.activation.geometry import ReLUGeometry
from deepopt.opt.policy import activation_plan
from deepopt.opt.quantization.activation_tensor import to_fp8_activation


_AVAILABLE = torch.cuda.is_available() and torch.version.hip is None


@unittest.skipUnless(_AVAILABLE, "An NVIDIA CUDA GPU is required")
class SpeedActivationCandidateCUDATests(unittest.TestCase):
    def test_group_owned_autograd_restores_layout_policy_for_backward(self):
        raw = torch.randn(2, 23, 7, 7, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        dy = torch.randn_like(raw)
        for function in (doF.silu, doF.relu):
            results = []
            for policy in ("baseline", "channels_last"):
                x = raw.detach().clone().requires_grad_()
                with activation_codec_scope(activation_layout=policy), deepopt.autocast("cuda", enabled=False, check_finite=False):
                    y = function(x, plan=activation_plan(False,
                        activation_compression={"enabled": True, "bits": 2, "group_size": 1024}))
                y.backward(dy)
                results.append((y.detach(), x.grad))
            for a, b in zip(*results):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_small_spatial_groups_match_existing_payload_with_fp8_boundaries(self):
        from deepopt.opt.backends.triton.activation import elementwise as compute
        torch.manual_seed(14)
        raw = torch.randn(2, 23, 7, 7, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        geometry = ReLUGeometry(tuple(raw.shape))
        for source_fp8 in (False, True):
            x = to_fp8_activation(raw, layout="channels") if source_fp8 else raw
            for output_fp8 in (False, True):
                for current in ((False, True) if output_fp8 else (False,)):
                    for bits in (2, 4):
                        with self.subTest(source_fp8=source_fp8, output_fp8=output_fp8, current=current, bits=bits):
                            options = dict(kind="silu", approximate="none", dtype=torch.float8_e4m3fn if output_fp8 else raw.dtype,
                                block_shape=None, packed=True, bits=bits, group_size=1024, save_mask=True, inplace=False,
                                scale=None if current or not output_fp8 else torch.tensor([[2.0]], device="cuda"))
                            with activation_codec_scope(activation_layout="baseline"):
                                expected = compute.forward(x, geometry, **options)
                            with activation_codec_scope(activation_layout="channels_last"):
                                actual = compute.forward(x, geometry, **options)
                            self.assertEqual(actual[-1]["forward_variant"], "channels_last_group_owned")
                            dense = actual[0].float() if output_fp8 else actual[0]
                            wanted = expected[0].float() if output_fp8 else geometry.restore(expected[0])
                            torch.testing.assert_close(dense, wanted, rtol=0, atol=0)
                            for a, b in zip(actual[2].tensors, expected[2].tensors):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
                            if output_fp8:
                                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_group_owned_forward_replays_changing_values(self):
        from deepopt.opt.backends.triton.activation import elementwise as compute
        x = torch.randn(2, 23, 7, 7, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        geometry = ReLUGeometry(tuple(x.shape))
        def step(policy):
            with activation_codec_scope(activation_layout=policy):
                result = compute.forward(x, geometry, kind="silu", approximate="none", dtype=x.dtype,
                    block_shape=None, packed=True, bits=2, group_size=1024, save_mask=True, inplace=False)
            return result[0], *result[2].tensors
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step("channels_last")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = step("channels_last")
        for _ in range(2):
            x.add_(0.125)
            graph.replay()
            expected = step("baseline")
            torch.testing.assert_close(actual[0], geometry.restore(expected[0]), rtol=0, atol=0)
            for a, b in zip(actual[1:], expected[1:]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_silu_recompute_matches_derivative_saves_including_channels_last_copy(self):
        for channels_last in (False, True):
            raw = torch.randn(2, 23, 7, 7, device="cuda", dtype=torch.bfloat16)
            if channels_last:
                raw = raw.contiguous(memory_format=torch.channels_last)
            dy = torch.randn_like(raw)
            for output_dtype in (torch.bfloat16, torch.float8_e4m3fn):
                results, sizes = [], []
                for state in ("derivative", "recompute"):
                    x = raw.detach().clone().requires_grad_()
                    saved, info = [], {}
                    with activation_codec_scope(silu_state=state), deepopt.autocast("cuda", enabled=False, check_finite=False), \
                            torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                        y = doF.silu(x, plan=activation_plan(False, activation_compression=False),
                                     output_dtype=output_dtype, _execution_info=info)
                    y.backward(dy)
                    results.append((getattr(y, "_data", y).detach().float(), x.grad))
                    sizes.append(sum(t.numel() * t.element_size() for t in saved))
                    if state == "recompute":
                        self.assertEqual(info["recompute_input_copy_bytes"], raw.numel() * 2 if channels_last else 0)
                self.assertEqual(sizes[0], sizes[1] * 2)
                for a, b in zip(*results):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
