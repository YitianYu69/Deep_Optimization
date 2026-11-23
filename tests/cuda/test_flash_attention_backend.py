"""Real installed FA2 forward/backward, codec selection and changing-value replay."""

import importlib.util
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

import deepopt
from deepopt.opt.nn import attention_backend as backend
from deepopt.opt.nn.functional import do_scaled_dot_product_attention
from deepopt.opt.backends._shared.activation_codec import activation_codec_scope


_AVAILABLE = torch.cuda.is_available() and torch.version.hip is None and importlib.util.find_spec("flash_attn") is not None
# CUDA blocks must be powers of two; odd matrix dimensions still test edge masks.
_DEPTHWISE_BLOCK_SHAPES = (None, (4, 2))


@unittest.skipUnless(_AVAILABLE, "An NVIDIA CUDA GPU and installed flash-attn 2 are required")
class InstalledFlashAttentionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("timm") is not None, "timm is required for fit/graph integration")
    def test_timm_trainer_fit_eager_and_graph_updates_match(self):
        modes = [(False, False), (False, True)]
        if torch.cuda.get_device_capability()[0] >= 9:
            modes += [(True, False), (True, True)]
        for fp8, compression in modes:
            with self.subTest(fp8=fp8, compression=compression):
                self.check_timm_fit(fp8, compression)

    def check_timm_fit(self, fp8, compression):
        import timm
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(14)
        def make():
            return timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32, num_classes=3).cuda()
        initial = make().state_dict()
        initial = {name: value.detach().clone() for name, value in initial.items()}
        images, labels = torch.randn(4, 3, 32, 32), torch.tensor([0, 2, 1, 0])
        results = []
        for graph in (False, True):
            model = make()
            model.load_state_dict(initial)
            trainer = deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=.001), device="cuda", precision="bf16",
                loss_fn=torch.nn.CrossEntropyLoss(), execution_mode="max_speed",
                config={"runtime": {"attention": {"backend": "flash_attn_2"},
                         "fp8": {"enabled": fp8, "backend": "triton", "trace_mode": "modules",
                                 "backward_backend": "native_fast", "intercept_linear": False},
                         "activation_compression": {"enabled": compression, "attention_bits": 4, "group_size": 1024}}})
            routes = []
            with backend.observe_attention(lambda values, out, info: routes.append(info)):
                captured = None
                try:
                    if graph:
                        with trainer.runtime.autocast():
                            captured = deepopt.CUDAGraphCore("cuda", config=trainer.config).capture_model(
                                trainer.model, (torch.zeros_like(images[:2], device="cuda"),), warmup=3, autograd=True)
                    launches = []
                    replay = torch.cuda.CUDAGraph.replay
                    def observed_replay(handle):
                        launches.append(id(handle))
                        return replay(handle)
                    with patch.object(torch.cuda.CUDAGraph, "replay", observed_replay) if graph else nullcontext():
                        trainer.fit(DataLoader(TensorDataset(images, labels), batch_size=2),
                                    DataLoader(TensorDataset(images, labels), batch_size=2),
                                    epochs=1, progress_bar=False)
                    if captured is not None:
                        self.assertEqual(captured.statistics["forward_replays"], 2)
                        self.assertEqual(captured.statistics["eager_autocast_fallbacks"], 0)
                        self.assertEqual(captured.statistics["eager_memory_policy_fallbacks"], 0)
                        self.assertEqual(captured.statistics["eager_mode_fallbacks"], 2)
                        self.assertGreaterEqual(len(launches), 4)
                        self.assertGreaterEqual(len(set(launches)), 2)  # Forward AND backward graph objects.
                    results.append({name: p.detach().cpu().clone() for name, p in trainer.model.named_parameters()})
                finally:
                    if captured is not None:
                        captured.close()
            self.assertTrue(routes)
            self.assertTrue(all(row["backend"] == "flash_attn_2" for row in routes))
        for name in results[0]:
            torch.testing.assert_close(results[0][name], results[1][name], rtol=1e-4, atol=1e-6)

    def test_depthwise_candidate_oracle_and_changing_value_replay(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest('FP8 depthwise candidate requires Hopper or newer')
        from deepopt.opt.backends import convolution as executor
        from deepopt.opt.backends.reference import convolution as reference
        from deepopt.opt.nn.convolution import ConvGeometry, source_matrix
        from deepopt.opt.quantization.quantizer import quantize_fp8
        x = torch.randn(2, 5, 7, 9, device='cuda')
        w = torch.randn(5, 1, 3, 3, device='cuda')
        geometry = ConvGeometry.from_tensors(x, w, stride=2, padding=1, groups=5)
        for block in _DEPTHWISE_BLOCK_SHAPES:
            qx, qw, qg = [quantize_fp8(v, block_shape=block) for v in
                          (source_matrix(x), w.reshape(5, -1), source_matrix(torch.randn(geometry.output_shape, device='cuda')))]
            def step():
                return (executor.forward(qx, qw, geometry, backend='triton', kernel_variant='depthwise'),
                        executor.input_gradient(qg, qw, geometry, backend='triton', kernel_variant='depthwise'),
                        executor.weight_gradient(qx, qg, geometry, backend='triton', kernel_variant='depthwise'))
            def check(actual):
                expected = (reference.forward(qx, qw, geometry), reference.input_gradient(qg, qw, geometry),
                            reference.weight_gradient(qx, qg, geometry))
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=3e-3, atol=3e-3)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = step()
            torch.cuda.current_stream().wait_stream(stream)
            replacement = quantize_fp8(source_matrix(x * 1.2 + .3), block_shape=block)
            qx.data.copy_(replacement.data); qx.scale_inv.copy_(replacement.scale_inv)
            graph.replay()
            check(output)

    def test_measured_auto_packing_matches_separate_payload(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest('Measured auto packing targets Hopper')
        from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
        source = torch.randn(512, 197, 3, 12, 64, dtype=torch.bfloat16, device='cuda')
        for values in (source.unbind(2), tuple(x.contiguous() for x in source.unbind(2))):
            for bits in (4, 8):
                with activation_codec_scope(attention_pack='separate'):
                    expected = backend._pack_operands(values, bits, 1024)
                with activation_codec_scope(attention_pack='auto'):
                    actual = backend._pack_operands(values, bits, 1024)
                for (a, _), (e, _) in zip(actual, expected):
                    for av, ev in zip(a, e):
                        torch.testing.assert_close(av, ev, rtol=0, atol=0)

    def test_explicit_gradient_layouts_match_adapter_backward(self):
        from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
        qkv = tuple(torch.randn(2, 3, 17, 64, dtype=torch.bfloat16, device='cuda', requires_grad=True) for _ in range(3))
        dy = torch.randn_like(qkv[0])
        expected = None
        for layout in ('auto', 'bshd', 'bhsd', 'interleaved'):
            with activation_codec_scope(attention_grad_layout=layout):
                out = backend.flash_attention_2(*qkv, compression={'enabled': True, 'attention_bits': 4, 'group_size': 256})
            gradients = torch.autograd.grad(out, qkv, dy)
            if expected is not None:
                for a, e in zip(gradients, expected):
                    torch.testing.assert_close(a, e, rtol=1e-3, atol=1e-3)
            expected = gradients

    def test_dense_adapter_matches_public_library_forward_and_backward(self):
        from flash_attn import flash_attn_func
        torch.manual_seed(14)
        for requires in ((True, True, True), (True, False, False), (False, False, True)):
            for dropout in (0., .15):
                with self.subTest(requires=requires, dropout=dropout):
                    values = tuple(torch.randn(2, 3, 17, 64, device="cuda", dtype=torch.bfloat16).requires_grad_(r)
                                   for r in requires)
                    reference = tuple(x.detach().clone().requires_grad_(r) for x, r in zip(values, requires))
                    state = torch.cuda.get_rng_state()
                    expected = flash_attn_func(*(x.transpose(1, 2) for x in reference), dropout_p=dropout,
                                               deterministic=torch.are_deterministic_algorithms_enabled()).transpose(1, 2)
                    torch.cuda.set_rng_state(state)
                    actual = backend.flash_attention_2(*values, dropout_p=dropout)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    dy = torch.randn_like(actual)
                    actual.backward(dy); expected.backward(dy)
                    for value, other, required in zip(values, reference, requires):
                        if required:
                            torch.testing.assert_close(value.grad, other.grad, rtol=.015, atol=.015)

    def test_packed_backward_uses_canonical_reconstruction_and_original_forward_state(self):
        from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward
        for bits in (1, 2, 4, 8):
            qkv = [torch.randn(2, 3, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
            policy = {"enabled": True, "bits": 2, "attention_bits": bits, "group_size": 1024}
            forward = backend._kernels()[0]
            q, k, v = (x.detach().transpose(1, 2) for x in qkv)
            expected, lse, _, rng = forward(q, k, v, 0., 64 ** -.5, False, -1, -1, 0., None, False)
            decoded = [backend._restore_operand(*backend._pack_operand(x, bits, 1024)) for x in (q, k, v)]
            gradients = [torch.empty_like(x) for x in decoded]
            dy = torch.randn_like(expected)
            _wrapped_flash_attn_backward(dy, *decoded, expected, lse, *gradients, 0., 64 ** -.5,
                                         False, -1, -1, 0., None, torch.are_deterministic_algorithms_enabled(), rng_state=rng)
            actual = backend.flash_attention_2(*qkv, compression=policy)
            torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=0, atol=0)
            actual.backward(dy.transpose(1, 2))
            for value, wanted in zip(qkv, gradients):
                torch.testing.assert_close(value.grad, wanted.transpose(1, 2), rtol=.015, atol=.015)

    def test_amp_cast_and_timm_style_sdpa_routing(self):
        q = torch.randn(1, 2, 17, 64, device="cuda", requires_grad=True)
        records = []
        with backend.observe_attention(lambda values, out, info: records.append(info)), \
                deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=False, attention_backend="flash_attn_2",
                                 activation_compression={"enabled": True, "attention_bits": 4}):
            result = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        result.sum().backward()
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(records[-1]["backend"], "flash_attn_2")
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_joint_packer_has_identical_payload_for_canonical_interleaved_and_dense_views(self):
        from deepopt.opt.quantization.activation import _quantize_pack_qkv_activation, _quantize_pack_strided_activation
        source = torch.randn(2, 17, 3, 3, 64, dtype=torch.bfloat16, device="cuda")
        views = source.unbind(2)
        for values in (views, tuple(x.contiguous() for x in views),
                       tuple(x.transpose(1, 2).contiguous().transpose(1, 2) for x in views)):
            for bits in (1, 2, 4, 8):
                for group in (256, 1024):
                    actual = _quantize_pack_qkv_activation(values, group, bits=bits)
                    expected = tuple(_quantize_pack_strided_activation(x, group, bits=bits) for x in values)
                    for a, b in zip(actual, expected):
                        for x, y in zip((a.data, a.step, a.minimum), (b.data, b.step, b.minimum)):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_graph_replay_changes_inputs_and_matches_eager_gradients(self):
        qkv = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dy = torch.randn_like(qkv[0])
        policy = {"enabled": True, "attention_bits": 4, "group_size": 1024}
        def step():
            for x in qkv:
                x.grad = None
            with activation_codec_scope(attention_pack="joint"):
                y = backend.flash_attention_2(*qkv, compression=policy)
            y.backward(dy)
            return y
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = step()
        for offset in (.1, -.3):
            with torch.no_grad():
                qkv[0].add_(offset)
            graph.replay()
            actual = [x.grad.clone() for x in qkv]
            output = captured_output.clone()
            reference = [x.detach().clone().requires_grad_() for x in qkv]
            expected = backend.flash_attention_2(*reference, compression=policy)
            expected.backward(dy)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            for x, other in zip(actual, reference):
                torch.testing.assert_close(x, other.grad, rtol=.015, atol=.015)

    def test_explicit_selection_rejects_masks_instead_of_falling_back(self):
        q = torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "unmasked"):
            do_scaled_dot_product_attention(q, q, q, attn_mask=torch.ones(17, 17, device="cuda"),
                                            attention_backend="flash_attn_2")


if __name__ == "__main__":
    unittest.main()
