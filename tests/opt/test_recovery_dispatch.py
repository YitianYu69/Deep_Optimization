"""Measured packing policy and explicit mixed-convolution contracts on CPU."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from deepopt import FP8Config
from deepopt.config import ConfigError, fp8_config_from_dict
from deepopt.opt.backends._shared.activation_codec import (
    activation_codec_scope, attention_pack_dispatch, attention_pack_plan,
    activation_pack_warps, strided_indexing,
    attention_gradient_layout,
)
from deepopt.opt.nn.functional import do_conv2d
from deepopt.opt.policy import conv_plan, linear_plan


class RecoveryDispatchTests(unittest.TestCase):
    def test_auto_packing_keeps_dense_separate_and_scopes_launch_settings(self):
        shape = (512, 197, 12, 64)
        with patch('deepopt.opt.backends._shared.normalization.measured_norm_device', return_value=True):
            with activation_codec_scope(attention_pack="auto"):
                for strides, implementation, warps in [((453888, 2304, 64, 1), "joint", 4),
                                                       ((151296, 64, 12608, 1), "joint", 4),
                                                       ((151296, 768, 64, 1), "separate", 2)]:
                    selection = attention_pack_plan(shape, strides, torch.bfloat16, 4, 1024, torch.device('cuda'))
                    self.assertEqual(selection['implementation'], implementation)
                    before = (activation_pack_warps(), strided_indexing())
                    with attention_pack_dispatch(selection):
                        self.assertEqual((activation_pack_warps(), strided_indexing()), (warps, 'bounded'))
                    self.assertEqual((activation_pack_warps(), strided_indexing()), before)
                selection = attention_pack_plan(shape, (453888, 2304, 64, 1), torch.bfloat16, 4, 512, torch.device('cuda'))
                self.assertEqual(selection['implementation'], 'separate')
                self.assertIsNone(selection['indexing'])

    def test_forced_packing_and_unmeasured_devices_preserve_controls(self):
        for mode in ('separate', 'joint'):
            with activation_codec_scope(attention_pack=mode, num_warps=8, strided_index='wide'):
                s = attention_pack_plan((512, 197, 12, 64), (453888, 2304, 64, 1),
                                        torch.bfloat16, 4, 1024, torch.device('cuda'))
                self.assertEqual(s['implementation'], mode)
                with attention_pack_dispatch(s):
                    self.assertEqual((activation_pack_warps(), strided_indexing()), (8, 'wide'))
        with patch('deepopt.opt.backends._shared.normalization.measured_norm_device', return_value=False):
            s = attention_pack_plan((512, 197, 12, 64), (453888, 2304, 64, 1),
                                    torch.bfloat16, 4, 1024, torch.device('cuda'))
            self.assertEqual(s['implementation'], 'separate')

    def test_mixed_policy_roundtrips_and_rejects_unknown_values(self):
        value = FP8Config(enabled=True, convolution_policy='native_spatial')
        self.assertEqual(fp8_config_from_dict(asdict(value)), value)
        self.assertEqual(conv_plan(value)['fp8_training']['convolution_policy'], 'native_spatial')
        with self.assertRaises(ConfigError):
            fp8_config_from_dict({'convolution_policy': 'unknown'})

    def test_mixed_spatial_matches_native_with_compression_on_and_off(self):
        torch.manual_seed(17)
        for compression in (False, True):
            for groups in (1, 4):
                with self.subTest(compression=compression, groups=groups):
                    x = torch.randn(2, 4, 7, 9, requires_grad=True)
                    w = torch.randn(4, 4 // groups, 3, 3, requires_grad=True)
                    b = torch.randn(4, requires_grad=True)
                    policy = {'enabled': compression, 'bits': 2, 'group_size': 17}
                    mixed = conv_plan(FP8Config(enabled=True, convolution_policy='native_spatial'),
                                      activation_compression=policy)
                    control = linear_plan(False, activation_compression=policy)
                    info = {}
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        actual = do_conv2d(x, w, b, padding=1, groups=groups, plan=mixed, _execution_info=info)
                        expected = do_conv2d(x, w, b, padding=1, groups=groups, plan=control)
                    dy = torch.randn_like(actual)
                    a = torch.autograd.grad(actual, (x, w, b), dy)
                    e = torch.autograd.grad(expected, (x, w, b), dy)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for av, ev in zip(a, e):
                        torch.testing.assert_close(av, ev, rtol=0, atol=0)
                    self.assertEqual(info['compute_precision'], 'native')
                    self.assertTrue(info['fp8_requested'])
                    self.assertEqual(info['compression'], compression)

    def test_mixed_pointwise_retains_fp8_dispatch(self):
        from deepopt.opt.nn.convolution import ConvGeometry
        from deepopt.opt.nn.convolution.dispatch import convolution_execution
        geometry = ConvGeometry.from_tensors(torch.empty(2, 4, 7, 9), torch.empty(8, 4, 1, 1))
        info = convolution_execution(geometry, conv_plan(FP8Config(enabled=True, backend='triton', convolution_policy='native_spatial')))
        self.assertTrue(info['pointwise'])
        self.assertEqual(info['compute_precision'], 'fp8')

    def test_gradient_destinations_and_shared_storage(self):
        from deepopt.opt.nn.attention_backend import _gradient_buffers
        values = tuple(torch.randn(2, 5, 3, 8) for _ in range(3))
        for layout in ('auto', 'bshd', 'bhsd', 'interleaved'):
            gradients = _gradient_buffers(values, tuple(x.stride() for x in values), layout)
            for x, g in zip(values, gradients):
                self.assertEqual(g.shape, x.shape)
                self.assertEqual(g.stride(-1), 1)
            if layout == 'interleaved':
                self.assertEqual(len({g.untyped_storage()._cdata for g in gradients}), 1)
                self.assertEqual(gradients[0].untyped_storage().nbytes(), sum(x.numel() * x.element_size() for x in values))

    def test_invalid_scope_and_backward_lifetime_do_not_leak_gradient_policy(self):
        before = attention_gradient_layout()
        with self.assertRaises(ValueError), activation_codec_scope(attention_grad_layout='interleaved', pointwise_tile=3):
            pass
        self.assertEqual(attention_gradient_layout(), before)
        with activation_codec_scope(attention_grad_layout='bhsd'):
            self.assertEqual(attention_gradient_layout(), 'bhsd')
        self.assertEqual(attention_gradient_layout(), before)
        from deepopt.opt.nn import attention_backend as backend
        from tests.opt.test_flash_attention_backend import fake_forward, fake_backward
        qkv = tuple(torch.randn(2, 3, 5, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        with patch.object(backend, '_validate_and_cast', side_effect=lambda q, k, v, *a: (q, k, v)), \
                patch.object(backend, '_kernels', return_value=(fake_forward, fake_backward)), \
                patch.object(backend, '_gradient_buffers', wraps=backend._gradient_buffers) as buffers:
            with activation_codec_scope(attention_grad_layout='interleaved'):
                output = backend.flash_attention_2(*qkv, compression={'enabled': True, 'attention_bits': 4, 'group_size': 17})
            output.float().square().sum().backward()
            self.assertEqual(buffers.call_args.args[2], 'interleaved')
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in qkv))

    def test_mixed_policy_preserves_output_dtype_and_master_gradients_without_amp(self):
        x = torch.randn(2, 3, 7, 9, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(4, 3, 3, 3, requires_grad=True)
        p = conv_plan(FP8Config(enabled=True, convolution_policy='native_spatial', out_dtype='fp32'))
        output = do_conv2d(x, w, padding=1, plan=p)
        self.assertEqual(output.dtype, torch.float32)
        output.square().mean().backward()
        self.assertEqual(w.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all())

    def test_prepared_mixed_model_keeps_parameter_identity_and_compression(self):
        from torch import nn
        from deepopt.opt.policy import prepare_fp8_model
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(4, 2))
        original = list(model.parameters())
        settings = FP8Config(enabled=True, convolution_policy='native_spatial', intercept_linear=False)
        model, _ = prepare_fp8_model(model, settings, activation_compression={'enabled': True, 'bits': 2, 'group_size': 17})
        self.assertEqual(list(model.parameters()), original)
        model(torch.randn(2, 3, 7, 9)).square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in original))
        info = [m.execution_info for m in model.modules() if hasattr(m, 'execution_info')]
        self.assertTrue(any(v.get('fp8_requested') and v.get('compute_precision') == 'native' and v.get('compression') for v in info))


if __name__ == '__main__':
    unittest.main()
