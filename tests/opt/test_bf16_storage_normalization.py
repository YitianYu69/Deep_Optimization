"""Normalization storage is independent of codecs and captured by the forward."""
from copy import deepcopy
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt._bf16_policy import bf16_policy_scope, BF16Policy
from deepopt.opt.nn.normalization.dispatch import normalization
from deepopt.opt.nn.normalization.layer_dispatch import layer_norm
from deepopt.opt.nn.normalization.geometry import NormGeometry
from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
from deepopt.opt.backends.reference import normalization as reference
from deepopt.opt.backends.reference.normalization import layer as layer_reference


class BF16StorageNormalizationTests(unittest.TestCase):
    device = "cpu"

    def test_owned_plan_resolves_without_wrapper_and_fails_before_state_effects(self):
        device = torch.device(self.device).type
        x = torch.randn(2, 4, 257, device=self.device, requires_grad=True)
        plan = {'fp8': False, 'bf16_policy': {'name': 'bf16_storage', 'version': 1},
                'activation_compression': {'enabled': False, 'group_size': 1024}}
        with torch.autocast(device, dtype=torch.bfloat16):
            y = normalization('group', x, None, None, groups=2, plan=plan)
        self.assertEqual(y.dtype, torch.bfloat16)
        y.sum().backward()
        self.assertEqual(x.grad.dtype, torch.float32)
        from deepopt import autocast

        with autocast(device, enabled=False, fp8=False):
            suspended = normalization('group', x, None, None, groups=2, plan=plan)
        self.assertEqual(suspended.dtype, torch.float32)
        with bf16_policy_scope('torch_amp', device), torch.autocast(device, dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'differs'):
                normalization('group', x, None, None, groups=2, plan=plan)
        with self.assertRaisesRegex(ValueError, 'BF16 compute'):
            normalization('group', x, None, None, groups=2, plan=plan)
        invalid = deepcopy(plan)
        invalid['fp8_training'] = {}
        running = [torch.zeros(4, device=self.device), torch.ones(4, device=self.device)]
        before = deepcopy(running)
        with torch.autocast(device, dtype=torch.bfloat16):
            with self.assertRaises(ValueError):
                normalization('batch', x.unsqueeze(-1), None, None, plan=invalid,
                    running_mean=running[0], running_var=running[1], training=True)
        for a, e in zip(running, before):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def execute(self, kind, x, weight, bias, *, compression=False, bits=2, training=True,
                running=None, info=None, explicit=None):
        plan = {'fp8': False, 'activation_compression': {'enabled': compression, 'bits': bits,
                'group_size': 1024, 'norm_xhat_recenter': True}}
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            if kind == 'layer':
                return layer_norm(x, x.shape[-2:], weight, bias, plan=plan, execution_info=info, output_dtype=explicit)
            return normalization(kind, x, weight, bias, groups=2, plan=plan, training=training,
                running_mean=None if running is None else running[0],
                running_var=None if running is None else running[1], momentum=0.1, execution_info=info)

    def test_storage_and_external_leaf_dtypes_do_not_depend_on_compression(self):
        for kind in ('group', 'batch', 'layer'):
            for dtype in (torch.bfloat16, torch.float32):
                for compressed in (False, True):
                    for bits in (1, 2, 4, 8):
                        with self.subTest(kind=kind, dtype=dtype, compressed=compressed, bits=bits):
                            x = torch.randn(2, 4, 3, 5, device=self.device, dtype=dtype, requires_grad=True)
                            affine = (3, 5) if kind == 'layer' else (4,)
                            w = torch.randn(affine, device=self.device, requires_grad=True)
                            b = torch.randn_like(w, requires_grad=True)
                            info = {}
                            saved = []
                            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                                with bf16_policy_scope('bf16_storage', x.device.type):
                                    y = self.execute(kind, x, w, b, compression=compressed, bits=bits, info=info)
                            self.assertEqual(y.dtype, torch.bfloat16)
                            self.assertEqual(info['bf16_policy'], BF16Policy('bf16_storage').as_dict())
                            # Switching policy before backward cannot change this invocation.
                            with bf16_policy_scope('torch_amp', x.device.type):
                                y.backward(torch.randn_like(y))
                            self.assertEqual(x.grad.dtype, dtype)
                            self.assertEqual(w.grad.dtype, torch.float32)
                            self.assertEqual(b.grad.dtype, torch.float32)
                            self.assertTrue(torch.isfinite(x.grad).all())
                            if compressed:
                                self.assertTrue(any(t.dtype == torch.int32 for t in saved))
                            self.assertTrue(any(t.dtype == torch.float32 and t.numel() < x.numel() for t in saved))

    def test_default_torch_amp_compression_off_remains_native(self):
        for kind in ('group', 'batch', 'layer'):
            x = torch.randn(2, 4, 3, 5, device=self.device, requires_grad=True)
            affine = (3, 5) if kind == 'layer' else (4,)
            w = torch.randn(affine, device=self.device, requires_grad=True)
            b = torch.randn_like(w, requires_grad=True)
            with bf16_policy_scope('torch_amp', x.device.type):
                actual = self.execute(kind, x, w, b)
            function = (lambda: torch.nn.functional.layer_norm(x, (3, 5), w, b) if kind == 'layer' else
                        torch.nn.functional.group_norm(x, 2, w, b) if kind == 'group' else
                        torch.nn.functional.batch_norm(x, None, None, w, b, True))
            expected = function()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, expected.dtype)

    def test_fixed_upstream_oracle_layouts_masks_and_scope_exit(self):
        for kind in ('group', 'batch', 'layer'):
            for layout in ('contiguous', 'channels_last', 'strided'):
                for needs in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
                    with self.subTest(kind=kind, layout=layout, needs=needs):
                        shape = (2, 4, 3, 5)
                        source = torch.randn(*shape[:-1], shape[-1] * (2 if layout == 'strided' else 1), device=self.device)
                        if layout == 'strided':
                            source = source[..., ::2]
                        if layout == 'channels_last':
                            source = source.contiguous(memory_format=torch.channels_last)
                        x = source.detach().requires_grad_(needs[0])
                        affine = (3, 5) if kind == 'layer' else (4,)
                        w = torch.randn(affine, device=self.device, requires_grad=needs[1])
                        b = torch.randn(affine, device=self.device, requires_grad=needs[2])
                        with bf16_policy_scope('bf16_storage', x.device.type):
                            y = self.execute(kind, x, w, b)
                        seed = torch.randn_like(y)
                        y.backward(seed)
                        compression = dict(enabled=False, bits=2, group_size=1024)
                        if kind == 'layer':
                            geometry = LayerNormGeometry(shape, (3, 5), 1e-5, 'last')
                            expected, _, state, *_ = layer_reference.forward(x.detach(), geometry,
                                weight=w.detach(), bias=b.detach(), dtype=torch.bfloat16, block_shape=None,
                                compression=compression, check_finite=False)
                            grads = layer_reference.backward(seed, w.detach(), geometry, packed=None, x=x.detach(),
                                mean=state.mean, inverse=state.inverse, need_dx=needs[0], need_dw=needs[1],
                                need_db=needs[2], input_dtype=x.dtype)
                        else:
                            geometry = NormGeometry(kind, shape, 2, 1e-5, True)
                            expected, mean, inverse, _ = reference.forward(x.detach(), w.detach(), b.detach(), geometry,
                                compression, out_dtype=torch.bfloat16)
                            grads = reference.backward(seed, w.detach(), geometry, x=x.detach(), mean=mean, inverse=inverse,
                                need_dx=needs[0], need_dw=needs[1], need_db=needs[2], input_dtype=x.dtype)
                        if kind == 'layer':
                            expected = geometry.transport.restore(expected)
                        tolerance = dict(rtol=0, atol=0) if self.device == 'cpu' else dict(rtol=3e-4, atol=3e-5)
                        torch.testing.assert_close(y, expected, **tolerance)
                        for actual, wanted, need in zip((x.grad, w.grad, b.grad), grads, needs):
                            if need:
                                torch.testing.assert_close(actual, wanted, **tolerance)
                            else:
                                self.assertIsNone(actual)

    def test_batch_running_statistics_eval_and_empty_do_not_corrupt_state(self):
        x = torch.randn(2, 4, 3, 5, device=self.device, requires_grad=True)
        w = torch.ones(4, device=self.device, requires_grad=True)
        b = torch.zeros_like(w, requires_grad=True)
        running = [torch.zeros(4, device=self.device), torch.ones(4, device=self.device)]
        expected_running = deepcopy(running)
        torch.nn.functional.batch_norm(x, *expected_running, w, b, True, 0.1)
        with bf16_policy_scope('bf16_storage', x.device.type):
            self.execute('batch', x, w, b, running=running)
        for a, e in zip(running, expected_running):
            torch.testing.assert_close(a, e)
        before = deepcopy(running)
        for training in (True, False):
            with bf16_policy_scope('bf16_storage', x.device.type):
                empty = self.execute('batch', x[:0], w, b, running=running, training=training)
            self.assertEqual(empty.dtype, torch.bfloat16)
            for a, e in zip(running, before):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
        with bf16_policy_scope('bf16_storage', x.device.type):
            evaluated = self.execute('batch', x, w, b, running=running, training=False)
        self.assertEqual(evaluated.dtype, torch.bfloat16)
        evaluated.sum().backward()
        self.assertEqual(x.grad.dtype, x.dtype)

    def test_layer_explicit_dtype_and_fake_match_actual_output(self):
        with bf16_policy_scope('bf16_storage', torch.device(self.device).type):
            x = torch.randn(2, 4, 3, 5, device=self.device)
            w, b = torch.ones(3, 5, device=self.device), torch.zeros(3, 5, device=self.device)
            self.assertEqual(self.execute('layer', x, w, b, explicit=torch.float32).dtype, torch.float32)
            with FakeTensorMode() as mode:
                fake = self.execute('layer', mode.from_tensor(x), mode.from_tensor(w), mode.from_tensor(b))
                self.assertEqual(fake.dtype, torch.bfloat16)

    def test_fake_all_normalization_families_preserve_storage_and_leaf_gradient_dtypes(self):
        for kind in ('group', 'batch', 'layer'):
            for dtype in (torch.float32, torch.bfloat16):
                for compressed in (False, True):
                    with self.subTest(kind=kind, dtype=dtype, compressed=compressed):
                        x = torch.randn(2, 4, 3, 5, device=self.device, dtype=dtype, requires_grad=True)
                        affine = (3, 5) if kind == 'layer' else (4,)
                        w = torch.ones(affine, device=self.device, requires_grad=True)
                        b = torch.zeros_like(w, requires_grad=True)
                        with bf16_policy_scope('bf16_storage', x.device.type), FakeTensorMode() as mode:
                            fx, fw, fb = (mode.from_tensor(t) for t in (x, w, b))
                            y = self.execute(kind, fx, fw, fb, compression=compressed)
                            self.assertEqual(y.dtype, torch.bfloat16)
                            y.backward(torch.ones_like(y))
                            self.assertEqual(fx.grad.dtype, dtype)
                            self.assertEqual(fw.grad.dtype, torch.float32)
                            self.assertEqual(fb.grad.dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), 'Actual BF16 normalization stores require CUDA')
class BF16StorageNormalizationCudaTests(BF16StorageNormalizationTests):
    device = 'cuda:0'

    def test_direct_bf16_stores_without_activation_sized_fp32_staging(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class Allocations(TorchDispatchMode):
            def __init__(self):
                self.events = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func._schema.name in ('aten::empty', 'aten::empty_strided', 'aten::_to_copy'):
                    if isinstance(result, torch.Tensor):
                        self.events.append((func._schema.name, result.numel(), result.dtype))
                return result

        for kind in ('group', 'batch', 'layer'):
            x = torch.randn(2, 8, 16, 16, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            affine = (16, 16) if kind == 'layer' else (8,)
            w = torch.ones(affine, device=self.device, requires_grad=True)
            b = torch.zeros_like(w, requires_grad=True)
            with bf16_policy_scope('bf16_storage', 'cuda'), Allocations() as allocations:
                y = self.execute(kind, x, w, b)
                y.backward(torch.ones_like(y))
                torch.cuda.synchronize()
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertEqual(x.grad.dtype, torch.bfloat16)
            self.assertFalse([event for event in allocations.events
                              if event[1] >= x.numel() and event[2] == torch.float32], allocations.events)
