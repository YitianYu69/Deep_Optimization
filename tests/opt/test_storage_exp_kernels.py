"""Execute exp JIT bodies on CPU; target CUDA execution is separate."""
import ast
from importlib.resources import files
from types import SimpleNamespace
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from tests.opt.test_activation_codec_kernels import CodecLanguage
from tests.opt.test_relu_kernels import TypedPointer, CPUCast


class StorageExpKernelTests(unittest.TestCase):
    def test_cuda_host_selector_allocates_only_bf16_or_packed_residuals(self):
        events = []

        class Launch:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    events.append((self.name, grid, args, kwargs))
                return launch

        path = files('deepopt.opt.backends.triton').joinpath('activation/storage_exp.py')
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('_validate', 'forward', 'backward')]
        for node in functions:
            node.decorator_list = []
        from math import prod
        host = {'torch': torch, 'prod': prod, 'triton': SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
                '_storage_exp_forward': Launch('forward'), '_storage_exp_backward': Launch('backward')}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), host)
        with FakeTensorMode(), patch('torch.cuda.device', return_value=nullcontext()):
            x = torch.empty((2, 1031), device='cuda', dtype=torch.bfloat16)
            for packed in (False, True):
                for bits in (1, 2, 4, 8):
                    y, *saved = host['forward'](x, packed=packed, bits=bits, group_size=1024)
                    self.assertEqual(y.dtype, torch.bfloat16)
                    self.assertEqual([t.dtype for t in saved],
                                     [torch.int32, torch.float32, torch.float32] if packed else [torch.bfloat16])
                    dx = host['backward'](y, *saved, shape=tuple(x.shape), packed=packed, bits=bits, group_size=1024)
                    self.assertEqual(dx.dtype, torch.bfloat16)
                    self.assertEqual(events[-2][0], 'forward')
                    self.assertEqual(events[-1][0], 'backward')
                    self.assertFalse(events[-1][3]['enable_fp_fusion'])
            with self.assertRaises(ValueError):
                host['forward'](x, group_size=0)
            with self.assertRaises(ValueError):
                host['forward'](x.float())
            with self.assertRaises(ValueError):
                host['backward'](x, shape=tuple(x.shape), packed=True)

    def setUp(self):
        self.tl = CodecLanguage()
        self.k = {'tl': self.tl, 'triton': SimpleNamespace(cdiv=self.tl.cdiv),
                  'libdevice': SimpleNamespace(exp=torch.exp)}
        package = files('deepopt.opt.backends.triton')
        for path, names in (
                ('quantization/activation/affine.py', ('_affine_step', '_affine_quantize_pack_with_metadata', '_affine_quantize_pack')),
                ('activation/elementwise_backward.py', ('_coefficient',)),
                ('activation/storage_exp.py', ('_logical_address', '_storage_exp_forward', '_storage_exp_backward'))):
            tree = ast.parse(package.joinpath(path).read_text())
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            for node in functions:
                node.decorator_list = []
            module = CPUCast().visit(ast.Module(body=functions, type_ignores=[]))
            exec(compile(ast.fix_missing_locations(module), path, 'exec'), self.k)

    def launch(self, name, programs, *args):
        args = tuple(TypedPointer(t) if isinstance(t, torch.Tensor) else t for t in args)
        for pid in range(programs):
            self.tl.programs = (pid, 0, 0)
            self.k[name](*args)

    def test_pre_rounding_coefficient_codec_and_exact_input_recompute(self):
        torch.manual_seed(137)
        for source in (torch.randn(2, 3, 517).bfloat16(), torch.randn(3, 2, 37).bfloat16().transpose(0, 1),
                       torch.full((1, 1025), -1.25, dtype=torch.bfloat16), torch.tensor(0.5, dtype=torch.bfloat16)):
            for packed in (False, True):
                for bits in (1, 2, 4, 8):
                    with self.subTest(shape=source.shape, packed=packed, bits=bits):
                        n, group = source.numel(), 1024
                        groups = (n + group - 1) // group
                        output = torch.empty(source.shape, dtype=torch.bfloat16)
                        coefficient = source.float().exp()
                        q = quantize_pack_activation(coefficient.reshape(1, -1), group, bits=bits)
                        data = torch.empty_like(q.data) if packed else torch.empty_like(output)
                        steps, minima = torch.empty_like(q.step), torch.empty_like(q.minimum)
                        self.launch('_storage_exp_forward', groups, source, output, data, steps, minima,
                            n, tuple(source.shape), tuple(source.stride()), packed, bits, group)
                        torch.testing.assert_close(output, coefficient.bfloat16(), rtol=0, atol=0)
                        if packed:
                            for actual, expected in zip((data, steps, minima), (q.data, q.step, q.minimum)):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            expected_coefficient = dequantize_unpack_activation(q, dtype=torch.float32).reshape_as(source)
                        else:
                            torch.testing.assert_close(data, source, rtol=0, atol=0)
                            expected_coefficient = coefficient
                        # A strided cotangent must use its own physical strides.
                        dy = torch.randn(source.shape, dtype=torch.bfloat16)
                        if dy.ndim > 1:
                            dy = dy.transpose(0, 1).contiguous().transpose(0, 1)
                        dx = torch.empty_like(output)
                        self.launch('_storage_exp_backward', (n + 511) // 512, dy, data, steps, minima,
                            dx, n, tuple(source.shape), tuple(dy.stride()), packed, bits, group, 512)
                        torch.testing.assert_close(dx, (dy.float() * expected_coefficient).bfloat16(), rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), 'Requires actual CUDA exp kernels')
class StorageExpCudaTests(unittest.TestCase):
    def test_observed_kernels_and_no_global_fp32_coefficient_allocation(self):
        from deepopt.opt.backends.triton.activation import storage_exp
        from torch.utils._python_dispatch import TorchDispatchMode

        class Allocations(TorchDispatchMode):
            def __init__(self):
                self.events = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func._schema.name in ('aten::empty', 'aten::empty_strided', 'aten::_to_copy'):
                    if isinstance(result, torch.Tensor):
                        self.events.append((result.numel(), result.dtype))
                return result

        x = torch.randn(2, 3, 1031, device='cuda', dtype=torch.bfloat16)
        for packed in (False, True):
            with Allocations() as allocations, torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                y, *saved = storage_exp.forward(x, packed=packed, group_size=1024)
                storage_exp.backward(y, *saved, shape=tuple(x.shape), packed=packed, group_size=1024)
                torch.cuda.synchronize()
            self.assertFalse([event for event in allocations.events
                              if event[0] >= x.numel() and event[1] == torch.float32])
            kernels = [event.name for event in profile.events()
                       if event.device_type == torch.autograd.DeviceType.CUDA]
            for symbol in ('_storage_exp_forward', '_storage_exp_backward'):
                self.assertTrue(any(symbol in name for name in kernels), kernels)

    def test_direct_store_and_inline_decode_match_oracle(self):
        from deepopt.opt.backends.triton.activation import storage_exp

        for packed in (False, True):
            for bits in (1, 2, 4, 8):
                x = torch.randn(2, 3, 517, device='cuda', dtype=torch.bfloat16).transpose(0, 1)
                actual, *saved = storage_exp.forward(x, packed=packed, bits=bits, group_size=1024)
                torch.testing.assert_close(actual, x.float().exp().bfloat16(), rtol=0, atol=0)
                if packed:
                    from deepopt.opt.quantization.activation import PackedActivation
                    coefficient = dequantize_unpack_activation(PackedActivation(*saved, (1, x.numel()), 1024, bits=bits), dtype=torch.float32).reshape_as(x)
                else:
                    coefficient = x.float().exp()
                dy = torch.randn_like(actual)
                dx = storage_exp.backward(dy, *saved, shape=tuple(x.shape), packed=packed, bits=bits, group_size=1024)
                torch.testing.assert_close(dx, (dy.float() * coefficient).bfloat16(), rtol=0, atol=0)
                self.assertEqual(dx.dtype, torch.bfloat16)
