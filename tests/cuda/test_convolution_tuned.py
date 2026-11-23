"""Hopper compilation, decoded-operand oracle and changing-value replay checks."""

from contextlib import ExitStack
import re
import unittest
from unittest.mock import patch

import torch

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class TunedConvolutionCUDATests(unittest.TestCase):
    @staticmethod
    def _choose(key, candidates, launch, **kwargs):
        if key[0] in ('forward', 'dx'):
            return next(c for c in candidates if c.get('TAP_MAJOR') and c['GROUP_M'] == 4)
        candidate = next(c for c in candidates if c.get('GROUP_M') == 4)
        return {**candidate, 'SPLIT_K':3}

    @staticmethod
    def _operands(geometry, *, scale_shift=0):
        """An exact oracle domain, independent of native FP8 accumulation noise.

        Ternary FP8 codes and power-of-two scales keep every partial sum exact
        for these short reductions. This tests indexing and scaling without
        imposing a random FP8-versus-ideal accumulation tolerance on H100.
        """
        from deepopt.opt.quantization.tensor import QuantizedTensor
        shapes = (geometry.input_matrix_shape, geometry.weight_matrix_shape, geometry.grad_matrix_shape)
        scales = (2. ** (-2 + scale_shift), 2. ** (1 + scale_shift), 2. ** (-1 + scale_shift))
        return tuple(QuantizedTensor(torch.randint(-1,2,shape,dtype=torch.int8).to(torch.float8_e4m3fn),
                                     torch.tensor([[scale]],dtype=torch.float32))
                     for shape, scale in zip(shapes,scales))

    def _check(self, outputs, cpu, geometry):
        from deepopt.opt.backends.reference import convolution as oracle
        qx, qw, qg = cpu
        self.assertLess(max(geometry.in_channels_per_group * geometry.kernel_volume,
                            geometry.out_channels_per_group * geometry.kernel_volume,
                            geometry.batch_size * geometry.output_volume), 2048)
        expected = (oracle.forward(qx,qw,geometry), oracle.input_gradient(qg,qw,geometry),
                    oracle.weight_gradient(qx,qg,geometry))
        for actual, reference in zip(outputs, expected):
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def _launch(self, operands, geometry):
        from deepopt.opt.backends.triton.convolution import forward, input_gradient, weight_gradient
        qx, qw, qg = operands
        return (forward(qx,qw,geometry,num_stages=3), input_gradient(qg,qw,geometry,num_stages=3),
                weight_gradient(qx,qg,geometry,num_stages=3))

    def test_tap_major_grouped_m_and_split_dw_match_independent_oracle(self):
        from deepopt.opt.backends._shared import convolution as policy
        from deepopt.opt.nn.convolution import ConvGeometry
        from deepopt.opt.quantization.tensor import QuantizedTensor
        from deepopt.opt.backends.triton.convolution.forward_impl import _forward_kernel
        from deepopt.opt.backends.triton.convolution.backward import _input_gradient_kernel, _weight_gradient_kernel
        torch.manual_seed(1721)
        compiled_ptx = {}
        def capture(name, original):
            def run(*args, **kwargs):
                compiled = original(*args, **kwargs)
                compiled_ptx[name] = compiled.asm['ptx']
                return compiled
            return run
        cases = (
            ((2,32,9,11),(65,32,3,3),dict(stride=2,padding=1)),
            ((2,17,11,13),(33,17,3,2),dict(stride=(2,1),padding=2,dilation=(2,1))),
        )
        for shape, weight_shape, options in cases:
            geometry = ConvGeometry.from_tensors(torch.empty(shape),torch.empty(weight_shape),**options)
            cpu = self._operands(geometry)
            gpu = tuple(QuantizedTensor(q.data.cuda(),q.scale_inv.cuda()) for q in cpu)
            with self.subTest(shape=shape), ExitStack() as stack, policy.convolution_kernel_scope('tuned'), \
                    patch.object(policy,'select_convolution_launch',side_effect=self._choose):
                for name, kernel in (('forward',_forward_kernel),('dx',_input_gradient_kernel),('dw',_weight_gradient_kernel)):
                    stack.enter_context(patch.object(kernel,'run',side_effect=capture(name,kernel.run)))
                self._check(self._launch(gpu,geometry),cpu,geometry)
        self.assertEqual(set(compiled_ptx), {'forward','dx','dw'})
        for name, ptx in compiled_ptx.items():
            with self.subTest(kernel=name):
                instructions = re.findall(r'[^\n]*(?:wgmma\.mma_async|mma\.sync|tcgen05\.mma)[^\n]*',ptx)
                self.assertTrue(any(re.search(r'e4m3|e5m2|f8f6f4', instruction) for instruction in instructions),ptx)
                self.assertFalse(any('tf32' in instruction for instruction in instructions),ptx)

    def test_changing_payload_scales_and_weights_replay_without_stale_values(self):
        from deepopt.opt.backends._shared import convolution as policy
        from deepopt.opt.nn.convolution import ConvGeometry
        from deepopt.opt.quantization.tensor import QuantizedTensor
        torch.manual_seed(1722)
        geometry = ConvGeometry.from_tensors(torch.empty(2,16,7,9),torch.empty(33,16,3,3),padding=1)
        cpu = self._operands(geometry)
        gpu = tuple(QuantizedTensor(q.data.cuda(),q.scale_inv.cuda()) for q in cpu)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with policy.convolution_kernel_scope('tuned'), \
                patch.object(policy,'select_convolution_launch',side_effect=self._choose):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._launch(gpu,geometry)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                outputs = self._launch(gpu,geometry)
            for replay in range(2):
                cpu = self._operands(geometry,scale_shift=replay + 1)
                for target, source in zip(gpu,cpu):
                    target.data.copy_(source.data)
                    target.scale_inv.copy_(source.scale_inv)
                graph.replay()
                self._check(outputs,cpu,geometry)


if __name__ == '__main__':
    unittest.main()
