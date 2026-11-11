"""Execute installed Triton bodies on CPU; not a CUDA compilation test."""

from itertools import product
import unittest

import torch

from deepopt.opt.backends.reference import convolution_transpose as oracle
from deepopt.opt.nn.convolution import source_matrix
from deepopt.opt.nn.convolution_transpose.geometry import ConvTransposeGeometry
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor
from tests.opt import test_convolution_blockwise_fp8 as support
from tests.opt.convolution_kernel_support import kernel_namespace
from tests.opt.test_convolution_transpose import CASES


class TransposedKernelBodyTests(unittest.TestCase):
    _launch = support.BlockwiseConvolutionKernelTests._launch

    def test_real_kernel_addressing_scales_bias_and_split_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(136)
        layouts = ((None, None, None), ((2, 4), None, None), (None, (2, 4), None),
                   (None, None, (2, 4)), ((4, 2), (2, 4), (2, 2)))
        for case, blocks, compressed in product(CASES, layouts, (False, True)):
            rank, shape, wshape, opts = case
            with self.subTest(rank=rank, options=opts, blocks=blocks, compressed=compressed):
                self.language, self.kernels = kernel_namespace()
                x, w = torch.randn(shape), torch.randn(wshape)
                geom = ConvTransposeGeometry.from_tensors(x, w, **opts)
                qx = quantize_fp8(source_matrix(x), block_shape=blocks[0])
                qw = quantize_fp8(w.reshape(geom.weight_matrix_shape), block_shape=blocks[1])
                qg = quantize_fp8(source_matrix(torch.randn(geom.output_shape)), block_shape=blocks[2])
                saved = qx
                if compressed:
                    packed = quantize_pack_activation(qx.data, 17, qx.block_shape, bits=2)
                    saved = QuantizedTensor(dequantize_unpack_activation(packed, dtype=qx.data.dtype), qx.scale_inv, qx.block_shape)
                # Non-contiguous data/scales must retain their physical strides.
                qx, qw, qg, saved = [QuantizedTensor(q.data.T.contiguous().T, q.scale_inv.T.contiguous().T, q.block_shape)
                                      for q in (qx, qw, qg, saved)]
                b = torch.randn(geom.out_channels * 2)[::2]
                y = self._launch("input_gradient", qx, qw, geom.adjoint, bias=b, index64=True)
                dx = self._launch("forward", qg, qw, geom.adjoint, index64=True)
                dw = self._launch("weight_gradient", saved, qg, geom.adjoint, splits=3, index64=True)
                torch.testing.assert_close(y, oracle.forward(qx, qw, geom, bias=b), rtol=3e-5, atol=3e-5)
                torch.testing.assert_close(dx, oracle.input_gradient(qg, qw, geom), rtol=3e-5, atol=3e-5)
                torch.testing.assert_close(dw, oracle.weight_gradient(saved, qg, geom), rtol=3e-5, atol=3e-5)
                self.assertGreater(self.language.dots, 0)


if __name__ == "__main__":
    unittest.main()
