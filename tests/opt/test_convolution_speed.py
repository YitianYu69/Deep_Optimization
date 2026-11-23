"""Convolution tuning caches only launch metadata and retains original operands."""

import unittest
import torch

from deepopt.opt.backends._shared import convolution as policy
from deepopt.opt.backends.reference import convolution as oracle
from deepopt.opt.backends.triton.convolution.launch import geometry_arguments, operand_arguments, weight_splits
from deepopt.opt.nn.convolution import ConvGeometry, source_matrix
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.opt.convolution_kernel_support import Pointer, kernel_namespace


class ConvolutionSpeedTests(unittest.TestCase):
    def test_capture_uses_baseline_without_poisoning_later_warm_tuning(self):
        policy._choices.clear()
        candidates=[dict(BLOCK_M=64,BLOCK_N=n,BLOCK_K=64,num_warps=4,num_stages=3,CHANNEL_FAST=False) for n in (32,64)]
        calls=[]
        def launch(config): calls.append(config['BLOCK_N']);return 100/config['BLOCK_N']
        with policy.record_convolution_launches() as recorded:
            selected=policy.select_convolution_launch(('forward','test'),candidates,launch,capturing=False,benchmark=lambda call:call())
        self.assertEqual(selected['BLOCK_N'],64)
        self.assertEqual(recorded[0]['tile']['BLOCK_N'],64)
        with self.assertRaisesRegex(ValueError,'variant'),policy.convolution_kernel_scope('missing'):
            pass
        self.assertEqual(policy.select_convolution_launch(('forward','test'),candidates,launch,autotune=False)['BLOCK_N'],32)
        selected=policy.select_convolution_launch(('forward','cold'),candidates,launch,capturing=True,
            benchmark=lambda call: self.fail('benchmark inside capture'))
        self.assertEqual(selected['BLOCK_N'],32)
        self.assertEqual(policy.select_convolution_launch(('forward','cold'),candidates,launch,capturing=False,
            benchmark=lambda call:call())['BLOCK_N'],64)
        self.assertEqual(policy.select_convolution_launch(('forward','cold'),candidates,launch,capturing=True,
            benchmark=lambda call:self.fail('benchmark inside capture'))['BLOCK_N'],64)
        for key,value in policy._choices.items():
            self.assertFalse(any(isinstance(v,torch.Tensor) for v in value.values()))

    def test_long_dw_reductions_target_multiple_device_waves_with_bounded_scratch(self):
        self.assertEqual(weight_splits(144,9,1_000_000,144),1)
        splits=weight_splits(1,9,1_000_000,144,num_sms=132)
        self.assertGreater(splits,1)
        self.assertLessEqual(splits*1*9*144*4,32*2**20)
        for rows,cols in ((64,576),(1024,1024),(4096,4096)):
            splits=weight_splits(rows,cols,100_000,1,num_sms=132)
            if splits>1:self.assertLessEqual(splits*rows*cols*4,32*2**20)

    def test_channel_fast_forward_dx_dw_indexing_matches_quantized_oracle(self):
        torch.manual_seed(194)
        tl,k=kernel_namespace()
        for rank in (1,2,3):
            shape=(5,7,5)[-rank:]
            x=torch.randn(2,3,*shape)
            w=torch.randn(5,3,*((3,)*rank))
            geometry=ConvGeometry.from_tensors(x,w,padding=1)
            g=torch.randn(geometry.output_shape)
            qx=quantize_fp8(source_matrix(x));qw=quantize_fp8(w.reshape(5,-1));qg=quantize_fp8(source_matrix(g))
            def arguments(a,b):
                return {**geometry_arguments(geometry),**operand_arguments('A',a),**operand_arguments('B',b),
                        'INDEX64':False,'TENSORWISE':True,'BLOCK_M':64,'BLOCK_N':32,'BLOCK_K':32,'CHANNEL_FAST':True}
            for name,a,b,expected in (('_forward_kernel',qx,qw,oracle.forward(qx,qw,geometry)),
                                       ('_input_gradient_kernel',qg,qw,oracle.input_gradient(qg,qw,geometry))):
                output=torch.empty_like(expected)
                for m in range((output.shape[0]+63)//64):
                    for n in range((output.shape[1]+31)//32):
                        tl.programs=(m*((output.shape[1]+31)//32)+n,0,0)
                        extra=dict(HAS_BIAS=False,BIAS_STRIDE=1)
                        args=[a.data,b.data,a.scale_inv,b.scale_inv,output]
                        if name=='_forward_kernel':args.append(output)
                        k[name](*(Pointer(v) for v in args),**arguments(a,b),**extra)
                torch.testing.assert_close(output,expected,rtol=3e-5,atol=3e-5)
            for splits in (1,3):
                partial=torch.empty(splits,*geometry.weight_matrix_shape)
                tile_n=(geometry.weight_matrix_shape[1]+31)//32
                for split in range(splits):
                    for n in range(tile_n):
                        tl.programs=(n,split,0)
                        k['_weight_gradient_kernel'](*(Pointer(v) for v in (qg.data,qx.data,qg.scale_inv,qx.scale_inv,partial)),
                            **arguments(qg,qx),SPLIT_K=splits)
                torch.testing.assert_close(partial.sum(0),oracle.weight_gradient(qx,qg,geometry),rtol=3e-5,atol=3e-5)

    def test_preflight_dense_geometry_all_gradient_tiles_match_quantized_oracle(self):
        """Exercise the failed H100 shape without depending on cuDNN precision.

        These are the actual JIT Python bodies with bounds-checked CPU loads.
        Hardware FP8 accumulation remains the responsibility of the GPU test.
        """
        torch.manual_seed(14)
        language, kernels = kernel_namespace()
        x = torch.randn(2, 32, 9, 11)
        weight = torch.randn(64, 32, 3, 3)
        geometry = ConvGeometry.from_tensors(x, weight, padding=1)
        gradient = torch.randn(geometry.output_shape)
        qx, qw, qg = [quantize_fp8(value) for value in (
            source_matrix(x), weight.reshape(64, -1), source_matrix(gradient))]
        expected_forward = oracle.forward(qx, qw, geometry)
        expected_dx = oracle.input_gradient(qg, qw, geometry)
        expected_dw = oracle.weight_gradient(qx, qg, geometry)

        # Include the baseline, reordered reduction, wider K, and wider M.
        # All dimensions have tails in at least one of forward/dX/dW.
        for bm, bn, bk, channel_fast in (
            (64, 32, 64, False),
            (64, 64, 64, True),
            (64, 64, 128, True),
            (128, 64, 64, True),
        ):
            tile = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                        CHANNEL_FAST=channel_fast)

            def arguments(a, b):
                return {**geometry_arguments(geometry),
                        **operand_arguments("A", a), **operand_arguments("B", b),
                        "INDEX64": False, "TENSORWISE": True, **tile}

            for name, a, b, expected in (
                ("_forward_kernel", qx, qw, expected_forward),
                ("_input_gradient_kernel", qg, qw, expected_dx),
            ):
                with self.subTest(kernel=name, **tile):
                    output = torch.full_like(expected, float("nan"))
                    tiles_n = (output.shape[1] + bn - 1) // bn
                    for m in range((output.shape[0] + bm - 1) // bm):
                        for n in range(tiles_n):
                            language.programs = (m * tiles_n + n, 0, 0)
                            tensors = [a.data, b.data, a.scale_inv, b.scale_inv, output]
                            if name == "_forward_kernel":
                                tensors.append(output)
                            kernels[name](*(Pointer(value) for value in tensors),
                                          **arguments(a, b), HAS_BIAS=False, BIAS_STRIDE=1)
                    torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-5)

            for splits in (1, 3):
                with self.subTest(kernel="_weight_gradient_kernel", splits=splits, **tile):
                    partial = torch.full((splits, *geometry.weight_matrix_shape), float("nan"))
                    tiles_n = (geometry.weight_matrix_shape[1] + bn - 1) // bn
                    for split in range(splits):
                        for m in range((geometry.out_channels + bm - 1) // bm):
                            for n in range(tiles_n):
                                language.programs = (m * tiles_n + n, split, 0)
                                kernels["_weight_gradient_kernel"](
                                    *(Pointer(value) for value in (
                                        qg.data, qx.data, qg.scale_inv, qx.scale_inv, partial)),
                                    **arguments(qg, qx), SPLIT_K=splits)
                    torch.testing.assert_close(partial.sum(0), expected_dw, rtol=3e-5, atol=3e-5)


if __name__=='__main__':
    unittest.main()
