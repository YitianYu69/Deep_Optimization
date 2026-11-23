"""Ownership and saved-state regressions reproduced from the v5 GPU results."""

import unittest

import torch
from torch.nn import functional as F

from deepopt.opt.nn import functional as D
from deepopt.opt.policy import conv_plan, linear_plan
from deepopt.opt.quantization.activation import (
    quantize_pack_activation, dequantize_unpack_activation,
    _quantize_pack_strided_activation, _dequantize_unpack_spatial_activation,
)


class RecoveryV6OwnershipTests(unittest.TestCase):
    def test_prepared_cnn_inplace_chain_runs_all_five_modes_under_real_scopes(self):
        import deepopt
        from deepopt.opt.policy import prepare_fp8_model
        for mode in ("torch_bf16","do_bf16","compression","fp8","fp8_compression"):
            torch.manual_seed(592)
            model=torch.nn.Sequential(torch.nn.Conv2d(3,8,3),torch.nn.BatchNorm2d(8),torch.nn.SiLU(inplace=True),
                torch.nn.Conv2d(8,8,5),torch.nn.SiLU(inplace=True),torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(1),torch.nn.Dropout(.2,inplace=True),torch.nn.Linear(8,3))
            source=torch.randn(2,3,7,7)
            fp8=mode in ("fp8","fp8_compression")
            compressed=mode in ("compression","fp8_compression")
            config=deepopt.FP8Config(enabled=fp8,backend="reference")
            if mode!="torch_bf16":model,_=prepare_fp8_model(model,config,activation_compression=compressed)
            context=(torch.autocast("cpu",dtype=torch.bfloat16) if mode=="torch_bf16" else
                deepopt.autocast("cpu",dtype=torch.bfloat16,fp8=config,activation_compression=compressed,check_finite=False))
            with context:output=model(source)
            output.float().square().mean().backward()
            self.assertEqual(tuple(output.shape),(2,3))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),mode)

    def test_fresh_restore_returns_owned_storage_without_copying(self):
        from deepopt.opt.nn.autograd._output import restore_fresh_output
        x = torch.randn(6, 8)
        pointer = x.data_ptr()
        result = restore_fresh_output(x, lambda a: a.reshape(2, 3, 8).transpose(1, 2))
        self.assertIsNone(result._base)
        self.assertEqual(result.data_ptr(), pointer)
        self.assertEqual(result.shape, (2, 8, 3))
        torch.testing.assert_close(result.stride(), (24, 1, 8))

    def test_fp8_convolution_singleton_spatial_output_allows_inplace_consumer(self):
        torch.manual_seed(492)
        raw=torch.randn(2,4,3,3)
        initial=torch.randn(8,4,3,3)
        plan=conv_plan(fp8={"enabled":True,"backend":"reference"},
                       activation_compression={"enabled":True,"bits":2,"group_size":17})
        results=[]
        for inplace in (False,True):
            x,w=raw.clone().requires_grad_(),initial.clone().requires_grad_()
            y=D.conv2d(x,w,plan=plan)
            z=F.silu(y,inplace=inplace)
            z.square().sum().backward()
            results.append((z.detach(),x.grad,w.grad))
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_pool_flatten_inplace_dropout_preserves_backward(self):
        raw=torch.randn(2,3,5,7)
        results=[]
        plan=linear_plan(False,activation_compression=True)
        for inplace in (False,True):
            x=raw.clone().requires_grad_()
            y=D.adaptive_avg_pool2d(x,(1,1),plan=plan).flatten(1)
            torch.manual_seed(903)
            z=F.dropout(y,p=.25,training=True,inplace=inplace)
            z.square().sum().backward()
            results.append((z.detach(),x.grad))
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_out_of_place_activation_can_feed_inplace_consumer(self):
        for operation in (D.relu,D.silu,D.gelu):
            x=torch.randn(2,3,7,requires_grad=True)
            y=operation(x,plan=linear_plan(False,activation_compression=True))
            F.relu(y,inplace=True).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())


class RecoveryV6SpatialCodecTests(unittest.TestCase):
    def test_coalesced_spatial_group_packing_keeps_payload_across_image_and_group_tails(self):
        from tests.opt.test_activation_kernels import ActivationKernelTests
        from tests.opt.test_recovery_v5_kernels import launch
        h=ActivationKernelTests();h.setUp()
        for shape,group,batch in (((2,8,3,5),32,4),((2,64,1,3),128,4)):
            x=torch.randn(shape)
            view=x.movedim(1,-1)
            matrix=view.bfloat16().reshape(-1,shape[1])
            for bits in (1,2,4,8):
                expected=quantize_pack_activation(matrix,group,bits=bits)
                data,step,minimum=torch.empty_like(expected.data),torch.empty_like(expected.step),torch.empty_like(expected.minimum)
                launch(h,"_quantize_pack_spatial_coalesced",(len(step)+batch-1)//batch,x,data,step,minimum,
                       matrix.shape[0],shape[1],shape[2]*shape[3],group,data.shape[1],bits,len(step),batch,
                       SOURCE_DTYPE=torch.bfloat16)
                for a,b in ((data,expected.data),(step,expected.step),(minimum,expected.minimum)):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_direct_source_matches_channel_row_payload_and_restored_memory_formats(self):
        for shape in ((2,3,7), (2,3,5,7), (2,3,3,5,7)):
            formats = [torch.contiguous_format]
            if len(shape) > 3:
                formats.append(torch.channels_last if len(shape) == 4 else torch.channels_last_3d)
            for fmt in formats:
                raw = torch.randn(shape).to(memory_format=fmt)
                for bits in (1,2,4,8):
                    packed = _quantize_pack_strided_activation(raw.movedim(1,-1),17,bits=bits,source_dtype=torch.bfloat16)
                    matrix = raw.to(torch.bfloat16).movedim(1,-1).reshape(-1,shape[1])
                    expected = quantize_pack_activation(matrix,17,bits=bits)
                    for a,b in ((packed.data,expected.data),(packed.step,expected.step),(packed.minimum,expected.minimum)):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    decoded = _dequantize_unpack_spatial_activation(packed,shape,dtype=torch.bfloat16,memory_format=fmt)
                    oracle = dequantize_unpack_activation(expected).reshape(shape[0],*shape[2:],shape[1]).movedim(-1,1)
                    torch.testing.assert_close(decoded,oracle,rtol=0,atol=0)
                    self.assertTrue(decoded.is_contiguous(memory_format=fmt))

    def test_spatial_jit_decode_and_rounded_strided_pack_match_oracles(self):
        from tests.opt.test_activation_kernels import ActivationKernelTests
        from tests.opt.test_recovery_v5_kernels import launch
        h = ActivationKernelTests(); h.setUp()
        shape=(2,3,5,7)
        raw=torch.randn(shape)
        view=raw.movedim(1,-1)
        rows,cols=2*5*7,3
        for bits in (1,2,4,8):
            expected=quantize_pack_activation(view.bfloat16().reshape(rows,cols),17,bits=bits)
            data,step,minimum=torch.empty_like(expected.data),torch.empty_like(expected.step),torch.empty_like(expected.minimum)
            launch(h,"_quantize_pack_groups",data.shape[0],view,data,step,minimum,
                   *view.stride()[:2],rows,cols,17,0,0,data.shape[1],bits,32,
                   SOURCE_SHAPE=tuple(view.shape),SOURCE_STRIDES=tuple(view.stride()),
                   SOURCE_NARROW=True,SOURCE_DTYPE=torch.bfloat16)
            for a,b in ((data,expected.data),(step,expected.step),(minimum,expected.minimum)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for fmt in (torch.contiguous_format,torch.channels_last):
                output=torch.empty(shape,dtype=torch.bfloat16,memory_format=fmt)
                launch(h,"_dequantize_unpack_spatial",2,data,step,minimum,output,raw.numel(),3,35,
                       17,data.shape[1],bits,fmt==torch.channels_last,0.,128)
                oracle=_dequantize_unpack_spatial_activation(expected,shape,dtype=torch.bfloat16,memory_format=fmt)
                torch.testing.assert_close(output,oracle,rtol=0,atol=0)


class RecoveryV6LinearTests(unittest.TestCase):
    def test_shared_gradient_statistics_match_amax_and_bias_oracles(self):
        from tests.opt.test_activation_kernels import ActivationKernelTests
        from tests.opt.test_relu_kernels import TypedPointer
        h=ActivationKernelTests();h.setUp()
        # Activation mask tests use an integer sum; gradient statistics need
        # the ordinary floating reduction from the interpreted JIT body.
        h.tl.sum=lambda value,axis:value.sum(dim=axis)
        value=torch.randn(37,65).T
        m,n=value.shape
        partial=torch.empty(2,n)
        amax=torch.empty(2,2)
        errors=torch.empty(2,2,dtype=torch.int32)
        for row in range(2):
            for col in range(2):
                h.tl.programs=(row,col,0)
                h.k["_gradient_statistics"](*(TypedPointer(x) for x in (value,partial,amax,errors)),
                    m,n,*value.stride(),64,2,True,32)
        reduced=torch.empty(1,n)
        for col in range(2):
            h.tl.programs=(0,col,0)
            h.k["_merge_gradient_bias"](*(TypedPointer(x) for x in (partial,reduced)),2,n,128,32)
        torch.testing.assert_close(amax.max(),value.abs().max(),rtol=0,atol=0)
        torch.testing.assert_close(reduced[0],value.sum(0),rtol=2e-6,atol=3e-6)
        self.assertFalse(errors.any())

    def test_shared_statistics_scratch_is_bounded_and_scope_restores(self):
        from deepopt.opt.backends._shared.gemm_policy import (gradient_statistics_geometry,linear_backward_scope,
            gradient_reduction_policy,linear_forward_backend)
        for shape in ((100864,768),(100864,3072),(6304,3072),(3,17)):
            geometry=gradient_statistics_geometry(*shape)
            self.assertIsNotNone(geometry)
            self.assertLessEqual(geometry[2],8*2**20)
        with linear_backward_scope(gradient_reduction="shared",forward_backend="native_precise"):
            self.assertEqual(gradient_reduction_policy(),"shared")
            self.assertEqual(linear_forward_backend(),"native_precise")
        self.assertEqual(gradient_reduction_policy(),"separate")
        self.assertEqual(linear_forward_backend(),"inherit")

    def test_native_preparation_keeps_eligible_operand_identity(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from deepopt.opt.backends._shared.gemm_policy import prepare_native_operands
        from deepopt.opt.quantization.tensor import QuantizedTensor
        with FakeTensorMode():
            a=QuantizedTensor(torch.empty(32,64,device="cuda",dtype=torch.float8_e4m3fn),torch.ones(1,1,device="cuda"))
            b=QuantizedTensor(torch.empty(64,48,device="cuda",dtype=torch.float8_e4m3fn).T,torch.ones(1,1,device="cuda"))
            pa,pb,info=prepare_native_operands(a,b)
            self.assertIs(pa,a)
            self.assertIsNot(pb,b)
            self.assertFalse(info["copied_a"])
            self.assertEqual(info["copied_bytes"],48*64)

    def test_layer_norm_only_producer_policy_does_not_fuse_gelu(self):
        from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
        from deepopt.opt.policy import activation_plan,normalization_plan
        x=torch.randn(3,17,dtype=torch.bfloat16,requires_grad=True)
        def plan(factory):
            return factory(fp8={"enabled":True,"backend":"reference"},
                activation_compression={"enabled":True,"group_size":32,"bits":4})
        with activation_codec_scope(producer_pack="layer_norm"):
            norm=D.layer_norm(x,(17,),plan=plan(normalization_plan),output_dtype=torch.float8_e4m3fn)
            gelu=D.gelu(x,plan=plan(activation_plan),output_dtype=torch.float8_e4m3fn)
        self.assertIsNotNone(norm.linear_packed(group_size=32,block_shape=None,bits=4))
        self.assertIsNone(gelu.linear_packed(group_size=32,block_shape=None,bits=4))


class RecoveryV6PoolingTests(unittest.TestCase):
    def test_fixed_pooling_geometry_matches_general_routing_and_gradients(self):
        from tests.opt.test_pooling_kernels import PoolingKernelTests
        from deepopt.opt.backends._shared.pooling import pooling_kernel_scope
        for compressed in (False,True):
            for encoded in (False,True):
                for layout in ("last","channels"):
                    for ceil in (False,True):
                        h=PoolingKernelTests();h.setUp()
                        try:
                            source=torch.randn(2,4,9,11,dtype=torch.bfloat16)
                            source[0,0,:3,:3]=1
                            options=dict(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=ceil,return_indices=True)
                            result=h.check("max_pool2d",source=source,options=options,compressed=compressed,encoded=encoded,layout=layout)
                            self.assertEqual(result[-1]["forward_variant"],"max_pool3x3")
                            events=[name for name,_ in h.h.events]
                            self.assertIn("_max_pool3x3_backward",events)
                            with pooling_kernel_scope("generic"):
                                baseline=h.check("max_pool2d",source=source,options=options,compressed=compressed,encoded=encoded,layout=layout)
                            torch.testing.assert_close(result[0],baseline[0],rtol=0,atol=0)
                        finally:h.doCleanups()


if __name__=='__main__':unittest.main()
