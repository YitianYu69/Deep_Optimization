"""Execute the new JIT bodies against payload and gradient oracles on CPU."""

import unittest

import torch

from deepopt.opt.nn.activation.geometry import tensor_arguments
from deepopt.opt.quantization.activation import quantize_pack_activation


def launch(harness, name, programs, *args, **kwargs):
    from tests.opt.test_relu_kernels import TypedPointer
    def pointer(value):
        return TypedPointer(value) if isinstance(value, torch.Tensor) else value
    for i in range(programs):
        harness.tl.programs = (i, 0, 0)
        harness.k[name](*(pointer(v) for v in args), **{k:pointer(v) for k,v in kwargs.items()})


class RecoveryV5KernelTests(unittest.TestCase):
    def test_bounded_strided_addresses_match_wide_arithmetic(self):
        from tests.opt.test_activation_kernels import ActivationKernelTests
        h=ActivationKernelTests();h.setUp()
        shape,strides=(2,3,5,7),(315,7,63,1)
        index=torch.arange(2*3*5*7,dtype=torch.int64)
        args=(index//7,index%7,7,strides[0],strides[1],shape,strides)
        wide=h.k["_source_address"](*args)
        narrow=h.k["_source_address"](*args,SOURCE_NARROW=True)
        torch.testing.assert_close(wide,narrow.long(),rtol=0,atol=0)
        self.assertEqual(narrow.dtype,torch.int32)

    def test_norm_epilogue_packs_rounded_fp8_across_row_boundaries(self):
        from tests.opt.test_layer_norm_kernels import LayerNormKernelTests
        h = LayerNormKernelTests()
        h.setUp()
        torch.manual_seed(93)
        m,k,group = 3,17,32
        x = torch.randn(m,k)
        _,inv,meta = tensor_arguments(x)
        var,mean = torch.var_mean(x,dim=1,correction=0)
        rstd = torch.rsqrt(var+1e-5)
        weight,bias = torch.randn(k),torch.randn(k)
        expected = (((x-mean[:,None])*rstd[:,None])*weight+bias)*2
        expected = expected.clamp(-448,448).to(torch.float8_e4m3fn)
        for bits in (1,2,4,8):
            target = quantize_pack_activation(expected,group,bits=bits)
            data,step,minimum = torch.empty_like(target.data),torch.empty_like(target.step),torch.empty_like(target.minimum)
            out = torch.empty_like(expected)
            scales,oinv = torch.full((1,1),2.),torch.empty(1,1)
            observations,snapshot,errors = torch.empty(2),torch.empty(2),torch.empty(2,dtype=torch.int32)
            launch(h,"_linear_output_epilogue",2,x,inv,weight,bias,mean,rstd,out,scales,oinv,data,step,minimum,
                   observations,snapshot,errors,m,k,(m,k),(k,),meta,(1,),(1,),True,True,
                   group,target.data.shape[1],bits,448.,True,True,True,32)
            for a,b in [(out.float(),expected.float()),(data,target.data),(step,target.step),(minimum,target.minimum)]:
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertFalse(errors.any())

    def test_gelu_forward_output_pack_and_recomputed_backward(self):
        from tests.opt.test_activation_kernels import ActivationKernelTests
        h = ActivationKernelTests()
        h.setUp()
        torch.manual_seed(923)
        m,k,group = 5,33,32
        x = torch.randn(m,k,dtype=torch.bfloat16)
        _,inv,meta = tensor_arguments(x)
        for bits in (1,2,4,8):
            words = group//(32//bits)
            size = (m*k+group-1)//group
            saved = torch.empty(size,words,dtype=torch.int32)
            steps,minima = torch.empty(size),torch.empty(size)
            output = torch.empty(m,k,dtype=torch.float8_e4m3fn)
            scale,oinv = torch.full((1,1),2.),torch.empty(1,1)
            op,os,om = torch.empty_like(saved),torch.empty_like(steps),torch.empty_like(minima)
            obs,snapshot,errors = torch.empty(size),torch.empty(2),torch.empty(size,dtype=torch.int32)
            launch(h,"_activation_forward",size,x,inv,output,saved,steps,minima,scale,oinv,obs,snapshot,errors,
                   m,k,(m,k),False,meta,"gelu","none",group,0,0,words,True,True,bits,False,
                   True,False,False,m,k,1,448.,False,False,False,True,1,32,
                   OUTPUT_PACK=True,output_words=op,output_steps=os,output_minima=om,FLAT_GRID=True)
            expected = quantize_pack_activation(output,group,bits=bits)
            for a,b in [(op,expected.data),(os,expected.step),(om,expected.minimum)]:
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertFalse(errors.any())
        dy = torch.randn(m,k)
        _,_,gm = tensor_arguments(dy)
        _,derivative = h.k["_evaluate"](x.float(),"gelu","none")
        outputs = []
        for recompute,saved in [(False,derivative),(True,x)]:
            out = torch.empty_like(x)
            launch(h,"_activation_backward",2,dy,inv,saved,steps,minima,out,scale,oinv,obs,
                   m,k,(m,k),gm,False,False,m,1,k,group,0,0,words,bits,False,False,
                   False,False,1,1,False,0.,128,RECOMPUTE=recompute,KIND="gelu",APPROXIMATE="none")
            outputs.append(out)
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
