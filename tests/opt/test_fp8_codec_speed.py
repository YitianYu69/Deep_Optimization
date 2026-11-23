"""Finite-range pack specialization preserves RNE codes, metadata and transpose."""

from types import SimpleNamespace
import unittest
import torch

from tests.opt import test_activation_codec_kernels as support
from deepopt.opt.quantization.activation import quantize_pack_activation,dequantize_unpack_activation
from deepopt.opt.backends._shared.activation_codec import activation_codec_scope,attention_gradient_plan
from unittest.mock import patch


class FP8CodecSpeedTests(unittest.TestCase):
    def setUp(self):
        self.h=support.ActivationCodecKernelTests();self.h.setUp()
        self.h.k['libdevice']=SimpleNamespace(float2int_rn=lambda x:torch.round(x).to(torch.int32))

    def test_hardware_rounding_keeps_halfway_ties_even(self):
        for bits in (1,2,4,8):
            maximum=(1<<bits)-1
            values=torch.arange(0,maximum*2+1,dtype=torch.float32)*.5
            padded=max(32//bits,1<<(len(values)-1).bit_length())
            source=torch.zeros(padded);source[:len(values)]=values
            mask=torch.arange(padded)<len(values)
            old=self.h.k['_affine_quantize_pack_with_metadata'](source,mask,torch.tensor(1.),torch.tensor(0.),bits,padded)
            new=self.h.k['_affine_quantize_pack_with_metadata'](source,mask,torch.tensor(1.),torch.tensor(0.),bits,padded,FINITE_RANGE=True)
            torch.testing.assert_close(old,new,rtol=0,atol=0)

    def test_fp8_pack_and_narrow_transpose_match_original_payload(self):
        for dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
            value=(torch.randn(37,65)*3).to(dtype)
            for bits in (1,2,4,8):
                expected=quantize_pack_activation(value,32,bits=bits)
                data,step,low=torch.empty_like(expected.data),torch.empty_like(expected.step),torch.empty_like(expected.minimum)
                self.h.launch('_quantize_pack_groups',(data.shape[0],),value,data,step,low,
                    *value.stride(),37,65,32,0,0,data.shape[1],bits,32,FP8_SOURCE=True)
                for a,b in ((data,expected.data),(step,expected.step),(low,expected.minimum)):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
                output=torch.empty(65,37,dtype=dtype)
                self.h.launch('_dequantize_unpack_transpose',((37+15)//16,(65+15)//16),
                    data,step,low,output,*output.stride(),37,65,32,0,0,data.shape[1],bits,
                    torch.finfo(dtype).max,16,16,INDEX64=False)
                wanted=dequantize_unpack_activation(expected,dtype=dtype).T
                torch.testing.assert_close(output.float(),wanted.float(),rtol=0,atol=0)

    def test_auto_gradient_layout_uses_only_measured_domain_and_preserve_is_explicit(self):
        shape=(512,197,12,64);strides=((151296,64,12608,1),)*3
        compression={'enabled':True,'bits':2,'group_size':1024}
        with patch('deepopt.opt.backends._shared.normalization.measured_norm_device',return_value=True):
            self.assertEqual(attention_gradient_plan(shape,strides,torch.bfloat16,torch.device('cuda'),compression),'bshd')
            self.assertEqual(attention_gradient_plan((2,197,12,64),strides,torch.bfloat16,torch.device('cuda'),compression),'auto')
            with activation_codec_scope(attention_grad_layout='preserve'):
                self.assertEqual(attention_gradient_plan(shape,strides,torch.bfloat16,torch.device('cuda'),compression),'auto')


if __name__=='__main__':
    unittest.main()
