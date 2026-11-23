"""Execute coalesced BatchNorm JIT bodies against the unchanged codec oracle."""

import unittest
from math import prod
from types import SimpleNamespace
import torch

from deepopt.opt.nn.normalization import NormGeometry
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from tests.opt import test_normalization_kernels as support


class ChannelsLastNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.executor = support.NormalizationKernelTests()
        self.executor.setUp()
        self.launch, self.reduce, self.k = self.executor.launch, self.executor.reduce, self.executor.k

    def test_dispatch_is_limited_to_two_spatial_dimensions(self):
        from deepopt.opt.backends._shared.normalization import channels_last_batch,normalization_kernel_scope
        for shape in ((2,5,7),(2,5,3,7),(2,5,2,3,7)):
            geo=SimpleNamespace(kind='batch',shape=shape,channels=5,elements=2*prod(shape[2:]))
            x=torch.randn(2,*shape[2:],5).movedim(-1,1)
            _,_,meta=self.k['input_arguments'](x)
            with normalization_kernel_scope('channels_last'):
                self.assertEqual(channels_last_batch(geo,meta,128,x.device),len(shape)==4)

    def test_coalesced_statistics_cover_channel_and_spatial_tails(self):
        x = torch.randn(2,5,3,7).bfloat16().contiguous(memory_format=torch.channels_last)
        geometry = NormGeometry('batch',tuple(x.shape))
        source, scale, meta = self.k['input_arguments'](x)
        elements, c, spatial = geometry.elements, geometry.channels, geometry.spatial
        parts = (elements+15)//16
        partial=torch.empty(3,c,parts)
        self.launch('_channels_last_stats',((c+3)//4,parts),source,scale,partial,c,spatial,elements,parts,meta,16,4)
        result=self.reduce(partial,c,parts,3,'_merge_stats')[:, :, 0]
        matrix=geometry.matrix(x.float())
        torch.testing.assert_close(result[0],torch.full((c,),float(elements)))
        torch.testing.assert_close(result[1],matrix.mean(1),rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(result[2],((matrix-matrix.mean(1)[:,None])**2).sum(1),rtol=2e-6,atol=2e-6)

    def test_coalesced_pack_recenter_sums_and_dx_match_same_codec(self):
        for bits in (1,2,4,8):
            for compressed in (False,True):
                with self.subTest(bits=bits,compressed=compressed):
                    self.check(bits,compressed)

    def check(self,bits,compressed):
        x=torch.randn(2,5,3,7).bfloat16().contiguous(memory_format=torch.channels_last)
        dy=torch.randn_like(x,dtype=torch.float32)
        geo=NormGeometry('batch',tuple(x.shape));c,s,e=geo.channels,geo.spatial,geo.elements
        matrix=geo.matrix(x.float());mu=matrix.mean(1);inv=torch.rsqrt(matrix.var(1,unbiased=False)+1e-5)
        h=(matrix-mu[:,None])*inv[:,None]
        group=17;groups=(e+group-1)//group;words=(group+(32//bits)-1)//(32//bits)
        weight=torch.randn(c);bias=torch.randn(c)
        expected=quantize_pack_activation(h,group,(1,group),bits=bits)
        data=torch.empty_like(expected.data);step=torch.empty_like(expected.step);low=torch.empty_like(expected.minimum)
        output=torch.empty_like(x,dtype=torch.float32)
        center=torch.empty(c);errors=torch.empty(2,c,groups)
        source,scale,meta=self.k['input_arguments'](x)
        self.launch('_channels_last_apply',((c+3)//4,groups),source,scale,weight,bias,mu,inv,output,
            data,step,low,center,errors,c,s,e,group,groups,words,bits,meta,output.stride(),True,True,compressed,
            compressed,32,4)
        torch.testing.assert_close(output,geo.restore(h)*weight[None,:,None,None]+bias[None,:,None,None],rtol=0,atol=0)
        if compressed:
            torch.testing.assert_close(data,expected.data,rtol=0,atol=0)
            torch.testing.assert_close(step,expected.step,rtol=0,atol=0)
            torch.testing.assert_close(low,expected.minimum,rtol=0,atol=0)
            self.launch('_merge_center',(c,1),errors,center,center,c,groups,1,True,4)
            decoded=dequantize_unpack_activation(expected, dtype=torch.float32)
            torch.testing.assert_close(center,(decoded-h).mean(1),rtol=2e-5,atol=2e-7)
            h=decoded-center[:,None]
        dy_data,dy_scale,dy_meta=self.k['input_arguments'](dy)
        parts=(e+15)//16;partial=torch.empty(2,c,parts)
        correction={'center_shift':center,'RECENTER':compressed}
        self.launch('_channels_last_sums',((c+3)//4,parts),dy_data,dy_scale,data,step,low,source,scale,
            mu,inv,partial,c,s,e,parts,compressed,group,words,bits,dy_meta,meta,True,16,4,**correction)
        sums=self.reduce(partial,c,parts,2,'_merge_sums')
        g=geo.matrix(dy)
        torch.testing.assert_close(sums[0,:,0],(g*h).sum(1),rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(sums[1,:,0],g.sum(1),rtol=2e-5,atol=2e-6)
        dx=torch.empty_like(dy)
        self.launch('_channels_last_dx',((c+3)//4,(e+15)//16),dy_data,dy_scale,data,step,low,source,scale,
            mu,inv,weight,sums,dx,c,s,e,True,compressed,group,words,bits,True,dy_meta,meta,dx.stride(),x.shape[-1],16,4,
            **correction)
        expected_dx=(g-g.mean(1)[:,None]-h*(g*h).mean(1)[:,None])*weight[:,None]*inv[:,None]
        torch.testing.assert_close(dx,geo.restore(expected_dx),rtol=2e-5,atol=2e-6)


if __name__=='__main__':
    unittest.main()
