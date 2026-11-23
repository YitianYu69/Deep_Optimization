"""Coalesced activation traversal keeps flat codec groups across channel tails."""

from math import gcd, prod
import ast
from importlib.resources import files
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from deepopt.opt.backends._shared.activation_codec import activation_codec_scope, channels_last_activation_plan
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.boolean import pack_boolean_mask
from tests.opt import test_activation_kernels as support


class ChannelsLastActivationTests(unittest.TestCase):
    def setUp(self):
        self.executor=support.ActivationKernelTests();self.executor.setUp()
        self.k,self.launch=self.executor.k,self.executor.launch

    def test_codec_groups_are_identical_with_misaligned_channel_boundaries(self):
        for kind in ('silu','gelu','tanh','sigmoid','relu'):
            for packed in (False,True):
                for bits in ((1,) if kind=='relu' else (1,2,4,8)):
                    with self.subTest(kind=kind,packed=packed,bits=bits):
                        self.check(kind,packed,bits,(1,5,4,12),32)
        self.check('relu',True,1,(1,5,8,12),128)
        self.check('silu',True,2,(2,7,4,12),32)

    def check(self,kind,packed,bits,shape,group):
        x=torch.randn(shape,dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        output=torch.full_like(x,float('nan'))
        total=x.numel();c=shape[1];s=prod(shape[2:]);groups=(total+group-1)//group
        period=group//gcd(s,group);programs=(c+period*4-1)//(period*4)
        value=x.float().contiguous().flatten()
        if kind=='relu':
            y=torch.where(value<0,0.,value);derivative=(~(value<=0)).float()
            saved=(torch.full(((total+31)//32,),-1,dtype=torch.int32) if packed else torch.empty(total,dtype=torch.bool))
            expected=pack_boolean_mask(derivative.bool()).data if packed else derivative.bool()
            step=low=torch.empty(1);words=group//32
        else:
            y,derivative=self.k['_evaluate'](value,kind,'none')
            reference=quantize_pack_activation(derivative.reshape(-1,shape[-1]),group,bits=bits)
            if packed:
                saved=torch.empty_like(reference.data);step=torch.empty_like(reference.step);low=torch.empty_like(reference.minimum)
            else:
                saved=torch.empty(total);step=low=torch.empty(1)
            words=(group+32//bits-1)//(32//bits)
        errors=torch.full((groups,),-1,dtype=torch.int32)
        self.launch('_channels_last_activation',(shape[0]*period*programs,(s+group-1)//group),
                    x,output,saved,step,low,errors,shape[0],c,s,group,groups,words,period,programs,
                    kind,'none',True,packed,bits,kind!='relu',group,4)
        torch.testing.assert_close(output,y.reshape(shape).to(output.dtype),rtol=0,atol=0)
        if kind=='relu':
            torch.testing.assert_close(saved,expected,rtol=0,atol=0)
            coefficient=derivative
        elif packed:
            torch.testing.assert_close(saved,reference.data,rtol=0,atol=0)
            torch.testing.assert_close(step,reference.step,rtol=0,atol=0)
            torch.testing.assert_close(low,reference.minimum,rtol=0,atol=0)
            self.assertFalse(errors.any())
            coefficient=dequantize_unpack_activation(reference,dtype=torch.float32).flatten()
        else:
            torch.testing.assert_close(saved,derivative,rtol=0,atol=0);coefficient=derivative
        dy=torch.randn_like(x);dx=torch.full_like(x,float('nan'))
        self.launch('_channels_last_activation_dx',(total+63)//64,dy,saved,step,low,dx,c,s,total,shape[-1],
                    group,words,bits,kind=='relu',packed,64)
        torch.testing.assert_close(dx,(dy.float().contiguous().flatten()*coefficient).reshape(shape).to(dx.dtype),rtol=0,atol=0)

    def test_dispatch_selects_bounded_group_ownership_for_small_spatial_maps(self):
        shape=(1,5,4,12);strides=(240,1,60,5)
        with activation_codec_scope(activation_layout='channels_last'):
            self.assertEqual(channels_last_activation_plan(shape,strides,32,torch.device('cpu'))['period'],2)
            selection=channels_last_activation_plan(shape,strides,1024,torch.device('cpu'))
            self.assertEqual(selection['variant'],'group_owned')
            shape49=(2,23,7,7);strides49=(23*49,1,7*23,23)
            selection=channels_last_activation_plan(shape49,strides49,1024,torch.device('cpu'))
            self.assertEqual((selection['spatial_tile'],selection['channel_tile']),(64,32))
            self.assertIsNone(channels_last_activation_plan((1,5,7,12),(420,1,60,5),1024,torch.device('cpu')))
        with activation_codec_scope(activation_layout='baseline'):
            self.assertIsNone(channels_last_activation_plan(shape,strides,32,torch.device('cpu')))
        self.assertIsNone(channels_last_activation_plan(shape49,strides49,1024,torch.device('cpu')))

    def test_group_owned_codec_preserves_order_across_channel_batch_and_word_tails(self):
        from deepopt.opt.quantization.activation_tensor import to_fp8_activation
        for kind in ('silu','gelu','sigmoid','tanh'):
            for bits in (1,2,4,8):
                for input_fp8,output_fp8 in ((False,False),(False,True),(True,False),(True,True)):
                    with self.subTest(kind=kind,bits=bits,input_fp8=input_fp8,output_fp8=output_fp8):
                        shape=(2,23,7,7);group=1024
                        x=torch.randn(shape,dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
                        q=to_fp8_activation(x,layout='channels') if input_fp8 else None
                        values=q.dequantize(torch.float32) if q is not None else x.float()
                        y,derivative=self.k['_evaluate'](values.contiguous().flatten(),kind,'none')
                        expected=quantize_pack_activation(derivative.reshape(-1,7),group,bits=bits)
                        data=torch.empty_like(expected.data);steps=torch.empty_like(expected.step);minimum=torch.empty_like(expected.minimum)
                        output=(torch.empty((x.numel()//7,7),dtype=torch.float8_e4m3fn) if output_fp8 else torch.empty_like(x))
                        scale=torch.tensor([[2.0]]);out_inverse=torch.empty_like(scale)
                        errors=torch.full((len(steps),),-1,dtype=torch.int32)
                        self.launch('_channels_last_group_activation',len(steps),q._data if q is not None else x,
                            output,data,steps,minimum,errors,23,49,x.numel(),group,expected.data.shape[1],kind,'none',
                            True,True,bits,True,64,32,group,q._scale_inv if q is not None else scale,scale,
                            out_inverse,input_fp8,output_fp8,448.0)
                        oracle=((y*2).clamp(-448,448).to(output.dtype).reshape(output.shape) if output_fp8
                                else y.reshape(shape).to(output.dtype))
                        torch.testing.assert_close(output.float(),oracle.float(),rtol=0,atol=0)
                        for actual,wanted in ((data,expected.data),(steps,expected.step),(minimum,expected.minimum)):
                            torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
                        self.assertFalse(errors.any())
                        if output_fp8:
                            torch.testing.assert_close(out_inverse,scale.reciprocal(),rtol=0,atol=0)

    def test_group_owned_relu_preserves_boolean_payload_and_final_word_tail(self):
        shape=(2,23,7,7);group=1024
        for packed in (False,True):
            x=torch.randn(shape,dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
            output=torch.empty_like(x)
            saved=(torch.empty((x.numel()+31)//32,dtype=torch.int32) if packed
                   else torch.empty(x.numel(),dtype=torch.bool))
            self.launch('_channels_last_group_activation',(x.numel()+group-1)//group,
                x,output,saved,output,output,output,23,49,x.numel(),group,32,'relu','none',
                True,packed,1,False,64,32,group)
            expected=~(x.contiguous().flatten()<=0)
            torch.testing.assert_close(output,torch.relu(x),rtol=0,atol=0)
            torch.testing.assert_close(saved,pack_boolean_mask(expected).data if packed else expected,rtol=0,atol=0)

    def test_actual_host_launcher_passes_saved_word_geometry_and_returns_channel_storage(self):
        from deepopt.opt.nn.activation.geometry import ReLUGeometry,tensor_arguments
        from deepopt.opt.backends.reference.activation.elementwise import allocate_saved
        from deepopt.opt.nn.activation.state import SavedActivation
        source=files('deepopt.opt.backends.triton').joinpath('activation/elementwise.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_forward')
        host={'__package__':'deepopt.opt.backends.triton.activation','torch':torch,'prod':prod,
              'tensor_arguments':tensor_arguments,'SavedActivation':SavedActivation,'allocate_saved':allocate_saved,
              'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length())}
        launch=self.launch
        class Kernel:
            def __getitem__(self,grid):
                return lambda *args,**kwargs:launch('_channels_last_activation',grid,*args)
        module=SimpleNamespace(_channels_last_activation=Kernel())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'elementwise.py','exec'),host)
        x=torch.randn(2,5,8,12,dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        geometry=ReLUGeometry(tuple(x.shape))
        with patch.dict('sys.modules',{'deepopt.opt.backends.triton.activation.channels_last':module}),activation_codec_scope(activation_layout='channels_last'):
            y,_,saved,_,_,info=host['_forward'](x,geometry,kind='silu',approximate='none',dtype=x.dtype,
                block_shape=None,packed=True,bits=2,group_size=128,save_mask=True,inplace=False)
        expected,derivative=self.k['_evaluate'](x.float().contiguous().reshape(geometry.matrix_shape),'silu','none')
        packed=quantize_pack_activation(derivative,128,bits=2)
        torch.testing.assert_close(y,expected.reshape(x.shape).to(x.dtype),rtol=0,atol=0)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))
        torch.testing.assert_close(saved.data,packed.data,rtol=0,atol=0)
        self.assertEqual(info['forward_variant'],'channels_last_grouped')

    def test_host_group_owned_fp8_path_keeps_logical_layout_and_supplied_scale(self):
        from deepopt.opt.nn.activation.geometry import ReLUGeometry,tensor_arguments
        from deepopt.opt.backends.reference.activation.elementwise import allocate_saved
        from deepopt.opt.nn.activation.state import SavedActivation
        from deepopt.opt.quantization.activation_tensor import to_fp8_activation
        source=files('deepopt.opt.backends.triton').joinpath('activation/elementwise.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_forward')
        host={'__package__':'deepopt.opt.backends.triton.activation','torch':torch,'prod':prod,
              'tensor_arguments':tensor_arguments,'SavedActivation':SavedActivation,'allocate_saved':allocate_saved,
              'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length())}
        launch=self.launch
        class Kernel:
            def __getitem__(self,grid):
                def call(*args,**kwargs):
                    keys=('input_inverse','scale','output_inverse','INPUT_FP8','OUTPUT_FP8','MAXIMUM')
                    launch('_channels_last_group_activation',grid[0],*args,*(kwargs[k] for k in keys))
                return call
        module=SimpleNamespace(_channels_last_group_activation=Kernel())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'elementwise.py','exec'),host)
        raw=torch.randn(2,23,7,7,dtype=torch.bfloat16)
        x=to_fp8_activation(raw,layout='channels');geometry=ReLUGeometry(tuple(x.shape))
        scale=torch.tensor([[2.0]])
        with patch.dict('sys.modules',{'deepopt.opt.backends.triton.activation.channels_last':module}),activation_codec_scope(activation_layout='channels_last'):
            y,inverse,saved,_,_,info=host['_forward'](x,geometry,kind='silu',approximate='none',dtype=torch.float8_e4m3fn,
                block_shape=None,packed=True,bits=2,group_size=1024,save_mask=True,inplace=False,scale=scale)
        dense=geometry.matrix(x.dequantize(torch.float32))
        expected,derivative=self.k['_evaluate'](dense,'silu','none')
        torch.testing.assert_close(y.float(),(expected*2).clamp(-448,448).to(y.dtype).float(),rtol=0,atol=0)
        torch.testing.assert_close(inverse,scale.reciprocal(),rtol=0,atol=0)
        packed=quantize_pack_activation(derivative,1024,bits=2)
        torch.testing.assert_close(saved.data,packed.data,rtol=0,atol=0)
        self.assertEqual(info['forward_variant'],'channels_last_group_owned')
        self.assertEqual(info['scale_launches'],0)


if __name__=='__main__':
    unittest.main()
