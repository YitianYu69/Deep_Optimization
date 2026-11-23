"""Standalone H100 implementation preflight for the runtime speed kernels."""

from copy import deepcopy
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
from deepopt.opt.backends._shared.normalization import normalization_kernel_scope
from deepopt.opt.backends._shared.convolution import convolution_kernel_scope,record_convolution_launches


_AVAILABLE = torch.cuda.is_available() and torch.version.hip is None
RUNTIME_SPEED_PREFLIGHT_REVISION = 4
CONVOLUTION_DIAGNOSTICS = []
CONVOLUTION_ARTIFACT_PATH = None
CONVOLUTION_ARTIFACT_WRITTEN = False
_CONVOLUTION_ARTIFACT_ROWS = {}


def exact_convolution_operands(geometry,device):
    """Exact signed sums isolate indexing/scales from native FP8 accumulation.

    The tested reductions have fewer than 2048 terms. Ternary products and all
    partial sums fit exactly even in FP16; binary scales stay exact as well.
    """
    from deepopt.opt.quantization.tensor import QuantizedTensor
    reductions=(geometry.in_channels_per_group*geometry.kernel_volume,
                geometry.out_channels_per_group*geometry.kernel_volume,
                geometry.batch_size*geometry.output_volume)
    if max(reductions)>=2048:
        raise ValueError('Exact FP8 preflight requires reductions below 2048')
    shapes=(geometry.input_matrix_shape,geometry.weight_matrix_shape,geometry.grad_matrix_shape)
    return tuple(QuantizedTensor(torch.randint(-1,2,shape,device=device).to(torch.float8_e4m3fn),
                                 torch.full((1,1),scale,device=device,dtype=torch.float32))
                 for shape,scale in zip(shapes,(.5,2.,.25)))


@torch.no_grad()
def advance_exact_operands(values):
    for value in values:
        value.data.copy_((-value.data.float()).roll(1,0).to(value.data.dtype))
        value.scale_inv.mul_(.5)


@torch.no_grad()
def convolution_cpu_reference(qx,qw,qg,geometry):
    """Use the existing GPU codes/scales without cuDNN's ambient TF32 policy.

    Copy on every call so graph-replay checks observe current operands. Do not
    independently quantize CPU random inputs: their codes/scales can differ.
    """
    from deepopt.opt.backends.reference import convolution as reference
    from deepopt.opt.quantization.tensor import QuantizedTensor
    x,w,g=(QuantizedTensor(q.data.detach().to(device='cpu',copy=True),
                          q.scale_inv.detach().to(device='cpu',copy=True),q.block_shape) for q in (qx,qw,qg))
    return (reference.forward(x,w,geometry),reference.input_gradient(g,w,geometry),
            reference.weight_gradient(x,g,geometry))


def assert_convolution_close(actual,expected,*,variant,policy,launches,exact=False):
    tolerance=0. if exact else .003
    for role,a,b in zip(('forward','dx','dw'),actual,expected):
        torch.testing.assert_close(a.detach().cpu(),b,rtol=tolerance,atol=tolerance,
            msg=lambda message,role=role: (f'{variant}/{policy}/{role}; reference=CPU FP32 from identical FP8 payload/scales; '
                                          f'launches={launches}\n{message}'))


def convolution_error_diagnostics(actual,expected):
    rows=[]
    for role,a,b in zip(('forward','dx','dw'),actual,expected):
        a,b=a.detach().cpu().double(),b.detach().cpu().double()
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        difference=(a-b).abs()
        rows.append({'role':role,'finite':finite,'max_abs':float(difference.max()) if finite else None,
                     'relative_l2':float(difference.norm()/b.norm().clamp_min(1e-30)) if finite else None,
                     'mismatched_elements':int((~torch.isclose(a,b,rtol=.003,atol=.003)).sum())})
    return rows


def assert_convolution_scaling(actual,unit,scales,*,variant,policy):
    x,w,g=scales
    for role,a,raw,(first,second) in zip(('forward','dx','dw'),actual,unit,((x,w),(g,w),(g,x))):
        expected=(raw.float()*first)*second
        torch.testing.assert_close(a,expected,rtol=2e-6,atol=2e-6,
            msg=lambda message,role=role:f'{variant}/{policy}/{role}; original versus unit-scale native accumulation\n{message}')


def record_native_convolution(actual,expected,values,geometry,*,variant,policy,stage,launches):
    global CONVOLUTION_ARTIFACT_WRITTEN
    row={'variant':variant,'policy':policy,'stage':stage,'launches':list(launches),
         'kind':'native_fp8_vs_ideal_fp32','precision_gate':False,
         'pointwise_diagnostic_tolerance':{'rtol':.003,'atol':.003},
         'metrics':convolution_error_diagnostics(actual,expected)}
    CONVOLUTION_DIAGNOSTICS.append(row)
    if CONVOLUTION_ARTIFACT_PATH is not None:
        key=f'{variant}-{policy}-{stage}'
        operands=[{'data':q.data.detach().to(device='cpu',copy=True),
                   'scale_inv':q.scale_inv.detach().to(device='cpu',copy=True),'block_shape':q.block_shape} for q in values]
        _CONVOLUTION_ARTIFACT_ROWS[key]={'diagnostics':row,'operands':operands,
            'input_shape':geometry.input_shape,'weight_shape':geometry.weight_shape,
            'stride':geometry.stride,'padding':geometry.padding,'dilation':geometry.dilation,'groups':geometry.groups,
            'actual':tuple(a.detach().cpu() for a in actual),'ideal_fp32':expected}
        temporary=CONVOLUTION_ARTIFACT_PATH.with_suffix('.tmp')
        torch.save(_CONVOLUTION_ARTIFACT_ROWS,temporary)
        temporary.replace(CONVOLUTION_ARTIFACT_PATH)
        CONVOLUTION_ARTIFACT_WRITTEN = True
    print('Native FP8 convolution diagnostic: '+json.dumps(row,allow_nan=False),flush=True)
    return row


@unittest.skipUnless(_AVAILABLE, "An NVIDIA CUDA GPU is required")
class RuntimeSpeedCUDATests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("The speed preflight requires Hopper FP8")
        torch.manual_seed(14)

    def graph_check(self, step, change, check):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = step()
        for _ in range(2):
            change()
            graph.replay()
            check(actual)

    def test_channels_last_batch_norm_codec_gradients_and_replay(self):
        from deepopt.opt.backends.triton import normalization as compute
        from deepopt.opt.nn.normalization import NormGeometry
        from deepopt.opt.quantization.activation import quantize_pack_activation,dequantize_unpack_activation
        x = torch.randn(2,5,8,12,device='cuda',dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        dy = torch.randn_like(x);weight=torch.randn(5,device='cuda');bias=torch.randn_like(weight)
        geometry=NormGeometry('batch',tuple(x.shape));policy={'enabled':True,'bits':2,'group_size':128}
        def step():
            center=torch.empty(5,device='cuda')
            with normalization_kernel_scope('channels_last'):
                y,mu,inv,packed=compute.forward(x,weight,bias,geometry,policy,center_shift=center)
                dx,dw,db=compute.backward(dy,weight,geometry,packed=packed,mean=mu,inverse=inv,
                    center_shift=center,input_dtype=x.dtype)
            return y,mu,inv,packed,center,dx,dw,db
        def check(values):
            y,mu,inv,packed,center,dx,dw,db=values
            h=(geometry.matrix(x.float())-mu[:,None])*inv[:,None]
            expected=quantize_pack_activation(h,128,(1,128),bits=2)
            torch.testing.assert_close(packed.data,expected.data,rtol=0,atol=0)
            for a,b in ((packed.step,expected.step),(packed.minimum,expected.minimum)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            wanted=geometry.restore(h)*weight[None,:,None,None]+bias[None,:,None,None]
            torch.testing.assert_close(y,wanted.to(y.dtype),rtol=.005,atol=.005)
            decoded=dequantize_unpack_activation(packed,dtype=torch.float32)-center[:,None]
            g=geometry.matrix(dy.float())
            wanted_dx=(g-g.mean(1)[:,None]-decoded*(g*decoded).mean(1)[:,None])*weight[:,None]*inv[:,None]
            torch.testing.assert_close(dx,geometry.restore(wanted_dx).to(dx.dtype),rtol=.01,atol=.01)
            torch.testing.assert_close(dw,(g*decoded).sum(1),rtol=.003,atol=.003)
            torch.testing.assert_close(db,g.sum(1),rtol=.003,atol=.003)
        check(step())
        def change():
            x.add_(.125);dy.mul_(.9);weight.add_(.01)
        self.graph_check(step,change,check)

    def test_channels_last_activation_payload_and_backward_scope(self):
        from deepopt.opt.nn import functional as DF
        from deepopt.opt.policy import linear_plan
        from deepopt.opt.backends.reference.activation.math import evaluate
        from deepopt.opt.quantization.activation import quantize_pack_activation,dequantize_unpack_activation
        plan=linear_plan(False,activation_compression={'enabled':True,'bits':2,'group_size':1024})
        for kind,function in (('silu',DF.do_silu),('gelu',DF.do_gelu),('relu',DF.do_relu)):
            x=torch.randn(2,5,32,48,device='cuda',dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last).requires_grad_()
            dy=torch.randn_like(x);info={}
            with activation_codec_scope(activation_layout='channels_last'):
                y=function(x,plan=plan,_execution_info=info)
            y.backward(dy)
            if kind=='relu':
                expected=F.relu(x.detach());coefficient=(~(x.detach()<=0)).float()
            else:
                expected,derivative=evaluate(x.detach().float(),kind,'none')
                packed=quantize_pack_activation(derivative.reshape(-1,48),1024,bits=2)
                coefficient=dequantize_unpack_activation(packed,dtype=torch.float32).reshape(x.shape)
            torch.testing.assert_close(y,expected.to(y.dtype),rtol=.02,atol=.008)
            torch.testing.assert_close(x.grad,(dy.float()*coefficient).to(x.dtype),rtol=.03,atol=.02)
            self.assertEqual(info['forward_variant'],'channels_last_grouped')

    def test_dense_and_depthwise_tuning_and_changing_value_replay(self):
        from deepopt.opt.backends import convolution as executor
        from deepopt.opt.nn.convolution import ConvGeometry,source_matrix
        from deepopt.opt.quantization.quantizer import quantize_fp8
        for variant,ci,co,groups in (('strict',32,64,1),('depthwise',5,5,5)):
            with self.subTest(variant=variant):
                spatial=(9,11) if variant=='strict' else (23,29)
                x=torch.randn(2,ci,*spatial,device='cuda');w=torch.randn(co,ci//groups,3,3,device='cuda')
                geometry=ConvGeometry.from_tensors(x,w,padding=1,groups=groups)
                g=torch.randn(geometry.output_shape,device='cuda')
                initial_sources=(x,w,g)
                for policy in ('baseline','tuned'):
                    with self.subTest(policy=policy):
                        x,w,g=(value.clone() for value in initial_sources)
                        # Exact arithmetic is a strict implementation gate;
                        # native Hopper FP8 does not promise ideal FP32 sums.
                        qx,qw,qg=exact_convolution_operands(geometry,'cuda')
                        launches=[]
                        def step():
                            options=dict(backend='triton',kernel_variant=variant)
                            with convolution_kernel_scope(policy),record_convolution_launches() as records:
                                actual=(executor.forward(qx,qw,geometry,**options),
                                        executor.input_gradient(qg,qw,geometry,**options),
                                        executor.weight_gradient(qx,qg,geometry,**options))
                            launches[:]=records
                            return actual
                        def check(actual):
                            expected=convolution_cpu_reference(qx,qw,qg,geometry)
                            assert_convolution_close(actual,expected,variant=variant,policy=policy,launches=launches,exact=True)
                        check(step())
                        self.graph_check(step,lambda:advance_exact_operands((qx,qw,qg)),check)

                        # Preserve full-range random coverage and require exact
                        # eager/graph agreement on the same native launch.
                        qx,qw,qg=[quantize_fp8(t) for t in (source_matrix(x),w.reshape(co,-1),source_matrix(g))]
                        replay=0
                        def check_random(actual):
                            nonlocal replay
                            expected=convolution_cpu_reference(qx,qw,qg,geometry)
                            row=record_native_convolution(actual,expected,(qx,qw,qg),geometry,
                                variant=variant,policy=policy,stage=f'random-{replay}',launches=launches)
                            self.assertTrue(all(r['finite'] for r in row['metrics']),'Nonfinite convolution result')
                            saved_scales=tuple(q.scale_inv.clone() for q in (qx,qw,qg))
                            try:
                                for q in (qx,qw,qg):q.scale_inv.fill_(1.)
                                unit=step()
                            finally:
                                for q,scale in zip((qx,qw,qg),saved_scales):q.scale_inv.copy_(scale)
                            assert_convolution_scaling(actual,unit,saved_scales,variant=variant,policy=policy)
                            eager=step()
                            for role,a,b in zip(('forward','dx','dw'),actual,eager):
                                torch.testing.assert_close(a,b,rtol=0,atol=0,
                                    msg=lambda message,role=role:f'{variant}/{policy}/{role}; graph/eager native replay\n{message}')
                            replay+=1
                        check_random(step())
                        def change():
                            for quantized,source in zip((qx,qw,qg),(source_matrix(x+.2),w.reshape(co,-1)*.9,source_matrix(g-.1))):
                                replacement=quantize_fp8(source)
                                quantized.data.copy_(replacement.data);quantized.scale_inv.copy_(replacement.scale_inv)
                            x.add_(.1);w.mul_(.99);g.add_(.05)
                        self.graph_check(step,change,check_random)

    def test_fp8_finite_range_packer_and_transposed_decode(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation,dequantize_unpack_activation
        for dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
            x=torch.randn(257,129,device='cuda').to(dtype)
            for bits in (1,2,4,8):
                actual=quantize_pack_activation(x,1024,bits=bits)
                expected=quantize_pack_activation(x.float(),1024,bits=bits)
                for a,b in ((actual.data,expected.data),(actual.step,expected.step),(actual.minimum,expected.minimum)):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
                decoded=dequantize_unpack_activation(actual,dtype=dtype,transpose=True)
                reference=dequantize_unpack_activation(expected,dtype=dtype).T
                torch.testing.assert_close(decoded.float(),reference.float(),rtol=0,atol=0)

    def test_maxpool_routes_ties_nans_and_changing_replay(self):
        from deepopt.opt.nn import functional as DF
        from deepopt.opt.policy import linear_plan
        x=torch.randn(2,5,9,11,device='cuda',dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last).requires_grad_()
        with torch.no_grad():
            x[0,0,:3,:3]=1.;x[0,1,1,1]=float('nan');x[0,1,2,2]=float('nan')
        plan=linear_plan(False,activation_compression={'enabled':True,'bits':2,'group_size':128})
        dy=torch.randn(2,5,5,6,device='cuda',dtype=torch.bfloat16)
        def step():
            x.grad=None
            y,index=DF.do_max_pool2d(x,3,stride=2,padding=1,return_indices=True,plan=plan)
            y.backward(dy)
            return y,index
        def check(actual):
            grad=x.grad.clone();other=x.detach().clone().requires_grad_()
            y,index=F.max_pool2d(other,3,stride=2,padding=1,return_indices=True);y.backward(dy)
            torch.testing.assert_close(actual[0],y,rtol=0,atol=0,equal_nan=True)
            torch.testing.assert_close(actual[1],index,rtol=0,atol=0)
            torch.testing.assert_close(grad,other.grad,rtol=0,atol=0)
        check(step())
        def change():
            with torch.no_grad():x.copy_(x.roll(1,-1))
        self.graph_check(step,change,check)

    def test_fp8_attention_block_destinations_and_graph_updates(self):
        from deepopt.opt.policy import prepare_fp8_model
        class Block(nn.Module):
            def __init__(self):
                super().__init__();self.qkv=nn.Linear(64,192);self.proj=nn.Linear(64,64)
            def forward(self,x):
                q,k,v=self.qkv(x).reshape(2,32,3,2,32).unbind(2)
                q,k,v=(t.transpose(1,2).contiguous() for t in (q,k,v))
                y=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(2,32,64)
                return self.proj(y)
        settings=deepopt.FP8Config(enabled=True,backend='triton',trace_mode='modules',backward_backend='native_fast')
        compression={'enabled':True,'bits':2,'attention_bits':4,'group_size':128,'norm_xhat_recenter':True}
        config={'runtime':{'attention':{'backend':'flash_attn_2'},'activation_compression':compression}}
        source=Block().cuda();control=deepcopy(source)
        model,_=prepare_fp8_model(source,settings,config=config)
        control,_=prepare_fp8_model(control,settings,config=config)
        x=torch.randn(2,32,64,device='cuda');other=x.clone()
        def run(current,input):
            current.zero_grad(set_to_none=True)
            with activation_codec_scope(attention_grad_layout='bshd'),deepopt.autocast('cuda',dtype=torch.bfloat16,
                    fp8=settings,activation_compression=compression,attention_backend='flash_attn_2',cache_enabled=False,check_finite=False):
                y=current(input)
            y.square().mean().backward()
            return y
        def change():
            x.add_(.01);other.copy_(x)
            with torch.no_grad():
                for a,b in zip(model.parameters(),control.parameters()):a.add_(.001);b.copy_(a)
        def check(y):
            gradients=[p.grad.clone() for p in model.parameters()]
            expected=run(control,other)
            torch.testing.assert_close(y,expected,rtol=.01,atol=.01)
            for a,b in zip(gradients,control.parameters()):torch.testing.assert_close(a,b.grad,rtol=.02,atol=.01)
        run(model,x);run(control,other)
        self.graph_check(lambda:run(model,x),change,check)


if __name__=='__main__':
    unittest.main()
