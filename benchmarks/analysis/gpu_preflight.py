"""Compile the v5 failure paths before admitting the expensive model matrix."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import traceback


def _lease_cli(args, argv=None):
    if __name__ != "__main__" or not getattr(args, "device", "cuda:0").startswith("cuda"):
        return None
    import runpy
    bootstrap = Path(__file__).with_name("_gpu_lease.py")
    if not bootstrap.is_file():
        bootstrap = Path(__file__).resolve().parents[1] / "_gpu_lease.py"
    if not bootstrap.is_file():
        raise SystemExit("Keep _gpu_lease.py and analysis/ beside GPU benchmark programs")
    return runpy.run_path(str(bootstrap))["ensure"](__file__, argv, getattr(args, "device", "cuda:0"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-source-sha256")
    args=parser.parse_args(argv)
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    import torch
    import deepopt
    from deepopt.opt.nn import functional as D
    from deepopt.opt.policy import activation_plan,conv_plan,linear_plan
    from deepopt.opt.quantization.activation_tensor import to_fp8_activation
    from deepopt.opt.quantization.activation import _quantize_pack_strided_activation
    from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
    from deepopt.opt.backends._shared.gemm_policy import linear_backward_scope
    from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms

    digest=sha256()
    root=Path(deepopt.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode()+path.read_bytes())
    report={"status":"running","installed_source_sha256":digest.hexdigest(),"checks":[],
            "torch":str(torch.__version__),"cuda_runtime":torch.version.cuda}
    def checkpoint():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");tmp.replace(args.output)
    if args.expected_source_sha256 and digest.hexdigest()!=args.expected_source_sha256:
        report.update(status="error",reason="Installed DeepOpt source does not match the benchmark export; install the v6 wheel first")
        checkpoint();return 2
    if not torch.cuda.is_available() or torch.version.hip:
        report.update(status="error",reason="An NVIDIA CUDA GPU is required for release acceptance")
        checkpoint();return 2
    report["gpu"]=str(torch.cuda.get_device_properties(0))
    def check(name,call):
        try:
            call();torch.cuda.synchronize()
            row={"name":name,"status":"ok"}
        except Exception as error:
            row={"name":name,"status":"error","reason":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()}
        report["checks"].append(row);checkpoint()
        print(json.dumps({k:v for k,v in row.items() if k!="traceback"}),flush=True)
    def policy(factory):
        return factory(fp8={"enabled":True,"backend":"triton","num_stages":3,"backward_backend":"native_fast"},
            activation_compression={"enabled":True,"bits":4,"group_size":128,"norm_xhat_recenter":True})
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    with deepopt.autocast("cuda",enabled=False,check_finite=False):
        for name in ("relu","silu","gelu","sigmoid","tanh","relu6"):
            for fmt in (torch.bfloat16,torch.float8_e4m3fn,torch.float8_e5m2):
                def activation(name=name,fmt=fmt):
                    raw=torch.randn(3,33,device="cuda",dtype=torch.bfloat16,requires_grad=True)
                    x=raw if fmt==torch.bfloat16 else to_fp8_activation(raw,fp8_dtype=fmt)
                    y=getattr(D,name)(x,plan=policy(activation_plan),output_dtype=torch.bfloat16)
                    y.backward(torch.ones_like(raw))
                    if not torch.isfinite(raw.grad).all():raise AssertionError("Nonfinite activation gradient")
                check(f"{name}-{str(fmt)}-masked-tail",activation)
        def ownership():
            raw=torch.randn(2,4,3,3,device="cuda",dtype=torch.bfloat16)
            weight=torch.randn(8,4,3,3,device="cuda",dtype=torch.bfloat16)
            values=[]
            for inplace in (False,True):
                x,w=raw.clone().requires_grad_(),weight.clone().requires_grad_()
                y=D.conv2d(x,w,plan=policy(conv_plan))
                y=torch.nn.functional.silu(y,inplace=inplace)
                y.float().square().sum().backward()
                values.append((y.detach(),x.grad,w.grad))
            for a,b in zip(*values):torch.testing.assert_close(a,b,rtol=0,atol=0)
            values=[]
            pool_source=torch.randn(2,8,3,5,device="cuda",dtype=torch.bfloat16)
            for inplace in (False,True):
                x=pool_source.clone().requires_grad_()
                y=D.adaptive_avg_pool2d(x,1,plan=linear_plan(False,activation_compression=True)).flatten(1)
                torch.manual_seed(924)
                y=torch.nn.functional.dropout(y,p=.25,training=True,inplace=inplace)
                y.float().square().sum().backward()
                values.append((y.detach(),x.grad))
            for a,b in zip(*values):torch.testing.assert_close(a,b,rtol=0,atol=0)
        check("singleton-conv-inplace-silu-and-pool-flatten-inplace-dropout",ownership)
        def maxpool():
            from deepopt.opt.backends._shared.pooling import pooling_kernel_scope
            source=torch.randn(2,4,9,11,device="cuda",dtype=torch.bfloat16)
            values=[]
            for variant in ("auto","generic"):
                x=source.clone().requires_grad_()
                with pooling_kernel_scope(variant):
                    y,index=D.max_pool2d(x,3,2,1,return_indices=True,plan=linear_plan(False,activation_compression=True))
                y.float().square().sum().backward()
                values.append((y.detach(),index,x.grad))
            for a,b in zip(*values):torch.testing.assert_close(a,b,rtol=0,atol=0)
        check("max-pool3x3-packed-routing-and-completed-gradient",maxpool)
        def gelu_auto():
            source=torch.randn(7,33,device="cuda",dtype=torch.bfloat16)
            dy=torch.randn_like(source)
            values=[]
            plan=activation_plan(fp8={"enabled":True,"backend":"triton"})
            for state in ("derivative","recompute","auto"):
                x=source.clone().requires_grad_()
                with activation_codec_scope(gelu_state=state):
                    y=D.gelu(x,plan=plan,output_dtype=torch.float8_e4m3fn)
                y.backward(dy)
                values.append((y.dequantize().detach(),x.grad))
            for output,gradient in values[1:]:
                torch.testing.assert_close(output,values[0][0],rtol=0,atol=0)
                torch.testing.assert_close(gradient,values[0][1],rtol=0,atol=0)
        check("fp8-only-gelu-auto-matches-derivative-state",gelu_auto)
        def gradient_retile():
            from deepopt.opt.quantization.quantizer import quantize_fp8
            from deepopt.opt.backends.triton.gradient import transpose_quantize,bias
            q=quantize_fp8(torch.randn(9,7,device="cuda"),fp8_dtype=torch.float8_e5m2,block_shape=(2,4))
            transposed=transpose_quantize(q,torch.float8_e5m2,(1,4))
            expected=quantize_fp8(q.dequantize().T,fp8_dtype=torch.float8_e5m2,block_shape=(1,4))
            torch.testing.assert_close(transposed.data.float(),expected.data.float(),rtol=0,atol=0)
            torch.testing.assert_close(transposed.scale_inv,expected.scale_inv,rtol=2e-7,atol=0)
            torch.testing.assert_close(bias(q),q.dequantize().sum(0),rtol=3e-6,atol=1e-6)
        check("fp8-gradient-transpose-and-bias-masked-tails",gradient_retile)
        for source in range(3):
            for bits,group,warps in ((2,128,4),(4,128,4),(8,128,4),(4,1024,2)):
                def bounded(source=source,bits=bits,group=group,warps=warps):
                    qkv=torch.randn(2,19,3,3,17,device="cuda",dtype=torch.bfloat16)
                    value=qkv.permute(2,0,3,1,4).unbind(0)[source]
                    with activation_codec_scope(strided_index="wide",num_warps=warps):
                        expected=_quantize_pack_strided_activation(value,group,bits=bits)
                    with activation_codec_scope(strided_index="bounded",num_warps=warps):
                        actual=_quantize_pack_strided_activation(value,group,bits=bits)
                    for a,b in ((actual.data,expected.data),(actual.step,expected.step),(actual.minimum,expected.minimum)):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream),activation_codec_scope(strided_index="bounded",num_warps=warps):
                        replay=_quantize_pack_strided_activation(value,group,bits=bits)
                    with torch.no_grad():qkv.copy_(torch.randn_like(qkv))
                    graph.replay()
                    with activation_codec_scope(strided_index="wide",num_warps=warps):
                        expected=_quantize_pack_strided_activation(value,group,bits=bits)
                    for a,b in ((replay.data,expected.data),(replay.step,expected.step),(replay.minimum,expected.minimum)):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                check(f"qkv-{source}-b{bits}-g{group}-w{warps}-bounded-replay",bounded)
        def direct_codec():
            from deepopt.opt.quantization.activation import (_dequantize_unpack_spatial_activation,
                quantize_pack_activation,dequantize_unpack_activation)
            for shape in ((2,3,7,9),(2,64,3,5)):
                for fmt in (torch.contiguous_format,torch.channels_last):
                    x=torch.randn(shape,device="cuda").to(memory_format=fmt)
                    for bits in (1,2,4,8):
                        for group in (17,128):
                            p=_quantize_pack_strided_activation(x.movedim(1,-1),group,bits=bits,source_dtype=torch.bfloat16)
                            expected=quantize_pack_activation(x.bfloat16().movedim(1,-1).reshape(-1,shape[1]),group,bits=bits)
                            for a,b in ((p.data,expected.data),(p.step,expected.step),(p.minimum,expected.minimum)):
                                torch.testing.assert_close(a,b,rtol=0,atol=0)
                            out=_dequantize_unpack_spatial_activation(p,x.shape,dtype=torch.bfloat16,memory_format=fmt)
                            oracle=dequantize_unpack_activation(expected).reshape(shape[0],*shape[2:],shape[1]).movedim(-1,1)
                            torch.testing.assert_close(out,oracle,rtol=0,atol=0)
        check("direct-convolution-codec-fp32-amp-rounding-all-bits-and-layouts",direct_codec)
        if torch.cuda.get_device_capability()[0]>=9:
            def linear_candidates():
                from deepopt.opt.quantization.quantizer import _quantize_gradient_pair_and_bias,_quantize_tensorwise_pair
                raw=torch.randn(129,64,device="cuda",dtype=torch.bfloat16)
                a,b,bias=_quantize_gradient_pair_and_bias(raw,fp8_dtype=torch.float8_e4m3fn,check_finite=True)
                expected,transpose=_quantize_tensorwise_pair(raw)
                for x,y in ((a.data.float(),expected.data.float()),(b.data.float(),transpose.data.float()),
                            (a.scale_inv,expected.scale_inv)):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
                torch.testing.assert_close(bias,raw.sum(0,dtype=torch.float32),rtol=2e-5,atol=1e-5)
                plan=linear_plan(fp8={"enabled":True,"backend":"triton","num_stages":3,"backward_backend":"native_fast"})
                x=raw.clone().requires_grad_();weight=torch.randn(96,64,device="cuda",requires_grad=True)
                bias=torch.randn(96,device="cuda",requires_grad=True)
                with linear_backward_scope(gradient_reduction="shared",forward_backend="native_precise",native_layout="prepare"),record_gemms() as records:
                    y=D.linear(x,weight,bias,plan=plan)
                y.float().square().mean().backward()
                if not torch.isfinite(bias.grad).all():raise AssertionError("Nonfinite native bias gradient")
                forward=next(r for r in records if r["role"]=="forward")
                if forward["effective_backend"]!="native_precise":raise AssertionError("Native forward candidate was inactive")
            check("shared-gradient-statistics-and-native-biased-linear",linear_candidates)
    report["status"]="ok" if all(r["status"]=="ok" for r in report["checks"]) else "error"
    checkpoint()
    return int(report["status"]!="ok")


if __name__ == "__main__":
    raise SystemExit(main())
