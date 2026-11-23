#!/usr/bin/env python3
"""Compare convolution layout-copy codecs against direct packing/decoding."""

import argparse
from functools import partial
import json
from hashlib import sha256
from pathlib import Path
import traceback


def shape(text):
    try:value=tuple(int(n) for n in text.lower().split("x"))
    except ValueError:raise argparse.ArgumentTypeError("shape must be NxCxspatial") from None
    if len(value) not in (3,4,5) or min(value)<1:
        raise argparse.ArgumentTypeError("shape requires three to five positive dimensions")
    return value


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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes",nargs="+",type=shape,default=[(128,64,56,56),(512,32,112,112),(512,256,14,14)])
    p.add_argument("--source-dtypes",nargs="+",choices=("bf16","fp32"),default=["bf16","fp32"])
    p.add_argument("--layouts",nargs="+",choices=("contiguous","channels_last"),default=["contiguous","channels_last"])
    p.add_argument("--bits",nargs="+",type=int,choices=(1,2,4,8),default=[1,2,4,8])
    p.add_argument("--group-sizes",nargs="+",type=int,default=[1024])
    p.add_argument("--execution",choices=("eager","graph","both"),default="both")
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if min(args.iterations,args.repeats,*args.group_sizes)<1 or args.warmup<0:
        p.error("iterations/repeats/group sizes must be positive; warmup must be nonnegative")
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    import torch
    import deepopt
    from benchmark_activation_pack import helpers,resume_identity,cached_results
    from deepopt.opt.quantization.activation import (quantize_pack_activation,dequantize_unpack_activation,
        _quantize_pack_strided_activation,_dequantize_unpack_spatial_activation)
    from deepopt.opt.nn.convolution.geometry import source_matrix,restore_matrix
    if not torch.cuda.is_available() or torch.version.hip:raise SystemExit("An NVIDIA CUDA GPU is required")
    h=helpers()
    torch.cuda.set_device(args.device);torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    identity=resume_identity(args,deepopt,torch,Path(h.__file__),Path(__file__),return_metadata=True)
    report={"identity":identity,"resume_key":sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),"results":[],
            "notes":["Channel-row group order and AMP rounding are unchanged.",
                     "Both materialization and direct implementations include their layout work in timing.",
                     "Full Torch BF16/DeepOpt BF16/compression/FP8/combined updates are in the model and operator matrices."]}
    cached=cached_results(args.output,report["resume_key"],args,h)
    rows=dict(cached)
    expected_count=sum(1 for s in args.shapes for layout in args.layouts if layout=="contiguous" or len(s)>3)
    expected_count*=len(args.source_dtypes)*len(args.bits)*len(args.group_sizes)*4
    def save():
        report["results"]=list(rows.values())
        report["manifest"]={"expected":expected_count,"completed":len(rows),"missing":expected_count-len(rows),
            "failed":sum(r.get("status")=="error" for r in rows.values())}
        h.write_checkpoint(args.output,report)
    for s in args.shapes:
        for layout in args.layouts:
            if layout=="channels_last" and len(s)==3:continue
            fmt=(torch.contiguous_format if layout=="contiguous" else torch.channels_last if len(s)==4 else torch.channels_last_3d)
            for source_dtype in args.source_dtypes:
                x=torch.randn(s,device=args.device,dtype=torch.bfloat16 if source_dtype=="bf16" else torch.float32).to(memory_format=fmt)
                for bits in args.bits:
                    for group in args.group_sizes:
                        calls={
                            "pack-materialized":lambda:quantize_pack_activation(source_matrix(x).to(torch.bfloat16),group,bits=bits),
                            "pack-direct":partial(_quantize_pack_strided_activation,x.movedim(1,-1),group,bits=bits,source_dtype=torch.bfloat16)}
                        packed=None
                        try:packed=calls["pack-materialized"]()
                        except Exception:pass
                        calls["unpack-materialized"]=lambda:restore_matrix(dequantize_unpack_activation(packed,dtype=torch.bfloat16),s,fmt)
                        calls["unpack-direct"]=lambda:_dequantize_unpack_spatial_activation(packed,s,dtype=torch.bfloat16,memory_format=fmt)
                        for variant,call in calls.items():
                            key=f"{'x'.join(map(str,s))}-{layout}-{source_dtype}-b{bits}-g{group}-{variant}"
                            if key in cached:continue
                            row={"case_id":key,"shape":list(s),"source_dtype":source_dtype,"layout":layout,
                                 "bits":bits,"group_size":group,"implementation":variant}
                            try:
                                if packed is None:packed=calls["pack-materialized"]()
                                actual=call()
                                if variant.startswith("pack"):
                                    for a,b in ((actual.data,packed.data),(actual.step,packed.step),(actual.minimum,packed.minimum)):
                                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                                else:
                                    torch.testing.assert_close(actual,calls["unpack-materialized"](),rtol=0,atol=0)
                                    if not actual.is_contiguous(memory_format=fmt):raise AssertionError("Wrong decoded layout")
                                row.update(status="ok",exact=True,timing=h.timings(call,torch,args))
                            except Exception as error:
                                row.update(status="error",exact=False,reason=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
                            rows[key]=row;save();print(json.dumps(row),flush=True)
    save()
    return int(report["manifest"]["failed"]>0 or report["manifest"]["missing"]!=0)


if __name__ == "__main__":
    raise SystemExit(main())
