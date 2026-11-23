#!/usr/bin/env python3
"""Shared-operand FP8 products: representation, codec, accumulation and layout costs."""

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path


PROJECTIONS = {"qkv": (768, 2304), "attention": (768, 768), "fc1": (768, 3072), "fc2": (3072, 768)}


def comparison(actual, expected, torch, error_stats):
    result=error_stats(actual,expected,torch)
    if result["finite"]:
        a,b=actual.double().flatten(),expected.double().flatten()
        na,nb=a.norm(),b.norm()
        difference=a-b
        result["relative_l2"]=float(difference.norm()/nb.clamp_min(1e-30))
        result["max_abs"]=float(difference.abs().max())
        result["reference_l2"]=float(nb)
        result["cosine"]=float((a@b)/(na*nb)) if na and nb else float(not na and not nb)
        result["sign_mismatch_fraction"]=float(((a*b)<0).float().mean())
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tokens", type=int, default=197)
    p.add_argument("--projections", nargs="+", choices=tuple(PROJECTIONS), default=list(PROJECTIONS))
    p.add_argument("--roles", nargs="+", choices=("forward", "dx", "dw"), default=["forward", "dx", "dw"])
    p.add_argument("--output-dtypes", nargs="+", choices=("bf16", "fp32"), default=["bf16", "fp32"])
    p.add_argument("--bits", nargs="+", type=int, choices=(1, 2, 4, 8), default=[2, 8])
    p.add_argument("--group-sizes", nargs="+", type=int, default=[128, 1024])
    p.add_argument("--upstream-scale", type=float, default=1e-4)
    p.add_argument("--execution", choices=("eager", "graph", "both"), default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    return p


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
    p = parser()
    args = p.parse_args(argv)
    if min(args.batch_size, args.tokens, args.iterations, args.repeats, *args.group_sizes) < 1 or args.warmup < 0:
        p.error("sizes/iterations/repeats must be positive and warmup nonnegative")
    import math
    if not math.isfinite(args.upstream_scale) or args.upstream_scale <= 0:
        p.error("upstream-scale must be positive and finite")
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    import torch
    import deepopt
    import importlib.util
    helper_path = Path(__file__).with_name("benchmark_activation_pack.py")
    if not helper_path.is_file():
        helper_path = Path(__file__).resolve().parents[1]/"activation_compression/benchmark_activation_pack.py"
    spec = importlib.util.spec_from_file_location("accumulation_pack_helpers",helper_path)
    pack_helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack_helpers)
    helpers, resume_identity = pack_helpers.helpers, pack_helpers.resume_identity
    from benchmark_fp8_linear_compare import error_stats
    from deepopt.opt.backends import gemm_nt
    from deepopt.opt.backends._shared.gemm_policy import prepare_native_operands
    from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
    from deepopt.opt.quantization.quantizer import quantize_fp8
    from deepopt.opt.quantization.tensor import QuantizedTensor
    if not torch.cuda.is_available():
        raise SystemExit("An NVIDIA CUDA GPU is required")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    h = helpers()
    identity = resume_identity(args, deepopt, torch, Path(h.__file__), Path(__file__), return_metadata=True)
    identity["error_helper"] = sha256(Path(__import__("benchmark_fp8_linear_compare").__file__).read_bytes()).hexdigest()
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report = {"resume_key": key, "identity": identity, "results": [],
              "reference": "FP32 dequantized-operand matmul with TF32 disabled; BF16 original operands are a separate control",
              "notes": ["All backend comparisons use identical FP8 bytes/scales; preparation is a byte copy, never requantization.",
                        "Codec changes only the activation operand saved for dW. Original-input forward/dx are controls.",
                        "No learning/convergence claim follows from these isolated products."]}
    cached = {}
    if args.resume and args.output.exists():
        try:
            old = json.loads(args.output.read_text())
        except (ValueError,OSError):
            old = {}
        if old.get("resume_key") == key:
            for r in old["results"]:
                if (r.get("status") == "ok" and set(r.get("backends",{})) == {"triton","native_fast","native_precise"}
                        and all(h.reusable_row({"exact":True,"timing":v.get("timing",{})},args)
                                for v in r["backends"].values())):
                    cached[r["case_id"]] = r
            report["results"] = list(cached.values())
    for projection in args.projections:
        torch.manual_seed(args.seed)
        k, n = PROJECTIONS[projection]
        m = args.batch_size * args.tokens
        x = torch.randn(m, k, device=args.device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=args.device, dtype=torch.bfloat16)
        g = (torch.randn(m, n, device=args.device, dtype=torch.float32) * args.upstream_scale).bfloat16()
        qx, qw, qg = (quantize_fp8(v, check_finite=False) for v in (x, w, g))
        gt = QuantizedTensor(qg.data.T.contiguous(), qg.scale_inv)
        for role in args.roles:
            raw_a, raw_b = {"forward": (x, w), "dx": (g, w.T), "dw": (g.T, x.T)}[role]
            a, b = {"forward": (qx, qw), "dx": (qg, qw.transpose(contiguous=False)),
                    "dw": (gt, qx.transpose(contiguous=False))}[role]
            recipes = [(None, None)] + ([(bits, group) for bits in args.bits for group in args.group_sizes] if role == "dw" else [])
            for bits, group in recipes:
                operand = b
                if bits is not None:
                    packed = quantize_pack_activation(qx.data, group, bits=bits)
                    values = dequantize_unpack_activation(packed, dtype=qx.data.dtype, transpose=True)
                    operand = QuantizedTensor(values, qx.scale_inv)
                for label in args.output_dtypes:
                    case_key = f"{projection}-{role}-{label}-b{bits}-g{group}"
                    if case_key in cached:
                        continue
                    dtype = torch.bfloat16 if label == "bf16" else torch.float32
                    pa, pb, preparation = prepare_native_operands(a, operand)
                    # Native requires row-major NT storage, including controls.
                    torch.testing.assert_close(pa.data.float(), a.data.float(), rtol=0, atol=0)
                    torch.testing.assert_close(pb.data.float(), operand.data.float(), rtol=0, atol=0)
                    decoded_ref = a.dequantize() @ operand.dequantize().T
                    original_quantized_ref = a.dequantize() @ b.dequantize().T
                    original_fp32 = raw_a.float() @ raw_b.float().T
                    bf16 = (raw_a @ raw_b.T).to(dtype)
                    row = {"case_id": case_key, "projection": projection, "role": role, "bits": bits, "group_size": group,
                           "a_shape": list(a.data.shape), "b_shape": list(operand.data.shape), "output_dtype": str(dtype),
                           "preparation": preparation, "status": "ok", "backends": {},
                           "codec_delta_on_fp32_reference": comparison(decoded_ref, original_quantized_ref, torch,error_stats),
                           "fp8_representation_vs_original_fp32":comparison(original_quantized_ref,original_fp32,torch,error_stats),
                           "bf16_rounding_vs_original_fp32":comparison(bf16,original_fp32,torch,error_stats),
                           "quantized_reference_vs_original_bf16": comparison(original_quantized_ref, bf16, torch,error_stats)}
                    fast_value=None
                    for backend in ("triton", "native_fast", "native_precise"):
                        call = partial(gemm_nt, pa, pb, backend=backend, out_dtype=dtype, num_stages=3)
                        try:
                            value = call()
                            result = {"vs_decoded_fp32": comparison(value, decoded_ref, torch,error_stats),
                                      "vs_original_bf16": comparison(value, bf16, torch,error_stats),
                                      "finite": bool(torch.isfinite(value).all()), "timing": h.timings(call, torch, args)}
                            if backend=="native_fast":
                                fast_value=value
                            elif backend=="native_precise" and fast_value is not None:
                                row["fast_vs_precise_same_operands"]=comparison(fast_value,value,torch,error_stats)
                            del value
                            if preparation is not None:
                                def complete(backend=backend):
                                    ca, cb, _ = prepare_native_operands(a, operand)
                                    return gemm_nt(ca, cb, backend=backend, out_dtype=dtype, num_stages=3)
                                result["including_preparation"] = h.timings(complete, torch, args)
                            if not result["finite"]:
                                row["status"] = "error"
                            row["backends"][backend] = result
                        except Exception as error:
                            row["status"] = "error"
                            row["backends"][backend] = {"error": f"{type(error).__name__}: {error}"}
                    report["results"].append(row)
                    h.write_checkpoint(args.output, report)
                    print(case_key, row["status"], flush=True)
                    del decoded_ref, original_quantized_ref, original_fp32, bf16, fast_value
        del x,w,g,qx,qw,qg,gt
    return int(any(r["status"] != "ok" for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
