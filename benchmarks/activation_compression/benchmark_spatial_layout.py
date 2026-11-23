#!/usr/bin/env python3
"""Equal-operand compact/aligned FP8 spatial convolution and reconstruction probes."""

import argparse
from hashlib import sha256
from functools import partial
import json
from pathlib import Path


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--channels", nargs="+", type=int, default=[3])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out-channels", type=int, default=768)
    p.add_argument("--kernel-size", type=int, default=16)
    p.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    p.add_argument("--group-size", type=int, default=1024)
    p.add_argument("--execution", choices=("eager", "graph", "both"), default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if min(args.batch_size, args.image_size, args.out_channels, args.kernel_size,
           args.group_size, args.iterations, args.repeats, *args.channels) < 1 or args.warmup < 0:
        p.error("sizes/iterations/repeats must be positive; warmup may be zero")
    if args.image_size % args.kernel_size:
        p.error("the nonoverlapping patch probe requires image size divisible by kernel size")
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    import torch
    import deepopt
    from benchmark_activation_pack import helpers, resume_identity, cached_results
    from deepopt.opt.nn.convolution import ConvGeometry
    from deepopt.opt.quantization.quantizer import quantize_fp8
    from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
    from deepopt.opt.quantization.tensor import QuantizedTensor
    from deepopt.opt.backends.triton import convolution

    h = helpers()
    if not torch.cuda.is_available():
        raise SystemExit("An NVIDIA CUDA GPU is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    identity = resume_identity(args, deepopt, torch, Path(h.__file__), Path(__file__), return_metadata=True)
    report = {"resume_key": sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), "identity": identity,
              "configuration": vars(args) | {"output": str(args.output)}, "results": [],
              "gpu": str(torch.cuda.get_device_properties(args.device)),
              "note": "Original and decoded operands are separate controls; each layout pair has identical values/scales. Kernel outputs must agree exactly."}
    cached = cached_results(args.output, report["resume_key"], args, h)
    report["results"] = list(cached.values())
    for channels in args.channels:
        shape = (args.batch_size, channels, args.image_size, args.image_size)
        weights = (args.out_channels, channels, args.kernel_size, args.kernel_size)
        geometry = ConvGeometry(shape, weights, (args.kernel_size,) * 2, (0, 0), (1, 1), 1)
        rows = args.batch_size * args.image_size**2
        qx = quantize_fp8(torch.randn(rows, channels, device=args.device, dtype=torch.bfloat16), check_finite=False)
        qw = quantize_fp8(torch.randn(args.out_channels, channels * args.kernel_size**2,
                                     device=args.device, dtype=torch.bfloat16), check_finite=False)
        qg = quantize_fp8(torch.randn(args.batch_size * geometry.output_volume, args.out_channels,
                                     device=args.device, dtype=torch.bfloat16), check_finite=False)
        packed = quantize_pack_activation(qx.data, args.group_size, bits=args.bits)
        for representation in ("original", "decoded"):
            values = qx.data if representation == "original" else dequantize_unpack_activation(packed, dtype=qx.data.dtype, row_alignment=1)
            expected = {}
            for alignment in (1, 16):
                data = torch.empty_strided(values.shape, (((channels + alignment - 1) // alignment) * alignment, 1),
                                           device=args.device, dtype=values.dtype)
                data.copy_(values)
                operand = QuantizedTensor(data, qx.scale_inv)
                calls = {"forward": partial(convolution.forward, operand, qw, geometry, out_dtype=torch.bfloat16, num_stages=3),
                         "dw": partial(convolution.weight_gradient, operand, qg, geometry, out_dtype=torch.bfloat16, num_stages=3)}
                if representation == "decoded":
                    calls["decode"] = partial(dequantize_unpack_activation, packed, dtype=values.dtype, row_alignment=alignment)
                for operation, call in calls.items():
                    actual = call()
                    if operation in expected:
                        torch.testing.assert_close(actual.float(), expected[operation].float(), rtol=0, atol=0)
                    else:
                        expected[operation] = actual
                    key = f"c{channels}-{representation}-align{alignment}-{operation}"
                    if key in cached:
                        continue
                    row = {"case_id": key, "representation": representation, "channels": channels,
                           "row_alignment": alignment, "strides": list(data.stride()), "operation": operation,
                           "exact": True, "timing": h.timings(call, torch, args)}
                    report["results"].append(row)
                    h.write_checkpoint(args.output, report)
                    print(json.dumps(row), flush=True)
    h.write_checkpoint(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
