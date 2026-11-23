#!/usr/bin/env python3
"""Compare materialized and direct-strided attention packing over identical Q/K/V layouts."""

import argparse
from hashlib import sha256
from functools import partial
from itertools import product
import json
from pathlib import Path
import traceback


def manifest(report, expected):
    rows=report["results"]
    return {"expected":expected,"completed":len(rows),"ok":sum(r.get("status","ok")=="ok" for r in rows),
            "failed":sum(r.get("status")=="error" for r in rows),"missing":expected-len(rows)}


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
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--sequence-length", type=int, default=197)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--bits", nargs="+", type=int, choices=(1, 2, 4, 8), default=[4, 8])
    p.add_argument("--group-sizes", nargs="+", type=int, default=[1024])
    p.add_argument("--execution", choices=("eager", "graph", "both"), default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-warps", nargs="+", type=int, choices=(1, 2, 4, 8), default=[4])
    p.add_argument("--indexing", nargs="+", choices=("wide", "bounded"), default=["wide"])
    p.add_argument("--source-layouts", nargs="+", choices=("bhsd", "bshd", "interleaved"),
                   default=["bhsd", "bshd", "interleaved"])
    p.add_argument("--qkv-only", action="store_true", help="Time complete separate/joint QKV packs, omitting per-operand controls")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if min(args.batch_size, args.heads, args.sequence_length, args.head_dim,
           args.iterations, args.repeats, *args.group_sizes) < 1 or args.warmup < 0:
        p.error("sizes/iterations/repeats must be positive; warmup may be zero")
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    import torch
    import deepopt
    from benchmark_activation_pack import helpers, resume_identity, cached_results
    from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
    from deepopt.opt.quantization.activation import quantize_pack_activation, _quantize_pack_strided_activation, _quantize_pack_qkv_activation
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from analysis.attention_comparison import synthetic_workload
    except ImportError:
        from attention_comparison import synthetic_workload

    h = helpers()
    if not torch.cuda.is_available():
        raise SystemExit("An NVIDIA CUDA GPU is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    identity = resume_identity(args, deepopt, torch, Path(h.__file__), Path(__file__), return_metadata=True)
    report = {"resume_key": sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), "identity": identity,
              "configuration": vars(args) | {"output": str(args.output)}, "results": [],
              "gpu": str(torch.cuda.get_device_properties(args.device))}
    cached = cached_results(args.output, report["resume_key"], args, h)
    report["results"] = list(cached.values())
    expected_count=(2 if args.qkv_only else 8)*len(args.source_layouts)*len(args.bits)*len(args.group_sizes)*len(args.num_warps)*len(args.indexing)
    report["manifest"]=manifest(report,expected_count)
    for layout in args.source_layouts:
        fixture = synthetic_workload(args.batch_size, args.heads, args.sequence_length, args.head_dim,
                                     dtype=torch.bfloat16, layout=layout, device=args.device)
        # FA2 saves BSHD, including original K. The source-layout name describes
        # physical storage; it never changes the logical quantization groups.
        values = tuple(x.transpose(1, 2) for x in fixture["qkv"])
        for bits in args.bits:
            for group in args.group_sizes:
                def separate():
                    return tuple(_quantize_pack_strided_activation(value, group, bits=bits) for value in values)
                calls = [("qkv", "separate", separate),
                         ("qkv", "joint", partial(_quantize_pack_qkv_activation, values, group, bits=bits))]
                if not args.qkv_only:
                    for name, value in zip(("q", "k", "v"), values):
                        def materialized(value=value):
                            return quantize_pack_activation(value.contiguous().reshape(-1, args.head_dim), group, bits=bits)
                        calls.extend(((name, "materialized", materialized),
                                      (name, "strided", partial(_quantize_pack_strided_activation, value, group, bits=bits))))
                expected = None
                for warps, indexing in product(args.num_warps, args.indexing):
                    for name, implementation, call in calls:
                        key = f"{layout}-{name}-b{bits}-g{group}-w{warps}-i{indexing}-{implementation}"
                        if key in cached:
                            continue
                        row = {"case_id": key, "source": name, "shape": list(values[0].shape),
                                   "source_strides": [list(x.stride()) for x in values],
                                   "source_layout": layout, "saved_layout": "bshd", "bits": bits, "group_size": group,
                                   "num_warps": warps, "indexing": indexing, "implementation": implementation}
                        try:
                            if expected is None:
                                with activation_codec_scope(num_warps=4,strided_index="wide"):
                                    expected=tuple(quantize_pack_activation(x.contiguous().reshape(-1, args.head_dim), group, bits=bits)
                                                   for x in values)
                            with activation_codec_scope(num_warps=warps, strided_index=indexing):
                                actual = call()
                                wanted = expected if name == "qkv" else (expected[("q", "k", "v").index(name)],)
                                actual = actual if name == "qkv" else (actual,)
                                for a, b in zip(actual, wanted):
                                    for av, bv in zip((a.data, a.step, a.minimum), (b.data, b.step, b.minimum)):
                                        torch.testing.assert_close(av, bv, rtol=0, atol=0)
                                row.update(status="ok",exact=True,timing=h.timings(call, torch, args))
                        except Exception as error:
                            errors=args.output.parent/(args.output.stem+"-errors")
                            errors.mkdir(parents=True,exist_ok=True)
                            trace=errors/(key+".txt");trace.write_text(traceback.format_exc())
                            row.update(status="error",exact=False,reason=f"{type(error).__name__}: {error}",traceback=str(trace))
                        report["results"].append(row)
                        report["manifest"]=manifest(report,expected_count)
                        h.write_checkpoint(args.output, report)
                        print(json.dumps(row), flush=True)
    h.write_checkpoint(args.output, report)
    return int(report["manifest"]["failed"]>0 or report["manifest"]["missing"]!=0)


if __name__ == "__main__":
    raise SystemExit(main())
