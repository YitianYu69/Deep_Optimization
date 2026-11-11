"""Compare installed-library grouped, tiled, and decode-plus-copy reconstruction.

CUDA execution is required. --help and argument validation do not import Torch.
The default matrices reproduce the dominant saved inputs in the H100 ViT trace.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median, pstdev
from time import perf_counter


def shape(text):
    try:
        dimensions = tuple(int(x) for x in text.lower().split("x"))
    except ValueError:
        dimensions = ()
    if len(dimensions) != 2 or min(dimensions, default=0) < 1:
        raise argparse.ArgumentTypeError("use positive MxK dimensions, e.g. 100864x3072")
    return dimensions


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shapes", nargs="+", type=shape, default=[(100864, 3072), (100864, 768), (4096, 768), (37, 65)])
    result.add_argument("--implementations", nargs="+", choices=("grouped", "copy", "tiled", "auto"),
                        default=["grouped", "copy", "tiled", "auto"])
    result.add_argument("--source-dtype", choices=("bf16", "e4m3fn", "e5m2"), default="e4m3fn")
    result.add_argument("--dtypes", nargs="+", choices=("bf16", "e4m3fn", "e5m2"), default=["e4m3fn"])
    result.add_argument("--bits", nargs="+", type=int, choices=(1, 2, 4, 8), default=[2])
    result.add_argument("--group-sizes", nargs="+", type=int, default=[1024])
    result.add_argument("--block", type=shape, help="Compression/FP8 scale rectangle; overrides group size")
    result.add_argument("--execution", choices=("eager", "graph", "both"), default="both")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--profile-dir", type=Path)
    result.add_argument("--output", type=Path)
    return result


def measure(call, torch, args):
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize(args.device)
    baseline = torch.cuda.memory_allocated(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    wall, stream = [], []
    for _ in range(args.repeats):
        torch.cuda.synchronize(args.device)
        first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        first.record()
        start = perf_counter()
        for _ in range(args.iterations):
            call()
        last.record()
        last.synchronize()
        wall.append((perf_counter() - start) * 1000 / args.iterations)
        stream.append(first.elapsed_time(last) / args.iterations)
    return {"wall_ms_median": median(wall), "wall_ms_samples": wall, "wall_ms_stdev": pstdev(wall),
            "cuda_stream_ms_median": median(stream), "cuda_stream_ms_samples": stream,
            "baseline_allocated_bytes": baseline, "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(args.device) - baseline}


def timings(call, torch, args):
    result = {}
    if args.execution in ("eager", "both"):
        result["eager"] = measure(call, torch, args)
    if args.execution in ("graph", "both"):
        stream = torch.cuda.Stream(device=args.device)
        stream.wait_stream(torch.cuda.current_stream(args.device))
        with torch.cuda.stream(stream):
            for _ in range(max(3, args.warmup)):
                call()
        torch.cuda.current_stream(args.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = call()
        result["graph"] = measure(graph.replay, torch, args)
        del graph, output
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if min(args.iterations, args.repeats, *args.group_sizes) < 1 or args.warmup < 0:
        arguments.error("iterations/repeats/groups must be positive; warmup may be zero")
    if not args.device.startswith("cuda"):
        arguments.error("reconstruction kernel measurements require CUDA")
    if args.block and (args.block == (1, 1) or any(x & (x - 1) for x in args.block)):
        arguments.error("CUDA scale blocks require powers of two excluding 1x1")
    import torch
    import deepopt
    if not torch.cuda.is_available() or torch.version.hip:
        raise SystemExit("An NVIDIA CUDA device is required")
    from deepopt.opt.backends.triton.quantization.activation.affine import dequantize_unpack_affine
    from deepopt.opt.quantization.activation import quantize_pack_activation
    from deepopt.opt.quantization.quantizer import quantize_fp8
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    dtypes = {"bf16": torch.bfloat16, "e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device),
              "deepopt": {"path": str(Path(deepopt.__file__).resolve()), "version": version("deep-optimization")},
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "notes": ["Timing is reconstruction only; forward packing/quantization and correctness checks are outside timing.",
                        "All strategies consume the identical packed payload, metadata and original compression groups.",
                        "copy decodes contiguously then copies a transpose into the same aligned output layout.",
                        "CUDA stream intervals include host submission gaps. Profiling is a separate pass.",
                        "Graph allocator peaks include retained capture pools. Compare graph and eager memory separately.",
                        "Old grouped and tiled arithmetic must match exactly; any mismatch aborts instead of reporting a speedup."],
              "results": []}
    for dimensions in args.shapes:
        source = torch.randn(dimensions, dtype=torch.bfloat16, device=args.device)
        if args.source_dtype != "bf16":
            source = quantize_fp8(source, fp8_dtype=dtypes[args.source_dtype], block_shape=args.block,
                                  check_finite=False).data
        for group in args.group_sizes:
            for bits in args.bits:
                packed = quantize_pack_activation(source, group, args.block, bits=bits)
                for dtype_name in args.dtypes:
                    def decode(implementation, payload=packed, group=group, bits=bits, dtype=dtypes[dtype_name]):
                        return dequantize_unpack_affine(payload.data, payload.step, payload.minimum,
                            payload.shape, group, args.block, dtype=dtype, bits=bits,
                            transpose=True, implementation=implementation)
                    expected = decode("grouped")
                    baseline = None
                    for implementation in dict.fromkeys(["grouped", *args.implementations]):
                        actual = decode(implementation)
                        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
                        if actual.stride() != expected.stride():
                            raise AssertionError("Reconstruction changed the native GEMM output alignment")
                        del actual
                        call = lambda decode=decode, implementation=implementation: decode(implementation)
                        measured = timings(call, torch, args)
                        if implementation == "grouped":
                            baseline = measured
                        row = {"shape": dimensions, "dtype": dtype_name, "bits": bits, "group_size": group,
                               "block": args.block, "implementation": implementation, "exact": True,
                               "timing": measured, "speedup_vs_grouped": {
                                   mode: baseline[mode]["wall_ms_median"] / value["wall_ms_median"]
                                   for mode, value in measured.items()},
                               "packed_payload_and_metadata_bytes": sum(x.numel() * x.element_size()
                                   for x in (packed.data, packed.step, packed.minimum))}
                        if args.profile_dir:
                            args.profile_dir.mkdir(parents=True, exist_ok=True)
                            label = f"{'x'.join(map(str, dimensions))}-{dtype_name}-b{bits}-g{group}-{implementation}"
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA]) as capture:
                                call()
                                torch.cuda.synchronize(args.device)
                            path = args.profile_dir / f"{label}.json"
                            capture.export_chrome_trace(str(path))
                            row["trace"] = str(path)
                        report["results"].append(row)
                    del expected, decode, call
                del packed
        del source
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
