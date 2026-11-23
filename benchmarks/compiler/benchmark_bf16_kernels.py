"""BF16 kernel/pipeline timing with complete cast/pack/decode costs included."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shapes", nargs="+", default=["4096x768x3072", "4096x3072x768"])
    parser.add_argument("--backends", nargs="+", choices=("triton", "tilelang"), default=["triton"])
    parser.add_argument("--tiles", nargs="+", type=int, choices=(32, 64, 128), default=[64])
    parser.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--max-extra-memory-mib", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        shapes = [tuple(int(d) for d in value.split("x")) for value in args.shapes]
        if any(len(s) != 3 or min(s) < 1 for s in shapes):
            raise ValueError
    except ValueError:
        parser.error("shapes must be positive MxKxN triples")
    if min(args.group_size, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error("group-size/iterations/repeats must be positive and warmup nonnegative")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if args.device.startswith("cuda"):
        from _gpu_lease import ensure

        leased = ensure(__file__, argv, args.device)
        if leased is not None:
            return leased
    import torch
    import deepopt
    from deepopt.kernel_policy import kernel_policy_scope
    from deepopt.opt.backends.bf16 import product, prepare
    from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
    from analysis.measurement import measure, synchronize

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(714)
    source = sha256()
    for path in sorted(Path(deepopt.__file__).parent.rglob("*.py")):
        source.update(path.relative_to(Path(deepopt.__file__).parent).as_posix().encode())
        source.update(path.read_bytes())
    versions = {}
    for name in ("deep-optimization", "triton", "tilelang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = {"source_sha256": source.hexdigest(), "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "measurement_sha256": sha256((root / "analysis/measurement.py").read_bytes()).hexdigest(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "packages": versions,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU semantic reference",
                "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING")}
    records = []
    for m, k, n in shapes:
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        compression = {"enabled": True, "bits": args.bits, "group_size": args.group_size}
        packed = quantize_pack_activation(b.T.contiguous(), args.group_size, bits=args.bits)
        reference = product(a, b)
        compressed_reference = product(a, None, packed_b=packed)
        prepare_reference = prepare(a, activation=("gelu", "none"), compression=compression)
        variants = [("product", "native", 64, "none"), ("product", "native", 64, "materialized"),
                    ("prepare", "native", 64, "packed")]
        for backend in args.backends:
            variants += [("product", backend, tile, decode) for tile in args.tiles for decode in ("none", "materialized", "tiled")]
            if args.group_size <= 16384:
                variants.append(("prepare", backend, 64, "packed"))
        random.Random(714).shuffle(variants)
        for operation, backend, tile, decode in variants:
            def call():
                with kernel_policy_scope("triton" if backend == "native" else backend), torch.no_grad():
                    if operation == "prepare":
                        return prepare(a, activation=("gelu", "none"), compression=compression, backend=backend)
                    operand = b
                    if decode == "materialized":
                        operand = dequantize_unpack_activation(packed, dtype=torch.bfloat16, transpose=True)
                    return product(a, None if decode == "tiled" else operand, backend=backend, tile=tile,
                                   packed_b=packed if decode == "tiled" else None)
            try:
                synchronize(torch, device)
                began = perf_counter()
                actual = call()
                synchronize(torch, device)
                cold_ms = (perf_counter() - began) * 1000
                if operation == "prepare":
                    torch.testing.assert_close(actual[0], prepare_reference[0], rtol=2e-2, atol=2e-2)
                    actual_dense = dequantize_unpack_activation(actual[1], dtype=torch.bfloat16)
                    expected_dense = dequantize_unpack_activation(prepare_reference[1], dtype=torch.bfloat16)
                    torch.testing.assert_close(actual_dense, expected_dense, rtol=2e-2, atol=2e-2)
                    del actual_dense, expected_dense
                else:
                    torch.testing.assert_close(actual, reference if decode == "none" else compressed_reference, rtol=2e-2, atol=2e-2)
                del actual
                timing = measure(call, torch, device, args.warmup, args.iterations, args.repeats)
                baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                actual = call()
                synchronize(torch, device)
                memory = {"baseline_allocated_bytes": baseline,
                          "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline if baseline is not None else None,
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if baseline is not None else None}
                del actual
                profile = None
                if args.profile:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities) as profiler:
                        call()
                        synchronize(torch, device)
                    trace = args.output.parent / (args.output.stem + "-traces") / f"{m}-{k}-{n}-{operation}-{backend}-{tile}-{decode}.json"
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    profiler.export_chrome_trace(str(trace))
                    kernels = [e for e in profiler.events() if str(e.device_type).endswith("CUDA")]
                    durations = defaultdict(float)
                    for event in kernels:
                        durations[event.name] += event.time_range.elapsed_us()
                    profile = {"observed_cuda_events": len(kernels), "counts": dict(Counter(e.name for e in kernels)),
                               "duration_us": dict(durations), "trace": str(trace)}
                records.append({"shape_m_k_n": [m, k, n], "operation": operation, "backend": backend,
                                "tile": tile, "decode": decode, "correctness": "pass", "cold_ms": cold_ms,
                                "timing": timing, "memory": memory, "profile": profile,
                                "execution": "cuda" if device.type == "cuda" else "cpu_reference"})
            except Exception as error:
                actual = actual_dense = expected_dense = None
                del actual, actual_dense, expected_dense
                records.append({"shape_m_k_n": [m, k, n], "operation": operation,
                                "backend": backend, "tile": tile, "decode": decode,
                                "status": "error", "correctness": "unqualified",
                                "reason": f"{type(error).__name__}: {error}"})

    recommendations = []
    groups = defaultdict(list)
    for row in records:
        if row.get("correctness") != "pass":
            continue
        key = (tuple(row["shape_m_k_n"]), row["operation"], row["decode"] != "none")
        extra = row["memory"]["peak_extra_allocated_bytes"]
        if args.max_extra_memory_mib is None or extra is not None and extra <= args.max_extra_memory_mib * 2**20:
            groups[key].append(row)
    for (shape, operation, compressed), candidates in groups.items():
        best = min(candidates, key=lambda r: (r["timing"]["cuda_stream_ms"] or r["timing"]["wall_ms"])["median"])
        recommendations.append({"shape": shape, "operation": operation, "compressed": compressed,
                                "backend": best["backend"], "tile": best["tile"], "decode": best["decode"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"identity": identity, "settings": vars(args) | {"output": str(args.output)},
        "results": records, "recommendations": recommendations,
        "note": "Recommendations apply only to this source/hardware/recipe identity and are not installed as defaults. CPU results do not qualify GPU schedules."}, indent=2) + "\n")
    return int(any(row.get("status") == "error" for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
