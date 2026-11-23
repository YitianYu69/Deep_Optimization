"""Compare installed-library grouped, tiled, and decode-plus-copy reconstruction.

CUDA execution is required. --help and argument validation do not import Torch.
The default matrices reproduce the dominant saved inputs in the H100 ViT trace.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import importlib.util
from hashlib import sha256
import json
import math
import os
import subprocess
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
    result.add_argument("--tiles", nargs="+", type=shape, help="Explicit tiled-kernel shapes, e.g. 32x32 32x64 64x64")
    result.add_argument("--warps", nargs="+", type=int, choices=(4, 8), default=[4])
    result.add_argument("--deepopt-backend", choices=("triton", "tilelang", "mixed"), default="triton")
    result.add_argument("--deepopt-kernel-policy", type=json.loads, default={}, metavar="JSON",
                        help="runtime.kernels mapping; route overrides require --deepopt-backend mixed")
    result.add_argument("--resume", action="store_true", help="Reuse matching completed cases from --output; retry incomplete cases")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--time-budget", type=float, default=2.0, help="Warm timing target per variant in seconds; 0 uses exhaustive counts")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--profile-dir", type=Path)
    result.add_argument("--output", type=Path)
    return result


def backend_sources(packages):
    """Hash installed compiler sources/templates/runtime without importing them."""
    result = {}
    for name, installed_version in sorted(packages.items()):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
        digest = sha256()
        relevant = {".py", ".h", ".hpp", ".cuh", ".c", ".cc", ".cpp", ".cu", ".so", ".dylib", ".dll", ".pyd", ".bc", ".ptx", ".cubin"}
        sources = ([] if origin is None else sorted(path for path in origin.parent.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts and any(suffix in relevant for suffix in path.suffixes)))
        for path in sources:
            digest.update(path.relative_to(origin.parent).as_posix().encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        result[name] = {"version": installed_version, "path": str(origin) if origin is not None else None,
                        "source_sha256": digest.hexdigest() if sources else None,
                        "source_files": len(sources)}
    return result


def codec_schedule(module, operation, requested):
    """Separate semantic controls from the schedule actually implemented."""
    backend = module.__name__.split(".backends.", 1)[-1].split(".", 1)[0]
    if backend == "tilelang":
        return {"backend": backend, "actual_schedule": "serial_per_group" if operation == "pack" else "flat_elementwise",
                "threads": 32 if operation == "pack" else 128,
                "groups_per_program": 32 if operation == "pack" else None,
                "tuning_controls_active": False, "accepted_but_inactive_controls": requested}
    return {"backend": backend, "actual_schedule": "backend_selected",
            "tuning_controls_active": True, "requested_controls": requested}


def codec_module():
    from deepopt.opt.backends.dispatch import backend_module
    return backend_module("activation_codec", "quantization.activation.affine")


def codec_choices(args, operation, schedule):
    if operation == "pack":
        if not schedule["tuning_controls_active"]:
            return [("grouped", 1, args.num_warps[0])]
        return ([("grouped", 1, warps) for warps in args.num_warps]
                + [(variant, batch, warps) for variant in args.variants
                   for batch in args.groups_per_program for warps in args.num_warps])
    result = [("grouped", None, 4)]
    if schedule["tuning_controls_active"]:
        for implementation in dict.fromkeys(args.implementations):
            if implementation == "tiled":
                result.extend((implementation, tile, warps) for tile in (args.tiles or [None]) for warps in args.warps)
            elif implementation != "grouped":
                result.append((implementation, None, 4))
    return result


def write_checkpoint(path, report):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def reusable_row(row, args):
    modes = ("eager", "graph") if args.execution == "both" else (args.execution,)
    def complete(value):
        count = args.repeats
        sampling = value.get("sampling")
        if sampling is not None:
            budget = float(getattr(args, "time_budget", 0.0))
            if (sampling.get("requested_repeats") != args.repeats
                    or sampling.get("time_budget_seconds") != budget
                    or sampling.get("requested_iterations") != args.iterations
                    or sampling.get("requested_warmup") != args.warmup):
                return False
            count = sampling.get("actual_repeats")
            iterations = sampling.get("actual_iterations_per_repeat", [])
            if (type(count) is not int or not 1 <= count <= args.repeats
                    or len(iterations) != count
                    or any(type(n) is not int or not 1 <= n <= args.iterations for n in iterations)):
                return False
            warmup = sampling.get("actual_warmup")
            if type(warmup) is not int or not 0 <= warmup <= args.warmup:
                return False
            if not budget and (count != args.repeats or warmup != args.warmup
                               or any(n != args.iterations for n in iterations)):
                return False
        return len(value.get("wall_ms_samples", [])) == count
    return (row.get("exact") is True and all(
        isinstance((row.get("timing") or {}).get(mode, {}).get("wall_ms_median"), (int, float))
        and math.isfinite(row["timing"][mode]["wall_ms_median"]) and row["timing"][mode]["wall_ms_median"] > 0
        and complete(row["timing"][mode]) for mode in modes)
                and (getattr(args, "profile_dir", None) is None or bool(row.get("trace") and Path(row["trace"]).is_file())))


def cached_results(path, key, args):
    if not args.resume or path is None or not path.is_file():
        return {}
    try:
        previous = json.loads(path.read_text())
        if previous.get("resume_key") != key:
            return {}
        return {row["case_id"]: row for row in previous.get("results", [])
                if "case_id" in row and reusable_row(row, args)}
    except (ValueError, OSError, TypeError, KeyError):
        return {}


def resume_identity(args, deepopt, torch, kernel_backend):
    digest = sha256()
    root = Path(deepopt.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    identity = {"build": digest.hexdigest(), "script": sha256(Path(__file__).read_bytes()).hexdigest(),
                "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                            if k not in ("resume", "output")},
                "torch": str(torch.__version__), "triton": kernel_backend["packages"].get("triton"),
                "gpu": torch.cuda.get_device_name(args.device),
                "kernel_backend": kernel_backend, "backend_sources": backend_sources(kernel_backend["packages"]),
                "gpu_uuid": str(getattr(torch.cuda.get_device_properties(args.device), "uuid", "unavailable")),
                "cuda": torch.version.cuda, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        identity["driver"] = driver.stdout.strip() if driver.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        identity["driver"] = "unavailable"
    if not identity["driver"] or identity["driver"] == "unavailable":
        identity["unverified_run_nonce"] = os.urandom(16).hex()
    identity["allocator_environment"] = {key: os.environ.get(key) for key in
                                         ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}
    return identity


def measure(call, torch, args):
    from deepopt._benchmarking import plan_timing
    began = perf_counter()
    plan = plan_timing(call, args, synchronize=lambda: torch.cuda.synchronize(args.device))
    for _ in range(plan.warmup):
        call()
    torch.cuda.synchronize(args.device)
    baseline = torch.cuda.memory_allocated(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    wall, stream = [], []
    for _ in range(plan.repeats):
        torch.cuda.synchronize(args.device)
        first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        first.record()
        start = perf_counter()
        for _ in range(plan.iterations):
            call()
        last.record()
        last.synchronize()
        wall.append((perf_counter() - start) * 1000 / plan.iterations)
        stream.append(first.elapsed_time(last) / plan.iterations)
    return {"wall_ms_median": median(wall), "wall_ms_samples": wall, "wall_ms_stdev": pstdev(wall),
            "cuda_stream_ms_median": median(stream), "cuda_stream_ms_samples": stream,
            "baseline_allocated_bytes": baseline, "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(args.device) - baseline,
            "sampling": plan.sampling(perf_counter() - began)}


def timings(call, torch, args):
    result = {}
    if args.execution in ("eager", "both"):
        result["eager"] = measure(call, torch, args)
    if args.execution in ("graph", "both"):
        from deepopt._benchmarking import capture_warmup_steps
        capture_steps = capture_warmup_steps(args)
        stream = torch.cuda.Stream(device=args.device)
        stream.wait_stream(torch.cuda.current_stream(args.device))
        with torch.cuda.stream(stream):
            for _ in range(capture_steps):
                call()
        torch.cuda.current_stream(args.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = call()
        result["graph"] = measure(graph.replay, torch, args)
        result["graph"]["capture_warmup_steps"] = capture_steps
        del graph, output
    return result


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
    arguments = parser()
    args = arguments.parse_args(argv)
    if min(args.iterations, args.repeats, *args.group_sizes) < 1 or args.warmup < 0:
        arguments.error("iterations/repeats/groups must be positive; warmup may be zero")
    if not args.device.startswith("cuda"):
        arguments.error("reconstruction kernel measurements require CUDA")
    if args.block and (args.block == (1, 1) or any(x & (x - 1) for x in args.block)):
        arguments.error("CUDA scale blocks require powers of two excluding 1x1")
    if args.tiles and any(any(x not in (16, 32, 64, 128) for x in tile) or math.prod(tile) > 8192 for tile in args.tiles):
        arguments.error("tiles require dimensions 16..128, powers of two, and at most 8192 elements")
    if args.resume and args.output is None:
        arguments.error("--resume requires --output")
    from deepopt._benchmarking import BenchmarkBackend, timing_plan_from_cost
    try:
        selection = BenchmarkBackend.from_args(args)
        timing_plan_from_cost(args, 0.0)
    except (TypeError, ValueError) as error:
        arguments.error(str(error))
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    with selection.scope():
        return run_selected(args, selection)


def run_selected(args, selection):
    import torch
    import deepopt
    if not torch.cuda.is_available() or torch.version.hip:
        raise SystemExit("An NVIDIA CUDA device is required")
    selected_codec = codec_module()
    dequantize_unpack_affine = selected_codec.dequantize_unpack_affine
    from deepopt.opt.quantization.activation import quantize_pack_activation
    from deepopt.opt.quantization.quantizer import quantize_fp8
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    kernel_backend = selection.metadata(args.device)
    dtypes = {"bf16": torch.bfloat16, "e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device),
              "deepopt": {"path": str(Path(deepopt.__file__).resolve()), "version": version("deep-optimization")},
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "kernel_backend": kernel_backend,
              "notes": ["Timing is reconstruction only; forward packing/quantization and correctness checks are outside timing.",
                        "All strategies consume the identical packed payload, metadata and original compression groups.",
                        "Triton copy decodes contiguously then copies a transpose into the same aligned output layout.",
                        "TileLang retains a semantic grouped control with its actual flat elementwise schedule; inactive tuning controls are recorded and deduplicated.",
                        "CUDA stream intervals include host submission gaps. Profiling is a separate pass.",
                        "Graph allocator peaks include retained capture pools. Compare graph and eager memory separately.",
                        "Old grouped and tiled arithmetic must match exactly; any mismatch aborts instead of reporting a speedup."],
              "results": []}
    identity = resume_identity(args, deepopt, torch, kernel_backend)
    report["allocator_environment"] = identity["allocator_environment"]
    report["allocator_backend"] = torch.cuda.memory.get_allocator_backend()
    report["deepopt"]["source_sha256"] = identity["build"]
    report["identity"] = identity
    report["resume_key"] = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cached = cached_results(args.output, report["resume_key"], args)
    # Keep every valid cached checkpoint on disk even if the resumed process
    # stops before revisiting its position in the grid.
    report["results"] = [{**row, "resumed": True} for row in cached.values()]
    schedule = codec_schedule(selected_codec, "unpack", {
        "implementations": args.implementations, "tile_shape": args.tiles, "num_warps": args.warps})
    strategies = codec_choices(args, "unpack", schedule)
    for dimensions in args.shapes:
        source = torch.randn(dimensions, dtype=torch.bfloat16, device=args.device)
        if args.source_dtype != "bf16":
            source = quantize_fp8(source, fp8_dtype=dtypes[args.source_dtype], block_shape=args.block,
                                  check_finite=False).data
        for group in args.group_sizes:
            for bits in args.bits:
                packed = quantize_pack_activation(source, group, args.block, bits=bits)
                for dtype_name in args.dtypes:
                    def decode(implementation, tile=None, warps=4, payload=packed, group=group, bits=bits, dtype=dtypes[dtype_name]):
                        return dequantize_unpack_affine(payload.data, payload.step, payload.minimum,
                            payload.shape, group, args.block, dtype=dtype, bits=bits,
                            transpose=True, implementation=implementation, tile_shape=tile, num_warps=warps)
                    expected = decode("grouped")
                    call = None
                    baseline = None
                    for implementation, tile, warps in strategies:
                        case_id = sha256(json.dumps([dimensions, group, bits, dtype_name, implementation, tile, warps,
                            report["resume_key"]], sort_keys=True).encode()).hexdigest()[:16]
                        if case_id in cached:
                            row = {**cached[case_id], "resumed": True}
                            if implementation == "grouped":
                                baseline = row["timing"]
                            write_checkpoint(args.output, report)
                            continue
                        actual = decode(implementation, tile, warps)
                        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
                        storage_dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.uint8
                        if not torch.equal(actual.view(storage_dtype), expected.view(storage_dtype)):
                            raise AssertionError("Reconstruction changed decoded bytes")
                        if actual.stride() != expected.stride():
                            raise AssertionError("Reconstruction changed the native GEMM output alignment")
                        del actual
                        call = lambda decode=decode, implementation=implementation, tile=tile, warps=warps: decode(implementation, tile, warps)
                        measured = timings(call, torch, args)
                        if implementation == "grouped":
                            baseline = measured
                        row = {"case_id": case_id, "shape": dimensions, "dtype": dtype_name, "bits": bits, "group_size": group,
                               "tile": tile, "num_warps": warps,
                               "block": args.block, "implementation": implementation, "exact": True,
                               "kernel_backend": kernel_backend, "schedule": schedule,
                               "timing": measured, "speedup_vs_grouped": {
                                   mode: baseline[mode]["wall_ms_median"] / value["wall_ms_median"]
                                   for mode, value in measured.items()} if schedule["tuning_controls_active"] else {},
                               "packed_payload_and_metadata_bytes": sum(x.numel() * x.element_size()
                                   for x in (packed.data, packed.step, packed.minimum))}
                        if args.profile_dir:
                            args.profile_dir.mkdir(parents=True, exist_ok=True)
                            label = f"{'x'.join(map(str, dimensions))}-{dtype_name}-b{bits}-g{group}-{implementation}-{case_id}"
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA]) as capture:
                                call()
                                torch.cuda.synchronize(args.device)
                            path = args.profile_dir / f"{label}.json"
                            capture.export_chrome_trace(str(path))
                            row["trace"] = str(path)
                        report["results"].append(row)
                        cached[case_id] = row
                        write_checkpoint(args.output, report)
                    del expected, decode, call
                del packed
        del source
    text = json.dumps(report, indent=2, allow_nan=False)
    write_checkpoint(args.output, report)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
