"""Compare grouped and independent-group batched packing on identical inputs.

Keep benchmark_activation_unpack.py beside this file for shared timing and
atomic checkpoint helpers. Only the installed DeepOpt library is imported.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from functools import partial
import importlib.util
import json
import os
import platform
from pathlib import Path
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", nargs="+", default=["100864x3072", "100864x768"])
    p.add_argument("--source-dtypes", nargs="+", choices=("bf16", "e4m3fn", "e5m2"), default=["bf16", "e4m3fn"])
    p.add_argument("--bits", nargs="+", type=int, choices=(1, 2, 4, 8), default=[2])
    p.add_argument("--group-sizes", nargs="+", type=int, default=[1024])
    p.add_argument("--groups-per-program", nargs="+", type=int, choices=(1, 2, 4, 8), default=[4, 8])
    p.add_argument("--variants", nargs="+", choices=("batched", "serial"), default=["batched", "serial"])
    p.add_argument("--num-warps", nargs="+", type=lambda v: "auto" if v == "auto" else int(v),
                   choices=("auto",1,2,4,8), default=[4])
    p.add_argument("--deepopt-backend", choices=("triton", "tilelang", "mixed"), default="triton")
    p.add_argument("--deepopt-kernel-policy", type=json.loads, default={}, metavar="JSON",
                   help="runtime.kernels mapping; route overrides require --deepopt-backend mixed")
    p.add_argument("--execution", choices=("eager", "graph", "both"), default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--time-budget", type=float, default=2.0, help="Warm timing target per variant in seconds; 0 uses exhaustive counts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--profile-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p


def cached_results(path, key, args, helper):
    if not args.resume or not path.exists():
        return {}
    try:
        previous = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    if previous.get("resume_key") != key:
        return {}
    return {r["case_id"]: r for r in previous.get("results", [])
            if "case_id" in r and helper.reusable_row(r, args)}


def helpers():
    path = Path(__file__).with_name("benchmark_activation_unpack.py")
    spec = importlib.util.spec_from_file_location("activation_pack_measurement", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resume_identity(args, deepopt, torch, helper_path, script_path=None, *, return_metadata=False, kernel_backend=None):
    digest = sha256()
    root = Path(deepopt.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
        driver = query.stdout.strip() if query.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        driver = ""
    if kernel_backend is None:
        from deepopt._benchmarking import BenchmarkBackend
        kernel_backend = BenchmarkBackend.from_args(args).metadata(args.device)
    identity = {"build": digest.hexdigest(), "script": sha256(Path(script_path or __file__).read_bytes()).hexdigest(),
                "identity_helper": sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": sys.version, "platform": platform.platform(),
                "helper": sha256(helper_path.read_bytes()).hexdigest(), "torch": str(torch.__version__),
                "triton": kernel_backend["packages"].get("triton"), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "kernel_backend": kernel_backend, "backend_sources": helpers().backend_sources(kernel_backend["packages"]),
                "allocator_environment": {key: os.environ.get(key) for key in
                                          ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
                "math_environment": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "TRITON_DEFAULT_FP_FUSION",
                                     "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUDA_LAUNCH_BLOCKING")},
                "driver": driver or os.urandom(16).hex(),
                "gpu": str(torch.cuda.get_device_properties(args.device)),
                "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                            if k not in ("resume", "output")}}
    return identity if return_metadata else sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


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
    h = helpers()
    try:
        args.shapes = [h.shape(value) for value in args.shapes]
    except argparse.ArgumentTypeError as error:
        p.error(str(error))
    if min(args.iterations, args.repeats, *args.group_sizes) < 1 or args.warmup < 0:
        p.error("iterations/repeats/groups must be positive; warmup may be zero")
    if max(args.group_sizes) > 2048:
        p.error("batched packing currently supports group sizes at most 2048")
    if not args.device.startswith("cuda"):
        p.error("packing kernel measurements require CUDA")
    from deepopt._benchmarking import BenchmarkBackend, timing_plan_from_cost
    try:
        selection = BenchmarkBackend.from_args(args)
        timing_plan_from_cost(args, 0.0)
    except (TypeError, ValueError) as error:
        p.error(str(error))
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    with selection.scope():
        return run_selected(args, h, selection)


def run_selected(args, h, selection):
    import torch
    import deepopt
    if not torch.cuda.is_available() or torch.version.hip:
        raise SystemExit("An NVIDIA CUDA device is required")
    selected_codec = h.codec_module()
    quantize_pack_affine = selected_codec.quantize_pack_affine
    from deepopt.opt.quantization.quantizer import quantize_fp8

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    kernel_backend = selection.metadata(args.device)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "configuration": {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "results": [],
        "kernel_backend": kernel_backend,
        "notes": ["Each implementation receives identical source values and independent compression groups.",
                  "Payload, step, and minimum must match the grouped control exactly before timing.",
                  "TileLang retains a semantic grouped control with its actual serial per-group schedule; inactive tuning controls are recorded and deduplicated.",
                  "Packing is isolated; full-update latency and graph-pool memory require the operator/model suite."]}
    identity_metadata = resume_identity(args, deepopt, torch, Path(h.__file__), return_metadata=True,
                                        kernel_backend=kernel_backend)
    report["identity"] = identity_metadata
    report["resume_key"] = sha256(json.dumps(identity_metadata, sort_keys=True).encode()).hexdigest()
    report["environment"] = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": str(torch.cuda.get_device_properties(args.device)), "deepopt_path": str(deepopt.__file__),
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "allocator_environment": {key: os.environ.get(key) for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}}
    library_digest = sha256()
    library_root = Path(deepopt.__file__).resolve().parent
    for source in sorted(library_root.rglob("*.py")):
        library_digest.update(source.relative_to(library_root).as_posix().encode() + source.read_bytes())
    report["environment"]["deepopt_source_sha256"] = library_digest.hexdigest()
    cached = cached_results(args.output, report["resume_key"], args, h)
    report["results"] = list(cached.values())
    dtypes = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    schedule = h.codec_schedule(selected_codec, "pack", {
        "implementations": ["grouped", *args.variants], "groups_per_program": args.groups_per_program,
        "num_warps": args.num_warps})
    for shape in args.shapes:
        original = torch.randn(shape, device=args.device, dtype=torch.bfloat16)
        for source_dtype in args.source_dtypes:
            value = original if source_dtype == "bf16" else quantize_fp8(original, fp8_dtype=dtypes[source_dtype], check_finite=False).data
            for bits in args.bits:
                for group in args.group_sizes:
                    call = None
                    expected = quantize_pack_affine(value, group, bits=bits, implementation="grouped")
                    choices = h.codec_choices(args, "pack", schedule)
                    for implementation, batch, warps in choices:
                        key = {"shape": shape, "dtype": source_dtype, "bits": bits, "group": group,
                               "implementation": implementation, "groups_per_program": batch, "num_warps": warps}
                        case_id = sha256(json.dumps({"case": key, "resume_key": report["resume_key"]},
                                                   sort_keys=True).encode()).hexdigest()[:16]
                        if case_id in cached:
                            continue
                        call = partial(quantize_pack_affine, value, group, bits=bits,
                                       implementation=implementation, groups_per_program=batch, num_warps=warps)
                        actual = call()
                        if not all(torch.equal(a, b) for a, b in zip(actual, expected)):
                            raise AssertionError(f"Batched packing changed the represented payload: {key}")
                        del actual
                        row = {"case_id": case_id, **key, "exact": True, "kernel_backend": kernel_backend,
                               "schedule": schedule, "timing": h.timings(call, torch, args)}
                        if args.profile_dir is not None:
                            args.profile_dir.mkdir(parents=True, exist_ok=True)
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA]) as capture:
                                call()
                                torch.cuda.synchronize(args.device)
                            trace = args.profile_dir / f"{case_id}.json"
                            capture.export_chrome_trace(str(trace))
                            row["trace"] = str(trace.resolve())
                        report["results"].append(row)
                        h.write_checkpoint(args.output, report)
                        print(json.dumps({"case_id": case_id, **key, "timing": row["timing"]}), flush=True)
                    call = None
                    del expected
    h.write_checkpoint(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
