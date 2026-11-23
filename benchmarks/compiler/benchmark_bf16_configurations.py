"""Role-specific BF16 launch configurations and packed dW controls.

The input shape is tokens x input_features x output_features. Each independent
trial runs in a fresh process. CPU execution is only a semantic reference;
candidate timings never install a new default or qualify GPU performance.
Execution uses a non-editable installed DeepOpt distribution and pins that
installation in every worker. A checkout source tree cannot qualify.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
import uuid


BENCHMARKS = Path(__file__).resolve().parents[1]
ROLES = ("forward", "dx", "dw")
REPRESENTATIVE_CONFIGS = (
    (64, 64, 32, 4, 3, "column_major", 1),
    (32, 128, 64, 8, 3, "grouped_m", 8),
    (128, 64, 32, 4, 3, "row_major", 1),
    (64, 128, 64, 8, 4, "grouped_m", 4),
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--shapes", nargs="+", default=["6304x768x3072", "6304x3072x768"])
    result.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    result.add_argument("--configs", choices=("legacy", "representative", "candidates"), default="representative")
    result.add_argument("--max-configurations", type=int, default=4,
                        help="Bound each route's generated configurations, including the legacy control")
    result.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    result.add_argument("--group-size", type=int, default=1024)
    result.add_argument("--trials", type=int, default=3)
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--min-measure-ms", type=float, default=200,
                        help="Target total CUDA event window across repeats; 0 keeps fixed counts. CPU ignores this target")
    result.add_argument("--max-iterations", type=int, default=10000,
                        help="Maximum calls per measured repeat after CUDA calibration")
    result.add_argument("--seed", type=int, default=714)
    result.add_argument("--profile", action="store_true")
    result.add_argument("--panel-decoding", action="store_true", help="Add an explicit bounded panel-decoding dW candidate")
    result.add_argument("--telemetry-interval", type=float, default=0,
                        help="Optional per-worker GPU diagnostics outside timing, at least one second; 0 disables")
    result.add_argument("--timeout", type=float, default=1800,
                        help="Maximum seconds per fresh role/shape/trial worker")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    return result


def parse_shapes(values):
    shapes = []
    for value in values:
        shape = tuple(int(part) for part in value.split("x"))
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("shapes must be positive tokens x input_features x output_features triples")
        shapes.append(shape)
    if len(set(shapes)) != len(shapes):
        raise ValueError("shapes must not contain duplicates")
    return shapes


def role_shapes(shape, role):
    tokens, inputs, outputs = shape
    if role == "forward":
        return (tokens, inputs), (outputs, inputs)
    if role == "dx":
        return (tokens, outputs), (inputs, outputs)
    if role == "dw":
        return (outputs, tokens), (inputs, tokens)
    raise ValueError(f"Unknown role: {role}")


def make_plan(args):
    shapes = parse_shapes(args.shapes)
    if len(set(args.roles)) != len(args.roles):
        raise ValueError("roles must not contain duplicates")
    if min(args.max_configurations, args.group_size, args.trials, args.iterations, args.repeats) < 1:
        raise ValueError("configuration count, group size, trials, iterations and repeats must be positive")
    if args.warmup < 0 or not 0 < args.timeout < float("inf"):
        raise ValueError("warmup must be nonnegative and timeout finite and positive")
    if not math.isfinite(args.min_measure_ms) or args.min_measure_ms < 0:
        raise ValueError("minimum measurement duration must be finite and nonnegative")
    if args.max_iterations < args.iterations:
        raise ValueError("maximum iterations must be at least the requested iterations")
    if args.group_size != 1024:
        raise ValueError("This qualification recipe requires compression group size 1024")
    if not math.isfinite(args.telemetry_interval) or args.telemetry_interval < 0 or 0 < args.telemetry_interval < 1:
        raise ValueError("telemetry interval must be zero or at least one second")
    jobs = []
    for trial in range(args.trials):
        for shape in shapes:
            for role in args.roles:
                a, b = role_shapes(shape, role)
                jobs.append({"id": f"{'x'.join(map(str, shape))}-{role}-t{trial}",
                             "shape_tokens_in_out": shape, "role": role, "a_shape": a, "b_shape": b,
                             "output_shape": (a[0], b[0]), "trial": trial, "seed": args.seed + trial,
                             "routes": (["dense", "materialized", "tiled"] + (["panel"] if args.panel_decoding else [])) if role == "dw" else ["dense"],
                             "device": args.device, "configs": args.configs,
                             "max_configurations": args.max_configurations,
                             "bits": args.bits, "group_size": args.group_size,
                             "measurement": {"warmup": args.warmup, "iterations": args.iterations,
                                             "repeats": args.repeats, "profile": args.profile,
                                             "min_measure_ms": args.min_measure_ms,
                                             "telemetry_interval": args.telemetry_interval,
                                             "max_iterations": args.max_iterations}})
    random.Random(args.seed).shuffle(jobs)
    return {"schema_version": 1, "run_id": uuid.uuid4().hex, "jobs": jobs,
            "fresh_process_per_role_shape_trial": True,
            "routes": {"dense": "BF16 operand",
                       "materialized": "decode the saved X[tokens,input] once per measured dW call, then GEMM",
                       "tiled": "experimental direct packed-read dW; native control decodes the same saved X",
                       "panel": "opt-in bounded K-panel decoding with invocation-local FP32 accumulation"},
            "unimplemented_routes": ["split-k", "persistent", "scratch-reuse"],
            "qualification": "Matching-codec numerical and launch evidence only; no automatic winner promotion",
            "gpu_performance_qualified": False}


def source_identity():
    from compiler.installed_runtime import installed_identity

    installed = installed_identity(root=BENCHMARKS.parent)
    return {"installed_deepopt": installed,
            "deepopt_sources_sha256": installed["source_sha256"],
            "deepopt_import_path": installed["package_path"],
            "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "measurement_sha256": sha256((BENCHMARKS / "analysis/measurement.py").read_bytes()).hexdigest()}


def build_operands(torch, shape, role, device):
    tokens, inputs, outputs = shape
    source = torch.randn(tokens, inputs, device=device, dtype=torch.bfloat16)
    weight = torch.randn(outputs, inputs, device=device, dtype=torch.bfloat16)
    upstream = torch.randn(tokens, outputs, device=device, dtype=torch.bfloat16)
    if role == "forward":
        return source, weight, None
    if role == "dx":
        return upstream, weight.T, None
    # The right matrix is X.T[in,tokens]; pack X[tokens,in], not a newly
    # flattened transposed matrix. Group placement is part of the oracle.
    return upstream.T, source.T, source


def configurations(job, a, b, packed):
    from deepopt.opt.backends._shared.bf16_gemm import (
        BF16GemmConfig, BF16GemmKey, bf16_gemm_candidates, bf16_source_version,
    )

    base = [BF16GemmConfig(*values) for values in REPRESENTATIVE_CONFIGS]
    rejections = []
    if job["configs"] == "legacy":
        base = base[:1]
    elif job["configs"] == "candidates":
        import torch
        from deepopt.kernel_policy import current_kernel_policy

        # CPU candidates describe NVIDIA hypotheses; execution remains a
        # semantic reference and cannot validate any launch configuration.
        capability = torch.cuda.get_device_capability(a.device) if a.device.type == "cuda" else (9, 0)
        try:
            triton_version = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            triton_version = "unavailable"
        compression = (("affine_flat", "source_KN_right_KN", packed.bits, packed.group_size)
                       if packed is not None else ("dense",))
        key = BF16GemmKey(role=job["role"], m=a.shape[0], n=b.shape[0], k=a.shape[1],
            device=str(a.device) if a.device.type == "cuda" else "cuda:semantic-hypothesis",
            capability=tuple(capability), a_dtype=str(a.dtype),
            b_dtype=str(packed.data.dtype if packed is not None else b.dtype),
            a_stride=tuple(a.stride()), b_stride=(0, 0) if packed is not None else tuple(b.stride()),
            compression=compression, epilogue=(None,), deterministic=torch.are_deterministic_algorithms_enabled(),
            precision_policy=("bf16_multiplicands_fp32_accumulation", "fp_fusion_disabled"),
            kernel_policy=current_kernel_policy().fingerprint,
            runtime_version=(str(torch.__version__), torch.version.cuda, triton_version),
            source_version=bf16_source_version(), a_storage_offset=a.storage_offset(),
            b_storage_offset=0 if packed is not None else b.storage_offset())
        candidates = bf16_gemm_candidates(key)
        base = [entry.config for entry in candidates if entry.rejection_reason is None]
        rejections = [{"config": asdict(entry.config), "reason": entry.rejection_reason}
                      for entry in candidates if entry.rejection_reason]
        if not base:
            raise ValueError(f"No eligible BF16 configurations: {rejections}")
    return base[:job["max_configurations"]], rejections


def measure_role(call, torch, device, settings, measure):
    """Calibrate stateless GEMM calls without synchronizing each invocation.

    Calibration is separate from the reported repeats. The shared timer
    enqueues each repeat as one batch and synchronizes at its boundary.
    CUDA event intervals include host submission gaps, so the duration target
    is a stream window rather than an active-GPU utilization measurement.
    """
    requested = settings["iterations"]
    repeats = settings["repeats"]
    target = settings.get("min_measure_ms", 0)
    maximum = settings.get("max_iterations", max(10000, requested))
    iterations, warmup = requested, settings["warmup"]
    calibration = None
    required = requested
    if device.type == "cuda" and target > 0:
        calibration = measure(call, torch, device, warmup, requested, 1)
        cost = calibration["cuda_stream_ms"]["median"]
        if not math.isfinite(cost) or cost <= 0:
            raise RuntimeError("CUDA calibration must report a finite positive event duration")
        required = (maximum + 1 if target > cost * repeats * maximum
                    else max(requested, math.ceil(target / (cost * repeats))))
        iterations, warmup = min(maximum, required), 0
    attempts = []
    while True:
        timing = measure(call, torch, device, warmup, iterations, repeats)
        window = None
        if device.type == "cuda" and target > 0:
            samples = (timing.get("cuda_stream_ms") or {}).get("samples", [])
            if (len(samples) != repeats or any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0 for value in samples)):
                raise RuntimeError("CUDA measurement must report finite positive event samples for every repeat")
            window = sum(samples) * iterations
            if window < target and iterations < maximum:
                attempts.append({"iterations_per_repeat": iterations, "repeats": repeats,
                                 "cuda_event_total_ms": window, "included_in_reported_samples": False})
                required = (maximum + 1 if target > window * (maximum / iterations) else
                            max(iterations + 1, math.ceil(iterations * max(1.25, target / window * 1.05))))
                iterations, warmup = min(maximum, required), 0
                continue
        elif device.type == "cuda" and timing.get("cuda_stream_ms") is not None:
            window = sum(timing["cuda_stream_ms"]["samples"]) * iterations
        break
    timing["sampling"] = {
        "mode": "calibrated_cuda_event_window" if calibration is not None else "fixed_iterations",
        "requested_iterations_per_repeat": requested,
        "actual_iterations_per_repeat": iterations,
        "repeats": repeats,
        "max_iterations_per_repeat": maximum,
        "target_cuda_event_total_ms": target if device.type == "cuda" else None,
        "actual_cuda_event_total_ms": window,
        "target_reached": window >= target if window is not None and target > 0 else None,
        "iteration_cap_reached": calibration is not None and iterations >= maximum,
        "calibration_calls": requested if calibration is not None else 0,
        "calibration_cuda_event_total_ms": (calibration["cuda_stream_ms"]["median"] * requested
                                            if calibration is not None else None),
        "calibration_in_reported_samples": False,
        "discarded_window_attempts": attempts,
        "additional_calibration_calls": sum(attempt["iterations_per_repeat"] * repeats for attempt in attempts),
        "synchronization": "repeat_boundaries_only",
        "duration_scope": "CUDA stream event window including host submission gaps; not active GPU time",
    }
    return timing


def execute_worker(job, directory):
    from compiler.installed_runtime import identity_session
    from analysis.telemetry import UtilizationTelemetry

    with identity_session(BENCHMARKS.parent), UtilizationTelemetry(job.get("device", "cpu"),
            interval=job.get("measurement", {}).get("telemetry_interval", 0)) as telemetry:
        result = _execute_worker(job, directory)
    result["telemetry"] = telemetry.report()
    return result


def _execute_worker(job, directory):
    # Validate the imported distribution before executing any numerical work.
    # A fresh worker must use the same installed wheel as its supervisor.
    before = source_identity()
    expected = job.get("installed_deepopt")
    if expected is not None and before.get("installed_deepopt") != expected:
        raise RuntimeError("Worker DeepOpt installation differs from its supervisor")
    import torch
    from deepopt.kernel_policy import kernel_policy_scope
    from deepopt.opt.backends.bf16 import product
    from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
    from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
    from analysis.measurement import measure, synchronize, state_fingerprint

    device = torch.device(job["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Only CUDA and CPU semantic-reference devices are supported")
    torch.manual_seed(job["seed"])
    a, b, source = build_operands(torch, job["shape_tokens_in_out"], job["role"], device)
    fingerprint = state_fingerprint(torch.nn.Module(), (a, b), {})
    records = []
    with kernel_policy_scope("triton"), torch.no_grad():
        packed = (quantize_pack_activation(source, job["group_size"], bits=job["bits"])
                  if source is not None else None)
        for route in job["routes"]:
            is_compressed = route != "dense"
            selected, rejections = configurations(job, a, b, packed if route == "tiled" else None)
            variants = ([] if route == "panel" else [("native", None)]) + [("triton", config) for config in selected]
            random.Random(job["seed"] + len(route)).shuffle(variants)
            # Both packed-read and materialized controls use the exact same
            # codec payload and orientation. Setup pack cost is reported below.
            decoded = (dequantize_unpack_activation(packed, dtype=torch.bfloat16, transpose=True)
                       if is_compressed else b)
            expected = product(a, decoded, backend="native", role=job["role"])
            del decoded
            for index, (backend, config) in enumerate(variants):
                identity = {"route": route, "backend": backend, "config": asdict(config) if config else None,
                            "role": job["role"], "candidate_rejections": rejections}

                def call(backend=backend, config=config):
                    operand = b
                    if route == "materialized":
                        operand = dequantize_unpack_activation(packed, dtype=torch.bfloat16, transpose=True)
                    route_options = {"packed_route": "panel", "panel_size": 1024} if route == "panel" else {}
                    return product(a, None if route in ("tiled", "panel") else operand, backend=backend,
                                   packed_b=packed if route in ("tiled", "panel") else None,
                                   kernel_config=config, role=job["role"], **route_options)

                try:
                    synchronize(torch, device)
                    began = perf_counter()
                    actual = call()
                    synchronize(torch, device)
                    first_ms = (perf_counter() - began) * 1000
                    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                    error = (actual.float() - expected.float()).abs().max().item()
                    if not bool(torch.isfinite(actual).all()):
                        raise AssertionError("Nonfinite BF16 result")
                    del actual
                    settings = job["measurement"]
                    timing = measure_role(call, torch, device, settings, measure)
                    synchronize(torch, device)
                    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
                    if baseline is not None:
                        torch.cuda.reset_peak_memory_stats(device)
                    actual = call()
                    synchronize(torch, device)
                    memory = {"baseline_allocated_bytes": baseline,
                              "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline if baseline is not None else None,
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if baseline is not None else None,
                              "opaque_workspace_bytes": None, "decoded_scratch_intervals": None}
                    del actual
                    # Resource/launch recording is intentionally outside timing.
                    with record_gemms() as diagnostics:
                        observed = call()
                    del observed
                    launch = diagnostics[-1].get("kernel_launch") if diagnostics else None
                    generated = bool(launch and diagnostics[-1].get("execution") == "generated")
                    if device.type == "cuda" and backend == "triton" and not generated:
                        raise AssertionError("Requested Triton kernel has no generated launch evidence")
                    profile = None
                    if settings["profile"]:
                        activities = [torch.profiler.ProfilerActivity.CPU]
                        if device.type == "cuda":
                            activities.append(torch.profiler.ProfilerActivity.CUDA)
                        with torch.profiler.profile(activities=activities) as capture:
                            call()
                            synchronize(torch, device)
                        trace = directory / f"{route}-{index}-{backend}-trace.json"
                        capture.export_chrome_trace(str(trace))
                        kernels = [event for event in capture.events() if str(event.device_type).endswith("CUDA")]
                        profile = {"trace": str(trace), "cuda_kernel_count": len(kernels),
                                   "kernels": [{"name": event.name, "duration_us": event.time_range.elapsed_us()}
                                               for event in kernels]}
                    records.append({**identity, "status": "ok", "correctness": "pass", "max_abs_error": error,
                                    "first_measured_call_ms": first_ms, "timing": timing, "memory": memory,
                                    "diagnostics": diagnostics, "profile": profile,
                                    "execution": "cuda" if device.type == "cuda" else "cpu_semantic_reference",
                                    "generated_launch_observed": generated, "gpu_performance_qualified": False})
                except Exception as error:
                    actual = None
                    del actual
                    records.append({**identity, "status": "error", "correctness": "unqualified",
                                    "error": f"{type(error).__name__}: {error}",
                                    "resource_rejection": getattr(error, "evidence", None), "gpu_performance_qualified": False})
            del expected
    after = source_identity()
    versions = {}
    for name in ("deep-optimization", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    passed = before == after and bool(records) and all(row["status"] == "ok" for row in records)
    return {"status": "ok" if passed else "failed", "job": job, "source_before": before, "source_after": after,
            "installed_deepopt": before.get("installed_deepopt"),
            "source_unchanged": before == after, "initial_operands_sha256": fingerprint,
            "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "packages": versions,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "results": records, "packed_source_shape": tuple(source.shape) if source is not None else None,
            "packing_in_timed_region": False,
            "scope": "GEMM pipeline: materialized decoding is timed, source packing is setup. Not a full optimizer update.",
            "gpu_performance_qualified": False}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    sys.path.insert(0, str(BENCHMARKS))
    if args._worker is not None:
        job = json.loads(args._worker.read_text())
        try:
            result = execute_worker(job, args.output.parent)
        except Exception as error:
            result = {"status": "error", "job": job, "results": [], "error": f"{type(error).__name__}: {error}"}
        write_json(args.output, result)
        return int(result["status"] != "ok")
    try:
        plan = make_plan(args)
    except (ValueError, TypeError) as error:
        arguments.error(str(error))
    if args.dry_run:
        write_json(args.output, {**plan, "status": "dry_run", "results": []})
        return 0
    try:
        from compiler.installed_runtime import child_environment, installed_identity, prepare_imports

        prepare_imports(root=BENCHMARKS.parent)
        installed = installed_identity(root=BENCHMARKS.parent)
        environment = child_environment(root=BENCHMARKS.parent, identity=installed)
    except Exception as error:
        write_json(args.output, {**plan, "status": "error", "passed": False, "results": [],
                                 "error": f"{type(error).__name__}: {error}"})
        return 1
    if args.device.startswith("cuda"):
        from _gpu_lease import ensure

        leased = ensure(__file__, argv, args.device)
        if leased is not None:
            return leased
    directory = args.output.resolve().with_suffix("").with_name(args.output.stem + "-runs") / plan["run_id"]
    report = {**plan, "status": "running", "installed_deepopt": installed, "results": []}
    write_json(args.output, report)
    for job in plan["jobs"]:
        job["installed_deepopt"] = installed
        child = directory / job["id"]
        input_path, output_path = child / "job.json", child / "result.json"
        write_json(input_path, job)
        try:
            with (child / "process.log").open("w") as log:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(input_path),
                                          "--output", str(output_path)], stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout, check=False, env=environment)
            result = json.loads(output_path.read_text())
            if process.returncode and result.get("status") == "ok":
                result.update(status="failed", error=f"Worker exited {process.returncode}")
            if result.get("status") == "ok" and any(
                    result.get(key, {}).get("installed_deepopt") != installed
                    for key in ("source_before", "source_after")):
                result.update(status="failed", error="Worker DeepOpt installation differs from its supervisor")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result = {"status": "error", "job": job, "results": [], "error": f"{type(error).__name__}: {error}"}
        report["results"].append({**result, "result_path": str(output_path), "log": str(child / "process.log")})
        write_json(args.output, report)
    passed = bool(report["results"]) and all(row["status"] == "ok" for row in report["results"])
    report.update(status="completed" if passed else "failed", passed=passed)
    write_json(args.output, report)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
