"""Standalone eager FP8 Linear benchmark: installed DeepOpt, torchao, and TE.

Copy this file anywhere and run with the GPU environment's Python. Example:
    python benchmark_fp8_eager.py --shapes 4096x768x3072 --output fp8-eager.json

Shapes are MxKxN: input [M, K], weight [N, K], output [M, N]. All dimensions
must be multiples of 16. No source path changes, torch.compile, or CUDA Graphs.
"""

import argparse
import cProfile
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import io
import inspect
import json
import math
from pathlib import Path
import pstats
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter
import traceback
from unittest.mock import patch
from uuid import uuid4
import zipfile


IMPLEMENTATIONS = ("bf16", "deepopt", "torchao", "te")
MODES = ("forward", "forward-backward")
DEFAULT_SHAPES = [(4096, 768, 2304), (4096, 768, 768),
                  (4096, 768, 3072), (4096, 3072, 768)]


def _deepopt_linear_module():
    canonical = "deepopt.opt.nn.autograd.linear"
    try:
        return import_module(canonical)
    except ModuleNotFoundError as error:
        if error.name is None or not (canonical == error.name or canonical.startswith(error.name + ".")):
            raise
        return import_module("deepopt.opt.do_nn.AOTAutograd_Func.linear_autograd_func")
RESULT_PREFIX = "FP8_EAGER_RESULT="
NOTES = [
    "Eager execution, tensorwise current/dynamic FP8 scaling, FP32 parameters, BF16 AMP and BF16 inputs/outputs.",
    "Forward runs without autograd. Forward-backward computes dX, dW and optional db, including zero_grad(set_to_none=True).",
    "Quantization, layout conversion and per-call context overhead are timed; initialization and warmup are excluded.",
    "Autocast/DeepOpt weight caching is disabled; TE uses its default uncached weight path. No optimizer update is timed.",
    "CUDA stream intervals include host submission gaps. Wall timing includes the final device synchronization.",
    "Each library retains its accumulation/rounding defaults; matching FP8 formats does not imply identical numerics.",
    "Numerical errors against FP32 are measured separately and are not a convergence test.",
    "Optional CUDA/Python profiles run AFTER throughput timing. Instrumented durations are diagnostic, not throughput results.",
    "CUDA activity sums exclude host submission gaps and user annotation events; overlapping CPU/GPU or nested scope times must not be added.",
    "DeepOpt quantization/GEMM scopes include their internal layout work. Other libraries expose their native kernel names in the trace.",
    "Python cProfile covers the calling thread and may include blocking native calls; autograd worker threads are not covered.",
    "DeepOpt autotuning is disabled by default for a fixed-schedule baseline; --autotune requests tuning where the selected backend implements it.",
    "The time budget is a soft target per warm timing measurement, including warmup. A GPU call cannot be interrupted; cold setup/first launch are recorded separately.",
    "Budgeted measurements retain actual warmup, iteration and repeat counts; inspect those counts when comparing runs.",
]


def parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError:
        shape = ()
    if len(shape) != 3 or any(size <= 0 or size % 16 for size in shape):
        raise argparse.ArgumentTypeError("use MxKxN with three positive multiples of 16, e.g. 4096x768x3072")
    return shape


def parse_kernel_policy(value):
    try:
        policy = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("kernel policy must be a JSON object") from error
    if not isinstance(policy, dict):
        raise argparse.ArgumentTypeError("kernel policy must be a JSON object")
    return policy


def deepopt_kernel_configuration(args):
    """Resolve new backend controls without changing the legacy default path."""
    mapping = getattr(args, "deepopt_kernel_policy", None)
    if args.deepopt_backend in ("triton", "reference") and mapping is None:
        # Keep the standalone default compatible with previously installed
        # DeepOpt releases that predate runtime.kernels.
        return None
    from deepopt.kernel_policy import resolve_kernel_policy

    policy = dict(mapping or {})
    if "backend" in policy and policy["backend"] != args.deepopt_backend:
        raise ValueError("--deepopt-kernel-policy backend must match --deepopt-backend")
    policy["backend"] = args.deepopt_backend
    return resolve_kernel_policy(policy).as_dict()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS))
    result.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    result.add_argument("--shapes", nargs="+", type=parse_shape, default=DEFAULT_SHAPES, metavar="MxKxN")
    result.add_argument("--format", choices=("e4m3fn", "hybrid"), default="e4m3fn",
                        help="hybrid uses E4M3 inputs/weights and E5M2 upstream gradients in all three FP8 libraries")
    result.add_argument("--deepopt-backend", choices=("triton", "tilelang", "mixed"), default="triton")
    result.add_argument("--deepopt-kernel-policy", type=parse_kernel_policy, metavar="JSON",
                        help='runtime.kernels options, e.g. --deepopt-backend mixed '
                             '--deepopt-kernel-policy \'{"preferred":"tilelang","roles":{"linear.dw":"triton"}}\'')
    result.add_argument("--deepopt-backward-backend", choices=("inherit", "native_fast", "native_precise"),
                        default="inherit", help="keep the selected forward/quantization backend and select native backward GEMMs")
    result.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=False,
                        help="opt into DeepOpt autotuning; disabled by default for a fixed-schedule baseline")
    result.add_argument("--time-budget", type=float, default=2.0, metavar="SECONDS",
                        help="soft wall-time budget per timing measurement including warmup; 0 runs all requested samples")
    result.add_argument("--qualify-tilelang", action="store_true", help="Opt into independent executor qualification and robust timing calibration")
    result.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output", type=Path, help="save timings, errors, configurations and installed package paths as JSON")
    result.add_argument("--profile", action="store_true", help="collect separate CUDA/Python profiles and bundle this run into a ZIP")
    result.add_argument("--profile-steps", type=int, default=5, help="active steps in each separate diagnostic pass")
    result.add_argument("--profile-dir", type=Path, help="parent directory for this run's unique profile directory")
    result.add_argument("--_worker", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    result.add_argument("--_checkpoint", type=Path, help=argparse.SUPPRESS)
    return result


def package_info(name, distribution):
    module = import_module(name)
    try:
        installed_version = version(distribution)
    except PackageNotFoundError:
        installed_version = getattr(module, "__version__", None)
    return {"version": str(installed_version), "path": str(Path(module.__file__).resolve())}


class LinearCase:
    def __init__(self, args, implementation, shape, mode, *, device=None):
        import torch

        self.torch = torch
        self.implementation = implementation
        self.training = mode == "forward-backward"
        device = torch.device(args.device if device is None else device)
        m, k, n = shape
        # Reset before EVERY case, before constructors of optional libraries can
        # consume random numbers. All libraries receive exactly the same data.
        torch.manual_seed(args.seed)
        self.source = torch.nn.Linear(k, n, bias=args.bias, device=device, dtype=torch.float32)
        self.x = torch.randn(m, k, device=device, dtype=torch.bfloat16, requires_grad=self.training)
        self.grad = torch.randn(m, n, device=device, dtype=torch.bfloat16) / m if self.training else None
        self.model = self.source
        self.configuration = None
        self.kernel_policy = None
        self.amp = torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False)
        self.fp8_context = nullcontext

        if implementation == "deepopt":
            import deepopt
            from deepopt.opt.policy import prepare_fp8_model

            self.kernel_policy = deepopt_kernel_configuration(args)
            execution = ({"config": {"runtime": {"kernels": self.kernel_policy}}}
                         if self.kernel_policy is not None else {})
            options = dict(
                enabled=True, backend="inherit" if args.deepopt_backend == "mixed" else args.deepopt_backend, trace_mode="modules",
                intercept_linear=False, scaling_recipe="current", out_dtype="bf16",
                grad_dtype="e5m2" if args.format == "hybrid" else "e4m3fn",
                backward_backend=args.deepopt_backward_backend,
            )
            if "autotune" in inspect.signature(deepopt.FP8Config).parameters:
                options["autotune"] = getattr(args, "autotune", False)
            elif device.type != "cpu" or args.deepopt_backend != "reference":
                raise RuntimeError("Installed DeepOpt does not expose FP8Config.autotune; upgrade DeepOpt "
                                   "to honor --autotune/--no-autotune in GPU comparisons")
            settings = deepopt.FP8Config(**options)
            self.model, report = prepare_fp8_model(self.source, settings, **execution)
            if not report.converted_modules:
                raise RuntimeError("DeepOpt did not convert the requested Linear to FP8")
            self.configuration = asdict(settings)
            if self.kernel_policy is not None:
                self.configuration["kernel_policy"] = self.kernel_policy
            self.amp = deepopt.autocast(
                device.type, dtype=torch.bfloat16, fp8=settings, cache_enabled=False, check_finite=False, **execution)
        elif implementation == "torchao":
            from torchao.float8 import CastConfig, Float8LinearConfig, convert_to_float8_training

            configuration = Float8LinearConfig(
                cast_config_input=CastConfig(target_dtype=torch.float8_e4m3fn),
                cast_config_weight=CastConfig(target_dtype=torch.float8_e4m3fn),
                cast_config_grad_output=CastConfig(
                    target_dtype=torch.float8_e5m2 if args.format == "hybrid" else torch.float8_e4m3fn),
                emulate=False,
            )
            self.model = convert_to_float8_training(
                torch.nn.Sequential(self.source), config=configuration,
                module_filter_fn=lambda module, fqn: isinstance(module, torch.nn.Linear),
            )[0]
            if not type(self.model).__module__.startswith("torchao."):
                raise RuntimeError("torchao did not convert the requested Linear to FP8")
            self.configuration = repr(configuration)
        elif implementation == "te":
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            settings = recipe.Float8CurrentScaling(
                fp8_format=recipe.Format.HYBRID if args.format == "hybrid" else recipe.Format.E4M3,
                use_power_2_scales=False,
            )
            self.model = te.Linear(k, n, bias=args.bias, params_dtype=torch.float32,
                                   device=device, fuse_wgrad_accumulation=False)
            self.model.weight = self.source.weight
            if args.bias:
                self.model.bias = self.source.bias
            self.fp8_context = lambda: te.fp8_autocast(enabled=True, fp8_recipe=settings)
            self.configuration = repr(settings)
        elif implementation != "bf16":
            raise ValueError(f"Unknown implementation: {implementation}")

        if {id(p) for p in self.model.parameters()} != {id(p) for p in self.source.parameters()}:
            raise RuntimeError("FP8 conversion changed the original FP32 parameter ownership")
        self.model.train(self.training)

    def step(self):
        if self.training:
            self.model.zero_grad(set_to_none=True)
            self.x.grad = None
        with self.torch.set_grad_enabled(self.training):
            with self.amp, self.fp8_context():
                output = self.model(self.x)
            if self.training:
                output.backward(self.grad)
        return output

    def check(self):
        torch = self.torch
        output = self.step()
        with torch.no_grad():
            errors = {"output": error_metrics(
                output, torch.nn.functional.linear(self.x.float(), self.source.weight,
                                                    self.source.bias), torch)}
            if self.training:
                errors["input_gradient"] = error_metrics(self.x.grad, self.grad.float() @ self.source.weight, torch)
                errors["weight_gradient"] = error_metrics(self.source.weight.grad, self.grad.float().T @ self.x.float(), torch)
                if self.source.bias is not None:
                    errors["bias_gradient"] = error_metrics(self.source.bias.grad, self.grad.float().sum(dim=0), torch)
        return {"output_dtype": str(output.dtype), "errors_vs_fp32": errors,
                "gradient_dtypes": {"input": str(self.x.grad.dtype) if self.x.grad is not None else None,
                                    "weight": str(self.source.weight.grad.dtype) if self.source.weight.grad is not None else None}}


def error_metrics(actual, expected, torch):
    if actual is None:
        raise RuntimeError("Requested gradient was not computed")
    finite = bool(torch.isfinite(actual).all().item())
    if not finite:
        return {"finite": False, "relative_l2": None, "max_abs": None}
    difference = actual.detach().float() - expected
    return {"finite": True, "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-30)).item(),
            "max_abs": difference.abs().max().item()}


def measure(operation, args, torch):
    if getattr(args, "qualify_tilelang", False):
        return qualified_measure(operation, args, torch)
    budget = getattr(args, "time_budget", 2.0)
    measurement_started = perf_counter()
    deadline = measurement_started + budget if budget else float("inf")
    warmup_deadline = measurement_started + budget / 4 if budget else float("inf")
    actual_warmup = 0
    estimate = None
    for _ in range(args.warmup):
        if actual_warmup and perf_counter() >= warmup_deadline:
            break
        started = perf_counter()
        operation()
        actual_warmup += 1
        if budget:
            # Warmup is outside throughput timing. Synchronizing each warmup
            # prevents a long asynchronous backlog from bypassing the budget.
            torch.cuda.synchronize()
            estimate = max(perf_counter() - started, 1e-9)
    torch.cuda.synchronize()
    # Initialize events before timing so their lazy creation is not measured.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()
    wall, cuda, counts = [], [], []
    for repeat in range(args.repeats):
        remaining = deadline - perf_counter()
        if counts and remaining <= 0:
            break
        count = args.iterations
        if budget:
            # Reserve time for remaining repeats where possible. At least one
            # timed call is retained even when a single operation exceeds the
            # soft target; it must finish before CUDA can be synchronized.
            allowance = max(0., remaining) / (args.repeats - repeat)
            count = max(1, min(count, int(allowance / estimate))) if estimate else 1
        torch.cuda.synchronize()
        started = perf_counter()
        start.record()
        for _ in range(count):
            operation()
        end.record()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        wall.append(elapsed * 1000 / count)
        cuda.append(start.elapsed_time(end) / count)
        counts.append(count)
        estimate = max(elapsed / count, 1e-9)
    elapsed = perf_counter() - measurement_started
    truncated = (actual_warmup < args.warmup or len(counts) < args.repeats
                 or any(count < args.iterations for count in counts))
    return {"wall_ms": {"median": median(wall), "samples": wall},
            "cuda_stream_ms": {"median": median(cuda), "samples": cuda},
            "sampling": {"requested_warmup": args.warmup, "requested_iterations": args.iterations,
                         "requested_repeats": args.repeats, "actual_warmup": actual_warmup,
                         "actual_iterations_per_repeat": counts, "actual_repeats": len(counts),
                         "time_budget_seconds": budget, "budget_exhausted": bool(budget and (truncated or elapsed >= budget)),
                         "elapsed_seconds": elapsed}}


def qualified_measure(operation, args, torch):
    from deepopt._tilelang_benchmarking import qualified_timing_plan

    began = perf_counter()
    plan = qualified_timing_plan(operation, args, synchronize=torch.cuda.synchronize)
    for _ in range(plan.warmup):
        operation()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); torch.cuda.synchronize()
    wall, cuda = [], []
    for _ in range(plan.repeats):
        torch.cuda.synchronize()
        started = perf_counter()
        start.record()
        for _ in range(plan.iterations):
            operation()
        end.record(); end.synchronize()
        wall.append((perf_counter() - started) * 1000 / plan.iterations)
        cuda.append(start.elapsed_time(end) / plan.iterations)
    peak = torch.cuda.max_memory_allocated()
    return {"wall_ms": {"median": median(wall), "samples": wall},
            "cuda_stream_ms": {"median": median(cuda), "samples": cuda},
            "sampling": plan.sampling(perf_counter() - began),
            "memory": {"baseline_allocated_bytes": baseline, "peak_allocated_bytes": peak,
                       "incremental_peak_allocated_bytes": max(0, peak - baseline)}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def profiled_step(case):
    """Annotated equivalent of step(); never used for throughput or cProfile."""
    torch = case.torch
    if case.training:
        with torch.profiler.record_function("linear.zero_grad"):
            case.model.zero_grad(set_to_none=True)
            case.x.grad = None
    with torch.set_grad_enabled(case.training):
        with torch.profiler.record_function("linear.forward"), case.amp, case.fp8_context():
            output = case.model(case.x)
        if case.training:
            with torch.profiler.record_function("linear.backward"):
                output.backward(case.grad)
    return output


@contextmanager
def deepopt_ranges(case):
    """Temporary ranges on installed-library aliases, with no tensor readback."""
    manifest = {"status": "not_applicable", "first_step_gemms": [], "first_step_quantization": []}
    if case.implementation != "deepopt":
        yield manifest
        return
    module = _deepopt_linear_module()
    gemm_roles = ("forward", "input_gradient", "weight_gradient") if case.training else ("forward",)
    quant_roles = ("activation", "weight", "grad") if case.training else ("activation", "weight")
    counts = {"gemm": 0, "quantize": 0}

    def tensor_info(tensor):
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype)}

    def wrapper(function, kind, name):
        def call(*args, **kwargs):
            index = counts[kind]
            counts[kind] += 1
            roles = gemm_roles if kind == "gemm" else quant_roles
            role = roles[index % len(roles)]
            if index < len(roles):
                if kind == "gemm":
                    left, right = args[:2]
                    manifest["first_step_gemms"].append({
                        "role": role, "a": tensor_info(left.data), "b_nt": tensor_info(right.data),
                        "requested_backend": kwargs.get("backend"), "out_dtype": str(kwargs.get("out_dtype")),
                    })
                else:
                    manifest["first_step_quantization"].append({
                        "role": role, "function": name, "input": tensor_info(args[0]),
                    })
            with case.torch.profiler.record_function(f"deepopt.{kind}.{role}"):
                return function(*args, **kwargs)
        return call

    with ExitStack() as stack:
        for name, kind in (("gemm_nt", "gemm"), ("quantize_fp8", "quantize"),
                           ("_quantize_tensorwise_pair", "quantize"), ("_quantize_tensorwise_transpose", "quantize")):
            function = getattr(module, name)
            stack.enter_context(patch.object(module, name, wrapper(function, kind, name)))
        manifest["status"] = "ok"
        yield manifest


def profile_summary(profiler, torch, steps):
    cpu, scopes, cuda = [], [], {}
    events = list(profiler.events())
    # Follow actual CUDA activities back to their CPU owner. CPU range device
    # totals may also contain mirrored GPU annotations on some profiler builds.
    # Never add those annotation durations to the real kernel/copy durations.
    owners = {event.id: event for event in events if event.device_type == torch.autograd.DeviceType.CPU
              and not getattr(event, "is_async", False) and not getattr(event, "linked_correlation_id", 0)
              and getattr(event, "external_id", 1) != 0}
    attributed, unattributed = {}, 0.
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA or getattr(event, "is_user_annotation", False):
            continue
        duration = event.device_time_total / steps
        row = cuda.setdefault(event.name, {"name": event.name, "calls_per_step": 0., "us_per_step": 0.})
        row["calls_per_step"] += 1 / steps
        row["us_per_step"] += duration
        owner = owners.get(getattr(event, "linked_correlation_id", None))
        matched = set()
        while owner is not None:
            if owner.name.startswith(("linear.", "deepopt.")) and owner.name not in matched:
                attributed[owner.name] = attributed.get(owner.name, 0.) + duration
                matched.add(owner.name)
            owner = getattr(owner, "cpu_parent", None)
        if not matched:
            unattributed += duration
    for event in profiler.key_averages():
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        row = {"name": event.key, "calls_per_step": event.count / steps,
               "self_cpu_us_per_step": event.self_cpu_time_total / steps,
               "inclusive_cpu_us_per_step": event.cpu_time_total / steps}
        if event.key.startswith(("linear.", "deepopt.")):
            scopes.append({**row, "cuda_us_per_step": attributed.get(event.key)})
        else:
            cpu.append(row)
    activities = sorted(cuda.values(), key=lambda row: row["us_per_step"], reverse=True)
    return {"steps": steps, "cuda_activity_status": "recorded" if cuda else "unavailable",
            "cuda_scope_attribution_status": ("complete" if unattributed == 0 else "partial") if attributed else "unavailable",
            "cuda_scope_unattributed_us_per_step": unattributed if cuda else None,
            "cuda_activity_sum_us_per_step": sum(row["us_per_step"] for row in activities) if cuda else None,
            "cuda_activity_count_per_step": sum(row["calls_per_step"] for row in activities) if cuda else None,
            "cuda_activities": activities, "scopes": scopes,
            "top_cpu_ops": sorted(cpu, key=lambda row: row["self_cpu_us_per_step"], reverse=True)[:30]}


def synchronize(case):
    if case.x.device.type == "cuda":
        case.torch.cuda.synchronize(case.x.device)


def trace_scope_summary(trace, steps):
    """Attribute real GPU work using the CPU API that submitted each activity.

    CUPTI can associate every kernel with an outer autograd function instead
    of our nested ranges. The launch API's timestamp and thread recover those
    ranges without comparing asynchronous GPU timestamps with CPU intervals.
    """
    events = trace["traceEvents"]
    scopes = [event for event in events if event.get("ph") == "X" and event.get("cat") == "user_annotation"
              and event.get("name", "").startswith(("linear.", "deepopt."))]
    launches = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None and (correlation not in launches or event["dur"] < launches[correlation]["dur"]):
            launches[correlation] = event
    gpu = [event for event in events if event.get("ph") == "X"
           and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    totals, durations, unmatched = {}, {}, 0.
    for scope in scopes:
        durations.setdefault(scope["name"], []).append(scope["dur"])
    for event in gpu:
        launch = launches.get(event.get("args", {}).get("correlation"))
        names = set()
        if launch is not None:
            enclosing = [scope for scope in scopes if scope["pid"] == launch["pid"]
                         and scope["ts"] <= launch["ts"] < scope["ts"] + scope["dur"]]
            names = {scope["name"] for scope in enclosing if scope["tid"] == launch["tid"]}
            # The caller's backward range spans launches on autograd workers.
            # Use a cross-thread phase only when there is one unambiguous owner.
            phases = [scope for scope in enclosing if scope["name"].startswith("linear.")]
            if not any(name.startswith("linear.") for name in names) and len(phases) == 1:
                names.add(phases[0]["name"])
        if not names:
            unmatched += event["dur"] / steps
        for name in names:
            totals[name] = totals.get(name, 0.) + event["dur"] / steps
    rows = [{"name": name, "cuda_us_per_step": totals.get(name),
             "profiled_cpu_median_us": median(values), "profiled_cpu_max_us": max(values)}
            for name, values in durations.items()]
    return {"method": "trace_launch_correlation", "status": ("complete" if unmatched == 0 else "partial") if totals else "unavailable",
            "cuda_activity_sum_us_per_step": sum(event["dur"] for event in gpu) / steps if gpu else None,
            "cuda_activity_count_per_step": len(gpu) / steps if gpu else None,
            "unattributed_us_per_step": unmatched if gpu else None, "scopes": rows}


def cuda_profile(case, args, directory):
    torch = case.torch
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda_requested = case.x.device.type == "cuda"
    if cuda_requested:
        if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("CUDA profiling is unavailable; check the PyTorch/CUPTI installation and GPU profiling permissions")
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(case)
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=args.profile_steps, repeat=1)
    with torch.profiler.profile(activities=activities, schedule=schedule, record_shapes=True) as profiler, \
         deepopt_ranges(case) as manifest:
        for _ in range(args.profile_steps + 1):
            profiled_step(case)
            # Keep warmup and previous-step GPU work out of the next interval.
            synchronize(case)
            profiler.step()
    summary = profile_summary(profiler, torch, args.profile_steps)
    trace = directory / "cuda.trace.json"
    profiler.export_chrome_trace(str(trace))
    attribution = trace_scope_summary(json.loads(trace.read_text()), args.profile_steps)
    by_name = {row["name"]: row for row in attribution["scopes"]}
    for row in summary["scopes"]:
        if row["name"] in by_name:
            row.update(by_name[row["name"]])
    summary.update(cuda_scope_attribution_method=attribution["method"], cuda_scope_attribution_status=attribution["status"],
                   cuda_scope_unattributed_us_per_step=attribution["unattributed_us_per_step"])
    summary.update(status="ok", trace=str(trace), deepopt_ranges=manifest)
    if cuda_requested and summary["cuda_activity_status"] != "recorded":
        summary.update(status="error", error="No CUDA activity was captured; the trace cannot establish GPU kernel times")
    if not cuda_requested:
        summary["cuda_activity_status"] = "not_requested"
    return summary


def python_profile(case, args, directory):
    profiler = cProfile.Profile()
    for _ in range(args.profile_steps):
        synchronize(case)
        profiler.enable()
        try:
            case.step()
        finally:
            profiler.disable()
        synchronize(case)
    path = directory / "python.pstats"
    profiler.dump_stats(str(path))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(40)
    text_path = directory / "python.txt"
    text_path.write_text(stream.getvalue())
    rows = [{"file": name[0], "line": name[1], "function": name[2],
             "primitive_calls_per_step": values[0] / args.profile_steps,
             "calls_per_step": values[1] / args.profile_steps,
             "self_us_per_step": values[2] * 1e6 / args.profile_steps,
             "inclusive_us_per_step": values[3] * 1e6 / args.profile_steps}
            for name, values in stats.stats.items()]
    return {"status": "ok", "steps": args.profile_steps, "pstats": str(path), "text": str(text_path),
            "top_functions": sorted(rows, key=lambda row: row["inclusive_us_per_step"], reverse=True)[:40]}


def diagnostics(case, args, directory):
    directory.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, function in (("torch_profiler", cuda_profile), ("python", python_profile)):
        try:
            report[name] = function(case, args, directory)
        except Exception:
            report[name] = {"status": "error", "error": traceback.format_exc()}
    report["status"] = "ok" if all(value["status"] == "ok" for value in report.values()) else "error"
    return report


def worker(args):
    implementation = args._worker
    checkpoint = getattr(args, "_checkpoint", None)
    if checkpoint is None and args.profile:
        checkpoint = args.profile_dir / implementation / "result.json"
    if checkpoint is None and args.output:
        checkpoint = args.output.with_name(args.output.stem + f"-{implementation}-worker.json")
    result = {"implementation": implementation, "status": "running", "environment": {}, "cases": []}
    if checkpoint is not None:
        result["worker_checkpoint"] = str(checkpoint)

    def save():
        if checkpoint is not None:
            write_json(checkpoint, result)

    def progress(stage, shape=None, mode=None):
        label = "" if shape is None else " " + "x".join(map(str, shape)) + " " + mode
        print(f"[fp8-eager] {implementation}{label}: {stage}", file=sys.stderr, flush=True)

    progress("starting worker" + (f"; checkpoint {checkpoint}" if checkpoint is not None else ""))
    save()
    try:
        import torch

        environment = result["environment"]
        environment.update(python=sys.version, executable=sys.executable,
                           torch=package_info("torch", "torch"), cuda=torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use CUDA-enabled PyTorch on an NVIDIA SM90+ GPU")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        capability = torch.cuda.get_device_capability(device)
        environment.update(gpu=torch.cuda.get_device_name(device), capability=list(capability))
        if capability < (9, 0):
            raise RuntimeError("This three-library comparison requires an NVIDIA SM90+ GPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        names = {"deepopt": ("deepopt", "deep-optimization"), "torchao": ("torchao", "torchao"),
                 "te": ("transformer_engine.pytorch", "transformer-engine")}
        if implementation in names:
            environment[implementation] = package_info(*names[implementation])
        if implementation == "deepopt":
            policy = deepopt_kernel_configuration(args)
            if policy is not None:
                selected = {policy["backend"], policy["preferred"],
                            *policy["families"].values(), *policy["roles"].values()}
                if "tilelang" in selected:
                    environment["tilelang"] = package_info("tilelang", "tilelang")
        for shape in args.shapes:
            for mode in args.modes:
                row = {"shape_m_k_n": list(shape), "mode": mode, "status": "running", "stage": "setup"}
                result["cases"].append(row)
                case = None
                try:
                    progress("cold setup", shape, mode)
                    save()
                    cold_started = perf_counter()
                    case = LinearCase(args, implementation, shape, mode)
                    torch.cuda.synchronize(device)
                    row.update(model_type=f"{type(case.model).__module__}.{type(case.model).__qualname__}",
                               configuration=case.configuration)
                    row["cold_start"] = {"setup_wall_ms": (perf_counter() - cold_started) * 1000,
                                         "cache_state": "process-local; earlier cases may have warmed compatible kernels"}
                    row["stage"] = "first_launch"
                    progress("first launch (compilation/autotuning may occur)", shape, mode)
                    save()
                    cold_started = perf_counter()
                    case.step()
                    torch.cuda.synchronize(device)
                    row["cold_start"]["first_launch_wall_ms"] = (perf_counter() - cold_started) * 1000
                    row["stage"] = "timing"
                    progress("warm timing", shape, mode)
                    save()
                    row["timing"] = measure(case.step, args, torch)
                    row["stage"] = "validation"
                    save()
                    row.update(case.check())
                    if getattr(args, "qualify_tilelang", False) and implementation == "deepopt":
                        from deepopt._tilelang_benchmarking import qualify_linear
                        qualified_started = perf_counter()
                        row["validation"] = qualify_linear(case)
                        row["qualification_wall_seconds"] = perf_counter() - qualified_started
                        if row["validation"]["status"] != "passed":
                            raise RuntimeError("Independent Linear executor qualification failed")
                    if not all(value["finite"] for value in row["errors_vs_fp32"].values()):
                        raise RuntimeError("Non-finite output or gradients; timings are invalid")
                    row["status"] = "ok"
                    if args.profile:
                        name = f"{len(result['cases']):03d}-" + "x".join(map(str, shape)) + "-" + mode
                        directory = args.profile_dir / implementation / name
                        # A profiler/CUPTI crash must not erase completed timing.
                        row["stage"] = "profiling"
                        progress("separate diagnostics", shape, mode)
                        save()
                        row["diagnostics"] = diagnostics(case, args, directory)
                    row["stage"] = "complete"
                    progress("complete", shape, mode)
                except Exception:
                    row.update(status="error", failed_stage=row["stage"], stage="failed", error=traceback.format_exc())
                    progress("failed", shape, mode)
                finally:
                    save()
                    del case
                    gc.collect()
                    torch.cuda.empty_cache()
        result["status"] = "ok" if all(row["status"] == "ok" and row.get("diagnostics", {}).get("status", "ok") == "ok"
                                         for row in result["cases"]) else "partial"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    save()
    return result


def print_results(report):
    print("M x K x N               mode              library       wall (ms)   CUDA (ms)   BF16/wall")
    baseline = {(tuple(row["shape_m_k_n"]), row["mode"]): row["timing"]["wall_ms"]["median"]
                for result in report["results"] if result["implementation"] == "bf16"
                for row in result["cases"] if row["status"] == "ok"}
    for result in report["results"]:
        implementation = result["implementation"]
        if "error" in result:
            print(f"{implementation}: ERROR\n{result['error']}", file=sys.stderr)
        for row in result["cases"]:
            shape = "x".join(map(str, row["shape_m_k_n"]))
            if row["status"] != "ok":
                print(f"{shape} {row['mode']} {implementation}: ERROR\n{row['error']}", file=sys.stderr)
                continue
            wall = row["timing"]["wall_ms"]["median"]
            cuda = row["timing"]["cuda_stream_ms"]["median"]
            reference = baseline.get((tuple(row["shape_m_k_n"]), row["mode"]))
            ratio = f"{reference / wall:.3f}x" if reference is not None and wall > 0 else "-"
            print(f"{shape:<23} {row['mode']:<17} {implementation:<12} {wall:>10.4f} {cuda:>11.4f} {ratio:>11}")
            diagnostic = row.get("diagnostics", {})
            for name in ("torch_profiler", "python"):
                if diagnostic.get(name, {}).get("status") == "error":
                    print(f"{implementation}/{shape}/{row['mode']} {name}: {diagnostic[name]['error']}", file=sys.stderr)
            profile = diagnostic.get("torch_profiler", {})
            if profile.get("cuda_activity_status") == "recorded":
                print(f"  Diagnostic CUDA activity sum: {profile['cuda_activity_sum_us_per_step']:.2f} us/step; "
                      f"{profile['cuda_activity_count_per_step']:.1f} activities/step")


def run(args):
    # Inherit the environment unchanged and use the SAME interpreter. The
    # portable script imports installed libraries normally, never checkout/src.
    if args.profile:
        args.output = args.output or Path("fp8-eager-profile.json")
        parent = args.profile_dir or args.output.with_name(args.output.stem + "-profiles")
        identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        args.profile_dir = parent.resolve() / identity
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    checkpoint_root = (args.profile_dir if args.profile else
                       args.output.with_name(args.output.stem + "-workers-" + identity).resolve() if args.output else
                       Path(tempfile.mkdtemp(prefix="fp8-eager-workers-")))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results = []
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if not key.startswith("_") and not (key == "deepopt_kernel_policy" and value is None)
                and not (key == "qualify_tilelang" and not value)}
    report = {"settings": settings, "notes": NOTES, "results": results,
              "worker_directory": str(checkpoint_root),
              "progress": {"requested_implementations": len(args.implementations), "completed_implementations": 0}}
    if args.profile:
        report.update(profile_directory=str(args.profile_dir), profile_archive=str(args.output.with_suffix(".zip")))
    for implementation in args.implementations:
        checkpoint = checkpoint_root / implementation / "result.json"
        report["progress"].update(active_implementation=implementation, active_worker_checkpoint=str(checkpoint))
        if args.output:
            write_json(args.output, report)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                   "--_checkpoint", str(checkpoint), "--_worker", implementation]
        if args.profile:
            command.extend(["--profile-dir", str(args.profile_dir)])
        print(f"Benchmarking {implementation} ...", flush=True)
        # The result protocol remains on captured stdout. Inherit stderr so
        # cold-setup/JIT/measurement progress is visible while the child runs.
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=None, text=True)
        stderr = getattr(completed, "stderr", None) or ""
        if stderr:
            print(stderr, file=sys.stderr, end="")
        result = next((json.loads(line[len(RESULT_PREFIX):]) for line in completed.stdout.splitlines()
                       if line.startswith(RESULT_PREFIX)), None)
        if result is None:
            result = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
                "implementation": implementation, "cases": []}
            result.update(status="error", error=f"Worker exited {completed.returncode}\n{completed.stdout}\n{stderr}")
            for row in result["cases"]:
                if row.get("status") == "running":
                    row.update(status="error", error=f"Worker exited {completed.returncode} during {row.get('stage', 'case execution')}")
        elif completed.returncode:
            result["status"] = "error"
            result.setdefault("error", f"Worker exited {completed.returncode}")
        results.append(result)
        report["progress"]["completed_implementations"] = len(results)
        if args.output:
            write_json(args.output, report)
    report["progress"].update(active_implementation=None, active_worker_checkpoint=None)
    if args.output:
        write_json(args.output, report)
    if args.profile:
        with zipfile.ZipFile(report["profile_archive"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(args.output, "summary.json")
            archive.write(Path(__file__).resolve(), "benchmark.py")
            for path in sorted(args.profile_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, "profiles/" + path.relative_to(args.profile_dir).as_posix())
    print_results(report)
    if args.output:
        print(f"Saved {args.output}")
    if args.profile:
        print(f"Profiles and report: {report['profile_archive']}")
    return report


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        deepopt_kernel_configuration(args)
    except (ValueError, ImportError) as error:
        argument_parser.error(str(error))
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 1 or args.profile_steps < 1:
        argument_parser.error("warmup, iterations, repeats and profile-steps must be positive")
    if not math.isfinite(args.time_budget) or args.time_budget < 0:
        argument_parser.error("time-budget must be a finite nonnegative number; 0 runs all requested samples")
    if args.profile and args.output and args.output.suffix.lower() == ".zip":
        argument_parser.error("--output must name the JSON report; --profile creates a separate ZIP alongside it")
    if args.device != "cuda" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        argument_parser.error("device must be cuda or cuda:INDEX")
    if args._worker:
        print(RESULT_PREFIX + json.dumps(worker(args), allow_nan=False))
        return 0
    report = run(args)
    return int(any(result["status"] != "ok" for result in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
