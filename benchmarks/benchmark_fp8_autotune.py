"""Benchmark the DeepOpt library installed in the active Python environment.

python benchmarks/benchmark_fp8_autotune.py
python benchmarks/benchmark_fp8_autotune.py --preset vit --scaling tensorwise --implementation both
python benchmarks/benchmark_fp8_autotune.py --output results/fp8-benchmark.json

Install DeepOpt in this environment before running; an editable installation is
supported. The script can be copied outside the repository. Each implementation
and scope runs in a fresh process using normal Python imports. By default its
Triton disk cache is isolated, so first-use latency includes compilation. Warmed
timings use synchronized wall-clock measurements. No datasets or model downloads
are needed.
"""

import argparse
import cProfile
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import inspect
import json
from math import isfinite
import os
import pstats
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch


_RESULT_PREFIX = "DEEPOPT_BENCHMARK_RESULT="


def _config_record(config):
    return {"tile": dict(config.kwargs), "num_warps": config.num_warps,
            "num_stages": config.num_stages}


@contextmanager
def _record_autotune_trials():
    """Record cold-search timings for every key, outside the warmed path."""
    backend = sys.modules.get("deepopt.opt.backends.triton_fp8")
    kernel = getattr(backend, "_gemm_nt_kernel", None)
    trials = {}
    if kernel is None or not hasattr(kernel, "_bench"):
        yield trials
        return
    original = kernel._bench

    def record(*args, config, **kwargs):
        all_args = {**kernel.nargs, **kwargs}
        named = {name: value for name, value in all_args.items() if name in kernel.arg_names}
        key = (tuple(named[name] for name in kernel.keys if name in named)
               + tuple(str(value.dtype) for value in named.values() if hasattr(value, "dtype")))
        started = perf_counter()
        timings = original(*args, config=config, **kwargs)
        samples = timings if isinstance(timings, (list, tuple)) else [timings]
        trials.setdefault(key, []).append({
            **_config_record(config),
            "quantiles_ms": [float(value) if isfinite(value) else None for value in samples],
            "status": "ok" if all(isfinite(value) for value in samples) else "unavailable",
            "compile_and_benchmark_seconds": perf_counter() - started,
        })
        return timings

    with patch.object(kernel, "_bench", new=record):
        yield trials


def _autotune_diagnostics(trials=None):
    """Read cached winners after timing when the installed library exposes them."""
    backend = sys.modules.get("deepopt.opt.backends.triton_fp8")
    cache = getattr(getattr(backend, "_gemm_nt_kernel", None), "cache", None)
    if not isinstance(cache, dict):
        return None
    policy = sys.modules.get("deepopt.opt.backends._triton_autotune")
    key_names = getattr(policy, "AUTOTUNE_KEY", ())

    def serializable(value):
        return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)

    winners = []
    for key, config in cache.items():
        tile = getattr(config, "kwargs", None)
        if not isinstance(tile, dict):
            continue
        key_values = key if isinstance(key, (tuple, list)) else (key,)
        metadata = ({name: serializable(value) for name, value in zip(key_names, key_values)}
                    if key_names else {"cache_key": [serializable(value) for value in key_values]})
        winners.append({
            "key": metadata,
            "tile": {name: serializable(value) for name, value in tile.items()},
            "num_warps": serializable(getattr(config, "num_warps", None)),
            "num_stages": serializable(getattr(config, "num_stages", None)),
            "implicit_dtype_key": [serializable(value) for value in key_values[len(key_names):]],
            "candidates": (trials or {}).get(key, []),
        })
    return winners


def _prepare_gemm_products(x, weight, bias, grad, plan):
    """Capture the library's real autograd operands without warming GEMMs.

    The explicit upstream gradient determines backward inputs independently of
    the placeholder forward result. Quantization and saved orientations still
    execute the imported library's own implementation.
    """
    import torch
    from deepopt.opt.do_nn.functional import do_linear
    from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func

    captured = []

    def capture(a, b, **kwargs):
        captured.append((a, b, kwargs.get("bias"), kwargs["out_dtype"]))
        return torch.zeros((a.data.shape[0], b.data.shape[0]),
                           dtype=kwargs["out_dtype"], device=a.data.device)

    try:
        with patch.object(linear_autograd_func, "gemm_nt", new=capture):
            do_linear(x, weight, bias, plan=plan).backward(grad)
    finally:
        x.grad = weight.grad = bias.grad = None
    if len(captured) != 3:
        raise RuntimeError("Expected forward, input-gradient, and weight-gradient GEMMs")
    return [(name, *product) for name, product in
            zip(("forward", "input_gradient", "weight_gradient"), captured)]


def _event_times(operation, torch, device, args):
    """Stream elapsed time, including gaps while Python submits work."""
    samples = []
    with torch.cuda.device(device):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            start.record()
            for _ in range(args.iterations):
                operation()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iterations)
    return {"median_ms": median(samples), "samples_ms": samples}


def _memory_measurement(operation, torch, device, prepare):
    """Measure one warmed call above the resident input/cache baseline."""
    prepare()
    torch.cuda.synchronize(device)
    baseline = {"allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device)}
    torch.cuda.reset_peak_memory_stats(device)
    result = operation()  # Keep its output and gradients alive until after sampling.
    try:
        torch.cuda.synchronize(device)
        return {"baseline_allocated_bytes": baseline["allocated"],
                "baseline_reserved_bytes": baseline["reserved"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "extra_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline["allocated"],
                "extra_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) - baseline["reserved"],
                "allocated_with_output_and_gradients_bytes": torch.cuda.memory_allocated(device)}
    finally:
        del result
        prepare()


def _device_time(event):
    # Kineto's device terminology replaced the older CUDA-specific attributes.
    value = getattr(event, "device_time_total", None)
    return float(value if value is not None else getattr(event, "cuda_time_total", 0))


def _profile_summary(profiler, torch, iterations, limit):
    cpu = []
    for event in profiler.key_averages():
        if event.device_type == torch.autograd.DeviceType.CPU:
            cpu.append({"name": event.key, "count": event.count,
                        "self_cpu_ms_per_call": event.self_cpu_time_total / 1000 / iterations,
                        "total_cpu_ms_per_call": event.cpu_time_total / 1000 / iterations})
    cpu.sort(key=lambda row: row["self_cpu_ms_per_call"], reverse=True)
    activities = {}
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        row = activities.setdefault(event.name, {"name": event.name, "count": 0, "total_ms_per_call": 0.0})
        row["count"] += 1
        row["total_ms_per_call"] += _device_time(event) / 1000 / iterations
    gpu = sorted(activities.values(), key=lambda row: row["total_ms_per_call"], reverse=True)
    # This separate total includes all activities, even those omitted by top-N.
    return {"iterations": iterations, "top_cpu_ops": cpu[:limit],
            "cuda_runtime_ops": [row for row in cpu if row["name"].startswith(("cuda", "cuLaunch"))],
            "top_cuda_activities": gpu[:limit],
            "cuda_activity_status": "recorded" if gpu else "unavailable",
            "cuda_activity_count_per_call": sum(row["count"] for row in gpu) / iterations if gpu else None,
            "cuda_activity_sum_ms_per_call": sum(row["total_ms_per_call"] for row in gpu) if gpu else None}


def _profile_measurement(operation, torch, device, args, prepare):
    prepare()
    torch.cuda.synchronize(device)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(args.profile_iterations):
                operation()
            torch.cuda.synchronize(device)
        return {"status": "ok", **_profile_summary(profiler, torch, args.profile_iterations, args.profile_rows)}
    except RuntimeError as error:
        return {"status": "unavailable", "error": str(error)}
    finally:
        prepare()


def _host_profile(operation, torch, device, args, prepare):
    """Profile warmed Python submission separately from Kineto instrumentation."""
    prepare()
    torch.cuda.synchronize(device)
    profiler = cProfile.Profile()
    try:
        with profiler:
            for _ in range(args.profile_iterations):
                operation()
        torch.cuda.synchronize(device)
        rows = []
        for (filename, line, name), (_, calls, self_time, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append({"function": f"{filename}:{line}:{name}", "calls": calls,
                         "self_ms_per_call": self_time * 1000 / args.profile_iterations,
                         "cumulative_ms_per_call": cumulative * 1000 / args.profile_iterations})
        rows.sort(key=lambda row: row["self_ms_per_call"], reverse=True)
        return {"top_functions": rows[:args.profile_rows],
                "note": "Instrumented wall time includes synchronous waits; cumulative times are nested."}
    finally:
        prepare()


def _graph_measurement(operation, torch, device, args):
    """Replay the prepared GEMM without per-operation Python dispatch."""
    try:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    operation()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = operation()
            # Retain the captured output through all replays; do not allocate a
            # different output or scratch buffer for every measured invocation.
            graph.replay()
            timing = _event_times(graph.replay, torch, device, args)
            del output
        return {"status": "ok", **timing}
    except (RuntimeError, NotImplementedError) as error:
        return {"status": "unavailable", "error": str(error)}


def _measure(operation, torch, device, args, *, prepare=lambda: None):
    def elapsed(iterations):
        torch.cuda.synchronize(device)
        started = perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)
        return perf_counter() - started

    first_use = elapsed(1)
    for _ in range(args.warmups):
        operation()
    samples = [elapsed(args.iterations) * 1000 / args.iterations
               for _ in range(args.repeats)]
    result = {
        "first_use_seconds": first_use,
        "warm_median_ms": median(samples),
        "warm_samples_ms": samples,
    }
    if getattr(args, "detailed", False):
        result["cuda_events"] = _event_times(operation, torch, device, args)
        # Measure before graph capture creates a private allocation pool.
        result["memory"] = _memory_measurement(operation, torch, device, prepare)
        result["profile"] = _profile_measurement(operation, torch, device, args, prepare)
        if getattr(args, "host_profile", False):
            result["python_submission"] = _host_profile(operation, torch, device, args, prepare)
        result["cuda_graph"] = (_graph_measurement(operation, torch, device, args)
                                if args.scope == "gemm" else
                                {"status": "skipped", "reason": "Only prepared GEMMs are captured."})
    return result


def _benchmark_plan(args, torch):
    formats = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    training = {"backend": "triton", "num_stages": args.num_stages, "out_dtype": torch.bfloat16,
                "fp8_dtypes": {role: formats[getattr(args, role + "_format", default)]
                               for role, default in (("activation", "e4m3fn"), ("weight", "e4m3fn"), ("grad", "e5m2"))}}
    blocks = (dict.fromkeys(("activation", "weight", "grad")) if args.scaling == "tensorwise" else
              {"activation": (1, args.scale_k), "weight": (args.scale_k, args.scale_k),
               "grad": (args.scale_k, args.scale_k)})
    return {"fp8_training": training, "block_shapes": blocks}


def _library_identity(deepopt, gemm_nt):
    """Fingerprint the imported installation without modifying import paths."""
    root = Path(deepopt.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    try:
        recipe = importlib.import_module("deepopt.opt._recipe").fp8_recipe("triton")
    except ModuleNotFoundError as error:
        if error.name != "deepopt.opt._recipe":
            raise
        legacy = inspect.signature(gemm_nt).parameters.get("max_num_imprecise_acc")
        recipe = {"version": 1, "accumulation": "legacy-explicit" if legacy else "unknown",
                  "legacy_limit": legacy.default if legacy else None}
    return {"source_sha256": digest.hexdigest(), "numerical_recipe": recipe}


@contextmanager
def _fixed_gemm_config(spec):
    """Benchmark one validated tile without changing library configuration."""
    if spec is None:
        yield
        return
    import triton
    from deepopt.opt.backends.triton_fp8 import _gemm_nt_kernel as kernel
    from deepopt.opt.backends._triton_autotune import autotune_cache_key, prune_gemm_configs

    bm, bn, bk, warps, stages, split, group = spec
    config = triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "NUM_STAGES": stages,
                            "SPLIT_K": split, "GROUP_SIZE_M": group}, num_warps=warps, num_stages=stages)
    original, validated = kernel.run, set()

    def run(*args, **kwargs):
        named = {**dict(zip(kernel.arg_names, args)), **kwargs}
        key = autotune_cache_key(named)
        if key not in validated:
            prune_gemm_configs([config], named)
            validated.add(key)
        return original(*args, **kwargs)

    # Single-config Triton launches skip benchmarking and persistent winner
    # lookup. Validation also rejects incompatible scales/stages/scratch sizes.
    with patch.object(kernel, "configs", [config]), patch.object(kernel, "cache", {}), \
         patch.object(kernel, "run", new=run):
        yield


def _worker(args):
    # Keep --help and orchestration usable without importing torch or Triton.
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA-enabled PyTorch and a compatible NVIDIA GPU.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select a CUDA device.")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) < (9, 0):
        raise SystemExit("The Triton TMA FP8 backend requires compute capability 9.0 or newer.")
    import deepopt
    from importlib.metadata import PackageNotFoundError, version
    from deepopt.opt.backends import gemm_nt
    from deepopt.opt.do_nn.functional import do_linear
    # Load the tuner before measuring so the recorder sees the very first key.
    if args.implementation == "fp8":
        from deepopt.opt.backends import triton_fp8  # noqa: F401

    torch.manual_seed(42)
    # Initialize the CUDA context before timing any projection.
    torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    plan = _benchmark_plan(args, torch)
    # Keep strict comparisons compatible with versions predating check_finite.
    # A persistent disabled AMP context around BF16 measurements would keep
    # native weight-cast caches alive across forwards, unlike a training step.
    check_scope = (nullcontext() if args.check_finite or args.implementation == "bf16" else
                   deepopt.autocast("cuda", enabled=False, fp8=None, check_finite=False))
    config_scope = (_fixed_gemm_config(getattr(args, "gemm_config", None))
                    if args.implementation == "fp8" else nullcontext())
    with check_scope, config_scope, _record_autotune_trials() as trials:
        rows = _measure_batches(args, torch, device, plan, do_linear, gemm_nt)
    try:
        deepopt_version = version("deep-optimization")
    except PackageNotFoundError:
        deepopt_version = getattr(deepopt, "__version__", None)
    return {
        "scope": args.scope,
        "implementation": args.implementation,
        "scaling": args.scaling,
        "fixed_gemm_config": getattr(args, "gemm_config", None),
        "fp8_policy": {"block_shapes": plan["block_shapes"],
                       "formats": {role: str(dtype) for role, dtype in plan["fp8_training"]["fp8_dtypes"].items()}},
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        **_library_identity(deepopt, gemm_nt),
                        "triton": getattr(sys.modules.get("triton"), "__version__", None),
                        "deepopt_version": deepopt_version,
                        "deepopt_source": str(Path(deepopt.__file__).resolve()),
                        "gpu": torch.cuda.get_device_name(device),
                        "capability": torch.cuda.get_device_capability(device)},
        "timings": rows,
        "autotune_selections": _autotune_diagnostics(trials),
        "tuning_result_cache_enabled": bool(getattr(
            getattr(sys.modules.get("deepopt.opt.backends.triton_fp8"), "_gemm_nt_kernel", None),
            "cache_results", False)),
    }


def _projection_cases(args):
    if getattr(args, "preset", "classifier") == "vit":
        return [(name, batch if name == "head" else batch * args.tokens_per_sample, inputs, outputs)
                for batch in args.batches
                for name, inputs, outputs in (("qkv", 768, 2304), ("attention", 768, 768),
                                              ("fc1", 768, 3072), ("fc2", 3072, 768),
                                              ("head", 768, args.output_features))]
    return [("classifier", batch, args.input_features, args.output_features) for batch in args.batches]


def _dense_products(x, weight, bias, grad):
    import torch

    # Prepared GEMMs have no autograd bookkeeping in either implementation.
    x, weight, bias, grad = (value.detach() for value in (x, weight, bias, grad))
    input_dtype = x.dtype
    x = x.to(torch.bfloat16)
    return [("forward", x, weight.to(torch.bfloat16), bias.to(torch.bfloat16), torch.bfloat16),
            ("input_gradient", grad, weight.T.to(torch.bfloat16), None, input_dtype),
            ("weight_gradient", grad.T, x.T, None, torch.float32)]


def _error_metrics(actual, expected, torch):
    actual, expected = actual.detach().float(), expected.detach().float()
    error = actual - expected
    values = {"relative_l2": (error.norm() / expected.norm().clamp_min(1e-30)).item(),
              "max_abs": error.abs().max().item(), "rmse": error.square().mean().sqrt().item()}
    return {**{name: value if isfinite(value) else None for name, value in values.items()},
            "finite": bool(torch.isfinite(actual).all().item())}


def _dequantized_slice(value, rows, start, stop, torch):
    payload = value.data[rows, start:stop].double()
    if value.block_shape is None:
        return payload * value.scale_inv.double()
    bm, bn = value.block_shape
    row_ids = torch.arange(rows.start, rows.stop, device=payload.device) // bm
    col_ids = torch.arange(start, stop, device=payload.device) // bn
    return payload * value.scale_inv[row_ids[:, None], col_ids[None, :]].double()


def _numerical_breakdown(actual, left, right, bias, original, torch, *, chunk_rows=128, chunk_k=4096):
    """Separate quantization and accumulation errors against tiled FP64 products.

    Build neither full dequantized operands nor full FP64 reference outputs.
    These diagnostics run after timing and memory measurement.
    """
    original_left, original_right, original_bias = original
    m, k = left.data.shape
    n = right.data.shape[0]
    stats = torch.zeros((3, 4), dtype=torch.float64, device=actual.device)
    with torch.no_grad():
        for row in range(0, m, chunk_rows):
            rows = slice(row, min(row + chunk_rows, m))
            quantized = torch.zeros((rows.stop - row, n), dtype=torch.float64, device=actual.device)
            floating = torch.zeros_like(quantized)
            for start in range(0, k, chunk_k):
                stop = min(start + chunk_k, k)
                quantized.add_(_dequantized_slice(left, rows, start, stop, torch)
                               @ _dequantized_slice(right, slice(0, n), start, stop, torch).T)
                floating.add_(original_left[rows, start:stop].double()
                              @ original_right[:, start:stop].double().T)
            if bias is not None:
                quantized.add_(bias.detach().double())
            if original_bias is not None:
                floating.add_(original_bias.detach().double())
            observed = actual[rows].double()
            for index, (value, reference) in enumerate(((quantized, floating), (observed, quantized),
                                                       (observed, floating))):
                error = value - reference
                stats[index, 0] += error.square().sum()
                stats[index, 1] += reference.square().sum()
                stats[index, 2] = torch.maximum(stats[index, 2], error.abs().max())
                stats[index, 3] += (~torch.isfinite(value) | ~torch.isfinite(reference)).sum()
    records = {}
    for name, (squared, norm, maximum, bad) in zip(
            ("quantization", "accumulation_and_output_cast", "total"), stats.cpu().tolist()):
        values = {"relative_l2": (squared / max(norm, 1e-60)) ** .5,
                  "max_abs": maximum, "rmse": (squared / (m * n)) ** .5}
        records[name] = {**{key: value if isfinite(value) else None for key, value in values.items()},
                         "finite": bad == 0}
    return records


def _input_matrix(rows, cols, *, torch, device, dtype, pattern, role):
    if pattern == "zeros":
        return torch.zeros((rows, cols), device=device, dtype=dtype)
    if pattern == "cancellation":
        # Paired columns cancel in forward/dX; paired rows exercise long dW.
        value = torch.randn(((rows + 1) // 2, (cols + 1) // 2), device=device)
        value = value.repeat_interleave(2, 0).repeat_interleave(2, 1)[:rows, :cols]
        if role == "activation":
            value[1::2].neg_()
        if role in ("activation", "grad"):
            value[:, 1::2].neg_()
        value[-1, -1] += .001
        return value.to(dtype)
    value = torch.randn((rows, cols), device=device, dtype=dtype)
    if pattern == "mixed-magnitude":
        factors = torch.pow(2., torch.arange(cols, device=device) % 21 - 10)
        value.mul_(factors.to(dtype))
    return value


def _reference_product(name, x, weight, bias, grad, torch):
    # Accuracy is computed after timing; IEEE FP32 references must not silently
    # use a process-wide TF32 setting intended for performance measurements.
    cuda = x.device.type == "cuda"
    previous = torch.backends.cuda.matmul.allow_tf32 if cuda else None
    try:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        if name == "forward":
            return x.detach().float() @ weight.detach().float().T + bias.detach().float()
        if name == "input_gradient":
            return grad.float() @ weight.detach().float()
        return grad.float().T @ x.detach().float()
    finally:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous


def _measure_batches(args, torch, device, plan, do_linear, gemm_nt):
    rows = []
    dense = getattr(args, "implementation", "fp8") == "bf16"
    for projection_name, batch, inputs, outputs in _projection_cases(args):
        input_dtype = torch.float32 if getattr(args, "input_dtype", "bf16") == "fp32" else torch.bfloat16
        pattern = getattr(args, "input_pattern", "random")
        x = _input_matrix(batch, inputs, torch=torch, device=device, dtype=input_dtype,
                          pattern=pattern, role="activation").requires_grad_()
        # Keep FP32 master parameters, as with ordinary BF16 autocast training.
        weight = _input_matrix(outputs, inputs, torch=torch, device=device, dtype=torch.float32,
                               pattern=pattern, role="weight").requires_grad_()
        bias = torch.randn(outputs, device=device, requires_grad=True)
        grad = _input_matrix(batch, outputs, torch=torch, device=device, dtype=torch.bfloat16,
                             pattern=pattern, role="grad") / batch
        if args.scope == "linear":
            def clear_gradients():
                x.grad = weight.grad = bias.grad = None

            def projection():
                clear_gradients()
                if dense:
                    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                        output = torch.nn.functional.linear(x, weight, bias)
                else:
                    output = do_linear(x, weight, bias, plan=plan)
                output.backward(grad)
                return output.detach()

            row = {"projection": projection_name, "batch": batch, "operation": "forward_backward",
                   **_measure(projection, torch, device, args, prepare=clear_gradients)}
            if getattr(args, "accuracy", False):
                actual = projection()
                row["errors_vs_fp32"] = {
                    name: _error_metrics(value, _reference_product(name, x, weight, bias, grad, torch), torch)
                    for name, value in (("forward", actual), ("input_gradient", x.grad),
                                        ("weight_gradient", weight.grad))}
                del actual
                clear_gradients()
            rows.append(row)
        elif args.scope == "bias-gradient":
            rows.append({"projection": projection_name, "batch": batch, "operation": "bias_gradient",
                         **_measure(lambda: grad.sum(dim=0, dtype=torch.float32), torch, device, args)})
        elif args.scope in ("quantize", "quantize-pair", "scale", "dequantize"):
            from deepopt.opt.quantization.quantizer import quantize_fp8
            from deepopt.opt.scaling_strategy.amax import AmaxScaling

            operands = [("activation", x, "activation"), ("weight", weight, "weight"),
                        ("gradient", grad, "grad")]
            for name, value, role in tuple(operands):
                blocks = plan["block_shapes"][role]
                if role in ("activation", "grad") and blocks is not None and blocks[0] == 1 and blocks[1] > 1:
                    operands.append((name + "_transposed", value.T, role))
            for name, value, role in operands:
                def quantize():
                    return quantize_fp8(value, fp8_dtype=plan["fp8_training"]["fp8_dtypes"][role],
                                        block_shape=plan["block_shapes"][role])

                if args.scope == "scale":
                    def operation():
                        return AmaxScaling(plan["block_shapes"][role]).compute(
                            value, fp8_dtype=plan["fp8_training"]["fp8_dtypes"][role])
                elif args.scope == "dequantize":
                    # Prepare FP8 operands outside reconstruction timing.
                    operation = quantize().dequantize
                elif args.scope == "quantize-pair":
                    from deepopt.opt.quantization.quantizer import _quantize_tensorwise_pair

                    def operation():
                        return _quantize_tensorwise_pair(value, fp8_dtype=plan["fp8_training"]["fp8_dtypes"][role])
                else:
                    operation = quantize
                rows.append({"projection": projection_name, "batch": batch, "operation": name, "shape": list(value.shape),
                             "stride": list(value.stride()), **_measure(operation, torch, device, args)})
        else:
            products = _dense_products(x, weight, bias, grad) if dense else _prepare_gemm_products(x, weight, bias, grad, plan)
            for name, left, right, product_bias, output_dtype in products:
                def product():
                    if dense:
                        return torch.nn.functional.linear(left, right, product_bias).to(output_dtype)
                    return gemm_nt(left, right, backend="triton", bias=product_bias,
                                   out_dtype=output_dtype, num_stages=args.num_stages)

                row = {"projection": projection_name, "batch": batch, "operation": name,
                       "M_N_K": [left.shape[0] if dense else left.data.shape[0],
                                 right.shape[0] if dense else right.data.shape[0],
                                 left.shape[1] if dense else left.data.shape[1]],
                       **_measure(product, torch, device, args)}
                if getattr(args, "accuracy", False):
                    actual = product()
                    if dense:
                        row["errors_vs_fp32"] = _error_metrics(
                            actual, _reference_product(name, x, weight, bias, grad, torch), torch)
                    else:
                        originals = {"forward": (x.detach(), weight.detach(), bias.detach()),
                                     "input_gradient": (grad, weight.detach().T, None),
                                     "weight_gradient": (grad.T, x.detach().T, None)}
                        row["numerical_errors_vs_fp64"] = _numerical_breakdown(
                            actual, left, right, product_bias, originals[name], torch)
                    del actual
                rows.append(row)
    return rows


def _run(args):
    results = []
    scopes = ("linear", "gemm") if args.scope == "both" else (args.scope,)
    implementations = ("fp8", "bf16") if args.implementation == "both" else (args.implementation,)
    for implementation in implementations:
        for scope in scopes:
            print(f"Measuring installed DeepOpt ({implementation}, {scope}) in a fresh process...",
                  file=sys.stderr, flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--_worker",
                       "--scope", scope, "--device", args.device,
                       "--warmups", str(args.warmups), "--iterations", str(args.iterations),
                       "--repeats", str(args.repeats), "--batches", *map(str, args.batches)]
            command.extend(["--num-stages", str(args.num_stages)])
            command.extend(["--implementation", implementation, "--scaling", args.scaling,
                            "--preset", args.preset, "--tokens-per-sample", str(args.tokens_per_sample),
                            "--input-dtype", args.input_dtype])
            command.extend(["--input-pattern", args.input_pattern])
            if args.gemm_config is not None:
                command.extend(["--gemm-config", *map(str, args.gemm_config)])
            for role in ("activation", "weight", "grad"):
                command.extend(["--" + role + "-format", getattr(args, role + "_format")])
            if not args.accuracy:
                command.append("--no-accuracy")
            command.extend(["--input-features", str(args.input_features),
                            "--output-features", str(args.output_features), "--scale-k", str(args.scale_k)])
            if not args.check_finite:
                command.append("--no-check-finite")
            if args.detailed:
                command.extend(["--detailed", "--profile-iterations", str(args.profile_iterations),
                                "--profile-rows", str(args.profile_rows)])
            if args.host_profile:
                command.append("--host-profile")
            cache = nullcontext(None) if args.reuse_cache else tempfile.TemporaryDirectory(prefix="deepopt-fp8-")
            with cache as cache_dir:
                environment = os.environ.copy()
                if cache_dir is not None:
                    environment["TRITON_CACHE_DIR"] = cache_dir
                completed = subprocess.run(command, env=environment, capture_output=True, text=True)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            if completed.returncode:
                print(completed.stdout, file=sys.stderr, end="")
                raise SystemExit(completed.returncode)
            result = None
            for line in completed.stdout.splitlines():
                if line.startswith(_RESULT_PREFIX):
                    result = json.loads(line[len(_RESULT_PREFIX):])
                else:
                    print(line, file=sys.stderr)
            if result is None:
                raise RuntimeError("Benchmark worker returned no timing result")
            results.append(result)
    return {
        "settings": {key: value for key, value in vars(args).items() if key not in ("_worker", "output")},
        "report_version": 3,
        "disk_cache": "existing" if args.reuse_cache else "isolated per process",
        "notes": [
            "First-use measurements include compilation and, when enabled, candidate benchmarking.",
            "With --reuse-cache, earlier processes may warm the Triton disk cache for later processes.",
            "Later batch shapes may reuse already compiled kernels from earlier shapes in the same process.",
            "Warm timings are synchronized wall-clock milliseconds per invocation.",
            "Detailed CUDA-event timings include stream gaps while the host submits work; they are not kernel-only time.",
            "Graph replay captures only prepared GEMMs and reuses output/scratch storage; compare it with eager dispatch.",
            "Detailed profiling runs separately from timing. CUDA activity sums may overlap and exclude CPU submission gaps.",
            "CPU profiler totals are nested; do not add them to CUDA activities. Counts cover all profiled calls.",
            "Memory is one warmed operation above resident inputs/prepared operands/caches, before this operation's graph capture.",
            "Memory excludes full-model activation retention, optimizer/EMA and data loading; allocator reservations are not live tensors.",
            "Linear scope includes quantization, forward/backward, and gradient reset; no backbone, optimizer, or data loading.",
            "Quantize/scale/dequantize scopes measure one operation and allocation per operand and independently scaled transpose.",
            "Dequantize scope prepares FP8 operands before timing and does not perform finite validation during reconstruction.",
            "Finite checks are strict by default; --no-check-finite models unchecked runtime updates in supporting versions.",
            "GEMM scope uses prequantized inputs but includes packing, padding, and host dispatch; these are not kernel-only timings.",
            "Forward output is BF16; input gradient follows --input-dtype; master parameters and weight gradients are FP32.",
            "Prepared BF16 products disable autograd and prepare BF16 inputs outside timing; linear scope includes casts and autograd.",
            "Autotune selections are inspected after timing; null means the library exposes no compatible tuner cache.",
            "Candidate quantiles are search-time median/p20/p80 device milliseconds, including split-K reduction when used.",
            "GEMM operands are captured from the imported library's actual autograd path before timing.",
            "Accuracy compares forward/dX/dW with IEEE FP32 after timing; changing accumulation requires training-quality validation.",
            "Prepared FP8 errors separate quantization from accumulation, FP32 scaling and output cast using tiled FP64 references.",
            "Fixed configurations bypass autotuning; compatibility is checked once per key before launch.",
            "ViT preset uses separate qkv/attention/fc1/fc2/head projections; it does not measure a whole ViT model.",
            "Candidate wall times include compilation and benchmarking; native tuning-result persistence is reported separately.",
        ],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", choices=("both", "linear", "gemm", "quantize", "quantize-pair", "scale", "dequantize", "bias-gradient"), default="both")
    parser.add_argument("--check-finite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accuracy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--implementation", choices=("fp8", "bf16", "both"), default="fp8")
    parser.add_argument("--scaling", choices=("blockwise", "tensorwise"), default="blockwise")
    parser.add_argument("--preset", choices=("classifier", "vit"), default="classifier")
    parser.add_argument("--tokens-per-sample", type=int)
    parser.add_argument("--gemm-config", type=int, nargs=7, metavar=("BM", "BN", "BK", "WARPS", "STAGES", "SPLIT", "GROUP"),
                        help="Use one validated FP8 GEMM tile for controlled comparisons; omit to autotune.")
    for role, default in (("activation", "e4m3fn"), ("weight", "e4m3fn"), ("grad", "e5m2")):
        parser.add_argument("--" + role + "-format", choices=("e4m3fn", "e5m2"), default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-stages", choices=("auto", "1", "2", "3", "4", "5"), default="auto",
                        help="Tune pipeline stages by default, or select an explicit depth.")
    parser.add_argument("--batches", nargs="+", type=int,
                        help="Sample batches; defaults are 2048/848/1808 for classifier or 512/336/272 for ViT.")
    parser.add_argument("--input-features", type=int, default=512)
    parser.add_argument("--input-dtype", choices=("bf16", "fp32"), default="bf16",
                        help="Projection input dtype; FP32 tests casts and wider input gradients after FP32 normalization.")
    parser.add_argument("--input-pattern", choices=("random", "cancellation", "zeros", "mixed-magnitude"), default="random")
    parser.add_argument("--output-features", type=int, default=10)
    parser.add_argument("--scale-k", type=int, choices=(32, 64, 128, 256, 512, 1024), default=32,
                        help="Scale width; 128+ permits K=128 FP8 tiles, and larger grids exercise chunked fusion.")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--detailed", action="store_true",
                        help="Add CUDA events, warmed memory peaks, CPU/CUDA profiles and prepared-GEMM graph replay.")
    parser.add_argument("--profile-iterations", type=int, default=2)
    parser.add_argument("--profile-rows", type=int, default=20)
    parser.add_argument("--host-profile", action="store_true", help="With --detailed, profile warmed Python submission separately.")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse the normal Triton disk cache.")
    parser.add_argument("--output", type=Path, help="Also save the complete JSON report to this file.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = args.batches or ([512, 336, 272] if args.preset == "vit" else [2048, 848, 1808])
    args.tokens_per_sample = args.tokens_per_sample if args.tokens_per_sample is not None else (197 if args.preset == "vit" else 1)
    if args.implementation != "fp8" and args.scope not in ("both", "linear", "gemm"):
        parser.error("BF16 comparison supports linear and gemm scopes")
    if args.scope == "quantize-pair" and args.scaling != "tensorwise":
        parser.error("quantize-pair requires --scaling tensorwise")
    if args.gemm_config is not None:
        bm, bn, bk, warps, stages, split, group = args.gemm_config
        if (any(value not in (16, 32, 64, 128, 256) for value in (bm, bn))
                or bk not in (32, 64, 128) or warps not in (2, 4, 8) or stages not in (1, 2, 3, 4, 5)
                or split not in (1, 2, 4, 8) or group not in (1, 8)):
            parser.error("--gemm-config must use supported tile, warp, stage, split and group values")
    if args.tokens_per_sample < 1:
        parser.error("--tokens-per-sample must be positive")
    args.num_stages = args.num_stages if args.num_stages == "auto" else int(args.num_stages)
    if (any(batch < 1 for batch in args.batches) or args.input_features < 1 or args.output_features < 1
            or args.iterations < 1 or args.repeats < 1 or args.warmups < 0
            or args.profile_iterations < 1 or args.profile_rows < 1):
        parser.error("Dimensions, iterations, repeats, and profile limits must be positive; warmups must be nonnegative.")
    if args._worker:
        if args.scope == "both" or args.implementation == "both":
            parser.error("A worker needs one implementation and one scope.")
        print(_RESULT_PREFIX + json.dumps(_worker(args)))
    else:
        serialized = json.dumps(_run(args), indent=2)
        if args.output is not None:
            destination = args.output.expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(serialized + "\n", encoding="utf-8")
            print(f"Benchmark report saved to {destination.resolve()}", file=sys.stderr)
        print(serialized)


if __name__ == "__main__":
    main()
