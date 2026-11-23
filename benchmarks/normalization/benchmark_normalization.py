"""Benchmark installed DeepOpt normalization, xhat compression and FP8 edges."""

import argparse
from contextlib import nullcontext
import gc
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def shape(text):
    try:
        values = tuple(int(value) for value in text.lower().split("x"))
    except ValueError:
        values = ()
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("use positive dimensions separated by x")
    return values


def compression_cases(args):
    yield False, False
    if getattr(args, "compression", "both") == "off":
        return
    if args.norm_xhat_recenter in ("off", "both"):
        yield True, False
    if args.norm_xhat_recenter in ("on", "both"):
        yield True, True


def codec_diagnostics(represented, args, suffix, recenter):
    """Independent CPU codec accuracy diagnostic, outside all timed regions."""
    import torch
    from deepopt.opt.nn.normalization.geometry import NormGeometry
    from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
    from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

    geometry = (LayerNormGeometry(tuple(represented.shape), suffix) if suffix is not None else
                NormGeometry("batch" if args.op == "batch_norm" else "group", tuple(represented.shape), args.groups))
    values = geometry.matrix(represented.detach().float().cpu())
    variance, mean = torch.var_mean(values, 1, correction=0)
    h = (values - mean[:, None]) * torch.rsqrt(variance + geometry.eps)[:, None]
    group = min(args.group_size, geometry.elements)
    q = quantize_pack_activation(h, group, (1, group), bits=args.bits)
    decoded = dequantize_unpack_activation(q, dtype=torch.float32)
    raw_mean_error = (decoded.mean(1) - h.mean(1)).abs().max().item()
    if suffix is not None:
        decoded -= decoded.mean(1, keepdim=True)
        decoded_m2 = decoded.square().mean(1)
        gain = torch.where(decoded_m2 > 0, torch.sqrt(h.square().mean(1) / decoded_m2.clamp_min(1e-12)), 0.)
        decoded *= gain[:, None]
    elif recenter:
        decoded -= (decoded - h).mean(1, keepdim=True)
    return {"uncorrected_mean_max_abs_error": raw_mean_error,
            "mean_max_abs_error": (decoded.mean(1) - h.mean(1)).abs().max().item(),
            "layer_norm_xhat_moment_correction": suffix is not None,
            "second_moment_max_abs_error": (decoded.square().mean(1) - h.square().mean(1)).abs().max().item(),
            "xhat_relative_l2": ((decoded - h).double().norm() / h.double().norm().clamp_min(1e-30)).item()}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--op", choices=("batch_norm", "group_norm", "layer_norm"), default="batch_norm")
    result.add_argument("--deepopt-backend", choices=("triton", "tilelang", "mixed"), default="triton")
    result.add_argument("--deepopt-kernel-policy", type=json.loads, default={}, metavar="JSON")
    result.add_argument("--qualify-tilelang", action="store_true", help="Opt into independent isolated LayerNorm qualification")
    result.add_argument("--compression", choices=("off", "on", "both"), default="both")
    result.add_argument("--normalized-shape", type=shape, help="LayerNorm suffix; defaults to the final dimension")
    result.add_argument("--outputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"], help="Isolated LayerNorm outputs")
    result.add_argument("--fp8-block", type=shape, help="LayerNorm FP8 activation scale tile")
    result.add_argument("--fp8-format", choices=("e4m3fn", "e5m2"), default="e4m3fn")
    result.add_argument("--activation", choices=("none", "relu", "sigmoid", "silu", "gelu", "gelu_tanh", "tanh", "relu6"), default="none", help="Optional LayerNorm pipeline activation")
    result.add_argument("--execution", choices=("eager", "graph", "both"), default="eager", help="Normalization timing mode")
    result.add_argument("--kernel-variant", choices=("auto", "baseline", "candidate", "batch_shared", "small_forward", "layer_fused", "layer_tiled"), default="auto", help="Triton schedule controls; non-auto values require Triton normalization routes")
    result.add_argument("--row-chunk", type=int, choices=(1, 2, 4, 8, 16), default=4, help="Candidate LayerNorm rows per chunk")
    result.add_argument("--max-programs", type=int, default=512, help="Experimental program cap; 0 selects by SM count, new variants bound scratch to 8 MiB")
    result.add_argument("--scope-lifetime", choices=("fresh", "reused"), default="fresh")
    result.add_argument("--phase-steps", type=int, default=0, help="Separate eager forward/backward diagnostic iterations")
    result.add_argument("--profile-steps", type=int, default=0, help="Separate profiler iterations; not part of throughput")
    result.add_argument("--profile-dir", type=Path)
    result.add_argument("--output-scaling", choices=("current", "provided"), default="current", help="Isolated LayerNorm fixed-scale comparison")
    result.add_argument("--shape", type=shape, default=(16, 64, 56, 56))
    result.add_argument("--groups", type=int, default=32)
    result.add_argument("--inputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    result.add_argument("--group-size", type=int, default=1024)
    result.add_argument("--norm-xhat-recenter", choices=("off", "on", "both"), default="off",
                        help="Compare compressed norm backward with mean correction disabled/enabled")
    result.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    result.add_argument("--pipeline", choices=("none", "conv", "linear"), default="conv")
    result.add_argument("--layout", choices=("contiguous", "channels_last"), default="contiguous")
    result.add_argument("--scaling-recipe", choices=("current", "delayed"), default="current")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=30)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--time-budget", type=float, default=2.0, help="Warm timing target per variant in seconds; 0 uses exhaustive counts")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=14)
    result.add_argument("--output", type=Path)
    return result


def variant_scope(args):
    if args.kernel_variant == "auto":
        return nullcontext()
    from deepopt.opt.backends._shared.normalization import normalization_kernel_scope
    return normalization_kernel_scope(args.kernel_variant, row_chunk=args.row_chunk, max_programs=args.max_programs)


def measure_modes(step, args, torch, device):
    def measure(call):
        from deepopt._benchmarking import plan_timing
        began = perf_counter()
        if getattr(args, "qualify_tilelang", False):
            from deepopt._tilelang_benchmarking import qualified_timing_plan
            plan_timing = qualified_timing_plan
        plan = plan_timing(call, args, synchronize=lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None)
        for _ in range(plan.warmup):
            call()
        baseline = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if getattr(args, "qualify_tilelang", False):
                baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        wall, cuda = [], []
        for _ in range(plan.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                first.record()
            start = perf_counter()
            for _ in range(plan.iterations):
                call()
            if device.type == "cuda":
                last.record()
                last.synchronize()
                cuda.append(first.elapsed_time(last) / plan.iterations)
            wall.append((perf_counter() - start) * 1000 / plan.iterations)
        result = {"wall_ms": wall, "wall_ms_median": median(wall), "cuda_stream_ms": cuda,
                "cuda_stream_ms_median": median(cuda) if cuda else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                "sampling": plan.sampling(perf_counter() - began)}
        if baseline is not None:
            result["memory"] = {"baseline_allocated_bytes": baseline,
                                "incremental_peak_allocated_bytes": max(0, result["peak_allocated_bytes"] - baseline)}
        return result

    records = {}
    if args.execution in ("eager", "both"):
        records["eager"] = measure(step)
    if args.execution in ("graph", "both"):
        from deepopt._benchmarking import capture_warmup_steps
        capture_steps = capture_warmup_steps(args)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(capture_steps):
                step()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            live = step()
        records["graph"] = measure(graph.replay)
        records["graph"]["capture_warmup_steps"] = capture_steps
        torch.cuda.synchronize(device)
        del graph, live
    return records


def diagnostics(forward, gradient, clear, args, torch, device, label):
    phase = {"forward_host_ms": [], "backward_host_ms": [], "forward_cuda_stream_ms": [], "backward_cuda_stream_ms": []}
    for _ in range(args.phase_steps):
        clear()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            marks[0].record()
        start = perf_counter()
        output = forward()
        middle = perf_counter()
        if device.type == "cuda":
            marks[1].record()
        output.backward(gradient)
        end = perf_counter()
        if device.type == "cuda":
            marks[2].record()
            marks[2].synchronize()
            phase["forward_cuda_stream_ms"].append(marks[0].elapsed_time(marks[1]))
            phase["backward_cuda_stream_ms"].append(marks[1].elapsed_time(marks[2]))
        phase["forward_host_ms"].append((middle-start)*1000)
        phase["backward_host_ms"].append((end-middle)*1000)
    result = {"phase_medians": {key: median(value) if value else None for key,value in phase.items()},
              "phase_samples": phase, "note": "Separate eager diagnostics, including input preparation; stream intervals include host gaps."}
    if args.profile_steps:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        clear()
        forward().backward(gradient)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as capture:
            for _ in range(args.profile_steps):
                clear()
                forward().backward(gradient)
                capture.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        directory = args.profile_dir or ((args.output.parent / (args.output.stem + '-profiles')) if args.output else Path('normalization-profiles'))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (label + '.json')
        capture.export_chrome_trace(str(path))
        kernels = {}
        for event in capture.events():
            if (event.device_type == torch.autograd.DeviceType.CUDA
                    and not getattr(event, "is_user_annotation", False)
                    and not event.name.startswith(("Memcpy", "Memset", "[memory]"))):
                row = kernels.setdefault(event.name, {"launches": 0, "ms_per_step": 0.0})
                row["launches"] += 1
                row["ms_per_step"] += event.device_time_total / 1000 / args.profile_steps
        result["profile"] = {"trace": str(path), "kernel_launches_per_step": sum(v['launches'] for v in kernels.values()) / args.profile_steps,
                             "kernels": kernels, "steps": args.profile_steps}
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


def main():
    arguments = parser()
    args = arguments.parse_args()
    if args.qualify_tilelang and (args.op != "layer_norm" or args.pipeline != "none" or args.activation != "none"
                                 or args.scaling_recipe != "current"):
        arguments.error("--qualify-tilelang currently requires isolated current-scaling LayerNorm without an activation")
    if args.op == "batch_norm" and len(args.shape) != 4:
        arguments.error("BatchNorm2d requires four dimensions")
    if args.groups <= 0 or args.op == "group_norm" and (len(args.shape) < 2 or args.shape[1] % args.groups):
        arguments.error("--groups must be positive and divide channels")
    if args.pipeline == "conv" and len(args.shape) not in (3, 4, 5):
        arguments.error("the convolution pipeline requires 3, 4 or 5 dimensions")
    if args.layout == "channels_last" and len(args.shape) not in (4, 5):
        arguments.error("channels_last requires 4 or 5 dimensions")
    if min(args.group_size, args.iterations, args.repeats) <= 0 or min(args.warmup,args.phase_steps,args.profile_steps) < 0:
        arguments.error("counts must be positive; warmup may be zero")
    if not 0 <= args.max_programs <= 4096:
        arguments.error("--max-programs must be 0..4096")
    if args.execution != "eager" and not args.device.startswith("cuda"):
        arguments.error("CUDA Graph timing requires a CUDA device")
    if args.execution != "eager" and args.scaling_recipe == "delayed":
        arguments.error("delayed scale updates require eager execution")
    if args.op == "layer_norm":
        suffix = args.normalized_shape or (args.shape[-1],)
        if len(suffix) > len(args.shape) or tuple(args.shape[-len(suffix):]) != suffix:
            arguments.error("--normalized-shape must match the exact input suffix")
        if args.fp8_block is not None and len(args.fp8_block) != 2:
            arguments.error("--fp8-block requires two dimensions")
        if args.execution != "eager" and not args.device.startswith("cuda"):
            arguments.error("CUDA Graph timing requires a CUDA device")
        if args.execution != "eager" and args.scaling_recipe == "delayed":
            arguments.error("delayed scale updates do not support full-training CUDA Graph replay")
        if args.scaling_recipe == "delayed" and args.fp8_block is not None:
            arguments.error("delayed scaling requires tensorwise FP8 scaling")
    from deepopt._benchmarking import BenchmarkBackend, timing_plan_from_cost

    try:
        kernels = BenchmarkBackend.from_args(args)
        timing_plan_from_cost(args, 0.0)
    except (TypeError, ValueError) as error:
        arguments.error(str(error))
    if args.kernel_variant != "auto":
        for role in ("normalization.forward", "normalization.backward"):
            if (kernels.policy.route("normalization", role) != "triton"
                    or kernels.policy.fallback == "other" and not kernels.policy.pinned("normalization", role)):
                arguments.error("non-auto --kernel-variant is Triton-only; pin both normalization routes to Triton")
    if args.op == "layer_norm":
        lease_code = _lease_cli(args)
        if lease_code is not None:
            return lease_code
        with kernels.scope(), variant_scope(args):
            return layer_main(args, suffix, kernels)
    lease_code = _lease_cli(args)
    if lease_code is not None:
        return lease_code
    with kernels.scope(), variant_scope(args):
        return norm_main(args, kernels)


def norm_main(args, kernels):
    kernel_config = kernels.config

    import torch
    from torch import nn
    import deepopt
    from deepopt.opt.policy import prepare_fp8_model
    from deepopt.opt.precision import precision_report
    from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for normalization performance measurements")
    torch.cuda.set_device(device)
    report = {"deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
              "kernel_backend": kernels.metadata(device),
              "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                                if not (key == "qualify_tilelang" and not value) and not (key == "compression" and value == "both")},
              "notes": ["Input FP8 casting, normalization, xhat packing/decoding and backward are timed; no optimizer. Graph replay is labeled separately.",
                        "Precision capability analysis occurs during preparation, outside warmup and timing.",
                        "FP8 uses scale-aware storage with FP32 normalization reductions.",
                        "Output and gradient errors compare compression off/on at the same input precision.",
                        "Current/delayed output casting uses the shared FP8 quantizer after fused normalization/xhat packing.",
                        "Saved bytes count logical payloads without deduplicating aliases; CUDA intervals include submission gaps."],
              "results": []}
    for storage in args.inputs:
        torch.manual_seed(args.seed)
        native_norm = nn.BatchNorm2d(args.shape[1]) if args.op == "batch_norm" else nn.GroupNorm(args.groups, args.shape[1])
        native_layers = [native_norm]
        if args.pipeline == "conv":
            native_layers.extend((nn.ReLU(), getattr(nn, f"Conv{len(args.shape) - 2}d")(args.shape[1], args.shape[1], 3, padding=1)))
        elif args.pipeline == "linear":
            native_layers.append(nn.Linear(args.shape[-1], args.shape[-1]))
        native_model = nn.Sequential(*native_layers).to(device)
        native_source = torch.randn(args.shape, device=device, dtype=torch.bfloat16)
        if args.layout == "channels_last":
            native_source = native_source.to(memory_format=torch.channels_last if native_source.ndim == 4 else torch.channels_last_3d)
        native_source.requires_grad_()
        native_amp = torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False) if args.scope_lifetime == "reused" else None

        def native_forward(native_model=native_model, native_source=native_source, native_amp=native_amp, storage=storage):
            value = to_fp8_activation(native_source, fp8_dtype=getattr(torch, "float8_" + args.fp8_format)).dequantize(torch.bfloat16) if storage == "fp8" else native_source
            with native_amp if native_amp is not None else torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False):
                return native_model(value)

        with torch.no_grad():
            native_gradient = torch.randn_like(native_forward())

        def native_clear(native_model=native_model, native_source=native_source):
            native_model.zero_grad(set_to_none=False)
            if native_source.grad is not None:
                native_source.grad.zero_()

        def native_step(native_forward=native_forward, native_clear=native_clear, native_gradient=native_gradient):
            native_clear()
            y = native_forward()
            y.backward(native_gradient)
            return y

        native_timing = measure_modes(native_step, args, torch, device)
        native_diagnostic = diagnostics(native_forward, native_gradient, native_clear, args, torch, device,
            f"{args.op}-{storage}-native")
        report["results"].append({"implementation": "native_bf16", "input": storage,
                                  "timing": native_timing, "diagnostics": native_diagnostic})
        del native_model, native_source, native_gradient, native_step, native_forward, native_clear, native_amp, native_norm, native_layers
        gc.collect()
        torch.cuda.empty_cache()
        baseline = None
        for compressed, recenter in compression_cases(args):
            torch.manual_seed(args.seed)
            norm = nn.BatchNorm2d(args.shape[1]) if args.op == "batch_norm" else nn.GroupNorm(args.groups, args.shape[1])
            layers = [norm]
            if args.pipeline == "conv":
                layers.extend((nn.ReLU(), getattr(nn, f"Conv{len(args.shape) - 2}d")(args.shape[1], args.shape[1], 3, padding=1)))
            elif args.pipeline == "linear":
                layers.append(nn.Linear(args.shape[-1], args.shape[-1]))
            model = nn.Sequential(*layers).to(device)
            settings = deepopt.FP8Config(enabled=True, backend=kernels.fp8_backend(device), autotune=args.autotune, intercept_linear=False,
                                        scaling_recipe=args.scaling_recipe, activation_dtype=args.fp8_format)
            model, _ = prepare_fp8_model(model, settings, activation_compression={
                "enabled": compressed, "group_size": args.group_size, "bits": args.bits, "norm_xhat_recenter": recenter}, config=kernel_config)
            source = torch.randn(args.shape, device=device, dtype=torch.bfloat16)
            if args.layout == "channels_last":
                source = source.to(memory_format=torch.channels_last if source.ndim == 4 else torch.channels_last_3d)
            source.requires_grad_()

            amp_context = (deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings, cache_enabled=False,
                check_finite=False, activation_compression={"enabled": compressed, "group_size": args.group_size,
                "bits": args.bits, "norm_xhat_recenter": recenter}, config=kernel_config) if args.scope_lifetime == "reused" else None)
            def context(amp_context=amp_context, settings=settings, compressed=compressed, recenter=recenter):
                return amp_context if amp_context is not None else deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings, cache_enabled=False,
                    check_finite=False, activation_compression={"enabled": compressed, "group_size": args.group_size, "bits": args.bits, "norm_xhat_recenter": recenter}, config=kernel_config)

            with torch.no_grad(), context():
                gradient = torch.randn_like(model(source))

            def clear(model=model, source=source):
                model.zero_grad(set_to_none=False)
                if source.grad is not None:
                    source.grad.zero_()

            def forward(model=model, source=source, storage=storage, context=context):
                with context():
                    value = to_fp8_activation(source, fp8_dtype=getattr(torch, "float8_" + args.fp8_format)) if storage == "fp8" else source
                    return model(value)

            def step(model=model, source=source, gradient=gradient, storage=storage, clear=clear, context=context):
                clear()
                with context():
                    value = to_fp8_activation(source, fp8_dtype=getattr(torch, "float8_" + args.fp8_format)) if storage == "fp8" else source
                    output = model(value)
                    output.backward(gradient)
                return output

            timing = measure_modes(step, args, torch, device)
            primary = timing.get("eager", timing.get("graph"))
            wall, event = primary["wall_ms"], primary["cuda_stream_ms"]
            model.zero_grad(set_to_none=True)
            source.grad = None
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            saved = []

            def save(value):
                tensors = (value._data, value._scale_inv) if isinstance(value, FP8Activation) else (value,)
                saved.append(sum(t.numel() * t.element_size() for t in tensors))
                return value

            with torch.autograd.graph.saved_tensors_hooks(save, lambda value: value):
                output = step()
            torch.cuda.synchronize(device)
            peak_extra = torch.cuda.max_memory_allocated(device) - allocated
            diagnostic = diagnostics(forward, gradient, clear, args, torch, device,
                f"{args.op}-{storage}-b{args.bits}-g{args.group_size}-{compressed}-{recenter}-{args.kernel_variant}")
            row = {"implementation": "deepopt", "timing": timing, "diagnostics": diagnostic, "input": storage, "compression": compressed, "norm_xhat_recenter": recenter, "wall_ms_median": median(wall), "wall_ms": wall,
                   "cuda_stream_ms_median": median(event), "saved_tensor_bytes": sum(saved),
                   "peak_extra_allocated_bytes": peak_extra,
                   "precision_plan": precision_report(model), "normalization": model[0].execution_info}
            if args.pipeline == "conv":
                row["relu"] = model[1].execution_info
            snapshot = {"output": output.detach().float().cpu(), "dx": source.grad.float().cpu(),
                        **{name: parameter.grad.float().cpu() for name, parameter in model.named_parameters() if parameter.grad is not None}}
            if baseline is None:
                baseline = snapshot
            else:
                row["errors_vs_compression_off"] = {
                    name: {"max_abs": (value - baseline[name]).abs().max().item(),
                           "relative_l2": ((value - baseline[name]).double().norm() / baseline[name].double().norm().clamp_min(1e-30)).item()}
                    for name, value in snapshot.items()}
            if compressed:
                represented = to_fp8_activation(source.detach(), fp8_dtype=getattr(torch, "float8_" + args.fp8_format)).dequantize(torch.float32) if storage == "fp8" else source.detach()
                row["reference_codec_diagnostics"] = codec_diagnostics(represented, args, None, recenter)
                del represented
            report["results"].append(row)
            print(f"{storage} compression={compressed} wall={median(wall):.4f} ms saved={sum(saved):,} bytes", flush=True)
            del model, source, output, gradient, step, forward, clear, context, amp_context
            gc.collect()
            torch.cuda.empty_cache()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


def layer_main(args, suffix, kernels):
    """Installed-library LayerNorm comparisons; no neighboring script imports."""
    kernel_config = kernels.config
    from copy import deepcopy
    import torch
    from torch import nn
    import deepopt
    from deepopt.opt.nn import functional as doF
    from deepopt.opt.policy import prepare_fp8_model
    from deepopt.opt.precision import precision_report
    from deepopt.opt.quantization.activation_tensor import to_fp8_activation, dequantize_activation
    from deepopt.opt.scaling.delayed import DelayedScalingState

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise SystemExit("LayerNorm measurements require CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.version.hip:
            raise SystemExit("An NVIDIA CUDA device is required")
        torch.cuda.set_device(device)
    cfg = deepopt.FP8Config(enabled=True, backend=kernels.fp8_backend(device), autotune=args.autotune,
                            activation_dtype=args.fp8_format, grad_dtype="e5m2", intercept_linear=False,
                            backward_precision="auto", scaling_recipe=args.scaling_recipe,
                            block_shapes={"activation": args.fp8_block, "weight": None, "grad": None})
    report = {"deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
              "torch": torch.__version__, "device": str(device), "normalized_shape": suffix,
              "kernel_backend": kernels.metadata(device),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                if not (k == "qualify_tilelang" and not v) and not (k == "compression" and v == "both")},
              "notes": ["Complete forward/backward includes input FP8 conversion, statistics, xhat packing, output scaling/casts and gradient clearing.",
                        "Fixed explicit output scales are prepared outside timing; ordinary scale preparation is timed.",
                        "Pipeline output precision and FP8 dx use the cached planner; --outputs applies to isolated LayerNorm.",
                        "Current scaling and provided scales can use graph replay; delayed state updates are eager only.",
                        "Peak allocation is absolute and includes retained model/input state; numerical diagnostics are outside timing.",
                        "Mean-error diagnostics use an independent CPU codec oracle outside timing; correction production/reductions/backward are timed.",
                        "CPU uses the reference implementation and does not establish CUDA compilation or throughput."], "results": []}

    def clear(values):
        for value in values:
            if value.grad is not None:
                value.grad.zero_()

    def timings(step):
        return measure_modes(step, args, torch, device)

    for storage in args.inputs:
        torch.manual_seed(args.seed)
        layers = [nn.LayerNorm(suffix)]
        activation = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "silu": nn.SiLU, "gelu": nn.GELU,
                      "gelu_tanh": lambda: nn.GELU(approximate="tanh"), "tanh": nn.Tanh, "relu6": nn.ReLU6}.get(args.activation)
        if activation is not None:
            layers.append(activation())
        if args.pipeline == "linear":
            layers.append(nn.Linear(args.shape[-1], args.shape[-1]))
        elif args.pipeline == "conv":
            layers.append(getattr(nn, f"Conv{len(args.shape) - 2}d")(args.shape[1], args.shape[1], 1))
        template = nn.Sequential(*layers).to(device)
        raw = torch.randn(args.shape, device=device, dtype=torch.bfloat16)
        if args.layout == "channels_last":
            raw = raw.to(memory_format=torch.channels_last if raw.ndim == 4 else torch.channels_last_3d)
        gradient = torch.randn_like(raw)
        native = deepcopy(template)
        source = raw.detach().clone().requires_grad_()

        def native_step(source=source, native=native):
            clear((source, *native.parameters()))
            # Native baseline receives the same represented input; both timed
            # paths include the caller's requested FP8 input preparation.
            value = to_fp8_activation(source, layout="last", fp8_dtype=getattr(torch, "float8_" + args.fp8_format), block_shape=args.fp8_block) if storage == "fp8" else source
            value = dequantize_activation(value)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                y = native(value)
            y.backward(gradient)
            return y
        report["results"].append({"implementation": "native_bf16", "input": storage, "timing": timings(native_step)})
        del native, source, native_step
        baseline = {}
        for compressed, recenter in compression_cases(args):
            baseline_only = getattr(args, "compression", "both") == "on" and not compressed
            compression = {"enabled": compressed, "group_size": args.group_size, "bits": args.bits, "norm_xhat_recenter": recenter}
            model, _ = prepare_fp8_model(deepcopy(template), cfg, activation_compression=compression, config=kernel_config)
            source = raw.detach().clone().requires_grad_()
            norm = model[0]
            isolated = args.pipeline == "none" and args.activation == "none"
            for output_type in args.outputs if isolated else ("planned",):
                dtype = torch.bfloat16 if output_type == "bf16" else getattr(torch, "float8_" + args.fp8_format)
                if isolated and output_type == "fp8" and args.scaling_recipe == "delayed" and norm.fp8_scaling is None:
                    training = norm.plan["fp8_training"]
                    norm.fp8_scaling = DelayedScalingState(training["delayed_scaling"], training["fp8_dtypes"], device=device)
                rows, cols = raw.numel() // args.shape[-1], args.shape[-1]
                grid = (1, 1) if args.fp8_block is None else ((rows + args.fp8_block[0] - 1) // args.fp8_block[0], (cols + args.fp8_block[1] - 1) // args.fp8_block[1])
                scale = torch.ones(grid, device=device) if isolated and output_type == "fp8" and args.output_scaling == "provided" else None
                info = {}

                def make_context():
                    return deepopt.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16,
                        fp8=cfg, activation_compression=compression, check_finite=False, cache_enabled=False, config=kernel_config)
                amp_context = make_context() if args.scope_lifetime == "reused" else None

                def compute():
                    value = to_fp8_activation(source, layout="last", fp8_dtype=getattr(torch, "float8_" + args.fp8_format), block_shape=args.fp8_block) if storage == "fp8" else source
                    return (doF.layer_norm(value, suffix, norm.weight, norm.bias, norm.eps, plan=norm.plan,
                            output_dtype=dtype, output_scale=scale, _scaling_state=norm.fp8_scaling,
                            _execution_info=info) if isolated else model(value))

                def forward():
                    with deepopt.fp8_scaling_scope(), (amp_context if amp_context is not None else make_context()):
                        return compute()

                def step():
                    clear((source, *model.parameters()))
                    with deepopt.fp8_scaling_scope(), (amp_context if amp_context is not None else make_context()):
                        y = compute()
                        y.backward(gradient)
                    return y
                timing = None if baseline_only else timings(step)
                diagnostic = None if baseline_only else diagnostics(forward, gradient, lambda: clear((source, *model.parameters())),
                    args, torch, device, f"layer-{storage}-{output_type}-b{args.bits}-g{args.group_size}-{compressed}-{recenter}-{args.kernel_variant}")
                sample = step()
                native_output, dx = dequantize_activation(sample).detach().float(), source.grad.detach().float()
                errors = None
                if compressed:
                    previous_y, previous_dx = baseline[output_type]
                    errors = {"output_max_abs_vs_compression_off": (native_output.cpu() - previous_y).abs().max().item(),
                              "dx_max_abs_vs_compression_off": (dx.cpu() - previous_dx).abs().max().item()}
                else:
                    baseline[output_type] = native_output.cpu().clone(), dx.cpu().clone()
                if baseline_only:
                    del sample, native_output, dx
                    continue
                report["results"].append({"implementation": "deepopt", "input": storage, "output": output_type,
                                          "compression": compressed, "norm_xhat_recenter": recenter, "timing": timing, "diagnostics": diagnostic, "errors": errors,
                                          "reference_codec_diagnostics": codec_diagnostics(
                                              to_fp8_activation(raw, layout="last", fp8_dtype=getattr(torch, "float8_" + args.fp8_format), block_shape=args.fp8_block).dequantize(torch.float32)
                                              if storage == "fp8" else raw, args, suffix, recenter) if compressed else None,
                                          "execution": dict(info) if isolated else norm.execution_info,
                                          "precision": precision_report(model)})
                if getattr(args, "qualify_tilelang", False):
                    from deepopt._tilelang_benchmarking import qualify_layer_norm
                    began = perf_counter()
                    validation = qualify_layer_norm(forward, source, model, gradient, suffix, compressed=compressed)
                    report["results"][-1].update(validation=validation, qualification_wall_seconds=perf_counter() - began,
                                                 status="ok" if validation["status"] == "passed" else "error")
                del sample, native_output, dx
    if getattr(args, "qualify_tilelang", False):
        report["qualification_status"] = "passed" if all(row.get("validation", {}).get("status") == "passed"
            for row in report["results"] if row["implementation"] == "deepopt") else "failed"
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if report.get("qualification_status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
