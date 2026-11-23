"""Benchmark installed DeepOpt pooling precision, compression and complete steps."""

import argparse
from copy import deepcopy
import gc
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter


OPERATIONS = tuple(
    f"{prefix}{rank}d"
    for rank in (1, 2, 3)
    for prefix in ("max_pool", "avg_pool", "adaptive_max_pool", "adaptive_avg_pool", "lp_pool", "max_unpool",
                   *(("fractional_max_pool",) if rank > 1 else ()))
)


def dimensions(text):
    try:
        result = tuple(int(v) for v in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use dimensions such as 2x32x16x16") from error
    if not result or any(v <= 0 for v in result):
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return result


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPERATIONS, default="max_pool2d")
    parser.add_argument("--shape", type=dimensions)
    parser.add_argument("--kernel-size", type=dimensions, default=(2,))
    parser.add_argument("--stride", type=dimensions)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--ceil-mode", action="store_true")
    parser.add_argument("--output-size", type=dimensions, default=(1,))
    parser.add_argument("--norm-type", type=float, default=2)
    parser.add_argument("--return-indices", action="store_true")
    parser.add_argument("--pipeline", choices=("none", "linear", "conv"), default="none")
    parser.add_argument("--inputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    parser.add_argument("--outputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    parser.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--fp8-block", type=dimensions)
    parser.add_argument("--fp8-format", choices=("e4m3fn", "e5m2"), default="e4m3fn")
    parser.add_argument("--scaling-recipe", choices=("current", "delayed"), default="current")
    parser.add_argument("--output-scaling", choices=("current", "provided"), default="current")
    parser.add_argument("--execution", choices=("eager", "graph", "both"), default="eager")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-budget", type=float, default=2.0, help="Warm timing target per variant in seconds; 0 uses exhaustive counts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deepopt-backend", choices=("triton", "tilelang", "mixed"), default="triton")
    parser.add_argument("--deepopt-kernel-policy", type=json.loads, default={},
                        help="JSON runtime.kernels policy; its backend must match --deepopt-backend")
    parser.add_argument("--qualify-tilelang", action="store_true", help="Opt into independent deterministic isolated pooling qualification")
    parser.add_argument("--compression", choices=("off", "on", "both"), default="both")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    arguments = parser()
    args = arguments.parse_args()
    if args.qualify_tilelang and (args.pipeline != "none" or args.scaling_recipe != "current"
            or not args.op.startswith(("max_pool", "avg_pool", "adaptive_max_pool", "adaptive_avg_pool"))):
        arguments.error("--qualify-tilelang requires isolated current-scaling deterministic max/average pooling")
    rank = int(args.op[-2])
    if args.shape is not None and len(args.shape) != rank + 2:
        arguments.error("--shape must be batched N,C followed by the operation's spatial dimensions")
    if min(args.group_size, args.iterations, args.repeats) <= 0 or args.warmup < 0:
        arguments.error("counts must be positive; warmup may be zero")
    if args.fp8_block is not None and len(args.fp8_block) != 2:
        arguments.error("--fp8-block requires two dimensions")
    if args.execution != "eager" and not args.device.startswith("cuda"):
        arguments.error("graph timing requires a CUDA device")
    if args.execution != "eager" and args.scaling_recipe == "delayed":
        arguments.error("delayed scaling requires eager full-step execution")
    if args.scaling_recipe == "delayed" and args.fp8_block is not None:
        arguments.error("delayed scaling requires tensorwise blocks")
    if args.return_indices and not args.op.startswith(("max_pool", "adaptive_max_pool", "fractional_max_pool")):
        arguments.error("--return-indices requires maximum pooling")

    from deepopt._benchmarking import BenchmarkBackend, timing_plan_from_cost
    try:
        selection = BenchmarkBackend.from_args(args)
        timing_plan_from_cost(args, 0.0)
    except (TypeError, ValueError) as error:
        arguments.error(str(error))
    with selection.scope():
        return run(args, arguments, selection)


def run(args, arguments, selection):
    kernel_config = selection.config
    import torch
    from torch import nn
    import deepopt
    from deepopt.opt.nn import functional as D
    from deepopt.opt.nn.pooling.spec import SPECS, MODULE_SPECS
    from deepopt.opt.nn.pooling.geometry import geometry
    from deepopt.opt.policy import prepare_fp8_model
    from deepopt.opt.precision import precision_report
    from deepopt.opt.quantization.activation_tensor import to_fp8_activation, dequantize_activation
    from deepopt.opt.scaling.delayed import DelayedScalingState
    from deepopt.opt._execution import finite_check_scope

    spec = SPECS[args.op]
    shape = args.shape or (8, 32) + (16,) * spec.rank
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
        raise SystemExit("Pooling measurements require CPU or NVIDIA CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    raw = torch.randn(shape, device=device, dtype=torch.bfloat16)
    kernel = args.kernel_size[0] if len(args.kernel_size) == 1 else args.kernel_size
    stride = args.stride[0] if args.stride and len(args.stride) == 1 else args.stride
    size = args.output_size[0] if len(args.output_size) == 1 else args.output_size
    options = {}
    if spec.family.startswith("adaptive"):
        options["output_size"] = size
    else:
        options.update(kernel_size=kernel, stride=stride)
    if spec.family in ("max", "average", "unpool"):
        options["padding"] = args.padding
    if spec.family in ("max", "average", "power"):
        options["ceil_mode"] = args.ceil_mode
    if spec.family == "max":
        options["dilation"] = args.dilation
    if "max" in spec.family:
        options["return_indices"] = args.return_indices
    if spec.family == "power":
        options["norm_type"] = args.norm_type
    if spec.family == "fractional_max":
        options.pop("stride")
        options["output_size"] = size
    if spec.family == "unpool":
        raw, indices = getattr(torch.nn.functional, f"max_pool{spec.rank}d")(
            raw.float(), kernel, stride, args.padding, return_indices=True)
        raw = raw.bfloat16()
        options.update(indices=indices, output_size=shape[-spec.rank:])
    g = geometry(args.op, raw, options, "last")
    native_type = next((cls for cls, item in MODULE_SPECS.items() if item.name == args.op), None)
    if native_type is None:
        raise SystemExit("The installed Torch version does not expose this native module for benchmarking")
    constructor = {k: v for k, v in options.items() if not (spec.family == "unpool" and k in ("indices", "output_size"))}

    collect_indices, last_indices = False, [None]

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = native_type(**constructor)
            if args.pipeline == "conv":
                self.before = getattr(nn, f"Conv{spec.rank}d")(shape[1], shape[1], 1)
                self.after = getattr(nn, f"Conv{spec.rank}d")(shape[1], shape[1], 1)
            elif args.pipeline == "linear":
                self.before = nn.Linear(raw.shape[-1], raw.shape[-1])
                self.after = nn.Linear(g.output_shape[-1], g.output_shape[-1])
            else:
                self.before, self.after = nn.Identity(), nn.Identity()

        def forward(self, x):
            x = self.before(x)
            # Torch has CPU/BF16 gaps in pooling, especially fractional/3D
            # kernels. Keep this explicit FP32 native oracle in the report.
            if not hasattr(self.pool, "plan") and (device.type == "cpu" or spec.family == "fractional_max"):
                x = x.float()
            y = self.pool(x, options["indices"], options["output_size"]) if spec.family == "unpool" else self.pool(x)
            if args.return_indices and "max" in spec.family:
                if collect_indices:
                    last_indices[0] = y[1].detach().cpu()
                y = y[0]
            return self.after(y)

    template = Network().to(device)
    gradient = torch.randn(g.output_shape, device=device, dtype=torch.bfloat16)
    cfg = deepopt.FP8Config(enabled=True, backend=selection.fp8_backend(device), autotune=args.autotune,
                            intercept_linear=False, backward_precision="auto", activation_dtype=args.fp8_format,
                            grad_dtype="e5m2", scaling_recipe=args.scaling_recipe,
                            block_shapes={"activation": args.fp8_block, "weight": None, "grad": None})
    report = dict(deepopt=dict(version=version("deep-optimization"), path=str(Path(deepopt.__file__).resolve())),
                  torch=torch.__version__, device=str(device), input_shape=list(raw.shape), kernel_backend=selection.metadata(device),
                  configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                 if not (k == "qualify_tilelang" and not v) and not (k == "compression" and v == "both")},
                  notes=["Full steps time input FP8 preparation, pooling/RNG, compression, output scales/casts, backward and gradient clearing.",
                         "Native FP8-input baselines include dense reconstruction; optimized kernels decode locally.",
                         "Native CPU pooling and native fractional pooling use an explicit FP32 oracle; CPU timings do not establish CUDA speed.",
                         "Unpool input/indices are assembled before timing. Fractional random samples are generated inside each timed forward.",
                         "Fixed explicit scales are prepared before timing. Pipeline precision uses cached planner decisions; --outputs applies to isolated pooling.",
                         "Peak bytes are absolute allocator measurements; saved bytes count actual hooks without manufacturing retained input state."], results=[])

    def clear(source, model):
        for tensor in (source, *model.parameters()):
            if tensor.grad is not None:
                tensor.grad.zero_()

    def measure(step):
        from deepopt._benchmarking import plan_timing
        if getattr(args, "qualify_tilelang", False):
            from deepopt._tilelang_benchmarking import qualified_timing_plan
            plan_timing = qualified_timing_plan
        began = perf_counter()
        plan = plan_timing(step, args, synchronize=lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None)
        for _ in range(plan.warmup):
            step()
        gc.collect()
        baseline = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if getattr(args, "qualify_tilelang", False):
                baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        wall, events = [], []
        for _ in range(plan.repeats):
            if device.type == "cuda":
                first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device); first.record()
            start = perf_counter()
            for _ in range(plan.iterations):
                step()
            if device.type == "cuda":
                last.record(); last.synchronize()
                events.append(first.elapsed_time(last) / plan.iterations)
            wall.append((perf_counter() - start) * 1000 / plan.iterations)
        result = dict(wall_ms=wall, wall_ms_median=median(wall), cuda_stream_ms=events,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    sampling=plan.sampling(perf_counter() - began))
        if baseline is not None:
            result["memory"] = {"baseline_allocated_bytes": baseline,
                                "incremental_peak_allocated_bytes": max(0, result["peak_allocated_bytes"] - baseline)}
        return result

    def timings(step):
        result = {}
        if args.execution in ("eager", "both"):
            result["eager"] = measure(step)
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
            result["graph"] = measure(graph.replay)
            result["graph"]["capture_warmup_steps"] = capture_steps
            del live, graph
        return result

    class PhaseProfile:
        def start(self):
            if device.type == "cuda":
                self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                self.events[0].record()
            self.begin = perf_counter()

        def split(self):
            if device.type == "cuda":
                self.events[1].record()
            self.middle = perf_counter()

        def finish(self):
            end = perf_counter()
            if device.type == "cuda":
                self.events[2].record()
                self.events[2].synchronize()
            self.result = dict(forward_cpu_ms=(self.middle - self.begin) * 1000,
                               backward_and_flush_cpu_ms=(end - self.middle) * 1000,
                               forward_cuda_stream_ms=self.events[0].elapsed_time(self.events[1]) if device.type == "cuda" else None,
                               backward_and_flush_cuda_stream_ms=self.events[1].elapsed_time(self.events[2]) if device.type == "cuda" else None)

    phases, profile_phases = PhaseProfile(), False

    for storage in args.inputs:
        native, source = deepcopy(template), raw.detach().clone().requires_grad_()
        def native_step(native=native, source=source):
            clear(source, native)
            with finite_check_scope(False), torch.autocast(device.type, dtype=torch.bfloat16):
                if profile_phases:
                    phases.start()
                represented = to_fp8_activation(source, fp8_dtype=getattr(torch, "float8_" + args.fp8_format), block_shape=args.fp8_block) if storage == "fp8" else source
                y = native(dequantize_activation(represented))
                if profile_phases:
                    phases.split()
                y.backward(gradient)
            if profile_phases:
                phases.finish()
            return y
        timing = timings(native_step)
        profile_phases = True
        native_step()
        profile_phases = False
        report["results"].append(dict(implementation="native", input=storage, timing=timing, eager_phase_profile=dict(phases.result)))
        del native_step, native, source
        selected_compression = {"off": (False,), "on": (True,), "both": (False, True)}[getattr(args, "compression", "both")]
        for compressed in selected_compression:
            compression = dict(enabled=compressed, bits=args.bits, group_size=args.group_size)
            model, _ = prepare_fp8_model(deepcopy(template), cfg, config=kernel_config, activation_compression=compression)
            source = raw.detach().clone().requires_grad_()
            for output_type in args.outputs if args.pipeline == "none" else ("planned",):
                output_dtype = torch.bfloat16 if output_type == "bf16" else getattr(torch, "float8_" + args.fp8_format)
                if output_type == "fp8" and args.scaling_recipe == "delayed" and model.pool.fp8_scaling is None:
                    training = model.pool.plan["fp8_training"]
                    model.pool.fp8_scaling = DelayedScalingState(training["delayed_scaling"], training["fp8_dtypes"], device=device)
                rows, cols = g.transport.matrix_shape
                grid = (1, 1) if args.fp8_block is None else ((rows + args.fp8_block[0] - 1) // args.fp8_block[0], (cols + args.fp8_block[1] - 1) // args.fp8_block[1])
                scale = torch.ones(grid, device=device, dtype=torch.float32) if output_type == "fp8" and args.output_scaling == "provided" else None
                info = {}
                def step(source=source, model=model):
                    clear(source, model)
                    with deepopt.fp8_scaling_scope(), deepopt.autocast(device.type, dtype=torch.bfloat16, fp8=cfg,
                            activation_compression=compression, cache_enabled=False, check_finite=False, config=kernel_config):
                        if profile_phases:
                            phases.start()
                        value = to_fp8_activation(source, fp8_dtype=getattr(torch, "float8_" + args.fp8_format), block_shape=args.fp8_block) if storage == "fp8" else source
                        if args.pipeline == "none":
                            y = getattr(D, "do_" + args.op)(value, **options, plan=model.pool.plan, output_dtype=output_dtype,
                                output_layout="last", output_scale=scale, _scaling_state=model.pool.fp8_scaling, _execution_info=info)
                            if isinstance(y, tuple):
                                if collect_indices:
                                    last_indices[0] = y[1].detach().cpu()
                                y = y[0]
                        else:
                            y = model(value)
                        if profile_phases:
                            phases.split()
                        y.backward(gradient)
                    if profile_phases:
                        phases.finish()
                    return y
                timing = timings(step)
                profile_phases = True
                step()
                profile_phases = False
                phase_timing = dict(phases.result)
                saved, storages = [], {}
                def save(tensor):
                    saved.append(tensor.numel() * tensor.element_size())
                    storages[tensor.untyped_storage()._cdata] = tensor.untyped_storage().nbytes()
                    return tensor
                before_state = deepcopy(model.state_dict()) if compressed else None
                rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                collect_indices = True
                with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
                    sample = step()
                qualified_dx = source.grad.detach().cpu().clone() if getattr(args, "qualify_tilelang", False) else None
                actual_indices = last_indices[0]
                execution = dict(info) if args.pipeline == "none" else model.pool.execution_info
                errors = None
                if compressed:
                    actual_output = dequantize_activation(sample).detach().float().cpu()
                    actual_dx = source.grad.detach().float().cpu()
                    after_state = deepcopy(model.state_dict())
                    rng_after = torch.get_rng_state()
                    cuda_rng_after = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                    model.load_state_dict(before_state)
                    torch.set_rng_state(rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, device)
                    compression = {**compression, "enabled": False}
                    oracle = step()
                    oracle_output = dequantize_activation(oracle).detach().float().cpu()
                    oracle_dx = source.grad.detach().float().cpu()
                    errors = dict(output_max_abs_vs_compression_off=(actual_output - oracle_output).abs().max().item(),
                                  dx_max_abs_vs_compression_off=(actual_dx - oracle_dx).abs().max().item(),
                                  dx_relative_l2_vs_compression_off=((actual_dx - oracle_dx).double().norm() /
                                      oracle_dx.double().norm().clamp_min(1e-30)).item())
                    errors = {key: value if isfinite(value) else None for key, value in errors.items()}
                    errors["indices_equal_vs_compression_off"] = bool(torch.equal(actual_indices, last_indices[0])) if actual_indices is not None else None
                    compression = {**compression, "enabled": True}
                    model.load_state_dict(after_state)
                    torch.set_rng_state(rng_after)
                    if cuda_rng_after is not None:
                        torch.cuda.set_rng_state(cuda_rng_after, device)
                    del before_state, after_state, oracle, actual_output, actual_dx, oracle_output, oracle_dx
                collect_indices = False
                report["results"].append(dict(implementation="deepopt", input=storage, output=output_type, compression=compressed,
                    timing=timing, eager_phase_profile=phase_timing, saved_tensor_bytes=sum(saved), saved_storage_bytes=sum(storages.values()),
                    output_finite=bool(torch.isfinite(dequantize_activation(sample).float()).all()),
                    errors=errors, execution=execution,
                    precision_plan=precision_report(model)))
                if getattr(args, "qualify_tilelang", False):
                    from deepopt._tilelang_benchmarking import qualify_pooling
                    began = perf_counter()
                    represented = to_fp8_activation(source.detach(), fp8_dtype=getattr(torch, "float8_" + args.fp8_format),
                        block_shape=args.fp8_block) if storage == "fp8" else source.detach()
                    validation = qualify_pooling(sample, source, represented, gradient, args.op, options,
                        compressed=compressed, compression_errors=errors, actual_indices=actual_indices, actual_dx=qualified_dx)
                    report["results"][-1].update(validation=validation, qualification_wall_seconds=perf_counter() - began,
                                                 status="ok" if validation["status"] == "passed" else "error")
                del sample
            del step, model, source
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
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
    main()
