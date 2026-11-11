"""Measure installed DeepOpt activations, derivative compression and FP8 transport."""

import argparse
from copy import deepcopy
import gc
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter


VARIANTS = ("sigmoid", "silu", "gelu", "gelu_tanh", "tanh", "relu6")


def shape(text):
    try:
        values = tuple(int(v) for v in text.lower().split("x"))
    except ValueError:
        values = ()
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("use positive dimensions, for example 4096x768")
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--activations", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    result.add_argument("--shape", type=shape, default=(4096, 768))
    result.add_argument("--inputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    result.add_argument("--outputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    result.add_argument("--compression", nargs="+", choices=("off", "on"), default=["off", "on"])
    result.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    result.add_argument("--group-size", type=int, default=512)
    result.add_argument("--fp8-block", type=shape)
    result.add_argument("--output-scaling", choices=("current", "provided"), default="current")
    result.add_argument("--execution", choices=("eager", "graph", "both"), default="eager")
    result.add_argument("--pipeline", choices=("none", "linear", "conv"), default="none")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=100)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--output", type=Path)
    return result


def measure(step, args, torch, device):
    for _ in range(args.warmup):
        step()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    walls, intervals = [], []
    for _ in range(args.repeats):
        if device.type == "cuda":
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            first.record()
        start = perf_counter()
        for _ in range(args.iterations):
            step()
        if device.type == "cuda":
            last.record()
            last.synchronize()
            intervals.append(first.elapsed_time(last) / args.iterations)
        walls.append((perf_counter() - start) * 1000 / args.iterations)
    return {"wall_ms": walls, "wall_ms_median": median(walls), "cuda_stream_ms": intervals,
            "cuda_stream_ms_median": median(intervals) if intervals else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def measurements(step, args, torch, device):
    report = {}
    if args.execution in ("eager", "both"):
        report["eager"] = measure(step, args, torch, device)
    if args.execution in ("graph", "both"):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(max(3, args.warmup)):
                step()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = step()
        report["graph"] = measure(graph.replay, args, torch, device)
        # Keep outputs and captured autograd buffers live through the measurement.
        del captured_output, graph
    return report


def main():
    arguments = parser()
    args = arguments.parse_args()
    if min(args.iterations, args.repeats, args.group_size) < 1 or args.warmup < 0:
        arguments.error("iteration/repeat/group counts must be positive; warmup may be zero")
    if args.fp8_block is not None and len(args.fp8_block) != 2:
        arguments.error("--fp8-block requires two dimensions")
    if args.pipeline == "conv" and len(args.shape) not in (3, 4, 5):
        arguments.error("conv pipelines require 3, 4 or 5 dimensions")
    if args.execution != "eager" and not args.device.startswith("cuda"):
        arguments.error("CUDA Graph measurement requires a CUDA device")

    import torch
    from torch import nn
    from torch.nn import functional as F
    import deepopt
    from deepopt.opt.nn import functional as doF
    from deepopt.opt.policy import activation_plan, prepare_fp8_model
    from deepopt.opt.precision import precision_report
    from deepopt.opt.quantization.activation_tensor import to_fp8_activation, dequantize_activation
    from deepopt.opt.backends.reference.activation.math import evaluate

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        arguments.error("--device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.version.hip:
            raise SystemExit("An NVIDIA CUDA device is required")
        torch.cuda.set_device(device)
    torch.manual_seed(137)
    backend = "triton" if device.type == "cuda" else "reference"
    report = {"deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
              "torch": torch.__version__, "device": str(device),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "notes": ["Timing includes derivative preprocessing/packing, output scale preparation, backward decoding/casts and gradient clearing.",
                        "Operation input FP8 preparation is outside timing and uses leaf wrappers; pipeline input casting stays inside timing.",
                        "Provided scales are caller-owned fixed FP32 multipliers of one; their creation is outside timing.",
                        "Pipeline precision is selected by the planner; --inputs/--outputs apply to isolated operations.",
                        "Reported peak allocation is the absolute allocator peak for that measurement, including retained state.",
                        "Finite diagnostics are disabled; CPU measurements validate portability, not CUDA performance."],
              "results": []}
    def clear(values):
        for value in values:
            if value.grad is not None:
                value.grad.zero_()

    for variant in args.activations:
        kind, approximate = ("gelu", "tanh") if variant == "gelu_tanh" else (variant, "none")
        native_options = {"approximate": approximate} if kind == "gelu" else {}
        data = torch.randn(args.shape, device=device, dtype=torch.bfloat16)
        if args.pipeline != "none":
            module = {"sigmoid": nn.Sigmoid, "silu": nn.SiLU, "gelu": nn.GELU,
                      "tanh": nn.Tanh, "relu6": nn.ReLU6}[kind]
            width = args.shape[1] if args.pipeline == "conv" else args.shape[-1]
            projection = (lambda: getattr(nn, f"Conv{len(args.shape) - 2}d")(width, width, 1)) if args.pipeline == "conv" else lambda: nn.Linear(width, width)
            original = nn.Sequential(projection(), module(**native_options), projection()).to(device)
            native_model = deepcopy(original)
            native_x, native_dy = data.clone().requires_grad_(), torch.ones_like(data)
            def native_pipeline_step():
                clear((native_x, *native_model.parameters()))
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    y = native_model(native_x)
                y.backward(native_dy)
                return y
            report["results"].append({"activation": variant, "implementation": "native_bf16_pipeline", "pipeline": args.pipeline,
                                      "timing": measurements(native_pipeline_step, args, torch, device)})
            del native_model, native_x, native_dy
            for compressed in args.compression:
                source = deepcopy(original)
                cfg = deepopt.FP8Config(enabled=True, backend=backend, backward_precision="auto", grad_dtype="e5m2",
                                       block_shapes={"activation": args.fp8_block, "weight": None, "grad": None})
                model, _ = prepare_fp8_model(source, cfg, activation_compression={"enabled": compressed == "on", "bits": args.bits, "group_size": args.group_size})
                x = data.clone().requires_grad_()
                dy = torch.ones_like(x)
                def pipeline_step():
                    clear((x, *model.parameters()))
                    with deepopt.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16,
                                          fp8=cfg, check_finite=False):
                        y = model(x)
                    y.backward(dy)
                    return y
                timing = measurements(pipeline_step, args, torch, device)
                report["results"].append({"activation": variant, "implementation": "deepopt_pipeline", "pipeline": args.pipeline,
                                          "compression": compressed, "timing": timing, "precision": precision_report(model),
                                          "activation_execution": [m.execution_info for m in model.modules() if getattr(m, "kind", None) == kind]})
            continue
        for input_type in args.inputs:
            x = (to_fp8_activation(data, layout="last", block_shape=args.fp8_block).detach().requires_grad_()
                 if input_type == "fp8" else data.clone().requires_grad_())
            represented = dequantize_activation(x).detach().float()
            native = represented.bfloat16().requires_grad_()
            dy = torch.randn_like(data)
            def native_step():
                clear((native,))
                y = getattr(F, kind)(native, **native_options)
                y.backward(dy)
                return y
            native_saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda t: native_saved.append(t) or t, lambda t: t):
                getattr(F, kind)(native, **native_options)
            native_bytes = sum(t.numel() * t.element_size() for t in native_saved)
            native_saved.clear()
            report["results"].append({"activation": variant, "implementation": "native_bf16", "input": input_type,
                                      "saved_state_bytes": native_bytes,
                                      "timing": measurements(native_step, args, torch, device)})
            for output_type in args.outputs:
                for compressed in args.compression:
                    plan = activation_plan(fp8={"enabled": True, "backend": backend, "block_shapes": {"activation": args.fp8_block}},
                                           activation_compression={"enabled": compressed == "on", "bits": args.bits, "group_size": args.group_size})
                    info = {}
                    dtype = torch.float8_e4m3fn if output_type == "fp8" else torch.bfloat16
                    rows, cols = data.numel() // args.shape[-1], args.shape[-1]
                    scale_shape = (1, 1) if args.fp8_block is None else ((rows + args.fp8_block[0] - 1) // args.fp8_block[0], (cols + args.fp8_block[1] - 1) // args.fp8_block[1])
                    scale = torch.ones(scale_shape, device=device) if output_type == "fp8" and args.output_scaling == "provided" else None
                    def step():
                        clear((x,))
                        with deepopt.autocast(device.type, enabled=False, check_finite=False):
                            y = getattr(doF, kind)(x, plan=plan, output_dtype=dtype, output_scale=scale,
                                                  _execution_info=info, **native_options)
                        y.backward(dy)
                        return y
                    timing = measurements(step, args, torch, device)
                    sample = step()
                    y_ref, derivative = evaluate(represented, kind, approximate)
                    dx_ref = torch.where(derivative, dy.float(), 0) if kind == "relu6" else dy.float() * derivative
                    errors = {"forward_max_abs_vs_represented_fp32": (dequantize_activation(sample).float() - y_ref).abs().max().item(),
                              "dx_max_abs_vs_preprocessed_fp32": (dequantize_activation(x.grad).float() - dx_ref).abs().max().item()}
                    report["results"].append({"activation": variant, "implementation": "deepopt", "input": input_type,
                                              "output": output_type, "compression": compressed,
                                              "timing": timing,
                                              "execution": dict(info), "errors": errors})
                    del sample, y_ref, derivative, dx_ref
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
