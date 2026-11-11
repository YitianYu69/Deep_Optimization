"""Measure spatial convolutions through the installed DeepOpt library.

Copy this program outside the checkout if desired. Eager measured steps include
gradient clearing, layouts, FP8 casting, activation packing, reconstruction and
forward/dX/dW. No optimizer, compilation or CUDA Graph is included.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def positive_shape(text):
    try:
        values = tuple(int(value) for value in text.lower().split("x"))
    except ValueError:
        values = ()
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("use positive dimensions separated by x")
    return values


def padding_value(text):
    if text in ("same", "valid"):
        return text
    try:
        values = tuple(int(value) for value in text.lower().split("x"))
    except ValueError:
        values = ()
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("use same, valid, or nonnegative dimensions separated by x")
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--operation", choices=("conv", "conv_transpose"), default="conv")
    result.add_argument("--shape", type=positive_shape, default=(8, 64, 56, 56), help="NxCxspatial, 1 to 3 spatial dimensions")
    result.add_argument("--out-channels", type=int, default=128)
    result.add_argument("--kernel", type=positive_shape, default=(3,))
    result.add_argument("--stride", type=positive_shape, default=(1,))
    result.add_argument("--padding", type=padding_value, default=(1,))
    result.add_argument("--dilation", type=positive_shape, default=(1,))
    result.add_argument("--output-padding", type=padding_value, default=(0,))
    result.add_argument("--groups", type=int, default=1)
    result.add_argument("--layout", choices=("contiguous", "channels_last"), default="contiguous")
    result.add_argument("--paths", choices=("bf16", "fp8"), nargs="+", default=["bf16", "fp8"])
    result.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    result.add_argument("--group-size", type=int, default=512)
    result.add_argument("--fp8-block", type=positive_shape, help="source FP8 activation scaling rectangle, e.g. 1x32")
    result.add_argument("--fp8-weight-block", type=positive_shape, help="FP8 weight scaling rectangle")
    result.add_argument("--fp8-grad-block", type=positive_shape, help="FP8 output-gradient scaling rectangle")
    result.add_argument("--scaling-recipe", choices=("current", "delayed"), default="current")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=30)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--cuda-graph", action="store_true", help="capture model forward/backward; replay is timed")
    result.add_argument("--optimizer-step", action="store_true", help="include SGD update outside the model capture")
    result.add_argument("--profile-dir", type=Path, help="write one Chrome trace per case and report CUDA kernel phases")
    result.add_argument("--output", type=Path)
    return result


def dimensional(value, rank, option):
    if isinstance(value, str):
        return value
    if len(value) == 1:
        return value * rank
    if len(value) != rank:
        raise ValueError(f"{option} must have 1 or {rank} dimensions")
    return value


class Case:
    def __init__(self, args, path, compressed):
        import deepopt
        from deepopt.opt.nn import functional as deepopt_f
        from deepopt.opt.policy import conv_plan, prepare_fp8_model
        import torch

        self.torch, self.path, self.compressed = torch, path, compressed
        self.rank = len(args.shape) - 2
        torch.manual_seed(args.seed)
        extra = ({"output_padding": dimensional(args.output_padding, self.rank, "--output-padding")}
                 if args.operation == "conv_transpose" else {})
        self.model = getattr(torch.nn, f"{'ConvTranspose' if args.operation == 'conv_transpose' else 'Conv'}{self.rank}d")(
            args.shape[1], args.out_channels,
            dimensional(args.kernel, self.rank, "--kernel"),
            stride=dimensional(args.stride, self.rank, "--stride"),
            padding=dimensional(args.padding, self.rank, "--padding"),
            dilation=dimensional(args.dilation, self.rank, "--dilation"),
            groups=args.groups, device=args.device, dtype=torch.float32, **extra,
        )
        self.x = torch.randn(args.shape, device=args.device, dtype=torch.bfloat16)
        if args.layout == "channels_last":
            memory_format = torch.channels_last if self.rank == 2 else torch.channels_last_3d
            self.x = self.x.to(memory_format=memory_format)
            self.model.to(memory_format=memory_format)
        self.x.requires_grad_()
        compression = {"enabled": compressed, "group_size": args.group_size, "bits": args.bits}
        self.context = lambda: torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        if path == "fp8":
            settings = deepopt.FP8Config(
                enabled=True, backend="triton", trace_mode="modules", intercept_linear=False,
                scaling_recipe=args.scaling_recipe,
                block_shapes={"activation": args.fp8_block, "weight": args.fp8_weight_block, "grad": args.fp8_grad_block},
            )
            self.model, report = prepare_fp8_model(self.model, settings, activation_compression=compression)
            if not report.converted_modules:
                raise RuntimeError("DeepOpt did not convert the spatial convolution")
            self.call = self.model
            self.context = lambda: deepopt.autocast(
                "cuda", dtype=torch.bfloat16, fp8=settings, activation_compression=compression,
                cache_enabled=False, check_finite=False,
            )
        else:
            plan = conv_plan(False, activation_compression=compression)
            native = getattr(deepopt_f, f"{args.operation}{self.rank}d")
            self.call = lambda value: native(
                value, self.model.weight, self.model.bias, stride=self.model.stride,
                padding=self.model.padding, dilation=self.model.dilation, groups=self.model.groups, plan=plan, **extra,
            )
        self.model.train()
        with torch.no_grad(), self.context():
            output = self.call(self.x)
        self.grad = torch.randn_like(output) / output.numel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3) if args.optimizer_step else None
        self.captured = None
        if args.cuda_graph:
            class Call(torch.nn.Module):
                def __init__(self, owner, call):
                    super().__init__()
                    self.owner = owner
                    object.__setattr__(self, "call", call)

                def forward(self, value):
                    return self.call(value)

            target = Call(self.model, self.call)
            with self.context():
                self.captured = deepopt.CUDAGraphCore(self.x.device).capture_model(
                    target, (self.x,), warmup=max(3, args.warmup), autograd=True)
            self.call = target

    def clear(self):
        self.model.zero_grad(set_to_none=True)
        self.x.grad = None

    def step(self):
        self.clear()
        with self.context():
            output = self.call(self.x)
            output.backward(self.grad)
        if self.optimizer is not None:
            self.optimizer.step()
        return output

    def close(self):
        if self.captured is not None:
            self.captured.close()


def measure(case, args):
    torch = case.torch
    for _ in range(args.warmup):
        case.step()
    torch.cuda.synchronize(args.device)
    wall, events = [], []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(args.device)
        began = perf_counter()
        start.record()
        for _ in range(args.iterations):
            case.step()
        end.record()
        end.synchronize()
        wall.append((perf_counter() - began) * 1000 / args.iterations)
        events.append(start.elapsed_time(end) / args.iterations)

    case.clear()
    torch.cuda.synchronize(args.device)
    baseline = torch.cuda.memory_allocated(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    saved = []

    def record(value):
        saved.append({"dtype": str(value.dtype), "shape": list(value.shape),
                      "logical_bytes": value.numel() * value.element_size()})
        return value

    with torch.autograd.graph.saved_tensors_hooks(record, lambda value: value):
        output = case.step()
    torch.cuda.synchronize(args.device)
    peak = torch.cuda.max_memory_allocated(args.device) - baseline
    snapshots = {"output": output.detach().float().cpu(), "dx": case.x.grad.float().cpu(),
                 "dw": case.model.weight.grad.float().cpu(), "db": case.model.bias.grad.float().cpu()}
    profile_summary = None
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        trace = args.profile_dir / f"{args.operation}-{case.path}-compression-{int(case.compressed)}.json"
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            case.step()
            torch.cuda.synchronize(args.device)
        profile.export_chrome_trace(str(trace))
        kernels = [event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA
                   and not event.name.startswith(("Memcpy", "Memset", "[memory]"))]
        phases = {}
        for event in kernels:
            name = event.name
            phase = "preparation_or_other"
            if args.operation == "conv_transpose" and case.path == "fp8":
                if "_input_gradient_kernel" in name:
                    phase = "forward_spatial"
                elif "_forward_kernel" in name:
                    phase = "dx_spatial"
                elif "weight_gradient" in name:
                    phase = "dw_spatial"
            row = phases.setdefault(phase, {"launches": 0, "cuda_us": 0.0})
            row["launches"] += 1
            row["cuda_us"] += event.device_time_total
        profile_summary = {"trace": str(trace), "kernel_launches": len(kernels), "phases": phases,
                           "note": "Spatial phase buckets identify Triton kernel names; GEMM, native kernels, bias, casts and packing remain in preparation_or_other. Kernel durations exclude CPU submission gaps."}
    return {
        "path": case.path, "activation_compression": case.compressed,
        "cuda_graph": case.captured is not None, "optimizer_step": case.optimizer is not None,
        "profile": profile_summary,
        "execution": getattr(case.model, "execution_info", None),
        "wall_ms_median": median(wall), "wall_ms_repeats": wall,
        "cuda_stream_ms_median": median(events), "cuda_stream_ms_repeats": events,
        "saved_tensor_logical_bytes": sum(value["logical_bytes"] for value in saved),
        "saved_tensors": saved, "peak_extra_allocated_bytes": peak,
        "output_shape": list(output.shape), "parameter_dtype": str(case.model.weight.dtype),
    }, snapshots


def differences(actual, expected):
    results = {}
    for name in expected:
        error = (actual[name] - expected[name]).double()
        norm = expected[name].double().norm().item()
        results[name] = {"relative_l2": error.norm().item() / max(norm, 1e-30),
                         "max_abs": error.abs().max().item()}
    return results


def main():
    arguments = parser()
    args = arguments.parse_args()
    rank = len(args.shape) - 2
    if rank not in (1, 2, 3):
        arguments.error("--shape needs batch, channels and 1, 2 or 3 spatial dimensions")
    if min(args.out_channels, args.groups, args.group_size, args.iterations, args.repeats) <= 0 or args.warmup < 0:
        arguments.error("counts must be positive; --warmup may be zero")
    if args.shape[1] % args.groups or args.out_channels % args.groups:
        arguments.error("input and output channels must be divisible by --groups")
    if args.layout == "channels_last" and rank == 1:
        arguments.error("channels_last is available for Conv2d and Conv3d")
    blocks = (args.fp8_block, args.fp8_weight_block, args.fp8_grad_block)
    for name, block in zip(("--fp8-block", "--fp8-weight-block", "--fp8-grad-block"), blocks):
        if block is not None and (len(block) != 2 or any(value & (value - 1) for value in block) or block == (1, 1)):
            arguments.error(f"{name} requires two positive powers of two and cannot be 1x1")
    if any(block is not None for block in blocks) and args.scaling_recipe == "delayed":
        arguments.error("delayed FP8 scaling requires tensorwise scaling")
    if args.cuda_graph and args.scaling_recipe == "delayed" and "fp8" in args.paths:
        arguments.error("CUDAGraphCore requires current scaling; delayed scaling needs eager preprocessing")
    try:
        for name in ("kernel", "stride", "padding", "dilation", "output_padding"):
            dimensional(getattr(args, name), rank, "--" + name)
        if args.operation == "conv_transpose" and (isinstance(args.padding, str) or isinstance(args.output_padding, str)):
            arguments.error("transposed convolution requires numeric padding and output padding")
        if args.operation == "conv" and args.output_padding != (0,):
            arguments.error("--output-padding applies only to conv_transpose")
    except ValueError as error:
        arguments.error(str(error))

    import deepopt
    import torch

    device = torch.device(args.device)
    if device.type != "cuda":
        arguments.error("--device must name a CUDA device")
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA; no CPU performance results are produced.")
    torch.cuda.set_device(device)
    if "fp8" in args.paths and (torch.version.hip is not None or torch.cuda.get_device_capability(device)[0] < 9):
        raise SystemExit("Triton FP8 requires an NVIDIA GPU with compute capability >= 9.")
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "notes": [
            "Step includes gradient clearing, quantization, packing, decode and layout handling; optional SGD is outside model capture.",
            "CUDA Graph setup and warmup are excluded; replay includes captured casting and compression kernels.",
            "Optimizer-enabled numerical errors include accumulated training divergence; use optimizer off to isolate one-step operand error.",
            "FP32 master parameters and BF16 autocast; quantized gradients change FP8 training relative to native BF16.",
            "Compression changes only saved convolution inputs used by dW, never forward/dX/db operands.",
            "CUDA event intervals include stream submission gaps and are not sums of kernel durations.",
            "Saved tensor bytes sum logical tensors, without deduplicating aliased storage; allocator peaks include gradients/workspaces.",
            "General spatial convolutions use implicit patch addressing; source packing does not materialize unfolded activations.",
            "Tensorwise and blockwise convolution use FP8 dot operands; blockwise partials are scaled and accumulated in FP32.",
            "Fine source-row groups can require short padded weight-gradient partials; FP8 arithmetic does not guarantee a speedup.",
            "Channels-last and pointwise conversion costs are included. FP8 speedups require measurement on each workload.",
        ], "results": [],
    }
    native_reference = None
    for path in args.paths:
        off_snapshots, off_result = None, None
        for compressed in (False, True):
            case = Case(args, path, compressed)
            try:
                row, snapshots = measure(case, args)
            finally:
                case.close()
            if path == "bf16" and not compressed:
                native_reference = snapshots
            if native_reference is not None:
                row["errors_vs_native_bf16"] = differences(snapshots, native_reference)
            if compressed:
                row["errors_vs_compression_off"] = differences(snapshots, off_snapshots)
                row["wall_speedup_vs_compression_off"] = off_result["wall_ms_median"] / row["wall_ms_median"]
            else:
                off_snapshots, off_result = snapshots, row
            results["results"].append(row)
            print(f"{path:4s} compression={'on ' if compressed else 'off'} "
                  f"wall={row['wall_ms_median']:.4f} ms stream={row['cuda_stream_ms_median']:.4f} ms "
                  f"saved={row['saved_tensor_logical_bytes']:,} bytes", flush=True)
            del case
            torch.cuda.empty_cache()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
