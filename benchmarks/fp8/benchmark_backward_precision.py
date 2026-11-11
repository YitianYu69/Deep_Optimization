"""Compare installed DeepOpt native and planned FP8 activation-gradient transport."""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def block(text):
    try:
        values = tuple(int(v) for v in text.lower().split("x"))
    except ValueError:
        values = ()
    if len(values) != 2 or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("use two positive dimensions, for example 1x32")
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pipeline", choices=("linear", "conv1d", "conv2d", "conv3d", "attention"), default="linear")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--sequence-length", type=int, default=64)
    result.add_argument("--spatial", type=int, default=8)
    result.add_argument("--grad-block", type=block)
    result.add_argument("--scaling-recipe", choices=("current", "delayed"), default="current")
    result.add_argument("--compression", action="store_true")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=30)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--output", type=Path)
    return result


def build(args, nn):
    if args.pipeline == "linear":
        return nn.Sequential(nn.Linear(args.width, args.width), nn.ReLU(), nn.Linear(args.width, args.width)), (
            args.batch_size, args.sequence_length, args.width)
    if args.pipeline == "attention":
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(args.width, 2, batch_first=True)

            def forward(self, x):
                return self.attention(x, x, x, need_weights=False)[0]
        return Attention(), (args.batch_size, args.sequence_length, args.width)
    rank = int(args.pipeline[4])
    conv = getattr(nn, f"Conv{rank}d")
    return nn.Sequential(conv(args.width, args.width, 3, padding=1), nn.GroupNorm(2, args.width),
                         nn.ReLU(), conv(args.width, args.width, 3, padding=1)), (
                             args.batch_size, args.width, *((args.spatial,) * rank))


def main():
    arguments = parser()
    args = arguments.parse_args()
    if min(args.batch_size, args.width, args.sequence_length, args.spatial, args.iterations, args.repeats) <= 0 or args.warmup < 0:
        arguments.error("dimensions/counts must be positive; warmup may be zero")
    if args.width % 2:
        arguments.error("--width must be divisible by two")
    if args.grad_block is not None and args.scaling_recipe == "delayed":
        arguments.error("delayed scaling requires tensorwise gradients")

    import torch
    import deepopt
    from deepopt.opt.policy import prepare_fp8_model
    from deepopt.opt.precision import precision_report

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        arguments.error("--device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise SystemExit("An NVIDIA CUDA device is required")
        torch.cuda.set_device(device)
    torch.manual_seed(14)
    original, shape = build(args, torch.nn)
    original.to(device)
    source = torch.randn(shape, device=device, dtype=torch.bfloat16)
    dy = torch.randn(shape, device=device, dtype=torch.bfloat16)
    report = {"deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
              "torch": torch.__version__, "device": str(device),
              "configuration": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
              "notes": ["Full eager forward/backward includes gradient preparation, transport, casting and gradient clearing.",
                        "No optimizer updates, input transfer, initialization or warmup are timed.",
                        "CUDA intervals include host submission gaps. CPU uses reference kernels.",
                        "Current global scales can require FP32 workspace and extra reduction/cast launches.",
                        "Native attention cores remain explicit backward boundaries.",
                        "Saved bytes count logical tensors, excluding allocator reservations and deduplication."],
              "results": []}
    baseline = None
    for mode in ("native", "auto"):
        cfg = deepopt.FP8Config(enabled=True, backend="triton" if device.type == "cuda" else "reference",
                               backward_precision=mode, grad_dtype="e5m2", scaling_recipe=args.scaling_recipe,
                               block_shapes={"activation": None, "weight": None, "grad": args.grad_block})
        model, _ = prepare_fp8_model(deepcopy(original), cfg, activation_compression=args.compression)
        x = source.detach().clone().requires_grad_()

        def step():
            model.zero_grad(set_to_none=True)
            x.grad = None
            with deepopt.autocast(device.type, dtype=torch.bfloat16, fp8=cfg,
                                  activation_compression=args.compression, check_finite=False):
                output = model(x)
                output.backward(dy)
            return output

        for _ in range(args.warmup):
            step()
        wall, events = [], []
        for _ in range(args.repeats):
            if device.type == "cuda":
                first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
            start = perf_counter()
            if device.type == "cuda":
                first.record()
            for _ in range(args.iterations):
                step()
            if device.type == "cuda":
                last.record()
                last.synchronize()
            wall.append((perf_counter() - start) * 1000 / args.iterations)
            if device.type == "cuda":
                events.append(first.elapsed_time(last) / args.iterations)
        model.zero_grad(set_to_none=True)
        x.grad = None
        saved = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value.numel() * value.element_size()) or value,
                                                    lambda value: value):
            output = step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        snapshot = {"output": output.detach().float().cpu(), "dx": x.grad.float().cpu(),
                    **{name: parameter.grad.float().cpu() for name, parameter in model.named_parameters() if parameter.grad is not None}}
        errors = {}
        if baseline is None:
            baseline = snapshot
        else:
            errors = {name: {"max_abs": (value - baseline[name]).abs().max().item(),
                             "relative_l2": ((value - baseline[name]).double().norm() /
                                             baseline[name].double().norm().clamp_min(1e-30)).item()}
                      for name, value in snapshot.items()}
        row = {"mode": mode, "wall_ms": wall, "wall_ms_median": median(wall), "cuda_stream_ms": events,
               "cuda_stream_ms_median": median(events) if events else None, "saved_tensor_bytes": sum(saved),
               "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - allocated if device.type == "cuda" else None,
               "precision_plan": precision_report(model), "errors_vs_native_gradient_transport": errors}
        report["results"].append(row)
        print(f"{mode}: wall={median(wall):.4f} ms saved={sum(saved):,} bytes", flush=True)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
