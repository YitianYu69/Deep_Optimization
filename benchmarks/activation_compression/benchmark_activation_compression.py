"""Benchmark saved-activation compression using the INSTALLED DeepOpt library.

Copy this file anywhere, then run with the GPU environment's Python:
    python benchmark_activation_compression.py --shape 4096x768x3072 --output compression.json

Each path compares compression off/on. Timed steps include quantization,
packing, unpacking, all three GEMMs and gradient clearing, with no optimizer,
torch.compile or CUDA Graph. Both paths use FP32 weights and BF16 AMP.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def parse_shape(text):
    try:
        result = tuple(int(value) for value in text.lower().split("x"))
    except ValueError:
        result = ()
    if len(result) != 3 or any(value <= 0 or value % 16 for value in result):
        raise argparse.ArgumentTypeError("use MxKxN with positive multiples of 16, e.g. 4096x768x3072")
    return result


def parse_block(text):
    try:
        block = tuple(int(value) for value in text.lower().split("x"))
    except ValueError:
        block = ()
    if len(block) != 2 or any(value <= 0 or value & (value - 1) for value in block):
        raise argparse.ArgumentTypeError("use two positive powers of two, e.g. 1x32 or 32x32")
    return block


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shape", type=parse_shape, default=(4096, 768, 3072), metavar="MxKxN")
    result.add_argument("--paths", choices=("fp8", "bf16"), nargs="+", default=["fp8", "bf16"])
    result.add_argument("--group-size", type=int, default=512)
    result.add_argument("--fp8-block", type=parse_block, help="activation scale rectangle; overrides compression group size")
    result.add_argument("--scaling-recipe", choices=("current", "delayed"), default="current")
    result.add_argument("--backward-backend", choices=("inherit", "native_fast", "native_precise"), default="inherit")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output", type=Path)
    return result


class Case:
    def __init__(self, args, path, compressed):
        import deepopt
        import torch

        self.torch, self.path, self.compressed = torch, path, compressed
        torch.manual_seed(args.seed)
        m, k, n = args.shape
        self.model = torch.nn.Linear(k, n, device=args.device, dtype=torch.float32)
        self.x = torch.randn(m, k, device=args.device, dtype=torch.bfloat16, requires_grad=True)
        self.grad = torch.randn(m, n, device=args.device, dtype=torch.bfloat16) / m
        self.group_size, self.block = args.group_size, args.fp8_block
        self.context = lambda: torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        if path == "fp8":
            from deepopt.opt.policy import prepare_fp8_model

            settings = deepopt.FP8Config(
                enabled=True, backend="triton", trace_mode="modules", intercept_linear=False,
                scaling_recipe=args.scaling_recipe, backward_backend=args.backward_backend,
                block_shapes={"activation": args.fp8_block, "weight": None, "grad": None},
            )
            self.model, report = prepare_fp8_model(self.model, settings)
            if not report.converted_modules:
                raise RuntimeError("DeepOpt did not convert Linear to FP8")
            self.context = lambda: deepopt.autocast(
                "cuda", dtype=torch.bfloat16, fp8=settings, cache_enabled=False, check_finite=False,
                activation_compression={"enabled": compressed, "group_size": args.group_size},
            )
        self.call = self.model
        if path == "bf16":
            from deepopt.opt.nn.functional import do_linear
            from deepopt.opt.policy import linear_plan

            plan = linear_plan(False, activation_compression={"enabled": compressed, "group_size": self.group_size})
            self.call = partial(do_linear, weight=self.model.weight, bias=self.model.bias, plan=plan)
        self.model.train()

    def clear(self):
        self.model.zero_grad(set_to_none=True)
        self.x.grad = None

    def step(self):
        self.clear()
        with self.context():
            output = self.call(self.x)
            output.backward(self.grad)
        return output

    def activation_payload_bytes(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation

        if self.compressed:
            block = self.block if self.path == "fp8" else None
            value = self.x.detach()
            if block is not None and block[0] == 1 and block[1] > 1:
                value = value.T
            packed = quantize_pack_activation(value, group_size=self.group_size, block_shape=block)
            return sum(tensor.numel() * tensor.element_size() for tensor in
                       (packed.data, packed.minimum, packed.step))
        return self.x.numel() * (1 if self.path == "fp8" else self.x.element_size())


def measure(case, args):
    torch = case.torch
    for _ in range(args.warmup):
        case.step()
    torch.cuda.synchronize(args.device)
    wall, gpu = [], []
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
        gpu.append(start.elapsed_time(end) / args.iterations)

    # Hooks and allocator accounting run separately, outside throughput timing.
    case.clear()
    torch.cuda.synchronize(args.device)
    baseline = torch.cuda.memory_allocated(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    saved = []

    def record(tensor):
        saved.append({"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                      "bytes": tensor.numel() * tensor.element_size()})
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        output = case.step()
    torch.cuda.synchronize(args.device)
    peak = torch.cuda.max_memory_allocated(args.device) - baseline
    snapshots = {"output": output.detach().float().cpu(), "dx": case.x.grad.float().cpu(),
                 "dw": case.model.weight.grad.float().cpu(), "db": case.model.bias.grad.float().cpu()}
    return {
        "path": case.path, "compression": case.compressed,
        "wall_ms_median": median(wall), "wall_ms_repeats": wall,
        "cuda_stream_ms_median": median(gpu), "cuda_stream_ms_repeats": gpu,
        "saved_tensor_payload_bytes": sum(value["bytes"] for value in saved),
        "saved_tensors": saved, "peak_extra_allocated_bytes": peak,
        "activation_codec_payload_bytes": case.activation_payload_bytes(),
        "parameter_dtype": str(case.model.weight.dtype),
    }, snapshots


def differences(actual, expected):
    result = {}
    for name in expected:
        error = (actual[name] - expected[name]).double()
        norm = expected[name].double().norm().item()
        result[name] = {"relative_l2": error.norm().item() / max(norm, 1e-30),
                        "max_abs": error.abs().max().item()}
    return result


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.group_size <= 0:
        arg_parser.error("--group-size must be a positive integer")
    if args.warmup < 0 or min(args.iterations, args.repeats) < 1:
        arg_parser.error("warmup must be nonnegative; iterations and repeats must be positive")
    if args.fp8_block is not None and args.scaling_recipe == "delayed":
        arg_parser.error("delayed FP8 scaling currently supports tensorwise scaling only")

    import deepopt
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU; no CPU performance results are produced.")
    device = torch.device(args.device)
    if device.type != "cuda":
        arg_parser.error("--device must name a CUDA device")
    torch.cuda.set_device(device)
    if "fp8" in args.paths and (torch.version.hip is not None or torch.cuda.get_device_capability(device)[0] < 9):
        raise SystemExit("The Triton FP8 path requires an NVIDIA GPU with compute capability >= 9.")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "notes": [
            "Eager forward+backward includes gradient clearing and FP8 preparation or 2-bit packing/unpacking; no optimizer.",
            "Both event intervals and wall measurements include host submission gaps; events are not sums of kernel durations.",
            "Both paths use FP32 master weights and BF16 AMP with BF16 inputs/outputs; BF16 uses native matmul.",
            "Activation codec payload includes int32 words and FP32 minimum/step; excludes FP8 inverse-scale metadata.",
            "Saved payload sums logical tensor bytes; peak extra allocation also includes gradients and temporary workspaces.",
            "The 2-bit codec changes dW. Forward/dX/db should match compression off; errors are not a convergence test.",
            "FP8 block scaling uses exactly the scale rectangle for compression, overriding --group-size.",
        ],
        "results": [],
    }
    for path in args.paths:
        baseline = None
        baseline_result = None
        for compressed in (False, True):
            case = Case(args, path, compressed)
            row, snapshots = measure(case, args)
            if not compressed:
                baseline, baseline_result = snapshots, row
            else:
                row["errors_vs_compression_off"] = differences(snapshots, baseline)
                row["wall_speedup_vs_off"] = baseline_result["wall_ms_median"] / row["wall_ms_median"]
            result["results"].append(row)
            print(f"{path:4s} compression={'on ' if compressed else 'off'} "
                  f"wall={row['wall_ms_median']:.4f} ms "
                  f"stream={row['cuda_stream_ms_median']:.4f} ms "
                  f"saved={row['saved_tensor_payload_bytes']:,} bytes", flush=True)
            del case
            torch.cuda.empty_cache()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
