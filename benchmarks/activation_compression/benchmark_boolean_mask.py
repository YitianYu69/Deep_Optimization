"""Benchmark installed DeepOpt boolean mask packing and exact reconstruction."""

import argparse
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
        raise argparse.ArgumentTypeError("use positive dimensions, for example 1024x1024")
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shape", type=shape, default=(1024, 1024))
    result.add_argument("--layouts", nargs="+", choices=("contiguous", "transpose", "slice", "expand"),
                        default=["contiguous", "transpose", "slice", "expand"])
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=100)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=14)
    result.add_argument("--output", type=Path)
    return result


def make_mask(source, layout, torch):
    if layout == "transpose":
        return source.transpose(-1, -2).contiguous().transpose(-1, -2)
    if layout == "slice":
        storage = torch.empty((*source.shape[:-1], source.shape[-1] * 2), dtype=torch.bool, device=source.device)
        mask = storage[..., ::2]
        mask.copy_(source)
        return mask
    if layout == "expand":
        return source[:1].expand(source.shape)
    return source


def measure(function, args, torch, device):
    for _ in range(args.warmup):
        function()
    wall, intervals = [], []
    for _ in range(args.repeats):
        if device.type == "cuda":
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
        start = perf_counter()
        if device.type == "cuda":
            first.record()
        for _ in range(args.iterations):
            function()
        if device.type == "cuda":
            last.record()
            last.synchronize()
            intervals.append(first.elapsed_time(last) / args.iterations)
        wall.append((perf_counter() - start) * 1000 / args.iterations)
    return {"wall_ms": wall, "wall_ms_median": median(wall),
            "cuda_stream_ms": intervals, "cuda_stream_ms_median": median(intervals) if intervals else None}


def main():
    arguments = parser()
    args = arguments.parse_args()
    if min(args.iterations, args.repeats) <= 0 or args.warmup < 0:
        arguments.error("iterations/repeats must be positive; warmup may be zero")
    if "transpose" in args.layouts and len(args.shape) < 2:
        arguments.error("transpose layout requires at least two dimensions")

    import torch
    import deepopt
    from deepopt.opt.quantization.boolean import pack_boolean_mask, unpack_boolean_mask

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        arguments.error("--device must select cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise SystemExit("An NVIDIA CUDA device is required for GPU measurements")
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    source = torch.randint(2, args.shape, dtype=torch.bool, device=device)
    report = {
        "deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
        "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "notes": ["Eager timing includes output allocation and host submission; warmup is excluded.",
                  "CUDA stream intervals include host submission gaps, not just kernel execution.",
                  "clone_bool measures an ordinary contiguous bool copy, without compression.",
                  "Packed bytes exclude Python shape metadata and allocator reservations.",
                  "Expanded masks report logical bytes; their original shared storage can be smaller than packed storage.",
                  "CPU measurements use the reference implementation, not Triton."],
        "results": [],
    }
    for layout in args.layouts:
        mask = make_mask(source, layout, torch)
        packed = pack_boolean_mask(mask)
        torch.testing.assert_close(unpack_boolean_mask(packed), mask, rtol=0, atol=0)
        logical_bytes = mask.numel() * mask.element_size()
        packed_bytes = packed.data.numel() * packed.data.element_size()
        operations = {
            "pack": lambda: pack_boolean_mask(mask),
            "unpack": lambda: unpack_boolean_mask(packed),
            "roundtrip": lambda: unpack_boolean_mask(pack_boolean_mask(mask)),
            "clone_bool": lambda: mask.clone(memory_format=torch.contiguous_format),
        }
        row = {"layout": layout, "shape": list(mask.shape), "stride": list(mask.stride()),
               "exact_roundtrip": True, "logical_bool_bytes": logical_bytes, "packed_bytes": packed_bytes,
               "logical_compression_ratio": logical_bytes / packed_bytes,
               "timings": {name: measure(function, args, torch, device) for name, function in operations.items()}}
        report["results"].append(row)
        print(f"{layout}: {logical_bytes:,} -> {packed_bytes:,} bytes; "
              f"pack={row['timings']['pack']['wall_ms_median']:.4f} ms; "
              f"unpack={row['timings']['unpack']['wall_ms_median']:.4f} ms", flush=True)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
