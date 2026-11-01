"""Reference-checked CPU/CUDA/DDP mode and transfer benchmark.

python benchmarks/benchmark_execution.py --device cpu --batches 24
torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_execution.py --device cuda --accumulation 4 --pinned --non-blocking --profile
"""

import argparse
from contextlib import nullcontext
import json
import os
import platform
from statistics import median
from time import perf_counter

import torch
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

from deepopt import Trainer, initialize_distributed, shutdown_distributed


def benchmark(args):
    context = initialize_distributed(args.device)
    device = context.device
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(1700 + context.global_rank)
    batches = [(torch.randn(args.batch_size, args.features, generator=generator),
                torch.randn(args.batch_size, 1, generator=generator)) for _ in range(args.batches)]
    if args.pinned:
        if device.type != "cuda":
            raise ValueError("--pinned requires CUDA")
        batches = [(x.pin_memory(), y.pin_memory()) for x, y in batches]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def maximum(value):
        if context.is_distributed:
            tensor = torch.tensor(value, device=device, dtype=torch.float64)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return tensor.item()
        return value

    def run(mode, profile=False):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(args.features, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 1)).to(device)
        if args.compile:
            model = torch.compile(model, backend="inductor" if device.type == "cuda" else "eager")
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        if mode == "torch":
            wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None) if context.is_distributed else model
            trainer = None
        else:
            trainer = Trainer(model, optimizer, device=device, precision=args.precision,
                              loss_fn=nn.functional.mse_loss, accumulation_steps=args.accumulation,
                              execution_mode=None if mode == "legacy" else mode,
                              non_blocking=args.non_blocking)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities) if profile else nullcontext()
        synchronize()
        started = perf_counter()
        with profiler as trace:
            if trainer is not None:
                loss = trainer.train_epoch(batches)["loss"]
            else:
                total = torch.zeros((), dtype=torch.float64, device=device)
                for start in range(0, len(batches), args.accumulation):
                    window = batches[start:start + args.accumulation]
                    for index, (x, y) in enumerate(window):
                        sync = wrapped.no_sync() if context.is_distributed and index < len(window) - 1 else nullcontext()
                        with sync:
                            x, y = x.to(device, non_blocking=args.non_blocking), y.to(device, non_blocking=args.non_blocking)
                            precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
                            with precision:
                                value = nn.functional.mse_loss(wrapped(x), y)
                            total += value.detach().double()
                            value.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(len(window))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if context.is_distributed:
                    dist.all_reduce(total)
                loss = (total / (len(batches) * context.world_size)).item()
        synchronize()
        elapsed = maximum(perf_counter() - started)
        counts = ({event.key: event.count for event in trace.key_averages()
                   if event.key in {"aten::item", "aten::_local_scalar_dense", "c10d::allreduce_",
                                    "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync"}}
                  if profile else {})
        return elapsed, [parameter.detach().cpu().clone() for parameter in model.parameters()], loss, counts

    rows, expected, reference_loss = [], None, None
    for mode in ["torch", *args.modes]:
        for _ in range(args.warmups):
            run(mode)
        times = []
        for _ in range(args.repeats):
            elapsed, parameters, loss, _ = run(mode)
            times.append(elapsed)
            if expected is None:
                expected, reference_loss = parameters, loss
            for actual, reference in zip(parameters, expected):
                torch.testing.assert_close(actual, reference, rtol=3e-3 if args.precision == "bf16" else 1e-5, atol=1e-6)
            torch.testing.assert_close(torch.tensor(loss), torch.tensor(reference_loss), rtol=3e-3, atol=1e-6)
        profiling = run(mode, profile=True)[3] if args.profile else {}
        elapsed = median(times)
        rows.append({"mode": mode, "median_seconds": elapsed,
                     "samples_per_second": args.batches * args.batch_size * context.world_size / elapsed,
                     "relative_to_torch": elapsed / rows[0]["median_seconds"] if rows else 1.,
                     "profiler_counts": profiling})

    transfers = []
    if device.type == "cuda":
        for pinned in (False, True):
            source = torch.randn(args.batch_size, args.features)
            if pinned:
                source = source.pin_memory()
            for non_blocking in (False, True):
                times = []
                for _ in range(args.repeats):
                    synchronize()
                    started = perf_counter()
                    for _ in range(args.batches):
                        copied = source.to(device, non_blocking=non_blocking)
                    synchronize()
                    times.append(perf_counter() - started)
                    torch.testing.assert_close(copied.cpu(), source)
                transfers.append({"pinned": pinned, "non_blocking": non_blocking, "seconds": median(times)})
    if context.is_main_process:
        print(json.dumps({"environment": {**context.as_dict(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                                           "python": platform.python_version(), "platform": platform.platform(),
                                           "logging_level": os.environ.get("DEEPOPT_LOGGING_LEVEL", "INFO")},
                          "settings": vars(args), "correctness_checked": True, "training": rows,
                          "transfers": transfers, "note": "Transfer timings do not imply compute/copy stream overlap."}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--modes", nargs="+", choices=("legacy", "max_speed", "balanced", "analyze"),
                        default=["legacy", "max_speed", "balanced", "analyze"])
    for name, default in (("batch-size", 32), ("features", 64), ("hidden", 128), ("batches", 24),
                          ("accumulation", 4), ("repeats", 3), ("threads", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if any(getattr(args, name) < 1 for name in ("batch_size", "features", "hidden", "batches", "accumulation", "repeats", "threads")) or args.warmups < 0:
        parser.error("Workload sizes/repeats must be positive; warmups must be nonnegative")
    try:
        benchmark(args)
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
