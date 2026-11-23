"""Compare eager and compiled DeepOpt with identical training and numerical policy."""

import argparse
from contextlib import nullcontext
import gc
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor

from deepopt import FP8Config, autocast
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.norm = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 16)

    def forward(self, x):
        y = self.relu(self.norm(self.conv(x)) + x)
        return self.head(self.pool(y).flatten(1))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 128)

    def forward(self, x):
        y = self.attention(x, x, x, need_weights=False)[0]
        return self.head(self.norm(y + x))


class ViTBlock(nn.Module):
    def __init__(self, width=768, heads=12):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.fc1, self.fc2 = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(y.transpose(1, 2).reshape(batch, tokens, width))
        y = nn.functional.gelu(self.fc1(self.norm2(x)))
        return x + self.fc2(nn.functional.dropout(y, p=0.0, training=self.training))


def workload(name, *, width=768, heads=12, tokens=197):
    if name == "mlp":
        return nn.Sequential(
            nn.Linear(128, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, 128)
        ), (128,)
    if name == "residual":
        return Residual(), (16, 8, 8)
    if name == "vit-block":
        return ViTBlock(width, heads), (tokens, width)
    if name == "vit-base":
        try:
            import timm
        except ImportError as error:
            raise RuntimeError("The vit-base benchmark requires timm") from error
        return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10), (3, 224, 224)
    return Attention(), (8, 128)


def summarize(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
    }


def phase_total(row):
    return sum(row[key] for key in ("forward_loss_ms", "backward_ms", "optimizer_ms"))


def benchmark(args, backend, kernel_backend=None):
    from deepopt._benchmarking import BenchmarkBackend

    kernels = kernel_backend if kernel_backend is not None else BenchmarkBackend.from_args(args)
    device = torch.device(args.device)
    with kernels.scope(), (torch.cuda.device(device) if device.type == "cuda" else nullcontext()):
        return _benchmark(args, backend, kernels)


def _benchmark(args, backend, kernels):
    from deepopt._benchmarking import timing_plan_from_cost
    kernel_config = kernels.config
    torch.compiler.reset()
    gc.collect()
    torch.manual_seed(73)
    device = torch.device(args.device)
    model, shape = workload(args.model, width=args.width, heads=args.heads, tokens=args.tokens)
    projections_only = args.model in ("vit-block", "vit-base")
    excluded = tuple(
        name for name, module in model.named_modules()
        if name and not any(module.children())
        and (type(module) is not nn.Linear or name == "head")
    ) if projections_only else ()
    settings = FP8Config(
        enabled=args.fp8, backend=kernels.fp8_backend(device),
        autotune=args.autotune, exclude_modules=excluded,
        intercept_linear=not projections_only,
    )
    compression = {"enabled": args.compression, "bits": args.bits, "group_size": args.group_size}
    model = prepare_fp8_model(
        model.to(device), settings, activation_compression=compression, config=kernel_config,
    )[0]
    if backend != "eager":
        compile_model(
            model, CompileConfig(
                enabled=True, backend=backend, dynamic=False,
                options={
                    "fuse_deepopt_activations": args.fuse_deepopt_activations,
                    "bf16_gemm_backend": getattr(args, "bf16_gemm_backend", "native"),
                    "bf16_decode": getattr(args, "bf16_decode", "materialized"),
                } if backend == "deepopt" else {},
            ), config=kernel_config,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    generator = torch.Generator(device=device).manual_seed(101)
    x = torch.randn((args.batch_size, *shape), device=device, generator=generator)

    def scope():
        return autocast(
            device.type,
            enabled=args.precision == "bf16",
            dtype=torch.bfloat16,
            fp8=settings,
            activation_compression=None if projections_only else compression,
            check_finite=False,
            config=kernel_config,
        )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step(timed=False):
        optimizer.zero_grad(set_to_none=True)
        events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if timed and device.type == "cuda"
            else None
        )
        marks = []

        def mark(index):
            if events is not None:
                events[index].record()
            marks.append(time.perf_counter())

        mark(0)
        with scope():
            output = model(x)
            loss = output.float().square().mean()
        mark(1)
        loss.backward()
        mark(2)
        optimizer.step()
        mark(3)
        if timed:
            synchronize()
            phases = (
                [events[index].elapsed_time(events[index + 1]) for index in range(3)]
                if events is not None
                else [(marks[index + 1] - marks[index]) * 1000 for index in range(3)]
            )
            return loss, dict(
                zip(("forward_loss_ms", "backward_ms", "optimizer_ms"), phases)
            )
        return loss, None

    synchronize()
    started = time.perf_counter()
    step()
    synchronize()
    cold_ms = (time.perf_counter() - started) * 1000
    # Use one common plan: independent adaptation would compare model states
    # after different numbers of optimizer updates.
    plan = getattr(args, "_measurement_plan", None)
    if plan is None:
        plan = timing_plan_from_cost(args, cold_ms / 1000, iterations=args.steps, repeats=1)
        args._measurement_plan = plan
    measurement_began = time.perf_counter()
    for _ in range(plan.warmup):
        step()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measurements = []
    for _ in range(plan.iterations):
        started = time.perf_counter()
        loss, values = step(timed=True)
        values["step_ms"] = (time.perf_counter() - started) * 1000
        measurements.append(values)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    sampling = plan.sampling(time.perf_counter() - measurement_began)
    sampling.update(shared_eager_compiled_counts=True, cost_estimate_source="eager_cold_step",
                    cold_first_call_seconds=cold_ms / 1000)

    # Measure retained storage in a separate forward, outside timing. Do not
    # retain tensor references or count the same aliased storage twice.
    saved = {}

    def pack(value):
        if isinstance(value, FakeTensor) or isinstance(
            getattr(value, "_data", None), FakeTensor
        ):
            return value
        storage = value.untyped_storage()
        saved[(str(value.device), storage.data_ptr())] = storage.nbytes()
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        loss, _ = step()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Benchmark training produced a non-finite loss")

    profile = None
    if args.profile or args.trace_dir is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            loss, _ = step()
            synchronize()
        if args.trace_dir is not None:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.trace_dir / f"{backend}.json"))
        kernel_events = [
            event.name
            for event in profiler.events()
            if str(event.device_type).endswith("CUDA")
        ]
        profile = {
            "cuda_kernel_events": len(kernel_events),
            "kernel_names": sorted(set(kernel_events)),
            "cast_and_codec_operations": {
                event.key: event.count
                for event in profiler.key_averages()
                if any(
                    term in event.key.lower()
                    for term in ("cast", "quant", "pack", "aten::to")
                )
            },
            "normalization_operations": {
                event.key: event.count for event in profiler.key_averages()
                if "layer_norm" in event.key
            },
        }
    report = compilation_report(model)
    result = {
        "backend": backend,
        "kernel_backend": kernels.metadata(device),
        "sampling": sampling,
        "cold_step_ms": cold_ms,
        "timings": {
            **{key: summarize([row[key] for row in measurements]) for key in measurements[0]},
            "step_phases_ms": summarize([phase_total(row) for row in measurements]),
        },
        "step_phase_clock": "cuda_stream_events" if device.type == "cuda" else "cpu_perf_counter",
        "peak_cuda_allocated_bytes": peak,
        "peak_cuda_reserved_bytes": peak_reserved,
        "saved_storage_bytes": sum(saved.values()),
        "saved_storage_measurement": "outer saved-tensor hooks; inner operation hooks may retain additional state",
        "final_loss": float(loss.detach()),
        "compile_report": report.as_dict() if report else None,
        "profile": profile,
    }
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    return result, state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("mlp", "residual", "attention", "vit-block", "vit-base"), default="mlp"
    )
    parser.add_argument("--backend", choices=("deepopt", "inductor"), default="deepopt")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:INDEX")
    parser.add_argument("--deepopt-backend", choices=("triton", "tilelang", "mixed"), default="triton")
    parser.add_argument("--deepopt-kernel-policy", type=json.loads, default={}, metavar="JSON")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--bits", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=197)
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fuse-deepopt-activations", action="store_true")
    parser.add_argument("--bf16-gemm-backend", choices=("native", "generated"), default="native")
    parser.add_argument("--bf16-decode", choices=("materialized", "tiled"), default="materialized")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--time-budget", type=float, default=2.0, help="Warm timing target; use a common eager/compiled count plan; 0 disables")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, help="Write the same JSON report printed to stdout")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or args.warmup < 0:
        parser.error(
            "steps and batch-size must be positive; warmup must be nonnegative"
        )
    if args.width < 1 or args.heads < 1 or args.width % args.heads or args.tokens < 1 or args.group_size < 1:
        parser.error("width must be divisible by heads; dimensions and group-size must be positive")
    if args.fuse_deepopt_activations and args.backend != "deepopt":
        parser.error("activation fusion requires the deepopt backend")
    from deepopt._benchmarking import BenchmarkBackend, timing_plan_from_cost

    try:
        kernels = BenchmarkBackend.from_args(args)
        timing_plan_from_cost(args, 0.0, iterations=args.steps, repeats=1)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    try:
        device = torch.device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if device.type not in ("cpu", "cuda") or device.type == "cpu" and device.index is not None:
        parser.error("device must be cpu, cuda, or cuda:INDEX")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    eager, baseline = benchmark(args, "eager", kernels)
    compiled, actual = benchmark(args, args.backend, kernels)
    tolerance = 2e-2 if args.precision == "bf16" else 2e-4
    for name, value in actual.items():
        torch.testing.assert_close(
            value, baseline[name], rtol=tolerance, atol=tolerance, msg=name
        )
    triton_version = None
    if kernels.policy.backend != "tilelang":
        try:
            triton_version = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            pass
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    text = json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "triton": triton_version,
                    "cuda": torch.version.cuda,
                    "device": device_name,
                },
                "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                             if not key.startswith("_")},
                "numerical_parity": True,
                "kernel_backend": kernels.metadata(device),
                "cases": [eager, compiled],
            },
            indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
