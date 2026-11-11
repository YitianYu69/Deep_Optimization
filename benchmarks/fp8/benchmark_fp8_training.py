"""Diagnose full-model BF16/FP8 training using the installed DeepOpt library.

Copy this file anywhere and run it with the environment that contains DeepOpt,
CUDA PyTorch, timm and torchmetrics. No source-path injection, model downloads,
or library modifications are performed. Run directly with Python, not torchrun.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import cProfile
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import pstats
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import uuid


# Every ablation has a matching control; unchanged fields use the CLI settings.
CASES = {
    "bf16-native": dict(runtime=False, fp8=False, intercept=False, ema=True, fused=False),
    "bf16-runtime": dict(runtime=True, fp8=False, intercept=False, ema=True, fused=False),
    "fp8-intercept": dict(runtime=True, fp8=True, intercept=True, ema=True, fused=False),
    "fp8-prepared": dict(runtime=True, fp8=True, intercept=False, ema=True, fused=False),
    "fp8-delayed": dict(runtime=True, fp8=True, intercept=False, ema=True, fused=False, scaling_recipe="delayed"),
    "bf16-no-ema": dict(runtime=True, fp8=False, intercept=False, ema=False, fused=False),
    "fp8-no-ema": dict(runtime=True, fp8=True, intercept=False, ema=False, fused=False),
    "bf16-fused-adamw": dict(runtime=True, fp8=False, intercept=False, ema=True, fused=True),
    "fp8-fused-adamw": dict(runtime=True, fp8=True, intercept=False, ema=True, fused=True),
}
PAIRS = (
    ("runtime_overhead", "bf16-native", "bf16-runtime"),
    ("fp8_with_interception", "bf16-runtime", "fp8-intercept"),
    ("fp8_prepared", "bf16-runtime", "fp8-prepared"),
    ("fp8_delayed", "bf16-runtime", "fp8-delayed"),
    ("current_vs_delayed", "fp8-prepared", "fp8-delayed"),
    ("disable_interception", "fp8-intercept", "fp8-prepared"),
    ("disable_ema_bf16", "bf16-runtime", "bf16-no-ema"),
    ("disable_ema_fp8", "fp8-prepared", "fp8-no-ema"),
    ("fused_adamw_bf16", "bf16-runtime", "bf16-fused-adamw"),
    ("fused_adamw_fp8", "fp8-prepared", "fp8-fused-adamw"),
)
NOTES = [
    "Single-GPU optimizer updates, not a Trainer epoch: callbacks, validation and checkpoint I/O are excluded.",
    "All cases include CrossEntropyLoss and TorchMetrics Accuracy; EMA and optimizer implementation are controlled ablations.",
    "Each case/batch starts a fresh process with the same seed and initial-state fingerprint; weights update every step.",
    "Resident mode reuses one synthetic GPU batch. Pinned mode copies one fixed CPU batch each step. CIFAR10 uses real augmentation.",
    "CIFAR10 drops incomplete batches to keep each experiment's batch shape fixed. Tail sizes can be separate --batches values.",
    "Throughput timing has no component events, hooks or profiler. Its CUDA events include host submission gaps.",
    "Breakdown, memory, coverage, saved tensors and profiles are separate passes after warmup and continue updating weights.",
    "Component event times include stream gaps, not just kernel execution. Host submission includes implicit waits; do not add the columns.",
    "Profiler CPU totals are nested. Raw CUDA activities may overlap; do not add CPU and CUDA totals.",
    "CUDA activity totals exclude user annotation ranges. Profile-only synchronization prevents warmup work spilling into capture.",
    "Uninitialized-memory filling is controlled independently of deterministic algorithms; its default remains enabled.",
    "Saved-storage totals count distinct storages seen by autograd, including saved parameters; they are not simultaneous peak memory.",
    "Interception-off leaves unconverted native linears dense. FP8 ablations require matching observed FP8 projection calls.",
    "Losses on repeated synthetic data are diagnostics, not convergence or validation-accuracy evidence.",
    "Cases run sequentially. Telemetry snapshots/logs can reveal contention or drift but cannot prove exclusive GPU access.",
]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stats(values):
    samples = list(values)
    ordered = sorted(samples)
    return {"median": statistics.median(samples), "mean": statistics.mean(samples),
            "min": ordered[0], "max": ordered[-1],
            "p95": ordered[max(0, math.ceil(len(ordered) * .95) - 1)], "samples": samples}


def _distribution_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _tree_fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _state_fingerprint(model, torch):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            digest.update(name.encode() + b":" + json.dumps(value, sort_keys=True, allow_nan=False).encode())
            continue
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fp8_policy(args, case):
    if not case["fp8"]:
        return False
    policy = dict(enabled=True, backend="triton", trace_mode=args.trace_mode, intercept_linear=case["intercept"],
                  num_stages=args.num_stages, activation_dtype="e4m3fn", weight_dtype="e4m3fn", grad_dtype=args.grad_format,
                  block_shapes=dict.fromkeys(("activation", "weight", "grad")))
    if case.get("scaling_recipe", "current") == "delayed":
        policy.update(scaling_recipe="delayed", delayed_scaling={"history_length": args.history_length,
                      "amax_algorithm": args.amax_algorithm, "margin": args.margin, "reduce_amax": True})
    return policy


def _delayed_metadata_report(model):
    records = []
    for name, module in model.named_modules():
        cache = getattr(module, "_metadata_cache", None)
        if cache is not None and hasattr(cache, "statistics"):
            records.append({"module": name, **cache.statistics()})
    return records


def _smi_snapshot():
    if shutil.which("nvidia-smi") is None:
        return {"status": "unavailable", "reason": "nvidia-smi not found"}
    queries = {
        "gpus": ["--query-gpu=index,uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,"
                 "power.limit,utilization.gpu,memory.used,memory.total"],
        "compute_processes": ["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"],
    }
    result = {}
    for name, options in queries.items():
        try:
            completed = subprocess.run(["nvidia-smi", *options, "--format=csv"],
                                       capture_output=True, text=True, timeout=10)
            result[name] = {"returncode": completed.returncode, "csv": completed.stdout,
                            "error": completed.stderr.strip() or None}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    return result


@contextmanager
def _telemetry(directory, enabled):
    process = None
    result = {"status": "disabled"}
    with (directory / "gpu-telemetry.csv").open("w", encoding="utf-8") as output:
        if enabled and shutil.which("nvidia-smi"):
            try:
                process = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=timestamp,index,uuid,pstate,clocks.sm,clocks.mem,power.draw,"
                     "power.limit,utilization.gpu,utilization.memory,memory.used", "--format=csv", "--loop=1"],
                    stdout=output, stderr=subprocess.STDOUT)
                result = {"status": "started", "path": "gpu-telemetry.csv", "interval_seconds": 1}
            except OSError as error:
                result = {"status": "unavailable", "error": str(error)}
        elif enabled:
            result = {"status": "unavailable", "reason": "nvidia-smi not found"}
        try:
            yield result
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                result["returncode"] = process.returncode
                if process.returncode not in (0, -signal.SIGTERM):
                    result["status"] = "failed"


class BatchSource:
    def __init__(self, args, torch, device, batch_size):
        self.loader = self.iterator = None
        if args.input_mode == "cifar10":
            from torchvision.datasets import CIFAR10
            from torchvision.transforms import v2

            transform = v2.Compose([
                v2.ToImage(), v2.Resize((args.image_size, args.image_size), antialias=True),
                v2.RandomHorizontalFlip(), v2.RandAugment(num_ops=5, magnitude=9),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([.485, .456, .406], [.229, .224, .225]),
                v2.RandomErasing(p=.5, scale=(.03, .15), ratio=(.3, 3.3), value=0),
            ])
            dataset = CIFAR10(args.data_root, train=True, download=False, transform=transform)
            if len(dataset) < batch_size:
                raise ValueError("CIFAR10 has fewer samples than the requested batch size")
            options = (dict(prefetch_factor=args.prefetch_factor, persistent_workers=True)
                       if args.workers else {})
            self.loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                pin_memory=True, generator=torch.Generator().manual_seed(args.seed + 1), **options)
            return
        generator = torch.Generator().manual_seed(args.seed + 1)
        images = torch.randn(batch_size, 3, args.image_size, args.image_size, generator=generator)
        labels = torch.randint(args.classes, (batch_size,), generator=generator)
        self.fixed = (tuple(value.to(device) for value in (images, labels)) if args.input_mode == "resident"
                      else tuple(value.pin_memory() for value in (images, labels)))

    def next(self):
        if self.loader is None:
            return self.fixed
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class PhaseRecorder:
    def __init__(self, torch):
        self.torch = torch
        self.records = []

    @contextmanager
    def phase(self, name, cuda=True):
        start = self.torch.cuda.Event(enable_timing=True) if cuda else None
        end = self.torch.cuda.Event(enable_timing=True) if cuda else None
        if start is not None:
            start.record()
        began = time.perf_counter()
        try:
            yield
        finally:
            host_ms = (time.perf_counter() - began) * 1000
            if end is not None:
                end.record()
            self.records.append((name, host_ms, start, end))

    def summary(self):
        host, cuda = defaultdict(list), defaultdict(list)
        for name, host_ms, start, end in self.records:
            host[name].append(host_ms)
            if start is not None:
                cuda[name].append(start.elapsed_time(end))
        return {name: {"host_submission_ms": _stats(values),
                       "cuda_stream_ms": _stats(cuda[name]) if cuda[name] else None}
                for name, values in host.items()}


class TrainingSession:
    def __init__(self, args, case, batch_size, torch):
        import deepopt
        import timm
        from torchmetrics import Accuracy
        from deepopt.engine.training.ema import ExponentialMovingAverage

        self.args, self.case, self.torch = args, case, torch
        self.device = torch.device(args.device)
        model = timm.create_model(args.model, pretrained=False, num_classes=args.classes,
                                  img_size=args.image_size)
        self.initial_sha256 = _state_fingerprint(model, torch)
        model = model.to(self.device).train()
        # False leaves foreach/fused selection to native AdamW defaults, matching
        # the user's original optimizer. True is an explicit separate ablation.
        options = {"fused": True} if case["fused"] else {}
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **options)
        self.runtime = None
        if case["runtime"]:
            fp8 = _fp8_policy(args, case)
            self.runtime = deepopt.TrainingRuntime(
                model, self.optimizer, device=self.device, precision="bf16", execution_mode="max_speed",
                fp8=fp8, non_blocking=True)
            model = self.runtime.model
        self.model = model.train()
        self.ema = ExponentialMovingAverage(model, args.ema_decay) if case["ema"] else None
        self.metric = Accuracy(task="multiclass", num_classes=args.classes, sync_on_compute=True).to(self.device)
        self.source = BatchSource(args, torch, self.device, batch_size)
        self.last_loss = None
        self.recorder = None
        self.annotate = False
        self.optimizer.zero_grad(set_to_none=True)

    def phase(self, name, cuda=True):
        if self.recorder is not None:
            return self.recorder.phase(name, cuda)
        return self.torch.profiler.record_function("experiment." + name) if self.annotate else nullcontext()

    def forward(self, batch):
        def compute(values):
            logits = self.model(values[0])
            return logits, self.torch.nn.functional.cross_entropy(logits, values[1])

        if self.runtime is not None:
            return self.runtime.compute(batch, compute)
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            return compute(batch)

    def step(self):
        with self.phase("fetch", cuda=False):
            batch = self.source.next()
        with self.phase("h2d"):
            batch = tuple(value.to(self.device, non_blocking=True) for value in batch)
        with self.phase("forward_loss"):
            logits, loss = self.forward(batch)
        with self.phase("metrics"):
            self.metric.update(logits.detach(), batch[1])
        with self.phase("backward"):
            if self.runtime is not None:
                self.runtime.backward(loss)
            else:
                loss.backward()
        with self.phase("optimizer_zero_grad"):
            if self.runtime is not None:
                self.runtime.step()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            with self.phase("ema"):
                self.ema.update(self.model)
        self.last_loss = loss.detach()


def _timed_updates(session, args, torch, batch_size):
    wall, events, losses = [], [], []
    for repeat in range(args.repeats):
        torch.cuda.synchronize(session.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        began = time.perf_counter()
        start.record()
        for _ in range(args.steps):
            session.step()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - began) * 1000 / args.steps)
        events.append(start.elapsed_time(end) / args.steps)
        loss = float(session.last_loss.item())
        losses.append(loss if math.isfinite(loss) else None)
        print(f"repeat {repeat + 1}/{args.repeats}: {wall[-1]:.3f} ms/update", flush=True)
    return {"wall_ms_per_update": _stats(wall), "cuda_stream_ms_per_update": _stats(events),
            "samples_per_second": batch_size * 1000 / statistics.median(wall),
            "last_loss_per_repeat": losses, "finite_sampled_losses": all(loss is not None for loss in losses)}


def _memory(session, torch):
    session.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(session.device)
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    session.step()
    torch.cuda.synchronize(session.device)
    return {"baseline_allocated_bytes": allocated, "baseline_reserved_bytes": reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "extra_peak_allocated_bytes": torch.cuda.max_memory_allocated() - allocated,
            "allocated_after_update_bytes": torch.cuda.memory_allocated()}


def _coverage(session, torch):
    calls, saved, handles, projections = [], {}, [], []
    parameters = {value.untyped_storage().data_ptr() for value in session.model.parameters()}
    parameter_names = {id(value): name for name, value in session.model.named_parameters()}

    def observe(name):
        def hook(module, inputs):
            value = inputs[0]
            calls.append({"module": name, "type": type(module).__name__, "input_dtype": str(value.dtype),
                          "shape": list(value.shape), "stride": list(value.stride()),
                          "fp8_adapter": hasattr(module, "fp8_dtypes")})
        return hook

    def pack(value):
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        saved.setdefault(key, {"dtype": str(value.dtype), "bytes": storage.nbytes(),
                               "parameter": storage.data_ptr() in parameters})
        return value

    projection_observer = nullcontext()
    observation_error = None
    try:
        canonical = "deepopt.opt.nn.autograd.linear"
        try:
            linear_module = import_module(canonical)
        except ModuleNotFoundError as error:
            if error.name is None or not (canonical == error.name or canonical.startswith(error.name + ".")):
                raise
            linear_module = import_module("deepopt.opt.do_nn.AOTAutograd_Func.linear_autograd_func")
        _DOLinearFunction = linear_module._DOLinearFunction
        original = _DOLinearFunction.forward

        def forward(ctx, value, weight, *args, **kwargs):
            output = original(ctx, value, weight, *args, **kwargs)
            projections.append({"weight": parameter_names.get(id(weight), "<unnamed>"),
                                "shape": list(value.shape), "input_dtype": str(value.dtype),
                                "output_dtype": str(output.dtype), "weight_dtype": str(weight.dtype)})
            return output

        projection_observer = patch.object(_DOLinearFunction, "forward", new=staticmethod(forward))
    except (ImportError, AttributeError) as error:
        observation_error = str(error)
    try:
        for name, module in session.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) or hasattr(module, "fp8_dtypes"):
                handles.append(module.register_forward_pre_hook(observe(name)))
        with projection_observer, torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            session.step()
        torch.cuda.synchronize(session.device)
    finally:
        for handle in handles:
            handle.remove()
    grouped = defaultdict(int)
    for record in saved.values():
        grouped[("parameter:" if record["parameter"] else "other:") + record["dtype"]] += record["bytes"]
    report = session.runtime.transformation_report if session.runtime is not None else None
    return {"module_calls": calls, "fp8_projection_calls": projections,
            "projection_observation_error": observation_error,
            "transformation_report": asdict(report) if is_dataclass(report) else repr(report),
            "unique_saved_storages": len(saved), "saved_storage_bytes_by_kind_dtype": dict(grouped),
            "functional_calls_note": "Module hooks do not observe rewritten standalone functional projection calls."}


def _profile_summary(profiler, torch, steps, limit):
    cpu, cuda = [], {}
    for event in profiler.key_averages():
        if event.device_type == torch.autograd.DeviceType.CPU:
            cpu.append({"name": event.key, "count": event.count,
                        "self_cpu_ms_per_update": event.self_cpu_time_total / 1000 / steps,
                        "total_cpu_ms_per_update": event.cpu_time_total / 1000 / steps})
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and not getattr(event, "is_user_annotation", False):
            row = cuda.setdefault(event.name, {"name": event.name, "count": 0, "ms_per_update": 0.0})
            row["count"] += 1
            duration = getattr(event, "device_time_total", None)
            row["ms_per_update"] += (duration if duration is not None else event.cuda_time_total) / 1000 / steps
    cpu.sort(key=lambda row: row["self_cpu_ms_per_update"], reverse=True)
    gpu = sorted(cuda.values(), key=lambda row: row["ms_per_update"], reverse=True)
    return {"profiled_updates": steps, "top_cpu_ops": cpu[:limit], "top_cuda_activities": gpu[:limit],
            "cuda_activity_status": "recorded" if gpu else "unavailable",
            "cuda_activity_sum_ms_per_update": sum(row["ms_per_update"] for row in gpu) if gpu else None,
            "cuda_activity_count_per_update": sum(row["count"] for row in gpu) / steps if gpu else None,
            "runtime_calls": [row for row in cpu if row["name"].startswith(("cuda", "cuLaunch"))]}


def _profile(session, args, torch, directory):
    if not args.profile_steps:
        return {"status": "disabled"}
    torch.cuda.synchronize(session.device)
    session.annotate = True
    try:
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=args.profile_steps, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as profiler:
            for _ in range(args.profile_steps + 1):
                session.step()
                # Only the profiling pass waits per update. Finish warmup GPU
                # work before Kineto changes capture windows on the CPU.
                torch.cuda.synchronize(session.device)
                profiler.step()
        result = {"status": "ok", **_profile_summary(profiler, torch, args.profile_steps, args.profile_rows)}
        if args.trace:
            profiler.export_chrome_trace(str(directory / "training-trace.json"))
            result["trace"] = "training-trace.json"
        return result
    except RuntimeError as error:
        return {"status": "unavailable", "error": str(error)}
    finally:
        session.annotate = False


def _host_profile(session, args, torch):
    if not args.host_profile:
        return {"status": "disabled"}
    torch.cuda.synchronize(session.device)
    profiler = cProfile.Profile()
    with profiler:
        for _ in range(args.breakdown_steps):
            session.step()
    torch.cuda.synchronize(session.device)
    rows = [{"function": f"{path}:{line}:{name}", "count": count,
             "self_ms_per_update": self_time * 1000 / args.breakdown_steps,
             "cumulative_ms_per_update": cumulative * 1000 / args.breakdown_steps}
            for (path, line, name), (_, count, self_time, cumulative, _) in pstats.Stats(profiler).stats.items()]
    rows.sort(key=lambda row: row["self_ms_per_update"], reverse=True)
    return {"status": "ok", "updates": args.breakdown_steps, "top_functions": rows[:args.profile_rows]}


def _configure_determinism(args, torch):
    torch.use_deterministic_algorithms(args.deterministic)
    torch.utils.deterministic.fill_uninitialized_memory = args.fill_uninitialized_memory
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False


def _worker(payload):
    import torch
    import deepopt

    args = argparse.Namespace(**payload["settings"])
    directory = Path(payload["directory"])
    case = CASES[payload["case"]]
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch and an NVIDIA GPU are required")
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", 0)
        args.device = str(device)
    torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Run directly with Python on one GPU, not multi-rank torchrun")
    torch.manual_seed(args.seed)
    _configure_determinism(args, torch)
    stages = args.num_stages
    args.num_stages = stages if stages == "auto" else int(stages)
    properties = torch.cuda.get_device_properties(device)
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "deepopt": _distribution_version("deep-optimization"), "timm": _distribution_version("timm"),
                   "torchmetrics": _distribution_version("torchmetrics"), "triton": _distribution_version("triton"),
                   "deepopt_source": str(Path(deepopt.__file__).resolve()),
                   "deepopt_source_sha256": _tree_fingerprint(Path(deepopt.__file__).resolve().parent),
                   "gpu": properties.name, "capability": [properties.major, properties.minor],
                   "gpu_memory_bytes": properties.total_memory, "pid": os.getpid(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                   "torch_cpu_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                   "cpu_count": os.cpu_count(), "cpu_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                   "flags": {name: os.environ.get(name) for name in (
                       "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "TRITON_INTERPRET",
                       "TRITON_DEFAULT_FP_FUSION", "NVIDIA_TF32_OVERRIDE", "TORCH_COMPILE_DEBUG",
                       "CUBLAS_WORKSPACE_CONFIG")}}
    started = time.perf_counter()
    with _telemetry(directory, args.telemetry) as telemetry:
        print("Setting up model, optimizer, runtime and input source...", flush=True)
        session = TrainingSession(args, case, payload["batch_size"], torch)
        recipe = None
        if case["fp8"]:
            try:
                from deepopt.opt._recipe import fp8_recipe
                recipe = (fp8_recipe("triton", session.runtime.fp8) if "settings" in inspect.signature(fp8_recipe).parameters
                          else fp8_recipe("triton"))
            except ModuleNotFoundError as error:
                if error.name != "deepopt.opt._recipe":
                    raise
                recipe = {"status": "unavailable", "reason": "Installed library exposes no recipe identity."}
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - started
        print("First training update (may compile/autotune)...", flush=True)
        began = time.perf_counter()
        session.step()
        torch.cuda.synchronize(device)
        first_step_seconds = time.perf_counter() - began
        print(f"First update: {first_step_seconds:.3f}s; warming {args.warmups} updates...", flush=True)
        for _ in range(args.warmups):
            session.step()
        before = _smi_snapshot()
        timing = _timed_updates(session, args, torch, payload["batch_size"])
        after = _smi_snapshot()
        result = {"status": "running", "run_id": payload["run_id"], "case": payload["case"],
                  "batch_size": payload["batch_size"], "case_policy": case, "environment": environment,
                  "initial_state_sha256": session.initial_sha256, "setup_seconds": setup_seconds,
                  "first_update_seconds": first_step_seconds, "timing": timing, "telemetry": telemetry,
                  "effective_fp8_policy": asdict(session.runtime.fp8) if session.runtime is not None else None,
                  "numerical_recipe": recipe,
                  "delayed_metadata_after_timing": _delayed_metadata_report(session.model),
                  "gpu_before_timing": before, "gpu_after_timing": after}
        # Outside timing; compare the same case across allocation-fill runs.
        # A mismatch requires investigation, rather than accepting latency alone.
        result["after_timing_state_sha256"] = _state_fingerprint(session.model, torch)
        _write_json(directory / "result.json", result)
        print("Measuring warmed memory and component breakdown...", flush=True)
        result["memory"] = _memory(session, torch)
        _write_json(directory / "result.json", result)
        recorder = PhaseRecorder(torch)
        session.recorder = recorder
        try:
            for _ in range(args.breakdown_steps):
                session.step()
            torch.cuda.synchronize(device)
        finally:
            session.recorder = None
        result["breakdown"] = recorder.summary()
        result["coverage"] = _coverage(session, torch)
        _write_json(directory / "result.json", result)
        print("Collecting separate CUDA/CPU and Python profiles...", flush=True)
        result["profile"] = _profile(session, args, torch, directory)
        _write_json(directory / "result.json", result)
        result["host_profile"] = _host_profile(session, args, torch)
        accuracy = float(session.metric.compute().item())
        result.update(status="ok", diagnostic_accuracy=accuracy if math.isfinite(accuracy) else None)
        return result


def _comparisons(results):
    by_key = {(row["case"], row["batch_size"]): row for row in results if row["status"] == "ok"}
    comparisons = []
    for batch in sorted({row["batch_size"] for row in results}):
        for name, control, candidate in PAIRS:
            if (control, batch) not in by_key or (candidate, batch) not in by_key:
                continue
            a, b = by_key[control, batch], by_key[candidate, batch]
            baseline = a["timing"]["wall_ms_per_update"]["median"]
            tested = b["timing"]["wall_ms_per_update"]["median"]
            same = a["initial_state_sha256"] == b["initial_state_sha256"]
            coverage = None
            if CASES[control]["fp8"] and CASES[candidate]["fp8"]:
                ac = a.get("coverage", {}).get("fp8_projection_calls")
                bc = b.get("coverage", {}).get("fp8_projection_calls")
                coverage = (sorted(json.dumps(row, sort_keys=True) for row in ac)
                            == sorted(json.dumps(row, sort_keys=True) for row in bc)) if ac and bc else False
            finite = a["timing"].get("finite_sampled_losses", True) and b["timing"].get("finite_sampled_losses", True)
            ah = a.get("environment", {}).get("deepopt_source_sha256")
            bh = b.get("environment", {}).get("deepopt_source_sha256")
            same_library = ah == bh if ah and bh else None
            comparable = same and coverage is not False and finite and same_library is not False
            comparisons.append({"experiment": name, "batch_size": batch, "control": control, "candidate": candidate,
                                "speedup": baseline / tested if comparable else None,
                                "candidate_minus_control_ms": tested - baseline if comparable else None,
                                "same_initial_state": same, "same_fp8_projection_calls": coverage,
                                "finite_sampled_losses": finite, "same_library_source": same_library})
    return comparisons


def _run_child(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        try:
            elapsed = 0
            while True:
                try:
                    return process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    elapsed += 30
                    print(f"  still running ({elapsed}s); progress: {log_path}", file=sys.stderr, flush=True)
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _run(args):
    directory = args.output.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    settings = {name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items() if name != "_worker"}
    run_id = uuid.uuid4().hex
    report = {"report_version": 2, "run_id": run_id, "settings": settings, "notes": NOTES,
              "results": [], "comparisons": []}
    _write_json(directory / "summary.json", report)
    for batch in args.batches:
        for case in args.cases:
            child_dir = directory / f"{case}-b{batch}"
            child_dir.mkdir(parents=True, exist_ok=True)
            payload = {"settings": settings, "case": case, "batch_size": batch, "directory": str(child_dir), "run_id": run_id}
            print(f"Measuring {case}, batch {batch}...", file=sys.stderr, flush=True)
            code = _run_child([sys.executable, str(Path(__file__).resolve()), "--_worker", json.dumps(payload)],
                              child_dir / "worker.log")
            result_path = child_dir / "result.json"
            # A failed worker must never reuse an older successful report.
            if code == 0 and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("run_id") != run_id:
                    result = {"status": "error", "case": case, "batch_size": batch,
                              "error": "Worker returned no report for this run; an older result was ignored."}
            else:
                result = {"status": "error", "case": case, "batch_size": batch, "returncode": code,
                          "log": str(child_dir / "worker.log")}
                if result_path.exists():
                    failed = json.loads(result_path.read_text(encoding="utf-8"))
                    if failed.get("run_id") == run_id:
                        result = {**failed, **result}
            report["results"].append(result)
            report["comparisons"] = _comparisons(report["results"])
            _write_json(directory / "summary.json", report)
            print(f"  {result['status']}; summary saved", file=sys.stderr, flush=True)
            if result.get("error"):
                print(f"  {result['error']}", file=sys.stderr, flush=True)
    print(f"Report: {directory / 'summary.json'}", flush=True)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("fp8-training-results"), help="Result directory.")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=["bf16-runtime", "fp8-prepared", "fp8-delayed"])
    parser.add_argument("--batches", nargs="+", type=int, default=[512])
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01, help="Matches AdamW's default in the original script.")
    parser.add_argument("--ema-decay", type=float, default=.95)
    parser.add_argument("--trace-mode", choices=("full", "modules", "auto"), default="full")
    parser.add_argument("--num-stages", choices=("auto", "1", "2", "3", "4", "5"), default="3")
    parser.add_argument("--grad-format", choices=("e4m3fn", "e5m2"), default="e4m3fn")
    parser.add_argument("--history-length", type=int, default=1024)
    parser.add_argument("--amax-algorithm", choices=("max", "most_recent"), default="max")
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--input-mode", choices=("resident", "pinned", "cifar10"), default="resident")
    parser.add_argument("--data-root", help="Existing CIFAR10 directory; downloads are disabled.")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20, help="Updates per uninstrumented timing repeat.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--breakdown-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=3, help="0 disables CPU/CUDA profiling.")
    parser.add_argument("--profile-rows", type=int, default=40)
    parser.add_argument("--trace", action="store_true", help="Save Chrome traces alongside JSON summaries.")
    parser.add_argument("--host-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--telemetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-uninitialized-memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Control allocation filling independently of deterministic algorithms; disabling is a diagnostic experiment.")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args._worker:
        payload = json.loads(args._worker)
        destination = Path(payload["directory"]) / "result.json"
        try:
            _write_json(destination, _worker(payload))
        except Exception as error:
            partial = json.loads(destination.read_text()) if destination.exists() else {}
            if partial.get("run_id") != payload["run_id"]:
                partial = {}
            partial.update(status="error", run_id=payload["run_id"], case=payload["case"],
                           batch_size=payload["batch_size"], error=str(error), traceback=traceback.format_exc())
            _write_json(destination, partial)
            raise
        return
    if (any(value < 1 for value in (*args.batches, args.image_size, args.classes, args.steps,
                                    args.repeats, args.breakdown_steps, args.profile_rows, args.prefetch_factor))
            or args.warmups < 0 or args.profile_steps < 0 or args.workers < 0):
        parser.error("Dimensions/counts must be positive; warmups, profile-steps and workers may be zero.")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative, both finite.")
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        parser.error("EMA decay must be in [0, 1).")
    if not 1 <= args.history_length <= 4096 or not -32 <= args.margin <= 32:
        parser.error("Delayed history length must be 1–4096 and margin must be -32–32")
    if not args.device.startswith("cuda"):
        parser.error("--device must select CUDA")
    if args.input_mode == "cifar10" and not args.data_root:
        parser.error("--data-root is required for cifar10 input mode")
    if args.input_mode == "cifar10" and args.classes != 10:
        parser.error("CIFAR10 mode requires --classes 10")
    if len(set(args.cases)) != len(args.cases) or len(set(args.batches)) != len(args.batches):
        parser.error("Cases and batch sizes must not be repeated.")
    report = _run(args)
    if any(row["status"] != "ok" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
