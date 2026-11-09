"""Eager Linear diagnostics for installed DeepOpt, PyTorch BF16 and TorchAO FP8.

Copy this file anywhere and run it with the GPU environment's Python. No source
path injection, compilation, producer fusion, datasets or downloads are used.
Results are written incrementally; a ZIP contains this run's JSON/CSV/logs/traces.
"""

import argparse
import cProfile
from contextlib import contextmanager, ExitStack, nullcontext
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import pstats
import shutil
from statistics import mean, median
import subprocess
import sys
import tempfile
from time import perf_counter
import traceback
from unittest.mock import patch
from uuid import uuid4
import zipfile


IMPLEMENTATIONS = ("bf16", "deepopt-current", "deepopt-delayed", "torchao")
PROJECTIONS = {"qkv": (768, 2304, "fp32", 12), "attention": (768, 768, "bf16", 12),
               "fc1": (768, 3072, "fp32", 12), "fc2": (3072, 768, "bf16", 12),
               "head": (768, 10, "fp32", 1)}
REPORT_VERSION = 2
NOTES = [
    "All full Linear measurements execute eagerly. No torch.compile or graph replay is used by this harness.",
    "The default update timing includes zero_grad, forward, backward, delayed scale commit and a real AdamW step.",
    "Phase events, hooks, GEMM interception and profilers are used only in separate diagnostic passes.",
    "Phase CUDA stream intervals include host submission gaps; CUDA activity sums contain actual device activities.",
    "Matched-fp32 matches internal FP8 GEMM output dtypes, common external bias and BF16 external output. BF16 stays BF16 AMP.",
    "Equal formats/output dtypes do not establish identical scaling, accumulation, rounding or training quality.",
    "The ten-class head uses BF16 in every implementation; its FP8 conversion is explicitly skipped.",
    "Weighted isolated times are diagnostic sums, not predictions of full ViT training or its batched optimizer.",
    "Python cProfile covers the calling thread; autograd workers and GPU backpressure limit CPU-time attribution.",
    "Prepared GEMM replay excludes quantization and optimizer updates. Shared-operand probes omit bias in both backends.",
]


def stats(values):
    ordered = sorted(values)
    return {"median": median(values), "mean": mean(values), "min": ordered[0], "max": ordered[-1],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "samples": list(values)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cases(args, batch):
    return [dict(projection=name, M=batch if name == "head" else batch * args.tokens,
                 K=PROJECTIONS[name][0], N=PROJECTIONS[name][1], input_dtype=PROJECTIONS[name][2],
                 multiplicity=PROJECTIONS[name][3]) for name in args.projections]


def case_seed(args, case):
    identity = json.dumps([args.seed, case["projection"], case["M"], case["K"], case["N"]])
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


def tensor_info(value):
    return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
            "device": str(value.device), "contiguous": value.is_contiguous(),
            "address_mod_16": value.data_ptr() % 16}


def fingerprint(value, torch):
    """Bounded, deterministic sample; explicitly not a full-payload hash."""
    flat = value.detach().reshape(-1)
    count = min(flat.numel(), 65536)
    indices = torch.linspace(0, flat.numel() - 1, count, device=value.device, dtype=torch.float64).long()
    sample = flat[indices].contiguous().view(torch.uint8).cpu().numpy().tobytes()
    header = json.dumps([list(value.shape), str(value.dtype), count]).encode()
    return {"sha256": hashlib.sha256(header + sample).hexdigest(), "sample_elements": count,
            "total_elements": value.numel(), "scheme": "evenly-spaced-elements-v1"}


def source_record(module, subdirectory=None):
    root = Path(module.__file__).resolve().parent
    files = sorted((root / subdirectory if subdirectory else root).rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return {"file": str(Path(module.__file__).resolve()), "sha256": digest.hexdigest(), "python_files": len(files)}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def smi():
    if shutil.which("nvidia-smi") is None:
        return {"status": "unavailable"}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pstate,clocks.sm,"
                                 "clocks.mem,power.draw,power.limit,utilization.gpu,memory.used", "--format=csv"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "error": repr(error)}
    return {"returncode": result.returncode, "csv": result.stdout, "stderr": result.stderr}


def environment(args):
    import torch

    result = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
              "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
              "hip": torch.version.hip, "cuda_available": torch.cuda.is_available(), "packages": {},
              "environment_flags": {key: os.environ.get(key) for key in
                                    ("CUBLAS_WORKSPACE_CONFIG", "TRITON_DEFAULT_FP_FUSION", "TRITON_CACHE_DIR")}}
    for name in ("deepopt", "torchao", "triton"):
        try:
            module = importlib.import_module(name)
            entry = {"status": "ok", "version": package_version("deep-optimization" if name == "deepopt" else name)}
            if name != "triton":
                entry["source"] = source_record(module, "float8" if name == "torchao" else None)
            if name == "torchao":
                entry["configuration"] = repr(torchao_config(args, torch))
            result["packages"][name] = entry
        except Exception as error:
            result["packages"][name] = {"status": "unavailable", "error": repr(error)}
    if result["cuda_available"]:
        torch.cuda.set_device(args.device)
        properties = torch.cuda.get_device_properties(args.device)
        result.update(gpu=properties.name, capability=list(torch.cuda.get_device_capability(args.device)),
                      total_memory_bytes=properties.total_memory, multiprocessors=properties.multi_processor_count)
    return result


def configure(args, torch):
    torch.cuda.set_device(args.device)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.utils.deterministic.fill_uninitialized_memory = args.fill_uninitialized_memory
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def torchao_config(args, torch):
    from torchao.float8 import CastConfig, Float8GemmConfig, Float8LinearConfig

    grad = torch.float8_e5m2 if args.format == "hybrid" else torch.float8_e4m3fn
    casts = {"cast_config_input": CastConfig(target_dtype=torch.float8_e4m3fn),
             "cast_config_weight": CastConfig(target_dtype=torch.float8_e4m3fn),
             "cast_config_grad_output": CastConfig(target_dtype=grad)}
    if args.accumulation == "fast":
        casts.update({f"gemm_config_{role}": Float8GemmConfig(use_fast_accum=True)
                      for role in ("output", "grad_input", "grad_weight")})
    return Float8LinearConfig(**casts)


class Session:
    def __init__(self, args, implementation, math_policy, case, torch, *, deepopt_backend="triton"):
        self.args, self.implementation, self.math_policy, self.case, self.torch = args, implementation, math_policy, case, torch
        self.device = torch.device(args.device)
        self.stage, self.recorder, self.annotate = "outside", None, False
        torch.manual_seed(case_seed(args, case))
        self.source = torch.nn.Linear(case["K"], case["N"], bias=args.bias, device=self.device, dtype=torch.float32)
        dtype = torch.float32 if case["input_dtype"] == "fp32" else torch.bfloat16
        self.x = torch.randn(case["M"], case["K"], device=self.device, dtype=dtype, requires_grad=True)
        self.grad = torch.randn(case["M"], case["N"], device=self.device, dtype=torch.bfloat16) / case["M"]
        self.initial = {"weight": fingerprint(self.source.weight, torch), "input": fingerprint(self.x, torch),
                        "upstream_gradient": fingerprint(self.grad, torch),
                        "bias": fingerprint(self.source.bias, torch) if args.bias else None}
        self.parameters = tuple(self.source.parameters())
        dense = implementation == "bf16" or case["projection"] == "head"
        matched = math_policy == "matched-fp32"
        projection = self.source
        if matched:
            projection = torch.nn.Linear(case["K"], case["N"], bias=False, device="meta")
            projection.weight = self.source.weight
        self.settings = None
        self.boundary = nullcontext
        self.configuration = None
        if not dense and implementation.startswith("deepopt"):
            import deepopt
            from deepopt.opt.policy import prepare_fp8_model

            self.settings = deepopt.FP8Config(
                enabled=True, backend=deepopt_backend, trace_mode="modules", intercept_linear=False,
                backward_backend=args.backward_backend,
                out_dtype="fp32" if matched else "bf16", num_stages=args.num_stages,
                grad_dtype="e5m2" if args.format == "hybrid" else "e4m3fn",
                scaling_recipe="delayed" if implementation.endswith("delayed") else "current",
                delayed_scaling=deepopt.DelayedScalingConfig(args.history_length, args.amax_algorithm, args.margin))
            projection = prepare_fp8_model(projection, self.settings)[0]
            self.configuration = asdict(self.settings)
            self.amp = deepopt.autocast(self.device.type, dtype=torch.bfloat16, enabled=not matched,
                                        fp8=self.settings, check_finite=False)
            if implementation.endswith("delayed"):
                self.boundary = deepopt.fp8_scaling_scope
        else:
            self.amp = torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=dense or not matched)
            if not dense:
                from torchao.float8 import convert_to_float8_training

                configuration = torchao_config(args, torch)
                projection = convert_to_float8_training(projection, config=configuration)
                self.configuration = repr(configuration)
                if not type(projection).__module__.startswith("torchao."):
                    raise RuntimeError("TorchAO did not convert the requested projection")
        self.projection, self.matched, self.dense = projection, matched, dense
        # The benchmark owns the wrapper bias without registering a second Parameter.
        self.optimizer = torch.optim.AdamW(self.parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        if {id(p) for p in projection.parameters()} - {id(p) for p in self.parameters}:
            raise RuntimeError("Projection introduced different parameter objects")
        if {id(p) for p in projection.parameters()} != {id(p) for p in self.parameters if not matched or p is self.source.weight}:
            raise RuntimeError("Projection lost FP32 parameter ownership")
        self.completed_updates = 0

    @contextmanager
    def phase(self, name):
        previous, self.stage = self.stage, name
        context = (self.recorder.phase(name) if self.recorder else
                   self.torch.profiler.record_function("linear." + name) if self.annotate else nullcontext())
        try:
            with context:
                yield
        finally:
            self.stage = previous

    def clear(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.x.grad = None

    def step(self, *, update=None):
        update = self.args.optimizer_updates if update is None else update
        with self.phase("zero_grad"):
            self.clear()
        boundary = self.boundary()
        boundary.__enter__()
        try:
            with self.phase("forward"), self.amp:
                value = self.x.float() if self.matched else self.x
                output = self.projection(value)
                if self.matched:
                    output = output.float()
                    if self.source.bias is not None:
                        output = output + self.source.bias
                    output = output.to(self.torch.bfloat16)
            with self.phase("backward"):
                output.backward(self.grad)
        except BaseException:
            boundary.__exit__(*sys.exc_info())
            raise
        else:
            with self.phase("scale_update"):
                boundary.__exit__(None, None, None)
        if update:
            with self.phase("optimizer"):
                self.optimizer.step()
            self.completed_updates += 1
        return output.detach()


class PhaseRecorder:
    def __init__(self, torch):
        self.torch, self.records = torch, []

    @contextmanager
    def phase(self, name):
        start, end = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        host = perf_counter()
        yield
        elapsed = (perf_counter() - host) * 1000
        end.record()
        self.records.append((name, elapsed, start, end))

    def summary(self):
        host, device = {}, {}
        for name, elapsed, start, end in self.records:
            host.setdefault(name, []).append(elapsed)
            device.setdefault(name, []).append(start.elapsed_time(end))
        return {name: {"host_submission_ms": stats(host[name]), "cuda_stream_ms": stats(values)}
                for name, values in device.items()}


def measure(operation, args, torch, iterations=None):
    wall, stream = [], []
    count = iterations or args.iterations
    for _ in range(args.repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        started = perf_counter()
        start.record()
        for _ in range(count):
            operation()
        end.record()
        torch.cuda.synchronize()
        wall.append((perf_counter() - started) * 1000 / count)
        stream.append(start.elapsed_time(end) / count)
    return {"wall_ms": stats(wall), "cuda_stream_ms": stats(stream), "iterations_per_repeat": count}


def role_for(stage, output, case):
    if stage == "forward":
        return "forward"
    shape = tuple(output.shape)
    if shape == (case["M"], case["K"]) and case["M"] not in (case["K"], case["N"]):
        return "input_gradient"
    if shape in ((case["N"], case["K"]), (case["K"], case["N"])):
        return "weight_gradient"
    return "unresolved_backward"


class Capture:
    """Temporary diagnostics only. Timed production calls do not pass through this."""
    def __init__(self, session, *, retain=True):
        self.session, self.retain = session, retain
        self.gemms, self.quantization = [], []

    def record(self, kind, function, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        role = role_for(self.session.stage, tensor, self.session.case)
        if kind == "deepopt":
            a, b = args[:2]
            raw_a, raw_b = a.data, b.data.T
            scales = (a.scale_inv, b.scale_inv)
            record = {"a": tensor_info(raw_a), "b": tensor_info(raw_b),
                      "scale_a": tensor_info(scales[0]), "scale_b": tensor_info(scales[1])}
            requested = kwargs["backend"]
            fallback = None
            if requested in ("native_fast", "native_precise"):
                from deepopt.opt.backends.native_fp8 import unsupported_reason

                fallback = unsupported_reason(a, b)
                record["use_fast_accum"] = requested == "native_fast" if fallback is None else None
            record.update(requested_backend=requested, actual_backend="triton" if fallback else requested,
                          fallback_reason=fallback)
        else:
            names = [argument.name for argument in function._schema.arguments]
            bound = {**dict(zip(names, args)), **kwargs}
            raw_a = bound["mat1"] if kind == "addmm" else bound["self"]
            raw_b = bound["mat2"]
            scales = (bound.get("scale_a"), bound.get("scale_b"))
            record = {"a": tensor_info(raw_a), "b": tensor_info(raw_b),
                      "scale_a": tensor_info(scales[0]) if scales[0] is not None else None,
                      "scale_b": tensor_info(scales[1]) if scales[1] is not None else None}
            if kind == "scaled_mm":
                record["use_fast_accum"] = bound.get("use_fast_accum", False)
        record.update(kind=kind, role=role, output=tensor_info(tensor))
        payload = (function, args, kwargs, raw_a, raw_b, scales, tensor.dtype) if self.retain else None
        self.gemms.append((record, payload))

    @contextmanager
    def enter(self):
        torch, session = self.session.torch, self.session
        with ExitStack() as stack:
            if not session.dense and session.implementation.startswith("deepopt"):
                module = importlib.import_module("deepopt.opt.do_nn.AOTAutograd_Func.linear_autograd_func")
                original = module.gemm_nt

                def gemm(*args, **kwargs):
                    with torch.profiler.record_function("probe.deepopt_gemm." + session.stage):
                        result = original(*args, **kwargs)
                    self.record("deepopt", original, args, kwargs, result)
                    return result

                stack.enter_context(patch.object(module, "gemm_nt", gemm))
                if session.implementation.endswith("delayed"):
                    state_module = importlib.import_module("deepopt.opt.scaling_strategy.delayed")
                    quantize = state_module.DelayedScalingState.quantize

                    def delayed(state, value, role, *args, **kwargs):
                        self.quantization.append({"stage": session.stage, "role": role, "input": tensor_info(value)})
                        with torch.profiler.record_function("probe.quantize." + role):
                            return quantize(state, value, role, *args, **kwargs)

                    stack.enter_context(patch.object(state_module.DelayedScalingState, "quantize", delayed))
                else:
                    # The aliases are looked up by the real autograd function.
                    for name in ("quantize_fp8", "_quantize_tensorwise_pair", "_quantize_tensorwise_transpose"):
                        quantize = getattr(module, name)

                        def current(value, *args, _function=quantize, **kwargs):
                            self.quantization.append({"stage": session.stage, "role": "infer_from_input_shape",
                                                      "input": tensor_info(value)})
                            with torch.profiler.record_function("probe.quantize.current"):
                                return _function(value, *args, **kwargs)

                        stack.enter_context(patch.object(module, name, current))
            elif not session.dense:
                original = torch._scaled_mm
                schema_op = torch.ops.aten._scaled_mm.default

                def scaled(*args, **kwargs):
                    with torch.profiler.record_function("probe.native_scaled_mm." + session.stage):
                        result = original(*args, **kwargs)
                    self.record("scaled_mm", schema_op, args, kwargs, result)
                    return result

                stack.enter_context(patch.object(torch, "_scaled_mm", scaled))
                module = importlib.import_module("torchao.float8.float8_linear")
                if hasattr(module, "hp_tensor_to_float8_dynamic"):
                    quantize = module.hp_tensor_to_float8_dynamic

                    def dynamic(value, *args, **kwargs):
                        role = str(kwargs.get("gemm_input_role", "unknown"))
                        self.quantization.append({"stage": session.stage, "role": role, "input": tensor_info(value)})
                        with torch.profiler.record_function("probe.quantize." + role):
                            return quantize(value, *args, **kwargs)

                    stack.enter_context(patch.object(module, "hp_tensor_to_float8_dynamic", dynamic))
            else:
                from torch.utils._python_dispatch import TorchDispatchMode
                capture = self

                class DenseMode(TorchDispatchMode):
                    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                        kwargs = kwargs or {}
                        result = function(*args, **kwargs)
                        if function in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
                            capture.record("mm" if function is torch.ops.aten.mm.default else "addmm",
                                           function, args, kwargs, result)
                        return result

                stack.enter_context(DenseMode())
            yield self


def error_stats(actual, expected, torch):
    with torch.no_grad():
        finite = bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())
        if not finite:
            return {"finite": False, "relative_l2": None, "max_abs": None}
        difference = actual.float() - expected.float()
        return {"finite": True, "relative_l2": float(difference.norm() / expected.float().norm().clamp_min(1e-30)),
                "max_abs": float(difference.abs().max())}


def numerical(session, output):
    torch = session.torch
    with torch.no_grad(), torch.autocast(session.device.type, enabled=False):
        x, weight, grad = session.x.float(), session.source.weight, session.grad.float()
        expected = x @ weight.T
        if session.source.bias is not None:
            expected += session.source.bias
        errors = {"forward": error_stats(output, expected, torch)}
        del expected
        errors["input_gradient"] = error_stats(session.x.grad, grad @ weight, torch)
        errors["weight_gradient"] = error_stats(session.source.weight.grad, grad.T @ x, torch)
        if session.source.bias is not None:
            errors["bias_gradient"] = error_stats(session.source.bias.grad, grad.sum(0), torch)
    return {"reference": "FP32 matmul with original input, FP32 parameters and the explicit upstream gradient; TF32 disabled",
            "errors": errors, "all_finite": all(value["finite"] for value in errors.values())}


def prepared_gemms(capture, args):
    torch = capture.session.torch
    rows = []
    for record, payload in capture.gemms:
        function, positional, keywords, a, b, scales, dtype = payload
        replay = lambda function=function, positional=positional, keywords=keywords: function(*positional, **keywords)
        row = {"actual_call": record}
        with torch.no_grad(), torch.autocast(capture.session.device.type, enabled=False):
            for _ in range(args.probe_warmups):
                replay()
            row["actual_backend_replay"] = measure(replay, args, torch, args.probe_iterations)
            if scales[0] is not None and a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                try:
                    from deepopt.opt.backends import gemm_nt
                    from deepopt.opt.quantization.tensor import QuantizedTensor

                    qa = QuantizedTensor(a, scales[0].reshape(1, 1))
                    qb = QuantizedTensor(b.T, scales[1].reshape(1, 1))
                    operations = {"deepopt_triton": lambda: gemm_nt(qa, qb, backend="triton", out_dtype=dtype,
                                                                  num_stages=args.num_stages)}
                    for fast in (True, False):
                        operations["native_fast" if fast else "native_precise"] = (
                            lambda fast=fast: torch._scaled_mm(a, b, scale_a=scales[0].reshape(()),
                                                              scale_b=scales[1].reshape(()), out_dtype=dtype,
                                                              use_fast_accum=fast))
                    row["shared_operands_bias_free"] = {}
                    reference = None
                    for name, operation in operations.items():
                        try:
                            started = perf_counter()
                            value = operation()
                            torch.cuda.synchronize()
                            cold = perf_counter() - started
                            if isinstance(value, tuple):
                                value = value[0]
                            if reference is None:
                                reference = value
                                row["shared_operands_reference_backend"] = name
                            accuracy = error_stats(value, reference, torch)
                            del value
                            for _ in range(args.probe_warmups):
                                operation()
                            row["shared_operands_bias_free"][name] = {
                                "status": "ok", "first_use_seconds": cold,
                                "timing": measure(operation, args, torch, args.probe_iterations),
                                "difference_vs_first_available_backend": accuracy}
                        except Exception as error:
                            row["shared_operands_bias_free"][name] = {"status": "error", "error": repr(error)}
                    del reference
                except Exception as error:
                    row["shared_operands_error"] = repr(error)
        rows.append(row)
    return rows


def profile_summary(profiler, torch, count, limit):
    cpu, gpu = [], {}
    for event in profiler.key_averages():
        if event.device_type == torch.autograd.DeviceType.CPU:
            cpu.append({"name": event.key, "count": event.count,
                        "self_ms_per_update": event.self_cpu_time_total / 1000 / count,
                        "nested_total_ms_per_update": event.cpu_time_total / 1000 / count})
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA or getattr(event, "is_user_annotation", False):
            continue
        duration = getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0))
        row = gpu.setdefault(event.name, {"name": event.name, "count": 0, "ms_per_update": 0.})
        row["count"] += 1
        row["ms_per_update"] += duration / 1000 / count
    all_gpu = sorted(gpu.values(), key=lambda row: row["ms_per_update"], reverse=True)
    return {"profiled_updates": count, "top_cpu_ops": sorted(cpu, key=lambda row: row["self_ms_per_update"], reverse=True)[:limit],
            "cuda_api_ops": [row for row in cpu if row["name"].startswith(("cuda", "cuLaunch", "cuMemcpy"))],
            "cuda_activity_sum_ms_per_update": sum(row["ms_per_update"] for row in all_gpu) if all_gpu else None,
            "cuda_activities": all_gpu, "cuda_activity_status": "recorded" if all_gpu else "unavailable"}


def profile(session, args, directory):
    if not args.profile_steps:
        return {"status": "disabled"}
    torch = session.torch
    torch.cuda.synchronize()
    session.annotate = True
    try:
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=args.profile_steps, repeat=1)
        capture = Capture(session, retain=False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    schedule=schedule, record_shapes=True) as profiler, capture.enter():
            for _ in range(args.profile_steps + 1):
                session.step()
                # Prevent warmup GPU work spilling into the active interval.
                torch.cuda.synchronize()
                profiler.step()
        result = {"status": "ok", **profile_summary(profiler, torch, args.profile_steps, args.profile_rows)}
        if args.trace:
            path = directory / (session.case["projection"] + ".trace.json")
            profiler.export_chrome_trace(str(path))
            result["trace"] = path.name
        return result
    finally:
        session.annotate = False


def saved_tensors(session):
    unique, calls = {}, []

    def pack(value):
        raw = getattr(value, "_data", value)
        storage = raw.untyped_storage()
        key = (str(raw.device), storage._cdata)
        unique[key] = {"dtype": str(raw.dtype), "bytes": storage.nbytes()}
        calls.append(str(raw.dtype))
        return value

    with session.torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        session.step(update=False)
    totals = {}
    for value in unique.values():
        totals[value["dtype"]] = totals.get(value["dtype"], 0) + value["bytes"]
    return {"save_calls": len(calls), "unique_storage_bytes": sum(totals.values()), "bytes_by_physical_dtype": totals,
            "note": "Observed saved-tensor hooks; shared storages counted once. This is not total peak memory."}


def host_profile(session, args):
    if not args.host_profile:
        return {"status": "disabled"}
    profiler = cProfile.Profile()
    session.torch.cuda.synchronize()
    with profiler:
        for _ in range(args.diagnostic_steps):
            session.step()
    session.torch.cuda.synchronize()
    rows = [{"function": f"{path}:{line}:{name}", "count": count,
             "self_ms_per_update": own * 1000 / args.diagnostic_steps,
             "nested_total_ms_per_update": total * 1000 / args.diagnostic_steps}
            for (path, line, name), (_, count, own, total, _) in pstats.Stats(profiler).stats.items()]
    return {"status": "ok", "updates": args.diagnostic_steps,
            "top_functions": sorted(rows, key=lambda row: row["self_ms_per_update"], reverse=True)[:args.profile_rows]}


def library_state(session):
    result = {"projection_class": type(session.projection).__module__ + "." + type(session.projection).__name__,
              "parameters": [{"shape": list(p.shape), "dtype": str(p.dtype)} for p in session.parameters]}
    states = []
    for name, module in session.projection.named_modules():
        if hasattr(module, "scales") and hasattr(module, "history") and hasattr(module, "updates"):
            cache = getattr(module, "_metadata_cache", None)
            states.append({"name": name, "scales": module.scales.detach().cpu().tolist(),
                           "updates": module.updates.detach().cpu().tolist(),
                           "history_shape": list(module.history.shape),
                           "metadata_cache": cache.statistics() if cache is not None and hasattr(cache, "statistics") else None})
    result["delayed_states"] = states
    if getattr(session, "settings", None) is not None:
        from deepopt.opt._recipe import fp8_recipe

        result["numerical_recipe"] = fp8_recipe(session.settings.backend, session.settings)
        from deepopt.opt.backends._gemm_launch import gemm_launch_plan

        result["gemm_layout_cache"] = gemm_launch_plan.cache_info()._asdict()
    backend = sys.modules.get("deepopt.opt.backends.triton_fp8")
    kernel = getattr(backend, "_gemm_nt_kernel", None)
    result["triton_gemm_winners"] = [{"key": [str(value) for value in key], "tile": value.kwargs,
                                        "warps": value.num_warps, "stages": value.num_stages}
                                       for key, value in getattr(kernel, "cache", {}).items()]
    return result


def run_case(session, args, directory, row=None, checkpoint=lambda: None):
    torch = session.torch
    if row is None:
        row = {}
    row.update({**session.case, "status": "running", "initial": session.initial,
                "dense_head_fallback": session.case["projection"] == "head", "configuration": session.configuration,
                "optimizer_updates": args.optimizer_updates, "accumulation_request": args.accumulation})
    checkpoint()
    torch.cuda.synchronize()
    started = perf_counter()
    session.step()
    torch.cuda.synchronize()
    row["first_update_seconds"] = perf_counter() - started
    print(f"  First update: {row['first_update_seconds']:.3f} s; warming and measuring", flush=True)
    for _ in range(args.warmups):
        session.step()
    row["gpu_before_timing"] = smi()
    row["timing"] = measure(session.step, args, torch)
    row["gpu_after_timing"] = smi()
    row["updates_after_timing"] = session.completed_updates
    row["weight_after_timing"] = fingerprint(session.source.weight, torch)
    row["weight_changed_after_timing"] = row["weight_after_timing"] != row["initial"]["weight"]
    print(f"  Update median: {row['timing']['wall_ms']['median']:.4f} ms; collecting diagnostics", flush=True)
    checkpoint()
    recorder = PhaseRecorder(torch)
    session.recorder = recorder
    try:
        for _ in range(args.diagnostic_steps):
            session.step()
        torch.cuda.synchronize()
        row["phases"] = recorder.summary()
    finally:
        session.recorder = None
    session.clear()
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    session.step()
    torch.cuda.synchronize()
    row["memory"] = {"baseline_allocated_bytes": baseline, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "extra_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    checkpoint()
    print("  Capturing actual GEMM operands and internal dtypes", flush=True)
    # Capture real buffers and internal output dtypes only after throughput/memory passes.
    capture = Capture(session)
    with capture.enter():
        output = session.step(update=False)
    # Scalar readback is outside both throughput and profiler passes. Never add
    # these synchronizations to Capture.record(), which is also used in profiles.
    for record, payload in capture.gemms:
        scales = payload[5]
        if scales[0] is not None:
            record["scale_a_values"] = scales[0].detach().cpu().reshape(-1).tolist()
            record["scale_b_values"] = scales[1].detach().cpu().reshape(-1).tolist()
    row["gemm_calls"] = [record for record, _ in capture.gemms]
    row["quantization_calls"] = capture.quantization
    row["three_gemms_observed"] = len(capture.gemms) == 3 and {r["role"] for r, _ in capture.gemms} == {
        "forward", "input_gradient", "weight_gradient"}
    row["numerical"] = numerical(session, output)
    del output
    checkpoint()
    try:
        print("  Replaying prepared GEMMs" if args.prepared_gemms else "  Prepared GEMM replay disabled", flush=True)
        row["prepared_gemms"] = prepared_gemms(capture, args) if args.prepared_gemms else []
    except Exception as error:
        row["prepared_gemms_error"] = {"error": repr(error), "traceback": traceback.format_exc()}
    del capture
    checkpoint()
    print("  Collecting saved tensors and separate CUDA/CPU profiles", flush=True)
    for key, function in (("saved_tensors", lambda: saved_tensors(session)),
                          ("profile", lambda: profile(session, args, directory)),
                          ("host_profile", lambda: host_profile(session, args))):
        try:
            row[key] = function()
        except Exception as error:
            row[key] = {"status": "error", "error": repr(error), "traceback": traceback.format_exc()}
        checkpoint()
    try:
        row["library_state"] = library_state(session)
    except Exception as error:
        row["library_state"] = {"status": "error", "error": repr(error)}
    row["status"] = "ok"
    return row


def worker(payload):
    args = argparse.Namespace(**payload["settings"])
    directory = Path(payload["directory"])
    report = {"report_version": REPORT_VERSION, "run_id": payload["run_id"], "status": "running", "cases": []}
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report["environment"] = env = environment(args)
    if payload.get("check"):
        needed = set()
        if any(name.startswith("deepopt") for name in args.implementations):
            needed.update(("deepopt", "triton"))
        if "torchao" in args.implementations:
            needed.add("torchao")
        errors = [name + ": " + env["packages"][name].get("error", "unavailable")
                  for name in sorted(needed) if env["packages"][name]["status"] != "ok"]
        if not env["cuda_available"] or env.get("capability", [0])[0] < 9 or env.get("hip"):
            errors.append("Requires a real NVIDIA SM90+ CUDA device")
        if args.torchao_version and "torchao" in needed and env["packages"]["torchao"].get("version") != args.torchao_version:
            errors.append("Installed TorchAO version differs from --torchao-version")
        report.update(status="unavailable" if errors else "ok", errors=errors)
        write_json(directory / "result.json", report)
        return
    implementation, policy, batch = payload["implementation"], payload["math_policy"], payload["batch"]
    report.update(implementation=implementation, math_policy=policy, batch=batch)
    required = ("deepopt", "triton") if implementation.startswith("deepopt") else ("torchao",) if implementation == "torchao" else ()
    missing = [name for name in required if env["packages"][name]["status"] != "ok"]
    mismatch = implementation == "torchao" and args.torchao_version and env["packages"]["torchao"].get("version") != args.torchao_version
    if not env["cuda_available"] or missing or mismatch or env.get("capability", [0])[0] < 9:
        report.update(status="unavailable", reason=f"Requires NVIDIA SM90+ CUDA and compatible packages; missing={missing}, version_mismatch={bool(mismatch)}")
        write_json(directory / "result.json", report)
        return
    import torch

    configure(args, torch)
    report["controls"] = {"eager": True, "producer_fusion": False, "deterministic": torch.are_deterministic_algorithms_enabled(),
                          "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                          "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "bias": args.bias,
                          "format": args.format, "optimizer_updates": args.optimizer_updates,
                          "deepopt_backward_backend": args.backward_backend,
                          "accumulation": args.accumulation, "workload": "forward_backward_adamw" if args.optimizer_updates else "forward_backward_fixed_weights"}
    for case in cases(args, batch):
        print(f"Running {implementation}/{policy}/b{batch}/{case['projection']}", flush=True)
        row = {**case, "status": "running"}
        report["cases"].append(row)
        checkpoint = lambda: write_json(directory / "result.json", report)
        checkpoint()
        try:
            session = Session(args, implementation, policy, case, torch)
            run_case(session, args, directory, row, checkpoint)
        except Exception as error:
            row.update(status="error", error=repr(error), traceback=traceback.format_exc())
        finally:
            if "session" in locals():
                del session
            gc.collect()
            torch.cuda.empty_cache()
        write_json(directory / "result.json", report)
    report["status"] = "ok" if all(row["status"] == "ok" for row in report["cases"]) else "partial"
    write_json(directory / "result.json", report)


def environment_signature(job):
    env = job.get("environment", {})
    return ([env.get(key) for key in ("torch", "cuda", "gpu", "capability")],
            {name: (value.get("version"), value.get("source", {}).get("sha256"))
             for name, value in env.get("packages", {}).items()})


def fp32_products_observed(row):
    products = {record["role"]: record["output"]["dtype"] for record in row.get("gemm_calls", [])}
    return (row.get("three_gemms_observed", False) and set(products) == {"forward", "input_gradient", "weight_gradient"}
            and all(dtype == "torch.float32" for dtype in products.values()))


def comparisons(results):
    groups = {}
    for result in results:
        for row in result.get("cases", []):
            if row["status"] == "ok":
                groups.setdefault((result["math_policy"], result["batch"], row["projection"]), {})[result["implementation"]] = (result, row)
    output = []
    for (policy, batch, projection), group in groups.items():
        baseline = group.get("bf16")
        for name, (job, row) in group.items():
            if name == "bf16":
                continue
            same = bool(baseline and baseline[1]["initial"] == row["initial"])
            finite = bool(baseline and baseline[1]["numerical"]["all_finite"] and row["numerical"]["all_finite"])
            controls = bool(baseline and baseline[0].get("controls") == job.get("controls"))
            same_environment = bool(baseline and environment_signature(baseline[0]) == environment_signature(job))
            comparable = same and finite and controls and same_environment
            peer = group.get("torchao")
            peer_valid = bool(peer and peer[1]["initial"] == row["initial"] and peer[1]["numerical"]["all_finite"]
                              and row["numerical"]["all_finite"] and peer[0].get("controls") == job.get("controls")
                              and environment_signature(peer[0]) == environment_signature(job))
            if policy == "matched-fp32" and projection != "head":
                peer_valid = peer_valid and fp32_products_observed(row) and fp32_products_observed(peer[1])
            fp8_dtypes = {record["role"]: record["output"]["dtype"] for record in row.get("gemm_calls", [])}
            output.append({"math_policy": policy, "batch": batch, "projection": projection, "implementation": name,
                           "same_initial_samples": same, "finite": finite, "same_controls": controls,
                           "same_environment": same_environment, "observed_gemm_output_dtypes": fp8_dtypes,
                           "matched_fp32_outputs_observed": fp32_products_observed(row)
                           if policy == "matched-fp32" and projection != "head" else None,
                           "speedup_vs_bf16": baseline[1]["timing"]["wall_ms"]["median"] / row["timing"]["wall_ms"]["median"] if comparable else None,
                           "speedup_vs_torchao": peer[1]["timing"]["wall_ms"]["median"] / row["timing"]["wall_ms"]["median"] if peer_valid and name != "torchao" else None,
                           "internal_dtype_contract": "inspect gemm_calls; AMP FP8 policies may differ" if policy == "amp" else "FP8 GEMM output dtypes controlled; verify observed calls"})
    return output


def summarize(results):
    weighted = []
    for result in results:
        rows = result.get("cases", [])
        good = [row for row in rows if row["status"] == "ok"]
        weighted.append({"implementation": result.get("implementation"), "math_policy": result.get("math_policy"),
                         "batch": result.get("batch"), "complete": result["status"] == "ok" and len(good) == len(rows) and bool(good),
                         "projection_count": len(good),
                         "sum_isolated_update_ms": sum(row["multiplicity"] * row["timing"]["wall_ms"]["median"] for row in good)})
    return {"comparisons": comparisons(results), "weighted_isolated_times": weighted}


def save_report(directory, report):
    report["summary"] = summarize(report["results"])
    write_json(directory / "summary.json", report)
    fields = ["implementation", "math_policy", "batch", "projection", "status", "wall_ms", "cuda_stream_ms", "peak_allocated_bytes"]
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in report["results"]:
            for row in job.get("cases", []):
                writer.writerow({"implementation": job["implementation"], "math_policy": job["math_policy"], "batch": job["batch"],
                                 "projection": row["projection"], "status": row["status"],
                                 "wall_ms": row.get("timing", {}).get("wall_ms", {}).get("median"),
                                 "cuda_stream_ms": row.get("timing", {}).get("cuda_stream_ms", {}).get("median"),
                                 "peak_allocated_bytes": row.get("memory", {}).get("peak_allocated_bytes")})


def bundle(directory, run_id):
    destination = Path(str(directory) + ".zip")
    paths = [directory / "summary.json", directory / "summary.csv", *(directory / run_id).rglob("*")]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return destination


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    settings = {key: value for key, value in vars(args).items() if key not in ("output", "_worker")}
    report = {"report_version": REPORT_VERSION, "run_id": run_id, "settings": settings,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "notes": NOTES, "results": []}
    save_report(directory, report)

    def launch(payload, destination):
        destination.mkdir(parents=True, exist_ok=True)
        payload.update(run_id=run_id, settings=settings, directory=str(destination))
        with (destination / "worker.log").open("w") as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", json.dumps(payload)],
                                       stdout=log, stderr=subprocess.STDOUT)
        result_path = destination / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "error", "cases": []}
        if result.get("run_id") != run_id:
            result = {"status": "error", "cases": [], "error": "Worker produced no result for this run"}
        result["worker_returncode"] = completed.returncode
        result["artifacts"] = str(destination.relative_to(directory))
        if completed.returncode:
            result["status"] = "error"
        return result

    report["preflight"] = launch({"check": True}, directory / run_id / "preflight")
    save_report(directory, report)
    for name, item in report["preflight"].get("environment", {}).get("packages", {}).items():
        if item["status"] != "ok":
            print(f"Preflight: {name} unavailable: {item.get('error')}", flush=True)
    if not args.check and report["preflight"].get("environment", {}).get("cuda_available"):
        try:
            for implementation in args.implementations:
                for policy in args.math_policies:
                    for batch in args.batches:
                        destination = directory / run_id / f"{implementation}-{policy}-b{batch}"
                        print(f"Starting {implementation} / {policy} / batch {batch}; log: {destination / 'worker.log'}", flush=True)
                        payload = {"implementation": implementation, "math_policy": policy, "batch": batch}
                        if args.cold_cache:
                            with tempfile.TemporaryDirectory(prefix="deepopt-linear-cache-") as cache, patch.dict(os.environ, TRITON_CACHE_DIR=cache):
                                result = launch(payload, destination)
                        else:
                            result = launch(payload, destination)
                        result.update(implementation=implementation, math_policy=policy, batch=batch)
                        report["results"].append(result)
                        save_report(directory, report)
                        print(f"  {result['status']}; details: {destination / 'worker.log'}", flush=True)
        finally:
            save_report(directory, report)
            print(f"Bundle: {bundle(directory, run_id)}", flush=True)
    else:
        print(json.dumps(report["preflight"], indent=2))
        print(f"Bundle: {bundle(directory, run_id)}")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output", type=Path, default=Path("fp8-linear-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")))
    result.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS))
    result.add_argument("--math-policies", nargs="+", choices=("amp", "matched-fp32"), default=["amp", "matched-fp32"])
    result.add_argument("--batches", nargs="+", type=int, default=[512])
    result.add_argument("--projections", nargs="+", choices=tuple(PROJECTIONS), default=list(PROJECTIONS))
    result.add_argument("--tokens", type=int, default=197)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--format", choices=("e4m3fn", "hybrid"), default="e4m3fn")
    result.add_argument("--accumulation", choices=("fast", "torchao-default"), default="fast")
    result.add_argument("--torchao-version", help="Optional exact installed-version check; no packages are installed by the benchmark.")
    result.add_argument("--num-stages", choices=("auto", "1", "2", "3", "4", "5"), default="3")
    result.add_argument("--backward-backend", choices=("inherit", "native_fast", "native_precise"), default="inherit",
                        help="DeepOpt dX/dW executor; forward stays Triton. Native policies change accumulation.")
    result.add_argument("--history-length", type=int, default=1024)
    result.add_argument("--amax-algorithm", choices=("max", "most_recent"), default="max")
    result.add_argument("--margin", type=int, default=0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=.01)
    result.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--optimizer-updates", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--fill-uninitialized-memory", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--warmups", type=int, default=10)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--diagnostic-steps", type=int, default=3)
    result.add_argument("--probe-warmups", type=int, default=3)
    result.add_argument("--probe-iterations", type=int, default=10)
    result.add_argument("--prepared-gemms", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--profile-steps", type=int, default=3)
    result.add_argument("--profile-rows", type=int, default=40)
    result.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--host-profile", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--cold-cache", action="store_true", help="Isolate Triton disk caches; default reuses the installed environment's cache.")
    result.add_argument("--check", action="store_true", help="Only inspect dependencies/CUDA; no benchmark kernels run.")
    result.add_argument("--_worker", help=argparse.SUPPRESS)
    return result


def main():
    command = parser()
    args = command.parse_args()
    if args._worker:
        payload = json.loads(args._worker)
        try:
            worker(payload)
        except Exception as error:
            path = Path(payload["directory"]) / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {"run_id": payload["run_id"], "cases": []}
            result.update(status="error", error=repr(error), traceback=traceback.format_exc())
            write_json(path, result)
            raise
        return
    if (any(value < 1 for value in (*args.batches, args.tokens, args.iterations, args.repeats,
                                    args.diagnostic_steps, args.probe_iterations, args.profile_rows))
            or min(args.warmups, args.probe_warmups, args.profile_steps) < 0
            or not 1 <= args.history_length <= 4096 or not -32 <= args.margin <= 32
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        command.error("Invalid shapes, measurement counts, recipe or optimizer settings")
    args.num_stages = args.num_stages if args.num_stages == "auto" else int(args.num_stages)
    report = run(args)
    if (not report["preflight"].get("environment", {}).get("cuda_available")
            or (args.check and report["preflight"]["status"] != "ok")
            or any(row["status"] != "ok" for row in report["results"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
