"""Save durable, tensor-free TorchInductor training references for full models.

Planning/help use only the standard library. Each model/recipe runs in its own
process and temporary compiler cache. CPU preflight cannot produce GPU evidence.
The compiled scope is forward/loss plus PyTorch AOTAutograd backward; the real
optimizer update is eager. This is never DeepOpt native-autodiff qualification.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("resnet18", "vit-base", "efficientnet-b0", "vae-v2")
RECIPES = ("torch-bf16",)
SCOPE = {"compiler": "torch.compile / TorchInductor", "backward": "PyTorch AOTAutograd",
         "requested_compiled": ["forward", "loss", "backward"], "optimizer": "eager diagnostic update",
         "deepopt_native_autodiff": False,
         "coverage": "full native-Torch architecture; opaque custom calls/breaks disqualify the optimized baseline"}
PARITY_POLICIES = {
    "fp32": {"rtol": 1e-4, "atol": 1e-7, "max_relative_l2": 1e-3, "min_cosine": .99999},
    "update": {"rtol": 1e-4, "atol": 1e-8, "max_relative_l2": 1e-3, "min_cosine": .99999},
    "bf16_output": {"rtol": .016, "atol": 1e-4, "max_relative_l2": .02, "min_cosine": .9999},
}
COMPILER_ENV_KEYS = ("TORCH_COMPILE_DEBUG", "TORCH_COMPILE_DEBUG_SAVE_REAL", "TORCH_COMPILE_DEBUG_DIR",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TRITON_DUMP_DIR", "TRITON_KERNEL_DUMP",
    "TORCHDYNAMO_REPRO_LEVEL", "TORCHDYNAMO_REPRO_AFTER", "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_CACHE", "TORCHINDUCTOR_SAVE_COMPILE_ARGS")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    paths = {"runner": Path(__file__), "models": ROOT / "benchmarks/compiler/inductor_reference_models.py",
             "store": ROOT / "benchmarks/compiler/inductor_reference_store.py",
             "inductor_context": ROOT / "benchmarks/compiler/inductor_benchmark_artifacts.py",
             "measurement": ROOT / "benchmarks/analysis/measurement.py",
             "original_vae": ROOT / "examples/vae/mbconv_vae.py"}
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def paired_timing_summary(eager, compiled):
    if len(eager) != len(compiled) or not eager or any(
            not math.isfinite(value) or value <= 0 for value in [*eager, *compiled]):
        raise ValueError("Paired runtime samples must be finite, positive and equal in count")
    ratios = [left / right for left, right in zip(eager, compiled)]
    return {"eager_median_ms": statistics.median(eager), "inductor_median_ms": statistics.median(compiled),
            "speedup_eager_median_over_inductor_median": statistics.median(eager) / statistics.median(compiled),
            "paired_speedup_median": statistics.median(ratios), "paired_speedups": ratios,
            "samples_per_strategy": len(eager), "scope": "native-Torch eager versus Inductor runtime diagnostic"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=ROOT / "compiler_references/inductor")
    result.add_argument("--reference-root", type=Path, help="Optional secondary corpus; primary code/timing stay in --output")
    result.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    result.add_argument("--recipes", nargs="+", choices=RECIPES, default=["torch-bf16"])
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--steps", type=int, default=50)
    result.add_argument("--timeout", type=float, default=3600)
    result.add_argument("--mode", choices=("default", "max-autotune", "reduce-overhead"), default="default")
    result.add_argument("--arithmetic-preset", choices=("default", "cast-preserving"), default="default")
    result.add_argument("--fullgraph", action="store_true")
    result.add_argument("--profile", action="store_true", help="Separate scalar profiler summary, outside timing")
    result.add_argument("--max-reference-bytes", type=int, default=4 * 2**30)
    result.add_argument("--session-id", default=os.environ.get("DEEPOPT_REFERENCE_SESSION_ID", "manual-cli"))
    result.add_argument("--preflight", action="store_true", help="Persist dependency/device availability; do not compile")
    result.add_argument("--dry-run", action="store_true", help="Print the complete plan without imports or writes")
    result.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return result


def make_plan(args):
    if args.recipes != ["torch-bf16"]:
        raise ValueError("Inductor references allow native Torch operations only: torch-bf16")
    for name in ("batch_size", "image_size", "steps", "max_reference_bytes"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Warmup must be nonnegative and timeout finite/positive")
    if args.image_size != 224:
        raise ValueError("Full-model references require image_size=224; tiny/probe substitution is forbidden")
    if not args.device.startswith("cuda"):
        raise ValueError("Full references require CUDA; use --preflight to record unavailable CUDA on a CPU host")
    if len(set(args.models)) != len(args.models) or len(set(args.recipes)) != len(args.recipes):
        raise ValueError("Duplicate models/recipes would obscure independent capture identity")
    common = {name: getattr(args, name) for name in (
        "device", "batch_size", "image_size", "seed", "warmup", "steps", "mode", "fullgraph",
        "profile", "max_reference_bytes", "session_id", "preflight", "arithmetic_preset")}
    return {"schema_version": 1, "output": str(args.output.absolute()),
            "reference_root": str(args.reference_root.absolute()) if args.reference_root else None,
            "timeout_seconds": args.timeout,
            "scope": SCOPE, "group_size": 1024, "fresh_process_per_model_recipe": True,
            "durable_not_cache": True, "tensor_files_permitted": False,
            "jobs": [{**common, "model": model, "recipe": recipe, "group_size": 1024}
                     for model in args.models for recipe in args.recipes]}


def worker_environment(scratch):
    scratch = Path(scratch)
    environment = dict(os.environ)
    # Disable inherited reproduction/trace settings that can serialize tensors.
    for key in ("TORCH_TRACE", "TORCH_LOGS", "TORCH_COMPILE_DEBUG_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR", "TRITON_DUMP_DIR", "TRITON_OVERRIDE_DIR", "TORCHDYNAMO_REPRO_AFTER"):
        environment.pop(key, None)
    environment.update(TORCH_COMPILE_DEBUG="1", TORCH_COMPILE_DEBUG_SAVE_REAL="0",
        TORCH_COMPILE_DEBUG_DIR=str(scratch / "debug"), TORCHINDUCTOR_CACHE_DIR=str(scratch / "inductor-cache"),
        TRITON_CACHE_DIR=str(scratch / "triton-cache"), TRITON_DUMP_DIR=str(scratch / "triton-dumps"),
        TRITON_KERNEL_DUMP="1", TORCHDYNAMO_REPRO_LEVEL="0",
        TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_AUTOGRAD_CACHE="0",
        TORCHINDUCTOR_SAVE_COMPILE_ARGS="0", PYTHONDONTWRITEBYTECODE="1")
    return environment


def disable_dynamo_reproduction():
    """Disable reproduction with Python None, never an empty env string.

    The worker environment removes REPRO_AFTER before Torch import. This also
    handles an already-imported Dynamo config in diagnostic/in-process use.
    """
    config = importlib.import_module("torch._dynamo.config")
    requested = {"repro_after": None, "repro_level": 0}
    for key, value in requested.items():
        setattr(config, key, value)
    applied = {key: getattr(config, key) for key in requested}
    if applied != requested:
        raise RuntimeError("Cannot disable TorchDynamo reproduction for tensor-free benchmark capture")
    return {"requested": requested, "applied": applied, "environment_policy": "TORCHDYNAMO_REPRO_AFTER unset"}


def configure_diagnostics(torch, scratch):
    dynamo = disable_dynamo_reproduction()
    config = importlib.import_module("torch._inductor.config")
    requested = {"trace.enabled": True, "trace.save_real_tensors": False,
        "trace.debug_dir": str(Path(scratch) / "debug"), "trace.fx_graph": True,
        "trace.fx_graph_transformed": True, "trace.ir_pre_fusion": True, "trace.ir_post_fusion": True,
        "trace.output_code": True, "trace.debug_log": True, "trace.info_log": True,
        "trace.log_autotuning_results": True, "trace.compile_profile": False,
        "fx_graph_cache": False, "fx_graph_remote_cache": False,
        "autotune_remote_cache": False, "fallback_random": True}
    applied, unavailable = {}, []
    for key, value in requested.items():
        owner = config
        try:
            for part in key.split(".")[:-1]:
                owner = getattr(owner, part)
            getattr(owner, key.split(".")[-1])
            setattr(owner, key.split(".")[-1], value)
            applied[key] = value
        except AttributeError:
            unavailable.append(key)
    if applied.get("trace.save_real_tensors") is not False:
        raise RuntimeError("This Torch version cannot confirm trace.save_real_tensors=False; capture refused")
    logging = getattr(getattr(torch, "_logging", None), "set_logs", None)
    logged = []
    if logging is not None:
        supported = inspect.signature(logging).parameters
        logged = [key for key in ("aot_graphs", "graph_code", "output_code", "kernel_code", "schedule",
                  "fusion", "guards", "graph_breaks", "recompiles", "autotuning", "ir_pre_fusion",
                  "ir_post_fusion") if key in supported]
        logging(**dict.fromkeys(logged, True))
    return {"requested": requested, "applied": applied, "unavailable": unavailable, "logging_flags": logged,
            "dynamo_reproduction": dynamo,
            "rng_policy": "fallback_random=True for matching-eager stochastic streams" if applied.get("fallback_random")
                else "fallback_random unavailable; stochastic parity must still pass or capture is partial",
            "real_tensor_dumping": False}


def runtime_preflight(job):
    missing = [name for name in ("torch", "torchvision" if job["model"] != "vae-v2" else "torch")
               if importlib.util.find_spec(name) is None]
    missing = sorted(set(missing))
    report = {"model": job["model"], "recipe": job["recipe"], "missing_dependencies": missing,
              "gpu_capture_pending": True, "cuda_available": False}
    if "torch" not in missing:
        torch = importlib.import_module("torch")
        report.update(torch_version=str(torch.__version__), torch_git=getattr(torch.version, "git_version", None),
                      cuda_runtime=torch.version.cuda,
                      cuda_available=bool(torch.cuda.is_available()))
        if report["cuda_available"]:
            device = torch.device(job["device"])
            with torch.cuda.device(device):
                report.update(device_name=torch.cuda.get_device_name(device),
                    bf16_supported=bool(torch.cuda.is_bf16_supported()),
                    capability=list(torch.cuda.get_device_capability(device)))
                properties = torch.cuda.get_device_properties(device)
                report["gpu_properties"] = {key: getattr(properties, key) for key in (
                    "name", "total_memory", "multi_processor_count", "major", "minor", "warp_size")
                    if hasattr(properties, key)}
                if hasattr(properties, "uuid"):
                    report["gpu_properties"]["uuid"] = str(properties.uuid)
    blockers = [f"Missing dependency: {name}" for name in missing]
    report["requires_installed_deepopt"] = job["model"] == "vae-v2"
    if report["requires_installed_deepopt"]:
        try:
            helper = importlib.import_module("benchmarks.compiler.installed_runtime")
            report["installed_deepopt"] = helper.installed_identity()
        except Exception as error:
            blockers.append(f"Installed non-editable DeepOpt required: {type(error).__name__}: {error}")
    if not report["cuda_available"]:
        blockers.append("CUDA unavailable; full-model GPU compiler references remain pending")
    elif not report.get("bf16_supported"):
        blockers.append("Target CUDA device lacks BF16 support")
    report["blockers"] = blockers
    return report


def tensor_leaves(value, prefix="output"):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_leaves(value[key], f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, f"{prefix}.{index}")


def scalar_leaves(value, prefix):
    if type(value) in (float, int, bool):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from scalar_leaves(value[key], f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from scalar_leaves(child, f"{prefix}.{index}")


class ReferenceBudget:
    def __init__(self, maximum):
        self.maximum, self.used = maximum, 0

    def copy(self, tensor):
        self.reserve(tensor)
        return tensor.detach().to("cpu", copy=True)

    def reserve(self, tensor):
        size = tensor.numel() * tensor.element_size()
        if self.used + size > self.maximum:
            raise ValueError("In-memory matching-eager reference exceeds --max-reference-bytes; no disk fallback")
        self.used += size


def snapshot(model, inputs, outputs, initial, optimizer, budget):
    import torch

    values = {name: budget.copy(value) for name, value in tensor_leaves(outputs)}
    for name, value in tensor_leaves(inputs, "input"):
        values[f"input_grad.{name}"] = budget.copy(value.grad) if value.grad is not None else None
    for name, value in model.named_parameters():
        values[f"parameter_grad.{name}"] = budget.copy(value.grad) if value.grad is not None else None
        values[f"parameter.{name}"] = budget.copy(value)
        budget.reserve(values[f"parameter.{name}"])
        values[f"parameter_update.{name}"] = values[f"parameter.{name}"] - initial[name]
    values.update({f"buffer.{name}": budget.copy(value) for name, value in model.named_buffers()})
    for name, parameter in model.named_parameters():
        for state_name, value in tensor_leaves(optimizer.state.get(parameter, {}), "state"):
            values[f"optimizer.{name}.{state_name}"] = budget.copy(value)
        for state_name, value in scalar_leaves(optimizer.state.get(parameter, {}), "state"):
            tensor = torch.tensor(value, dtype=torch.float64 if type(value) is float else
                                  torch.bool if type(value) is bool else torch.int64)
            values[f"optimizer_scalar.{name}.{state_name}"] = budget.copy(tensor)
    return values


def compare_snapshots(actual, expected):
    """Separate declared BF16 output rounding from strict FP32 training state."""
    import torch
    from analysis.measurement import errors

    tensors, passed, finite = {}, True, True
    missing = object()
    for key in sorted(set(actual) | set(expected)):
        left, right = actual.get(key, missing), expected.get(key, missing)
        if left is missing or right is missing or (left is None) != (right is None):
            tensors[key] = {"passed": False, "reason": "missing_or_None_mismatch"}
        elif left is None:
            tensors[key] = {"passed": True, "both_None": True}
        elif left.dtype != right.dtype:
            tensors[key] = {"passed": False, "reason": "dtype_mismatch", "actual_dtype": str(left.dtype),
                            "expected_dtype": str(right.dtype)}
        else:
            policy = ("update" if key.startswith("parameter_update.") else "bf16_output"
                      if key.startswith("output.") and left.dtype == torch.bfloat16 else "fp32")
            a, b = {key: left}, {key: right}
            extra = {}
            if policy == "update":
                parameter_key = "parameter." + key.removeprefix("parameter_update.")
                if parameter_key in actual and parameter_key in expected:
                    a[parameter_key], b[parameter_key] = actual[parameter_key], expected[parameter_key]
                    extra["update_policy"] = {"parameter_rounding_ulps": 2}
            result = errors(a, b, **PARITY_POLICIES[policy], **extra)
            tensors[key] = {**result["tensors"][key], "policy": policy}
            finite &= result["finite"]
        passed &= tensors[key]["passed"]
    return {"passed": passed and finite, "finite": finite, "policies": PARITY_POLICIES,
            "dtype_and_None_equality_required": True, "tensors": tensors}


def gradient_coverage(model):
    """Scalar coverage of original initialization, including a zero ViT head."""
    rows, blocks = {}, {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        nonzero = int(gradient.count_nonzero()) if gradient is not None else 0
        rows[name] = {"present": gradient is not None, "nonzero_elements": nonzero,
                      "elements": parameter.numel(), "nonzero": bool(nonzero)}
        block = name.rsplit(".", 1)[0]
        counts = blocks.setdefault(block, {"parameters": 0, "nonzero_parameters": 0})
        counts["parameters"] += 1
        counts["nonzero_parameters"] += bool(nonzero)
    return {"parameters": rows, "blocks": blocks,
            "nonzero_parameters": sum(row["nonzero"] for row in rows.values()),
            "total_parameters": len(rows)}


VIT_BODY_GRADIENT_PARAMETERS = ("network.conv_proj.weight", *(
    f"network.encoder.layers.encoder_layer_{index}.mlp.0.weight" for index in range(12)))


def vit_body_gradient_coverage(coverage):
    """Require positive matching second-update coverage through all ViT blocks.

    These fixed full-architecture anchors exclude the classification head.
    Recompute this receipt in the parent; a child success boolean is insufficient.
    """
    result = {"passed": False, "update": 2, "declared_parameters": list(VIT_BODY_GRADIENT_PARAMETERS),
              "scope": "patch projection and MLP input weight in each of the original 12 transformer blocks"}
    if not isinstance(coverage, list):
        return {**result, "reason": "missing_gradient_coverage"}
    updates = [row for row in coverage if isinstance(row, dict) and type(row.get("update")) is int and row["update"] == 2]
    if len(updates) != 1:
        return {**result, "reason": "missing_or_duplicate_second_update"}
    row, failures = updates[0], []
    for name in VIT_BODY_GRADIENT_PARAMETERS:
        entries = []
        for strategy in ("eager", "inductor"):
            side = row.get(strategy)
            parameters = side.get("parameters") if isinstance(side, dict) else None
            entry = parameters.get(name) if isinstance(parameters, dict) else None
            if (not isinstance(entry, dict) or entry.get("present") is not True or entry.get("nonzero") is not True
                    or type(entry.get("elements")) is not int or entry["elements"] < 1
                    or type(entry.get("nonzero_elements")) is not int
                    or not 0 < entry["nonzero_elements"] <= entry["elements"]):
                failures.append({"parameter": name, "strategy": strategy, "reason": "missing_or_zero_body_gradient"})
            else:
                entries.append(entry)
        if len(entries) == 2 and any(entries[0][key] != entries[1][key] for key in ("elements", "nonzero_elements")):
            failures.append({"parameter": name, "reason": "mismatched_body_gradient_coverage"})
    return {**result, "passed": not failures, "failures": failures}


def rng_fingerprint(state):
    """Hash RNG bytes in RAM; persist neither generator state nor tensors."""
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__name__.encode() + b"\0")
        if isinstance(value, dict):
            for key in sorted(value):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif hasattr(value, "detach"):
            visit(str(value.dtype))
            visit(tuple(value.shape))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif hasattr(value, "tobytes"):
            visit(str(value.dtype))
            visit(tuple(value.shape))
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode() + b"\0")

    visit(state)
    return digest.hexdigest()


def finite_state(model, inputs, outputs, optimizer):
    """Run after synchronization and outside every measured/profiled window."""
    import torch

    bad = []
    values = list(tensor_leaves(outputs)) + list(tensor_leaves(inputs, "input"))
    for name, value in model.named_parameters():
        values.append((f"parameter.{name}", value))
        if value.grad is not None:
            values.append((f"parameter_grad.{name}", value.grad))
        values.extend(tensor_leaves(optimizer.state.get(value, {}), f"optimizer.{name}"))
        bad.extend(key for key, scalar in scalar_leaves(optimizer.state.get(value, {}), f"optimizer.{name}")
                   if type(scalar) is float and not math.isfinite(scalar))
    values.extend((f"buffer.{name}", value) for name, value in model.named_buffers())
    for name, value in tensor_leaves(inputs, "input"):
        if value.grad is not None:
            values.append((f"input_grad.{name}", value.grad))
    for name, value in values:
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            bad.append(name)
    return {"finite": not bad, "nonfinite": bad, "outside_measurement": True,
            "checked": "outputs/loss, inputs, parameters, gradients, buffers, optimizer tensor state"}


def phase_evidence(scratch):
    """Retain paths proving generated forward/backward phases, not model labels."""
    phases = {"forward": [], "backward": []}
    for path in sorted(Path(scratch).rglob("*.py")):
        with path.open(errors="replace") as stream:
            text = stream.read(65536)
        for phase in phases:
            if (("output_code" in path.name and phase in str(path.relative_to(scratch)).lower())
                    or any("AOT" in line and phase in line.lower() for line in text.splitlines()[:30])):
                phases[phase].append(path.relative_to(scratch).as_posix())
    return phases


def native_graph_provenance(graph_module):
    """Inspect callable identities, without executing or saving tensor values."""
    records, violations = [], []
    for module_name, module in graph_module.named_modules():
        if not hasattr(module, "graph"):
            continue
        for node in module.graph.nodes:
            target = node.target
            name, owner = str(target), getattr(target, "__module__", "") or ""
            namespace = getattr(target, "namespace", None)
            native = node.op in ("placeholder", "output", "get_attr")
            if node.op == "call_function":
                native = (namespace in ("aten", "prims", "inductor", "higher_order") if namespace is not None
                          else owner == "torch" or owner.startswith("torch.")
                          or owner in ("operator", "_operator", "math", "builtins"))
            elif node.op == "call_method":
                import torch

                native = hasattr(torch.Tensor, name)
            elif node.op == "call_module":
                child = module.get_submodule(name)
                native = type(child).__module__.startswith("torch.nn.")
            if "deepopt" in owner.lower() or "deepopt" in name.lower() or any(
                    marker in name.lower() for marker in ("autograd_function_apply", "triton_kernel_wrapper")):
                native = False
            record = {"graph": module_name, "op": node.op, "target": name,
                      "owner": owner, "namespace": namespace, "native_torch": bool(native)}
            records.append(record)
            if not native:
                violations.append(record)
    return {"native_ops_only": not violations, "nodes": records, "opaque_custom_boundaries": violations}


def inductor_compile_options(torch, requested_mode, backend_kwargs):
    """Translate callable-backend mode kwargs to Inductor's options API."""
    kwargs = dict(backend_kwargs)
    mode = kwargs.pop("mode", None) or requested_mode
    if mode != requested_mode:
        raise ValueError("Callable backend mode differs from the requested benchmark mode")
    extra = kwargs.pop("options", None) or {}
    if kwargs or not isinstance(extra, dict):
        raise ValueError(f"Unsupported callable backend arguments: {sorted(kwargs)}")
    modes = getattr(torch._inductor, "list_mode_options", None)
    if modes is None and mode != "default":
        raise RuntimeError("Requested Inductor mode has no supported mode-options API")
    options = dict(modes(mode)) if modes else {}
    options.update(extra)
    return options


def execute_capture(job, scratch):
    probe = runtime_preflight(job)
    if probe["blockers"] or job["preflight"]:
        return _execute_capture(job, scratch, probe)
    from benchmarks.compiler.inductor_benchmark_artifacts import inductor_reference_context

    # Keep the assumption live through first backward, subsequent shape guards,
    # recompilation, warmup and the instrumented profiling update.
    with inductor_reference_context(job.get("arithmetic_preset", "default")) as controls:
        return _execute_capture(job, scratch, probe, controls)


def _execute_capture(job, scratch, probe, controls=None):
    """CUDA-only worker. All tensors remain in RAM and are released on exit."""
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None
    base = {"scope": SCOPE, "preflight": probe, "request": job, "executed_updates": 0,
            "gpu_capture_observed": False, "deepopt_native_autodiff": False,
            "performance_qualified": False,
            "model": job["model"], "recipe": job["recipe"], "device_type": "cuda",
            "python_version": platform.python_version(), "python_build": sys.version,
            "torch_version": probe.get("torch_version"), "torch_git": probe.get("torch_git"),
            "triton_version": triton_version, "torchvision_version": torchvision_version,
            "source_sha256s": source_identity(), "cuda_runtime": probe.get("cuda_runtime"),
            "gpu_properties": probe.get("gpu_properties"), "requested_device": job["device"],
            "gpu_name": probe.get("device_name"), "gpu_capability": probe.get("capability"),
            "gpu_uuid": (probe.get("gpu_properties") or {}).get("uuid"),
            "installed_deepopt": probe.get("installed_deepopt"),
            "numerical_controls": controls,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if probe["blockers"] or job["preflight"]:
        return {**base, "status": "blocked", "reason": "; ".join(probe["blockers"]) or "Preflight only; no compiler execution requested"}
    import torch

    device = torch.device(job["device"])
    torch.cuda.set_device(device)
    base["diagnostic_configuration"] = configure_diagnostics(torch, scratch)
    # Adapter owns full architectures, weights=None, original VAE sampling and
    # an explicit diagnostic optimizer. It never replaces a missing model.
    adapter_module = importlib.import_module("benchmarks.compiler.inductor_reference_models")
    adapter = adapter_module.build_reference_model(job["model"], recipe=job["recipe"], device=job["device"],
        batch_size=job["batch_size"], image_size=job["image_size"], seed=job["seed"])
    base["model_metadata"] = adapter["metadata"]
    model, inputs, optimizer = adapter["model"], adapter["inputs"], adapter["optimizer"]
    reference = deepcopy(model)
    reference_inputs = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in inputs)
    reference_optimizer = adapter["optimizer_factory"](reference.parameters())
    budget = ReferenceBudget(job["max_reference_bytes"])
    initial = {name: budget.copy(value) for name, value in model.named_parameters()}
    capture_rng, restore_rng = adapter["capture_rng"], adapter["restore_rng"]
    rng = capture_rng()

    def forward_loss(*values):
        outputs = model(*values)
        return outputs, adapter["loss_fn"](outputs)

    def eager_forward_loss(*values):
        outputs = reference(*values)
        return outputs, adapter["loss_fn"](outputs)

    def update(call, owned_inputs, owned_optimizer, *, validate=False):
        owned_optimizer.zero_grad(set_to_none=True)
        for value in owned_inputs:
            value.grad = None
        with adapter["execution_context"](), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, loss = call(*owned_inputs)
        if loss.ndim or (validate and not bool(torch.isfinite(loss))):
            raise ValueError("Training loss must be finite and scalar")
        loss.backward()
        owned_optimizer.step()
        return outputs

    with adapter_module.observe_stochastic_branches(reference) as stochastic_first:
        eager_output = update(eager_forward_loss, reference_inputs, reference_optimizer, validate=True)
    base["stochastic_branch_coverage"] = [{"update": 1, "eager_observations": stochastic_first}]
    eager_rng_state = capture_rng()
    eager_rng_post = rng_fingerprint(eager_rng_state)
    expected = snapshot(reference, reference_inputs, eager_output, initial, reference_optimizer, budget)
    restore_rng(rng)
    torch._dynamo.utils.counters.clear()
    native_graphs = []

    def native_torch_inductor_reference(graph, example_inputs, **backend_kwargs):
        provenance = native_graph_provenance(graph)
        options = inductor_compile_options(torch, job["mode"], backend_kwargs)
        provenance["backend_options"] = options
        number = len(native_graphs)
        native_graphs.append(provenance)
        (Path(scratch) / f"fx_graph_native_{number:03d}.py").write_text(graph.code)
        write_json(Path(scratch) / f"native-provenance-{number:03d}.json", provenance)
        if not provenance["native_ops_only"]:
            raise RuntimeError("Opaque/custom operation in native-Torch Inductor reference; eager fallback forbidden")
        return torch._inductor.compile(graph, example_inputs, options=options)

    # Mode is applied above, not passed as an unhandled callable-backend kwarg.
    compiled = torch.compile(forward_loss, backend=native_torch_inductor_reference, fullgraph=job["fullgraph"])
    torch.cuda.synchronize(device)
    began = perf_counter()
    output = update(compiled, inputs, optimizer, validate=True)
    torch.cuda.synchronize(device)
    compiled_rng_post = rng_fingerprint(capture_rng())
    base["cold_compile_and_first_update_seconds"] = perf_counter() - began
    base["executed_updates"] = 1
    breaks = {str(key): value for key, value in torch._dynamo.utils.counters.get("graph_break", {}).items()}
    native_only = bool(native_graphs) and all(graph["native_ops_only"] for graph in native_graphs) and not breaks
    base.update(gpu_capture_observed=True, compiled_phase_artifacts=phase_evidence(scratch),
        native_ops_provenance={"native_ops_only": native_only, "graphs": native_graphs, "graph_breaks": breaks,
                               "opaque_custom_calls_allowed": False}, optimized_baseline_eligible=False,
        execution_phases={"compiled": bool(native_graphs), "first_update_executed": True, "numerically_accepted": False,
                          "timed": False, "profiled": False})
    actual = snapshot(model, inputs, output, initial, optimizer, budget)
    comparison = compare_snapshots(actual, expected)
    rng_parity = {"passed": eager_rng_post == compiled_rng_post, "eager_post_sha256": eager_rng_post,
                  "compiled_post_sha256": compiled_rng_post, "scope": "CPU/CUDA/Python/NumPy generator post-state"}
    base.update(matching_eager=comparison, rng_post_state_parity=rng_parity, reference_bytes=budget.used)
    base["first_update_finite_checks"] = {"eager": finite_state(reference, reference_inputs, eager_output, reference_optimizer),
                                          "inductor": finite_state(model, inputs, output, optimizer)}
    base["final_state_finite"] = all(check["finite"] for check in base["first_update_finite_checks"].values())
    base["gradient_coverage"] = [{"update": 1, "eager": gradient_coverage(reference),
                                 "inductor": gradient_coverage(model)}]
    del initial, expected, actual, eager_output, output
    if not comparison["passed"] or not rng_parity["passed"]:
        failures = {name: value for name, value in comparison["tensors"].items() if not value["passed"]}
        worst = max(failures, key=lambda key: failures[key].get("relative_l2", 0.), default=None)
        base["numerical_failures"] = {"count": len(failures), "first_failing_tensor": next(iter(failures), None),
                                      "worst_relative_l2_tensor": worst,
                                      "worst_category": worst.split(".", 1)[0] if worst else None}
    first_accepted = comparison["passed"] and rng_parity["passed"]
    base["execution_phases"]["numerically_accepted"] = first_accepted
    # Both timed trajectories start at exactly the accepted eager post-update
    # state. Deepcopy prevents sharing mutable momentum between strategies.
    model.load_state_dict(reference.state_dict())
    optimizer.load_state_dict(deepcopy(reference_optimizer.state_dict()))
    restore_rng(eager_rng_state)
    del eager_rng_state, rng
    base["paired_initial_state"] = "matching eager post-first-update model/optimizer state; independent optimizer tensors; same RNG"
    pair_reports = []

    def pair(index, *, timed, numerical=False):
        shared_rng = capture_rng()
        if numerical:
            comparison_budget = ReferenceBudget(job["max_reference_bytes"])
            before = {name: comparison_budget.copy(value) for name, value in reference.named_parameters()}
        durations, post_states, checks = {}, {}, []
        order = ("eager", "inductor") if index % 2 == 0 else ("inductor", "eager")
        outputs = {}
        for label in order:
            restore_rng(shared_rng)
            call, owned_inputs, owned_optimizer, owned_model = ((compiled, inputs, optimizer, model)
                if label == "inductor" else (eager_forward_loss, reference_inputs, reference_optimizer, reference))
            torch.cuda.synchronize(device)
            began = perf_counter()
            observer = (adapter_module.observe_stochastic_branches(owned_model)
                        if numerical and label == "eager" else nullcontext())
            with observer as observations:
                outputs[label] = update(call, owned_inputs, owned_optimizer)
            if numerical and label == "eager":
                base["stochastic_branch_coverage"].append({"update": 2, "eager_observations": observations})
            torch.cuda.synchronize(device)
            durations[label] = (perf_counter() - began) * 1000
            post_states[label] = rng_fingerprint(capture_rng())
            checks.append({"strategy": label, **finite_state(owned_model, owned_inputs, outputs[label], owned_optimizer)})
        report = {"order": list(order), "rng_post_state_equal": post_states["eager"] == post_states["inductor"],
                  "finite": all(check["finite"] for check in checks), "checks": checks}
        if timed:
            report["wall_ms"] = durations
        if numerical:
            expected = snapshot(reference, reference_inputs, outputs["eager"], before,
                                reference_optimizer, comparison_budget)
            actual = snapshot(model, inputs, outputs["inductor"], before, optimizer, comparison_budget)
            report["matching_eager"] = compare_snapshots(actual, expected)
            report["reference_bytes"] = comparison_budget.used
        return outputs["inductor"], report

    # Preserve original zero initialization; a separate untimed second update
    # exercises body gradients after the ViT classification head becomes nonzero.
    _, later = pair(0, timed=False, numerical=True)
    base["executed_updates"] += 1
    base["later_update_numerics"] = [{"update": 2, **later}]
    base["gradient_coverage"].append({"update": 2, "eager": gradient_coverage(reference),
                                      "inductor": gradient_coverage(model)})
    later_accepted = later["matching_eager"]["passed"] and later["finite"] and later["rng_post_state_equal"]
    if job["model"] == "vit-base":
        base["vit_body_gradient_coverage"] = vit_body_gradient_coverage(base["gradient_coverage"])
        later_accepted &= base["vit_body_gradient_coverage"]["passed"]
    base["execution_phases"]["numerically_accepted"] = first_accepted and later_accepted
    if not first_accepted or not later_accepted:
        base["source_sha256s_after"] = source_identity()
        base["benchmark_source_unchanged"] = base["source_sha256s"] == base["source_sha256s_after"]
        return {**base, "status": "partial", "reason": "First/later-update matching-recipe gradients/updates/state rejected",
                "later_update_scope": "untimed diagnostic from matching eager post-first-update state; does not erase first rejection"}
    base["paired_initial_state"] = "independent matching eager post-first-update state followed by accepted paired second update; same RNG"
    for index in range(job["warmup"]):
        _, report = pair(index, timed=False)
        base["executed_updates"] += 1
        if not report["finite"] or not report["rng_post_state_equal"]:
            return {**base, "status": "partial", "reason": "Paired warmup finite/RNG parity rejected", "warmup_failure": report}
    torch.cuda.synchronize(device)
    durations, eager_durations = [], []
    finite_checks = []
    for _ in range(job["steps"]):
        last_output, report = pair(len(durations), timed=True)
        durations.append(report["wall_ms"]["inductor"])
        eager_durations.append(report["wall_ms"]["eager"])
        pair_reports.append(report)
        base["executed_updates"] += 1
        check = {"finite": report["finite"] and report["rng_post_state_equal"], "outside_measurement": True,
                 "strategy_checks": report["checks"], "rng_post_state_equal": report["rng_post_state_equal"]}
        finite_checks.append({"phase": "timed", "step": len(durations) - 1, **check})
        if not check["finite"]:
            break
    base["timing"] = {"wall_ms": durations, "eager_wall_ms": eager_durations, "pairs": pair_reports,
                      "paired_speedup_eager_over_inductor": [left / right for left, right in zip(eager_durations, durations)],
                      "warmup_updates_per_strategy": job["warmup"], "cold_compile_in_samples": False,
                      "profiler_in_samples": False, "scope": "paired full-update diagnostic; not performance-policy qualification"}
    base["timing"].update(paired_timing_summary(eager_durations, durations))
    base["eager_executed_updates"] = 2 + job["warmup"] + len(durations)
    base["executed_updates_scope"] = "Inductor strategy, including its separate profiler update when requested"
    if job["profile"]:
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False,
                profile_memory=True) as profile:
            last_output = update(compiled, inputs, optimizer)
            torch.cuda.synchronize(device)
        base["executed_updates"] += 1
        trace = Path(scratch) / "profiler-chrome-trace.json"
        profile.export_chrome_trace(str(trace))
        base["profile"] = {"trace": trace.name, "profiled_updates": 1, "outside_timing": True,
            "scope": "instrumented CPU/CUDA scheduling chronology; no speedup qualification",
            "tensor_values_recorded": False, "paired_models_resident": True,
            "memory_scope": "both paired models resident; extra peak is above that measured baseline",
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "events": [{"name": event.key, "count": event.count,
                        "self_cpu_time_us": event.self_cpu_time_total,
                        "self_cuda_time_us": getattr(event, "self_device_time_total", None)}
                       for event in profile.key_averages()]}
        cuda_events = [{"name": event.name, "duration_us": event.time_range.elapsed_us()}
                       for event in profile.events() if str(event.device_type).endswith("CUDA")]
        base["profile"].update(cuda_activity_observed=bool(cuda_events), cuda_events=cuda_events,
            cuda_event_total_us=sum(event["duration_us"] for event in cuda_events),
            cuda_time_scope="sum of instrumented CUDA events; overlap means this is not elapsed wall time")
        finite_checks.append({"phase": "profile", **finite_state(model, inputs, last_output, optimizer)})
    evidence = phase_evidence(scratch)
    finite = all(check["finite"] for check in finite_checks)
    breaks = {str(key): value for key, value in torch._dynamo.utils.counters.get("graph_break", {}).items()}
    native_only = bool(native_graphs) and all(graph["native_ops_only"] for graph in native_graphs) and not breaks
    provenance = {"native_ops_only": native_only, "graphs": native_graphs, "graph_breaks": breaks,
                  "opaque_custom_calls_allowed": False}
    profile_valid = not job["profile"] or base["profile"]["cuda_activity_observed"]
    base.update(status="complete" if all(evidence.values()) and finite and native_only and profile_valid else "partial", gpu_capture_observed=True,
        native_ops_provenance=provenance,
        optimized_baseline_eligible=all(evidence.values()) and finite and native_only and profile_valid,
        final_state_finite=finite,
        finite_checks=finite_checks,
        compiled_phase_artifacts=evidence,
        dynamo_counters={str(key): dict(value) for key, value in torch._dynamo.utils.counters.items()})
    base["execution_phases"].update(timed=len(durations) == job["steps"], profiled=bool(job["profile"] and profile_valid))
    if not all(evidence.values()):
        base["reason"] = "Missing generated forward/backward phase evidence; partial references retained"
    if not finite:
        base["reason"] = "Nonfinite state/output after actual training updates; partial references retained"
    if not native_only:
        base["reason"] = "Custom/opaque operations, visible graph breaks or absent FX provenance disqualify native optimized baseline"
    if not profile_valid:
        base["reason"] = "Requested CUDA profiler activity missing; chronology remains partial"
    base["source_sha256s_after"] = source_identity()
    base["benchmark_source_unchanged"] = base["source_sha256s"] == base["source_sha256s_after"]
    if not base["benchmark_source_unchanged"]:
        base.update(status="partial", optimized_baseline_eligible=False, reason="Benchmark/adapter/store/VAE source changed during execution")
    write_json(Path(scratch) / "benchmark-results.json", base)
    return base


def worker_main(config):
    payload = json.loads(Path(config).read_text())
    try:
        result = execute_capture(payload["job"], payload["scratch"])
    except Exception as error:
        traceback.print_exc()
        result = {"status": "partial", "reason": f"{type(error).__name__}: {error}",
                  "scope": SCOPE, "gpu_capture_observed": False, "request": payload["job"]}
    write_json(payload["result"], result)
    return 0 if result["status"] in ("complete", "blocked") else 1


def copied_phase_evidence(run, phases, copied, namespace):
    inventory = {entry["path"]: entry for entry in copied.get("copied", [])}
    result = {"passed": True, "forward": [], "backward": [], "missing": []}
    for phase in ("forward", "backward"):
        paths = phases.get(phase) if isinstance(phases, dict) else None
        if not isinstance(paths, list) or not paths:
            result["passed"] = False
            result["missing"].append(f"{phase}: no phase paths")
            continue
        for relative in paths:
            if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                    or "\\" in relative):
                result["passed"] = False
                result["missing"].append(f"{phase}: invalid phase path")
                continue
            name = f"artifacts/{namespace}/{relative}"
            entry, path = inventory.get(name), Path(run) / name
            if (entry is None or not path.is_file() or path.is_symlink()
                    or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]):
                result["passed"] = False
                result["missing"].append(name)
            else:
                result[phase].append({"path": name, "sha256": entry["sha256"], "bytes": entry["bytes"]})
    return result


def read_worker_result(path, job, code):
    from benchmarks.compiler.inductor_benchmark_artifacts import reference_controls_valid

    if not path.exists():
        return {"reason": "Worker produced no result", "scope": SCOPE}, "partial"
    if path.stat().st_size > 16 * 2**20:
        raise ValueError("Worker result exceeds scalar metadata bound")
    outcome = json.loads(path.read_text())
    if not isinstance(outcome, dict) or outcome.get("request") != job:
        raise ValueError("Worker result identity differs from requested model/recipe")
    status = outcome.get("status", "partial")
    if status not in ("complete", "partial", "blocked", "failed"):
        raise ValueError("Invalid worker status")
    if code != 0 and status in ("complete", "blocked"):
        outcome["reason"] = "Nonzero/timeout worker exit invalidates reported success"
        status = "partial"
    matching, phases, updates = (outcome.get(key) for key in
                                 ("matching_eager", "compiled_phase_artifacts", "executed_updates"))
    rng = outcome.get("rng_post_state_parity")
    checks = outcome.get("finite_checks")
    native = outcome.get("native_ops_provenance")
    graphs = native.get("graphs") if isinstance(native, dict) else None
    later = outcome.get("later_update_numerics")
    if status == "complete" and job["model"] == "vit-base":
        body_coverage = vit_body_gradient_coverage(outcome.get("gradient_coverage"))
        if not body_coverage["passed"] or outcome.get("vit_body_gradient_coverage") != body_coverage:
            status = "partial"
            outcome["reason"] = "Missing/zero/mismatched second-update ViT body gradients"
    if status == "complete" and (outcome.get("gpu_capture_observed") is not True
            or not reference_controls_valid(outcome.get("numerical_controls"), job.get("arithmetic_preset", "default"))
            or not isinstance(later, list) or not later
            or not all(isinstance(report, dict) and report.get("update") == 2
                       and isinstance(report.get("matching_eager"), dict)
                       and report["matching_eager"].get("passed") is True
                       and report.get("finite") is True and report.get("rng_post_state_equal") is True for report in later)
            or not isinstance(matching, dict) or matching.get("passed") is not True
            or matching.get("dtype_and_None_equality_required") is not True
            or not isinstance(rng, dict) or rng.get("passed") is not True
            or outcome.get("final_state_finite") is not True
            or not isinstance(native, dict) or native.get("native_ops_only") is not True
            or not isinstance(native.get("graph_breaks"), dict) or native["graph_breaks"]
            or not isinstance(graphs, list) or not graphs
            or not all(isinstance(graph, dict) and graph.get("native_ops_only") is True
                       and isinstance(graph.get("nodes"), list) and graph["nodes"]
                       and graph.get("opaque_custom_boundaries") == [] for graph in graphs)
            or outcome.get("optimized_baseline_eligible") is not True
            or not isinstance(checks, list) or not checks
            or not all(isinstance(check, dict) and check.get("finite") is True
                       and check.get("outside_measurement") is True for check in checks)
            or sum(check.get("phase") == "timed" for check in checks) != job["steps"]
            or not isinstance(phases, dict) or not all(phases.get(phase) for phase in ("forward", "backward"))
            or type(updates) is not int or updates < 2 + job["warmup"] + job["steps"]):
        status = "partial"
        outcome["reason"] = "Missing GPU/update/dtype/RNG/finite/parity/forward-backward evidence"
    if status == "complete" and job["profile"]:
        profile = outcome.get("profile")
        if (not isinstance(profile, dict) or profile.get("outside_timing") is not True
                or profile.get("cuda_activity_observed") is not True
                or profile.get("trace") != "profiler-chrome-trace.json"
                or not (path.parent / "profiler-chrome-trace.json").is_file()
                or not any(check.get("phase") == "profile" for check in checks)):
            status = "partial"
            outcome["reason"] = "Requested profiler chronology/finite-state evidence is missing"
    if status == "complete":
        timing = outcome.get("timing")
        try:
            if not isinstance(timing, dict) or len(timing["wall_ms"]) != job["steps"]:
                raise ValueError("Missing paired runtime measurements")
            timing.update(paired_timing_summary(timing["eager_wall_ms"], timing["wall_ms"]))
        except (KeyError, TypeError, ValueError) as error:
            status = "partial"
            outcome["reason"] = f"Invalid paired eager/Inductor runtime baseline: {error}"
    return outcome, status


def run_plan(plan, *, worker_command=None):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    store = importlib.import_module("benchmarks.compiler.inductor_reference_store")
    results, interrupted = [], False
    for job in plan["jobs"]:
        request = {**job, "scope": SCOPE, "source_sha256s": source_identity(),
                   "requested_at_utc": datetime.now(timezone.utc).isoformat()}
        run = store.create_run(plan["output"], request)
        scratch = Path(tempfile.mkdtemp(prefix="deepopt-inductor-reference-")).resolve()
        child, code, copied = None, None, False
        status = "partial"
        outcome = {"scope": SCOPE, "request": job, "gpu_capture_observed": False,
                   "model": job["model"], "recipe": job["recipe"], "reason": "Worker did not complete"}
        try:
            result_path, config = scratch / "worker-result.json", scratch / "worker-request.json"
            write_json(config, {"job": job, "scratch": str(scratch), "result": str(result_path)})
            command = ([*worker_command, str(config)] if worker_command is not None else
                       [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(config)])
            with (scratch / "worker.stdout.log").open("wb") as stdout, (scratch / "worker.stderr.log").open("wb") as stderr:
                child = subprocess.Popen(command, env=worker_environment(scratch), stdout=stdout, stderr=stderr,
                                         start_new_session=True)
                code = child.wait(timeout=plan["timeout_seconds"])
            outcome, status = read_worker_result(result_path, job, code)
        except KeyboardInterrupt:
            interrupted = True
            outcome["reason"] = "Interrupted; partial compiler artifacts retained before cleanup"
            (scratch / "runner-error.log").write_text(traceback.format_exc())
        except Exception as error:
            outcome["reason"] = f"{type(error).__name__}: {error}; partial compiler artifacts retained"
            (scratch / "runner-error.log").write_text(traceback.format_exc())
        finally:
            # Stop every compiler descendant before the store freezes bytes.
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                    status = "partial"
                    outcome["reason"] = "Worker/descendants stopped before partial artifact capture"
                except ProcessLookupError:
                    pass
                child.wait()
            outcome["worker_exit_code"] = code
            outcome["interrupted"] = interrupted
            environment = worker_environment(scratch)
            outcome["requested_compiler_environment"] = {key: environment.get(key) for key in COMPILER_ENV_KEYS}
            copy_errors = []
            for namespace in ("worker", "recovery"):
                try:
                    outcome["artifact_copy_report"] = store.save_artifacts(run, scratch, namespace=namespace)
                    if outcome["artifact_copy_report"].get("errors") or outcome["artifact_copy_report"].get("status") == "partial":
                        status = "partial"
                        outcome["reason"] = "Compiler artifact copy was incomplete; inspect artifact_copy_report"
                    outcome["copied_phase_artifacts"] = copied_phase_evidence(run,
                        outcome.get("compiled_phase_artifacts"), outcome["artifact_copy_report"], namespace)
                    if status == "complete" and not outcome["copied_phase_artifacts"]["passed"]:
                        status = "partial"
                        outcome["reason"] = "Claimed forward/backward phase artifacts were not actually copied and hash-verified"
                    copied = True
                    break
                except Exception as error:
                    status = "partial"
                    copy_errors.append(f"{type(error).__name__}: {error}")
            if copy_errors:
                outcome["artifact_copy_exceptions"] = copy_errors
            if not copied:
                # A broken destination must not destroy the only failure dumps.
                # This emergency scratch is explicitly NOT an eligible reference.
                outcome.update(uncopied_scratch_retained=str(scratch), raw_scratch_is_reference=False)
            outcome["worker_reported_status"] = outcome.get("status")
            outcome["status"] = status
            if status != "complete":
                outcome["optimized_baseline_eligible"] = False
            write_json(run / "benchmark-results.json", outcome)
            try:
                try:
                    manifest = store.finalize_run(run, outcome, status, required_artifacts=("generated_code",))
                except Exception as error:
                    outcome["finalization_exception"] = f"{type(error).__name__}: {error}"
                    if (run / "manifest.json").exists():
                        # The manifest may have committed before an index error;
                        # finalized artifacts are immutable, so do not rewrite it.
                        manifest = json.loads((run / "manifest.json").read_text())
                    else:
                        manifest = store.finalize_run(run, outcome, "partial", required_artifacts=("generated_code",))
            finally:
                if copied:
                    shutil.rmtree(scratch)
            results.append({"run_dir": str(run), "manifest": manifest})
        reference_root = plan.get("reference_root")
        if reference_root and Path(reference_root).resolve() != Path(plan["output"]).resolve():
            secondary = None
            try:
                secondary = store.create_run(reference_root, {**request, "primary_benchmark_run": str(run),
                    "primary_manifest_sha256": sha256_file(run / "manifest.json")})
                archive_report = store.save_artifacts(secondary, run / "artifacts", namespace="benchmark")
                phases = {phase: [f"{namespace}/{name}" for name in outcome.get("compiled_phase_artifacts", {}).get(phase, [])]
                          for phase in ("forward", "backward")}
                secondary_phases = copied_phase_evidence(secondary, phases, archive_report, "benchmark")
                secondary_status = manifest["status"]
                if (archive_report.get("errors") or archive_report.get("status") == "partial"
                        or (secondary_status == "complete" and not secondary_phases["passed"])):
                    secondary_status = "partial"
                metadata = {**outcome, "primary_benchmark_run": str(run), "artifact_copy_report": archive_report,
                            "copied_phase_artifacts": secondary_phases, "status": secondary_status,
                            "optimized_baseline_eligible": secondary_status == "complete"
                            and outcome.get("optimized_baseline_eligible") is True}
                secondary_manifest = store.finalize_run(secondary, metadata, secondary_status,
                    required_artifacts=("generated_code",))
                results[-1]["secondary_reference"] = {"run_dir": str(secondary), "status": secondary_manifest["status"]}
            except Exception as error:
                results[-1]["secondary_reference_error"] = f"{type(error).__name__}: {error}"
                if secondary is not None and not (secondary / "manifest.json").exists():
                    store.finalize_run(secondary, {"primary_benchmark_run": str(run),
                        "reason": results[-1]["secondary_reference_error"]}, "partial")
        if interrupted:
            break
    store.refresh_index(plan["output"])
    return {"output": plan["output"], "scope": SCOPE, "runs": results, "interrupted": interrupted}


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.worker is not None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if str(ROOT / "benchmarks") not in sys.path:
            sys.path.insert(0, str(ROOT / "benchmarks"))
        return worker_main(args.worker)
    try:
        plan = make_plan(args)
    except ValueError as error:
        arguments.error(str(error))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    report = run_plan(plan)
    print(json.dumps({"output": report["output"], "scope": SCOPE,
        "runs": [{"run_dir": row["run_dir"], "status": row["manifest"]["status"],
                  "reference_eligible": row["manifest"].get("reference_eligible", False),
                  "timing": {key: row["manifest"].get("metadata", {}).get("timing", {}).get(key) for key in
                             ("eager_median_ms", "inductor_median_ms", "paired_speedup_median")},
                  "benchmark_results": str(Path(row["run_dir"]) / "benchmark-results.json"),
                  "reason": row["manifest"].get("metadata", {}).get("reason")}
                 for row in report["runs"]]}, indent=2))
    return 130 if report["interrupted"] else 0 if all(
        row["manifest"].get("status") == "complete" for row in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
