"""Compare explicit custom-kernel policies in isolated benchmark processes.

Run with the intended DeepOpt build installed. --help needs only Python's
standard library. CPU runs exercise portability and report reference timings.
Each worker retains its report and logs, including failed and timed-out cases.
"""

import argparse
import ast
from contextlib import contextmanager
import importlib
from importlib import metadata
import importlib.util
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import tempfile
import threading
from time import perf_counter
from types import SimpleNamespace


WORKLOADS = {
    "linear": ("fp8/benchmark_fp8_eager.py", True,
               ["--implementations", "deepopt", "--shapes", "32x64x96"],
               ["--implementations", "deepopt", "--shapes", "4096x768x3072", "4096x3072x768"]),
    "activation": ("activation/benchmark_activations.py", False,
                   ["--shape", "3x33", "--activations", "silu"],
                   ["--shape", "4096x768"]),
    "layer_norm": ("normalization/benchmark_normalization.py", False,
                   ["--op", "layer_norm", "--shape", "3x33", "--pipeline", "none"],
                   ["--op", "layer_norm", "--shape", "4096x768", "--pipeline", "none"]),
    "batch_norm": ("normalization/benchmark_normalization.py", True,
                   ["--op", "batch_norm", "--shape", "2x8x4x4", "--pipeline", "none"],
                   ["--op", "batch_norm", "--shape", "8x64x56x56", "--pipeline", "none"]),
    "group_norm": ("normalization/benchmark_normalization.py", True,
                   ["--op", "group_norm", "--shape", "2x8x4x4", "--groups", "4", "--pipeline", "none"],
                   ["--op", "group_norm", "--shape", "8x64x56x56", "--groups", "32", "--pipeline", "none"]),
    "pooling": ("pooling/benchmark_pooling.py", False,
                ["--op", "max_pool2d", "--shape", "2x3x6x6"],
                ["--op", "max_pool2d", "--shape", "8x32x32x32"]),
    "convolution": ("convolution/benchmark_convolution.py", True,
                    ["--shape", "2x8x8x8", "--out-channels", "16"],
                    ["--shape", "8x64x32x32", "--out-channels", "128"]),
    "pack": ("activation_compression/benchmark_activation_pack.py", True,
             ["--shapes", "37x65", "--source-dtypes", "bf16", "--group-sizes", "32",
              "--variants", "serial", "--groups-per-program", "1", "--num-warps", "4", "--execution", "eager"],
             ["--shapes", "4096x768", "--source-dtypes", "bf16", "--group-sizes", "512", "1024",
              "--variants", "serial", "--groups-per-program", "1", "--num-warps", "4", "--execution", "eager"]),
    "unpack": ("activation_compression/benchmark_activation_unpack.py", True,
               ["--shapes", "37x65", "--implementations", "grouped", "--group-sizes", "32", "--execution", "eager"],
               ["--shapes", "4096x768", "--implementations", "grouped", "--group-sizes", "512", "1024", "--execution", "eager"]),
    "compiler": ("compiler/benchmark_compiler.py", False,
                 ["--model", "mlp", "--batch-size", "2", "--fp8", "--compression"],
                 ["--model", "vit-block", "--width", "768", "--heads", "12", "--tokens", "197",
                  "--batch-size", "8", "--fp8", "--compression", "--precision", "bf16"]),
}

IDENTITY_FIELDS = ("implementation", "shape", "shape_m_k_n", "mode", "activation", "input", "output",
                   "compression", "activation_compression", "storage", "dtype", "source_dtype", "bits", "group", "group_size",
                   "path", "pipeline", "recenter", "backend", "transpose", "normalized_shape",
                   "groups_per_program", "num_warps", "tile", "warps")
TIMING_KEYS = {"wall_ms_median", "cuda_stream_ms_median", "cold_step_ms"}
MEMORY_KEYS = {"peak_allocated_bytes", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
               "peak_reserved_bytes", "peak_extra_allocated_bytes", "saved_tensor_logical_bytes",
               "saved_storage_bytes", "saved_state_bytes"}
SAMPLING_FIELDS = ("requested_warmup", "requested_iterations", "requested_repeats", "actual_warmup",
                   "actual_iterations_per_repeat", "actual_repeats", "time_budget_seconds", "budget_exhausted",
                   "elapsed_seconds")
PACK_HELPER_API = {"codec_module", "codec_choices", "codec_schedule", "backend_sources", "measure", "timings",
                   "write_checkpoint", "reusable_row", "shape"}
QUALIFIED_WORKLOADS = {"linear", "layer_norm", "pooling"}
QUALIFIED_OPTIONS = {"qualify_tilelang", "inputs", "outputs", "compression", "shape",
                     "linear_shapes", "linear_modes", "max_worker_runs"}


def json_object(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("use a JSON object") from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("use a JSON object")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parent)
    result.add_argument("--preset", choices=("smoke", "representative"), default="smoke")
    result.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS))
    result.add_argument("--backends", nargs="+", choices=("triton", "tilelang", "mixed"),
                        default=["triton", "tilelang", "mixed"])
    result.add_argument("--mixed-policy", type=json_object,
                        default={"preferred": "tilelang", "families": {"gemm": "triton"}})
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--iterations", type=int, default=10)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--trials", type=int, default=1, help="Independent processes per workload/backend")
    result.add_argument("--time-budget", type=float, default=2.0,
                        help="Seconds per timed measurement; requested counts are caps (0 disables)")
    result.add_argument("--exhaustive", action="store_true", help="Disable measurement budgets; retain all requested counts")
    result.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable Linear/compiler autotuning; disabled for ordinary comparisons")
    result.add_argument("--timeout", type=float, default=600, help="Seconds per worker process, including compilation")
    result.add_argument("--heartbeat-interval", type=float, default=15.0,
                        help="Seconds between elapsed-time/latest-log progress messages (0 disables)")
    result.add_argument("--seed", type=int, default=42, help="Deterministic rotation of backend launch order")
    result.add_argument("--dry-run", action="store_true", help="Write the resolved matrix without launching workers")
    result.add_argument("--resume", action="store_true", help="Reuse verified successful workers; retry failed or incomplete workers")
    result.add_argument("--output", type=Path, default=Path("work/tilelang-benchmarks"), help="New run directory")
    result.add_argument("--qualify-tilelang", action="store_true",
                        help="Opt in to independent numerical checks and focused controls for Linear/LayerNorm/pooling")
    result.add_argument("--inputs", nargs="+", choices=("bf16", "fp8"), help="Qualification: normalization/pooling input cases")
    result.add_argument("--outputs", nargs="+", choices=("bf16", "fp8"), help="Qualification: normalization/pooling output cases")
    result.add_argument("--compression", choices=("off", "on", "both"), help="Qualification: timed saved-state cases")
    result.add_argument("--shape", help="Qualification: override LayerNorm/pooling shape, for example 4096x768")
    result.add_argument("--linear-shapes", nargs="+", help="Qualification: override Linear MxKxN shapes")
    result.add_argument("--linear-modes", nargs="+", choices=("forward", "forward-backward"))
    result.add_argument("--max-worker-runs", type=int, help="Qualification: reject a larger experiment matrix before GPU admission")
    return result


def settings_snapshot(args):
    """New opt-in switches do not change legacy report settings."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            if args.qualify_tilelang or key not in QUALIFIED_OPTIONS}


def _replace_option(command, flag, values):
    command = list(command)
    if flag in command:
        start = command.index(flag)
        end = start + 1
        while end < len(command) and not command[end].startswith("--"):
            end += 1
        del command[start:end]
    return [*command, flag, *values]


def matrix(args):
    from deepopt._benchmarking import BenchmarkBackend

    cpu = args.device == "cpu"
    selected = list(dict.fromkeys(args.workloads or [name for name, item in WORKLOADS.items() if not cpu or not item[1]]))
    if args.qualify_tilelang:
        if not args.workloads or set(selected) - QUALIFIED_WORKLOADS:
            raise ValueError("--qualify-tilelang requires explicit --workloads from linear layer_norm pooling; other families remain unqualified")
        if args.max_worker_runs is not None and args.max_worker_runs < 1:
            raise ValueError("--max-worker-runs must be positive")
        if args.max_worker_runs is not None and len(selected) * len(set(args.backends)) * args.trials > args.max_worker_runs:
            raise ValueError("Selected matrix exceeds --max-worker-runs; select fewer cases or schedule a separately budgeted batch")
        if any(getattr(args, key) is not None for key in ("inputs", "outputs", "compression", "shape")) and not set(selected) & {"layer_norm", "pooling"}:
            raise ValueError("--inputs/--outputs/--compression/--shape require LayerNorm or pooling")
        if (args.linear_shapes or args.linear_modes) and "linear" not in selected:
            raise ValueError("--linear-shapes/--linear-modes require the linear workload")
    elif any(getattr(args, key) is not None for key in QUALIFIED_OPTIONS - {"qualify_tilelang"}):
        raise ValueError("Focused benchmark controls require --qualify-tilelang; legacy commands are unchanged")
    cases = []
    for name in selected:
        relative, gpu_only, smoke, representative = WORKLOADS[name]
        source = args.benchmark_root / relative
        if not source.is_file():
            source = args.benchmark_root / Path(relative).name
        if not source.is_file():
            raise ValueError(f"Missing benchmark script: {relative}")
        order = list(dict.fromkeys(args.backends))
        random.Random(f"{args.seed}:{name}").shuffle(order)
        for trial in range(args.trials):
            rotated = order[trial % len(order):] + order[:trial % len(order)]
            for mode in rotated:
                selection = BenchmarkBackend.from_args(SimpleNamespace(
                    deepopt_backend=mode, deepopt_kernel_policy=args.mixed_policy if mode == "mixed" else {}))
                identity = f"{name}-{mode}-{trial:02d}"
                output = args.output / (identity + ".json")
                command = [sys.executable, str(source.resolve()), *(smoke if args.preset == "smoke" else representative),
                           "--device", args.device, "--warmup", str(args.warmup),
                           "--steps" if name == "compiler" else "--iterations", str(args.iterations),
                           "--deepopt-backend", mode, "--output", str(output.resolve()),
                           "--time-budget", str(0.0 if args.exhaustive else args.time_budget)]
                if name != "compiler":
                    command += ["--repeats", str(args.repeats)]
                if name not in ("pack", "unpack"):
                    command.append("--autotune" if args.autotune else "--no-autotune")
                if mode == "mixed":
                    command += ["--deepopt-kernel-policy", json.dumps(selection.policy.as_dict(), sort_keys=True)]
                if args.qualify_tilelang:
                    command.append("--qualify-tilelang")
                    if name == "linear":
                        if args.linear_shapes:
                            command = _replace_option(command, "--shapes", args.linear_shapes)
                        if args.linear_modes:
                            command = _replace_option(command, "--modes", args.linear_modes)
                    else:
                        for flag in ("inputs", "outputs"):
                            if getattr(args, flag):
                                command += ["--" + flag, *getattr(args, flag)]
                        if args.compression:
                            command += ["--compression", args.compression]
                        if args.shape:
                            command = _replace_option(command, "--shape", [args.shape])
                cases.append({"id": identity, "workload": name, "backend": mode, "trial": trial,
                              "policy": selection.policy.as_dict(), "command": command,
                              "device": args.device, "benchmark_root": str(args.benchmark_root.resolve()),
                              "script_sha256": sha256(source.read_bytes()).hexdigest(),
                              "report": str(output.resolve()), "gpu_only": gpu_only})
                if args.qualify_tilelang:
                    cases[-1]["qualification"] = {"schema_version": 1, "policy": "tilelang-independent-v1"}
    return cases


def _script_contract(script):
    """Inspect literal argparse declarations and selected library helper imports."""
    source = ast.parse(Path(script).read_bytes(), filename=str(script))
    boolean_actions = {"BooleanOptionalAction"}
    for node in ast.walk(source):
        if isinstance(node, ast.ImportFrom) and node.module == "argparse":
            boolean_actions.update(alias.asname or alias.name for alias in node.names if alias.name == "BooleanOptionalAction")
    flags, helpers = set(), set()
    for node in ast.walk(source):
        if isinstance(node, ast.ImportFrom) and node.module == "deepopt._benchmarking":
            helpers.update(alias.name for alias in node.names if alias.name != "*")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        declared = {argument.value for argument in node.args if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str) and argument.value.startswith("--")}
        flags.update(declared)
        action = next((keyword.value for keyword in node.keywords if keyword.arg == "action"), None)
        if (isinstance(action, ast.Attribute) and action.attr == "BooleanOptionalAction"
                or isinstance(action, ast.Name) and action.id in boolean_actions):
            flags.update("--no-" + flag[2:] for flag in declared)
    definitions = {node.name for node in source.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return flags, helpers, definitions


def cli_capabilities(script):
    """Inspect supported options without importing or executing a worker."""
    return _script_contract(script)[0]


def preflight_cli(cases):
    """Reject a mixed-generation bundle before GPU discovery or any launch."""
    required = {}
    required_functions = {}
    for case in cases:
        command = case["command"]
        script = Path(command[1]).resolve()
        required.setdefault(script, set()).update(argument.split("=", 1)[0] for argument in command[2:]
                                                  if argument.startswith("--") and argument != "--")
        if case.get("workload") == "pack":
            root = Path(case.get("benchmark_root", script.parent))
            helper = root / WORKLOADS["unpack"][0]
            if not helper.is_file():
                helper = root / Path(WORKLOADS["unpack"][0]).name
            helper = helper.resolve()
            required.setdefault(helper, set())
            required_functions.setdefault(helper, set()).update(PACK_HELPER_API)
    failures = []
    available_helpers = vars(importlib.import_module("deepopt._benchmarking"))
    if any(case.get("qualification") for case in cases):
        try:
            qualification = importlib.import_module("deepopt._tilelang_benchmarking")
            required_api = {"tensor_check", "validation_result", "validation_valid", "qualify_linear",
                            "qualify_layer_norm", "qualify_pooling", "qualified_timing_plan"}
            missing_api = required_api - vars(qualification).keys()
            if qualification.QUALIFICATION_SCHEMA_VERSION != 1 or missing_api:
                failures.append("  selected DeepOpt build has incompatible TileLang qualification helpers")
        except (ImportError, AttributeError) as error:
            failures.append(f"  selected DeepOpt build is missing TileLang qualification helpers ({error})")
    for script, forwarded in sorted(required.items()):
        try:
            capabilities, helpers, definitions = _script_contract(script)
            missing = forwarded - capabilities
        except (OSError, SyntaxError, ValueError) as error:
            failures.append(f"  {script}: cannot inspect CLI declarations ({error})")
            continue
        issues = []
        if missing:
            issues.append(f"missing forwarded flags {', '.join(sorted(missing))}")
        missing_helpers = helpers - available_helpers.keys()
        if missing_helpers:
            issues.append("selected DeepOpt build is missing deepopt._benchmarking helpers "
                          + ", ".join(sorted(missing_helpers)))
        missing_functions = required_functions.get(script, set()) - definitions
        if missing_functions:
            issues.append("selected packing dependency is missing shared functions " + ", ".join(sorted(missing_functions)))
        if issues:
            failures.append(f"  {script}: " + "; ".join(issues))
    if failures:
        raise ValueError("Benchmark CLI preflight failed; stale or incomplete benchmark scripts:\n"
                         + "\n".join(failures)
                         + "\nUpdate/re-export the complete benchmark bundle and install its matching current DeepOpt build. "
                           "Copying only the runner is insufficient. No workers were started.")


def invalid_report(value):
    """Detect explicit worker failures; finite diagnostics alone are not parity."""
    if isinstance(value, dict):
        if value.get("status") in ("error", "partial", "unsupported", "timeout"):
            return True
        if isinstance(value.get("error"), str) and value["error"]:
            return True
        if any(value.get(key) is False for key in ("finite", "exact", "numerical_parity")):
            return True
        return any(invalid_report(item) for item in value.values())
    if isinstance(value, list):
        return any(invalid_report(item) for item in value)
    return isinstance(value, float) and not math.isfinite(value)


def _option_values(command, option, default=()):
    if option not in command:
        return list(default)
    start = command.index(option) + 1
    end = start
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    return command[start:end]


def invalid_qualified_report(report, workload, case=None):
    """Fail closed on required numerical evidence, only in the opt-in mode."""
    if not isinstance(report, dict) or invalid_report(report):
        return True
    found = []
    settings = report.get("settings", {})
    if not isinstance(settings, dict) or type(settings.get("bias", False)) is not bool:
        return True
    bias = settings.get("bias", False)

    def visit(value, custom=False):
        if isinstance(value, dict):
            if "implementation" in value:
                custom = value["implementation"] == "deepopt"
            if custom and "timing" in value:
                found.append(value)
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item, custom)
        elif isinstance(value, list):
            for item in value:
                visit(item, custom)

    visit(report)
    if not found:
        return True
    identities = []
    for row in found:
        if workload == "linear":
            shape = row.get("shape_m_k_n")
            if (row.get("mode") not in ("forward", "forward-backward") or not isinstance(shape, list)
                    or len(shape) != 3 or any(type(size) is not int or size < 1 for size in shape)):
                return True
            identities.append((tuple(shape), row["mode"]))
        else:
            if (row.get("input") not in ("bf16", "fp8") or row.get("output") not in ("bf16", "fp8")
                    or type(row.get("compression")) is not bool):
                return True
            identities.append((row["input"], row["output"], row["compression"]))
        validation = row.get("validation")
        if (row.get("output_finite") is False or not isinstance(validation, dict)
                or validation.get("schema_version") != 1 or validation.get("status") != "passed"
                or not isinstance(validation.get("oracle"), str) or not validation["oracle"]):
            return True
        required = {"output"}
        if workload != "linear" or row.get("mode") == "forward-backward":
            required.add("dx")
        if workload == "layer_norm" or workload == "linear" and row.get("mode") == "forward-backward":
            required.add("dweight")
            if workload == "layer_norm" or bias:
                required.add("dbias")
        declared = validation.get("required_checks", [])
        if not isinstance(declared, list) or not all(isinstance(name, str) for name in declared):
            return True
        required.update(declared)
        checks = validation.get("checks")
        if not isinstance(checks, dict) or not required <= checks.keys():
            return True
        for check in checks.values():
            if not isinstance(check, dict) or check.get("status") != "passed" or check.get("finite") is not True:
                return True
            if not isinstance(check.get("metric"), str) or not check["metric"]:
                return True
            metric, limit = check.get("value"), check.get("limit")
            if (type(metric) not in (int, float) or type(limit) not in (int, float)
                    or not math.isfinite(metric) or not math.isfinite(limit) or metric < 0 or limit < 0 or metric > limit):
                return True
    if len(set(identities)) != len(identities):
        return True
    if case is not None:
        command = case["command"]
        if workload == "linear":
            try:
                shapes = [tuple(int(size) for size in shape.split("x")) for shape in _option_values(command, "--shapes")]
            except ValueError:
                return True
            modes = _option_values(command, "--modes", ("forward", "forward-backward"))
            expected = {(shape, mode) for shape in shapes for mode in modes}
        else:
            inputs = _option_values(command, "--inputs", ("bf16", "fp8"))
            outputs = _option_values(command, "--outputs", ("bf16", "fp8"))
            compression = _option_values(command, "--compression", ("both",))[0]
            choices = (False, True) if compression == "both" else (compression == "on",)
            expected = {(source, target, compressed) for source in inputs for target in outputs for compressed in choices}
            try:
                expected_shape = [int(size) for size in _option_values(command, "--shape")[0].split("x")]
            except (ValueError, IndexError):
                return True
            if not isinstance(report.get("configuration"), dict) or report["configuration"].get("shape") != expected_shape:
                return True
        if set(identities) != expected:
            return True
    return False


def measurements(report):
    """Extract comparable semantic rows, retaining eager/graph and phase names."""
    found = {}

    def visit(value, path=()):
        if isinstance(value, dict):
            implementation = value.get("implementation", "")
            if implementation.startswith(("native", "bf16", "torchao", "te")) or value.get("backend") == "eager":
                return
            for key, item in value.items():
                if key in ("environment", "configuration", "settings", "kernel_backend", "identity", "notes"):
                    continue
                if key in TIMING_KEYS | MEMORY_KEYS or key == "median" and any(part in ("timing", "timings") for part in path):
                    if isinstance(item, (float, int)) and not isinstance(item, bool) and math.isfinite(item) and item >= 0:
                        metric = "/".join((*path, key))
                        if metric in found:
                            raise ValueError(f"Ambiguous repeated measurement identity: {metric}")
                        found[metric] = item
                elif isinstance(item, (dict, list)):
                    visit(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                identity = {key: item[key] for key in IDENTITY_FIELDS if key in item} if isinstance(item, dict) else {}
                label = json.dumps(identity, sort_keys=True, separators=(",", ":")) if identity else str(index)
                visit(item, (*path, label))

    visit(report)
    return found


def sampling_records(report):
    """Retain actual work counts alongside per-step averages from each worker."""
    found = {}

    def visit(value, path=()):
        if isinstance(value, dict):
            if isinstance(value.get("sampling"), dict):
                found["/".join(path)] = {key: value["sampling"][key] for key in SAMPLING_FIELDS if key in value["sampling"]}
            for key, item in value.items():
                if key not in ("sampling", "configuration", "settings", "environment", "kernel_backend"):
                    visit(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                identity = {key: item[key] for key in IDENTITY_FIELDS if key in item} if isinstance(item, dict) else {}
                label = json.dumps(identity, sort_keys=True, separators=(",", ":")) if identity else str(index)
                visit(item, (*path, label))

    visit(report)
    return found


def sampling_summary(records):
    repeats = [row["actual_repeats"] for row in records.values() if isinstance(row.get("actual_repeats"), int)]
    iterations = [count for row in records.values() for count in row.get("actual_iterations_per_repeat", [])]
    return {"measurements": len(records),
            "budget_exhausted_measurements": sum(row.get("budget_exhausted") is True for row in records.values()),
            "minimum_actual_repeats": min(repeats) if repeats else None,
            "minimum_iterations_per_repeat": min(iterations) if iterations else None,
            "confidence_note": "Per-step averages may use different actual counts; inspect sampling before interpreting small differences."}


def _fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_fingerprint(root, *, exclude=()):
    """Hash source/native implementation files, excluding generated caches/logs."""
    relevant = {".py", ".json", ".toml", ".h", ".hpp", ".cuh", ".c", ".cc", ".cpp", ".cu",
                ".so", ".dylib", ".dll", ".pyd", ".bc", ".ptx", ".cubin"}
    root = Path(root).resolve()
    excluded = tuple(Path(path).resolve() for path in exclude)
    digest, count = sha256(), 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or not any(suffix in relevant for suffix in path.suffixes):
            continue
        if any(path == directory or directory in path.parents for directory in excluded):
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def run_provenance(args):
    """Read build/dependency/device identities without initializing Torch/CUDA."""
    spec = importlib.util.find_spec("deepopt")
    if spec is None or not spec.origin:
        raise ValueError("Cannot fingerprint the DeepOpt build selected by this interpreter")
    library_root = Path(spec.origin).resolve().parent
    packages = {}
    for name in ("deep-optimization", "torch", "triton", "tilelang", "apache-tvm-ffi", "numpy"):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
            continue
        # RECORD identifies wheel contents without rereading multi-GB Torch
        # libraries. Editable/source DeepOpt is fingerprinted independently.
        manifest = {key: distribution.read_text(key) for key in ("METADATA", "WHEEL", "RECORD", "direct_url.json")}
        packages[name] = {"version": distribution.version, "manifest_sha256": _fingerprint(manifest)}
    environment = {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
        "CUBLAS_WORKSPACE_CONFIG", "TRITON_DEFAULT_FP_FUSION", "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUDA_LAUNCH_BLOCKING", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    device = {"requested": args.device, "host": platform.node(), "processor": platform.processor()}
    reusable = True
    if args.device != "cpu":
        try:
            if args.dry_run:
                raise ValueError("GPU identity is not probed by dry-run")
            scheduler = _lease_scheduler(args.benchmark_root)
            selected = scheduler.discover_devices(None, args.device)[0]
            driver = scheduler.smi([f"--id={selected.uuid}", "--query-gpu=driver_version"])
            if not driver or not driver[0]:
                raise ValueError("GPU driver identity is unavailable")
            device.update(uuid=selected.uuid, index=selected.index, name=selected.name,
                          total_memory_bytes=selected.total, driver=driver[0][0].strip())
        except (OSError, ValueError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
            device.update(verification="unavailable", reason=str(error))
            reusable = False
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "deepopt": {"path": str(library_root), **_source_fingerprint(library_root)},
            "benchmark_sources": _source_fingerprint(args.benchmark_root, exclude=(args.output,)), "packages": packages,
            "environment": environment, "device": device, "resume_eligible": reusable}


def planned_work(args, cases):
    """A qualified work estimate, separate from compilation/setup time."""
    calls = {"linear": 2 if args.preset == "smoke" else 4, "layer_norm": 10, "pooling": 10}
    if args.qualify_tilelang:
        calls["linear"] = len(args.linear_shapes or ([None] if args.preset == "smoke" else [None, None])) * len(args.linear_modes or ("forward", "forward-backward"))
        inputs, outputs = len(args.inputs or ("bf16", "fp8")), len(args.outputs or ("bf16", "fp8"))
        compression_cases = 2 if args.compression in (None, "both") else 1
        calls.update(layer_norm=inputs * (1 + outputs * compression_cases),
                     pooling=inputs * (1 + outputs * compression_cases))
    known = sum(calls.get(case["workload"], 0) for case in cases)
    unknown = sorted({case["workload"] for case in cases if case["workload"] not in calls})
    budget = 0.0 if args.exhaustive else args.time_budget
    return {"workers": len(cases), "trials_per_backend_workload": args.trials,
            "requested_warmup_per_measurement": args.warmup, "requested_iterations_per_repeat": args.iterations,
            "requested_repeats_per_measurement": args.repeats,
            "requested_steps_per_measurement_cap": args.warmup + args.iterations * args.repeats,
            "time_budget_seconds_per_measurement": budget, "estimated_measurement_calls": known,
            "estimated_measurement_budget_seconds": known * budget if budget else None,
            "measurement_estimate_complete": not unknown, "unestimated_workloads": unknown,
            "worker_timeout_envelope_seconds": len(cases) * args.timeout,
            "estimate_note": "Measurement calls include native controls and all default input/output/compression cases. "
                             "The budget target excludes compilation, setup and diagnostics; a single slow step can exceed it. "
                             "The timeout envelope excludes coordinator overhead and process cleanup."}


def reusable_worker(case, checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("status") != "ok" or checkpoint.get("worker_identity") != case.get("worker_identity"):
        return None
    try:
        payload = Path(case["report"]).read_bytes()
        if sha256(payload).hexdigest() != checkpoint.get("report_sha256"):
            return None
        report = json.loads(payload)
        if (invalid_qualified_report(report, case["workload"], case) if case.get("qualification") else invalid_report(report)) or not measurements(report):
            return None
    except (OSError, TypeError, ValueError):
        return None
    return {**checkpoint, **case, "reused": True, "measurements": measurements(report),
            "sampling": sampling_records(report), "sampling_summary": sampling_summary(sampling_records(report))}


@contextmanager
def output_lock(directory):
    import fcntl

    with (directory / ".coordinator.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another benchmark coordinator owns this output directory") from None
        yield


def worker_log_tail(case):
    """Read at most 4 KiB without loading potentially large compiler logs."""
    prefix = Path(case["report"]).with_suffix("")
    suffix = ".stderr.log" if case.get("device", "cpu") == "cpu" else ".combined.log"
    try:
        with prefix.with_suffix(suffix).open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 4096))
            lines = log.read(4096).decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            text = "".join(character if character.isprintable() else " " for character in line).strip()
            if text:
                return text[:180]
    except OSError:
        pass  # No log exists during startup/admission, or the worker moved it.
    return None


@contextmanager
def worker_heartbeat(case, interval):
    """UI-only supervision; stop and join the helper on every exit path."""
    if interval == 0:
        yield
        return
    stopped = threading.Event()
    started = perf_counter()

    def heartbeat():
        while not stopped.wait(interval):
            message = f"[{case['id']}] still running: {perf_counter() - started:.1f}s elapsed"
            latest = worker_log_tail(case)
            if latest:
                message += f"; latest log: {latest}"
            print(message, flush=True)

    thread = threading.Thread(target=heartbeat, name="deepopt-worker-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _lease_scheduler(benchmark_root):
    """Load the exported stdlib-only scheduler without a global path change."""
    root = Path(benchmark_root).resolve()
    directory = root / "analysis"
    initializer = directory / "__init__.py"
    if not initializer.is_file():
        raise ValueError("GPU benchmark exports must include the analysis/ GPU lease helpers")
    name = "_deepopt_kernel_benchmark_analysis_" + sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, initializer, submodule_search_locations=[str(directory)])
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load GPU lease helpers from {directory}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return importlib.import_module(name + ".scheduler")


def _logical_cuda_command(command):
    """The broker binds a physical UUID, exposed as logical cuda:0 in the child."""
    return ["cuda:0" if index and command[index - 1] == "--device" else
            "--device=cuda:0" if value.startswith("--device=") else value
            for index, value in enumerate(command)]


def _execute_gpu(case, timeout, log_prefix):
    """Own the outer GPU lease so child scripts never detach a second worker."""
    result = {"combined_log": str(log_prefix.with_suffix(".combined.log")),
              "stdout": None, "stderr": None, "log_format": "combined_stdout_stderr"}
    scheduler = _lease_scheduler(case["benchmark_root"])
    device = scheduler.discover_devices(None, case["device"])[0]
    command = _logical_cuda_command(case["command"])
    result["executed_command"] = command
    result["device_binding"] = {"requested": case["device"], "physical_uuid": device.uuid,
                                "physical_index": device.index, "logical_device": "cuda:0"}
    broker_path = Path(os.environ.get("DEEPOPT_BENCHMARK_LEASE_DB") or scheduler.default_broker_path())
    inherited = scheduler.inherited_lease()
    if inherited:
        with scheduler.Broker(broker_path).connect() as connection:
            leased = connection.execute("SELECT gpu FROM leases WHERE token=?",
                                        (os.environ["DEEPOPT_BENCHMARK_LEASE"],)).fetchone()
        if leased != (device.uuid,):
            raise ValueError("Inherited GPU lease does not cover the requested physical device")
    result["gpu_lease"] = "inherited" if inherited else "owned"
    with tempfile.TemporaryDirectory(prefix="deepopt-kernel-benchmark-timeout-") as temporary:
        cancel = Path(temporary) / "cancel"

        def expired():
            try:
                cancel.write_text("worker deadline exceeded\n")
            except FileNotFoundError:
                pass  # The completed invocation may already have closed its scope.

        timer = threading.Timer(timeout, expired)
        timer.daemon = True
        timer.start()
        try:
            if inherited:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": device.uuid,
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
                with Path(result["combined_log"]).open("w") as log:
                    result["returncode"] = scheduler.run_command(
                        command, env=env, log=log, timeout=timeout, cancel_file=cancel)
            else:
                result["returncode"] = scheduler.run_external(
                    command, device=device, broker_path=broker_path, timeout=timeout,
                    cancel=cancel, log_path=Path(result["combined_log"]),
                )
        except (TimeoutError, InterruptedError):
            result.update(status="timeout", error=f"Worker exceeded {timeout:g} seconds including GPU admission")
        finally:
            timer.cancel()
            timer.join(timeout=1)
    return result


def execute(case, timeout):
    started = perf_counter()
    log_prefix = Path(case["report"]).with_suffix("")
    result = {**case, "status": "error", "measurements": {}}
    if case.get("device", "cpu") != "cpu":
        try:
            result.update(_execute_gpu(case, timeout, log_prefix))
        except (OSError, ValueError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
            result["error"] = str(error)
    else:
        # CPU workers share no GPU lease bootstrap. File-backed logs avoid
        # retaining unbounded compiler diagnostics in the coordinator.
        result.update(stdout=str(log_prefix.with_suffix(".stdout.log")),
                      stderr=str(log_prefix.with_suffix(".stderr.log")),
                      log_format="separate_stdout_stderr", executed_command=list(case["command"]))
        with Path(result["stdout"]).open("w") as stdout, Path(result["stderr"]).open("w") as stderr:
            try:
                with subprocess.Popen(case["command"], stdout=stdout, stderr=stderr, start_new_session=os.name == "posix") as process:
                    try:
                        result["returncode"] = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        # CPU workers may spawn helpers in their owned group.
                        if os.name == "posix":
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        process.wait()
                        result.update(status="timeout", error=f"Worker exceeded {timeout:g} seconds")
            except OSError as error:
                result["error"] = str(error)
    result["process_wall_seconds"] = perf_counter() - started
    if result.get("returncode") == 0:
        try:
            payload = Path(case["report"]).read_bytes()
            report = json.loads(payload)
            result["measurements"] = measurements(report)
            result["sampling"] = sampling_records(report)
            result["sampling_summary"] = sampling_summary(result["sampling"])
            invalid = invalid_qualified_report(report, case["workload"], case) if case.get("qualification") else invalid_report(report)
            if invalid:
                result["error"] = "Worker report contains failures, nonfinite values or failed self-checks"
                if case.get("qualification"):
                    result["measurements"] = {}
            elif not result["measurements"]:
                result["error"] = "Worker report contains no recognized measurements"
            else:
                result["status"] = "ok"
                result["report_sha256"] = sha256(payload).hexdigest()
        except (OSError, ValueError, TypeError) as error:
            result["error"] = f"Invalid worker report: {error}"
    elif "error" not in result:
        result["error"] = "Worker exited unsuccessfully; inspect the retained logs"
    return result


def comparisons(results):
    baselines = {(row["workload"], row["trial"]): row for row in results
                 if row["backend"] == "triton" and row["status"] == "ok"}
    rows = []
    for candidate in results:
        baseline = baselines.get((candidate["workload"], candidate["trial"]))
        if candidate["backend"] == "triton" or candidate["status"] != "ok" or baseline is None:
            continue
        common = baseline["measurements"].keys() & candidate["measurements"].keys()
        ratios = {key: candidate["measurements"][key] / baseline["measurements"][key]
                  for key in sorted(common) if baseline["measurements"][key] > 0}
        warm = [ratio for key, ratio in ratios.items()
                if key.endswith(("wall_ms_median", "/wall_ms/median")) and ratio > 0]
        step_phases = [ratio for key, ratio in ratios.items() if key.endswith("/step_phases_ms/median")]
        baseline_sampling, candidate_sampling = baseline.get("sampling", {}), candidate.get("sampling", {})
        sampling_common = baseline_sampling.keys() & candidate_sampling.keys()
        count_fields = ("actual_warmup", "actual_iterations_per_repeat", "actual_repeats")
        different_counts = [key for key in sorted(sampling_common)
                            if any(baseline_sampling[key].get(field) != candidate_sampling[key].get(field) for field in count_fields)]
        rows.append({"workload": candidate["workload"], "backend": candidate["backend"], "trial": candidate["trial"],
                     "ratios_vs_triton": ratios, "matching_metrics": len(ratios),
                     "unmatched_baseline_metrics": sorted(baseline["measurements"].keys() - common),
                     "unmatched_candidate_metrics": sorted(candidate["measurements"].keys() - common),
                     "warm_wall_ratio_geomean": math.exp(sum(map(math.log, warm)) / len(warm)) if warm else None,
                     "warm_wall_cases_above_1_25": sum(ratio > 1.25 for ratio in warm),
                     "warm_step_phase_ratios": step_phases,
                     "sampling": {"matching_measurements": len(sampling_common), "different_effective_counts": different_counts,
                                  "baseline": baseline.get("sampling_summary", sampling_summary(baseline_sampling)),
                                  "candidate": candidate.get("sampling_summary", sampling_summary(candidate_sampling)),
                                  "caution": "Ratios compare per-step averages. Budget-limited or unequal actual sample counts can reduce confidence; inspect worker sampling records."}})
    return rows


def write_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_matrix(args, cases, provenance):
    started = perf_counter()
    for case in cases:
        case["worker_identity"] = _fingerprint({"case": {key: value for key, value in case.items() if key != "worker_identity"},
                                                "provenance": provenance})
    identity = _fingerprint({"schema_version": 2, "matrix": cases, "provenance": provenance})
    destination = args.output / "summary.json"
    previous = None
    if args.resume:
        try:
            previous = json.loads(destination.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot resume without a valid summary.json: {error}") from error
        if not isinstance(previous, dict) or previous.get("run_identity") != identity:
            raise ValueError("Resume identity mismatch: scripts, library build, policy, sampling caps, dependencies or device changed; use a new output directory")
        if (previous.get("schema_version") != 2 or not isinstance(previous.get("created_utc"), str)
                or not isinstance(previous.get("results"), list)
                or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in previous["results"])):
            raise ValueError("Resume summary has an invalid checkpoint schema")
        if not provenance["resume_eligible"] and any(row.get("status") == "ok" for row in previous.get("results", [])):
            raise ValueError("Cannot reuse successful CUDA workers without a verified physical GPU/driver identity")
    report = {"schema_version": 2, "created_utc": previous["created_utc"] if previous else datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "preset": args.preset, "device": args.device,
              "execution": "cpu_reference" if args.device == "cpu" else "cuda",
              "settings": settings_snapshot(args),
              "run_identity": identity, "provenance": provenance, "planned_work": planned_work(args, cases),
              "matrix": cases, "results": [], "comparisons": [],
              "notes": ["Reports and self-checks do not qualify complete operator parity; run the mandatory GPU test suite separately.",
                        "Process wall time includes startup, compilation, diagnostics and serialization; it is not cold kernel compilation time.",
                        "Ratios use matching semantic rows from the same trial; values above one are slower or use more memory.",
                        "Warm wall geometric means are unweighted across matching rows; unmatched rows remain visible.",
                        "Requested warmup/iterations/repeats are caps under a measurement budget; actual counts and budget exhaustion remain in worker sampling records.",
                        "No profiler or sanitizer is enabled during these throughput runs. Use family scripts for separate diagnostic passes."]}
    prior_rows = {row["id"]: row for row in previous.get("results", [])} if previous else {}
    cached = {}
    if args.resume and provenance["resume_eligible"]:
        for case in cases:
            checkpoint = prior_rows.get(case["id"], {})
            try:
                durable = json.loads((args.output / (case["id"] + ".checkpoint.json")).read_text())
                if isinstance(durable, dict):
                    checkpoint = durable
            except (OSError, ValueError):
                pass
            row = reusable_worker(case, checkpoint)
            if row is not None:
                cached[case["id"]] = row
    runs_started = 0
    if args.qualify_tilelang:
        prior_progress = previous.get("progress", {}) if previous else {}
        if not isinstance(prior_progress, dict):
            raise ValueError("Invalid qualification attempt ledger")
        runs_started = prior_progress.get("worker_runs_started", 0)
        if type(runs_started) is not int or runs_started < 0:
            raise ValueError("Invalid qualification attempt ledger")
        pending_runs = sum(case["id"] not in cached and not (args.device == "cpu" and case["gpu_only"]) for case in cases)
        if args.max_worker_runs is not None and runs_started + pending_runs > args.max_worker_runs:
            raise ValueError("Qualification batch exhausted --max-worker-runs, including previous failures. "
                             "Use a new output directory or explicitly increase the declared budget to retry.")
    work = report["planned_work"]
    print(f"Planned {len(cases)} isolated workers; {len(cached)} verified successful workers reusable. "
          f"Caps per measurement: warmup={args.warmup}, iterations={args.iterations}, repeats={args.repeats}.", flush=True)
    if work["estimated_measurement_budget_seconds"] is not None:
        qualifier = "partial" if not work["measurement_estimate_complete"] else "nominal"
        print(f"Measurement budget target ({qualifier}): {work['estimated_measurement_calls']} calls × "
              f"{work['time_budget_seconds_per_measurement']:g}s = {work['estimated_measurement_budget_seconds']:g}s; "
              "compilation/setup/diagnostics and slow single-step overruns are additional.", flush=True)
    else:
        print("Exhaustive measurement counts enabled; total duration depends on kernel speed and compilation.", flush=True)
    report["progress"] = {"completed_workers": 0, "reused_workers": len(cached), "pending_workers": len(cases) - len(cached),
                          "session_wall_seconds": 0.0, "rough_remaining_seconds": None}
    if args.qualify_tilelang:
        report["progress"].update(worker_runs_started=runs_started, max_worker_runs=args.max_worker_runs)
    write_report(destination, report)
    if not args.dry_run:
        for index, case in enumerate(cases, 1):
            checkpoint_path = args.output / (case["id"] + ".checkpoint.json")
            if case["id"] in cached:
                row = cached[case["id"]]
                print(f"[{index}/{len(cases)}] {case['id']}: reused verified success "
                      f"(original worker {row.get('process_wall_seconds', 0):.2f}s)", flush=True)
            elif args.device == "cpu" and case["gpu_only"]:
                row = {**case, "status": "unsupported", "error": "This workload requires a real CUDA GPU", "measurements": {}}
            else:
                print(f"[{index}/{len(cases)}] Running {case['id']}", flush=True)
                # A retry cannot accidentally validate a stale successful JSON
                # when its new worker returns without producing a report.
                Path(case["report"]).unlink(missing_ok=True)
                write_report(checkpoint_path, {**case, "status": "running"})
                if args.qualify_tilelang:
                    runs_started += 1
                    report["progress"]["worker_runs_started"] = runs_started
                    # Persist the charge before launching; crashes cannot erase
                    # an attempt and turn a resume into an unbounded retry.
                    write_report(destination, report)
                # Heartbeat is a console preference and is deliberately absent
                # from worker commands/identities and measurement records.
                with worker_heartbeat(case, args.heartbeat_interval):
                    row = execute(case, args.timeout)
                row["reused"] = False
                print(f"[{index}/{len(cases)}] {case['id']}: {row['status']} in "
                      f"{row.get('process_wall_seconds', 0):.2f}s", flush=True)
            write_report(checkpoint_path, row)
            report["results"].append(row)
            report["comparisons"] = comparisons(report["results"])
            elapsed = perf_counter() - started
            observed = [item["process_wall_seconds"] for item in report["results"] if item.get("process_wall_seconds", 0) > 0]
            remaining = sum(item["id"] not in cached for item in cases[index:])
            report["progress"].update(completed_workers=index, pending_workers=remaining, session_wall_seconds=elapsed,
                                      rough_remaining_seconds=sum(observed) / len(observed) * remaining if observed else None)
            write_report(destination, report)
    report["progress"]["session_wall_seconds"] = perf_counter() - started
    write_report(destination, report)
    print(f"Report: {destination}")
    return int(any(row["status"] != "ok" for row in report["results"]))


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if os.name != "posix":
        arguments.error("isolated benchmark process-group timeouts require a POSIX host")
    if min(args.iterations, args.repeats, args.trials) < 1 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        arguments.error("counts and timeout must be positive; warmup may be zero")
    if not math.isfinite(args.time_budget) or args.time_budget < 0:
        arguments.error("--time-budget must be finite and nonnegative; 0 disables the budget")
    if not math.isfinite(args.heartbeat_interval) or args.heartbeat_interval < 0:
        arguments.error("--heartbeat-interval must be finite and nonnegative; 0 disables heartbeats")
    if args.device != "cpu" and args.device != "cuda" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        arguments.error("--device must be cpu, cuda, or cuda:N")
    try:
        cases = matrix(args)
        preflight_cli(cases)
        if args.output.exists() and not args.resume:
            raise ValueError("--output must be a new directory so existing evidence cannot be overwritten; use --resume for matching checkpoints")
        if args.resume and not args.output.is_dir():
            raise ValueError("--resume requires an existing run directory")
        provenance = run_provenance(args)
        args.output.mkdir(parents=True, exist_ok=args.resume)
        with output_lock(args.output):
            return run_matrix(args, cases, provenance)
    except ImportError as error:
        arguments.error(f"Benchmark contract preflight cannot load the selected DeepOpt build ({error}). "
                        "Update/re-export the complete benchmark bundle and install its matching current DeepOpt build.")
    except (OSError, TypeError, ValueError) as error:
        arguments.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
