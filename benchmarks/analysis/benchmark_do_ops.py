"""Compare all DO operators/settings against Torch BF16 and isolate FP8/compression interaction.

Run --help, --list-ops or --dry-run without importing Torch. Copy the entire
analysis directory to a CUDA machine; execution imports its installed DeepOpt.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import uuid

if not __package__:
    # Only make this benchmark package importable, never select a repository src tree.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.catalog import OPS, family, inventory, normalize_workload, workload_list
from analysis.matrix import CASES, DEFAULT_SETTINGS, EXTENDED_GRID, fingerprint, incompatibility, merge, variants
from analysis.reporting import write_json, write_reports


REGRESSION_OPS = ("linear", "conv2d", "batch_norm", "group_norm", "layer_norm", "gelu", "relu",
                  "scaled_dot_product_attention", "linear_pipeline", "conv_pipeline", "attention_pipeline")

DEFAULT_TOLERANCES = {"default": {"rtol": .05, "atol": .01},
                      "gradients": {"rtol": .05, "atol": 1e-6, "max_relative_l2": .1, "min_cosine": .99},
                      "updates": {"rtol": .05, "atol": 1e-7, "max_relative_l2": .1, "min_cosine": .99}}


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--config", type=Path, help="JSON settings, grid, workloads and optional tolerances")
    result.add_argument("--preset", choices=("regression", "all", "extended"), default="regression")
    result.add_argument("--ops", nargs="+", choices=OPS, help="Select operators, overriding the preset")
    result.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    result.add_argument("--sweep", choices=("one-at-a-time", "cartesian"), default="one-at-a-time")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--backend", choices=("triton", "reference"), help="Override fp8.backend")
    result.add_argument("--small", action="store_true", help="Small built-in shapes for smoke testing")
    result.add_argument("--gradient-modes", nargs="+", choices=("all", "dx", "dw", "db", "none"), default=None,
                        help="Separate trainability probes; these are full executions, not additive backward phases")
    result.add_argument("--output", type=Path, default=Path("do-analysis-results"))
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--phase-steps", type=int, default=5)
    result.add_argument("--profile-steps", type=int, default=0, help="Separate CPU/CUDA profiler pass; 0 disables")
    result.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--capture-workloads", action="store_true", help="Record native module shapes/call counts in a separate pass")
    result.add_argument("--max-numerical-elements", type=int, default=65536,
                        help="Deterministic samples per float tensor for errors; 0 checks all values. Finiteness/indices are always checked in full")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--fill-uninitialized-memory", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--timeout", type=float, default=1800, help="Seconds allowed per isolated worker")
    result.add_argument("--max-experiments", type=int, default=10000, help="Reject oversized grids before launching workers")
    result.add_argument("--resume", action="store_true", help="Reuse matching successful results from this code/configuration")
    result.add_argument("--in-process", action="store_true", help="CPU smoke tests only; disables process/allocator isolation")
    result.add_argument("--strict-accuracy", action="store_true", help="Exit nonzero if a matched numerical comparison fails")
    result.add_argument("--require-fp8-plan", action="store_true",
                        help="For FP8 models/pipelines, reject planner fallbacks or an empty FP8 edge plan before timing")
    result.add_argument("--observed-epoch-seconds", nargs=4, type=float, metavar=("BF16", "COMPRESSION", "FP8", "COMBINED"),
                        help="User observations for context, e.g. 21 27 22 47; never treated as suite measurements")
    result.add_argument("--list-ops", action="store_true")
    result.add_argument("--write-example", type=Path, help="Write an editable complete JSON configuration and exit")
    result.add_argument("--dry-run", action="store_true", help="Validate and write plan.json without importing Torch or executing kernels")
    result.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    return result


def code_hash():
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configuration(args):
    data = json.loads(args.config.read_text()) if args.config else {}
    if not isinstance(data, dict) or set(data) - {"settings", "grid", "workloads", "tolerances"}:
        raise ValueError("Configuration accepts settings, grid, workloads and tolerances only")
    settings = merge(DEFAULT_SETTINGS, data.get("settings", {}))
    if args.backend:
        settings["fp8"]["backend"] = args.backend
    grid = deepcopy(data.get("grid", EXTENDED_GRID if args.preset == "extended" else {}))
    if args.backend:
        grid.pop("fp8.backend", None)
    selection = args.ops or (OPS if args.preset == "all" else REGRESSION_OPS)
    workloads = data.get("workloads", workload_list(selection, args.small))
    if args.ops and "workloads" in data:
        workloads = [w for w in workloads if w["op"] in args.ops]
    workloads = [normalize_workload(w) for w in workloads]
    if args.gradient_modes:
        workloads = [{**deepcopy(w), "name": w["name"] + "." + mode, "gradient_mode": mode}
                     for w in workloads for mode in dict.fromkeys(args.gradient_modes)]
    if not workloads or len({w["name"] for w in workloads}) != len(workloads):
        raise ValueError("Workloads must be nonempty with unique names")
    tolerances = data.get("tolerances", deepcopy(DEFAULT_TOLERANCES))
    if not isinstance(tolerances, dict):
        raise ValueError("tolerances must be an object")
    for name, policy in tolerances.items():
        if (not isinstance(policy, dict) or not {"rtol", "atol"} <= policy.keys()
                or set(policy) - {"rtol", "atol", "max_relative_l2", "min_cosine"}):
            raise ValueError(f"Invalid accuracy tolerances for {name}")
        for key, value in policy.items():
            if key in ("max_relative_l2", "min_cosine") and value is None:
                continue
            lower, upper = (-1, 1) if key == "min_cosine" else (0, math.inf)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Invalid accuracy {key} for {name}")
    return settings, grid, workloads, tolerances


def make_plan(args):
    if not args.device.startswith(("cpu", "cuda")):
        raise ValueError("device must be cpu or cuda[:index]")
    if args.in_process and args.device != "cpu":
        raise ValueError("--in-process is restricted to CPU validation; CUDA experiments require worker isolation")
    if min(args.iterations, args.repeats, args.max_experiments) < 1 or min(args.warmup, args.phase_steps, args.profile_steps, args.max_numerical_elements) < 0:
        raise ValueError("iterations/repeats/max-experiments must be positive; warmup/phase/profile counts must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if args.observed_epoch_seconds and any(not math.isfinite(x) or x <= 0 for x in args.observed_epoch_seconds):
        raise ValueError("Observed epoch times must be positive and finite")
    settings, grid, workloads, tolerances = configuration(args)
    options = {key: getattr(args, key) for key in (
        "device", "seed", "warmup", "iterations", "repeats", "phase_steps", "profile_steps", "memory",
        "capture_workloads", "max_numerical_elements", "deterministic", "cudnn_benchmark", "fill_uninitialized_memory", "require_fp8_plan")}
    tasks, digest = [], code_hash()
    for variant_id, variant in variants(settings, grid, args.sweep):
        for workload in workloads:
            cases = ("codec",) if workload["op"].endswith("codec") else tuple(dict.fromkeys(args.cases))
            for case in cases:
                task = {"variant_id": variant_id, "settings": variant, "workload": workload, "case": case}
                task["id"] = fingerprint({"task": task, "options": options, "code_sha256": digest})
                task["skip_reason"] = incompatibility(variant, case, workload, args.device) if case != "codec" else None
                tasks.append(task)
                if len(tasks) > args.max_experiments:
                    raise ValueError(f"Grid exceeds {args.max_experiments} experiments. Narrow the grid or explicitly raise --max-experiments.")
    return {"schema_version": 1, "code_sha256": digest, "sweep": args.sweep, "grid": grid,
            "options": options, "tolerances": tolerances, "inventory": inventory(), "tasks": tasks,
            "process_isolation": not args.in_process,
            "coverage_note": "One-at-a-time measures main effects; cartesian fully crosses the explicitly declared finite grid. "
                             "Neither mode enumerates all possible shapes or integer hyperparameters. Unsupported rows remain visible."}


def execute_task(task, options, directory):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if task["skip_reason"]:
        return {"status": "skipped", "reason": task["skip_reason"], "task": task}
    try:
        if task["case"] == "codec":
            from analysis.codecs import run_codec
            row = run_codec(task, options, directory)
        else:
            from analysis.engine import run_task
            row = run_task(task, options, directory)
    except Exception as error:
        from analysis.workloads import UnsupportedWorkload
        known = isinstance(error, UnsupportedWorkload)
        native_bf16_gap = (task["case"] in ("torch_bf16", "do_bf16") and family(task["workload"]["op"]) == "pooling"
                           and isinstance(error, (RuntimeError, NotImplementedError))
                           and "not implemented" in str(error).lower() and "bfloat16" in str(error).lower())
        known |= native_bf16_gap
        row = {"status": "skipped" if known else "error", "reason": f"{type(error).__name__}: {error}"}
        if not known:
            (directory / "exception.txt").write_text(traceback.format_exc())
            row["traceback"] = str(directory / "exception.txt")
    row.update(task=task, elapsed_seconds=time.perf_counter() - started)
    return row


def worker(path):
    request = json.loads(path.read_text())
    row = execute_task(request["task"], request["options"], path.parent)
    row["run_id"] = request["run_id"]
    write_json(path.parent / "result.json", row)
    return 1 if row["status"] == "error" else 0


def run(args, plan):
    from analysis.reporting import summarize
    from analysis.engine import installed_build_fingerprint

    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    build_hash = installed_build_fingerprint()
    plan["installed_build_sha256"] = build_hash
    write_json(directory / "plan.json", plan)
    pending = list(plan["tasks"])
    random.Random(args.seed).shuffle(pending)
    results, run_id = [], uuid.uuid4().hex
    for index, task in enumerate(pending):
        folder = directory / "cases" / task["id"]
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        row = None
        if args.resume and result_path.exists():
            cached = json.loads(result_path.read_text())
            if (cached.get("task", {}).get("id") == task["id"] and cached.get("status") == "ok"
                    and cached.get("environment", {}).get("deepopt_source_sha256") == build_hash) and (
                    task["case"] == "codec" or Path(cached.get("snapshot", "__missing__")).is_file()):
                row = cached
        if row is None:
            if args.in_process or task["skip_reason"]:
                row = execute_task(task, plan["options"], folder)
                row["run_id"] = run_id
                write_json(result_path, row)
            else:
                request = folder / "request.json"
                write_json(request, {"task": task, "options": plan["options"], "run_id": run_id})
                with (folder / "worker.log").open("w") as log:
                    try:
                        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(request)],
                                               stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                        if result_path.exists():
                            candidate = json.loads(result_path.read_text())
                            if candidate.get("run_id") == run_id and candidate.get("task", {}).get("id") == task["id"]:
                                row = candidate
                        if row is None:
                            row = {"status": "error", "task": task,
                                   "reason": f"Worker exited {child.returncode} without a fresh result; see {folder / 'worker.log'}"}
                    except subprocess.TimeoutExpired:
                        row = {"status": "error", "task": task, "reason": f"Worker timed out after {args.timeout}s"}
                row["run_id"] = run_id
                write_json(result_path, row)
        results.append(row)
        print(f"[{index + 1}/{len(pending)}] {task['workload']['name']} {task['case']} {task['variant_id']}: {row['status']}",
              file=sys.stderr, flush=True)
        # A partial manifest survives interruption, including unstarted experiment IDs.
        write_json(directory / "progress.json", {"run_id": run_id, "completed": len(results), "total": len(pending),
                                                "result_files": [str(directory / 'cases' / r['task']['id'] / 'result.json') for r in results]})
    epoch = dict(zip(("bf16", "compression", "fp8", "combined"), args.observed_epoch_seconds)) if args.observed_epoch_seconds else None
    summary = summarize(results, plan["tolerances"], epoch)
    summary.update(schema_version=1, run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(),
                   plan=str(directory / "plan.json"), process_isolation=not args.in_process,
                   results=results, coverage_note=plan["coverage_note"])
    write_reports(directory, summary)
    recorded = [w for row in results for w in row.get("recorded_workloads", [])]
    if recorded:
        # Baseline variants can record the same site repeatedly; keep one replay specification.
        unique = {w["name"]: w for w in recorded}
        write_json(directory / "recorded-workloads.json", {"settings": DEFAULT_SETTINGS, "workloads": list(unique.values())})
    print(json.dumps({"output": str(directory), "counts": summary["counts"], "report": str(directory / "report.md")}, indent=2))
    failed = summary["counts"]["error"] > 0
    if args.strict_accuracy:
        failed |= any(not row["accuracy_passed"] or not row["same_initial_state"] or not row["same_installed_build"]
                      for row in summary["comparisons"])
    return 1 if failed else 0


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args._worker:
        return worker(args._worker)
    if args.list_ops:
        print(json.dumps(inventory(), indent=2))
        return 0
    if args.write_example:
        write_json(args.write_example, {"settings": DEFAULT_SETTINGS,
                                       "grid": {"compression.bits": [1, 2, 4, 8],
                                                "fp8.backward_precision": ["native", "auto"]},
                                       "workloads": workload_list(REGRESSION_OPS, args.small),
                                       "tolerances": DEFAULT_TOLERANCES})
        print(args.write_example)
        return 0
    try:
        plan = make_plan(args)
    except (ValueError, TypeError, KeyError) as error:
        arguments.error(str(error))
    if args.dry_run:
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"experiments": len(plan["tasks"]), "known_skips": sum(bool(t["skip_reason"]) for t in plan["tasks"]),
                          "plan": str(args.output / "plan.json")}, indent=2))
        return 0
    return run(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
