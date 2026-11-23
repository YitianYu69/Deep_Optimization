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
import os
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
from analysis.matrix import CASES, DEFAULT_SETTINGS, EXTENDED_GRID, fingerprint, incompatibility, merge, variants, numerical_recipe_identity
from analysis.reporting import write_json, write_reports
from analysis.artifacts import snapshot_valid, profiles_valid


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
    result.add_argument("--profile-replay-steps", type=int, default=0, help="Separate replay profile for graph cases; capture occurs before profiling")
    result.add_argument("--profile-detail", choices=("summary", "trace", "stacks"), default="trace",
                        help="Summary counters, compact trace, or detailed Python stacks; outside throughput")
    result.add_argument("--trace-workloads", nargs="+", help="Only export traces for matching workload globs; other profiles retain counters")
    result.add_argument("--trace-cases", nargs="+", choices=tuple(CASES), help="Only export traces for these feature cases")
    result.add_argument("--trace-variants",choices=("all","baseline"),default="all",help="Select profiler variants independently from full timing coverage")
    result.add_argument("--compress-traces", action="store_true", help="Store raw traces as gzip JSON")
    result.add_argument("--trace-directory", help="Optional separate filesystem/directory for raw profiler artifacts")
    result.add_argument("--latency-steps", type=int, default=0, help="Serialized per-update diagnostic with GC and allocation telemetry; 0 disables")
    result.add_argument("--quality-steps", type=int, default=0, help="Separate eager learning-curve diagnostic; real task validation requires a factory validate(model) callback")
    result.add_argument("--quality-interval", type=int, default=20)
    result.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--numerical-comparison", action=argparse.BooleanOptionalAction, default=True,
                        help="Save sampled tensors and compare against BF16; disable for performance-only screening")
    result.add_argument("--numerical-snapshot-mode", choices=("disk", "memory"), default="disk",
                        help="Legacy disk artifacts or a live numerical IPC channel; compiler qualification requires memory")
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
    result.add_argument("--scheduler", choices=("serial","auto","local","dask"),default="serial",
                        help="Fresh-process supervisor; optional Dask uses the same physical GPU broker")
    result.add_argument("--devices", help="auto or comma-separated physical GPU UUIDs/indices within CUDA_VISIBLE_DEVICES")
    result.add_argument("--device-affinity-file",type=Path,help="Suite-owned model/workload to physical UUID mapping")
    result.add_argument("--cpu-workers",type=int,default=4,help="Concurrent fresh-process supervisors, never shared CUDA threads")
    result.add_argument("--gpu-concurrency",default="auto",help="Maximum admitted children per physical GPU (auto: up to four)")
    result.add_argument("--timing-isolation",choices=("exclusive","shared"),default="exclusive",
                        help="Shared mode is screening only; unknown memory footprints calibrate exclusively")
    result.add_argument("--memory-headroom-gib",type=float,default=4)
    result.add_argument("--lease-db",type=Path,help="Host-wide SQLite GPU broker (default: per-user system temporary directory)")
    result.add_argument("--preflight-manifest",type=Path,help="Dependency-specific failure manifest produced by the v6 suite")
    result.add_argument("--deduplicate-effective-settings",action="store_true",
                        help="Keep every requested row, reference canonical runs only for proven inactive settings")
    result.add_argument("--profile-policy",choices=("all","selected"),default="all",
                        help="selected disables profiling entirely outside --trace-workloads/--trace-cases")
    result.add_argument("--in-process", action="store_true", help="CPU smoke tests only; disables process/allocator isolation")
    result.add_argument("--strict-accuracy", action="store_true", help="Exit nonzero if a matched numerical comparison fails")
    result.add_argument("--require-fp8-plan", action="store_true",
                        help="For FP8 models/pipelines, reject planner fallbacks or an empty FP8 edge plan before timing")
    result.add_argument("--observed-epoch-seconds", nargs=4, type=float, metavar=("BF16", "COMPRESSION", "FP8", "COMBINED"),
                        help="Optional user epoch timings for context in the report")
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
    if args.strict_accuracy and not args.numerical_comparison:
        raise ValueError("--strict-accuracy requires --numerical-comparison")
    if args.numerical_snapshot_mode == "memory" and args.resume:
        raise ValueError("Memory numerical comparisons require fresh workers; --resume is a legacy disk option")
    if not args.device.startswith(("cpu", "cuda")):
        raise ValueError("device must be cpu or cuda[:index]")
    if args.in_process and args.device != "cpu":
        raise ValueError("--in-process is restricted to CPU validation; CUDA experiments require worker isolation")
    if args.cpu_workers < 1 or not math.isfinite(args.memory_headroom_gib) or args.memory_headroom_gib < 0:
        raise ValueError("cpu-workers must be positive; memory-headroom-gib must be finite and nonnegative")
    try:
        args.gpu_concurrency=min(4,args.cpu_workers) if args.gpu_concurrency=="auto" else int(args.gpu_concurrency)
    except ValueError:
        raise ValueError("gpu-concurrency must be auto or a positive integer") from None
    if args.gpu_concurrency < 1 or args.in_process and args.scheduler!="serial":
        raise ValueError("gpu-concurrency must be positive; --in-process requires --scheduler serial")
    if args.scheduler=="serial" and (args.devices or args.deduplicate_effective_settings or args.timing_isolation=="shared"):
        args.scheduler="local"
    if min(args.iterations, args.repeats, args.max_experiments, args.quality_interval) < 1 or min(args.warmup, args.phase_steps, args.profile_steps, args.profile_replay_steps, args.latency_steps, args.quality_steps, args.max_numerical_elements) < 0:
        raise ValueError("iterations/repeats/max-experiments must be positive; warmup/phase/profile counts must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if args.observed_epoch_seconds and any(not math.isfinite(x) or x <= 0 for x in args.observed_epoch_seconds):
        raise ValueError("Observed epoch times must be positive and finite")
    settings, grid, workloads, tolerances = configuration(args)
    options = {key: getattr(args, key) for key in (
        "device", "seed", "warmup", "iterations", "repeats", "phase_steps", "profile_steps", "memory",
        "capture_workloads", "max_numerical_elements", "deterministic", "cudnn_benchmark", "fill_uninitialized_memory", "require_fp8_plan",
        "profile_replay_steps", "latency_steps", "quality_steps", "quality_interval",
        "profile_detail", "trace_workloads", "trace_cases", "trace_variants", "compress_traces", "trace_directory", "profile_policy",
        "numerical_comparison", "numerical_snapshot_mode")}
    blocked=json.loads(args.preflight_manifest.read_text()) if args.preflight_manifest else {}
    tasks, digest = [], code_hash()
    for variant_id, variant in variants(settings, grid, args.sweep):
        for workload in workloads:
            cases = ("codec",) if workload["op"].endswith("codec") else tuple(dict.fromkeys(args.cases))
            for case in cases:
                task = {"variant_id": variant_id, "settings": variant, "workload": workload, "case": case,
                        "numerical_recipe": numerical_recipe_identity(variant, case),
                        "baseline_variant":variant_id==fingerprint(settings)}
                task["id"] = fingerprint({"task": task, "options": options, "code_sha256": digest})
                task["skip_reason"] = incompatibility(variant, case, workload, args.device) if case != "codec" else None
                if blocked:
                    from analysis.preflight import blocking_reason
                    task["skip_reason"]=task["skip_reason"] or blocking_reason(task,blocked)
                tasks.append(task)
                if len(tasks) > args.max_experiments:
                    raise ValueError(f"Grid exceeds {args.max_experiments} experiments. Narrow the grid or explicitly raise --max-experiments.")
    fills = sorted({t["settings"]["runtime"]["timing_fill_uninitialized_memory"]
                        if t["settings"]["runtime"]["timing_fill_uninitialized_memory"] is not None
                        else args.fill_uninitialized_memory for t in tasks})
    return {"schema_version": 1, "code_sha256": digest, "sweep": args.sweep, "grid": grid,
            "measurement_preflight": {"correctness_fill": args.fill_uninitialized_memory,
                                      "performance_fill_values": fills, "deterministic": args.deterministic,
                                      "profile_detail": args.profile_detail},
            "options": options, "tolerances": tolerances, "inventory": inventory(), "tasks": tasks,
            "process_isolation": not args.in_process,
            "coverage_note": "One-at-a-time measures main effects; cartesian fully crosses the explicitly declared finite grid. "
                             "Neither mode enumerates all possible shapes or integer hyperparameters. Unsupported rows remain visible."}


def execute_task(task, options, directory):
    from analysis.telemetry import UtilizationTelemetry

    interval = float(os.environ.get("DEEPOPT_BENCHMARK_TELEMETRY_INTERVAL", "0"))
    with UtilizationTelemetry(options.get("device", "cpu"), interval=interval) as telemetry:
        row = _execute_task(task, options, directory)
    row["telemetry"] = telemetry.report()
    return row


def _execute_task(task, options, directory):
    from analysis.scheduler import profile_options
    options=profile_options(task,options,options.get("profile_policy","all"))
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
        if isinstance(getattr(error, "inductor_artifacts", None), dict):
            row["inductor_artifacts"] = error.inductor_artifacts
        if not known:
            (directory / "exception.txt").write_text(traceback.format_exc())
            row["traceback"] = str(directory / "exception.txt")
    row.update(task=task, elapsed_seconds=time.perf_counter() - started)
    if options.get("device","").startswith("cuda"):
        import torch
        if torch.cuda.is_initialized():
            row["worker_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(options["device"])
    if "environment" in row:
        row["environment"]["resume_environment_sha256"] = options.get("resume_environment_sha256")
    return row


def worker(path):
    request = json.loads(path.read_text())
    request["options"]["_numerical_worker_run_id"] = request["run_id"]
    row = execute_task(request["task"], request["options"], path.parent)
    row["run_id"] = request["run_id"]
    write_json(path.parent / "result.json", row)
    return 1 if row["status"] == "error" else 0


def run(args, plan):
    inherited=os.environ.get("DEEPOPT_BENCHMARK_LEASE")
    if inherited:
        from analysis.scheduler import Broker,default_broker_path
        with Broker(os.environ.get("DEEPOPT_BENCHMARK_LEASE_DB") or default_broker_path()).connect() as db:
            if not db.execute("SELECT 1 FROM leases WHERE token=?",(inherited,)).fetchone():
                raise RuntimeError("The inherited GPU lease is no longer active")
        # Allocator trials are fresh serial subprocesses of the exclusive
        # standalone owner. A nested lease would deadlock against that owner.
        args.scheduler="serial"
    if not inherited and (args.scheduler!="serial" or args.device.startswith("cuda")):
        from analysis.scheduler import run_plan
        if args.scheduler=="serial":args.cpu_workers=args.gpu_concurrency=1
        results,run_id=run_plan(args,plan)
        return finalize(args,plan,results,run_id)
    from analysis.engine import installed_build_fingerprint, resume_environment_fingerprint

    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    print("Measurement preflight: " + json.dumps(plan["measurement_preflight"], sort_keys=True), file=sys.stderr, flush=True)
    build_hash = installed_build_fingerprint()
    plan["installed_build_sha256"] = build_hash
    environment_hash = resume_environment_fingerprint(args.device)
    plan["options"]["resume_environment_sha256"] = environment_hash
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
            try:
                cached = json.loads(result_path.read_text())
            except (ValueError, OSError):
                cached = {}
            if (cached.get("task", {}).get("id") == task["id"] and cached.get("status") == "ok"
                    and cached.get("environment", {}).get("deepopt_source_sha256") == build_hash
                    and cached.get("environment", {}).get("resume_environment_sha256") == environment_hash) and (
                    task["case"] == "codec" or snapshot_valid(cached)):
                if profiles_valid(cached, task, plan["options"]):
                    row = cached
        if row is None:
            worker_options = dict(plan["options"])
            core_path = folder / "core-result.json"
            if args.resume and core_path.exists() and task["case"] != "codec":
                try:
                    core = json.loads(core_path.read_text())
                    if (core.get("task") == task and core.get("status") == "ok" and snapshot_valid(core)
                            and core.get("environment", {}).get("deepopt_source_sha256") == build_hash
                            and core.get("environment", {}).get("resume_environment_sha256") == environment_hash):
                        worker_options["_resume_core"] = str(core_path)
                except (OSError, ValueError):
                    pass
            if args.in_process or task["skip_reason"]:
                worker_options["_numerical_worker_run_id"] = run_id
                row = execute_task(task, worker_options, folder)
                row["run_id"] = run_id
                write_json(result_path, row)
            else:
                request = folder / "request.json"
                write_json(request, {"task": task, "options": worker_options, "run_id": run_id})
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
                        elif child.returncode != 0:
                            row.update(status="error", reason=f"Worker exited {child.returncode} after publishing a result; see {folder / 'worker.log'}")
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
    return finalize(args,plan,results,run_id)


def finalize(args,plan,results,run_id):
    from analysis.reporting import summarize
    directory=args.output.resolve()
    epoch = dict(zip(("bf16", "compression", "fp8", "combined"), args.observed_epoch_seconds)) if args.observed_epoch_seconds else None
    summary = summarize(results, plan["tolerances"], epoch,
                        numerical_comparisons=getattr(args, "_numerical_comparisons", None))
    summary.update(schema_version=1, run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(),
                   plan=str(directory / "plan.json"), process_isolation=not args.in_process,
                   results=results, coverage_note=plan["coverage_note"])
    write_reports(directory, summary)
    recorded = [w for row in results for w in row.get("recorded_workloads", [])]
    if recorded:
        # Baseline variants can record the same site repeatedly; keep one replay specification.
        unique = {w["name"]: w for w in recorded}
        write_json(directory / "recorded-workloads.json", {"settings": DEFAULT_SETTINGS, "workloads": list(unique.values())})
    print(json.dumps({"output": str(directory), "counts": summary["counts"],
                      "qualification": summary["qualification"], "report": str(directory / "report.md")}, indent=2))
    failed = summary["counts"]["error"] > 0 or any(r.get("artifact_errors") for r in results)
    if args.strict_accuracy:
        failed |= any(not row["accuracy_passed"] or not row["same_initial_state"] or not row["same_installed_build"]
                      for row in summary["comparisons"])
    return 1 if failed else 0


def main(argv=None, *, numerical_comparisons=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    args._numerical_comparisons = numerical_comparisons
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
    from analysis.scheduler import output_lock,cancellation_signals
    with cancellation_signals(),output_lock(args.output):
        return run(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
