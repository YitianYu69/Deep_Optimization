"""Compare installed DeepOpt eager and native compiled VAE training execution.

Each route runs alone in a fresh process. This measures execution, runtime and
CUDA allocator peaks; it does not run a numerical or compression-error oracle.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MODES = ("eager", "native")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--from-result", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path("work/vae_eager_native.json"))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--updates", type=int, default=10)
    result.add_argument("--trials", type=int, default=1)
    result.add_argument("--project-root", type=Path)
    result.add_argument("--train-dir", type=Path)
    result.add_argument("--optimizer", choices=("custom", "sgd"))
    result.add_argument("--synthetic", action="store_true")
    result.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--dry-run", action="store_true", help="Print the paired plan without importing Torch")
    result.add_argument("--arm-manifest", type=Path, help="Pinned installed release for both routes")
    result.add_argument("--profile-after-measurement", action="store_true")
    result.add_argument("--profile-updates", type=int, default=1)
    result.add_argument("--profile-memory", action="store_true")
    result.add_argument("--profile-record-shapes", action="store_true")
    result.add_argument("--memory-audit", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--start-free-tolerance-mib", type=int, default=256)
    return result


def worker_command(args, mode, output):
    command = [sys.executable, "-B", str(ROOT / "examples/vae/native_execution.py"),
        "--from-result", str(args.from_result.resolve()), "--execution-mode", mode, "--benchmark",
        "--device", args.device, "--batch-size", str(args.batch_size), "--warmup", str(args.warmup),
        "--updates", str(args.updates), "--output", str(output)]
    for option in ("project_root", "train_dir", "optimizer"):
        value = getattr(args, option)
        if value is not None:
            command.extend(("--" + option.replace("_", "-"), str(value)))
    if args.synthetic:
        command.append("--synthetic")
    if args.deterministic is not None:
        command.append("--deterministic" if args.deterministic else "--no-deterministic")
    if getattr(args, "memory_audit", True) is False:
        command.append("--no-memory-audit")
    if getattr(args, "arm_manifest", None) is not None:
        command.extend(("--arm-manifest", str(args.arm_manifest.resolve())))
    if getattr(args, "profile_after_measurement", False):
        command.extend(("--profile-after-measurement", "--profile-updates", str(args.profile_updates)))
        if args.profile_memory:
            command.append("--profile-memory")
        if args.profile_record_shapes:
            command.append("--profile-record-shapes")
    return command


def benchmark_plan(args):
    from examples.vae import native_execution

    if args.trials < 1 or args.warmup < 1:
        raise ValueError("trials and benchmark warmup must be positive")
    tolerance_mib = getattr(args, "start_free_tolerance_mib", 256)
    if type(tolerance_mib) is not int or tolerance_mib < 0:
        raise ValueError("Starting free-memory tolerance must be a nonnegative integer MiB count")
    plans = {}
    for mode in MODES:
        worker_args = native_execution.parser().parse_args(worker_command(args, mode, args.output)[3:])
        plans[mode] = native_execution.execution_plan(worker_args)
    return {"schema_version": 1, "scope": "DeepOpt_eager_vs_native_compiled_execution_benchmark",
        "qualified": False, "qualification_eligible": False, "correctness_assessed": False,
        "numerical_comparisons_performed": False, "tensor_files_written": 0,
        "trials_requested": args.trials, "warmup": args.warmup, "updates": args.updates,
        "requested_input_shape": plans["native"]["requested_input_shape"],
        "origin": plans["native"]["origin"], "workers": plans,
        "order": [list(MODES if trial % 2 == 0 else reversed(MODES)) for trial in range(args.trials)],
        "process_isolation": "fresh sequential process per mode and trial; one model on the GPU at a time",
        "timing_scope": "GPU-resident repeated batch; loss, backward, optimizer and schedules included",
        "excluded_from_timing": ["dataset_loading", "H2D", "compile_and_warmup", "finite_state_scans",
            "numerical_oracles", "compiler_observers", "artifact_serialization", "source_hashing"],
        "memory_scope": "Torch CUDA allocator peaks; startup and measured windows reported separately",
        "memory_isolation_policy": {"audit_enabled": getattr(args, "memory_audit", True),
            "start_free_tolerance_bytes": tolerance_mib * 2**20,
            "release_boundary": "fresh sequential worker process exit", "warmup_pool_retained": True},
        "compiled_scope": "DeepOpt model forward and native AutoDiff backward; eager loss and optimizer"}


def comparable_identity(report):
    """Everything that must match across routes, excluding compiler selection."""
    settings = deepcopy(report.get("settings"))
    if isinstance(settings, dict):
        settings.get("runtime", {}).pop("compile", None)
    effective = deepcopy(report.get("runtime_config"))
    if isinstance(effective, dict):
        effective.get("runtime", {}).pop("compile", None)
    installed = report.get("installed", {})
    inputs = report.get("input", {})
    model = report.get("model", {})
    return {"origin_sha256": report.get("origin", {}).get("sha256"),
        "common_settings": settings, "initial_model_sha256": model.get("initial_state_sha256"),
        "effective_runtime_config": effective,
        "model_parameters": model.get("parameters"), "model_factory": report.get("model_factory"),
        "initial_optimizer_sha256": report.get("optimizer", {}).get("initial_state_sha256"),
        "input": {key: inputs.get(key) for key in ("sha256", "shape", "dtype", "stride", "contiguous",
            "storage_bytes", "requires_grad", "mode")},
        "optimizer_factory": report.get("optimizer_factory"),
        "installed": {key: installed.get(key) for key in ("version", "source_sha256", "files_hashed")},
        "provenance": report.get("provenance"), "helpers": report.get("helper_sha256"),
        "software": report.get("software"), "device": report.get("device"),
        "warmup": report.get("warmup"), "updates": report.get("updates"),
        "measurement_scope": {key: report.get("measurement", {}).get(key)
            for key in ("timing_scope", "wall_scope", "cuda_event_scope")}}


def _identity_errors(report):
    identity = comparable_identity(report)
    errors = []
    for field in ("origin_sha256", "initial_model_sha256", "initial_optimizer_sha256", "model_parameters", "common_settings",
                  "effective_runtime_config",
                  "provenance", "helpers", "software", "device", "warmup", "updates", "measurement_scope"):
        if not identity[field]:
            errors.append("missing_identity:" + field)
    for field in ("sha256", "shape", "dtype", "stride", "storage_bytes", "mode"):
        if not identity["input"][field]:
            errors.append("missing_identity:input." + field)
    for field in ("version", "source_sha256", "files_hashed"):
        if not identity["installed"][field]:
            errors.append("missing_identity:installed." + field)
    if str(report.get("device", "")).startswith("cuda"):
        if not (report.get("software") or {}).get("gpu", {}).get("uuid"):
            errors.append("missing_identity:gpu.uuid")
    for field in ("source_stable", "installed_stable", "helper_source_stable"):
        if report.get(field) is not True:
            errors.append("unstable_or_unverified:" + field)
    if report.get("input_stable") is not True:
        errors.append("unstable_or_unverified:input_stable")
    if not identity["measurement_scope"].get("timing_scope"):
        errors.append("missing_identity:measurement_scope.timing_scope")
    return errors


def _number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _metrics(report):
    measurement, memory = report.get("measurement", {}), report.get("memory", {})
    result = {key: measurement.get(key) for key in
              ("wall_ms_per_update", "cuda_event_ms_per_update", "images_per_second")}
    for prefix, segment in (("steady", measurement.get("memory", {})),
                            ("startup", memory.get("startup", {})), ("total", memory.get("total", {}))):
        for kind in ("allocated", "reserved"):
            result[f"{prefix}_peak_{kind}_bytes"] = segment.get(f"peak_{kind}_bytes")
    return result


def _statistics(values):
    values = [value for value in values if _number(value) or
              (type(value) in (int, float) and math.isfinite(value))]
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "samples": len(values)} if values else None


def _worker_errors(row, *, require_memory_audit=False):
    reasons = []
    if require_memory_audit or "memory_audit" in row:
        from examples.vae.memory_comparison import worker_memory_errors

        reasons.extend(worker_memory_errors(row))
    if row.get("status") != "completed" or row.get("execution_completed") is not True:
        reasons.append("worker_not_completed")
    if row.get("worker_exit_code", 0) != 0:
        reasons.append("nonzero_worker_exit")
    measurement = row.get("measurement", {})
    if measurement.get("valid") is not True:
        reasons.append("invalid_measurement")
        reasons.extend(measurement.get("invalid_reasons", []))
    if not _number(measurement.get("wall_ms_per_update"), positive=True):
        reasons.append("missing_positive_wall_runtime")
    if measurement.get("measured_updates") != row.get("updates"):
        reasons.append("incomplete_measured_updates")
    if (row.get("warmup_completed") != row.get("warmup") or row.get("updates_completed") != row.get("updates")
            or measurement.get("warmup_updates") != row.get("warmup")):
        reasons.append("incomplete_training_updates")
    postchecks = measurement.get("postchecks", {})
    if postchecks.get("finite_losses") is not True or postchecks.get("finite_parameters") is not True:
        reasons.append("finite_state_postchecks_unverified")
    if str(row.get("device", "")).startswith("cuda"):
        for name, value in _metrics(row).items():
            if ("peak_" in name or name == "cuda_event_ms_per_update") and not _number(value, positive=True):
                reasons.append("missing_positive_CUDA_metric:" + name)
    if row.get("execution_mode") == "native" and row.get("native_ownership_verified") is not True:
        reasons.append("native_ownership_unverified")
    compilation = (row.get("settings") or {}).get("runtime", {}).get("compile", {})
    if row.get("execution_mode") == "native":
        options = compilation.get("options", {})
        if (compilation.get("enabled") is not True or compilation.get("backend") != "deepopt"
                or compilation.get("fullgraph") is not True or options.get("use_native_autodiff") is not True
                or options.get("require_full_aot") is not True):
            reasons.append("invalid_native_compile_recipe")
    elif row.get("execution_mode") == "eager":
        if (compilation.get("enabled") is not False or row.get("eager_ownership_verified") is not True):
            reasons.append("eager_ownership_unverified")
    reasons.extend(_identity_errors(row))
    return reasons


def aggregate_trials(results, *, start_free_tolerance_bytes=256 * 2**20, require_memory_audit=False):
    """Summarize measured workers; never fabricate a ratio for failed/mismatched pairs."""
    summary = {"valid": False, "invalid_reasons": [], "modes": {}, "paired_trials": [], "ratios": {},
        "ratio_convention": "runtime speedup = eager/native; memory reduction percent = 100*(eager-native)/eager",
        "correctness_assessed": False, "qualification_eligible": False}
    if require_memory_audit or any("memory_audit" in row for row in results):
        from examples.vae.memory_comparison import compare_memory_baselines

        summary["memory_isolation"] = compare_memory_baselines(results, tolerance_bytes=start_free_tolerance_bytes)
        summary["invalid_reasons"].extend("memory_isolation:" + value
            for value in summary["memory_isolation"]["invalid_reasons"])
    else:
        summary["memory_isolation"] = {"valid": None, "evidence_scope": "legacy_receipts_without_start_audit"}
    for mode in MODES:
        selected = [row for row in results if row.get("execution_mode") == mode]
        successful = [row for row in selected if not _worker_errors(row, require_memory_audit=require_memory_audit)]
        names = set().union(*(_metrics(row).keys() for row in successful)) if successful else set()
        summary["modes"][mode] = {"successful_trials": len(successful), "total_trials": len(selected),
            "metrics": {name: _statistics([_metrics(row)[name] for row in successful]) for name in sorted(names)}}
    anchor = None
    trials = sorted({row.get("trial", 1) for row in results})
    for trial in trials:
        selected = [row for row in results if row.get("trial", 1) == trial]
        pair = {"trial": trial, "valid": False, "invalid_reasons": [], "ratios": {}}
        by_mode = {mode: [row for row in selected if row.get("execution_mode") == mode] for mode in MODES}
        if any(len(rows) != 1 for rows in by_mode.values()) or len(selected) != 2:
            pair["invalid_reasons"].append("expected_one_eager_and_one_native_worker")
        else:
            eager, native = by_mode["eager"][0], by_mode["native"][0]
            for mode, row in (("eager", eager), ("native", native)):
                pair["invalid_reasons"].extend(mode + ":" + error for error in
                    _worker_errors(row, require_memory_audit=require_memory_audit))
            left, right = comparable_identity(eager), comparable_identity(native)
            pair["identity_sha256"] = hashlib.sha256(json.dumps(left, sort_keys=True).encode()).hexdigest()
            for field in left:
                if left[field] != right[field]:
                    pair["invalid_reasons"].append("identity_mismatch:" + field)
            if not pair["invalid_reasons"]:
                if anchor is None:
                    anchor = left
                elif left != anchor:
                    pair["invalid_reasons"].append("identity_mismatch_across_trials")
            if not pair["invalid_reasons"]:
                pair["valid"] = True
                em, nm = _metrics(eager), _metrics(native)
                pair["ratios"]["runtime_speedup_eager_over_native"] = em["wall_ms_per_update"] / nm["wall_ms_per_update"]
                if _number(em["cuda_event_ms_per_update"], positive=True) and _number(nm["cuda_event_ms_per_update"], positive=True):
                    pair["ratios"]["cuda_event_speedup_eager_over_native"] = em["cuda_event_ms_per_update"] / nm["cuda_event_ms_per_update"]
                for segment in ("steady", "startup", "total"):
                    for kind in ("allocated", "reserved"):
                        name = f"{segment}_peak_{kind}_bytes"
                        if _number(em[name], positive=True) and _number(nm[name]):
                            pair["ratios"][f"{segment}_{kind}_reduction_percent"] = 100 * (em[name] - nm[name]) / em[name]
        summary["paired_trials"].append(pair)
        summary["invalid_reasons"].extend(f"trial_{trial}:" + error for error in pair["invalid_reasons"])
    if not trials:
        summary["invalid_reasons"].append("no_workers")
    names = set().union(*(row["ratios"].keys() for row in summary["paired_trials"])) if trials else set()
    summary["valid"] = bool(trials) and not summary["invalid_reasons"]
    if summary["valid"]:
        summary["ratios"] = {name: _statistics([row["ratios"][name] for row in summary["paired_trials"]
            if row["valid"] and name in row["ratios"]]) for name in sorted(names)}
    summary["target_cuda_comparison"] = summary["valid"] and all(row.get("physical_target_executed") is True for row in results)
    summary["evidence_scope"] = "target_CUDA_execution" if summary["target_cuda_comparison"] else "explicit_diagnostic_or_incomplete_target_execution"
    return summary


def _reference(path, parent):
    payload = path.read_bytes()
    return {"path": path.relative_to(parent).as_posix(), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _reject_json_constant(value):
    raise ValueError("Nonfinite JSON value in worker result: " + value)


def _failed_aggregation_summary(results, error, *, require_memory_audit):
    """Keep usable peer metrics if an unrelated aggregation failure occurs."""
    from examples.vae import benchmark_vae as benchmark

    modes = {}
    for mode in MODES:
        selected = [row for row in results if row.get("execution_mode") == mode]
        successful = []
        for row in selected:
            try:
                if not _worker_errors(row, require_memory_audit=require_memory_audit):
                    successful.append(_metrics(row))
            except Exception:
                continue
        names = set().union(*(row.keys() for row in successful)) if successful else set()
        modes[mode] = {"successful_trials": len(successful), "total_trials": len(selected),
            "metrics": {name: _statistics([row[name] for row in successful]) for name in sorted(names)}}
    return {"valid": False, "invalid_reasons": [], "modes": modes, "paired_trials": [], "ratios": {},
            "correctness_assessed": False, "qualification_eligible": False,
            "aggregation_error": benchmark.failure_error_details(error)}


def _print_summary(summary):
    print("\nMode      successful   wall ms/update    images/s   measured peak GiB (allocated / reserved)")
    for mode in MODES:
        row = summary["modes"][mode]
        metrics = row["metrics"]
        def value(name, divisor=1):
            result = metrics.get(name)
            return f"{result['median'] / divisor:.3f}" if result else "n/a"
        print(f"{mode:9} {row['successful_trials']}/{row['total_trials']:<9} "
              f"{value('wall_ms_per_update'):>14} {value('images_per_second'):>11} "
              f"{value('steady_peak_allocated_bytes', 2**30):>12} / {value('steady_peak_reserved_bytes', 2**30)}")
    for key, row in summary["ratios"].items():
        print(f"{key}: {row['median']:.4f} (min {row['min']:.4f}, max {row['max']:.4f}; {row['samples']} pair(s))")
    if summary["invalid_reasons"]:
        print("Comparison incomplete: " + "; ".join(summary["invalid_reasons"]))


def run_benchmark(args):
    from examples.vae import benchmark_vae as benchmark

    plan = benchmark_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh benchmark output; historical results are immutable")
    directory.mkdir(parents=True)
    report = {**plan, "status": "running", "results": [],
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    report["supervisor_sources"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), ROOT / "examples/vae/memory_comparison.py")}
    for name, digest in report["supervisor_sources"].items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Supervisor source changed before its snapshot")
        target = directory / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    benchmark.write_json(output, report)
    results = []
    require_memory_audit = plan["memory_isolation_policy"]["audit_enabled"]
    aggregation_options = {"start_free_tolerance_bytes": getattr(args, "start_free_tolerance_mib", 256) * 2**20,
                           "require_memory_audit": require_memory_audit}
    try:
        from benchmarks.compiler.installed_runtime import child_environment, installed_identity

        identity = installed_identity(ROOT, full=True)
        report["installed"] = identity
        environment = child_environment(ROOT, identity=identity)
        environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
        environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"
        for trial, order in enumerate(plan["order"], 1):
            for mode in order:
                worker_output = directory / f"trial-{trial}-{mode}.json"
                log = worker_output.with_suffix(".log")
                command = worker_command(args, mode, worker_output)
                print(f"VAE benchmark: {mode}, trial {trial}/{args.trials}; "
                      f"physical_shape={plan['requested_input_shape']}, {args.warmup} warmup + {args.updates} measured updates", flush=True)
                launch_error, exit_code = None, None
                with log.open("x") as stream:
                    try:
                        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                                   stderr=subprocess.STDOUT, check=False)
                        exit_code = completed.returncode
                    except OSError as error:
                        launch_error = str(error)
                        stream.write("Worker could not start: " + launch_error + "\n")
                row = None
                try:
                    row = json.loads(worker_output.read_text(), parse_constant=_reject_json_constant)
                    if not isinstance(row, dict):
                        raise ValueError("worker result must be a JSON object")
                    validation = _worker_errors(row, require_memory_audit=require_memory_audit)
                    if any(type(reason) is not str for reason in validation):
                        raise ValueError("worker invalid reasons must contain only strings")
                    _metrics(row)  # Validate nested metric structure on CPU diagnostics too.
                except Exception as error:
                    malformed = f"Worker result unavailable or malformed: {type(error).__name__}: {error}"
                    retained = ({key: row[key] for key in ("error", "traceback", "capacity_status")
                                 if type(row.get(key)) is str} if isinstance(row, dict) else {})
                    row = {**retained, "status": "failed", "execution_completed": False,
                        "error": launch_error or retained.get("error") or malformed,
                        "worker_result_error": malformed}
                row.setdefault("status", "failed")
                returned_mode = row.get("execution_mode")
                row.update(trial=trial, execution_mode=mode, worker_exit_code=exit_code)
                if returned_mode != mode:
                    row["status"] = "failed"
                    row.setdefault("error", "Worker did not identify the requested execution mode")
                if row.get("status") == "completed" and row.get("origin", {}).get("sha256") != plan["origin"]["sha256"]:
                    row["status"] = "failed"
                    row["error"] = "Source receipt changed after the paired plan was created"
                results.append(row)
                receipt = {"trial": trial, "execution_mode": mode, "status": row["status"],
                    "worker_exit_code": exit_code, "log": _reference(log, output.parent),
                    "capacity_status": row.get("capacity_status"), "measurement": row.get("measurement"),
                    "memory": row.get("memory"), "error": row.get("error"), "command": command,
                    "worker_result_error": row.get("worker_result_error")}
                if worker_output.is_file():
                    receipt["result"] = _reference(worker_output, output.parent)
                report["results"].append(receipt)
                report["summary"] = aggregate_trials(results, **aggregation_options)
                benchmark.write_json(output, report)
                print(f"  {row['status']}; log: {log}", flush=True)
        report["installed_after"] = installed_identity(ROOT, full=True)
        if report["installed_after"] != identity:
            raise RuntimeError("Installed package changed across benchmark trials")
        report["summary"] = aggregate_trials(results, **aggregation_options)
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != report["supervisor_sha256"]:
            raise RuntimeError("Supervisor source changed during the benchmark")
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
               for name, digest in report["supervisor_sources"].items()):
            raise RuntimeError("Supervisor helper source changed during the benchmark")
        report["status"] = "completed" if report["summary"]["valid"] else "failed"
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
        try:
            report["summary"] = aggregate_trials(results, **aggregation_options)
        except Exception as aggregation_error:
            report["summary"] = _failed_aggregation_summary(results, aggregation_error,
                require_memory_audit=require_memory_audit)
        report["summary"]["valid"] = False
        report["summary"]["ratios"] = {}
        report["summary"]["target_cuda_comparison"] = False
        report["summary"]["evidence_scope"] = "explicit_diagnostic_or_incomplete_target_execution"
        report["summary"]["invalid_reasons"].append("supervisor_failed")
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    _print_summary(report["summary"])
    print(f"Results: {output}")
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    return run_benchmark(parser().parse_args(argv))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
