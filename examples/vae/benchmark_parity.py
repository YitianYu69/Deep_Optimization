"""Fresh-process matched VAE AMP comparison across Torch and DeepOpt compilers."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "deepopt.vae.compiler_parity.cohort.v1"


def parser():
    from examples.vae.parity_execution import parser as worker_parser

    result = worker_parser()
    result.description = __doc__
    result.set_defaults(output=Path("work/vae_compiler_parity.json"))
    result.add_argument("--trials", type=int, default=1, help="Independent fresh-process blocks; final parity requires at least five")
    result.add_argument("--inductor-presets", choices=("default", "cast-preserving"), nargs="+", default=["default", "cast-preserving"])
    result.add_argument("--start-free-tolerance-mib", type=int, default=256)
    return result


def worker_command(args, route, preset, output, *, arm_manifest=None):
    command = [sys.executable, "-B", str(ROOT / "examples/vae/parity_execution.py"), "--from-result", str(args.from_result.resolve()),
        "--route", route, "--inductor-preset", preset, "--device", args.device, "--batch-size", str(args.batch_size),
        "--warmup", str(args.warmup), "--updates", str(args.updates), "--output", str(output)]
    for name in ("project_root", "train_dir", "optimizer", "arm_manifest", "expected_version"):
        value = arm_manifest if name == "arm_manifest" and arm_manifest is not None else getattr(args, name, None)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if route == "deepopt_native" and args.native_options_json is not None:
        command += ["--native-options-json", str(args.native_options_json)]
    for name in ("cpu_tiny", "synthetic", "profile_after_measurement", "profile_memory", "profile_record_shapes"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    if args.profile_after_measurement:
        command += ["--profile-updates", str(args.profile_updates)]
    if args.deterministic is not None:
        command.append("--deterministic" if args.deterministic else "--no-deterministic")
    return command


def benchmark_plan(args):
    from examples.vae import parity_execution as worker

    if args.trials < 1 or type(args.start_free_tolerance_mib) is not int or args.start_free_tolerance_mib < 0:
        raise ValueError("Require positive trials and nonnegative integer baseline tolerance")
    if len(set(args.inductor_presets)) != len(args.inductor_presets):
        raise ValueError("Inductor presets must be unique")
    if args.route != "deepopt_native" or args.execution_mode != "native":
        raise ValueError("The cohort always runs all four routes; route selection belongs to parity_execution.py")
    jobs = [(route, preset) for route in worker.ROUTES
            for preset in (args.inductor_presets if route == "native_torch_inductor" else ("cast-preserving",))]
    release = None
    if args.arm_manifest is not None:
        from examples.vae.native_execution import _json_input
        value, source = _json_input(args.arm_manifest)
        if value.get("native_options_sha256") is not None:
            raise ValueError("The cohort arm manifest pins the common release only; native_options_sha256 must be null")
        release = {"source": source, "manifest": value}
    plans = {}
    for route, preset in jobs:
        options = worker.parser().parse_args(worker_command(args, route, preset, args.output)[3:])
        if route == "deepopt_native" and args.native_options_json is not None:
            # The common release pin is validated by the other route plans.
            # Runtime receives a separate strict options-byte pin for this job.
            options.arm_manifest = options.expected_version = None
        plans[route + ":" + preset] = worker.execution_plan(options)
        if release is not None:
            plans[route + ":" + preset]["expected_installed_version"] = release["manifest"]["expected_installed"]["version"]
    return {"parity_schema": SCHEMA, "schema_version": 1, "trials_requested": args.trials,
        "order": [[route + ":" + preset for route, preset in (jobs if index % 2 == 0 else reversed(jobs))] for index in range(args.trials)],
        "workers": plans, "source_sha256": worker.source_identity(), "origin": next(iter(plans.values()))["origin"],
        "release_pin": release,
        "baseline_tolerance_bytes": args.start_free_tolerance_mib * 2**20,
        "independent_samples": "fresh process blocks; one DeepOpt native worker shared across the two preset comparisons",
        "scope": "same_common_AMP_model_only_compile_eager_FP32_loss_and_optimizer", "performance_policy_promoted": False,
        "target_CUDA_parity_established": False, "tensor_files_written": 0}


def _identity(row):
    result = {key: deepcopy(row.get(key)) for key in ("common_recipe_sha256", "origin", "installed", "input", "model", "optimizer", "provenance",
        "software", "helper_sha256", "compiled_scope", "warmup", "updates", "rng_before_warmup_sha256", "rng_after_warmup_sha256", "rng_after_measurement_sha256")}
    controls = row.get("numerical_controls", {})
    result["common_arithmetic_controls"] = {key: controls.get(key) for key in
        ("tf32_matmul", "tf32_cudnn", "bf16_reduced_precision_reduction", "backward_execution", "random_policy")}
    result["common_arithmetic_controls"].update({key: controls.get("effective", {}).get(key)
        for key in ("backward_pass_autocast", "fallback_random")})
    return result


def worker_errors(row):
    from examples.vae.memory_comparison import worker_memory_errors
    from examples.vae.parity_execution import SCHEMA as worker_schema
    from examples.vae.benchmark_execution import _number
    from benchmarks.compiler.inductor_benchmark_artifacts import reference_controls_valid

    reasons = worker_memory_errors(row)
    if row.get("parity_schema") != worker_schema:
        reasons.append("worker_schema")
    if row.get("status") != "completed" or row.get("execution_completed") is not True or row.get("worker_exit_code", 0) != 0:
        reasons.append("worker_incomplete")
    measurement = row.get("measurement", {})
    if measurement.get("valid") is not True or not _number(measurement.get("wall_ms_per_update"), positive=True):
        reasons.append("invalid_measurement")
    if (measurement.get("measured_updates") != row.get("updates") or row.get("updates_completed") != row.get("updates")
            or row.get("warmup_completed") != row.get("warmup")):
        reasons.append("incomplete_updates")
    if measurement.get("native_graph_count_before") != measurement.get("native_graph_count_after"):
        reasons.append("measured_graph_growth")
    if not all(measurement.get("postchecks", {}).get(name) is True for name in ("finite_losses", "finite_parameters")):
        reasons.append("unverified_finite_state")
    if (row.get("route_ownership_verified") is not True or row.get("installed_stable") is not True
            or row.get("helper_source_stable") is not True or row.get("source_stable") is not True):
        reasons.append("ownership_or_source_identity_unverified")
    if not reference_controls_valid(row.get("numerical_controls"), row.get("inductor_preset")):
        reasons.append("backward_RNG_or_arithmetic_controls_unverified")
    optimizer_hash = row.get("provenance", {}).get("optimizer_source_sha256")
    if type(optimizer_hash) is not str or len(optimizer_hash) != 64:
        reasons.append("optimizer_source_unverified")
    expected_owner = ("deepopt_native_autodiff" if row.get("route") == "deepopt_native" else
                      "pytorch_aotautograd" if row.get("route") == "native_torch_inductor" else "pytorch_autograd")
    ownership = row.get("ownership", {})
    if ownership.get("owner") != expected_owner or ownership.get("runtime_fallbacks") != 0:
        reasons.append("wrong_derivative_owner_or_fallback")
    if row.get("compression_enabled") is not False or row.get("external_input_requires_grad") is not False:
        reasons.append("wrong_parity_recipe")
    runtime = row.get("settings", {}).get("runtime", {})
    if (runtime.get("bf16_policy") != "torch_amp" or runtime.get("activation_compression", {}).get("enabled") is not False
            or runtime.get("activation_memory", {}).get("multiply_backward") != "materialized"
            or runtime.get("activation_memory", {}).get("pack_reuse") != "off"
            or runtime.get("activation_memory", {}).get("recompute_modules") != []):
        reasons.append("effective_runtime_recipe_drift")
    for key, value in _identity(row).items():
        if value is None or value == {}:
            reasons.append("missing_identity:" + key)
    if row.get("route") == "native_torch_inductor" and row.get("inductor_validation", {}).get("passed") is not True:
        reasons.append("missing_native_Torch_forward_backward_code")
    if row.get("route") == "deepopt_native" and row.get("native_ownership_verified") is not True:
        reasons.append("native_autodiff_unverified")
    if str(row.get("device", "")).startswith("cuda"):
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if not _number(measurement.get("memory", {}).get(key), positive=True):
                reasons.append("missing_CUDA_peak:" + key)
        if not _number(measurement.get("cuda_event_ms_per_update"), positive=True):
            reasons.append("missing_CUDA_event_time")
        snapshots = row.get("memory_audit", {}).get("snapshots", {})
        if any(snapshots.get(name, {}).get("ooms") != 0 or snapshots.get(name, {}).get("allocation_retries") != 0
               for name in ("before_model", "after_setup", "after_first_update", "after_warmup", "measurement_start", "measurement_end", "after_postchecks")):
            reasons.append("CUDA_OOM_or_allocator_retry")
    return reasons


def _verify_route_artifacts(row, directory):
    from benchmarks.compiler.inductor_benchmark_artifacts import validate_inductor_artifacts
    from examples.vae import benchmark_vae as benchmark

    if row["route"] == "native_torch_inductor":
        result = validate_inductor_artifacts(row.get("inductor_artifacts"), directory)
        if not result["passed"]:
            raise ValueError("Saved native Torch forward/backward evidence did not verify: " + result.get("reason", "unknown"))
        return result
    if row["route"] == "deepopt_native":
        artifact = row["compiler_report_artifact"]
        path = Path(artifact["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Native compiler report escapes its worker output")
        data = path.read_bytes()
        if len(data) != artifact["bytes"] or hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError("Native compiler report checksum mismatch")
        error = benchmark.native_report_error(json.loads(data))
        if error:
            raise ValueError(error)
    return {"passed": True, "scope": "eager_or_native_serialized_ownership"}


def aggregate(results, jobs, trials, *, tolerance_bytes=256 * 2**20):
    from examples.vae.memory_comparison import compare_memory_baselines
    from examples.vae.benchmark_execution import _metrics, _statistics

    summary = {"valid": False, "invalid_reasons": [], "modes": {}, "pairs": [], "ratios": {},
        "target_CUDA_parity_established": False, "performance_policy_promoted": False,
        "correctness_assessed": False, "independent_blocks": trials}
    baseline = compare_memory_baselines(results, tolerance_bytes=tolerance_bytes)
    summary["memory_isolation"] = baseline
    summary["invalid_reasons"] += baseline["invalid_reasons"]
    for job in jobs:
        selected = [row for row in results if row.get("job") == job]
        successful = [row for row in selected if not worker_errors(row)]
        summary["modes"][job] = {"successful_trials": len(successful), "total_trials": len(selected),
            "wall_ms_per_update": _statistics([row["measurement"]["wall_ms_per_update"] for row in successful])}
    anchors = {}
    for block in range(1, trials + 1):
        selected = [row for row in results if row.get("block") == block]
        by_job = {job: [row for row in selected if row.get("job") == job] for job in jobs}
        if len(selected) != len(jobs) or any(len(rows) != 1 for rows in by_job.values()):
            summary["invalid_reasons"].append(f"block_{block}:missing_or_duplicate_workers")
            continue
        rows = {job: value[0] for job, value in by_job.items()}
        for job, row in rows.items():
            summary["invalid_reasons"] += [f"{block}:{job}:" + reason for reason in worker_errors(row)]
            current = _identity(row)
            if job in anchors and current != anchors[job]:
                summary["invalid_reasons"].append(f"{block}:{job}:identity_drift_across_blocks")
            anchors.setdefault(job, current)
        reference = _identity(next(iter(rows.values())))
        for job, row in rows.items():
            if _identity(row) != reference:
                summary["invalid_reasons"].append(f"{block}:{job}:matched_recipe_state_input_rng_or_software_drift")
        native = rows.get("deepopt_native:cast-preserving")
        if native is not None:
            for job, reference_row in rows.items():
                if not job.startswith("native_torch_inductor:"):
                    continue
                reasons = worker_errors(native) + worker_errors(reference_row)
                if _identity(native) != _identity(reference_row):
                    reasons.append("pair_identity_drift")
                pair = {"block": block, "inductor_preset": reference_row["inductor_preset"], "valid": not reasons, "invalid_reasons": reasons, "ratios": {}}
                if not reasons:
                    nm, im = _metrics(native), _metrics(reference_row)
                    pair["ratios"]["native_over_inductor_wall"] = nm["wall_ms_per_update"] / im["wall_ms_per_update"]
                    if nm["steady_peak_allocated_bytes"] is not None and im["steady_peak_allocated_bytes"]:
                        pair["ratios"]["native_over_inductor_allocated_peak"] = nm["steady_peak_allocated_bytes"] / im["steady_peak_allocated_bytes"]
                summary["pairs"].append(pair)
    summary["valid"] = bool(results) and not summary["invalid_reasons"]
    if summary["valid"]:
        for preset in ("default", "cast-preserving"):
            pairs = [pair for pair in summary["pairs"] if pair["inductor_preset"] == preset]
            if pairs:
                keys = set.intersection(*(set(pair["ratios"]) for pair in pairs))
                summary["ratios"][preset] = {key: _statistics([pair["ratios"][key] for pair in pairs]) for key in sorted(keys)}
        target = trials >= 5 and all(row.get("target_CUDA_execution") is True for row in results)
        summary["five_pair_ratio_gate_passed"] = target and all(
            value.get("native_over_inductor_wall", {}).get("median", float("inf")) <= 1.05
            and value.get("native_over_inductor_allocated_peak", {}).get("median", float("inf")) <= 1
            for value in summary["ratios"].values()) and len(summary["ratios"]) == 2
        summary["parity_decision_requires_review"] = True
    return summary


def run_benchmark(args):
    from examples.vae import benchmark_vae as benchmark, parity_execution as worker
    from examples.vae.benchmark_execution import _reference

    plan = benchmark_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh parity cohort output")
    directory.mkdir(parents=True)
    report = {**plan, "status": "running", "results": [], "started_at_utc": datetime.now(timezone.utc).isoformat()}
    benchmark.write_json(output, report)
    results = []
    try:
        from benchmarks.compiler.installed_runtime import installed_identity, child_environment

        identity = installed_identity(ROOT, full=True)
        expected = (plan["release_pin"]["manifest"]["expected_installed"] if plan["release_pin"] is not None else
                    {"version": worker.EXPECTED_VERSION})
        if any(identity.get(name) != value for name, value in expected.items()):
            raise RuntimeError("Installed parity release differs from its version/source pin")
        environment = child_environment(ROOT, identity=identity)
        environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
        environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"
        for block, order in enumerate(plan["order"], 1):
            for job in order:
                route, preset = job.split(":")
                path = directory / f"block-{block}-{route}-{preset}.json"
                log = path.with_suffix(".log")
                pin = path.with_suffix(".arm.json")
                options = plan["workers"][job]["native_options_override"]
                pin.write_text(json.dumps({"schema_version": 1, "arm": route,
                    "expected_installed": {name: identity[name] for name in ("version", "source_sha256", "files_hashed")},
                    "wheel_sha256": (plan["release_pin"]["manifest"].get("wheel_sha256") if plan["release_pin"] else None),
                    "native_options_sha256": options["sha256"] if options is not None else None}, indent=2) + "\n")
                command = worker_command(args, route, preset, path, arm_manifest=pin)
                code = None
                with log.open("x") as stream:
                    try:
                        code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
                    except OSError as error:
                        stream.write(str(error) + "\n")
                row = None
                try:
                    from examples.vae.benchmark_execution import _reject_json_constant
                    row = json.loads(path.read_text(), parse_constant=_reject_json_constant)
                    if not isinstance(row, dict):
                        raise ValueError("Worker result must be an object")
                    worker_errors(row)
                    if row.get("route") != route or row.get("inductor_preset") != preset or row.get("installed") != identity:
                        raise ValueError("Worker route/preset/installation differs from its declared job")
                    if row.get("common_recipe_sha256") != plan["workers"][job]["common_recipe_sha256"] or row.get("origin") != plan["origin"]:
                        raise ValueError("Worker recipe or original receipt differs from planning")
                    if row.get("status") == "completed":
                        row["supervisor_artifact_validation"] = _verify_route_artifacts(row, path.parent / (path.stem + "-artifacts"))
                except Exception as error:
                    row = {"status": "failed", "error": (row.get("error") if isinstance(row, dict) else None) or str(error), "route": route, "inductor_preset": preset}
                row.update(job=job, block=block, worker_exit_code=code)
                results.append(row)
                receipt = {"job": job, "block": block, "status": row["status"], "worker_exit_code": code,
                    "error": row.get("error"), "measurement": row.get("measurement"), "log": _reference(log, output.parent)}
                if path.is_file():
                    receipt["result"] = _reference(path, output.parent)
                report["results"].append(receipt)
                benchmark.write_json(output, report)
        if worker.source_identity() != plan["source_sha256"] or installed_identity(ROOT, full=True) != identity:
            raise RuntimeError("Parity supervisor/helper/package sources changed")
        report["summary"] = aggregate(results, plan["workers"], args.trials, tolerance_bytes=plan["baseline_tolerance_bytes"])
        report["status"] = "completed" if report["summary"]["valid"] else "failed"
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
        report["summary"] = {"valid": False, "invalid_reasons": ["supervisor_failed"], "ratios": {}, "target_CUDA_parity_established": False}
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    print(json.dumps({"status": report["status"], "summary": report["summary"], "output": str(output)}, indent=2))
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    return run_benchmark(parser().parse_args(argv))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
