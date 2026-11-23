"""Run matched BF16/compression compiler ablations through shared workers.

Copy compiler/ and analysis/ together when exporting these benchmarks. --dry-run
uses only the standard library and writes independently reviewable run plans.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import uuid


def write_json(path, value):
    """Publish a complete checkpoint, including when an older suite exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _analysis_run_id(directory):
    identities = {_read_json(path).get("run_id") for name in ("summary.json", "progress.json")
                  if (path := Path(directory) / name).exists()}
    if len(identities) != 1 or not all(isinstance(value, str) and value for value in identities):
        raise ValueError("Missing or inconsistent analysis run identity")
    return identities.pop()


def _bf16_policy_receipt(row):
    from analysis.matrix import bf16_policy_identity, numerical_recipe_identity

    task = row["task"]
    runtime = task.get("settings", {}).get("runtime", {})
    expected = bf16_policy_identity({"runtime": runtime}, task["case"])
    requested = {"name": runtime.get("bf16_policy", "torch_amp"), "version": 1}
    receipt = {"policy": expected, "requested": requested}
    execution, metadata = row.get("execution", {}), row.get("numerical_snapshot", {})
    if not isinstance(execution, dict) or not isinstance(metadata, dict):
        return {**receipt, "passed": False, "reason": "malformed_policy_receipt"}
    # Older scalar fixtures without a declared policy remain AMP references.
    # Every current plan declares the policy and requires all three receipts.
    observed = (row.get("bf16_policy"), execution.get("bf16_policy"), metadata.get("bf16_policy"))
    recipe = numerical_recipe_identity(task.get("settings", {}), task["case"])
    recipes = (row.get("numerical_recipe"), execution.get("numerical_recipe"), metadata.get("numerical_recipe"))
    recipe_required = expected["name"] == "bf16_storage" or "numerical_recipe" in task
    recipe_passed = (all(json.dumps(value, sort_keys=True) == json.dumps(recipe, sort_keys=True) for value in recipes)
                     if recipe_required else True)
    if "bf16_policy" not in runtime and all(value is None for value in observed):
        return {**receipt, "passed": True, "legacy_implicit_torch_amp": True}
    def canonical(value, selected):
        return (isinstance(value, dict) and value.keys() == selected.keys()
                and type(value.get("version")) is int and value == selected)

    return {**receipt, "passed": recipe_passed and all(canonical(value, expected) for value in observed)
            and canonical(row.get("requested_bf16_policy"), requested)
            and canonical(execution.get("requested_bf16_policy"), requested)
            and execution.get("native_torch_ops_only") is (task["case"] == "torch_bf16"),
            "observed": list(observed), "numerical_recipe": recipe, "recipe_passed": recipe_passed}


@contextmanager
def numerical_job(job, suite_run_id, comparisons):
    """Receive concurrently with nested fresh workers; retain no tensor files."""
    from analysis.numerical_ipc import NumericalSnapshotBroker, IdleTimeout

    context = comparisons.start_job(job, suite_run_id)
    stopped = threading.Event()
    try:
        with NumericalSnapshotBroker(timeout=600, idle_timeout=1) as broker:
            environment = {**broker.environment, "DEEPOPT_NUMERICAL_CONTEXT": json.dumps(context),
                           "DEEPOPT_BENCHMARK_TELEMETRY_INTERVAL": str(job.get("telemetry_interval", 0)),
                           "DEEPOPT_INDUCTOR_BENCHMARK_DUMPS": "1" if job["strategy"] == "torch-inductor" else "0"}
            previous = {name: os.environ.get(name) for name in environment}

            def receive():
                while not stopped.is_set():
                    try:
                        broker.receive(on_snapshot=comparisons.accept)
                    except IdleTimeout:
                        continue
                    except Exception as error:
                        if not stopped.is_set():
                            comparisons.fail(error)
                        return

            receiver = threading.Thread(target=receive, name="model-numerical-receiver", daemon=True)
            try:
                os.environ.update(environment)
                receiver.start()
                yield
            finally:
                stopped.set()
                broker.close()
                if receiver.ident is not None:
                    receiver.join(timeout=10)
                if receiver.is_alive():
                    comparisons.fail(RuntimeError("Numerical receiver did not stop"))
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
    finally:
        comparisons.end_job()


def compare_strategies(jobs, *, cuda=False, require_native=False, numerical_comparisons=None):
    """Qualify the declared matrix, never just the result files that survived."""
    from analysis.artifacts import profiles_valid
    from analysis.measurement import fusion_candidate_evidence

    rows, eager, comparisons, identities = [], {}, [], set()
    for job in jobs:
        base = {"trial": job["trial"], "strategy": job["strategy"], "passed": False}
        identity = job["trial"], job["strategy"]
        if identity in identities:
            comparisons.append({**base, "reason": "duplicate_job"})
            continue
        identities.add(identity)
        if job.get("status", "completed") != "completed" or job.get("result", 0) != 0:
            comparisons.append({**base, "reason": "job_incomplete_or_failed", "job_status": job.get("status"),
                                "job_result": job.get("result"), "execution_error": job.get("error")})
        directory = Path(job["directory"])
        try:
            plan = _read_json(directory / "plan.json")
            tasks = job.get("expected_tasks", plan["tasks"])
            if not tasks or tasks != plan["tasks"] or (job.get("plan_code_sha256") is not None
                                                       and job["plan_code_sha256"] != plan.get("code_sha256")):
                raise ValueError("Saved analysis plan does not match the declared job")
            expected = {task["id"]: task for task in tasks}
            if len(expected) != len(tasks):
                raise ValueError("Duplicate task identities in analysis plan")
            run_id = _analysis_run_id(directory)
            if job.get("analysis_run_id", run_id) != run_id:
                raise ValueError("Saved analysis run does not match the declared job")
        except (OSError, ValueError, KeyError, TypeError) as error:
            comparisons.append({**base, "reason": "invalid_job_identity", "error": f"{type(error).__name__}: {error}"})
            continue
        for path in (directory / "cases").glob("*/result.json"):
            if path.parent.name not in expected:
                comparisons.append({**base, "reason": "unexpected_result", "result_file": str(path)})
        for task_id, task in expected.items():
            key = job["trial"], task["workload"]["name"], task["case"]
            path = directory / "cases" / task_id / "result.json"
            try:
                row = _read_json(path)
            except (OSError, ValueError) as error:
                comparisons.append({**base, "key": key, "reason": "missing_or_invalid_result",
                                    "error": f"{type(error).__name__}: {error}"})
                continue
            if row.get("task") != task or row.get("run_id") != run_id:
                comparisons.append({**base, "key": key, "reason": "stale_or_mismatched_result"})
                continue
            profile_options = dict(plan.get("options", {}))
            if "profile_steps" in job:
                profile_options["profile_steps"] = job["profile_steps"]
            if row.get("status") == "ok" and (row.get("artifact_errors") or not profiles_valid(row, task, profile_options)):
                comparisons.append({**base, "key": key, "reason": "invalid_profile_artifacts"})
                continue
            row["trial"], row["strategy"] = job["trial"], job["strategy"]
            row["_job_execution_valid"] = job.get("status", "completed") == "completed" and job.get("result", 0) == 0
            row["_profile_required"] = job.get("profile_steps", plan.get("options", {}).get("profile_steps", 0)) > 0
            rows.append(row)
            if job["strategy"] == "eager" and row.get("status") == "ok":
                eager[key] = row
    for row in rows:
        if row.get("status") != "ok":
            comparisons.append({"key": (row["trial"], row["task"]["workload"]["name"], row["task"]["case"]),
                "strategy": row["strategy"], "passed": False, "reason": "execution_error",
                "execution_error": row.get("reason")})
            continue
        if row["strategy"] == "eager":
            policy = _bf16_policy_receipt(row)
            if not policy["passed"]:
                comparisons.append({"key": (row["trial"], row["task"]["workload"]["name"], row["task"]["case"]),
                    "strategy": "eager", "passed": False, "reason": "invalid_bf16_policy_receipt", "bf16_policy": policy})
            evidence = (numerical_comparisons.validate_row(row) if numerical_comparisons is not None
                        else {"passed": False, "reason": "missing_live_numerical_comparison"})
            if not evidence["passed"] or row.get("finite_after_timing") is not True:
                comparisons.append({"key": (row["trial"], row["task"]["workload"]["name"], row["task"]["case"]),
                    "strategy": "eager", "passed": False, "reason": "missing_or_invalid_snapshot", "numerical": evidence})
            value = row.get("timing", {}).get("wall_ms", {}).get("median")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                comparisons.append({"key": (row["trial"], row["task"]["workload"]["name"], row["task"]["case"]),
                    "strategy": "eager", "passed": False, "reason": "invalid_timing"})
            continue
        key = row["trial"], row["task"]["workload"]["name"], row["task"]["case"]
        reference = eager.get(key)
        if reference is None:
            comparisons.append({"key": key, "strategy": row["strategy"], "passed": False, "reason": "missing_eager_oracle"})
            continue
        try:
            timings = [r["timing"]["wall_ms"]["median"] for r in (row, reference)]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                   for value in timings):
                raise ValueError("Wall timing medians must be finite and positive")
        except (KeyError, TypeError, ValueError) as error:
            comparisons.append({"key": key, "strategy": row["strategy"], "passed": False,
                                "reason": "invalid_timing", "error": str(error)})
            continue
        same = all(isinstance(row.get(k), str) and bool(row[k]) and row[k] == reference.get(k)
                   for k in ("initial_state_sha256", "rng_baseline_sha256", "shared_optimizer_baseline_sha256"))
        build = row.get("environment", {}).get("deepopt_source_sha256")
        same_build = isinstance(build, str) and bool(build) and build == reference.get("environment", {}).get("deepopt_source_sha256")
        same_measurement = bool(row.get("measurement_policy")) and row.get("measurement_policy") == reference.get("measurement_policy")
        if numerical_comparisons is None:
            comparisons.append({"key": key, "strategy": row["strategy"], "passed": False,
                                "reason": "missing_or_invalid_snapshot", "error": "A live numerical comparison is required"})
            continue
        numerical = numerical_comparisons.compare_rows(row, reference, rtol=0.02, atol=0.002, max_relative_l2=0.1)
        policy, reference_policy = _bf16_policy_receipt(row), _bf16_policy_receipt(reference)
        same_bf16_policy = (policy["passed"] and reference_policy["passed"]
                            and policy["policy"] == reference_policy["policy"])
        numerical["scope"] = "same_recipe_compiler_qualification"
        numerical["full_tensor_finite"] = all(r.get("numerical_snapshot", {}).get("full_tensor_finite") is True for r in (row, reference))
        numerical["finite_after_timing"] = all(r.get("finite_after_timing") is True for r in (row, reference))
        numerical["finite"] &= numerical["full_tensor_finite"] and numerical["finite_after_timing"]
        numerical["passed"] &= numerical["finite"] and numerical["same_output_and_gradient_shapes"]
        report = row.get("execution", {}).get("compiler_report") or {}
        inductor = row["strategy"] == "torch-inductor"
        inductor_artifacts = None
        if inductor:
            from compiler.inductor_benchmark_artifacts import validate_inductor_artifacts

            inductor_artifacts = validate_inductor_artifacts(row.get("inductor_artifacts"),
                Path(next(job["directory"] for job in jobs
                          if job["trial"] == row["trial"] and job["strategy"] == "torch-inductor")) / "cases" / row["task"]["id"])
            native_task = (row["task"]["case"] == "torch_bf16"
                           and row["task"].get("settings", {}).get("runtime", {}).get("compile", {}).get("backend") == "inductor")
            coverage = (native_task and inductor_artifacts.get("passed") is True
                        and (not cuda or inductor_artifacts.get("cuda_execution_observed") is True))
        else:
            coverage = report.get("static_regions", {}).get("bf16_linear_programs", 0) > 0
        native_coverage = (report.get("native_autodiff_operations", 0) > 0 and not report.get("legacy_backward_operations")
                           and report.get("runtime_fallbacks", 0) == 0
                           and all(not graph.get("eager_autograd_operations") for graph in report.get("graphs", [])))
        native_required = require_native and not inductor and "legacy-compiler" not in row["strategy"]
        positive = (cuda and "fused" in row["strategy"] and "unfused" not in row["strategy"]
                    or cuda and "packed-read" in row["strategy"])
        positive &= row["task"]["workload"]["factory_kwargs"]["kind"] == "mlp"
        profile = row.get("profile")
        fusion_evidence = fusion_candidate_evidence(report, profile)
        fusion = not positive or (report.get("fused_deepopt_regions", 0) > 0
                                   and fusion_evidence["selected_candidates"] > 0)
        observed = not positive or fusion_evidence["all_selected_observed"]
        comparisons.append({"key": key, "strategy": row["strategy"],
            "passed": (row["_job_execution_valid"] and reference["_job_execution_valid"]
                       and same and same_build and same_measurement and same_bf16_policy
                       and numerical["passed"] and coverage and fusion and observed
                       and (not native_required or native_coverage)),
            "same_initial_state": same, "same_installed_build": same_build, "same_measurement_policy": same_measurement, "numerical": numerical,
            "same_bf16_policy": same_bf16_policy, "bf16_policy": policy,
            "reference_bf16_policy": reference_policy,
            "compiler_coverage": coverage, "fusion_required": positive, "expected_fusion_selected": fusion,
            "fusion_execution_observed": (None if fusion_evidence["status"] == "unknown"
                                          else fusion_evidence["all_selected_observed"]),
            "fusion_qualification": fusion_evidence,
            "native_autodiff_required": native_required, "native_autodiff_coverage": native_coverage,
            "runtime_reference": "torch_inductor_native_ops" if inductor else "deepopt_compiler",
            "inductor_artifacts": inductor_artifacts,
            "wall_speedup_vs_eager_same_recipe": timings[1] / timings[0],
            "peak_extra_allocated_bytes": (row.get("memory") or {}).get("peak_extra_allocated_bytes"),
            "gpu_performance_qualified": False})
    # Native Inductor is a runtime reference for every route. Compression quality
    # remains governed by the matching eager recipe above, never by this ratio.
    inductor_times = {(row["trial"], row["task"]["workload"]["name"]): row for row in rows
                      if row["strategy"] == "torch-inductor" and row.get("status") == "ok"}
    for comparison in comparisons:
        key = comparison.get("key")
        if not key or comparison.get("strategy") == "torch-inductor":
            continue
        reference = inductor_times.get((key[0], key[1]))
        if reference is None:
            continue
        candidate = next((row for row in rows if row["trial"] == key[0]
                          and row["strategy"] == comparison["strategy"]
                          and row["task"]["workload"]["name"] == key[1] and row["task"]["case"] == key[2]), None)
        if candidate is None:
            continue
        try:
            native_ms = reference["timing"]["wall_ms"]["median"]
            candidate_ms = candidate["timing"]["wall_ms"]["median"]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                   for value in (native_ms, candidate_ms)):
                continue
        except (KeyError, TypeError):
            continue
        comparison["native_inductor_runtime_reference"] = {
            "native_inductor_wall_ms": native_ms, "candidate_wall_ms": candidate_ms,
            "native_inductor_time_over_candidate_time": native_ms / candidate_ms,
            "reference_case": "torch_bf16", "candidate_case": key[2],
            "same_recipe": key[2] == "torch_bf16", "scope": "descriptive_runtime_reference",
            "reference_bf16_policy": _bf16_policy_receipt(reference)["policy"],
            "candidate_bf16_policy": _bf16_policy_receipt(candidate)["policy"],
            "same_bf16_policy": (_bf16_policy_receipt(reference)["policy"]
                                  == _bf16_policy_receipt(candidate)["policy"]),
            "reference_accepted": any(item.get("strategy") == "torch-inductor" and item.get("key") == (key[0], key[1], "torch_bf16")
                                      and item.get("passed") is True for item in comparisons),
            "artifacts": reference.get("inductor_artifacts"),
        }
    return {"comparisons": comparisons, "passed": bool(comparisons) and all(row["passed"] for row in comparisons),
            "numerical_policy": {"qualification": "compiled_vs_matching_eager_recipe",
                                 "cross_bf16_policy_comparisons": "descriptive_only",
                                 "uncompressed_recipe_comparisons": "descriptive_quantization_quality_diagnostics"},
            "note": "Positive fusion fixtures require candidate-specific CUDA ranges and all expected kernel roles; missing attribution is unqualified. Speedups need repeated target-GPU qualification; CPU timing never qualifies a GPU schedule."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", choices=("linear", "mlp", "vit-block", "vit-base"), default=["linear", "mlp"])
    parser.add_argument("--batches", type=int, nargs="+", default=[8])
    parser.add_argument("--backends", nargs="+", choices=("triton", "tilelang"), default=["triton"])
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--tokens", type=int, default=197)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--telemetry-interval", type=float, default=0,
                        help="Optional per-worker diagnostics outside throughput; 0 disables, minimum one second")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-native-autodiff", action="store_true")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--no-inductor-reference", action="store_true",
                        help="Explicitly omit the native TorchInductor runtime/code reference; DeepOpt cases are retained")
    parser.add_argument("--inductor-arithmetic-preset", choices=("default", "cast-preserving"), default="default")
    parser.add_argument("--bf16-policy", choices=("torch_amp", "bf16_storage"), default="torch_amp",
                        help="DeepOpt BF16 recipe; raw Torch/Inductor controls always retain torch_amp")
    args = parser.parse_args(argv)
    if min(*args.batches, args.width, args.hidden, args.tokens, args.iterations, args.repeats, args.trials) < 1 or min(args.warmup, args.profile_steps) < 0:
        parser.error("dimensions, iterations, repeats and trials must be positive; warmup and profile steps must be nonnegative")
    if any(len(values) != len(set(values)) for values in (args.models, args.batches, args.backends)):
        parser.error("models, batches and backends must not contain duplicates")
    if not math.isfinite(args.telemetry_interval) or args.telemetry_interval < 0 or 0 < args.telemetry_interval < 1:
        parser.error("telemetry interval must be zero or at least one second")
    args.output = args.output.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.scheduler import output_lock

    installed = None
    if not args.dry_run:
        from compiler.installed_runtime import installed_identity, child_environment

        root = Path(__file__).resolve().parents[2]
        installed = installed_identity(root)
        os.environ.update(child_environment(root, identity=installed))
    with output_lock(args.output):
        if installed is None:
            result = _run_suite(args)
        else:
            from compiler.installed_runtime import identity_session

            with identity_session(root):
                result = _run_suite(args)
    return result


def _run_suite(args):
    from analysis.benchmark_do_ops import main as run_analysis, make_plan, parser as analysis_parser

    strategies = [("eager", "eager", {})]
    if not args.no_inductor_reference:
        strategies.append(("torch-inductor", "compile", {}))
    for backend in args.backends:
        if args.include_legacy:
            strategies.append((f"{backend}-legacy-compiler", "compile", {"kernel_backend": backend,
                "use_native_autodiff": False, "require_full_aot": False, "bf16_gemm_backend": "generated",
                "fuse_deepopt_activations": True}))
        for name, options in (
            ("native-control", {"bf16_gemm_backend": "native"}),
            ("generated-unfused", {"bf16_gemm_backend": "generated"}),
            ("generated-fused", {"bf16_gemm_backend": "generated", "fuse_deepopt_activations": True}),
            ("generated-packed-read", {"bf16_gemm_backend": "generated", "bf16_decode": "tiled", "fuse_deepopt_activations": True}),
            ("generated-dw-first", {"bf16_gemm_backend": "generated", "bf16_backward_order": "dw_first", "fuse_deepopt_activations": True}),
        ):
            strategies.append((f"{backend}-{name}", "compile", {"kernel_backend": backend,
                "use_native_autodiff": True, "require_full_aot": args.require_native_autodiff, **options}))
    workloads = [{"name": f"{model}-b{batch}", "op": "model", "shape": [batch, args.tokens, args.width],
                  "factory": "analysis.compiler_workloads:build", "factory_kwargs": {
                      "kind": model, "batch": batch, "tokens": args.tokens, "width": args.width,
                      "hidden": args.hidden, "output": args.width}}
                 for model in args.models for batch in args.batches]
    args.output.mkdir(parents=True, exist_ok=True)
    # One trial's eager samples bound retention. Candidate order remains
    # randomized, but the matching eager oracle must arrive before candidates.
    jobs = []
    for trial in range(args.trials):
        candidates = list(strategies[1:])
        random.Random(714 + trial).shuffle(candidates)
        jobs.extend((trial, strategy) for strategy in [strategies[0], *candidates])
    suite_run_id = uuid.uuid4().hex
    manifest = {"schema_version": 2, "suite_run_id": suite_run_id, "status": "planning", "jobs": [],
                "bf16_policy": {"name": getattr(args, "bf16_policy", "torch_amp"), "version": 1},
                "numerical_policy": {"transport": "memory_only", "max_eager_retained_bytes": 512 * 1024 * 1024,
                    "qualification": "compiled_vs_matching_eager_recipe",
                    "cross_bf16_policy_comparisons": "descriptive_only",
                    "uncompressed_recipe_comparisons": "descriptive_quantization_quality_diagnostics"},
                "matrix": [{"trial": trial, "strategy": strategy[0]} for trial, strategy in jobs],
                "inductor_reference": {"enabled": not args.no_inductor_reference, "case": "torch_bf16",
                    "bf16_policy": {"name": "torch_amp", "version": 1},
                    "arithmetic_preset": args.inductor_arithmetic_preset,
                    "native_torch_ops_only": True, "deepopt_custom_ops": False,
                    "compiled_code_location": "per-case output/inductor/", "optimizer": "same eager optimizer as other strategies"},
                "comparison": str(args.output / "comparison.json"),
                "note": "Per-job artifacts belong to this invocation's unique runs directory. Compare matching initial-state/recipe fingerprints; compressed dW is intentionally approximate."}
    manifest_path = args.output / "manifest.json"
    comparison_path = args.output / "comparison.json"
    comparison = {"suite_run_id": suite_run_id, "status": "not_requested" if args.dry_run else "pending",
                  "passed": False, "comparisons": []}
    # Invalidate old qualification before planning can be interrupted or fail.
    write_json(comparison_path, comparison)
    write_json(manifest_path, manifest)
    try:
        _plan_jobs(args, jobs, workloads, manifest, make_plan, analysis_parser)
    except BaseException as error:
        interrupted = not isinstance(error, Exception) or isinstance(error, InterruptedError)
        message = f"{type(error).__name__}: {error}"
        manifest.update(status="interrupted" if interrupted else "failed", planning_error=message)
        write_json(manifest_path, manifest)
        write_json(comparison_path, {**comparison, "status": manifest["status"], "reason": "planning_error", "error": message})
        if interrupted:
            raise
        return 1

    manifest["status"] = "planned" if args.dry_run else "running"
    write_json(manifest_path, manifest)
    if args.dry_run:
        for job in manifest["jobs"]:
            job.update(status="planned", result=0)
        manifest["status"] = "dry_run"
        write_json(manifest_path, manifest)
        return 0

    from compiler.model_numerics import ModelNumericalComparisons

    numerical_comparisons = ModelNumericalComparisons(max_retained_bytes=512 * 1024 * 1024)
    failed = False
    active_trial = None
    for job in manifest["jobs"]:
        if active_trial != job["trial"]:
            numerical_comparisons.clear_trial()
            active_trial = job["trial"]
        job["status"] = "running"
        write_json(manifest_path, manifest)
        try:
            with numerical_job(job, suite_run_id, numerical_comparisons):
                job["result"] = int(run_analysis(job["command"], numerical_comparisons=numerical_comparisons))
            job["status"] = "failed" if job["result"] else "completed"
        except BaseException as error:
            if isinstance(error, Exception) and not isinstance(error, InterruptedError):
                job.update(status="failed", result=1, error=f"{type(error).__name__}: {error}")
            else:
                job.update(status="interrupted", error=f"{type(error).__name__}: {error}")
                manifest.update(status="interrupted", interrupted_job={"trial": job["trial"], "strategy": job["strategy"]})
                write_json(manifest_path, manifest)
                write_json(comparison_path, {**comparison, "status": "interrupted"})
                raise
        try:
            job["analysis_run_id"] = _analysis_run_id(job["directory"])
        except (OSError, ValueError, TypeError) as error:
            job.update(status="failed", result=job["result"] or 1)
            job.setdefault("error", f"{type(error).__name__}: {error}")
        failed |= job["status"] != "completed"
        write_json(manifest_path, manifest)
    try:
        numerical_comparisons.clear_trial()
        comparison = compare_strategies(manifest["jobs"], cuda=args.device.startswith("cuda"),
                                        require_native=args.require_native_autodiff,
                                        numerical_comparisons=numerical_comparisons)
        comparison.update(suite_run_id=suite_run_id, status="completed")
    except BaseException as error:
        if isinstance(error, Exception) and not isinstance(error, InterruptedError):
            comparison = {"suite_run_id": suite_run_id, "status": "failed", "passed": False, "comparisons": [],
                          "reason": "comparison_error", "error": f"{type(error).__name__}: {error}"}
        else:
            manifest.update(status="interrupted", comparison_error=f"{type(error).__name__}: {error}")
            write_json(manifest_path, manifest)
            write_json(comparison_path, {**comparison, "status": "interrupted"})
            raise
    write_json(comparison_path, comparison)
    failed |= not comparison["passed"]
    manifest.update(status="failed" if failed else "completed", comparison_passed=comparison["passed"])
    write_json(manifest_path, manifest)
    return int(failed)


def _plan_jobs(args, jobs, workloads, manifest, make_plan, analysis_parser):
    for trial, (name, execution, options) in jobs:
        inductor = name == "torch-inductor"
        suite_run_id = manifest["suite_run_id"]
        directory = args.output / "runs" / suite_run_id / f"trial-{trial}" / name
        directory.mkdir(parents=True, exist_ok=True)
        configuration = {
            "settings": {"compression": {"group_size": 1024}, "runtime": {"execution": execution, "binding": "prepared", "optimizer": "sgd",
                "bf16_policy": "torch_amp" if inductor else getattr(args, "bf16_policy", "torch_amp"),
                "optimizer_warmup_steps": 3, "check_finite": False,
                "compile": {"backend": "inductor" if inductor else "deepopt", "dynamic": False,
                    "fullgraph": not inductor and args.require_native_autodiff and "legacy-compiler" not in name,
                    "options": {"emulate_precision_casts": args.inductor_arithmetic_preset == "cast-preserving"}
                    if inductor else options}}},
            "workloads": deepcopy(workloads),
        }
        config_path = directory / "config.json"
        write_json(config_path, configuration)
        cases = (["torch_bf16"] if inductor else
                 ["torch_bf16", "do_bf16", "compression"] if execution == "eager" else ["do_bf16", "compression"])
        command = ["--config", str(config_path), "--device", args.device,
                   "--cases", *cases, "--scheduler", "serial",
                   "--numerical-snapshot-mode", "memory",
                   "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                   "--repeats", str(args.repeats), "--profile-steps", str(args.profile_steps),
                   "--output", str(directory / "results")]
        # The shared planner is standard-library-only. Declare exact normalized
        # tasks before any worker, CUDA query, or analysis invocation can fail.
        plan = make_plan(analysis_parser().parse_args(command))
        write_json(directory / "results" / "plan.json", plan)
        manifest["jobs"].append({"trial": trial, "strategy": name, "status": "pending", "result": None,
            "bf16_policy": {"name": configuration["settings"]["runtime"]["bf16_policy"], "version": 1},
            "telemetry_interval": args.telemetry_interval,
            "configuration": str(config_path), "directory": str(directory / "results"), "command": command,
            "expected_tasks": plan["tasks"], "plan_code_sha256": plan["code_sha256"], "profile_steps": args.profile_steps})
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
