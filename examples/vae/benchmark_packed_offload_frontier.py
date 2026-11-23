"""Run immutable physical-capacity points on one installed candidate.

Host capacity always has an explicit ceiling. A feasible arena upper bound is
not a minimum: all-eligible planning is still attempted below that bound when
the logical lower bound and consumer staging floors fit. Every point uses new
resident/demand/predictive processes; no numerical oracle or tensor export.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from examples.vae import benchmark_packed_offload as offload


def parser():
    result = offload.parser()
    result.description = __doc__
    result.set_defaults(output=Path("work/vae_packed_offload_frontier.json"))
    result.add_argument("--max-host-pinned-mib", type=int, required=True,
                        help="Explicit safe physical pinned ceiling; never inferred as all host RAM")
    result.add_argument("--host-pinned-mib", type=int, nargs="+",
                        help="Intermediate physical budgets; default ordinary16GiB capped by the ceiling")
    result.add_argument("--inventory-json", type=Path, help="Same-candidate compact native inventory; otherwise bootstrap untimed demand")
    return result


def _write(path, value):
    from examples.vae.benchmark_vae import write_json

    write_json(path, value)


def _ref(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(2**20), b""):
            digest.update(data)
            size += len(data)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _compact_context(value):
    if not isinstance(value, dict):
        return None
    return {key[:128]: item[:1024] if isinstance(item, str) else item
        for key, item in islice(value.items(), 16) if isinstance(key, str)
        and (item is None or type(item) in (str, bool, int))}


def _bootstrap_failure(report, phase, message):
    """Keep the child primary failure; receipt diagnostics are secondary."""
    message = str(message)[:4096]
    report.setdefault("receipt_diagnostics", []).append({"phase": phase, "error": message})
    if not report.get("error"):
        report["error"] = message
    if not report.get("failure_context"):
        report["failure_context"] = {"stage": "inventory_bootstrap", "phase": phase}
    return None, report


def _base(args):
    if args.configuration_preflight_only:
        raise ValueError("Capacity frontier uses exact per-point preflight automatically; use --dry-run for planning")
    if type(args.max_host_pinned_mib) is not int or args.max_host_pinned_mib < 1:
        raise ValueError("max-host-pinned-mib must be a positive explicit capacity ceiling")
    budgets = args.host_pinned_mib or [min(16384, args.max_host_pinned_mib)]
    if any(type(value) is not int or value < 1 or value > args.max_host_pinned_mib for value in budgets):
        raise ValueError("Intermediate host budgets must be positive and within the explicit ceiling")
    plan = offload.benchmark_plan(args)
    if any(policy.get("version", 1) != 2 for policy in plan["manifest"]["policies"].values()):
        raise ValueError("Physical-capacity frontier requires explicit version2 policies; legacy version1 retains logical semantics")
    plan.update(scope="same_candidate_physical_pinned_capacity_frontier", ceiling_bytes=args.max_host_pinned_mib * 2**20,
        intermediate_bytes=sorted({value * 2**20 for value in budgets} | {args.max_host_pinned_mib * 2**20}),
        inventory_required=True, inventory_source="provided" if args.inventory_json else "fresh_untimed_demand_bootstrap",
        no_automatic_all_RAM_pinning=True, performance_policy_promoted=False)
    return plan


def _validate_inventory(inventory, plan):
    from examples.vae.execution_inventory import comparison_options

    identity = {key: plan["installed"][key] for key in ("version", "source_sha256", "files_hashed")}
    if (inventory.get("schema_version") != 1 or inventory.get("status") != "available"
            or inventory.get("installed") != identity or inventory.get("origin") != plan["origin"]
            or inventory.get("requested_input_shape") != plan["workers"]["demand_offload"]["requested_input_shape"]
            or inventory.get("comparison_native_options_sha256") != _digest(comparison_options(plan["workers"]["demand_offload"]["settings"]))):
        raise ValueError("Inventory differs from this candidate, workload or native math options")
    for key in ("eligible_history_bytes", "all_eligible_host_physical_capacity_bytes", "required_staging_bytes", "required_d2h_inflight_bytes"):
        if type(inventory.get(key)) is not int or inventory[key] < 1:
            raise ValueError("Inventory lacks a supported positive capacity/floor: " + key)
    if inventory.get("all_eligible_capacity_is_feasible_upper_bound_not_minimum") is not True:
        raise ValueError("Inventory must label its feasible arena upper bound, not claim a minimum")


def capacity_points(plan, inventory):
    _validate_inventory(inventory, plan)
    points = [{"name": "peak-" + str(value // 2**20) + "MiB", "host_pinned_bytes": value,
        "selection_mode": "peak", "preflight_status": "attempt"} for value in plan["intermediate_bytes"]]
    policy = plan["manifest"]["policies"]["demand_offload"]
    blockers = []
    # Logical history is a necessary floor. The rounded single-arena feasible
    # upper bound is deliberately not used to reject a smaller ceiling.
    canonical_capacity = policy["host_pageable_bytes"] or plan["ceiling_bytes"]
    if inventory["eligible_history_bytes"] > canonical_capacity:
        blockers.append("logical_history_exceeds_fixed_pageable_budget" if policy["host_pageable_bytes"]
            else "logical_history_exceeds_explicit_physical_ceiling")
    if 0 < policy["device_staging_bytes"] < inventory["required_staging_bytes"]:
        blockers.append("all_eligible_consumer_staging_floor_exceeds_declared_budget")
    if 0 < policy["d2h_inflight_bytes"] < inventory["required_d2h_inflight_bytes"]:
        blockers.append("all_eligible_D2H_floor_exceeds_declared_budget")
    points.append({"name": "all-eligible-ceiling", "host_pinned_bytes": plan["ceiling_bytes"],
        "fixed_host_pageable_bytes": policy["host_pageable_bytes"],
        "selection_mode": "all_eligible", "preflight_status": "resource_limited" if blockers else "attempt",
        "resource_limits": blockers, "feasible_arena_upper_bound_bytes": inventory["all_eligible_host_physical_capacity_bytes"],
        "upper_bound_is_not_minimum": True, "requires_actual_planner_and_runtime_admission": not blockers})
    return points


def _command(args, *, manifest, options, output):
    result = [sys.executable, "-B", str(ROOT / "examples/vae/benchmark_packed_offload.py"),
        "--from-result", str(args.from_result.resolve()), "--experiment-manifest", str(manifest),
        "--native-options-json", str(options), "--device", args.device, "--batch-size", str(args.batch_size),
        "--warmup", str(args.warmup), "--updates", str(args.updates), "--trials", str(args.trials),
        "--arms", args.arms, "--start-free-tolerance-mib", str(args.start_free_tolerance_mib),
        "--host-headroom-mib", str(args.host_headroom_mib), "--output", str(output)]
    for key in ("project_root", "train_dir", "optimizer"):
        if getattr(args, key) is not None:
            result += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.synthetic:
        result.append("--synthetic")
    if args.deterministic is not None:
        result.append("--deterministic" if args.deterministic else "--no-deterministic")
    if args.cold_diagnostics:
        result.append("--cold-diagnostics")
    if args.profile_after_measurement:
        result += ["--profile-after-measurement", "--profile-updates", str(args.profile_updates)]
        if args.profile_memory:
            result.append("--profile-memory")
        if args.profile_record_shapes:
            result.append("--profile-record-shapes")
    return result


def _run(command, directory, label, environment):
    log = directory / (label + ".log")
    with log.open("x") as stream:
        try:
            code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        except OSError as error:
            stream.write(f"{type(error).__name__}: {error}\n")
            code = None
    return code, _ref(log)


def _bootstrap(args, plan, directory, environment):
    from examples.vae.execution_memory import host_capacity_admission
    from examples.vae.native_execution import _json_input

    bootstrap = deepcopy(plan)
    budget = min(16384 * 2**20, plan["ceiling_bytes"])
    admission = host_capacity_admission(budget + plan["manifest"]["policies"]["demand_offload"]["host_pageable_bytes"],
        headroom_bytes=args.host_headroom_mib * 2**20)
    if not admission["admitted"]:
        return None, {"status": admission["status"], "host_capacity_preflight": admission,
            "scope": "fresh_untimed_native_demand_first_update_not_a_benchmark_point",
            "worker_started": False, "CUDA_execution_proven": False,
            "error": admission.get("reason") or "Host capacity admission failed",
            "failure_context": {"stage": "inventory_bootstrap", "phase": "host_capacity_admission"}}
    policy = {**bootstrap["manifest"]["policies"]["demand_offload"], "version": 2, "enabled": True,
        "policy": "demand", "host_pinned_bytes": budget}
    bootstrap["manifest"]["policies"]["demand_offload"] = policy
    bootstrap["workers"]["demand_offload"]["settings"] = offload._apply_policy(bootstrap["workers"]["demand_offload"]["settings"], policy)
    bootstrap["workers"]["demand_offload"]["settings"]["runtime"]["compile"]["options"]["packed_offload_selection_mode"] = "peak"
    options = deepcopy(args)
    options.warmup, options.updates = 0, 1
    options.profile_after_measurement = False
    output, log, command = offload._pinned_command(options, bootstrap, directory, "inventory-bootstrap", "demand_offload")
    command.remove("--benchmark")
    inventory_path = directory / "inventory.json"
    command += ["--inventory-output", str(inventory_path)]
    code, log_ref = _run(command, directory, "inventory-bootstrap", environment)
    report = {"status": "failed", "worker_exit_code": code, "log": log_ref,
        "scope": "fresh_untimed_native_demand_first_update_not_a_benchmark_point", "command": command,
        "host_capacity_preflight": admission, "worker_started": True, "expected_worker_path": str(output.resolve())}
    if not output.is_file():
        return _bootstrap_failure(report, "worker_receipt", f"Inventory worker produced no receipt (exit={code})")
    try:
        report["worker"] = _ref(output)
        raw = json.loads(output.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Worker receipt must be an object")
    except (OSError, UnicodeError, ValueError) as error:
        return _bootstrap_failure(report, "worker_receipt", f"Inventory worker receipt invalid (exit={code}): {type(error).__name__}: {error}")
    child_error = raw.get("error")
    if isinstance(child_error, str) and child_error:
        report["error"] = child_error[:4096]
        report["error_truncated"] = len(child_error) > 4096
    context_key = "failure_context" if child_error or raw.get("status") == "failed" else "worker_context"
    report[context_key] = _compact_context(raw.get("failure_context"))
    report.update(worker_status=raw.get("status")[:64] if isinstance(raw.get("status"), str) else None,
        worker_execution_completed=raw.get("execution_completed") is True,
        worker_native_ownership_verified=raw.get("native_ownership_verified") is True)
    if (code != 0 or raw.get("status") != "completed" or raw.get("execution_completed") is not True
            or raw.get("native_ownership_verified") is not True or child_error is not None):
        return _bootstrap_failure(report, "worker_completion",
            f"Inventory worker did not complete successfully (exit={code}, status={report['worker_status']}, "
            f"execution_completed={report['worker_execution_completed']}, native_ownership_verified={report['worker_native_ownership_verified']})")
    # A successful worker's last context is not the failure location of a
    # missing, corrupt or stale compact export. Keep it without misattribution.
    if not inventory_path.is_file():
        return _bootstrap_failure(report, "inventory_receipt", "Completed native inventory worker did not write the requested compact inventory")
    try:
        report["inventory"] = _ref(inventory_path)
        inventory, source = _json_input(inventory_path)
        _validate_inventory(inventory, plan)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        return _bootstrap_failure(report, "inventory_validation", f"Compact inventory invalid: {type(error).__name__}: {error}")
    report.update(status="completed", inventory=source)
    return inventory, report


def _metrics(point):
    result = {}
    if not point.get("valid"):
        return result
    summary = point["summary"]
    resident = summary["arms"].get("resident", {}).get("metrics", {})
    for arm, row in summary["arms"].items():
        metrics = row["metrics"]
        value = {name: metric["median"] for name, metric in metrics.items() if isinstance(metric, dict) and "median" in metric}
        for scope in ("total", "steady"):
            key = scope + "_peak_allocated_bytes"
            if resident.get(key) and metrics.get(key):
                before, after = resident[key]["median"], metrics[key]["median"]
                value[scope + "_GPU_saved_bytes_vs_matched_resident"] = before - after
                value[scope + "_GPU_reduction_percent_vs_matched_resident"] = 100*(before-after)/before
        if resident.get("wall_ms_per_update") and metrics.get("wall_ms_per_update"):
            value["wall_overhead_percent_ratio_of_medians"] = 100*(metrics["wall_ms_per_update"]["median"] / resident["wall_ms_per_update"]["median"]-1)
        result[arm] = value
    return result


def _point_identity(result, result_path):
    first = next((row for row in result.get("results", ()) if row.get("arm") == "resident"), None)
    if first is None or not isinstance(first.get("result"), dict):
        raise ValueError("Completed capacity point lacks a resident worker reference")
    source = result_path.parent / first["result"]["path"]
    metadata = _ref(source)
    if any(metadata[key] != first["result"][key] for key in ("bytes", "sha256")):
        raise ValueError("Capacity point resident receipt hash mismatch")
    identity = offload._identity(json.loads(source.read_text()))
    # Selection is the declared residency axis. Every other mathematical,
    # input/state/RNG/software/GPU/compiler option remains comparable.
    identity["native_math_options"].pop("packed_offload_selection_mode", None)
    return identity


def _first_validation_error(result):
    for owner in (result.get("configuration_preflight"), result.get("summary"), *result.get("results", ())):
        if isinstance(owner, dict):
            reasons = owner.get("invalid_reasons", owner.get("validation_errors", ()))
            if isinstance(reasons, (list, tuple)):
                first = next((reason for reason in reasons if isinstance(reason, str) and reason), None)
                if first is not None:
                    return first[:4096]
    return None


def _demand_oom_evidence(result, result_path, plan, spec):
    """Only an explicitly typed, identity-checked demand OOM is recoverable.

    This classifies a failed attempt for continuation, never creates a timing
    point. Missing/malformed evidence and other failed peers remain structural.
    """
    configuration = result.get("configuration_preflight")
    if result.get("error") is not None or not isinstance(configuration, dict) or configuration.get("valid") is not True:
        return None
    rows = result.get("results")
    expected = {(block, arm) for block, order in enumerate(plan["order"], 1) for arm in order}
    if (not isinstance(rows, list) or any(not isinstance(row, dict)
            or type(row.get("block")) is not int or not isinstance(row.get("arm"), str) for row in rows)
            or len(rows) != len(expected) or {(row.get("block"), row.get("arm")) for row in rows} != expected):
        return None
    policy = {**plan["manifest"]["policies"]["demand_offload"], "version": 2,
        "host_pinned_bytes": spec["host_pinned_bytes"]}
    settings = offload._apply_policy(plan["workers"]["demand_offload"]["settings"], policy)
    settings["runtime"]["compile"]["options"]["packed_offload_selection_mode"] = spec["selection_mode"]
    evidence = []
    for row in rows:
        if row.get("status") == "completed" and row.get("worker_exit_code") == 0 and row.get("validation_errors") == []:
            continue
        if row.get("arm") != "demand_offload" or row.get("status") != "failed" or row.get("worker_exit_code") != 1:
            return None
        try:
            reference = row["result"]
            source = result_path.parent / reference["path"]
            metadata = _ref(source)
            if any(metadata[key] != reference[key] for key in ("bytes", "sha256")):
                return None
            raw = json.loads(source.read_text())
            if (not isinstance(raw, dict) or raw.get("status") != "failed" or raw.get("capacity_status") != "cuda_oom"
                    or not isinstance(raw.get("error"), str) or not raw["error"].startswith("OutOfMemoryError:")
                    or not str(raw.get("device", "")).startswith("cuda") or raw.get("collection_errors")
                    or raw.get("worker_result_error") or raw.get("installed") != plan["installed"]
                    or raw.get("origin") != plan["origin"] or raw.get("settings") != settings
                    or any(raw.get(key) is not True for key in ("source_stable", "installed_stable", "helper_source_stable"))):
                return None
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            return None
        evidence.append({"worker": metadata, "capacity_status": "cuda_oom", "error_class": "OutOfMemoryError",
            "failure_context": _compact_context(raw.get("failure_context"))})
    return evidence or None


def _point_progress(point, index, total, output, event):
    elapsed = f" elapsed={point['elapsed_seconds']:.3f}s" if "elapsed_seconds" in point else ""
    reason = point.get("first_validation_error") or point.get("error")
    print(f"[frontier {index}/{total}] {event} {point['name']} status={point['status']}{elapsed} progress={output}"
        + (f" reason={reason}" if reason else ""), flush=True)


def run_frontier(args):
    from benchmarks.compiler.installed_runtime import child_environment
    from examples.vae.native_execution import _json_input
    from examples.vae.execution_memory import host_capacity_admission

    plan = _base(args)
    inventory, inventory_source = (None, None) if args.inventory_json is None else _json_input(args.inventory_json)
    if args.dry_run:
        print(json.dumps({**plan, "capacity_points": None if inventory is None else capacity_points(plan, inventory),
            "inventory_status": "requires_fresh_native_bootstrap" if inventory is None else "provided"}, indent=2, allow_nan=False))
        return 0
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh capacity-frontier output")
    directory.mkdir(parents=True)
    report = {**plan, "schema_version": 1, "status": "running", "points": [], "tensor_files_written": 0,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "CUDA_execution_proven": False,
        "maximum_all_eligible_measured": False, "all_eligible_status": "not_reached", "stopped_early": False,
        "maximum_point": {"selection_mode": "all_eligible", "host_pinned_bytes": plan["ceiling_bytes"],
            "status": "requires_inventory", "timing_point_created": False},
        "host_metric_scopes": {"physical_pinned_peak_bytes": "one shared offload allocator-domain physical arena peak; process cache is separate",
            "Torch_host_pinned_allocated_peak_bytes_approximate": "process-wide Torch rounded pinned allocator high-water mark; approximate",
            "actual_admitted_history_bytes": "unique regional bundle IDs per forward ordinal; exact model-scope value only for one shared manager"},
        "host_capacity_guard": {"headroom_bytes": args.host_headroom_mib * 2**20,
            "per_point_and_worker_checks": True, "requested_capacity_clamped": False}}
    report["frontier_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment = child_environment(ROOT, identity=plan["installed"])
    environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
    environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"
    _write(output, report)
    try:
        if inventory is None:
            inventory, bootstrap = _bootstrap(args, plan, directory, environment)
            report["inventory_bootstrap"] = bootstrap
            _write(output, report)
            if inventory is None:
                report["maximum_point"].update(status="not_reached", reason="inventory_bootstrap_unavailable")
                raise RuntimeError("Fresh native inventory bootstrap failed: " + str(bootstrap.get("error") or bootstrap["status"])
                    + "; no capacity timing point was run")
        points = capacity_points(plan, inventory)
        report["points"] = [{**spec, "valid": False, "worker_started": False,
            "timing_point_created": False, "status": "pending"} for spec in points]
        report["inventory"] = inventory
        report["inventory_source"] = inventory_source or report["inventory_bootstrap"]["inventory"]
        if args.inventory_output is not None:
            target = args.inventory_output.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((json.dumps(inventory, indent=2, allow_nan=False)+"\n").encode())
            report["inventory_export"] = _ref(target)
        options_base = deepcopy(plan["workers"]["demand_offload"]["settings"]["runtime"]["compile"]["options"])
        for name in ("bf16_policy", "activation_memory", "packed_activation_offload"):
            options_base.pop(name, None)
        anchor = None
        for index, point in enumerate(report["points"], 1):
            spec = points[index - 1]
            started = time.perf_counter()
            point.update(status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
            _write(output, report)
            _point_progress(point, index, len(points), output, "START")
            if spec["preflight_status"] != "attempt":
                point.update(status="resource_limited", elapsed_seconds=time.perf_counter() - started)
                _write(output, report)
                _point_progress(point, index, len(points), output, "COMPLETE")
                continue
            point["host_capacity_preflight"] = host_capacity_admission(
                spec["host_pinned_bytes"] + plan["manifest"]["policies"]["demand_offload"]["host_pageable_bytes"],
                headroom_bytes=args.host_headroom_mib * 2**20)
            if not point["host_capacity_preflight"]["admitted"]:
                point.update(status=point["host_capacity_preflight"]["status"], timing_point_created=False,
                    worker_started=False, error="Physical host/job RAM capacity is unavailable or insufficient; requested point was not clamped",
                    elapsed_seconds=time.perf_counter() - started)
                _write(output, report)
                _point_progress(point, index, len(points), output, "COMPLETE")
                continue
            point_directory = directory / spec["name"]
            point_directory.mkdir()
            manifest = deepcopy(plan["manifest"])
            for policy in manifest["policies"].values():
                policy.update(version=2, host_pinned_bytes=spec["host_pinned_bytes"])
            manifest_path = point_directory / "experiment.json"
            options_path = point_directory / "native-options.json"
            _write(manifest_path, manifest)
            _write(options_path, {**options_base, "packed_offload_selection_mode": spec["selection_mode"]})
            result_path = point_directory / "comparison.json"
            command = _command(args, manifest=manifest_path, options=options_path, output=result_path)
            code, log = _run(command, point_directory, "comparison", environment)
            point.update(worker_exit_code=code, log=log, command=command, timing_point_created=False, worker_started=True)
            result = None
            if result_path.is_file():
                try:
                    point["result"] = _ref(result_path)
                    result = json.loads(result_path.read_text())
                    if (not isinstance(result, dict) or not isinstance(result.get("summary"), dict)
                            or not isinstance(result.get("results", []), list)):
                        raise ValueError("Comparison receipt must contain object summary and worker result list")
                    child_error = result.get("error")
                    if child_error is not None and not isinstance(child_error, str):
                        raise ValueError("Comparison receipt error must be a string or null")
                    point.update(summary=result["summary"], worker_status=result.get("status"),
                        valid=code == 0 and result.get("status") == "completed" and result["summary"].get("valid") is True and child_error is None,
                        error=child_error[:4096] if child_error else None, first_validation_error=_first_validation_error(result))
                    point["status"] = "completed" if point["valid"] else "failed"
                    if point["valid"]:
                        actual_identity = _point_identity(result, result_path)
                        if anchor is not None and anchor != actual_identity:
                            raise ValueError("Input/state/RNG/software/GPU/math identity drift across capacity points")
                        anchor = actual_identity if anchor is None else anchor
                        point["comparison_identity_sha256"] = _digest(actual_identity)
                    point["metrics"] = _metrics(point)
                except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as error:
                    point.update(valid=False, status="failed", comparison_error=f"{type(error).__name__}: {error}")
            else:
                point.update(status="failed", error="No comparison receipt")
            point["timing_point_created"] = point["valid"]
            point.setdefault("metrics", {})
            if not point["valid"]:
                point["error"] = point.get("error") or point.get("comparison_error") or point.get("first_validation_error") or "Capacity comparison did not complete with valid evidence"
                report.setdefault("error", f"{point['name']}: {point['error']}")
                if point.get("first_validation_error"):
                    report.setdefault("first_validation_error", point["first_validation_error"])
                resource = (_demand_oom_evidence(result, result_path, plan, spec)
                    if isinstance(result, dict) and not point.get("comparison_error") else None)
                point["failure_classification"] = "demand_cuda_oom" if resource else "structural_or_evidence_failure"
                if resource:
                    point["capacity_failure_evidence"] = resource
                else:
                    report.update(stopped_early=index < len(points), stopped_after_point=point["name"], stop_reason=point["error"])
                    for remaining in report["points"][index:]:
                        remaining.update(status="not_executed", stopped_after_point=point["name"],
                            not_executed_reason="Stopped after structural/evidence failure: " + point["error"])
            point["elapsed_seconds"] = time.perf_counter() - started
            _write(output, report)
            _point_progress(point, index, len(points), output, "COMPLETE")
            if not point["valid"] and point["failure_classification"] != "demand_cuda_oom":
                break
        successful = [point for point in report["points"] if point["valid"]]
        baseline = successful[0] if successful else None
        if baseline is not None:
            for point in successful:
                for arm, metrics in point["metrics"].items():
                    base = baseline["metrics"].get(arm, {})
                    for scope in ("total", "steady"):
                        key = scope + "_GPU_saved_bytes_vs_matched_resident"
                        if key in metrics and key in base:
                            metrics[scope + "_extra_GPU_saved_bytes_vs_first_successful_budget"] = metrics[key] - base[key]
                    key = "wall_overhead_percent_ratio_of_medians"
                    if key in metrics and key in base:
                        metrics["extra_wall_overhead_percentage_points_vs_first_successful_budget"] = metrics[key] - base[key]
                    key = "wall_ms_per_update"
                    if key in metrics and key in base:
                        metrics["wall_ms_per_update_delta_vs_first_successful_budget"] = metrics[key] - base[key]
        maximum = next(point for point in report["points"] if point["selection_mode"] == "all_eligible")
        report["maximum_point"] = {key: maximum.get(key) for key in ("selection_mode", "host_pinned_bytes", "status",
            "timing_point_created", "resource_limits", "host_capacity_preflight", "error", "not_executed_reason", "stopped_after_point")}
        complete = len(successful) == len(report["points"])
        report.update(status="completed" if complete else "partial" if successful else "failed", successful_capacity_points=len(successful),
            all_eligible_status=maximum["status"], maximum_all_eligible_measured=maximum["valid"],
            unexecuted_capacity_points=sum(point["status"] == "not_executed" for point in report["points"]),
            resource_limited_or_failed_points_preserved=True,
            CUDA_execution_proven=bool(successful) and all(point["summary"].get("CUDA_execution_proven") for point in successful))
        if not complete and not report.get("error"):
            report["error"] = "Resource-limited capacity points were not measured; completed points remain preserved"
        if offload.source_identity() != plan["source_sha256"] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != report["frontier_source_sha256"]:
            raise RuntimeError("Frontier or worker helper source changed during the sweep")
    except Exception as error:
        from examples.vae.benchmark_vae import failure_error_details

        report.update(status="failed", **failure_error_details(error))
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write(output, report)
    bootstrap = report.get("inventory_bootstrap")
    print(json.dumps({"status": report["status"], "output": str(output),
        "successful_capacity_points": report.get("successful_capacity_points", 0),
        "all_eligible_status": report.get("all_eligible_status"), "error": report.get("error"),
        **({"bootstrap_failure": {key: bootstrap.get(key) for key in ("status", "worker_exit_code", "error", "failure_context",
            "worker_execution_completed", "worker_native_ownership_verified", "worker", "log")}}
            if isinstance(bootstrap, dict) and bootstrap.get("status") != "completed" else {})}, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    return run_frontier(parser().parse_args(argv))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
